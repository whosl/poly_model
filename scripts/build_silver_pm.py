from __future__ import annotations

import argparse
import json
from datetime import datetime
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import polars as pl

from preprocess.asset_mapping import load_combined_mapping, mapping_to_frame
from preprocess.config import load_config, resolve_path
from preprocess.dataset_io import write_partitioned_parquet
from preprocess.logging_utils import setup_logging
from preprocess.paths import bootstrap_layout
from preprocess.pm_quote_state import build_asset_quote_state, build_asset_state_on_grid, normalize_quote_events
from preprocess.reporting import markdown_table, write_markdown

logger = logging.getLogger(__name__)

NEEDED_ORDERBOOK_COLS = ["market_id", "asset_id", "ts_event", "ts_recv", "best_bid", "best_ask", "bid_size_1", "ask_size_1", "source_file"]
NEEDED_PRICE_CHANGE_COLS = ["market_id", "asset_id", "ts_event", "ts_recv", "best_bid", "best_ask", "price", "event_type", "source_file"]
SILVER_COLUMNS = [
    "sample_ts", "market_id", "yes_asset_id", "no_asset_id", "mapping_status",
    "yes_bid", "yes_ask", "yes_mid", "yes_spread", "yes_bid_depth_5", "yes_ask_depth_5", "yes_depth_imbalance_5", "yes_quote_age_seconds", "yes_is_stale",
    "no_bid", "no_ask", "no_mid", "no_spread", "no_bid_depth_5", "no_ask_depth_5", "no_depth_imbalance_5", "no_quote_age_seconds", "no_is_stale",
    "pair_bid_sum", "pair_ask_sum", "pair_mid_sum", "seconds_since_last_pm_update",
    "pm_yes_mid_change_1s_past", "pm_yes_mid_change_5s_past", "pm_no_mid_change_1s_past", "pm_no_mid_change_5s_past",
    "market_start_ts", "market_end_ts", "time_elapsed_seconds", "time_to_expiry_seconds",
    "yes_last_quote_update_ts", "no_last_quote_update_ts", "yes_crossed_quote", "no_crossed_quote", "quote_source_state_available", "date",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Polymarket 1s silver quote table from orderbook + price_change events.")
    p.add_argument("--config", required=True)
    p.add_argument("--market-id")
    p.add_argument("--limit-markets", type=int)
    p.add_argument("--market-batch-size", type=int, default=100)
    p.add_argument("--date", help="Single YYYY-MM-DD / YYYYMMDD market date")
    p.add_argument("--start-date", help="Inclusive YYYYMMDD or YYYY-MM-DD market_start_ts date filter")
    p.add_argument("--end-date", help="Inclusive YYYYMMDD or YYYY-MM-DD market_start_ts date filter")
    p.add_argument("--force", action="store_true", help="Overwrite selected date/market silver partitions")
    return p.parse_args()


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if "-" in text:
        return text
    return datetime.strptime(text, "%Y%m%d").date().isoformat()


def to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return pl.Series([value]).str.to_datetime(time_zone="UTC")[0]


def has_parquet(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def market_date(row: dict[str, Any]) -> str | None:
    value = row.get("market_start_ts")
    if value is None:
        return None
    return str(value)[:10]


def selected_market_files(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    seen: set[Path] = set()
    for row in rows:
        mid = row["market_id"]
        d = market_date(row)
        roots = [root / f"date={d}" / f"market_id={mid}"] if d else []
        if not roots:
            roots = [p / f"market_id={mid}" for p in root.glob("date=*")]
        for r in roots:
            if r.exists():
                for f in r.glob("*.parquet"):
                    if f not in seen:
                        seen.add(f); files.append(str(f))
    return files


def read_selected(root: Path, rows: list[dict[str, Any]], cols: list[str], label: str) -> pl.DataFrame:
    files = selected_market_files(root, rows)
    if not files:
        logger.info("No %s parquet files for current batch", label)
        return pl.DataFrame()
    logger.info("Reading %s: %d parquet files", label, len(files))
    lf = pl.scan_parquet(files, hive_partitioning=True)
    available = lf.collect_schema().names()
    keep = [c for c in cols if c in available]
    return lf.select(keep).collect()


def clear_selected_partitions(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        d = market_date(row)
        if not d:
            continue
        out_dir = root / f"date={d}" / f"market_id={row['market_id']}"
        if out_dir.exists():
            shutil.rmtree(out_dir)


def prepare_mapping(config: dict, args: argparse.Namespace) -> pl.DataFrame:
    mapping = load_combined_mapping(config)
    df = mapping_to_frame(mapping).filter(pl.col("mapping_status") == "ok")
    if args.market_id:
        df = df.filter(pl.col("market_id") == args.market_id)
    date = normalize_date(args.date)
    start = date or normalize_date(args.start_date)
    end = date or normalize_date(args.end_date)
    if start or end:
        df = df.with_columns(pl.col("market_start_ts").cast(pl.String).str.slice(0, 10).alias("__market_date"))
        if start:
            df = df.filter(pl.col("__market_date") >= start)
        if end:
            df = df.filter(pl.col("__market_date") <= end)
        df = df.drop("__market_date")
    df = df.sort("market_start_ts")
    if args.limit_markets:
        df = df.head(args.limit_markets)
    return df


def build_market_silver(mapping_row: dict[str, Any], quote_state: pl.DataFrame, ttl_seconds: int) -> pl.DataFrame:
    market_id = mapping_row["market_id"]
    yes_asset_id = str(mapping_row["yes_asset_id"])
    no_asset_id = str(mapping_row["no_asset_id"])
    start_dt = to_dt(mapping_row["market_start_ts"])
    end_dt = to_dt(mapping_row["market_end_ts"])
    if start_dt is None or end_dt is None:
        return pl.DataFrame()
    grid = pl.DataFrame({"sample_ts": pl.datetime_range(start_dt, end_dt, interval="1s", eager=True, time_zone="UTC")})
    market_state = quote_state.filter(pl.col("market_id") == market_id)
    state = build_asset_state_on_grid(grid, market_state, yes_asset_id, "yes", ttl_seconds)
    state = build_asset_state_on_grid(state, market_state, no_asset_id, "no", ttl_seconds)
    pm_updates = market_state.select("ts_event").unique().sort("ts_event") if not market_state.is_empty() else pl.DataFrame(schema={"ts_event": pl.Datetime(time_zone="UTC")})
    if pm_updates.is_empty():
        state = state.with_columns(pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("last_pm_update_ts"))
    else:
        state = state.join_asof(pm_updates, left_on="sample_ts", right_on="ts_event", strategy="backward").rename({"ts_event": "last_pm_update_ts"})
    state = state.with_columns(
        ((pl.col("sample_ts") - pl.col("last_pm_update_ts")).dt.total_milliseconds() / 1000.0).alias("seconds_since_last_pm_update"),
        (pl.col("yes_bid") + pl.col("no_bid")).alias("pair_bid_sum"),
        (pl.col("yes_ask") + pl.col("no_ask")).alias("pair_ask_sum"),
        (pl.col("yes_mid") + pl.col("no_mid")).alias("pair_mid_sum"),
        pl.lit(market_id).alias("market_id"),
        pl.lit(yes_asset_id).alias("yes_asset_id"),
        pl.lit(no_asset_id).alias("no_asset_id"),
        pl.lit("ok").alias("mapping_status"),
        pl.lit(start_dt).alias("market_start_ts"),
        pl.lit(end_dt).alias("market_end_ts"),
        ((pl.col("sample_ts") - pl.lit(start_dt)).dt.total_milliseconds() / 1000.0).alias("time_elapsed_seconds"),
        ((pl.lit(end_dt) - pl.col("sample_ts")).dt.total_milliseconds() / 1000.0).alias("time_to_expiry_seconds"),
        (pl.col("yes_mid") - pl.col("yes_mid").shift(1)).alias("pm_yes_mid_change_1s_past"),
        (pl.col("yes_mid") - pl.col("yes_mid").shift(5)).alias("pm_yes_mid_change_5s_past"),
        (pl.col("no_mid") - pl.col("no_mid").shift(1)).alias("pm_no_mid_change_1s_past"),
        (pl.col("no_mid") - pl.col("no_mid").shift(5)).alias("pm_no_mid_change_5s_past"),
        (pl.col("yes_last_quote_update_ts").is_not_null() & pl.col("no_last_quote_update_ts").is_not_null()).alias("quote_source_state_available"),
        pl.col("sample_ts").dt.date().cast(pl.String).alias("date"),
    )
    for col in SILVER_COLUMNS:
        if col not in state.columns:
            state = state.with_columns(pl.lit(None).alias(col))
    return state.select(SILVER_COLUMNS)


def batch_rows(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[i:i+size] for i in range(0, len(rows), size)]


def write_empty_report(config: dict, reason: str, mapped: int = 0) -> None:
    lines = ["# Silver Polymarket Report", "", "- rows: `0`", f"- mapped_markets_processed: `{mapped}`", f"- reason: `{reason}`"]
    write_markdown(resolve_path(config, "reports/silver_pm_report.md"), lines)


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    bootstrap_layout(config)
    mapping_df = prepare_mapping(config, args)
    if mapping_df.is_empty():
        write_empty_report(config, "no mapping_status=ok markets after filters", 0)
        logger.info("No mapped markets after filters")
        return
    rows = mapping_df.to_dicts()
    ttl_seconds = int(config["polymarket"]["quote_ttl_seconds"])
    orderbook_root = resolve_path(config, "data/bronze/pm_orderbook")
    price_change_root = resolve_path(config, "data/bronze/pm_price_change")
    out_root = resolve_path(config, "data/silver/pm_1s")
    if args.force:
        clear_selected_partitions(out_root, rows)
    total_rows = 0
    total_events = 0
    total_updates = 0
    total_stale_yes = 0
    total_stale_no = 0
    constant_markets = 0
    processed_markets = 0
    for i, batch in enumerate(batch_rows(rows, max(1, args.market_batch_size)), start=1):
        dates = sorted({market_date(r) for r in batch if market_date(r)})
        logger.info("Processing PM silver batch %d/%d markets=%d dates=%s", i, (len(rows)+args.market_batch_size-1)//args.market_batch_size, len(batch), ",".join(dates))
        ob = read_selected(orderbook_root, batch, NEEDED_ORDERBOOK_COLS, "pm_orderbook")
        pc = read_selected(price_change_root, batch, NEEDED_PRICE_CHANGE_COLS, "pm_price_change")
        logger.info("Raw rows: orderbook=%d price_change=%d", ob.height, pc.height)
        events = normalize_quote_events(ob, pc)
        event_updates = events.filter(pl.col("update_bid").is_not_null() | pl.col("update_ask").is_not_null()) if not events.is_empty() else pl.DataFrame()
        quote_state = build_asset_quote_state(events)
        logger.info("Quote events=%d bid/ask update rows=%d quote_state_rows=%d crossed_rows=%d", events.height, event_updates.height, quote_state.height, 0 if quote_state.is_empty() else quote_state.filter(pl.col("crossed_quote")).height)
        total_events += events.height; total_updates += event_updates.height
        out_frames=[]
        for row in batch:
            silver_m = build_market_silver(row, quote_state, ttl_seconds)
            if silver_m.is_empty():
                continue
            processed_markets += 1
            total_rows += silver_m.height
            total_stale_yes += int(silver_m.get_column("yes_is_stale").sum())
            total_stale_no += int(silver_m.get_column("no_is_stale").sum())
            if int(silver_m.get_column("yes_mid").n_unique()) <= 1:
                constant_markets += 1
            out_frames.append(silver_m)
        if out_frames:
            batch_silver = pl.concat(out_frames, how="diagonal_relaxed")
            write_partitioned_parquet(batch_silver, out_root, ["date", "market_id"], basename="silver")
            logger.info("Generated silver rows=%d stale_yes=%d stale_no=%d constant_market_ratio_so_far=%.4f", batch_silver.height, int(batch_silver["yes_is_stale"].sum()), int(batch_silver["no_is_stale"].sum()), constant_markets/max(processed_markets,1))
    if total_rows == 0:
        write_empty_report(config, "no silver rows generated", mapping_df.height)
        return
    manifest_path = resolve_path(config, "reports/audit/pm_silver_last_build_markets.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"market_ids": [r["market_id"] for r in rows], "rows": total_rows}, indent=2), encoding="utf-8")
    report_lines = ["# Silver Polymarket Report", ""]
    report_lines += [
        f"- rows: `{total_rows}`",
        f"- mapped_markets_processed: `{mapping_df.height}`",
        f"- processed_markets_with_rows: `{processed_markets}`",
        f"- quote_events_rows: `{total_events}`",
        f"- quote_events_with_bid_or_ask_update_rows: `{total_updates}`",
        f"- yes_stale_rows: `{total_stale_yes}`",
        f"- no_stale_rows: `{total_stale_no}`",
        f"- constant_yes_mid_market_ratio: `{constant_markets/max(processed_markets,1):.6f}`",
    ]
    report_lines.append("")
    report_lines.append("## Stale Quote Ratio")
    report_lines.extend(markdown_table(["field", "true_ratio"], [["yes_is_stale", total_stale_yes/max(total_rows,1)], ["no_is_stale", total_stale_no/max(total_rows,1)]]))
    write_markdown(resolve_path(config, "reports/silver_pm_report.md"), report_lines)
    logger.info("Wrote PM silver rows=%d to %s", total_rows, out_root)

if __name__ == "__main__":
    main()
