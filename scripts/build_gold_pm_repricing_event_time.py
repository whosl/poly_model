"""Build an event-time executable repricing dataset preview.

This is a dataset/eval probe, not a full training pipeline.  Sampling points are
PM quote update event timestamps from bronze pm_price_change.  Features are
as-of joined from the existing 1s gold repricing feature state.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.dataset as ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--price-change", default="data/bronze/pm_price_change")
    p.add_argument("--silver", default="data/silver/pm_1s")
    p.add_argument("--gold", default="data/gold/pm_repricing_1s")
    p.add_argument("--out", default="data/gold/pm_repricing_event_time")
    p.add_argument("--max-events", type=int, default=500_000)
    p.add_argument("--date", default=None, help="Optional YYYY-MM-DD partition/date filter for the event-time probe.")
    p.add_argument("--latencies", default="0,0.25,0.5,1.0,2.0")
    p.add_argument("--horizons", default="1,3,5,10")
    return p.parse_args()


def floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def scan(path: str | Path, cols: list[str] | None = None) -> pl.DataFrame:
    p = Path(path)
    lf = pl.scan_parquet(str(p / "**" / "*.parquet") if p.is_dir() else str(p), extra_columns="ignore")
    if cols:
        schema = set(lf.collect_schema().names())
        lf = lf.select([c for c in cols if c in schema])
    return lf.collect()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_price_change_events(path: str | Path, date: str | None, max_events: int) -> pl.DataFrame:
    """Read a bounded number of quote-update timestamps without Polars scanning all files."""
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    filt = (ds.field("date") == date) if date else None
    scanner = dataset.scanner(columns=["market_id", "ts_event", "best_bid", "best_ask"], filter=filt, batch_size=65536)
    frames: list[pl.DataFrame] = []
    total = 0
    for batch in scanner.to_batches():
        df = pl.from_arrow(batch)
        if df.height == 0:
            continue
        df = (
            df.filter(pl.col("ts_event").is_not_null() & (pl.col("best_bid").is_not_null() | pl.col("best_ask").is_not_null()))
            .select([pl.col("market_id"), pl.col("ts_event").alias("sample_ts")])
            .unique()
        )
        if df.height == 0:
            continue
        need = max_events - total
        frames.append(df.head(need))
        total += min(df.height, need)
        if total >= max_events:
            break
    if not frames:
        return pl.DataFrame({"market_id": [], "sample_ts": []}, schema={"market_id": pl.Utf8, "sample_ts": pl.Datetime("us", "UTC")})
    return pl.concat(frames, how="vertical").unique().sort(["market_id", "sample_ts"]).head(max_events)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/stage1")
    report_dir.mkdir(parents=True, exist_ok=True)
    latencies = floats(args.latencies)
    horizons = ints(args.horizons)

    pc = load_price_change_events(args.price_change, args.date, args.max_events)
    gold_cols = [
        "market_id",
        "sample_ts",
        "split",
        "time_to_expiry_seconds",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "no_bid",
        "no_ask",
        "no_mid",
        "yes_spread",
        "pair_mid_sum",
        "btc_return_1s",
        "btc_return_5s",
        "btc_return_30s",
        "realized_vol_10s",
        "realized_vol_30s",
        "formula_p_yes",
        "formula_p_yes_minus_yes_mid",
        "formula_p_yes_minus_yes_ask",
        "pm_yes_mid_change_1s_past",
        "pm_yes_mid_change_5s_past",
        "seconds_since_last_pm_update",
    ]
    gold = scan(args.gold, gold_cols).sort(["market_id", "sample_ts"])
    base = pc.join_asof(gold, on="sample_ts", by="market_id", strategy="backward", tolerance="2s")
    silver_cols = ["market_id", "sample_ts", "yes_bid", "yes_ask", "no_bid", "no_ask", "yes_is_stale", "no_is_stale", "yes_crossed_quote", "no_crossed_quote"]
    silver = scan(args.silver, silver_cols).sort(["market_id", "sample_ts"])
    entry = silver.rename({"sample_ts": "entry_quote_ts", "yes_ask": "entry_yes_ask", "no_ask": "entry_no_ask", "yes_is_stale": "entry_yes_is_stale", "no_is_stale": "entry_no_is_stale", "yes_crossed_quote": "entry_yes_crossed_quote", "no_crossed_quote": "entry_no_crossed_quote"}).select(["market_id", "entry_quote_ts", "entry_yes_ask", "entry_no_ask", "entry_yes_is_stale", "entry_no_is_stale", "entry_yes_crossed_quote", "entry_no_crossed_quote"])
    exitq = silver.rename({"sample_ts": "exit_quote_ts", "yes_bid": "exit_yes_bid", "no_bid": "exit_no_bid", "yes_is_stale": "exit_yes_is_stale", "no_is_stale": "exit_no_is_stale", "yes_crossed_quote": "exit_yes_crossed_quote", "no_crossed_quote": "exit_no_crossed_quote"}).select(["market_id", "exit_quote_ts", "exit_yes_bid", "exit_no_bid", "exit_yes_is_stale", "exit_no_is_stale", "exit_yes_crossed_quote", "exit_no_crossed_quote"])
    frames = []
    rows = []
    for latency in latencies:
        lat_ms = int(round(latency * 1000))
        for horizon in horizons:
            left = base.with_columns(
                [
                    pl.lit(latency).alias("latency_seconds"),
                    pl.lit(horizon).alias("exit_horizon_seconds"),
                    (pl.col("sample_ts") + pl.duration(milliseconds=lat_ms)).alias("entry_time"),
                    (pl.col("sample_ts") + pl.duration(milliseconds=lat_ms + horizon * 1000)).alias("exit_time"),
                ]
            ).filter(pl.col("time_to_expiry_seconds") >= latency + horizon)
            joined = (
                left.sort(["market_id", "entry_time"])
                .join_asof(entry, left_on="entry_time", right_on="entry_quote_ts", by="market_id", strategy="forward", tolerance="1s")
                .sort(["market_id", "exit_time"])
                .join_asof(exitq, left_on="exit_time", right_on="exit_quote_ts", by="market_id", strategy="forward", tolerance="1s")
            )
            filt = joined.filter(
                pl.col("entry_yes_ask").is_not_null()
                & pl.col("entry_no_ask").is_not_null()
                & pl.col("exit_yes_bid").is_not_null()
                & pl.col("exit_no_bid").is_not_null()
                & (~pl.col("entry_yes_is_stale").fill_null(True))
                & (~pl.col("entry_no_is_stale").fill_null(True))
                & (~pl.col("exit_yes_is_stale").fill_null(True))
                & (~pl.col("exit_no_is_stale").fill_null(True))
                & (~pl.col("entry_yes_crossed_quote").fill_null(True))
                & (~pl.col("entry_no_crossed_quote").fill_null(True))
            ).with_columns(
                [
                    (pl.col("exit_yes_bid") - pl.col("entry_yes_ask")).alias("pnl_up"),
                    (pl.col("exit_no_bid") - pl.col("entry_no_ask")).alias("pnl_down"),
                    ((pl.col("exit_yes_bid") - pl.col("entry_yes_ask")) > 0).cast(pl.Int8).alias("label_up_profitable"),
                    ((pl.col("exit_no_bid") - pl.col("entry_no_ask")) > 0).cast(pl.Int8).alias("label_down_profitable"),
                ]
            )
            frames.append(filt)
            rows.append({"latency_seconds": latency, "exit_horizon_seconds": horizon, "rows": filt.height, "up_win_rate": float(filt["label_up_profitable"].mean()) if filt.height else None, "down_win_rate": float(filt["label_down_profitable"].mean()) if filt.height else None, "avg_pnl_up": float(filt["pnl_up"].mean()) if filt.height else None, "avg_pnl_down": float(filt["pnl_down"].mean()) if filt.height else None})
    result = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
    result.write_parquet(out / "part-00000.parquet")
    write_csv(report_dir / "pm_repricing_event_time_label_distribution.csv", rows)
    lines = [
        "# PM Repricing Event-time Dataset Report\n\n",
        f"- raw event sample rows: `{pc.height}`\n",
        f"- output rows: `{result.height}`\n",
        f"- max_events setting: `{args.max_events}`\n\n",
        "This is an event-time probe using quote-update timestamps and existing 1s gold features as-of joined backward. Entry/exit quotes are still sampled from silver quote state, so this is not yet a full tick-level execution simulator.\n\n",
        "| latency | horizon | rows | up_win_rate | avg_pnl_up | down_win_rate | avg_pnl_down |\n",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for r in rows:
        lines.append(f"| {r['latency_seconds']} | {r['exit_horizon_seconds']} | {r['rows']} | {r['up_win_rate']} | {r['avg_pnl_up']} | {r['down_win_rate']} | {r['avg_pnl_down']} |\n")
    (report_dir / "pm_repricing_event_time_dataset_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
