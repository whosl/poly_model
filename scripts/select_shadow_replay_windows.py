"""Select strong historical replay windows for PM repricing shadow replay."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml


PRED_PATH = "reports/stage1/pm_repricing_test_predictions.parquet"
SILVER_PATH = "data/silver/pm_1s"
OUT_JSON = "data/shadow/replay_windows/repricing_replay_windows.json"
OUT_MD = "reports/shadow/replay_window_selection_report.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/shadow_repricing.yaml")
    p.add_argument("--predictions-path", default=PRED_PATH)
    p.add_argument("--silver-path", default=SILVER_PATH)
    return p.parse_args()


def read_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def tte_bucket_expr() -> pl.Expr:
    return (
        pl.when((pl.col("time_to_expiry_seconds") >= 240) & (pl.col("time_to_expiry_seconds") <= 300)).then(pl.lit("[240,300]"))
        .when((pl.col("time_to_expiry_seconds") >= 180) & (pl.col("time_to_expiry_seconds") < 240)).then(pl.lit("[180,240)"))
        .when((pl.col("time_to_expiry_seconds") >= 120) & (pl.col("time_to_expiry_seconds") < 180)).then(pl.lit("[120,180)"))
        .when((pl.col("time_to_expiry_seconds") >= 60) & (pl.col("time_to_expiry_seconds") < 120)).then(pl.lit("[60,120)"))
        .when((pl.col("time_to_expiry_seconds") >= 30) & (pl.col("time_to_expiry_seconds") < 60)).then(pl.lit("[30,60)"))
        .when((pl.col("time_to_expiry_seconds") >= 10) & (pl.col("time_to_expiry_seconds") < 30)).then(pl.lit("[10,30)"))
        .otherwise(pl.lit("[0,10)"))
    )


def greedy_select(df: pl.DataFrame, max_rows: int) -> pl.DataFrame:
    if df.is_empty():
        return df
    seen_dates: set[str] = set()
    seen_markets: set[str] = set()
    seen_buckets: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for row in df.to_dicts():
        score = 0
        if row["date"] not in seen_dates:
            score += 4
        if row["market_id"] not in seen_markets:
            score += 3
        if row["tte_bucket"] not in seen_buckets:
            score += 2
        row["_diversity_score"] = score
        chosen.append(row)
    chosen = sorted(
        chosen,
        key=lambda r: (
            -r["_diversity_score"],
            -(r.get("offline_p_up_5s", 0.0) if r["expected_direction"] == "UP" else r.get("offline_p_down_5s", 0.0)),
            str(r["date"]),
            str(r["market_id"]),
            str(r["center_sample_ts"]),
        ),
    )
    final: list[dict[str, Any]] = []
    for row in chosen:
        if len(final) >= max_rows:
            break
        final.append(row)
        seen_dates.add(str(row["date"]))
        seen_markets.add(str(row["market_id"]))
        seen_buckets.add(str(row["tte_bucket"]))
    return pl.DataFrame(final).drop("_diversity_score")


def build_candidates(pred: pl.DataFrame) -> pl.DataFrame:
    return (
        pred.with_columns(
            tte_bucket_expr().alias("tte_bucket"),
            pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"),
        )
        .select(
            [
                "market_id",
                "sample_ts",
                "date",
                "tte_bucket",
                "time_to_expiry_seconds",
                "yes_bid",
                "yes_ask",
                "no_bid",
                "no_ask",
                "yes_spread",
                "p_up_5s",
                "p_down_5s",
            ]
        )
    )


def choose_windows(pred: pl.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "baseline_repricing_070_up": build_candidates(pred).filter((pl.col("p_up_5s") >= 0.70) & (pl.col("time_to_expiry_seconds") >= 10)).with_columns(pl.lit("UP").alias("expected_direction"), pl.lit("baseline_repricing_070").alias("expected_config_name"), pl.lit(0.70).alias("expected_threshold"), pl.col("sample_ts").alias("center_sample_ts")),
        "baseline_repricing_070_down": build_candidates(pred).filter((pl.col("p_down_5s") >= 0.70) & (pl.col("time_to_expiry_seconds") >= 10)).with_columns(pl.lit("DOWN").alias("expected_direction"), pl.lit("baseline_repricing_070").alias("expected_config_name"), pl.lit(0.70).alias("expected_threshold"), pl.col("sample_ts").alias("center_sample_ts")),
        "high_conf_tight_spread_up": build_candidates(pred).filter((pl.col("p_up_5s") >= 0.80) & (pl.col("yes_spread") <= 0.02) & (pl.col("time_to_expiry_seconds") >= 60)).with_columns(pl.lit("UP").alias("expected_direction"), pl.lit("high_conf_tight_spread").alias("expected_config_name"), pl.lit(0.80).alias("expected_threshold"), pl.col("sample_ts").alias("center_sample_ts")),
        "high_conf_medium_spread_up": build_candidates(pred).filter((pl.col("p_up_5s") >= 0.80) & (pl.col("yes_spread") <= 0.05) & (pl.col("time_to_expiry_seconds") >= 30)).with_columns(pl.lit("UP").alias("expected_direction"), pl.lit("high_conf_medium_spread").alias("expected_config_name"), pl.lit(0.80).alias("expected_threshold"), pl.col("sample_ts").alias("center_sample_ts")),
    }
    selected: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    idx = 0
    for name, df in groups.items():
        chosen = greedy_select(df.sort(["date", "market_id", "sample_ts"]), 20)
        stats[name] = {
            "candidate_count": df.height,
            "selected_count": chosen.height,
            "market_count": chosen.select(pl.col("market_id").n_unique()).item() if chosen.height else 0,
            "dates": [] if chosen.height == 0 else sorted(chosen["date"].unique().to_list()),
        }
        for row in chosen.to_dicts():
            center = row["center_sample_ts"]
            payload = {
                "window_id": f"w{idx:04d}",
                "market_id": row["market_id"],
                "center_sample_ts": center.isoformat(),
                "start_ts": (center - timedelta(seconds=60)).isoformat(),
                "end_ts": (center + timedelta(seconds=90)).isoformat(),
                "expected_direction": row["expected_direction"],
                "expected_config_name": row["expected_config_name"],
                "expected_threshold": row["expected_threshold"],
                "offline_p_up_5s": row["p_up_5s"],
                "offline_p_down_5s": row["p_down_5s"],
                "offline_yes_bid": row["yes_bid"],
                "offline_yes_ask": row["yes_ask"],
                "offline_no_bid": row["no_bid"],
                "offline_no_ask": row["no_ask"],
                "date": row["date"],
                "time_to_expiry_seconds": row["time_to_expiry_seconds"],
                "yes_spread": row["yes_spread"],
                "tte_bucket": row["tte_bucket"],
            }
            selected.append(payload)
            idx += 1
    return selected, stats


def write_report(path: str, windows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    ensure_parent(path)
    df = pl.DataFrame(windows) if windows else pl.DataFrame()
    lines = ["# Shadow Replay Window Selection Report\n\n"]
    for name, meta in stats.items():
        lines.append(f"## {name}\n\n")
        lines.append(f"- candidate_count: `{meta['candidate_count']}`\n")
        lines.append(f"- selected_count: `{meta['selected_count']}`\n")
        lines.append(f"- market_count: `{meta['market_count']}`\n")
        lines.append(f"- dates: `{meta['dates']}`\n\n")
    if not df.is_empty():
        lines.append("## Selected window aggregate stats\n\n")
        lines.append(f"- total_windows: `{df.height}`\n")
        lines.append(f"- distinct_markets: `{df.select(pl.col('market_id').n_unique()).item()}`\n")
        lines.append(f"- distinct_dates: `{df.select(pl.col('date').n_unique()).item()}`\n\n")
        for col in ["offline_p_up_5s", "offline_p_down_5s", "yes_spread", "time_to_expiry_seconds"]:
            part = df.select(
                pl.col(col).min().alias("min"),
                pl.col(col).median().alias("p50"),
                pl.col(col).quantile(0.9).alias("p90"),
                pl.col(col).max().alias("max"),
            ).to_dicts()[0]
            lines.append(f"- {col}: min=`{part['min']}`, p50=`{part['p50']}`, p90=`{part['p90']}`, max=`{part['max']}`\n")
        lines.append("\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _cfg = read_yaml(args.config)
    pred = pl.read_parquet(args.predictions_path)
    windows, stats = choose_windows(pred)
    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_predictions_path": args.predictions_path,
        "source_silver_path": args.silver_path,
        "windows": windows,
    }
    ensure_parent(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(OUT_MD, windows, stats)
    print(f"Wrote {OUT_JSON} with {len(windows)} windows")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
