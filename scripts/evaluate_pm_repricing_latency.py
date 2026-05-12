"""Latency sensitivity for executable PM repricing signals."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pm_repricing_executable import (  # noqa: E402
    COOLDOWNS,
    add_tte_bucket,
    max_drawdown,
    parse_float_list,
    parse_int_list,
    read_parquet_dataset,
    select_mode,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="reports/stage1/pm_repricing_test_predictions.parquet")
    p.add_argument("--silver", default="data/silver/pm_1s")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--thresholds", default="0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--horizons", default="1,5,10,30")
    p.add_argument("--latencies", default="0,1,2,3,5")
    return p.parse_args()


def load_silver_quotes(path: str | Path) -> pl.DataFrame:
    cols = [
        "market_id",
        "sample_ts",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "yes_is_stale",
        "no_is_stale",
        "yes_crossed_quote",
        "no_crossed_quote",
    ]
    df = read_parquet_dataset(path, cols)
    for c in ["yes_is_stale", "no_is_stale", "yes_crossed_quote", "no_crossed_quote"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(False).alias(c))
    return df.sort(["market_id", "sample_ts"])


def add_latency_quotes(base: pl.DataFrame, silver: pl.DataFrame, latency: int, horizon: int) -> pl.DataFrame:
    left = base.with_columns(
        [
            (pl.col("sample_ts") + pl.duration(seconds=latency)).alias("entry_ts"),
            (pl.col("sample_ts") + pl.duration(seconds=latency + horizon)).alias("exit_ts"),
        ]
    ).sort(["market_id", "entry_ts"])
    entry = silver.rename(
        {
            "sample_ts": "entry_quote_ts",
            "yes_bid": "entry_yes_bid",
            "yes_ask": "entry_yes_ask",
            "no_bid": "entry_no_bid",
            "no_ask": "entry_no_ask",
            "yes_is_stale": "entry_yes_is_stale",
            "no_is_stale": "entry_no_is_stale",
            "yes_crossed_quote": "entry_yes_crossed_quote",
            "no_crossed_quote": "entry_no_crossed_quote",
        }
    )
    out = left.join_asof(
        entry,
        left_on="entry_ts",
        right_on="entry_quote_ts",
        by="market_id",
        strategy="forward",
        tolerance="1s",
    )
    out = out.sort(["market_id", "exit_ts"])
    exitq = silver.rename(
        {
            "sample_ts": "exit_quote_ts",
            "yes_bid": "exit_yes_bid",
            "yes_ask": "exit_yes_ask",
            "no_bid": "exit_no_bid",
            "no_ask": "exit_no_ask",
            "yes_is_stale": "exit_yes_is_stale",
            "no_is_stale": "exit_no_is_stale",
            "yes_crossed_quote": "exit_yes_crossed_quote",
            "no_crossed_quote": "exit_no_crossed_quote",
        }
    )
    return out.join_asof(
        exitq,
        left_on="exit_ts",
        right_on="exit_quote_ts",
        by="market_id",
        strategy="forward",
        tolerance="1s",
    )


def make_latency_signals(df: pl.DataFrame, direction: str, threshold: float, latency: int, horizon: int) -> pl.DataFrame:
    if direction == "UP":
        prob = "p_up_5s"
        entry, exitp = "entry_yes_ask", "exit_yes_bid"
        stale_cols = ["entry_yes_is_stale", "exit_yes_is_stale", "entry_yes_crossed_quote", "exit_yes_crossed_quote"]
    else:
        prob = "p_down_5s"
        entry, exitp = "entry_no_ask", "exit_no_bid"
        stale_cols = ["entry_no_is_stale", "exit_no_is_stale", "entry_no_crossed_quote", "exit_no_crossed_quote"]
    cond = (
        (pl.col(prob) >= threshold)
        & (pl.col("time_to_expiry_seconds") >= latency + horizon)
        & pl.col(entry).is_not_null()
        & pl.col(exitp).is_not_null()
    )
    for c in stale_cols:
        if c in df.columns:
            cond = cond & (~pl.col(c).fill_null(True))
    return (
        df.filter(cond)
        .select(["market_id", "sample_ts", "date", "tte_bucket", "time_to_expiry_seconds", prob, entry, exitp])
        .rename({prob: "signal_probability", entry: "entry_price", exitp: "exit_price"})
        .with_columns([pl.lit(direction).alias("direction"), (pl.col("exit_price") - pl.col("entry_price")).alias("pnl")])
        .filter(pl.col("entry_price").is_finite() & pl.col("exit_price").is_finite())
    )


def metric(trades: pl.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    if trades.height == 0:
        row.update({"trades": 0, "total_pnl": 0.0, "avg_pnl": None, "win_rate": None, "median_pnl": None, "pnl_std": None, "sharpe_like": None, "max_drawdown": None, "avg_entry_price": None, "avg_exit_price": None, "avg_time_to_expiry": None})
        return row
    pnl = trades["pnl"].to_numpy().astype(float)
    std = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    avg = float(np.nanmean(pnl))
    row.update(
        {
            "trades": int(trades.height),
            "avg_entry_price": float(trades["entry_price"].mean()),
            "avg_exit_price": float(trades["exit_price"].mean()),
            "win_rate": float((trades["pnl"] > 0).mean()),
            "total_pnl": float(np.nansum(pnl)),
            "avg_pnl": avg,
            "median_pnl": float(np.nanmedian(pnl)),
            "pnl_std": std,
            "sharpe_like": None if std == 0 else avg / std,
            "max_drawdown": max_drawdown(pnl),
            "avg_time_to_expiry": float(trades["time_to_expiry_seconds"].mean()),
        }
    )
    return row


def run_latency(predictions: str | Path, silver_path: str | Path, thresholds: list[float], horizons: list[int], latencies: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = read_parquet_dataset(predictions, ["market_id", "sample_ts", "time_to_expiry_seconds", "p_up_5s", "p_down_5s", "p_flat_5s"])
    base = add_tte_bucket(base)
    silver = load_silver_quotes(silver_path)
    rows: list[dict[str, Any]] = []
    by_date: list[dict[str, Any]] = []
    by_tte: list[dict[str, Any]] = []
    for latency in latencies:
        for horizon in horizons:
            q = add_latency_quotes(base, silver, latency, horizon)
            for threshold in thresholds:
                for direction in ["UP", "DOWN"]:
                    raw = make_latency_signals(q, direction, threshold, latency, horizon)
                    for mode in COOLDOWNS:
                        trades = select_mode(raw, mode)
                        meta = {"latency_seconds": latency, "mode": mode, "direction": direction, "threshold": threshold, "exit_horizon": f"{horizon}s"}
                        rows.append(metric(trades, meta))
                        if trades.height:
                            for r in trades.group_by("date").agg(pl.len().alias("trades"), pl.col("pnl").sum().alias("total_pnl"), pl.col("pnl").mean().alias("avg_pnl"), (pl.col("pnl") > 0).mean().alias("win_rate")).to_dicts():
                                by_date.append({**meta, **r})
                            for r in trades.group_by("tte_bucket").agg(pl.len().alias("trades"), pl.col("pnl").sum().alias("total_pnl"), pl.col("pnl").mean().alias("avg_pnl"), (pl.col("pnl") > 0).mean().alias("win_rate")).to_dicts():
                                by_tte.append({**meta, **r})
    return rows, by_date, by_tte


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_int_list(args.horizons)
    latencies = parse_int_list(args.latencies)
    rows, by_date, by_tte = run_latency(args.predictions, args.silver, thresholds, horizons, latencies)
    write_csv(out_dir / "pm_repricing_latency_thresholds.csv", rows)
    write_csv(out_dir / "pm_repricing_latency_by_date.csv", by_date)
    write_csv(out_dir / "pm_repricing_latency_by_tte.csv", by_tte)
    key = [
        r
        for r in rows
        if r["mode"] == "first_signal_per_market_side"
        and r["threshold"] in {0.7, 0.75}
        and r["exit_horizon"] in {"5s", "10s", "30s"}
        and r["trades"]
    ]
    key = sorted(key, key=lambda r: (r["latency_seconds"], -(r["avg_pnl"] or -999)))
    lines = [
        "# PM Repricing Latency Sensitivity\n\n",
        "Entry and exit quotes are sampled from `data/silver/pm_1s` with ask entry and future bid exit. Stale/crossed/missing quotes are filtered.\n\n",
        "| latency | mode | direction | threshold | horizon | trades | win_rate | total_pnl | avg_pnl |\n",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |\n",
    ]
    for r in key[:80]:
        lines.append(f"| {r['latency_seconds']} | {r['mode']} | {r['direction']} | {r['threshold']} | {r['exit_horizon']} | {r['trades']} | {r['win_rate']} | {r['total_pnl']} | {r['avg_pnl']} |\n")
    (out_dir / "pm_repricing_latency_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
