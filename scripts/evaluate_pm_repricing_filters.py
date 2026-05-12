"""Spread / quote-quality filter grid for executable repricing predictions."""

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

from evaluate_pm_repricing_executable import COOLDOWNS, add_tte_bucket, max_drawdown, select_mode, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="reports/stage1/pm_repricing_executable_predictions.parquet")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--prob-thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--max-spreads", default="0.02,0.03,0.05,0.08,0.10,null")
    p.add_argument("--max-quote-ages", default="0.5,1.0,2.0,5.0")
    p.add_argument("--min-ttes", default="5,10,30,60")
    return p.parse_args()


def floats(s: str, allow_null: bool = False) -> list[float | None]:
    out: list[float | None] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if allow_null and x.lower() in {"null", "none", "nan"}:
            out.append(None)
        else:
            out.append(float(x))
    return out


def read(path: str | Path) -> pl.DataFrame:
    return pl.scan_parquet(str(path), extra_columns="ignore").collect()


def summarize(trades: pl.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    if trades.height == 0:
        row.update({"trades": 0, "unique_markets": 0, "avg_trades_per_market": None, "win_rate": None, "total_pnl": 0.0, "avg_pnl": None, "median_pnl": None, "pnl_p25": None, "pnl_p75": None, "pnl_std": None, "sharpe_like": None, "max_drawdown": None, "positive_date_count": 0, "negative_date_count": 0, "worst_date_pnl": None, "best_date_pnl": None, "positive_tte_bucket_count": 0, "negative_tte_bucket_count": 0})
        return row
    pnl = trades["pnl"].to_numpy().astype(float)
    std = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    markets = trades.select("market_id").n_unique()
    date_pnl = trades.group_by("date").agg(pl.col("pnl").sum().alias("pnl"))
    tte_pnl = trades.group_by("tte_bucket").agg(pl.col("pnl").sum().alias("pnl"))
    avg = float(np.nanmean(pnl))
    row.update(
        {
            "trades": int(trades.height),
            "unique_markets": int(markets),
            "avg_trades_per_market": float(trades.height / markets) if markets else None,
            "win_rate": float((trades["pnl"] > 0).mean()),
            "total_pnl": float(np.nansum(pnl)),
            "avg_pnl": avg,
            "median_pnl": float(np.nanmedian(pnl)),
            "pnl_p25": float(np.nanpercentile(pnl, 25)),
            "pnl_p75": float(np.nanpercentile(pnl, 75)),
            "pnl_std": std,
            "sharpe_like": None if std == 0 else avg / std,
            "max_drawdown": max_drawdown(pnl),
            "positive_date_count": int((date_pnl["pnl"] > 0).sum()),
            "negative_date_count": int((date_pnl["pnl"] < 0).sum()),
            "worst_date_pnl": float(date_pnl["pnl"].min()) if date_pnl.height else None,
            "best_date_pnl": float(date_pnl["pnl"].max()) if date_pnl.height else None,
            "positive_tte_bucket_count": int((tte_pnl["pnl"] > 0).sum()),
            "negative_tte_bucket_count": int((tte_pnl["pnl"] < 0).sum()),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_tte_bucket(read(args.predictions)).sort(["market_id", "direction", "sample_ts"])
    thresholds = [float(x) for x in floats(args.prob_thresholds)]
    spreads = floats(args.max_spreads, allow_null=True)
    ages = [float(x) for x in floats(args.max_quote_ages)]
    min_ttes = [float(x) for x in floats(args.min_ttes)]
    grid_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    tte_rows: list[dict[str, Any]] = []
    combos = df.select(["latency_seconds", "exit_horizon_seconds"]).unique().sort(["latency_seconds", "exit_horizon_seconds"]).to_dicts()
    for combo in combos:
        latency = combo["latency_seconds"]
        horizon = combo["exit_horizon_seconds"]
        d_combo = df.filter((pl.col("latency_seconds") == latency) & (pl.col("exit_horizon_seconds") == horizon))
        for direction in ["UP", "DOWN"]:
            d0 = d_combo.filter(pl.col("direction") == direction)
            for thr in thresholds:
                for max_spread in spreads:
                    for max_age in ages:
                        for min_tte in min_ttes:
                            cond = (pl.col("p_model") >= thr) & (pl.col("side_quote_age_seconds") <= max_age) & (pl.col("time_to_expiry_seconds") >= min_tte)
                            if max_spread is not None:
                                cond = cond & (pl.col("side_spread") <= max_spread)
                            raw = d0.filter(cond)
                            for mode in COOLDOWNS:
                                trades = select_mode(raw, mode)
                                meta = {"latency_seconds": latency, "exit_horizon_seconds": horizon, "direction": direction, "prob_threshold": thr, "max_spread": max_spread, "max_quote_age": max_age, "min_time_to_expiry": min_tte, "mode": mode}
                                grid_rows.append(summarize(trades, meta))
                                if trades.height:
                                    for r in trades.group_by("date").agg(pl.len().alias("trades"), pl.col("pnl").sum().alias("total_pnl"), pl.col("pnl").mean().alias("avg_pnl"), (pl.col("pnl") > 0).mean().alias("win_rate")).to_dicts():
                                        date_rows.append({**meta, **r})
                                    for r in trades.group_by("tte_bucket").agg(pl.len().alias("trades"), pl.col("pnl").sum().alias("total_pnl"), pl.col("pnl").mean().alias("avg_pnl"), (pl.col("pnl") > 0).mean().alias("win_rate")).to_dicts():
                                        tte_rows.append({**meta, **r})
    write_csv(out_dir / "pm_repricing_filter_grid.csv", grid_rows)
    write_csv(out_dir / "pm_repricing_filter_by_date.csv", date_rows)
    write_csv(out_dir / "pm_repricing_filter_by_tte.csv", tte_rows)
    best = [r for r in grid_rows if r["mode"] in {"first_signal_per_market_side", "cooldown_10s", "cooldown_30s"} and r["trades"]]
    best = sorted(best, key=lambda r: (r["avg_pnl"] or -999, r["trades"]), reverse=True)[:40]
    lines = ["# PM Repricing Filter Grid Report\n\n", "Filters are applied to executable model predictions and evaluated separately by latency/horizon combo.\n\n", "| latency | horizon | mode | direction | prob | max_spread | max_age | min_tte | trades | avg_pnl | total_pnl | win_rate | pos_dates | neg_dates |\n", "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"]
    for r in best:
        lines.append(f"| {r['latency_seconds']} | {r['exit_horizon_seconds']} | {r['mode']} | {r['direction']} | {r['prob_threshold']} | {r['max_spread']} | {r['max_quote_age']} | {r['min_time_to_expiry']} | {r['trades']} | {r['avg_pnl']} | {r['total_pnl']} | {r['win_rate']} | {r['positive_date_count']} | {r['negative_date_count']} |\n")
    (out_dir / "pm_repricing_filter_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
