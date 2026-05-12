"""Stability analysis for executable PM repricing signals."""

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
    ensure_future_quotes,
    make_signals,
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
    p.add_argument("--gold", default="data/gold/pm_repricing_1s")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--thresholds", default="0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--horizons", default="1,5,10,30")
    return p.parse_args()


def summarize(trades: pl.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    if trades.height == 0:
        row.update(
            {
                "trades": 0,
                "unique_market_count": 0,
                "avg_trades_per_market": None,
                "win_rate": None,
                "total_pnl": 0.0,
                "avg_pnl": None,
                "median_pnl": None,
                "pnl_p25": None,
                "pnl_p75": None,
                "pnl_std": None,
                "sharpe_like": None,
                "max_drawdown": None,
                "positive_date_count": 0,
                "negative_date_count": 0,
                "worst_date_pnl": None,
                "best_date_pnl": None,
                "positive_tte_bucket_count": 0,
                "negative_tte_bucket_count": 0,
                "top_market_pnl": None,
                "top_market_pnl_share_abs": None,
            }
        )
        return row
    pnl = trades["pnl"].to_numpy().astype(float)
    std = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    markets = trades.select("market_id").n_unique()
    date_pnl = trades.group_by("date").agg(pl.col("pnl").sum().alias("pnl"))
    tte_pnl = trades.group_by("tte_bucket").agg(pl.col("pnl").sum().alias("pnl"))
    market_pnl = trades.group_by("market_id").agg(pl.col("pnl").sum().alias("pnl")).sort("pnl", descending=True)
    total = float(np.nansum(pnl))
    top_market = float(market_pnl["pnl"][0]) if market_pnl.height else None
    row.update(
        {
            "trades": int(trades.height),
            "unique_market_count": int(markets),
            "avg_trades_per_market": float(trades.height / markets) if markets else None,
            "win_rate": float((trades["pnl"] > 0).mean()),
            "total_pnl": total,
            "avg_pnl": float(np.nanmean(pnl)),
            "median_pnl": float(np.nanmedian(pnl)),
            "pnl_p25": float(np.nanpercentile(pnl, 25)),
            "pnl_p75": float(np.nanpercentile(pnl, 75)),
            "pnl_std": std,
            "sharpe_like": None if std == 0 else float(np.nanmean(pnl)) / std,
            "max_drawdown": max_drawdown(pnl),
            "positive_date_count": int((date_pnl["pnl"] > 0).sum()),
            "negative_date_count": int((date_pnl["pnl"] < 0).sum()),
            "worst_date_pnl": float(date_pnl["pnl"].min()) if date_pnl.height else None,
            "best_date_pnl": float(date_pnl["pnl"].max()) if date_pnl.height else None,
            "positive_tte_bucket_count": int((tte_pnl["pnl"] > 0).sum()),
            "negative_tte_bucket_count": int((tte_pnl["pnl"] < 0).sum()),
            "top_market_pnl": top_market,
            "top_market_pnl_share_abs": None if not total else abs(top_market or 0.0) / max(abs(total), 1e-12),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_int_list(args.horizons)

    cols = [
        "market_id",
        "sample_ts",
        "time_to_expiry_seconds",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "p_up_5s",
        "p_down_5s",
        "p_flat_5s",
    ]
    df = read_parquet_dataset(args.predictions, cols)
    df = ensure_future_quotes(df, args.gold, horizons)
    df = add_tte_bucket(df)

    stability_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    tte_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    dd_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        for threshold in thresholds:
            for direction in ["UP", "DOWN"]:
                raw = make_signals(df, direction, threshold, horizon)
                for mode in COOLDOWNS:
                    trades = select_mode(raw, mode)
                    meta = {"mode": mode, "direction": direction, "threshold": threshold, "exit_horizon": f"{horizon}s"}
                    stability_rows.append(summarize(trades, meta))
                    if trades.height:
                        for r in trades.group_by("date").agg(
                            pl.len().alias("trades"),
                            pl.col("pnl").sum().alias("total_pnl"),
                            pl.col("pnl").mean().alias("avg_pnl"),
                            (pl.col("pnl") > 0).mean().alias("win_rate"),
                        ).to_dicts():
                            date_rows.append({**meta, **r})
                        for r in trades.group_by("tte_bucket").agg(
                            pl.len().alias("trades"),
                            pl.col("pnl").sum().alias("total_pnl"),
                            pl.col("pnl").mean().alias("avg_pnl"),
                            (pl.col("pnl") > 0).mean().alias("win_rate"),
                        ).to_dicts():
                            tte_rows.append({**meta, **r})
                        for r in trades.group_by("market_id").agg(
                            pl.len().alias("trades"),
                            pl.col("pnl").sum().alias("total_pnl"),
                            pl.col("pnl").mean().alias("avg_pnl"),
                            (pl.col("pnl") > 0).mean().alias("win_rate"),
                        ).sort("total_pnl", descending=True).head(200).to_dicts():
                            market_rows.append({**meta, **r})
                        equity = trades.sort("sample_ts").with_columns(pl.col("pnl").cum_sum().alias("cum_pnl"))
                        peak = equity["cum_pnl"].cum_max()
                        for i, r in enumerate(equity.select(["sample_ts", "market_id", "pnl", "cum_pnl"]).to_dicts()):
                            if i % max(1, equity.height // 200) == 0 or i == equity.height - 1:
                                dd_rows.append({**meta, **r, "drawdown": float(peak[i] - r["cum_pnl"])})

    write_csv(out_dir / "pm_repricing_stability_by_date_threshold.csv", date_rows)
    write_csv(out_dir / "pm_repricing_stability_by_tte_threshold.csv", tte_rows)
    write_csv(out_dir / "pm_repricing_stability_by_market.csv", market_rows)
    write_csv(out_dir / "pm_repricing_stability_drawdown.csv", dd_rows)

    # Keep the most important rows in the Markdown.
    key_rows = [
        r
        for r in stability_rows
        if r["mode"] in {"first_signal_per_market_side", "cooldown_10s", "cooldown_30s"}
        and r["threshold"] in {0.7, 0.75}
        and r["exit_horizon"] in {"5s", "10s", "30s"}
        and r["trades"]
    ]
    key_rows = sorted(key_rows, key=lambda r: (r["avg_pnl"] or -999), reverse=True)[:30]
    lines = [
        "# PM Repricing Stability Report\n\n",
        "Focus: thresholds 0.60-0.80, horizons 1/5/10/30s, all executable modes.\n\n",
        "## Key rows\n\n",
        "| mode | direction | threshold | horizon | trades | markets | avg_pnl | total_pnl | win_rate | positive_dates | negative_dates | worst_date_pnl | top_market_share |\n",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for r in key_rows:
        lines.append(
            f"| {r['mode']} | {r['direction']} | {r['threshold']} | {r['exit_horizon']} | {r['trades']} | "
            f"{r['unique_market_count']} | {r['avg_pnl']} | {r['total_pnl']} | {r['win_rate']} | "
            f"{r['positive_date_count']} | {r['negative_date_count']} | {r['worst_date_pnl']} | {r['top_market_pnl_share_abs']} |\n"
        )
    lines.extend(
        [
            "\n## Output files\n\n",
            "- `reports/stage1/pm_repricing_stability_by_date_threshold.csv`\n",
            "- `reports/stage1/pm_repricing_stability_by_tte_threshold.csv`\n",
            "- `reports/stage1/pm_repricing_stability_by_market.csv`\n",
            "- `reports/stage1/pm_repricing_stability_drawdown.csv`\n",
        ]
    )
    (out_dir / "pm_repricing_stability_report.md").write_text("".join(lines), encoding="utf-8")
    # Also persist the full combo table under an intuitive extra name for downstream use.
    write_csv(out_dir / "pm_repricing_stability_combos.csv", stability_rows)


if __name__ == "__main__":
    main()
