"""Slippage / stress-cost test for executable PM repricing signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pm_repricing_executable import COOLDOWNS, parse_float_list, parse_int_list, select_mode, write_csv  # noqa: E402
from evaluate_pm_repricing_latency import add_latency_quotes, add_tte_bucket, load_silver_quotes, make_latency_signals, metric, read_parquet_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="reports/stage1/pm_repricing_test_predictions.parquet")
    p.add_argument("--silver", default="data/silver/pm_1s")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--thresholds", default="0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--horizons", default="1,5,10,30")
    p.add_argument("--latencies", default="0,1,2,3,5")
    p.add_argument("--costs", default="0.00,0.005,0.01,0.02,0.03")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_int_list(args.horizons)
    latencies = parse_int_list(args.latencies)
    costs = parse_float_list(args.costs)
    base = read_parquet_dataset(args.predictions, ["market_id", "sample_ts", "time_to_expiry_seconds", "p_up_5s", "p_down_5s", "p_flat_5s"])
    base = add_tte_bucket(base)
    silver = load_silver_quotes(args.silver)

    rows: list[dict[str, Any]] = []
    for latency in latencies:
        for horizon in horizons:
            q = add_latency_quotes(base, silver, latency, horizon)
            for threshold in thresholds:
                for direction in ["UP", "DOWN"]:
                    raw = make_latency_signals(q, direction, threshold, latency, horizon)
                    for mode in COOLDOWNS:
                        trades0 = select_mode(raw, mode)
                        for cost in costs:
                            trades = trades0.with_columns((pl.col("pnl") - cost).alias("pnl")) if trades0.height else trades0
                            meta = {
                                "latency_seconds": latency,
                                "stress_cost_per_trade": cost,
                                "mode": mode,
                                "direction": direction,
                                "threshold": threshold,
                                "exit_horizon": f"{horizon}s",
                            }
                            rows.append(metric(trades, meta))
    write_csv(out_dir / "pm_repricing_stress_thresholds.csv", rows)

    # Best break-even cost by combo among first-signal/cooldown modes.
    combo: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r["mode"] not in {"first_signal_per_market_side", "cooldown_10s", "cooldown_30s"}:
            continue
        key = (r["latency_seconds"], r["mode"], r["direction"], r["threshold"], r["exit_horizon"])
        combo.setdefault(key, []).append(r)
    be_rows = []
    for key, vals in combo.items():
        vals = sorted(vals, key=lambda r: r["stress_cost_per_trade"])
        nonneg = [v for v in vals if (v.get("avg_pnl") is not None and float(v["avg_pnl"]) >= 0)]
        if nonneg:
            best = max(nonneg, key=lambda r: r["stress_cost_per_trade"])
            be_rows.append({**best, "break_even_stress_cost_floor": best["stress_cost_per_trade"]})
    be_rows = sorted(be_rows, key=lambda r: (r["latency_seconds"], -(r.get("break_even_stress_cost_floor") or 0), -(r.get("avg_pnl") or -999)))[:50]

    lines = [
        "# PM Repricing Stress Test\n\n",
        "Stress cost is subtracted from every trade PnL after executable ask-entry / bid-exit calculation.\n\n",
        "| latency | mode | direction | threshold | horizon | trades | break_even_cost_floor | avg_pnl | total_pnl |\n",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |\n",
    ]
    for r in be_rows:
        lines.append(
            f"| {r['latency_seconds']} | {r['mode']} | {r['direction']} | {r['threshold']} | {r['exit_horizon']} | {r['trades']} | "
            f"{r['break_even_stress_cost_floor']} | {r['avg_pnl']} | {r['total_pnl']} |\n"
        )
    (out_dir / "pm_repricing_stress_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
