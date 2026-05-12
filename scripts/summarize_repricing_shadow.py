"""Summarize no-trade PM repricing shadow runs for a given date."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from run_pm_repricing_shadow import normalize_shadow_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/shadow_repricing.yaml")
    p.add_argument("--date", required=True)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def read_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_latest_diagnostics(base_dir: Path, date_str: str, run_id: str | None) -> dict[str, Any]:
    diag_dir = base_dir / "run_diagnostics" / f"date={date_str}"
    files = sorted(diag_dir.glob("run_diagnostics-*.json"))
    if not files:
        return {}
    if run_id:
        files = [p for p in files if run_id in p.name]
        if not files:
            return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_parquet_dir(base_dir: Path, subdir: str, date_str: str, run_id: str | None) -> pl.DataFrame:
    root = base_dir / subdir / f"date={date_str}"
    files = list(root.glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    if run_id:
        if "run_id" in df.columns:
            df = df.filter(pl.col("run_id") == run_id)
        else:
            return pl.DataFrame()
    if "is_forced_signal" not in df.columns and subdir == "repricing_signals":
        df = df.with_columns(pl.lit(False).alias("is_forced_signal"))
    if "is_forced_signal" not in df.columns and subdir == "repricing_outcomes":
        df = df.with_columns(pl.lit(False).alias("is_forced_signal"))
    return df


def fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    args = parse_args()
    cfg = normalize_shadow_config(read_yaml(args.config))
    base_dir = Path(cfg["output"]["base_dir"])
    diagnostics = load_latest_diagnostics(base_dir, args.date, args.run_id)
    run_id = args.run_id or diagnostics.get("run_id")
    signals = load_parquet_dir(base_dir, "repricing_signals", args.date, run_id)
    outcomes = load_parquet_dir(base_dir, "repricing_outcomes", args.date, run_id)
    latency = load_parquet_dir(base_dir, "latency_metrics", args.date, run_id)

    lines = [
        f"# Repricing Shadow Daily Report {args.date}\n\n",
        "SHADOW MODE ONLY - NO ORDERS WILL BE PLACED\n\n",
        f"- run_id: `{run_id}`\n",
        f"- signal_count: `{signals.height}`\n",
        f"- outcome_count: `{outcomes.height}`\n",
        f"- latency_metric_count: `{latency.height}`\n\n",
    ]
    if diagnostics:
        lines.extend(
            [
                "## Runtime counts\n\n",
                f"- quote snapshot count: `{diagnostics.get('quote_snapshot_count')}`\n",
                f"- feature vector count: `{diagnostics.get('feature_vector_count')}`\n",
                f"- feature_ready count: `{diagnostics.get('feature_ready_count')}`\n",
                f"- model inference count: `{diagnostics.get('model_inference_count')}`\n",
                f"- max p_up: `{fmt(diagnostics.get('max_p_up'))}`\n",
                f"- max p_down: `{fmt(diagnostics.get('max_p_down'))}`\n",
                f"- p_up quantiles: `{diagnostics.get('p_up_quantiles', {})}`\n",
                f"- p_down quantiles: `{diagnostics.get('p_down_quantiles', {})}`\n\n",
                "## filtered_by_reason\n\n",
            ]
        )
        for k, v in sorted((diagnostics.get("filtered_by_reason") or {}).items()):
            lines.append(f"- {k}: `{v}`\n")
        lines.append("\n")

    if signals.height:
        agg = signals.group_by(["config_name", "direction", "is_forced_signal"]).agg(
            pl.len().alias("signals"),
            pl.col("p_up").max().alias("max_p_up"),
            pl.col("p_down").max().alias("max_p_down"),
        ).sort(["config_name", "direction", "is_forced_signal"])
        lines.append("## Signal counts\n\n")
        lines.append("| config | direction | forced | signals | max_p_up | max_p_down |\n")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |\n")
        for r in agg.to_dicts():
            lines.append(f"| {r['config_name']} | {r['direction']} | {r['is_forced_signal']} | {r['signals']} | {fmt(r['max_p_up'])} | {fmt(r['max_p_down'])} |\n")
        lines.append("\n")

    if outcomes.height:
        agg = outcomes.group_by(["config_name", "direction", "entry_latency_ms", "exit_horizon_seconds", "is_forced_signal"]).agg(
            pl.len().alias("rows"),
            pl.col("pnl").mean().alias("avg_pnl"),
            pl.col("pnl").sum().alias("total_pnl"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
        ).sort(["config_name", "direction", "entry_latency_ms", "exit_horizon_seconds", "is_forced_signal"])
        lines.append("## Simulated PnL by latency / horizon\n\n")
        lines.append("| config | direction | latency_ms | horizon_s | forced | rows | avg_pnl | total_pnl | win_rate |\n")
        lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n")
        for r in agg.to_dicts():
            lines.append(f"| {r['config_name']} | {r['direction']} | {r['entry_latency_ms']} | {r['exit_horizon_seconds']} | {r['is_forced_signal']} | {r['rows']} | {fmt(r['avg_pnl'])} | {fmt(r['total_pnl'])} | {fmt(r['win_rate'])} |\n")
        lines.append("\n")

    if signals.height == 0:
        reasons = diagnostics.get("filtered_by_reason", {}) if diagnostics else {}
        top_reason = None if not reasons else max(reasons.items(), key=lambda kv: kv[1])
        lines.append("## why_no_signal summary\n\n")
        if top_reason is None:
            lines.append("- No diagnostics available.\n")
        else:
            lines.append(f"- top filtered reason: `{top_reason[0]}` = `{top_reason[1]}`\n")
            lines.append(f"- all filtered reasons: `{reasons}`\n")

    out = Path(cfg["output"].get("report_dir", "reports/shadow")) / f"repricing_daily_{args.date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
