from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import polars as pl
import yaml

from preprocess.reporting import markdown_table, write_markdown

TTE_BUCKETS = [(240, 300, "[240, 300]"), (180, 240, "[180, 240)"), (120, 180, "[120, 180)"), (60, 120, "[60, 120)"), (30, 60, "[30, 60)"), (10, 30, "[10, 30)"), (0, 10, "[0, 10)")]
DEFAULT_THRESHOLDS = [0.00, 0.005, 0.01, 0.02, 0.03, 0.05]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimistic maker upper-bound evaluation for PM terminal predictions.")
    ap.add_argument("--config", default="configs/model_stage1.yaml")
    ap.add_argument("--predictions", default=None)
    return ap.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def bucket_expr(col: str, buckets: list[tuple[float | None, float | None, str]]) -> pl.Expr:
    expr = None
    c = pl.col(col)
    for lo, hi, label in buckets:
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (c >= lo)
        if hi is not None:
            cond = cond & ((c <= hi) if label == "[240, 300]" else (c < hi))
        expr = pl.when(cond).then(pl.lit(label)) if expr is None else expr.when(cond).then(pl.lit(label))
    return expr.otherwise(pl.lit("unbucketed"))


def max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    curve = np.cumsum(np.asarray(pnls, dtype=float))
    peaks = np.maximum.accumulate(curve)
    return float((curve - peaks).min()) if len(curve) else 0.0


def signal_frame(pred: pl.DataFrame, threshold: float) -> pl.DataFrame:
    yes_edge = pl.col("p_model") - pl.col("yes_bid")
    no_edge = (1.0 - pl.col("p_model")) - pl.col("no_bid")
    yes = pred.filter(yes_edge >= threshold).with_columns(
        pl.lit("YES").alias("side"), pl.lit(threshold).alias("threshold"),
        pl.col("yes_bid").alias("entry_price"), (pl.col("settled_yes") - pl.col("yes_bid")).alias("pnl"),
        yes_edge.alias("maker_edge_yes_bid"), no_edge.alias("maker_edge_no_bid"),
    )
    no = pred.filter(no_edge >= threshold).with_columns(
        pl.lit("NO").alias("side"), pl.lit(threshold).alias("threshold"),
        pl.col("no_bid").alias("entry_price"), ((1 - pl.col("settled_yes")) - pl.col("no_bid")).alias("pnl"),
        yes_edge.alias("maker_edge_yes_bid"), no_edge.alias("maker_edge_no_bid"),
    )
    cols = ["market_id", "sample_ts", "date", "side", "threshold", "entry_price", "pnl", "settled_yes", "time_to_expiry_seconds", "label_source", "maker_edge_yes_bid", "maker_edge_no_bid"]
    if yes.height or no.height:
        return pl.concat([yes.select(cols), no.select(cols)], how="vertical").with_columns((pl.col("pnl") / pl.col("entry_price")).alias("roi")).sort("sample_ts")
    return pl.DataFrame()


def first_signal_per_market_side(signals: pl.DataFrame) -> pl.DataFrame:
    if signals.is_empty():
        return signals
    return signals.sort(["market_id", "side", "sample_ts"]).unique(subset=["market_id", "side"], keep="first", maintain_order=True).sort("sample_ts")


def summarize(signals: pl.DataFrame, mode: str, group_cols: list[str]) -> pl.DataFrame:
    if signals.is_empty():
        return pl.DataFrame()
    rows = []
    for key, part in signals.partition_by(group_cols, as_dict=True).items():
        if not isinstance(key, tuple):
            key = (key,)
        pnls = part.sort("sample_ts").get_column("pnl").to_list()
        pnl_std = float(part.get_column("pnl").std() or 0.0)
        row = {"mode": mode, **{c: v for c, v in zip(group_cols, key)}}
        row.update({
            "trades": part.height,
            "avg_entry_price": float(part.get_column("entry_price").mean() or 0.0),
            "win_rate": float((part.get_column("pnl") > 0).mean() or 0.0),
            "total_pnl": float(part.get_column("pnl").sum() or 0.0),
            "avg_pnl": float(part.get_column("pnl").mean() or 0.0),
            "avg_roi": float(part.get_column("roi").mean() or 0.0),
            "median_roi": float(part.get_column("roi").median() or 0.0),
            "pnl_std": pnl_std,
            "sharpe_like": None if pnl_std == 0 else float((part.get_column("pnl").mean() or 0.0) / pnl_std),
            "max_drawdown": max_drawdown(pnls),
            "avg_time_to_expiry": float(part.get_column("time_to_expiry_seconds").mean() or 0.0),
        })
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_dir = PROJECT_ROOT / cfg["reports"]["stage1_dir"]
    pred_path = PROJECT_ROOT / (args.predictions or (cfg["reports"]["stage1_dir"] + "/pm_terminal_test_predictions.parquet"))
    stage_dir.mkdir(parents=True, exist_ok=True)
    pred = pl.read_parquet(pred_path).with_columns(pl.col("sample_ts").dt.date().cast(pl.String).alias("date"))
    all_frames, first_frames = [], []
    for thr in DEFAULT_THRESHOLDS:
        sig = signal_frame(pred, thr)
        if not sig.is_empty():
            all_frames.append(sig.with_columns(pl.lit("all_signals").alias("mode")))
            first_frames.append(first_signal_per_market_side(sig).with_columns(pl.lit("first_signal_per_market_side").alias("mode")))
    all_sig = pl.concat(all_frames, how="vertical") if all_frames else pl.DataFrame()
    first_sig = pl.concat(first_frames, how="vertical") if first_frames else pl.DataFrame()

    outs = {"thresholds": [], "by_tte": [], "by_date": [], "by_label_source": []}
    for mode, sig in [("all_signals", all_sig), ("first_signal_per_market_side", first_sig)]:
        if sig.is_empty():
            continue
        sig = sig.with_columns(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
        outs["thresholds"].append(summarize(sig, mode, ["side", "threshold"]))
        outs["by_tte"].append(summarize(sig, mode, ["side", "threshold", "time_to_expiry_bucket"]))
        outs["by_date"].append(summarize(sig, mode, ["side", "threshold", "date"]))
        outs["by_label_source"].append(summarize(sig, mode, ["side", "threshold", "label_source"]))
    thresholds = pl.concat(outs["thresholds"], how="diagonal_relaxed") if outs["thresholds"] else pl.DataFrame()
    by_tte = pl.concat(outs["by_tte"], how="diagonal_relaxed") if outs["by_tte"] else pl.DataFrame()
    by_date = pl.concat(outs["by_date"], how="diagonal_relaxed") if outs["by_date"] else pl.DataFrame()
    by_label = pl.concat(outs["by_label_source"], how="diagonal_relaxed") if outs["by_label_source"] else pl.DataFrame()
    thresholds.write_csv(stage_dir / "pm_terminal_maker_upper_bound_thresholds.csv")
    by_tte.write_csv(stage_dir / "pm_terminal_maker_upper_bound_by_tte.csv")
    by_date.write_csv(stage_dir / "pm_terminal_maker_upper_bound_by_date.csv")
    by_label.write_csv(stage_dir / "pm_terminal_maker_upper_bound_by_label_source.csv")

    lines = ["# PM Terminal Maker Optimistic Upper-Bound", "", "This is not an executable backtest.", "", "Important limitations:", "- fill probability is not modeled", "- queue position is not modeled", "- adverse selection is not modeled", "- if this upper-bound is not positive, maker terminal is not worth prioritizing", "- if positive, next step is a fill / adverse-selection model", ""]
    lines.append(f"- prediction_rows: `{pred.height}`")
    lines.append(f"- all_signal_rows: `{all_sig.height if not all_sig.is_empty() else 0}`")
    lines.append(f"- first_signal_rows: `{first_sig.height if not first_sig.is_empty() else 0}`")
    lines.append("")
    lines.append("## By threshold / side")
    if not thresholds.is_empty():
        lines.extend(markdown_table(thresholds.columns, thresholds.sort(["mode", "threshold", "side"]).rows()))
    write_markdown(stage_dir / "pm_terminal_maker_upper_bound_report.md", lines)
    print(f"Wrote maker upper-bound outputs to {stage_dir}")


if __name__ == "__main__":
    main()
