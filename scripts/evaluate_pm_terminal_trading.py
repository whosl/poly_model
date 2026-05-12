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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PM terminal trading rules from test predictions.")
    parser.add_argument("--config", default="configs/model_stage1.yaml")
    parser.add_argument("--predictions", default=None)
    return parser.parse_args()


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
        if expr is None:
            expr = pl.when(cond).then(pl.lit(label))
        else:
            expr = expr.when(cond).then(pl.lit(label))
    return expr.otherwise(pl.lit("unbucketed"))


def max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    curve = np.cumsum(np.asarray(pnls, dtype=float))
    peaks = np.maximum.accumulate(curve)
    dd = curve - peaks
    return float(dd.min()) if len(dd) else 0.0


def signal_frame(pred: pl.DataFrame, threshold: float) -> pl.DataFrame:
    yes = pred.filter(pl.col("edge_model_yes_ask") >= threshold).with_columns(
        pl.lit("YES").alias("side"),
        pl.lit(threshold).alias("threshold"),
        pl.col("yes_ask").alias("entry_price"),
        (pl.col("settled_yes") - pl.col("yes_ask")).alias("pnl"),
    )
    no = pred.filter(pl.col("edge_model_no_ask") >= threshold).with_columns(
        pl.lit("NO").alias("side"),
        pl.lit(threshold).alias("threshold"),
        pl.col("no_ask").alias("entry_price"),
        ((1 - pl.col("settled_yes")) - pl.col("no_ask")).alias("pnl"),
    )
    cols = [
        "market_id",
        "sample_ts",
        "date",
        "side",
        "threshold",
        "entry_price",
        "pnl",
        "settled_yes",
        "time_to_expiry_seconds",
        "label_source",
        "edge_model_yes_ask",
        "edge_model_no_ask",
    ]
    out = pl.concat([yes.select(cols), no.select(cols)], how="vertical") if (yes.height or no.height) else pl.DataFrame(schema={c: pl.Float64 for c in cols})
    if out.is_empty():
        return out
    return out.with_columns((pl.col("pnl") / pl.col("entry_price")).alias("roi")).sort("sample_ts")


def first_signal_per_market_side(signals: pl.DataFrame) -> pl.DataFrame:
    if signals.is_empty():
        return signals
    return signals.sort(["market_id", "side", "sample_ts"]).unique(subset=["market_id", "side"], keep="first", maintain_order=True).sort("sample_ts")


def summarize(signals: pl.DataFrame, mode: str, group_cols: list[str]) -> pl.DataFrame:
    if signals.is_empty():
        schema = {"mode": pl.String, **{c: pl.String for c in group_cols}, "trades": pl.Int64}
        return pl.DataFrame(schema=schema)
    rows = []
    for key, part in signals.partition_by(group_cols, as_dict=True).items():
        if not isinstance(key, tuple):
            key = (key,)
        pnls = part.sort("sample_ts").get_column("pnl").to_list()
        pnl_std = float(part.get_column("pnl").std() or 0.0)
        row = {"mode": mode}
        row.update({col: val for col, val in zip(group_cols, key)})
        row.update(
            {
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
                "min_time_to_expiry": float(part.get_column("time_to_expiry_seconds").min() or 0.0),
                "max_time_to_expiry": float(part.get_column("time_to_expiry_seconds").max() or 0.0),
            }
        )
        rows.append(row)
    return pl.DataFrame(rows)


def build_all_signals(pred: pl.DataFrame, thresholds: list[float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_frames = []
    first_frames = []
    for thr in thresholds:
        sig = signal_frame(pred, thr)
        if not sig.is_empty():
            all_frames.append(sig.with_columns(pl.lit("all_signals").alias("mode")))
            first_frames.append(first_signal_per_market_side(sig).with_columns(pl.lit("first_signal_per_market_side").alias("mode")))
    all_sig = pl.concat(all_frames, how="vertical") if all_frames else pl.DataFrame()
    first_sig = pl.concat(first_frames, how="vertical") if first_frames else pl.DataFrame()
    return all_sig, first_sig


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_dir = PROJECT_ROOT / cfg["reports"]["stage1_dir"]
    pred_path = PROJECT_ROOT / (args.predictions or (cfg["reports"]["stage1_dir"] + "/pm_terminal_test_predictions.parquet"))
    thresholds = [float(x) for x in cfg["evaluation"]["edge_thresholds"]]
    stage_dir.mkdir(parents=True, exist_ok=True)

    pred = pl.read_parquet(pred_path).with_columns(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
    all_sig, first_sig = build_all_signals(pred, thresholds)
    combined = pl.concat([x for x in [all_sig, first_sig] if not x.is_empty()], how="vertical") if (not all_sig.is_empty() or not first_sig.is_empty()) else pl.DataFrame()

    threshold_frames = []
    tte_frames = []
    date_frames = []
    label_frames = []
    for mode, sig in [("all_signals", all_sig), ("first_signal_per_market_side", first_sig)]:
        if sig.is_empty():
            continue
        sig = sig.with_columns(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
        threshold_frames.append(summarize(sig, mode, ["side", "threshold"]))
        tte_frames.append(summarize(sig, mode, ["side", "threshold", "time_to_expiry_bucket"]))
        date_frames.append(summarize(sig, mode, ["side", "threshold", "date"]))
        label_frames.append(summarize(sig, mode, ["side", "threshold", "label_source"]))

    thresholds_out = pl.concat(threshold_frames, how="diagonal_relaxed") if threshold_frames else pl.DataFrame()
    by_tte = pl.concat(tte_frames, how="diagonal_relaxed") if tte_frames else pl.DataFrame()
    by_date = pl.concat(date_frames, how="diagonal_relaxed") if date_frames else pl.DataFrame()
    by_label = pl.concat(label_frames, how="diagonal_relaxed") if label_frames else pl.DataFrame()

    thresholds_out.write_csv(stage_dir / "pm_terminal_trading_thresholds.csv")
    by_tte.write_csv(stage_dir / "pm_terminal_trading_by_tte.csv")
    by_date.write_csv(stage_dir / "pm_terminal_trading_by_date.csv")
    by_label.write_csv(stage_dir / "pm_terminal_trading_by_label_source.csv")

    lines = ["# PM Terminal Trading Evaluation", ""]
    lines.append(f"- prediction_rows: `{pred.height}`")
    lines.append(f"- all_signal_rows: `{all_sig.height if not all_sig.is_empty() else 0}`")
    lines.append(f"- first_signal_rows: `{first_sig.height if not first_sig.is_empty() else 0}`")
    lines.append("")
    lines.append("## By threshold / side")
    if not thresholds_out.is_empty():
        lines.extend(markdown_table(thresholds_out.columns, thresholds_out.sort(["mode", "threshold", "side"]).rows()))
    lines.append("")
    lines.append("## First-signal highlights")
    if not thresholds_out.is_empty():
        hi = thresholds_out.filter(pl.col("mode") == "first_signal_per_market_side").sort(["total_pnl"], descending=True).head(10)
        lines.extend(markdown_table(hi.columns, hi.rows()))
    lines.append("")
    lines.append("## Output files")
    lines.extend([
        "- `reports/stage1/pm_terminal_trading_thresholds.csv`",
        "- `reports/stage1/pm_terminal_trading_by_tte.csv`",
        "- `reports/stage1/pm_terminal_trading_by_date.csv`",
        "- `reports/stage1/pm_terminal_trading_by_label_source.csv`",
    ])
    write_markdown(stage_dir / "pm_terminal_trading_report.md", lines)
    print(f"Wrote PM terminal trading evaluation to {stage_dir}")


if __name__ == "__main__":
    main()
