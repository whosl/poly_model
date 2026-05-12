from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import polars as pl
import yaml

from preprocess.reporting import markdown_table, write_markdown

THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TTE_BUCKETS = [(240, 300, "[240, 300]"), (180, 240, "[180, 240)"), (120, 180, "[120, 180)"), (60, 120, "[60, 120)"), (30, 60, "[30, 60)"), (10, 30, "[10, 30)"), (0, 10, "[0, 10)")]


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser(description="PM repricing trading-style sanity eval.")
    ap.add_argument("--config", default="configs/model_stage1.yaml")
    ap.add_argument("--predictions", default=None)
    return ap.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh: return yaml.safe_load(fh) or {}


def bucket_expr(col: str, buckets: list[tuple[float | None, float | None, str]]) -> pl.Expr:
    expr=None; c=pl.col(col)
    for lo,hi,label in buckets:
        cond=pl.lit(True)
        if lo is not None: cond=cond & (c>=lo)
        if hi is not None: cond=cond & ((c<=hi) if label=="[240, 300]" else (c<hi))
        expr=pl.when(cond).then(pl.lit(label)) if expr is None else expr.when(cond).then(pl.lit(label))
    return expr.otherwise(pl.lit("unbucketed"))


def signal_frame(pred: pl.DataFrame, thr: float) -> pl.DataFrame:
    up=pred.filter(pl.col("p_up_5s")>=thr).with_columns(pl.lit("UP").alias("direction"), pl.lit(thr).alias("threshold"), pl.col("markout_5s").alias("directional_markout"))
    down=pred.filter(pl.col("p_down_5s")>=thr).with_columns(pl.lit("DOWN").alias("direction"), pl.lit(thr).alias("threshold"), (-pl.col("markout_5s")).alias("directional_markout"))
    cols=["market_id","sample_ts","date","threshold","direction","time_to_expiry_seconds","yes_spread","markout_5s","directional_markout"]
    return pl.concat([up.select(cols),down.select(cols)], how="vertical") if (up.height or down.height) else pl.DataFrame()


def summarize(sig: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    if sig.is_empty(): return pl.DataFrame()
    return sig.group_by(group_cols).agg(
        pl.len().alias("signals"),
        pl.col("directional_markout").mean().alias("mean_abs_directional_markout"),
        pl.col("directional_markout").median().alias("median_abs_directional_markout"),
        (pl.col("directional_markout")>0).mean().alias("positive_direction_rate"),
        (pl.col("directional_markout")>0.005).mean().alias("gt_0_5c_rate"),
        (pl.col("directional_markout")>0.01).mean().alias("gt_1c_rate"),
        (pl.col("directional_markout")>0.02).mean().alias("gt_2c_rate"),
        pl.col("yes_spread").mean().alias("avg_yes_spread"),
        (pl.col("directional_markout") - 0.5*pl.col("yes_spread")).mean().alias("markout_minus_half_spread"),
        (pl.col("directional_markout") - pl.col("yes_spread")).mean().alias("markout_minus_full_spread"),
    ).sort(group_cols)


def main() -> None:
    args=parse_args(); cfg=load_config(args.config)
    stage_dir=PROJECT_ROOT/cfg["reports"]["stage1_dir"]; stage_dir.mkdir(parents=True, exist_ok=True)
    pred_path=PROJECT_ROOT/(args.predictions or (cfg["reports"]["stage1_dir"]+"/pm_repricing_test_predictions.parquet"))
    pred=pl.read_parquet(pred_path).with_columns(pl.col("sample_ts").dt.date().cast(pl.String).alias("date"))
    frames=[]
    for thr in THRESHOLDS:
        sig=signal_frame(pred,thr)
        if not sig.is_empty(): frames.append(sig)
    all_sig=pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    if not all_sig.is_empty():
        all_sig=all_sig.with_columns(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
    thresholds=summarize(all_sig,["direction","threshold"])
    by_tte=summarize(all_sig,["direction","threshold","time_to_expiry_bucket"])
    by_date=summarize(all_sig,["direction","threshold","date"])
    thresholds.write_csv(stage_dir/"pm_repricing_sanity_thresholds.csv")
    by_tte.write_csv(stage_dir/"pm_repricing_sanity_by_tte.csv")
    by_date.write_csv(stage_dir/"pm_repricing_sanity_by_date.csv")
    lines=["# PM Repricing Sanity Eval", "", "This is not a full trading backtest; it checks whether high-confidence 5s repricing signals cover spread.", "", f"- prediction_rows: `{pred.height}`", f"- signal_rows: `{all_sig.height if not all_sig.is_empty() else 0}`", "", "## Threshold summary"]
    if not thresholds.is_empty(): lines.extend(markdown_table(thresholds.columns, thresholds.rows()))
    lines += ["", "Interpretation: if mean directional markout does not cover half spread, taker repricing is weak; full-spread coverage is needed before execution modeling."]
    write_markdown(stage_dir/"pm_repricing_sanity_report.md", lines)
    print(f"Wrote repricing sanity outputs to {stage_dir}")

if __name__ == "__main__": main()
