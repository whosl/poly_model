from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess.reporting import markdown_table, write_json, write_markdown


BANNED_EXACT = {
    "settled_yes",
    "btc_close_price",
    "label_source",
    "split",
    "market_id",
    "sample_ts",
    "market_start_ts",
    "market_end_ts",
    "matched_binance_sample_ts",
    "binance_quote_age_seconds",
    "binance_is_stale",
    "date",
}
BANNED_PREFIXES = ("future_", "markout_", "label_")
EDGE_BUCKETS = [
    (None, -0.10, "[-inf, -0.10)"),
    (-0.10, -0.05, "[-0.10, -0.05)"),
    (-0.05, -0.03, "[-0.05, -0.03)"),
    (-0.03, -0.01, "[-0.03, -0.01)"),
    (-0.01, 0.00, "[-0.01, 0.00)"),
    (0.00, 0.01, "[0.00, 0.01)"),
    (0.01, 0.03, "[0.01, 0.03)"),
    (0.03, 0.05, "[0.03, 0.05)"),
    (0.05, 0.10, "[0.05, 0.10)"),
    (0.10, None, "[0.10, inf)"),
]
TTE_BUCKETS = [(240, 300, "[240, 300]"), (180, 240, "[180, 240)"), (120, 180, "[120, 180)"), (60, 120, "[60, 120)"), (30, 60, "[30, 60)"), (10, 30, "[10, 30)"), (0, 10, "[0, 10)")]
YES_ASK_BUCKETS = [(None, 0.10, "[-inf, 0.10)"), (0.10, 0.25, "[0.10, 0.25)"), (0.25, 0.40, "[0.25, 0.40)"), (0.40, 0.60, "[0.40, 0.60)"), (0.60, 0.75, "[0.60, 0.75)"), (0.75, 0.90, "[0.75, 0.90)"), (0.90, None, "[0.90, inf)")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 PM terminal baseline models.")
    parser.add_argument("--config", default="configs/model_stage1.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def scan_dataset(path: str | Path) -> pl.DataFrame:
    root = PROJECT_ROOT / Path(path)
    return pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True).collect()


def read_json(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_features(features: list[str]) -> None:
    banned = [c for c in features if c in BANNED_EXACT or c.startswith(BANNED_PREFIXES)]
    if banned:
        raise ValueError(f"Forbidden/leaky feature columns detected: {banned}")


def check_market_split(df: pl.DataFrame) -> None:
    leakage = df.group_by("market_id").agg(pl.col("split").n_unique().alias("split_count")).filter(pl.col("split_count") > 1)
    if leakage.height:
        raise ValueError(f"market_id crosses splits for {leakage.height} markets")


def to_xy(df: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df.select(features).to_numpy().astype(float)
    y = df.get_column("settled_yes").to_numpy().astype(int)
    return x, y


def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(p) & np.isfinite(y)
    y2 = y[mask].astype(int)
    p2 = np.clip(p[mask].astype(float), 1e-6, 1 - 1e-6)
    if len(y2) == 0:
        return {"rows": 0, "auc": None, "logloss": None, "brier": None, "accuracy_0p5": None}
    auc = None if len(np.unique(y2)) < 2 else float(roc_auc_score(y2, p2))
    return {
        "rows": int(len(y2)),
        "auc": auc,
        "logloss": float(log_loss(y2, p2, labels=[0, 1])),
        "brier": float(brier_score_loss(y2, p2)),
        "accuracy_0p5": float(accuracy_score(y2, p2 >= 0.5)),
    }




def add_market_baseline_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str], list[str]]:
    """Add market-implied probability baselines when source columns exist."""
    model_cols: dict[str, str] = {}
    skipped: list[str] = []
    exprs = []
    if "yes_mid" in df.columns:
        exprs.append(pl.col("yes_mid").cast(pl.Float64).alias("p_market_mid"))
        model_cols["market_yes_mid"] = "p_market_mid"
    else:
        skipped.append("market_yes_mid: missing yes_mid")
    if "yes_mid" in df.columns and "no_mid" in df.columns:
        exprs.append((pl.col("yes_mid") / (pl.col("yes_mid") + pl.col("no_mid"))).cast(pl.Float64).alias("p_market_norm"))
        model_cols["normalized_market_mid"] = "p_market_norm"
    else:
        skipped.append("normalized_market_mid: missing yes_mid or no_mid")
    if "yes_bid" in df.columns and "yes_ask" in df.columns:
        exprs.append(((pl.col("yes_bid") + pl.col("yes_ask")) / 2.0).cast(pl.Float64).alias("p_market_ba"))
        model_cols["market_bid_ask_mid"] = "p_market_ba"
    else:
        skipped.append("market_bid_ask_mid: missing yes_bid or yes_ask")
    if exprs:
        df = df.with_columns(exprs)
    return df, model_cols, skipped


def metrics_rows(metrics: dict[str, Any], split: str = "test") -> pl.DataFrame:
    rows = []
    for model, vals in metrics.get(split, {}).items():
        rows.append({"split": split, "model": model, **vals})
    return pl.DataFrame(rows)

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


def grouped_metrics(df: pl.DataFrame, prob_col: str, group_col: str) -> list[dict[str, Any]]:
    rows = []
    for key_df in df.partition_by(group_col, as_dict=False):
        key = key_df.get_column(group_col)[0]
        y = key_df.get_column("settled_yes").to_numpy().astype(int)
        p = key_df.get_column(prob_col).to_numpy().astype(float)
        row = {group_col: key}
        row.update(safe_metrics(y, p))
        rows.append(row)
    return rows


def calibration_table(df: pl.DataFrame, model_cols: dict[str, str]) -> pl.DataFrame:
    frames = []
    edges = [i / 10 for i in range(11)]
    for model, col in model_cols.items():
        d = df.filter(pl.col(col).is_finite()).with_columns(
            pl.when(pl.col(col) >= 1.0)
            .then(pl.lit(9))
            .otherwise((pl.col(col) * 10).floor().cast(pl.Int64).clip(0, 9))
            .alias("prob_bin")
        )
        tab = d.group_by("prob_bin").agg(
            pl.len().alias("rows"),
            pl.col(col).mean().alias("mean_pred"),
            pl.col("settled_yes").mean().alias("settled_yes_rate"),
        ).with_columns(
            pl.lit(model).alias("model"),
            pl.col("prob_bin").map_elements(lambda i: f"[{edges[int(i)]:.1f}, {edges[int(i)+1]:.1f})" if int(i) < 9 else "[0.9, 1.0]", return_dtype=pl.String).alias("prob_bucket")
        ).select(["model", "prob_bin", "prob_bucket", "rows", "mean_pred", "settled_yes_rate"]).sort(["model", "prob_bin"])
        frames.append(tab)
    return pl.concat(frames, how="vertical")


def decile_table(df: pl.DataFrame, model_cols: dict[str, str]) -> pl.DataFrame:
    frames = []
    for model, col in model_cols.items():
        src = df.filter(pl.col(col).is_finite()).sort(col)
        n = src.height
        if n == 0:
            continue
        tab = src.with_row_index("_idx").with_columns(((pl.col("_idx") * 10 / n).floor().clip(0, 9).cast(pl.Int64) + 1).alias("decile")).group_by("decile").agg(
            pl.len().alias("rows"),
            pl.col(col).min().alias("pred_min"),
            pl.col(col).max().alias("pred_max"),
            pl.col(col).mean().alias("mean_pred"),
            pl.col("settled_yes").mean().alias("settled_yes_rate"),
        ).with_columns(pl.lit(model).alias("model")).select(["model", "decile", "rows", "pred_min", "pred_max", "mean_pred", "settled_yes_rate"]).sort(["model", "decile"])
        frames.append(tab)
    return pl.concat(frames, how="vertical") if frames else pl.DataFrame()


def split_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return (df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "valid"), df.filter(pl.col("split") == "test"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = PROJECT_ROOT / cfg["data"]["pm_terminal"]
    stage_dir = PROJECT_ROOT / cfg["reports"]["stage1_dir"]
    model_dir = PROJECT_ROOT / "models" / "pm_terminal"
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(cfg["data"]["pm_terminal"])
    features = read_json(data_root / "features_pm_terminal.json")
    labels = read_json(data_root / "labels_pm_terminal.json")
    if labels != ["settled_yes"]:
        raise ValueError(f"Unexpected labels: {labels}")
    check_features(features)
    missing = [c for c in features + ["settled_yes", "split", "market_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    check_market_split(df)
    df = df.filter(pl.col("settled_yes").is_not_null())
    train_df, valid_df, test_df = split_rows(df)
    print(f"Rows train={train_df.height} valid={valid_df.height} test={test_df.height}; features={len(features)}")

    x_train, y_train = to_xy(train_df, features)
    x_valid, y_valid = to_xy(valid_df, features)
    x_test, y_test = to_xy(test_df, features)

    # Formula baseline.
    p_formula_valid = valid_df.get_column("formula_p_yes").to_numpy().astype(float)
    p_formula_test = test_df.get_column("formula_p_yes").to_numpy().astype(float)

    lr_cfg = cfg["models"].get("logistic_regression", {})
    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=int(lr_cfg.get("max_iter", 1000)), class_weight=lr_cfg.get("class_weight", None), n_jobs=-1)),
        ]
    )
    lr.fit(x_train, y_train)
    p_lr_valid = lr.predict_proba(x_valid)[:, 1]
    p_lr_test = lr.predict_proba(x_test)[:, 1]
    joblib.dump(lr, model_dir / "logistic_regression.joblib")

    lgb_cfg = cfg["models"].get("lightgbm", {})
    early_stopping_rounds = int(lgb_cfg.pop("early_stopping_rounds", 50)) if "early_stopping_rounds" in lgb_cfg else 50
    lgbm = lgb.LGBMClassifier(objective="binary", random_state=42, n_jobs=-1, verbosity=-1, **lgb_cfg)
    lgbm.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    p_lgb_valid = lgbm.predict_proba(x_valid)[:, 1]
    p_lgb_test = lgbm.predict_proba(x_test)[:, 1]
    joblib.dump(lgbm, model_dir / "lightgbm.joblib")
    (model_dir / "features_pm_terminal.json").write_text(json.dumps(features, indent=2), encoding="utf-8")

    metrics = {
        "valid": {
            "formula_p_yes": safe_metrics(y_valid, p_formula_valid),
            "logistic_regression": safe_metrics(y_valid, p_lr_valid),
            "lightgbm": safe_metrics(y_valid, p_lgb_valid),
        },
        "test": {
            "formula_p_yes": safe_metrics(y_test, p_formula_test),
            "logistic_regression": safe_metrics(y_test, p_lr_test),
            "lightgbm": safe_metrics(y_test, p_lgb_test),
        },
        "rows": {"train": train_df.height, "valid": valid_df.height, "test": test_df.height},
        "features": features,
        "market_baseline_skipped": [],
    }

    pred = test_df.select(
        [
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
            "no_spread",
            "pair_mid_sum",
            "formula_p_yes",
            "settled_yes",
            "label_source",
        ]
    ).with_columns(
        pl.Series("p_formula", p_formula_test),
        pl.Series("p_logistic", p_lr_test),
        pl.Series("p_lightgbm", p_lgb_test),
        pl.Series("p_model", p_lgb_test),
    ).with_columns(
        (pl.col("p_model") - pl.col("yes_ask")).alias("edge_model_yes_ask"),
        ((1.0 - pl.col("p_model")) - pl.col("no_ask")).alias("edge_model_no_ask"),
        pl.col("sample_ts").dt.date().cast(pl.String).alias("date"),
    )
    pred, market_model_cols, market_skipped = add_market_baseline_columns(pred)
    metrics["market_baseline_skipped"] = market_skipped
    for model_name, prob_col in market_model_cols.items():
        metrics["test"][model_name] = safe_metrics(y_test, pred.get_column(prob_col).to_numpy().astype(float))
    pred.write_parquet(stage_dir / "pm_terminal_test_predictions.parquet")

    model_cols = {"formula_p_yes": "p_formula", "logistic_regression": "p_logistic", "lightgbm": "p_lightgbm"}
    model_cols.update(market_model_cols)
    cal = calibration_table(pred, model_cols)
    dec = decile_table(pred, model_cols)
    cal.write_csv(stage_dir / "pm_terminal_calibration.csv")
    dec.write_csv(stage_dir / "pm_terminal_deciles.csv")

    pred_bucketed = pred.with_columns(
        bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"),
        bucket_expr("yes_ask", YES_ASK_BUCKETS).alias("yes_ask_bucket"),
        bucket_expr("edge_model_yes_ask", EDGE_BUCKETS).alias("edge_model_yes_ask_bucket"),
    )
    metrics["test_by_time_to_expiry_bucket"] = grouped_metrics(pred_bucketed, "p_model", "time_to_expiry_bucket")
    metrics["test_by_yes_ask_bucket"] = grouped_metrics(pred_bucketed, "p_model", "yes_ask_bucket")
    metrics["test_by_edge_bucket"] = grouped_metrics(pred_bucketed, "p_model", "edge_model_yes_ask_bucket")

    market_csv = metrics_rows(metrics, "test")
    market_csv.write_csv(stage_dir / "pm_terminal_market_baselines.csv")
    market_lines = ["# PM Terminal Market Baseline Comparison", ""]
    market_lines.append("This compares LightGBM/formula/logistic probabilities against Polymarket-implied probabilities on the PM terminal test split.")
    if market_skipped:
        market_lines.append("")
        market_lines.append("## Skipped baselines")
        for item in market_skipped:
            market_lines.append(f"- {item}")
    market_lines.append("")
    market_lines.append("## Test metrics")
    market_lines.extend(markdown_table(market_csv.columns, market_csv.rows()))
    write_markdown(stage_dir / "pm_terminal_market_baseline_report.md", market_lines)

    write_json(stage_dir / "pm_terminal_metrics.json", metrics)

    lines = ["# PM Terminal Stage 1 Baseline", ""]
    lines.append(f"- rows_train: `{train_df.height}`")
    lines.append(f"- rows_valid: `{valid_df.height}`")
    lines.append(f"- rows_test: `{test_df.height}`")
    lines.append(f"- features: `{len(features)}`")
    lines.append("")
    lines.append("## Test metrics")
    metric_rows = []
    for model, vals in metrics["test"].items():
        metric_rows.append([model, vals["rows"], vals["auc"], vals["logloss"], vals["brier"], vals["accuracy_0p5"]])
    lines.extend(markdown_table(["model", "rows", "auc", "logloss", "brier", "accuracy_0p5"], metric_rows))
    lines.append("")
    lines.append("## LightGBM by time_to_expiry_bucket")
    lines.extend(markdown_table(["bucket", "rows", "auc", "logloss", "brier", "accuracy_0p5"], [[r["time_to_expiry_bucket"], r["rows"], r["auc"], r["logloss"], r["brier"], r["accuracy_0p5"]] for r in metrics["test_by_time_to_expiry_bucket"]]))
    lines.append("")
    lines.append("## LightGBM by yes_ask_bucket")
    lines.extend(markdown_table(["bucket", "rows", "auc", "logloss", "brier", "accuracy_0p5"], [[r["yes_ask_bucket"], r["rows"], r["auc"], r["logloss"], r["brier"], r["accuracy_0p5"]] for r in metrics["test_by_yes_ask_bucket"]]))
    lines.append("")
    lines.append("## LightGBM by edge_model_yes_ask_bucket")
    lines.extend(markdown_table(["bucket", "rows", "auc", "logloss", "brier", "accuracy_0p5"], [[r["edge_model_yes_ask_bucket"], r["rows"], r["auc"], r["logloss"], r["brier"], r["accuracy_0p5"]] for r in metrics["test_by_edge_bucket"]]))
    lines.append("")
    lines.append("## Output files")
    lines.extend([
        "- `reports/stage1/pm_terminal_metrics.json`",
        "- `reports/stage1/pm_terminal_test_predictions.parquet`",
        "- `reports/stage1/pm_terminal_calibration.csv`",
        "- `reports/stage1/pm_terminal_deciles.csv`",
        "- `reports/stage1/pm_terminal_market_baselines.csv`",
        "- `reports/stage1/pm_terminal_market_baseline_report.md`",
        "- `models/pm_terminal/lightgbm.joblib`",
        "- `models/pm_terminal/logistic_regression.joblib`",
    ])
    write_markdown(stage_dir / "pm_terminal_report.md", lines)
    print(f"Wrote PM terminal training outputs to {stage_dir}")


if __name__ == "__main__":
    main()
