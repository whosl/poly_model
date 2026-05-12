from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess.reporting import markdown_table, write_json, write_markdown

BANNED_EXACT = {"settled_yes", "btc_close_price", "label_source", "split", "market_id", "sample_ts", "market_start_ts", "market_end_ts", "matched_binance_sample_ts", "binance_quote_age_seconds", "binance_is_stale", "date"}
BANNED_PREFIXES = ("future_", "markout_", "label_")
CLASSES = ["DOWN", "FLAT", "UP"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
TTE_BUCKETS = [(240, 300, "[240, 300]"), (180, 240, "[180, 240)"), (120, 180, "[120, 180)"), (60, 120, "[60, 120)"), (30, 60, "[30, 60)"), (10, 30, "[10, 30)"), (0, 10, "[0, 10)")]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Stage 1 PM repricing baselines.")
    ap.add_argument("--config", default="configs/model_stage1.yaml")
    return ap.parse_args()


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


def split_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "valid"), df.filter(pl.col("split") == "test")


def to_x(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select(features).to_numpy().astype(float)


def to_y(df: pl.DataFrame, label: str) -> np.ndarray:
    return np.asarray([CLASS_TO_INT[x] for x in df.get_column(label).to_list()], dtype=int)


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist(),
        "per_class": {CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(3)},
    }


def prob_decile_markout(df: pl.DataFrame, prob_col: str, markout_col: str, label: str) -> pl.DataFrame:
    src = df.filter(pl.col(prob_col).is_finite()).sort(prob_col)
    n = src.height
    if n == 0:
        return pl.DataFrame()
    return (src.with_row_index("_idx")
        .with_columns(((pl.col("_idx") * 10 / n).floor().clip(0,9).cast(pl.Int64)+1).alias("decile"))
        .group_by("decile")
        .agg(pl.len().alias("rows"), pl.col(prob_col).mean().alias("mean_prob"), pl.col(markout_col).mean().alias("mean_markout"), pl.col(markout_col).median().alias("median_markout"))
        .with_columns(pl.lit(label).alias("probability"))
        .select(["probability", "decile", "rows", "mean_prob", "mean_markout", "median_markout"])
        .sort(["probability", "decile"]))


def top_decile(df: pl.DataFrame, prob_col: str, markout_col: str, direction: str, horizon: str) -> dict[str, Any]:
    n = df.height
    if n == 0:
        return {"horizon": horizon, "direction": direction, "rows": 0}
    cutoff = df.get_column(prob_col).quantile(0.9)
    part = df.filter(pl.col(prob_col) >= cutoff)
    m = pl.col(markout_col)
    row = part.select(
        pl.len().alias("rows"), m.mean().alias("mean_markout_h"), m.median().alias("median_markout_h"), m.quantile(0.25).alias("p25_markout_h"), m.quantile(0.75).alias("p75_markout_h")
    ).to_dicts()[0]
    row.update({"horizon": horizon, "direction": direction, "prob_cutoff": float(cutoff)})
    if direction == "UP":
        row["win_rate_up"] = float((part.get_column(markout_col) > 0).mean() or 0.0)
        row["markout_gt_1c_rate"] = float((part.get_column(markout_col) > 0.01).mean() or 0.0)
    else:
        row["win_rate_down"] = float((part.get_column(markout_col) < 0).mean() or 0.0)
        row["markout_lt_minus_1c_rate"] = float((part.get_column(markout_col) < -0.01).mean() or 0.0)
    return row


def by_tte_metrics(df: pl.DataFrame, label_col: str, pred_col: str, horizon: str) -> list[dict[str, Any]]:
    out=[]
    d=df.with_columns(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
    for key, part in d.partition_by("time_to_expiry_bucket", as_dict=True).items():
        y=to_y(part,label_col); yp=np.asarray([CLASS_TO_INT[x] for x in part.get_column(pred_col).to_list()], dtype=int)
        row={"horizon":horizon,"time_to_expiry_bucket": key[0] if isinstance(key, tuple) else key}
        row.update({k:v for k,v in safe_metrics(y,yp).items() if k not in ["confusion_matrix","per_class"]})
        out.append(row)
    return out


def main() -> None:
    args = parse_args(); cfg = load_config(args.config)
    data_root = PROJECT_ROOT / cfg["data"]["pm_repricing"]
    stage_dir = PROJECT_ROOT / cfg["reports"]["stage1_dir"]; stage_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "models" / "pm_repricing"; model_dir.mkdir(parents=True, exist_ok=True)
    df = scan_dataset(cfg["data"]["pm_repricing"])
    features = read_json(data_root / "features_pm_repricing.json")
    labels = read_json(data_root / "labels_pm_repricing.json")
    check_features(features)
    missing = [c for c in features + labels + ["split","market_id","sample_ts"] if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    train_df, valid_df, test_df = split_rows(df)
    x_train, x_valid, x_test = to_x(train_df, features), to_x(valid_df, features), to_x(test_df, features)
    lr_cfg = cfg["models"].get("logistic_regression", {})
    lgb_cfg_base = dict(cfg["models"].get("lightgbm", {})); early = int(lgb_cfg_base.pop("early_stopping_rounds", 50))
    metrics: dict[str, Any] = {"rows":{"train":train_df.height,"valid":valid_df.height,"test":test_df.height}, "features":features, "horizons":{}}
    pred = test_df.select([c for c in ["market_id","sample_ts","split","time_to_expiry_seconds","yes_bid","yes_ask","yes_mid","no_bid","no_ask","no_mid","yes_spread","pair_mid_sum","markout_1s","markout_5s","markout_30s","label_reprice_1s","label_reprice_5s","label_reprice_30s"] if c in test_df.columns]).with_columns(pl.col("sample_ts").dt.date().cast(pl.String).alias("date"))
    top_rows=[]; tte_rows=[]; decile_frames=[]
    for label_col in labels:
        horizon = label_col.replace("label_reprice_", "")
        markout_col = f"markout_{horizon}"
        y_train, y_valid, y_test = to_y(train_df,label_col), to_y(valid_df,label_col), to_y(test_df,label_col)
        lr = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=int(lr_cfg.get("max_iter",1000)), class_weight=lr_cfg.get("class_weight","balanced"), n_jobs=-1))])
        lr.fit(x_train, y_train)
        p_lr = lr.predict_proba(x_test); y_lr = p_lr.argmax(axis=1)
        joblib.dump(lr, model_dir / f"logistic_regression_{horizon}.joblib")
        lgbm = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=42, n_jobs=-1, verbosity=-1, class_weight="balanced", **lgb_cfg_base)
        lgbm.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(early, verbose=False)])
        p_lgb = lgbm.predict_proba(x_test); y_lgb = p_lgb.argmax(axis=1)
        joblib.dump(lgbm, model_dir / f"lightgbm_{horizon}.joblib")
        metrics["horizons"][horizon] = {"logistic_regression": safe_metrics(y_test,y_lr), "lightgbm": safe_metrics(y_test,y_lgb)}
        pred_labels = [CLASSES[i] for i in y_lgb]
        pred = pred.with_columns(
            pl.Series(f"p_down_{horizon}", p_lgb[:,0]), pl.Series(f"p_flat_{horizon}", p_lgb[:,1]), pl.Series(f"p_up_{horizon}", p_lgb[:,2]), pl.Series(f"pred_reprice_{horizon}", pred_labels)
        )
        tmp = pred.select(["market_id","sample_ts","time_to_expiry_seconds", markout_col, label_col, f"p_up_{horizon}", f"p_down_{horizon}", f"pred_reprice_{horizon}"])
        top_rows.append(top_decile(tmp, f"p_up_{horizon}", markout_col, "UP", horizon))
        top_rows.append(top_decile(tmp, f"p_down_{horizon}", markout_col, "DOWN", horizon))
        decile_frames.append(prob_decile_markout(tmp, f"p_up_{horizon}", markout_col, f"p_up_{horizon}"))
        decile_frames.append(prob_decile_markout(tmp, f"p_down_{horizon}", markout_col, f"p_down_{horizon}"))
        tte_rows.extend(by_tte_metrics(pred, label_col, f"pred_reprice_{horizon}", horizon))
    pred.write_parquet(stage_dir / "pm_repricing_test_predictions.parquet")
    pl.DataFrame(top_rows).write_csv(stage_dir / "pm_repricing_top_deciles.csv")
    pl.DataFrame(tte_rows).write_csv(stage_dir / "pm_repricing_by_tte.csv")
    deciles = pl.concat(decile_frames, how="vertical") if decile_frames else pl.DataFrame()
    deciles.write_csv(stage_dir / "pm_repricing_probability_deciles.csv")
    write_json(stage_dir / "pm_repricing_metrics.json", metrics)
    (model_dir / "features_pm_repricing.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    lines=["# PM Repricing Stage 1 Baseline", "", f"- rows_train: `{train_df.height}`", f"- rows_valid: `{valid_df.height}`", f"- rows_test: `{test_df.height}`", f"- features: `{len(features)}`", ""]
    for h, vals in metrics["horizons"].items():
        lines.append(f"## Horizon {h}")
        lines.extend(markdown_table(["model","accuracy","balanced_accuracy","macro_f1"], [[m, v["accuracy"], v["balanced_accuracy"], v["macro_f1"]] for m,v in vals.items()]))
        lines.append("")
    lines.append("## LightGBM top deciles")
    td=pl.DataFrame(top_rows)
    lines.extend(markdown_table(td.columns, td.rows()))
    write_markdown(stage_dir / "pm_repricing_report.md", lines)
    print(f"Wrote PM repricing outputs to {stage_dir}")

if __name__ == "__main__":
    main()
