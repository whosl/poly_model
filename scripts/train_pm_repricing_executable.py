"""Train executable PM repricing profitable-trade classifiers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BANNED_EXACT = {"split", "market_id", "sample_ts", "market_start_ts", "market_end_ts", "date"}
BANNED_PREFIXES = ("future_", "pnl_", "roi_", "label_", "exit_", "entry_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/gold/pm_repricing_executable_1s")
    p.add_argument("--config", default="configs/model_stage1.yaml")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--model-dir", default="models/pm_repricing_executable")
    p.add_argument("--combos", default="1:10,1:5,0:10,0:5")
    return p.parse_args()


def parse_combos(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def read_dataset(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    return pl.scan_parquet(str(p / "**" / "*.parquet"), extra_columns="ignore").collect()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def check_features(features: list[str]) -> None:
    banned = [c for c in features if c in BANNED_EXACT or c.startswith(BANNED_PREFIXES)]
    if banned:
        raise ValueError(f"Forbidden executable feature columns: {banned}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    out: dict[str, Any] = {"rows": int(len(y)), "positive_rate": float(np.mean(y)) if len(y) else None}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auc": None, "logloss": None, "brier": None})
    else:
        out.update({"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))})
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        pred = (p >= thr).astype(int)
        prec, rec, _f, _sup = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
        mask = p >= thr
        out[f"thr_{thr}_signals"] = int(mask.sum())
        out[f"thr_{thr}_precision"] = float(prec[0]) if len(prec) else None
        out[f"thr_{thr}_recall"] = float(rec[0]) if len(rec) else None
    return out


def deciles(df: pl.DataFrame, prob_col: str, pnl_col: str, label_col: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    src = df.select([prob_col, pnl_col, label_col]).drop_nulls().sort(prob_col)
    n = src.height
    rows = []
    if n == 0:
        return rows
    src = src.with_row_index("_idx").with_columns(((pl.col("_idx") * 10 / n).floor().clip(0, 9).cast(pl.Int64) + 1).alias("decile"))
    for r in src.group_by("decile").agg(
        pl.len().alias("rows"),
        pl.col(prob_col).min().alias("prob_min"),
        pl.col(prob_col).max().alias("prob_max"),
        pl.col(prob_col).mean().alias("prob_mean"),
        pl.col(pnl_col).mean().alias("avg_pnl"),
        pl.col(pnl_col).median().alias("median_pnl"),
        pl.col(label_col).mean().alias("win_rate"),
    ).sort("decile").to_dicts():
        rows.append({**meta, **r})
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    lr_cfg = cfg.get("models", {}).get("logistic_regression", {})
    lgb_cfg = dict(cfg.get("models", {}).get("lightgbm", {}))
    early = int(lgb_cfg.pop("early_stopping_rounds", 50))
    lgb_cfg.setdefault("n_estimators", 600)
    lgb_cfg.setdefault("learning_rate", 0.03)
    features = json.loads((Path(args.data) / "features_pm_repricing_executable.json").read_text(encoding="utf-8"))
    check_features(features)
    df = read_dataset(args.data).with_columns(pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"))
    missing = [c for c in features + ["split", "market_id", "sample_ts"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    metric_out: dict[str, Any] = {"features": features, "models": {}}
    pred_frames = []
    decile_rows: list[dict[str, Any]] = []
    by_rows: list[dict[str, Any]] = []

    for latency, horizon in parse_combos(args.combos):
        combo = df.filter((pl.col("latency_seconds") == latency) & (pl.col("exit_horizon_seconds") == horizon))
        if combo.height == 0:
            continue
        train = combo.filter(pl.col("split") == "train")
        valid = combo.filter(pl.col("split") == "valid")
        test = combo.filter(pl.col("split") == "test")
        x_train = train.select(features).to_numpy().astype(float)
        x_valid = valid.select(features).to_numpy().astype(float)
        x_test = test.select(features).to_numpy().astype(float)
        for direction, label_col, pnl_col, spread_col, age_col in [
            ("UP", "label_up_profitable", "pnl_up", "yes_spread", "yes_quote_age_seconds"),
            ("DOWN", "label_down_profitable", "pnl_down", "no_spread", "no_quote_age_seconds"),
        ]:
            y_train = train[label_col].to_numpy().astype(int)
            y_valid = valid[label_col].to_numpy().astype(int)
            y_test = test[label_col].to_numpy().astype(int)
            key = f"lat{latency}_h{horizon}_{direction}"
            lr = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=int(lr_cfg.get("max_iter", 1000)), class_weight=lr_cfg.get("class_weight", "balanced"), n_jobs=-1)),
                ]
            )
            lr.fit(x_train, y_train)
            p_lr = lr.predict_proba(x_test)[:, 1]
            lgbm = lgb.LGBMClassifier(objective="binary", random_state=42, n_jobs=-1, verbosity=-1, class_weight="balanced", **lgb_cfg)
            lgbm.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(early, verbose=False)])
            p_lgb = lgbm.predict_proba(x_test)[:, 1]
            joblib.dump(lr, model_dir / f"logistic_{key}.joblib")
            joblib.dump(lgbm, model_dir / f"lightgbm_{key}.joblib")
            metric_out["models"][key] = {"latency_seconds": latency, "exit_horizon_seconds": horizon, "direction": direction, "logistic_regression": metrics(y_test, p_lr), "lightgbm": metrics(y_test, p_lgb)}
            pred = test.select(
                [
                    "market_id",
                    "sample_ts",
                    "date",
                    "split",
                    "time_to_expiry_seconds",
                    "latency_seconds",
                    "exit_horizon_seconds",
                    "yes_spread",
                    "no_spread",
                    "yes_quote_age_seconds",
                    "no_quote_age_seconds",
                    "entry_yes_ask",
                    "exit_yes_bid",
                    "entry_no_ask",
                    "exit_no_bid",
                    pnl_col,
                    label_col,
                ]
            ).rename({pnl_col: "pnl", label_col: "label_profitable"})
            side_spread_expr = pl.col(spread_col) if spread_col in pred.columns else pl.lit(None)
            side_age_expr = pl.col(age_col) if age_col in pred.columns else pl.lit(None)
            pred = pred.with_columns(
                [
                    pl.lit(direction).alias("direction"),
                    side_spread_expr.alias("side_spread"),
                    side_age_expr.alias("side_quote_age_seconds"),
                    pl.Series("p_logistic", p_lr),
                    pl.Series("p_model", p_lgb),
                ]
            )
            pred_frames.append(pred)
            decile_rows.extend(deciles(pred, "p_model", "pnl", "label_profitable", {"latency_seconds": latency, "exit_horizon_seconds": horizon, "direction": direction, "model": "lightgbm"}))
            for group_col in ["date"]:
                for r in pred.group_by(group_col).agg(pl.len().alias("rows"), pl.col("pnl").mean().alias("avg_pnl"), pl.col("label_profitable").mean().alias("win_rate")).to_dicts():
                    by_rows.append({"latency_seconds": latency, "exit_horizon_seconds": horizon, "direction": direction, "group_type": group_col, **r})
    preds = pl.concat(pred_frames, how="diagonal") if pred_frames else pl.DataFrame()
    preds.write_parquet(out_dir / "pm_repricing_executable_predictions.parquet")
    (out_dir / "pm_repricing_executable_model_metrics.json").write_text(json.dumps(metric_out, indent=2), encoding="utf-8")
    write_csv(out_dir / "pm_repricing_executable_deciles.csv", decile_rows)
    write_csv(out_dir / "pm_repricing_executable_model_by_group.csv", by_rows)

    lines = ["# PM Repricing Executable Model Report\n\n", "Models predict executable profitable labels, not mid-markout labels.\n\n", "| combo | model | AUC | logloss | Brier | positive_rate |\n", "| --- | --- | ---: | ---: | ---: | ---: |\n"]
    for key, m in metric_out["models"].items():
        for model_name in ["logistic_regression", "lightgbm"]:
            vals = m[model_name]
            lines.append(f"| {key} | {model_name} | {vals['auc']} | {vals['logloss']} | {vals['brier']} | {vals['positive_rate']} |\n")
    lines.append("\n## Top-decile check\n\n")
    for r in decile_rows:
        if r.get("decile") == 10:
            lines.append(f"- {r['direction']} lat={r['latency_seconds']} h={r['exit_horizon_seconds']} top decile: rows={r['rows']}, avg_pnl={r['avg_pnl']}, win_rate={r['win_rate']}\n")
    (out_dir / "pm_repricing_executable_model_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
