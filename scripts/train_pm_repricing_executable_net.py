"""Train executable PM repricing models on net taker PnL labels.

This is stricter than the current executable model:

* Entry is taker buy at ask.
* Exit is taker sell at bid.
* Net PnL subtracts Polymarket taker fees on entry and exit.
* Net PnL can also subtract an additional per-share slippage/safety buffer.
* Labels can require a positive net edge buffer, not merely > 0.

The goal is to select thresholds by out-of-sample net PnL, not classification
accuracy. This is the first training target aligned with live taker execution.
"""

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
    p.add_argument("--out-dir", default="reports/stage1/net_exec_v1")
    p.add_argument("--model-dir", default="models/pm_repricing_executable_net_v1")
    p.add_argument("--combos", default="1:5,1:10,0:5,0:10")
    p.add_argument("--fee-rate", type=float, default=0.07, help="Polymarket crypto taker feeRate in fee=C*r*p*(1-p).")
    p.add_argument("--slippage-buffer", type=float, default=0.0025, help="Extra per-share adverse execution buffer.")
    p.add_argument("--edge-buffer", type=float, default=0.0, help="Label positive only if net pnl > this value.")
    p.add_argument("--min-tte", type=float, default=30.0)
    p.add_argument("--max-spread", type=float, default=0.05)
    p.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    return p.parse_args()


def parse_combos(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in s.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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


def taker_fee_expr(price_col: str, fee_rate: float) -> pl.Expr:
    p = pl.col(price_col).clip(0.0, 1.0)
    return fee_rate * p * (1.0 - p)


def model_metrics(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    out: dict[str, Any] = {"rows": int(len(y)), "positive_rate": float(np.mean(y)) if len(y) else None}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auc": None, "logloss": None, "brier": None})
    else:
        out.update({"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))})
    for thr in thresholds:
        pred = (p >= thr).astype(int)
        prec, rec, _f, _sup = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
        mask = p >= thr
        out[f"thr_{thr}_signals"] = int(mask.sum())
        out[f"thr_{thr}_precision"] = float(prec[0]) if len(prec) else None
        out[f"thr_{thr}_recall"] = float(rec[0]) if len(rec) else None
    return out


def max_drawdown(pnl: np.ndarray) -> float | None:
    if len(pnl) == 0:
        return None
    equity = np.cumsum(pnl.astype(float))
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def threshold_rows(pred: pl.DataFrame, thresholds: list[float], meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        trades = pred.filter(pl.col("p_model") >= thr).sort(["market_id", "sample_ts"])
        row: dict[str, Any] = {**meta, "threshold": thr, "trades": trades.height}
        if trades.height == 0:
            row.update({"win_rate": None, "total_net_pnl": 0.0, "avg_net_pnl": None, "median_net_pnl": None, "pnl_std": None, "sharpe_like": None, "max_drawdown": None})
        else:
            pnl = trades["net_pnl"].to_numpy().astype(float)
            avg = float(np.nanmean(pnl))
            std = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else 0.0
            row.update(
                {
                    "win_rate": float((trades["net_pnl"] > 0).mean()),
                    "total_net_pnl": float(np.nansum(pnl)),
                    "avg_net_pnl": avg,
                    "median_net_pnl": float(np.nanmedian(pnl)),
                    "pnl_std": std,
                    "sharpe_like": None if std == 0 else avg / std,
                    "max_drawdown": max_drawdown(pnl),
                    "avg_entry_price": float(trades["entry_price"].mean()),
                    "avg_exit_price": float(trades["exit_price"].mean()),
                    "avg_tte": float(trades["time_to_expiry_seconds"].mean()),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.thresholds)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    lr_cfg = cfg.get("models", {}).get("logistic_regression", {})
    lgb_cfg = dict(cfg.get("models", {}).get("lightgbm", {}))
    early = int(lgb_cfg.pop("early_stopping_rounds", 50))
    lgb_cfg.setdefault("n_estimators", 800)
    lgb_cfg.setdefault("learning_rate", 0.03)

    features = json.loads((Path(args.data) / "features_pm_repricing_executable.json").read_text(encoding="utf-8"))
    check_features(features)
    df = read_dataset(args.data).with_columns(pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"))
    missing = [c for c in features + ["split", "market_id", "sample_ts", "latency_seconds", "exit_horizon_seconds"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_out: dict[str, Any] = {
        "cost_model": {
            "fee_rate": args.fee_rate,
            "fee_formula_per_share": "fee_rate * price * (1 - price)",
            "entry_and_exit_taker_fees": True,
            "slippage_buffer": args.slippage_buffer,
            "edge_buffer": args.edge_buffer,
            "min_tte": args.min_tte,
            "max_spread": args.max_spread,
        },
        "features": features,
        "models": {},
    }
    pred_frames: list[pl.DataFrame] = []
    threshold_out: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for latency, horizon in parse_combos(args.combos):
        combo = df.filter((pl.col("latency_seconds") == latency) & (pl.col("exit_horizon_seconds") == horizon) & (pl.col("time_to_expiry_seconds") >= args.min_tte))
        if combo.height == 0:
            continue
        for direction, gross_pnl_col, entry_col, exit_col, spread_col, label_name in [
            ("UP", "pnl_up", "entry_yes_ask", "exit_yes_bid", "yes_spread", "label_up_net_profitable"),
            ("DOWN", "pnl_down", "entry_no_ask", "exit_no_bid", "no_spread", "label_down_net_profitable"),
        ]:
            side = combo.filter(pl.col(spread_col) <= args.max_spread).with_columns(
                [
                    taker_fee_expr(entry_col, args.fee_rate).alias("entry_fee"),
                    taker_fee_expr(exit_col, args.fee_rate).alias("exit_fee"),
                ]
            ).with_columns(
                [
                    (pl.col(gross_pnl_col) - pl.col("entry_fee") - pl.col("exit_fee") - pl.lit(args.slippage_buffer)).alias("net_pnl"),
                ]
            ).with_columns((pl.col("net_pnl") > args.edge_buffer).cast(pl.Int8).alias(label_name))
            if side.height == 0:
                continue

            train = side.filter(pl.col("split") == "train")
            valid = side.filter(pl.col("split") == "valid")
            test = side.filter(pl.col("split") == "test")
            if min(train.height, valid.height, test.height) == 0:
                continue

            x_train = train.select(features).to_numpy().astype(float)
            x_valid = valid.select(features).to_numpy().astype(float)
            x_test = test.select(features).to_numpy().astype(float)
            y_train = train[label_name].to_numpy().astype(int)
            y_valid = valid[label_name].to_numpy().astype(int)
            y_test = test[label_name].to_numpy().astype(int)
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

            joblib.dump(lr, model_dir / f"logistic_net_{key}.joblib")
            joblib.dump(lgbm, model_dir / f"lightgbm_net_{key}.joblib")
            metric_out["models"][key] = {
                "latency_seconds": latency,
                "exit_horizon_seconds": horizon,
                "direction": direction,
                "rows": {"train": train.height, "valid": valid.height, "test": test.height},
                "positive_rate": {"train": float(y_train.mean()), "valid": float(y_valid.mean()), "test": float(y_test.mean())},
                "logistic_regression": model_metrics(y_test, p_lr, thresholds),
                "lightgbm": model_metrics(y_test, p_lgb, thresholds),
            }
            pred = test.select(
                [
                    "market_id",
                    "sample_ts",
                    "date",
                    "split",
                    "time_to_expiry_seconds",
                    "latency_seconds",
                    "exit_horizon_seconds",
                    spread_col,
                    entry_col,
                    exit_col,
                    gross_pnl_col,
                    "entry_fee",
                    "exit_fee",
                    "net_pnl",
                    label_name,
                ]
            ).rename({spread_col: "side_spread", entry_col: "entry_price", exit_col: "exit_price", gross_pnl_col: "gross_pnl", label_name: "label_net_profitable"})
            pred = pred.with_columns([pl.lit(direction).alias("direction"), pl.Series("p_logistic", p_lr), pl.Series("p_model", p_lgb)])
            pred_frames.append(pred)
            threshold_out.extend(threshold_rows(pred, thresholds, {"latency_seconds": latency, "exit_horizon_seconds": horizon, "direction": direction, "model": "lightgbm"}))
            for r in pred.group_by("date").agg(pl.len().alias("rows"), pl.col("net_pnl").mean().alias("avg_net_pnl"), pl.col("label_net_profitable").mean().alias("net_win_rate")).to_dicts():
                group_rows.append({"latency_seconds": latency, "exit_horizon_seconds": horizon, "direction": direction, "group_type": "date", **r})

    preds = pl.concat(pred_frames, how="diagonal") if pred_frames else pl.DataFrame()
    preds.write_parquet(out_dir / "pm_repricing_executable_net_predictions.parquet")
    (out_dir / "pm_repricing_executable_net_metrics.json").write_text(json.dumps(metric_out, indent=2), encoding="utf-8")
    write_csv(out_dir / "pm_repricing_executable_net_thresholds.csv", threshold_out)
    write_csv(out_dir / "pm_repricing_executable_net_by_group.csv", group_rows)

    lines = [
        "# PM Repricing Executable Net v1\n\n",
        "Target: taker ask-entry / bid-exit net PnL after fees and slippage buffer.\n\n",
        f"- fee_rate: `{args.fee_rate}`\n",
        f"- slippage_buffer: `{args.slippage_buffer}`\n",
        f"- edge_buffer: `{args.edge_buffer}`\n",
        f"- min_tte: `{args.min_tte}`\n",
        f"- max_spread: `{args.max_spread}`\n\n",
        "## Model metrics\n\n",
        "| combo | AUC | logloss | Brier | test_positive_rate |\n",
        "| --- | ---: | ---: | ---: | ---: |\n",
    ]
    for key, m in metric_out["models"].items():
        vals = m["lightgbm"]
        lines.append(f"| {key} | {vals['auc']} | {vals['logloss']} | {vals['brier']} | {m['positive_rate']['test']} |\n")
    lines.append("\n## Positive net PnL thresholds\n\n")
    good = [r for r in threshold_out if r.get("trades", 0) and (r.get("avg_net_pnl") or 0) > 0]
    if not good:
        lines.append("No tested threshold produced positive average net PnL on test.\n")
    else:
        lines.append("| combo | threshold | trades | avg_net_pnl | total_net_pnl | win_rate | max_drawdown |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in sorted(good, key=lambda x: (x["avg_net_pnl"] or -999), reverse=True)[:50]:
            combo = f"{r['direction']} lat={r['latency_seconds']} h={r['exit_horizon_seconds']}"
            lines.append(f"| {combo} | {r['threshold']} | {r['trades']} | {r['avg_net_pnl']} | {r['total_net_pnl']} | {r['win_rate']} | {r['max_drawdown']} |\n")
    (out_dir / "pm_repricing_executable_net_report.md").write_text("".join(lines), encoding="utf-8")
    (model_dir / "features_pm_repricing_executable_net.json").write_text(json.dumps(features, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
