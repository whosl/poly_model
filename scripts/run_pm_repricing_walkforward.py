"""Rolling walk-forward training/evaluation for PM repricing 5s model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pm_repricing_executable import COOLDOWNS, add_tte_bucket, ensure_future_quotes, make_signals, parse_float_list, read_parquet_dataset, select_mode, write_csv  # noqa: E402
from evaluate_pm_repricing_latency import add_latency_quotes, load_silver_quotes, make_latency_signals, metric as latency_metric  # noqa: E402


CLASSES = ["DOWN", "FLAT", "UP"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
FOLDS = [
    ("fold1", "2026-04-22", "2026-04-25", "2026-04-26", "2026-04-27"),
    ("fold2", "2026-04-22", "2026-04-26", "2026-04-27", "2026-04-28"),
    ("fold3", "2026-04-22", "2026-04-27", "2026-04-28", "2026-04-29"),
    ("fold4", "2026-04-22", "2026-04-28", "2026-04-29", "2026-04-30"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/model_stage1.yaml")
    p.add_argument("--data", default="data/gold/pm_repricing_1s")
    p.add_argument("--silver", default="data/silver/pm_1s")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--thresholds", default="0.70,0.75")
    return p.parse_args()


def load_cfg(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def to_x(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select(features).to_numpy().astype(float)


def to_y(df: pl.DataFrame, label: str) -> np.ndarray:
    return np.asarray([CLASS_TO_INT[x] for x in df[label].to_list()], dtype=int)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def exec_metric(trades: pl.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    if trades.height == 0:
        row.update({"trades": 0, "total_pnl": 0.0, "avg_pnl": None, "win_rate": None})
        return row
    row.update(
        {
            "trades": int(trades.height),
            "total_pnl": float(trades["pnl"].sum()),
            "avg_pnl": float(trades["pnl"].mean()),
            "win_rate": float((trades["pnl"] > 0).mean()),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    data_root = Path(args.data)
    features = json.loads((data_root / "features_pm_repricing.json").read_text(encoding="utf-8"))
    label = "label_reprice_5s"
    cols = sorted(set(features + [label, "market_id", "sample_ts", "time_to_expiry_seconds", "yes_bid", "yes_ask", "no_bid", "no_ask", "markout_5s"]))
    df = read_parquet_dataset(data_root, cols).with_columns(pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"))
    thresholds = parse_float_list(args.thresholds)
    horizons = [5, 10, 30]
    lgb_cfg = dict(cfg.get("models", {}).get("lightgbm", {}))
    early = int(lgb_cfg.pop("early_stopping_rounds", 50))
    lgb_cfg.setdefault("num_leaves", 63)
    lgb_cfg.setdefault("learning_rate", 0.03)
    lgb_cfg.setdefault("n_estimators", 1000)

    silver = load_silver_quotes(args.silver)
    metric_rows: list[dict[str, Any]] = []
    exec_rows: list[dict[str, Any]] = []
    lat_rows: list[dict[str, Any]] = []

    for fold, train_start, train_end, valid_date, test_date in FOLDS:
        train = df.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
        valid = df.filter(pl.col("date") == valid_date)
        test = df.filter(pl.col("date") == test_date)
        if min(train.height, valid.height, test.height) < 1000:
            metric_rows.append({"fold": fold, "status": "skipped", "train_rows": train.height, "valid_rows": valid.height, "test_rows": test.height})
            continue
        x_train, y_train = to_x(train, features), to_y(train, label)
        x_valid, y_valid = to_x(valid, features), to_y(valid, label)
        x_test, y_test = to_x(test, features), to_y(test, label)
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=42, n_jobs=-1, verbosity=-1, class_weight="balanced", **lgb_cfg)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(early, verbose=False)])
        p = model.predict_proba(x_test)
        yp = p.argmax(axis=1)
        metric_rows.append(
            {
                "fold": fold,
                "status": "ok",
                "train_start": train_start,
                "train_end": train_end,
                "valid_date": valid_date,
                "test_date": test_date,
                "train_rows": train.height,
                "valid_rows": valid.height,
                "test_rows": test.height,
                "accuracy": float(accuracy_score(y_test, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, yp)),
                "macro_f1": float(f1_score(y_test, yp, average="macro", zero_division=0)),
            }
        )
        pred = test.select(["market_id", "sample_ts", "time_to_expiry_seconds", "yes_bid", "yes_ask", "no_bid", "no_ask", "markout_5s"]).with_columns(
            [
                pl.Series("p_down_5s", p[:, 0]),
                pl.Series("p_flat_5s", p[:, 1]),
                pl.Series("p_up_5s", p[:, 2]),
            ]
        )
        pred_path = out_dir / f"pm_repricing_walkforward_{fold}_predictions.parquet"
        pred.write_parquet(pred_path)
        pred_exec = add_tte_bucket(ensure_future_quotes(pred, args.data, horizons))
        for horizon in horizons:
            for threshold in thresholds:
                for direction in ["UP", "DOWN"]:
                    raw = make_signals(pred_exec, direction, threshold, horizon)
                    for mode in ["first_signal_per_market_side", "cooldown_10s", "cooldown_30s"]:
                        trades = select_mode(raw, mode)
                        exec_rows.append(exec_metric(trades, {"fold": fold, "test_date": test_date, "mode": mode, "direction": direction, "threshold": threshold, "exit_horizon": f"{horizon}s"}))
        base_lat = add_tte_bucket(pred.select(["market_id", "sample_ts", "time_to_expiry_seconds", "p_up_5s", "p_down_5s", "p_flat_5s"]))
        for latency in [0, 1]:
            for horizon in [5, 10]:
                q = add_latency_quotes(base_lat, silver, latency, horizon)
                for threshold in thresholds:
                    for direction in ["UP", "DOWN"]:
                        raw = make_latency_signals(q, direction, threshold, latency, horizon)
                        for mode in ["first_signal_per_market_side", "cooldown_10s", "cooldown_30s"]:
                            trades = select_mode(raw, mode)
                            lat_rows.append(latency_metric(trades, {"fold": fold, "test_date": test_date, "latency_seconds": latency, "mode": mode, "direction": direction, "threshold": threshold, "exit_horizon": f"{horizon}s"}))

    write_rows(out_dir / "pm_repricing_walkforward_metrics.csv", metric_rows)
    write_rows(out_dir / "pm_repricing_walkforward_executable.csv", exec_rows)
    write_rows(out_dir / "pm_repricing_walkforward_latency.csv", lat_rows)
    lines = ["# PM Repricing Walk-forward Report\n\n", "## Fold model metrics\n\n", "| fold | test_date | rows | accuracy | balanced_accuracy | macro_f1 |\n", "| --- | --- | ---: | ---: | ---: | ---: |\n"]
    for r in metric_rows:
        lines.append(f"| {r.get('fold')} | {r.get('test_date')} | {r.get('test_rows')} | {r.get('accuracy')} | {r.get('balanced_accuracy')} | {r.get('macro_f1')} |\n")
    lines.append("\n## Key executable rows\n\n")
    lines.append("| fold | date | mode | direction | threshold | horizon | trades | total_pnl | avg_pnl | win_rate |\n")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |\n")
    for r in exec_rows:
        if r.get("mode") == "first_signal_per_market_side" and r.get("trades", 0):
            lines.append(f"| {r['fold']} | {r['test_date']} | {r['mode']} | {r['direction']} | {r['threshold']} | {r['exit_horizon']} | {r['trades']} | {r['total_pnl']} | {r['avg_pnl']} | {r['win_rate']} |\n")
    (out_dir / "pm_repricing_walkforward_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
