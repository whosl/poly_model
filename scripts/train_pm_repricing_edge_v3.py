"""Train v3 PM repricing edge models.

This version is designed around executable *expected net edge* rather than a
high-probability rare-event classifier:

* entry = taker buy at ask after latency
* exit = taker sell at bid after horizon
* label/target = net PnL after Polymarket fee + slippage buffer
* selection = choose edge/probability thresholds on validation, report test PnL

All engineered features are derived from columns that the live shadow runtime
already computes (or can compute cheaply), so the model can be deployed without
needing offline-only leakage columns.
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
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


BASE_FEATURES_PATH = "models/pm_repricing_executable_net_v2_spread02_tte60/features_pm_repricing_executable_net.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/gold/pm_repricing_executable_1s_v2")
    p.add_argument("--base-features", default=BASE_FEATURES_PATH)
    p.add_argument("--out-dir", default="reports/stage1/edge_v3")
    p.add_argument("--model-dir", default="models/pm_repricing_edge_v3")
    p.add_argument("--combos", default="0:5,0:10,1:5,1:10")
    p.add_argument("--fee-rate", type=float, default=0.07)
    p.add_argument("--slippage-buffer", type=float, default=0.0025)
    p.add_argument("--edge-buffer", type=float, default=0.0)
    p.add_argument("--min-tte", type=float, default=60.0)
    p.add_argument("--max-spread", type=float, default=0.03)
    p.add_argument("--min-valid-trades", type=int, default=20)
    p.add_argument("--threshold-grid", default="-0.005,-0.0025,0,0.001,0.0025,0.005,0.0075,0.01,0.015,0.02")
    p.add_argument("--prob-grid", default="0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--max-train-rows", type=int, default=600000)
    return p.parse_args()


def parse_combos(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def scan_dataset(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    return pl.scan_parquet(str(p / "**" / "*.parquet"), extra_columns="ignore").collect()


def scan_lazy(path: str | Path) -> pl.LazyFrame:
    p = Path(path)
    return pl.scan_parquet(str(p / "**" / "*.parquet"), extra_columns="ignore")


def fee(price: pl.Expr, fee_rate: float) -> pl.Expr:
    p = price.clip(0.0, 1.0)
    return fee_rate * p * (1.0 - p)


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(None)


def add_v3_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add leakage-safe features derived from live-available columns."""
    eps = 1e-9
    df = df.with_columns(
        [
            pl.col("time_to_expiry_seconds").log1p().alias("log_tte"),
            safe_div(pl.col("time_elapsed_seconds"), pl.col("time_elapsed_seconds") + pl.col("time_to_expiry_seconds")).alias("time_frac_elapsed"),
            (pl.col("time_to_expiry_seconds") <= 300).cast(pl.Int8).alias("tte_le_300"),
            (pl.col("time_to_expiry_seconds") <= 180).cast(pl.Int8).alias("tte_le_180"),
            (pl.col("time_to_expiry_seconds") <= 120).cast(pl.Int8).alias("tte_le_120"),
            (pl.col("time_to_expiry_seconds") <= 60).cast(pl.Int8).alias("tte_le_60"),
            (pl.col("yes_ask") - pl.col("formula_p_yes")).alias("yes_ask_minus_formula_p_yes"),
            (pl.col("no_ask") - (1.0 - pl.col("formula_p_yes"))).alias("no_ask_minus_formula_p_no"),
            (pl.col("formula_p_yes") - pl.col("yes_mid")).abs().alias("abs_formula_gap_yes_mid"),
            (pl.col("pair_ask_sum") - 1.0).alias("pair_ask_overround"),
            (1.0 - pl.col("pair_bid_sum")).alias("pair_bid_underround"),
            (pl.col("pair_mid_sum_live") - 1.0).alias("pair_mid_sum_excess"),
            (pl.col("yes_ask_depth_5") - pl.col("yes_bid_depth_5")).alias("yes_depth_skew_abs"),
            (pl.col("no_ask_depth_5") - pl.col("no_bid_depth_5")).alias("no_depth_skew_abs"),
            safe_div(pl.col("yes_ask_depth_5"), pl.col("yes_bid_depth_5") + eps).log1p().alias("log_yes_ask_bid_depth_ratio"),
            safe_div(pl.col("no_ask_depth_5"), pl.col("no_bid_depth_5") + eps).log1p().alias("log_no_ask_bid_depth_ratio"),
            (pl.col("yes_spread") + pl.col("no_spread")).alias("sum_side_spreads"),
            (pl.col("yes_spread") - pl.col("no_spread")).alias("spread_diff_yes_no"),
            (pl.col("yes_mid") - pl.col("no_mid")).alias("yes_no_mid_diff"),
            (pl.col("pm_yes_mid_change_1s_past") - pl.col("pm_no_mid_change_1s_past")).alias("pm_mid_change_1s_rel"),
            (pl.col("pm_yes_mid_change_5s_past") - pl.col("pm_no_mid_change_5s_past")).alias("pm_mid_change_5s_rel"),
            (pl.col("btc_return_1s") * pl.col("time_to_expiry_seconds").log1p()).alias("btc_ret1_x_logtte"),
            (pl.col("btc_return_5s") * pl.col("time_to_expiry_seconds").log1p()).alias("btc_ret5_x_logtte"),
            (pl.col("btc_return_10s") * pl.col("time_to_expiry_seconds").log1p()).alias("btc_ret10_x_logtte"),
            (pl.col("btc_return_30s") * pl.col("time_to_expiry_seconds").log1p()).alias("btc_ret30_x_logtte"),
            safe_div(pl.col("btc_return_5s"), pl.col("btc_realized_vol_60s") + eps).alias("btc_ret5_vol_adj"),
            safe_div(pl.col("btc_return_10s"), pl.col("btc_realized_vol_60s") + eps).alias("btc_ret10_vol_adj"),
            safe_div(pl.col("btc_return_30s"), pl.col("btc_realized_vol_60s") + eps).alias("btc_ret30_vol_adj"),
            (pl.col("btc_return_1s") - pl.col("btc_return_5s")).alias("btc_ret_1m5_reversal"),
            (pl.col("btc_return_5s") - pl.col("btc_return_30s")).alias("btc_ret_5m30_reversal"),
            (pl.col("btc_trade_imbalance_1s") - pl.col("btc_trade_imbalance_5s")).alias("btc_trade_imb_1m5"),
            (pl.col("btc_trade_imbalance_5s") - pl.col("btc_trade_imbalance_30s")).alias("btc_trade_imb_5m30"),
            (pl.col("btc_depth_imbalance_5") * pl.col("btc_trade_imbalance_5s")).alias("btc_depth_x_trade_imb5"),
            (pl.col("yes_quote_age_seconds").fill_null(999) + pl.col("no_quote_age_seconds").fill_null(999)).alias("sum_quote_age"),
            (pl.max_horizontal("yes_quote_age_seconds", "no_quote_age_seconds")).alias("max_quote_age"),
        ]
    )
    return df.with_columns(
        [
            (pl.col("formula_p_yes_minus_yes_ask") * pl.col("btc_ret5_vol_adj")).alias("formula_yes_edge_x_btc_ret5_vol"),
            (pl.col("formula_p_no_minus_no_ask") * (-pl.col("btc_ret5_vol_adj"))).alias("formula_no_edge_x_btc_ret5_vol"),
        ]
    )


DERIVED_FEATURES = [
    "log_tte", "time_frac_elapsed", "tte_le_300", "tte_le_180", "tte_le_120", "tte_le_60",
    "yes_ask_minus_formula_p_yes", "no_ask_minus_formula_p_no", "abs_formula_gap_yes_mid",
    "pair_ask_overround", "pair_bid_underround", "pair_mid_sum_excess",
    "yes_depth_skew_abs", "no_depth_skew_abs", "log_yes_ask_bid_depth_ratio", "log_no_ask_bid_depth_ratio",
    "sum_side_spreads", "spread_diff_yes_no", "yes_no_mid_diff", "pm_mid_change_1s_rel", "pm_mid_change_5s_rel",
    "btc_ret1_x_logtte", "btc_ret5_x_logtte", "btc_ret10_x_logtte", "btc_ret30_x_logtte",
    "btc_ret5_vol_adj", "btc_ret10_vol_adj", "btc_ret30_vol_adj", "btc_ret_1m5_reversal", "btc_ret_5m30_reversal",
    "btc_trade_imb_1m5", "btc_trade_imb_5m30", "btc_depth_x_trade_imb5",
    "formula_yes_edge_x_btc_ret5_vol", "formula_no_edge_x_btc_ret5_vol", "sum_quote_age", "max_quote_age",
]


def max_drawdown(pnl: np.ndarray) -> float | None:
    if len(pnl) == 0:
        return None
    equity = np.cumsum(pnl)
    return float(np.max(np.maximum.accumulate(equity) - equity))


def summarize_trades(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {"trades": 0, "total_net_pnl": 0.0, "avg_net_pnl": None, "win_rate": None, "max_drawdown": None}
    pnl = df["net_pnl"].to_numpy().astype(float)
    return {
        "trades": int(df.height),
        "total_net_pnl": float(np.nansum(pnl)),
        "avg_net_pnl": float(np.nanmean(pnl)),
        "win_rate": float((pnl > 0).mean()),
        "max_drawdown": max_drawdown(pnl),
        "avg_tte": float(df["time_to_expiry_seconds"].mean()),
        "avg_entry": float(df["entry_price"].mean()),
    }


def choose_threshold(valid_pred: pl.DataFrame, edge_grid: list[float], prob_grid: list[float], min_trades: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for edge_thr in edge_grid:
        for prob_thr in prob_grid:
            trades = valid_pred.filter((pl.col("pred_edge") >= edge_thr) & (pl.col("pred_prob") >= prob_thr)).sort(["market_id", "sample_ts"])
            s = summarize_trades(trades)
            if s["trades"] < min_trades:
                continue
            score = (s["avg_net_pnl"] or -999) - 0.05 * (s["max_drawdown"] or 0)
            row = {"edge_threshold": edge_thr, "prob_threshold": prob_thr, "score": score, **s}
            if best is None or row["score"] > best["score"]:
                best = row
    if best is None:
        # Fall back to the most conservative edge-positive threshold.
        return {"edge_threshold": 0.0, "prob_threshold": 0.5, "score": None, **summarize_trades(valid_pred.filter((pl.col("pred_edge") >= 0.0) & (pl.col("pred_prob") >= 0.5)))}
    return best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)

    base_features = json.loads(Path(args.base_features).read_text(encoding="utf-8"))
    features = base_features + [c for c in DERIVED_FEATURES if c not in base_features]
    # Keep only numeric/live-safe features.
    banned_exact = {"market_id", "sample_ts", "date", "split", "market_start_ts", "market_end_ts"}
    banned_prefix = ("future_", "pnl_", "roi_", "label_", "entry_", "exit_")
    features = [c for c in features if c not in banned_exact and not c.startswith(banned_prefix)]

    edge_grid = floats(args.threshold_grid)
    prob_grid = floats(args.prob_grid)
    rows: list[dict[str, Any]] = []
    pred_frames = []
    lf_all = scan_lazy(args.data)
    base_schema = set(lf_all.collect_schema().names())
    source_cols = sorted((set(base_features) | {
        "market_id", "sample_ts", "date", "split", "time_to_expiry_seconds", "time_elapsed_seconds",
        "latency_seconds", "exit_horizon_seconds", "pnl_up", "pnl_down",
        "entry_yes_ask", "entry_no_ask", "exit_yes_bid", "exit_no_bid", "yes_spread", "no_spread",
    }) & base_schema)

    for latency, horizon in parse_combos(args.combos):
        for direction, gross_col, entry_col, exit_col, spread_col in [
            ("UP", "pnl_up", "entry_yes_ask", "exit_yes_bid", "yes_spread"),
            ("DOWN", "pnl_down", "entry_no_ask", "exit_no_bid", "no_spread"),
        ]:
            needed = sorted(set(source_cols) | {gross_col, entry_col, exit_col, spread_col})
            combo = (
                lf_all.select(needed)
                .filter(
                    (pl.col("latency_seconds") == latency)
                    & (pl.col("exit_horizon_seconds") == horizon)
                    & (pl.col("time_to_expiry_seconds") >= args.min_tte)
                    & (pl.col(spread_col) <= args.max_spread)
                )
                .collect()
            )
            combo = add_v3_features(combo).with_columns(pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"))
            features_here = [c for c in features if c in combo.columns]
            side = combo.with_columns(
                [
                    fee(pl.col(entry_col), args.fee_rate).alias("entry_fee"),
                    fee(pl.col(exit_col), args.fee_rate).alias("exit_fee"),
                ]
            ).with_columns((pl.col(gross_col) - pl.col("entry_fee") - pl.col("exit_fee") - args.slippage_buffer).alias("net_pnl"))
            side = side.with_columns((pl.col("net_pnl") > args.edge_buffer).cast(pl.Int8).alias("label_net"))
            train, valid, test = [side.filter(pl.col("split") == s) for s in ["train", "valid", "test"]]
            if min(train.height, valid.height, test.height) < 100 or len(set(train["label_net"].to_list())) < 2:
                continue
            if train.height > args.max_train_rows:
                train_fit = train.sample(n=args.max_train_rows, seed=42, shuffle=True)
            else:
                train_fit = train
            x_train = train_fit.select(features_here).to_numpy().astype(float)
            x_valid = valid.select(features_here).to_numpy().astype(float)
            x_test = test.select(features_here).to_numpy().astype(float)
            y_train = train_fit["label_net"].to_numpy().astype(int)
            y_valid = valid["label_net"].to_numpy().astype(int)
            y_test = test["label_net"].to_numpy().astype(int)
            edge_train = train_fit["net_pnl"].to_numpy().astype(float)
            edge_valid = valid["net_pnl"].to_numpy().astype(float)

            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.025, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)),
            ])
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(objective="regression", n_estimators=900, learning_rate=0.025, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, random_state=43, n_jobs=-1, verbosity=-1)),
            ])
            clf.fit(x_train, y_train)
            reg.fit(x_train, edge_train)
            p_valid_raw = clf.predict_proba(x_valid)[:, 1]
            p_test_raw = clf.predict_proba(x_test)[:, 1]
            # Calibrate on validation to avoid "high threshold never fires" behavior.
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p_valid_raw, y_valid)
            p_valid = cal.predict(p_valid_raw)
            p_test = cal.predict(p_test_raw)
            e_valid = reg.predict(x_valid)
            e_test = reg.predict(x_test)

            valid_pred = valid.select(["market_id", "sample_ts", "date", "time_to_expiry_seconds", entry_col, exit_col, gross_col, "net_pnl", "label_net"]).rename({entry_col:"entry_price", exit_col:"exit_price", gross_col:"gross_pnl"}).with_columns([pl.Series("pred_prob", p_valid), pl.Series("pred_edge", e_valid)])
            test_pred = test.select(["market_id", "sample_ts", "date", "time_to_expiry_seconds", entry_col, exit_col, gross_col, "net_pnl", "label_net"]).rename({entry_col:"entry_price", exit_col:"exit_price", gross_col:"gross_pnl"}).with_columns([pl.Series("pred_prob", p_test), pl.Series("pred_edge", e_test), pl.lit(direction).alias("direction"), pl.lit(latency).alias("latency_seconds"), pl.lit(horizon).alias("exit_horizon_seconds")])
            chosen = choose_threshold(valid_pred, edge_grid, prob_grid, args.min_valid_trades)
            trades = test_pred.filter((pl.col("pred_edge") >= chosen["edge_threshold"]) & (pl.col("pred_prob") >= chosen["prob_threshold"])).sort(["market_id", "sample_ts"])
            test_sum = summarize_trades(trades)
            auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) == 2 else None
            key = f"lat{latency}_h{horizon}_{direction}"
            joblib.dump({"classifier": clf, "regressor": reg, "calibrator": cal, "features": features_here, "edge_threshold": chosen["edge_threshold"], "prob_threshold": chosen["prob_threshold"], "direction": direction, "latency_seconds": latency, "exit_horizon_seconds": horizon}, model_dir / f"edge_v3_{key}.joblib")
            rows.append({"key": key, "direction": direction, "latency_seconds": latency, "exit_horizon_seconds": horizon, "train_rows": train.height, "valid_rows": valid.height, "test_rows": test.height, "test_pos_rate": float(y_test.mean()), "test_auc": auc, **{f"valid_{k}": v for k,v in chosen.items()}, **{f"test_{k}": v for k,v in test_sum.items()}})
            pred_frames.append(test_pred.with_columns(((pl.col("pred_edge") >= chosen["edge_threshold"]) & (pl.col("pred_prob") >= chosen["prob_threshold"])).alias("selected")))

    write_csv(out_dir / "edge_v3_summary.csv", rows)
    if pred_frames:
        pl.concat(pred_frames, how="diagonal").write_parquet(out_dir / "edge_v3_test_predictions.parquet")
    (model_dir / "features_pm_repricing_edge_v3.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (out_dir / "edge_v3_report.md").write_text(
        "# PM Repricing Edge v3\n\n"
        + f"- features: `{len(features)}`\n"
        + f"- fee_rate: `{args.fee_rate}`\n- slippage_buffer: `{args.slippage_buffer}`\n- min_tte: `{args.min_tte}`\n- max_spread: `{args.max_spread}`\n\n"
        + "See `edge_v3_summary.csv` for validation-selected thresholds and test PnL.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
