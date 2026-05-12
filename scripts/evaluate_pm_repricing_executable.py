"""Executable PM repricing evaluation using ask entry and bid exit.

This is deliberately stricter than the markout sanity check:

* UP signal enters by buying YES at current yes_ask.
* DOWN signal enters by buying NO at current no_ask.
* Exits use future bid, never mid.
* Missing entry/exit quotes are filtered.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
HORIZONS = [1, 5, 10, 30]
COOLDOWNS = {
    "all_signals": 0,
    "first_signal_per_market_side": None,
    "cooldown_5s": 5,
    "cooldown_10s": 10,
    "cooldown_30s": 30,
}
TTE_BUCKETS = [
    ("[240,300]", 240, 300),
    ("[180,240)", 180, 240),
    ("[120,180)", 120, 180),
    ("[60,120)", 60, 120),
    ("[30,60)", 30, 60),
    ("[10,30)", 10, 30),
    ("[0,10)", 0, 10),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="reports/stage1/pm_repricing_test_predictions.parquet")
    p.add_argument("--gold", default="data/gold/pm_repricing_1s")
    p.add_argument("--out-dir", default="reports/stage1")
    p.add_argument("--thresholds", default=",".join(str(x) for x in THRESHOLDS))
    p.add_argument("--horizons", default=",".join(str(x) for x in HORIZONS))
    return p.parse_args()


def read_parquet_dataset(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    p = Path(path)
    if p.is_dir():
        lf = pl.scan_parquet(str(p / "**" / "*.parquet"), extra_columns="ignore")
    else:
        lf = pl.scan_parquet(str(p), extra_columns="ignore")
    if columns:
        available = set(lf.collect_schema().names())
        lf = lf.select([c for c in columns if c in available])
    return lf.collect()


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip().rstrip("s")) for x in s.split(",") if x.strip()]


def add_tte_bucket(df: pl.DataFrame) -> pl.DataFrame:
    bucket_expr = pl.lit(None, dtype=pl.Utf8)
    for label, lo, hi in reversed(TTE_BUCKETS):
        if hi == 300:
            cond = (pl.col("time_to_expiry_seconds") >= lo) & (pl.col("time_to_expiry_seconds") <= hi)
        else:
            cond = (pl.col("time_to_expiry_seconds") >= lo) & (pl.col("time_to_expiry_seconds") < hi)
        bucket_expr = pl.when(cond).then(pl.lit(label)).otherwise(bucket_expr)
    return df.with_columns(
        [
            pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"),
            bucket_expr.alias("tte_bucket"),
        ]
    )


def ensure_future_quotes(pred: pl.DataFrame, gold_path: str | Path, horizons: list[int]) -> pl.DataFrame:
    needed = []
    for h in horizons:
        for col in [
            f"future_yes_bid_{h}s",
            f"future_yes_ask_{h}s",
            f"future_no_bid_{h}s",
            f"future_no_ask_{h}s",
        ]:
            if col not in pred.columns:
                needed.append(col)
    if not needed:
        return pred

    quote_cols = ["market_id", "sample_ts", "yes_bid", "yes_ask", "no_bid", "no_ask"]
    gold = read_parquet_dataset(gold_path, quote_cols).sort(["market_id", "sample_ts"])
    out = pred.sort(["market_id", "sample_ts"])
    for h in horizons:
        if all(c in out.columns for c in [f"future_yes_bid_{h}s", f"future_no_bid_{h}s"]):
            continue
        right = gold.rename(
            {
                "sample_ts": f"future_sample_ts_{h}s",
                "yes_bid": f"future_yes_bid_{h}s",
                "yes_ask": f"future_yes_ask_{h}s",
                "no_bid": f"future_no_bid_{h}s",
                "no_ask": f"future_no_ask_{h}s",
            }
        )
        left = out.with_columns((pl.col("sample_ts") + pl.duration(seconds=h)).alias(f"exit_ts_{h}s"))
        out = left.join_asof(
            right,
            left_on=f"exit_ts_{h}s",
            right_on=f"future_sample_ts_{h}s",
            by="market_id",
            strategy="forward",
            tolerance="1s",
        )
    return out


def max_drawdown(pnls: np.ndarray) -> float | None:
    if len(pnls) == 0:
        return None
    equity = np.cumsum(pnls.astype(float))
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def metric_row(trades: pl.DataFrame, mode: str, direction: str, threshold: float, horizon: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mode": mode,
        "direction": direction,
        "threshold": threshold,
        "exit_horizon": f"{horizon}s",
    }
    if extra:
        row.update(extra)
    if trades.height == 0:
        row.update(
            {
                "trades": 0,
                "avg_entry_price": None,
                "avg_exit_price": None,
                "win_rate": None,
                "total_pnl": 0.0,
                "avg_pnl": None,
                "median_pnl": None,
                "pnl_std": None,
                "sharpe_like": None,
                "max_drawdown": None,
                "avg_time_to_expiry": None,
            }
        )
        return row
    pnl = trades["pnl"].to_numpy().astype(float)
    std = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    avg = float(np.nanmean(pnl))
    row.update(
        {
            "trades": int(trades.height),
            "avg_entry_price": float(trades["entry_price"].mean()),
            "avg_exit_price": float(trades["exit_price"].mean()),
            "win_rate": float((trades["pnl"] > 0).mean()),
            "total_pnl": float(np.nansum(pnl)),
            "avg_pnl": avg,
            "median_pnl": float(np.nanmedian(pnl)),
            "pnl_std": std,
            "sharpe_like": None if std == 0 or math.isnan(std) else avg / std,
            "max_drawdown": max_drawdown(pnl),
            "avg_time_to_expiry": float(trades["time_to_expiry_seconds"].mean()),
        }
    )
    return row


def select_mode(signals: pl.DataFrame, mode: str) -> pl.DataFrame:
    if signals.height == 0 or mode == "all_signals":
        return signals
    signals = signals.sort(["market_id", "direction", "sample_ts"])
    if mode == "first_signal_per_market_side":
        return signals.group_by(["market_id", "direction"], maintain_order=True).head(1)
    cooldown = COOLDOWNS[mode]
    if cooldown is None:
        return signals
    kept: list[dict[str, Any]] = []
    last_key: tuple[str, str] | None = None
    last_ts: Any = None
    for r in signals.iter_rows(named=True):
        key = (str(r["market_id"]), str(r["direction"]))
        ts = r["sample_ts"]
        if key != last_key:
            kept.append(r)
            last_key, last_ts = key, ts
            continue
        # Polars returns Python datetime for datetime columns.
        if (ts - last_ts).total_seconds() >= float(cooldown):
            kept.append(r)
            last_ts = ts
    return pl.DataFrame(kept, schema=signals.schema) if kept else signals.head(0)


def make_signals(df: pl.DataFrame, direction: str, threshold: float, horizon: int) -> pl.DataFrame:
    if direction == "UP":
        prob_col = "p_up_5s"
        entry_col = "yes_ask"
        exit_col = f"future_yes_bid_{horizon}s"
    else:
        prob_col = "p_down_5s"
        entry_col = "no_ask"
        exit_col = f"future_no_bid_{horizon}s"
    if prob_col not in df.columns or entry_col not in df.columns or exit_col not in df.columns:
        return df.head(0)
    cols = [
        "market_id",
        "sample_ts",
        "date",
        "tte_bucket",
        "time_to_expiry_seconds",
        prob_col,
        entry_col,
        exit_col,
    ]
    cols = [c for c in cols if c in df.columns]
    out = (
        df.filter((pl.col(prob_col) >= threshold) & pl.col(entry_col).is_not_null() & pl.col(exit_col).is_not_null())
        .select(cols)
        .rename({prob_col: "signal_probability", entry_col: "entry_price", exit_col: "exit_price"})
        .with_columns(
            [
                pl.lit(direction).alias("direction"),
                (pl.col("exit_price") - pl.col("entry_price")).alias("pnl"),
            ]
        )
        .filter(pl.col("entry_price").is_finite() & pl.col("exit_price").is_finite())
    )
    return out


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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_int_list(args.horizons)

    pred_cols = [
        "market_id",
        "sample_ts",
        "time_to_expiry_seconds",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "no_bid",
        "no_ask",
        "no_mid",
        "markout_1s",
        "markout_5s",
        "markout_30s",
        "p_up_5s",
        "p_down_5s",
        "p_flat_5s",
    ]
    for h in horizons:
        pred_cols.extend(
            [
                f"future_yes_bid_{h}s",
                f"future_yes_ask_{h}s",
                f"future_no_bid_{h}s",
                f"future_no_ask_{h}s",
            ]
        )
    df = read_parquet_dataset(args.predictions, pred_cols)
    df = ensure_future_quotes(df, args.gold, horizons)
    df = add_tte_bucket(df)

    threshold_rows: list[dict[str, Any]] = []
    by_date_rows: list[dict[str, Any]] = []
    by_tte_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        for threshold in thresholds:
            for direction in ["UP", "DOWN"]:
                raw = make_signals(df, direction, threshold, horizon)
                for mode in COOLDOWNS:
                    trades = select_mode(raw, mode)
                    threshold_rows.append(metric_row(trades, mode, direction, threshold, horizon))
                    if trades.height:
                        for date in trades.select("date").drop_nulls().unique().to_series().to_list():
                            sub = trades.filter(pl.col("date") == date)
                            by_date_rows.append(metric_row(sub, mode, direction, threshold, horizon, {"date": date}))
                        for bucket in trades.select("tte_bucket").drop_nulls().unique().to_series().to_list():
                            sub = trades.filter(pl.col("tte_bucket") == bucket)
                            by_tte_rows.append(metric_row(sub, mode, direction, threshold, horizon, {"tte_bucket": bucket}))

    write_csv(out_dir / "pm_repricing_executable_thresholds.csv", threshold_rows)
    write_csv(out_dir / "pm_repricing_executable_by_date.csv", by_date_rows)
    write_csv(out_dir / "pm_repricing_executable_by_tte.csv", by_tte_rows)

    first_rows = [r for r in threshold_rows if r["mode"] == "first_signal_per_market_side" and r["trades"]]
    top = sorted(first_rows, key=lambda r: (r.get("avg_pnl") is not None, r.get("avg_pnl") or -999), reverse=True)[:12]
    lines = [
        "# PM Repricing Executable Evaluation\n\n",
        "This evaluation uses executable bid/ask assumptions: ask entry and future bid exit. It does not use mid markout as PnL.\n\n",
        f"- prediction rows: `{df.height}`\n",
        f"- thresholds: `{thresholds}`\n",
        f"- exit horizons: `{horizons}` seconds\n\n",
        "## Best first_signal_per_market_side rows by avg_pnl\n\n",
        "| direction | threshold | exit_horizon | trades | win_rate | avg_entry_price | avg_exit_price | total_pnl | avg_pnl | sharpe_like |\n",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for r in top:
        lines.append(
            f"| {r['direction']} | {r['threshold']} | {r['exit_horizon']} | {r['trades']} | "
            f"{r.get('win_rate')} | {r.get('avg_entry_price')} | {r.get('avg_exit_price')} | "
            f"{r.get('total_pnl')} | {r.get('avg_pnl')} | {r.get('sharpe_like')} |\n"
        )
    lines.extend(
        [
            "\n## Output files\n\n",
            "- `reports/stage1/pm_repricing_executable_thresholds.csv`\n",
            "- `reports/stage1/pm_repricing_executable_by_date.csv`\n",
            "- `reports/stage1/pm_repricing_executable_by_tte.csv`\n",
        ]
    )
    (out_dir / "pm_repricing_executable_report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
