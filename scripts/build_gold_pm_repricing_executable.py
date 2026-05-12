"""Build executable PM repricing gold labels.

Labels are ask-entry / bid-exit executable PnL labels, not mid markouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


LATENCIES = [0, 1, 2]
HORIZONS = [1, 5, 10, 30]
BANNED_EXACT = {
    "split",
    "market_id",
    "sample_ts",
    "date",
    "market_start_ts",
    "market_end_ts",
}
BANNED_PREFIXES = ("future_", "pnl_", "roi_", "label_", "exit_", "entry_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pm-silver", default="data/silver/pm_1s")
    p.add_argument("--binance-silver", default="data/silver/binance_1s")
    p.add_argument("--gold-repricing", default="data/gold/pm_repricing_1s")
    p.add_argument("--out", default="data/gold/pm_repricing_executable_1s")
    p.add_argument("--report", default="reports/stage1/pm_repricing_executable_gold_report.md")
    p.add_argument("--latencies", default="0,1,2")
    p.add_argument("--horizons", default="1,5,10,30")
    return p.parse_args()


def ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def scan(path: str | Path, cols: list[str] | None = None) -> pl.DataFrame:
    p = Path(path)
    lf = pl.scan_parquet(str(p / "**" / "*.parquet") if p.is_dir() else str(p), extra_columns="ignore")
    if cols:
        schema = set(lf.collect_schema().names())
        lf = lf.select([c for c in cols if c in schema])
    return lf.collect()


def main() -> None:
    args = parse_args()
    latencies = ints(args.latencies)
    horizons = ints(args.horizons)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    gold_cols = [
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
        "yes_depth_imbalance_5",
        "pair_mid_sum",
        "btc_return_1s",
        "btc_return_5s",
        "btc_return_30s",
        "realized_vol_10s",
        "realized_vol_30s",
        "formula_p_yes",
        "formula_p_yes_minus_yes_mid",
        "formula_p_yes_minus_yes_ask",
        "pm_yes_mid_change_1s_past",
        "pm_yes_mid_change_5s_past",
        "seconds_since_last_pm_update",
        "date",
    ]
    base = scan(args.gold_repricing, gold_cols)
    silver_cols = [
        "market_id",
        "sample_ts",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "yes_spread",
        "yes_bid_depth_5",
        "yes_ask_depth_5",
        "yes_depth_imbalance_5",
        "yes_quote_age_seconds",
        "yes_is_stale",
        "yes_crossed_quote",
        "no_bid",
        "no_ask",
        "no_mid",
        "no_spread",
        "no_bid_depth_5",
        "no_ask_depth_5",
        "no_depth_imbalance_5",
        "no_quote_age_seconds",
        "no_is_stale",
        "no_crossed_quote",
        "pair_bid_sum",
        "pair_ask_sum",
        "pair_mid_sum",
        "pm_no_mid_change_1s_past",
        "pm_no_mid_change_5s_past",
        "time_elapsed_seconds",
    ]
    silver = scan(args.pm_silver, silver_cols).sort(["market_id", "sample_ts"])
    # Add current fields missing from gold.
    current = silver.rename(
        {
            "yes_quote_age_seconds": "cur_yes_quote_age_seconds",
            "no_quote_age_seconds": "cur_no_quote_age_seconds",
            "yes_is_stale": "cur_yes_is_stale",
            "no_is_stale": "cur_no_is_stale",
            "yes_crossed_quote": "cur_yes_crossed_quote",
            "no_crossed_quote": "cur_no_crossed_quote",
            "no_spread": "cur_no_spread",
            "yes_bid_depth_5": "cur_yes_bid_depth_5",
            "yes_ask_depth_5": "cur_yes_ask_depth_5",
            "yes_depth_imbalance_5": "cur_yes_depth_imbalance_5",
            "no_bid_depth_5": "cur_no_bid_depth_5",
            "no_ask_depth_5": "cur_no_ask_depth_5",
            "no_depth_imbalance_5": "cur_no_depth_imbalance_5",
            "pair_bid_sum": "cur_pair_bid_sum",
            "pair_ask_sum": "cur_pair_ask_sum",
            "pair_mid_sum": "cur_pair_mid_sum",
            "pm_no_mid_change_1s_past": "cur_pm_no_mid_change_1s_past",
            "pm_no_mid_change_5s_past": "cur_pm_no_mid_change_5s_past",
            "time_elapsed_seconds": "cur_time_elapsed_seconds",
        }
    ).select(
        [
            "market_id",
            "sample_ts",
            "cur_yes_quote_age_seconds",
            "cur_no_quote_age_seconds",
            "cur_yes_is_stale",
            "cur_no_is_stale",
            "cur_yes_crossed_quote",
            "cur_no_crossed_quote",
            "cur_no_spread",
            "cur_yes_bid_depth_5",
            "cur_yes_ask_depth_5",
            "cur_yes_depth_imbalance_5",
            "cur_no_bid_depth_5",
            "cur_no_ask_depth_5",
            "cur_no_depth_imbalance_5",
            "cur_pair_bid_sum",
            "cur_pair_ask_sum",
            "cur_pair_mid_sum",
            "cur_pm_no_mid_change_1s_past",
            "cur_pm_no_mid_change_5s_past",
            "cur_time_elapsed_seconds",
        ]
    )
    base = base.join(current, on=["market_id", "sample_ts"], how="left").with_columns(
        [
            pl.col("cur_yes_quote_age_seconds").alias("yes_quote_age_seconds"),
            pl.col("cur_no_quote_age_seconds").alias("no_quote_age_seconds"),
            pl.col("cur_no_spread").alias("no_spread"),
            pl.col("cur_yes_bid_depth_5").alias("yes_bid_depth_5"),
            pl.col("cur_yes_ask_depth_5").alias("yes_ask_depth_5"),
            pl.col("cur_yes_depth_imbalance_5").alias("yes_depth_imbalance_5_live"),
            pl.col("cur_no_bid_depth_5").alias("no_bid_depth_5"),
            pl.col("cur_no_ask_depth_5").alias("no_ask_depth_5"),
            pl.col("cur_no_depth_imbalance_5").alias("no_depth_imbalance_5"),
            pl.col("cur_pair_bid_sum").alias("pair_bid_sum"),
            pl.col("cur_pair_ask_sum").alias("pair_ask_sum"),
            pl.col("cur_pair_mid_sum").alias("pair_mid_sum_live"),
            pl.col("cur_pm_no_mid_change_1s_past").alias("pm_no_mid_change_1s_past"),
            pl.col("cur_pm_no_mid_change_5s_past").alias("pm_no_mid_change_5s_past"),
            pl.col("cur_time_elapsed_seconds").alias("time_elapsed_seconds"),
            ((1.0 - pl.col("formula_p_yes")) - pl.col("no_ask")).alias("formula_p_no_minus_no_ask"),
        ]
    )
    binance_cols = [
        "sample_ts",
        "spread",
        "microprice_minus_mid",
        "return_10s",
        "realized_vol_60s",
        "buy_volume_1s",
        "sell_volume_1s",
        "net_volume_1s",
        "total_volume_1s",
        "trade_count_1s",
        "trade_imbalance_1s",
        "buy_volume_5s",
        "sell_volume_5s",
        "net_volume_5s",
        "total_volume_5s",
        "trade_count_5s",
        "trade_imbalance_5s",
        "buy_volume_30s",
        "sell_volume_30s",
        "net_volume_30s",
        "total_volume_30s",
        "trade_count_30s",
        "trade_imbalance_30s",
        "bid_depth_5",
        "ask_depth_5",
        "depth_imbalance_5",
    ]
    binance = scan(args.binance_silver, binance_cols).sort("sample_ts").rename(
        {
            "spread": "btc_spread",
            "microprice_minus_mid": "btc_microprice_minus_mid",
            "return_10s": "btc_return_10s",
            "realized_vol_60s": "btc_realized_vol_60s",
            "buy_volume_1s": "btc_buy_volume_1s",
            "sell_volume_1s": "btc_sell_volume_1s",
            "net_volume_1s": "btc_net_volume_1s",
            "total_volume_1s": "btc_total_volume_1s",
            "trade_count_1s": "btc_trade_count_1s",
            "trade_imbalance_1s": "btc_trade_imbalance_1s",
            "buy_volume_5s": "btc_buy_volume_5s",
            "sell_volume_5s": "btc_sell_volume_5s",
            "net_volume_5s": "btc_net_volume_5s",
            "total_volume_5s": "btc_total_volume_5s",
            "trade_count_5s": "btc_trade_count_5s",
            "trade_imbalance_5s": "btc_trade_imbalance_5s",
            "buy_volume_30s": "btc_buy_volume_30s",
            "sell_volume_30s": "btc_sell_volume_30s",
            "net_volume_30s": "btc_net_volume_30s",
            "total_volume_30s": "btc_total_volume_30s",
            "trade_count_30s": "btc_trade_count_30s",
            "trade_imbalance_30s": "btc_trade_imbalance_30s",
            "bid_depth_5": "btc_bid_depth_5",
            "ask_depth_5": "btc_ask_depth_5",
            "depth_imbalance_5": "btc_depth_imbalance_5",
        }
    )
    base = base.sort("sample_ts").join_asof(binance, on="sample_ts", strategy="backward")

    entry = silver.rename(
        {
            "sample_ts": "entry_quote_ts",
            "yes_ask": "entry_yes_ask",
            "yes_bid": "entry_yes_bid",
            "no_ask": "entry_no_ask",
            "no_bid": "entry_no_bid",
            "yes_is_stale": "entry_yes_is_stale",
            "no_is_stale": "entry_no_is_stale",
            "yes_crossed_quote": "entry_yes_crossed_quote",
            "no_crossed_quote": "entry_no_crossed_quote",
        }
    ).select(
        [
            "market_id",
            "entry_quote_ts",
            "entry_yes_ask",
            "entry_yes_bid",
            "entry_no_ask",
            "entry_no_bid",
            "entry_yes_is_stale",
            "entry_no_is_stale",
            "entry_yes_crossed_quote",
            "entry_no_crossed_quote",
        ]
    )
    exitq = silver.rename(
        {
            "sample_ts": "exit_quote_ts",
            "yes_bid": "exit_yes_bid",
            "yes_ask": "exit_yes_ask",
            "no_bid": "exit_no_bid",
            "no_ask": "exit_no_ask",
            "yes_is_stale": "exit_yes_is_stale",
            "no_is_stale": "exit_no_is_stale",
            "yes_crossed_quote": "exit_yes_crossed_quote",
            "no_crossed_quote": "exit_no_crossed_quote",
        }
    ).select(
        [
            "market_id",
            "exit_quote_ts",
            "exit_yes_bid",
            "exit_yes_ask",
            "exit_no_bid",
            "exit_no_ask",
            "exit_yes_is_stale",
            "exit_no_is_stale",
            "exit_yes_crossed_quote",
            "exit_no_crossed_quote",
        ]
    )

    frames: list[pl.DataFrame] = []
    stats = []
    for latency in latencies:
        for horizon in horizons:
            left = base.with_columns(
                [
                    pl.lit(latency).alias("latency_seconds"),
                    pl.lit(horizon).alias("exit_horizon_seconds"),
                    (pl.col("sample_ts") + pl.duration(seconds=latency)).alias("entry_time"),
                    (pl.col("sample_ts") + pl.duration(seconds=latency + horizon)).alias("exit_time"),
                ]
            ).filter(pl.col("time_to_expiry_seconds") >= latency + horizon)
            joined = (
                left.sort(["market_id", "entry_time"])
                .join_asof(entry, left_on="entry_time", right_on="entry_quote_ts", by="market_id", strategy="forward", tolerance="1s")
                .sort(["market_id", "exit_time"])
                .join_asof(exitq, left_on="exit_time", right_on="exit_quote_ts", by="market_id", strategy="forward", tolerance="1s")
            )
            filt = (
                joined.filter(
                    pl.col("entry_yes_ask").is_not_null()
                    & pl.col("entry_no_ask").is_not_null()
                    & pl.col("exit_yes_bid").is_not_null()
                    & pl.col("exit_no_bid").is_not_null()
                    & (~pl.col("entry_yes_is_stale").fill_null(True))
                    & (~pl.col("entry_no_is_stale").fill_null(True))
                    & (~pl.col("exit_yes_is_stale").fill_null(True))
                    & (~pl.col("exit_no_is_stale").fill_null(True))
                    & (~pl.col("entry_yes_crossed_quote").fill_null(True))
                    & (~pl.col("entry_no_crossed_quote").fill_null(True))
                    & (~pl.col("exit_yes_crossed_quote").fill_null(True))
                    & (~pl.col("exit_no_crossed_quote").fill_null(True))
                    & (pl.col("entry_yes_ask") > 0)
                    & (pl.col("entry_no_ask") > 0)
                    & (pl.col("entry_yes_ask") <= 1)
                    & (pl.col("entry_no_ask") <= 1)
                )
                .with_columns(
                    [
                        (pl.col("exit_yes_bid") - pl.col("entry_yes_ask")).alias("pnl_up"),
                        ((pl.col("exit_yes_bid") - pl.col("entry_yes_ask")) / pl.col("entry_yes_ask")).alias("roi_up"),
                        ((pl.col("exit_yes_bid") - pl.col("entry_yes_ask")) > 0).cast(pl.Int8).alias("label_up_profitable"),
                        (pl.col("exit_no_bid") - pl.col("entry_no_ask")).alias("pnl_down"),
                        ((pl.col("exit_no_bid") - pl.col("entry_no_ask")) / pl.col("entry_no_ask")).alias("roi_down"),
                        ((pl.col("exit_no_bid") - pl.col("entry_no_ask")) > 0).cast(pl.Int8).alias("label_down_profitable"),
                    ]
                )
            )
            frames.append(filt)
            stats.append({"latency": latency, "horizon": horizon, "rows": filt.height, "base_rows_after_tte": left.height})

    result = pl.concat(frames, how="diagonal")
    result.write_parquet(out / "part-00000.parquet")

    labels = ["label_up_profitable", "label_down_profitable"]
    feature_cols = []
    for c in result.columns:
        if c in BANNED_EXACT:
            continue
        if c.startswith(BANNED_PREFIXES):
            continue
        if c in {"entry_time", "exit_time", "entry_quote_ts", "exit_quote_ts"}:
            continue
        if c.startswith("cur_"):
            continue
        if c in labels:
            continue
        # Exclude QC stale flags that are constants after filtering.
        if c.endswith("_is_stale") or c.endswith("_crossed_quote"):
            continue
        feature_cols.append(c)
    (out / "features_pm_repricing_executable.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (out / "labels_pm_repricing_executable.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    lines = [
        "# PM Repricing Executable Gold Report\n\n",
        f"- input gold rows: `{base.height}`\n",
        f"- output rows: `{result.height}`\n",
        f"- features: `{len(feature_cols)}`\n\n",
        "| latency | horizon | base_rows_after_tte | output_rows |\n",
        "| ---: | ---: | ---: | ---: |\n",
    ]
    for s in stats:
        lines.append(f"| {s['latency']} | {s['horizon']} | {s['base_rows_after_tte']} | {s['rows']} |\n")
    lines.append("\n## Label rates\n\n")
    label_summary = result.group_by(["latency_seconds", "exit_horizon_seconds"]).agg(
        [
            pl.len().alias("rows"),
            pl.col("label_up_profitable").mean().alias("up_profitable_rate"),
            pl.col("pnl_up").mean().alias("avg_pnl_up"),
            pl.col("label_down_profitable").mean().alias("down_profitable_rate"),
            pl.col("pnl_down").mean().alias("avg_pnl_down"),
        ]
    ).sort(["latency_seconds", "exit_horizon_seconds"])
    lines.append("| latency | horizon | rows | up_win_rate | avg_pnl_up | down_win_rate | avg_pnl_down |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in label_summary.to_dicts():
        lines.append(f"| {r['latency_seconds']} | {r['exit_horizon_seconds']} | {r['rows']} | {r['up_profitable_rate']} | {r['avg_pnl_up']} | {r['down_profitable_rate']} | {r['avg_pnl_down']} |\n")
    report_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
