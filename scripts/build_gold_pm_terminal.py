from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import polars as pl

from preprocess.config import load_config, resolve_path
from preprocess.dataset_io import scan_parquet_dir, write_partitioned_parquet
from preprocess.feature_metadata import write_feature_and_label_lists
from preprocess.gold_utils import compute_formula_p_yes_expr, compute_settled_yes_proxy
from preprocess.logging_utils import setup_logging
from preprocess.paths import bootstrap_layout
from preprocess.reporting import write_markdown
from preprocess.split import apply_market_split

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PM terminal probability gold dataset.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def has_parquet_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def add_binance_quote_age(pm_df: pl.DataFrame, binance_df: pl.DataFrame, stale_seconds: float = 5.0) -> pl.DataFrame:
    binance_for_join = binance_df.with_columns(pl.col("sample_ts").alias("matched_binance_sample_ts"))
    joined = pm_df.join_asof(binance_for_join, left_on="sample_ts", right_on="sample_ts", strategy="backward")
    return joined.with_columns(
        ((pl.col("sample_ts") - pl.col("matched_binance_sample_ts")).dt.total_milliseconds() / 1000.0).alias("binance_quote_age_seconds")
    ).with_columns(
        (pl.col("matched_binance_sample_ts").is_null() | (pl.col("binance_quote_age_seconds") > stale_seconds)).alias("binance_is_stale")
    )


def quote_age_report_rows(df: pl.DataFrame) -> list[str]:
    if df.is_empty() or "binance_quote_age_seconds" not in df.columns:
        return []
    stats = df.select(
        pl.col("binance_quote_age_seconds").null_count().alias("null_count"),
        pl.col("binance_quote_age_seconds").min().alias("min"),
        pl.col("binance_quote_age_seconds").quantile(0.50).alias("p50"),
        pl.col("binance_quote_age_seconds").quantile(0.90).alias("p90"),
        pl.col("binance_quote_age_seconds").quantile(0.99).alias("p99"),
        pl.col("binance_quote_age_seconds").max().alias("max"),
    ).to_dicts()[0]
    return [f"- binance_quote_age_seconds_{key}: `{value}`" for key, value in stats.items()]


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    bootstrap_layout(config)

    pm_root = resolve_path(config, "data/silver/pm_1s")
    binance_root = resolve_path(config, "data/silver/binance_1s")
    out_root = resolve_path(config, "data/gold/pm_terminal_1s")

    if not has_parquet_files(pm_root) or not has_parquet_files(binance_root):
        write_feature_and_label_lists(out_root, "features_pm_terminal.json", "labels_pm_terminal.json", [], [])
        write_markdown(resolve_path(config, "reports/gold_pm_terminal_report.md"), ["# Gold PM Terminal Report", "", "No data available."])
        return

    pm_df = scan_parquet_dir(pm_root).collect().sort("sample_ts")
    binance_df = scan_parquet_dir(binance_root).collect().sort("sample_ts")

    if pm_df.is_empty() or binance_df.is_empty():
        write_feature_and_label_lists(out_root, "features_pm_terminal.json", "labels_pm_terminal.json", [], [])
        write_markdown(resolve_path(config, "reports/gold_pm_terminal_report.md"), ["# Gold PM Terminal Report", "", "No data available."])
        return

    raw_pm_rows = pm_df.height
    pm_df = pm_df.filter(
        (pl.col("mapping_status") == "ok")
        & (~pl.col("yes_is_stale"))
        & (~pl.col("no_is_stale"))
        & (pl.col("time_to_expiry_seconds") > 0)
        & (pl.col("time_elapsed_seconds") >= 0)
    ).drop_nulls(["yes_ask", "no_ask", "yes_mid", "no_mid"])
    if pm_df.is_empty():
        write_feature_and_label_lists(out_root, "features_pm_terminal.json", "labels_pm_terminal.json", [], [])
        write_markdown(resolve_path(config, "reports/gold_pm_terminal_report.md"), ["# Gold PM Terminal Report", "", "All rows filtered out."])
        return

    pm_filtered_rows = pm_df.height
    joined_pre_stale = add_binance_quote_age(pm_df, binance_df)
    binance_stale_rows = joined_pre_stale.filter(pl.col("binance_is_stale")).height
    joined = joined_pre_stale.filter(~pl.col("binance_is_stale"))
    if joined.is_empty():
        write_feature_and_label_lists(out_root, "features_pm_terminal.json", "labels_pm_terminal.json", [], [])
        report_lines = [
            "# Gold PM Terminal Report",
            "",
            f"- raw_pm_rows: `{raw_pm_rows}`",
            f"- after_pm_filters_rows: `{pm_filtered_rows}`",
            f"- binance_stale_filtered_rows: `{binance_stale_rows}`",
            "- rows: `0`",
            "",
            "All rows filtered out by Binance quote-age filter.",
        ]
        report_lines.extend(quote_age_report_rows(joined_pre_stale))
        write_markdown(resolve_path(config, "reports/gold_pm_terminal_report.md"), report_lines)
        return
    open_px = (
        joined.group_by("market_id")
        .agg(
            pl.first("market_start_ts").alias("market_start_ts_first"),
            pl.first("market_end_ts").alias("market_end_ts_first"),
        )
        .sort("market_start_ts_first")
        .join_asof(binance_df.select(["sample_ts", "mid_price"]).sort("sample_ts"), left_on="market_start_ts_first", right_on="sample_ts", strategy="backward")
        .rename({"mid_price": "btc_open_price"})
        .sort("market_end_ts_first")
        .join_asof(binance_df.select(["sample_ts", "mid_price"]).sort("sample_ts"), left_on="market_end_ts_first", right_on="sample_ts", strategy="backward", suffix="_close")
        .rename({"mid_price": "btc_close_price"})
        .select(["market_id", "btc_open_price", "btc_close_price"])
    )
    joined = joined.join(open_px, on="market_id", how="left").with_columns(
        pl.col("mid_price").alias("btc_current_price"),
        (pl.col("mid_price") / pl.col("btc_open_price") - 1.0).alias("return_since_open"),
        (pl.col("mid_price") - pl.col("btc_open_price")).alias("distance_to_open"),
        compute_formula_p_yes_expr("mid_price", "btc_open_price", "realized_vol_60s", "time_to_expiry_seconds").alias("formula_p_yes"),
    ).with_columns(
        (pl.col("formula_p_yes") - pl.col("yes_ask")).alias("edge_to_yes_ask"),
        ((1.0 - pl.col("formula_p_yes")) - pl.col("no_ask")).alias("edge_to_no_ask"),
        pl.struct(["btc_open_price", "btc_close_price"]).map_elements(
            lambda row: compute_settled_yes_proxy(row["btc_open_price"], row["btc_close_price"]),
            return_dtype=pl.Int64,
        ).alias("settled_yes"),
        pl.lit("binance_proxy").alias("label_source"),
    )

    split_cfg = config["split"]
    joined = apply_market_split(
        joined,
        market_col="market_id",
        order_col="market_start_ts",
        train_ratio=float(split_cfg["train_ratio"]),
        valid_ratio=float(split_cfg["valid_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )

    gold = joined.select(
        [
            "market_id",
            "sample_ts",
            "market_start_ts",
            "market_end_ts",
            "time_elapsed_seconds",
            "time_to_expiry_seconds",
            "btc_open_price",
            "btc_current_price",
            "return_since_open",
            "distance_to_open",
            "realized_vol_10s",
            "realized_vol_30s",
            "realized_vol_60s",
            "trade_imbalance_1s",
            "trade_imbalance_5s",
            "trade_imbalance_30s",
            "depth_imbalance_5",
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
            "edge_to_yes_ask",
            "edge_to_no_ask",
            "settled_yes",
            "label_source",
            "matched_binance_sample_ts",
            "binance_quote_age_seconds",
            "binance_is_stale",
            "split",
        ]
    ).rename(
        {
            "trade_imbalance_1s": "binance_trade_imbalance_1s",
            "trade_imbalance_5s": "binance_trade_imbalance_5s",
            "trade_imbalance_30s": "binance_trade_imbalance_30s",
            "depth_imbalance_5": "binance_depth_imbalance_5",
        }
    ).with_columns(pl.col("sample_ts").dt.date().cast(pl.String).alias("date"))

    write_partitioned_parquet(gold, out_root, ["date", "split"], basename="gold")
    label_cols = ["settled_yes"]
    write_feature_and_label_lists(
        out_root,
        "features_pm_terminal.json",
        "labels_pm_terminal.json",
        gold.columns,
        label_cols,
        extra_exclude={
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
        },
    )
    report_lines = [
        "# Gold PM Terminal Report",
        "",
        f"- raw_pm_rows: `{raw_pm_rows}`",
        f"- after_pm_filters_rows: `{pm_filtered_rows}`",
        f"- binance_stale_filtered_rows: `{binance_stale_rows}`",
        f"- rows: `{gold.height}`",
        f"- filtered_total_rows: `{raw_pm_rows - gold.height}`",
    ]
    report_lines.append("")
    report_lines.append("## Binance Quote Age Distribution Before Filtering")
    report_lines.append("")
    report_lines.extend(quote_age_report_rows(joined_pre_stale))
    write_markdown(resolve_path(config, "reports/gold_pm_terminal_report.md"), report_lines)
    logger.info("Wrote PM terminal gold dataset with %d rows", gold.height)


if __name__ == "__main__":
    main()
