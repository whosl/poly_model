from __future__ import annotations

import argparse
from datetime import date
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
from preprocess.logging_utils import setup_logging
from preprocess.paths import bootstrap_layout
from preprocess.reporting import markdown_table, write_markdown

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Binance 1s silver feature table.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--symbol", default="BTCUSDT")
    return parser.parse_args()


def apply_date_filter(lf: pl.LazyFrame, start_date: str | None, end_date: str | None) -> pl.LazyFrame:
    if lf.collect_schema().names() == []:
        return lf
    if start_date:
        start = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:]))
        lf = lf.filter(pl.col("date") >= start)
    if end_date:
        end = date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:]))
        lf = lf.filter(pl.col("date") <= end)
    return lf


def build_grid(book_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    book_df = book_df.filter(pl.col("symbol") == symbol).sort("ts_event")
    if book_df.is_empty():
        return pl.DataFrame(schema={"sample_ts": pl.Datetime(time_zone="UTC"), "symbol": pl.String})
    start = book_df["ts_event"].min().replace(microsecond=0)
    end = book_df["ts_event"].max().replace(microsecond=0)
    return pl.DataFrame(
        {
            "sample_ts": pl.datetime_range(start, end, interval="1s", eager=True, time_zone="UTC"),
            "symbol": [symbol] * ((int((end - start).total_seconds()) + 1)),
        }
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    bootstrap_layout(config)

    bronze_root = resolve_path(config, config["output_paths"]["bronze"])
    book_lf = apply_date_filter(scan_parquet_dir(bronze_root / "binance_bookticker"), args.start_date, args.end_date)
    trade_lf = apply_date_filter(scan_parquet_dir(bronze_root / "binance_aggtrade"), args.start_date, args.end_date)
    depth_lf = apply_date_filter(scan_parquet_dir(bronze_root / "binance_depth"), args.start_date, args.end_date)

    book_df = book_lf.filter(pl.col("symbol") == args.symbol).select(
        ["ts_event", "symbol", "bid_price", "ask_price", "bid_qty", "ask_qty", "mid_price", "spread", "microprice"]
    ).sort("ts_event").collect()
    trade_df = trade_lf.filter(pl.col("symbol") == args.symbol).select(
        ["ts_event", "symbol", "qty", "side", "notional"]
    ).sort("ts_event").collect()
    depth_df = depth_lf.filter(pl.col("symbol") == args.symbol).select(
        ["ts_event", "symbol", "bid_depth_5", "ask_depth_5", "depth_imbalance_5"]
    ).sort("ts_event").collect()

    grid = build_grid(book_df, args.symbol)
    if grid.is_empty():
        out_root = resolve_path(config, "data/silver/binance_1s")
        write_partitioned_parquet(pl.DataFrame(), out_root, ["date", "symbol"])
        write_markdown(resolve_path(config, "reports/silver_binance_report.md"), ["# Silver Binance Report", "", "No data available."])
        return

    silver = grid.join_asof(book_df, left_on="sample_ts", right_on="ts_event", by="symbol", strategy="backward")
    silver = silver.join_asof(depth_df, left_on="sample_ts", right_on="ts_event", by="symbol", strategy="backward", suffix="_depth")
    silver = silver.with_columns((pl.col("microprice") - pl.col("mid_price")).alias("microprice_minus_mid"))

    per_sec_trades = (
        trade_df.with_columns(pl.col("ts_event").dt.truncate("1s").alias("sample_ts"))
        .with_columns(
            pl.when(pl.col("side") == "buy_aggressor").then(pl.col("qty")).otherwise(0.0).alias("buy_qty"),
            pl.when(pl.col("side") == "sell_aggressor").then(pl.col("qty")).otherwise(0.0).alias("sell_qty"),
        )
        .group_by(["symbol", "sample_ts"])
        .agg(
            pl.sum("buy_qty").alias("buy_volume_1s_raw"),
            pl.sum("sell_qty").alias("sell_volume_1s_raw"),
            pl.sum("qty").alias("total_volume_1s_raw"),
            pl.len().alias("trade_count_1s_raw"),
        )
    )
    silver = silver.join(per_sec_trades, on=["symbol", "sample_ts"], how="left").with_columns(
        pl.col("buy_volume_1s_raw").fill_null(0.0),
        pl.col("sell_volume_1s_raw").fill_null(0.0),
        pl.col("total_volume_1s_raw").fill_null(0.0),
        pl.col("trade_count_1s_raw").fill_null(0),
    )
    silver = silver.sort("sample_ts").with_columns(
        (pl.col("buy_volume_1s_raw") - pl.col("sell_volume_1s_raw")).alias("net_volume_1s"),
        pl.when(pl.col("total_volume_1s_raw") > 0)
        .then((pl.col("buy_volume_1s_raw") - pl.col("sell_volume_1s_raw")) / pl.col("total_volume_1s_raw"))
        .otherwise(None)
        .alias("trade_imbalance_1s"),
    )
    for window in (1, 5, 30):
        silver = silver.with_columns(
            pl.col("buy_volume_1s_raw").rolling_sum(window_size=window).alias(f"buy_volume_{window}s"),
            pl.col("sell_volume_1s_raw").rolling_sum(window_size=window).alias(f"sell_volume_{window}s"),
            pl.col("total_volume_1s_raw").rolling_sum(window_size=window).alias(f"total_volume_{window}s"),
            pl.col("trade_count_1s_raw").rolling_sum(window_size=window).alias(f"trade_count_{window}s"),
        ).with_columns(
            (pl.col(f"buy_volume_{window}s") - pl.col(f"sell_volume_{window}s")).alias(f"net_volume_{window}s"),
            pl.when(pl.col(f"total_volume_{window}s") > 0)
            .then((pl.col(f"buy_volume_{window}s") - pl.col(f"sell_volume_{window}s")) / pl.col(f"total_volume_{window}s"))
            .otherwise(None)
            .alias(f"trade_imbalance_{window}s"),
        )

    silver = silver.with_columns(
        (pl.col("mid_price") / pl.col("mid_price").shift(1) - 1.0).alias("return_1s"),
        (pl.col("mid_price") / pl.col("mid_price").shift(5) - 1.0).alias("return_5s"),
        (pl.col("mid_price") / pl.col("mid_price").shift(10) - 1.0).alias("return_10s"),
        (pl.col("mid_price") / pl.col("mid_price").shift(30) - 1.0).alias("return_30s"),
        (pl.col("mid_price").log() - pl.col("mid_price").shift(1).log()).alias("log_return_1s"),
    )
    for window in (10, 30, 60):
        silver = silver.with_columns(
            pl.col("log_return_1s").rolling_std(window_size=window).alias(f"realized_vol_{window}s")
        )

    silver = silver.with_columns(pl.col("sample_ts").dt.date().cast(pl.String).alias("date")).select(
        [
            "sample_ts",
            "symbol",
            "bid_price",
            "ask_price",
            "bid_qty",
            "ask_qty",
            "mid_price",
            "spread",
            "microprice",
            "microprice_minus_mid",
            "return_1s",
            "return_5s",
            "return_10s",
            "return_30s",
            "realized_vol_10s",
            "realized_vol_30s",
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
            "date",
        ]
    )

    out_root = resolve_path(config, "data/silver/binance_1s")
    write_partitioned_parquet(silver, out_root, ["date", "symbol"], basename="silver")

    null_rows = [[col, round(float(silver[col].null_count() / max(silver.height, 1)), 6)] for col in silver.columns]
    report_lines = [
        "# Silver Binance Report",
        "",
        f"- rows: `{silver.height}`",
        f"- symbol: `{args.symbol}`",
        f"- sample_ts_min: `{silver['sample_ts'].min()}`",
        f"- sample_ts_max: `{silver['sample_ts'].max()}`",
        "",
        "## Null Ratio",
        "",
    ]
    report_lines.extend(markdown_table(["column", "null_ratio"], null_rows))
    write_markdown(resolve_path(config, "reports/silver_binance_report.md"), report_lines)
    logger.info("Wrote silver Binance dataset with %d rows", silver.height)


if __name__ == "__main__":
    main()
