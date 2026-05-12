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
from preprocess.gold_utils import classify_three_way
from preprocess.logging_utils import setup_logging
from preprocess.paths import bootstrap_layout
from preprocess.reporting import write_markdown

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BTC direction gold dataset.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    bootstrap_layout(config)

    silver_root = resolve_path(config, "data/silver/binance_1s")
    df = scan_parquet_dir(silver_root).collect().sort("sample_ts")
    if df.is_empty():
        out_root = resolve_path(config, "data/gold/btc_direction_1s")
        write_feature_and_label_lists(out_root, "features_btc_direction.json", "labels_btc_direction.json", [], [])
        write_markdown(resolve_path(config, "reports/gold_btc_direction_report.md"), ["# Gold BTC Direction Report", "", "No data available."])
        return

    threshold = float(config["labeling"]["btc_direction"]["fixed_bps"]) / 10000.0
    gold = df.with_columns(
        pl.col("mid_price").shift(-1).alias("future_mid_1s"),
        pl.col("mid_price").shift(-5).alias("future_mid_5s"),
        pl.col("mid_price").shift(-30).alias("future_mid_30s"),
    ).with_columns(
        (pl.col("future_mid_1s") / pl.col("mid_price") - 1.0).alias("future_return_1s"),
        (pl.col("future_mid_5s") / pl.col("mid_price") - 1.0).alias("future_return_5s"),
        (pl.col("future_mid_30s") / pl.col("mid_price") - 1.0).alias("future_return_30s"),
    ).with_columns(
        classify_three_way(pl.col("future_return_1s"), threshold).alias("label_1s"),
        classify_three_way(pl.col("future_return_5s"), threshold).alias("label_5s"),
        classify_three_way(pl.col("future_return_30s"), threshold).alias("label_30s"),
    )

    out_root = resolve_path(config, "data/gold/btc_direction_1s")
    write_partitioned_parquet(gold, out_root, ["date", "symbol"], basename="gold")
    label_cols = ["label_1s", "label_5s", "label_30s"]
    write_feature_and_label_lists(out_root, "features_btc_direction.json", "labels_btc_direction.json", gold.columns, label_cols)
    write_markdown(
        resolve_path(config, "reports/gold_btc_direction_report.md"),
        [
            "# Gold BTC Direction Report",
            "",
            f"- rows: `{gold.height}`",
            f"- threshold_fixed_bps: `{config['labeling']['btc_direction']['fixed_bps']}`",
        ],
    )
    logger.info("Wrote BTC direction gold dataset with %d rows", gold.height)


if __name__ == "__main__":
    main()
