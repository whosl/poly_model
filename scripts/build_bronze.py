from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from preprocess.bronze import (
    BronzeWriter,
    build_bronze_for_file,
    iter_raw_files,
    write_bronze_quality_report,
    write_corrupt_raw_report,
)
from preprocess.config import load_config
from preprocess.logging_utils import setup_logging
from preprocess.paths import bootstrap_layout, link_or_copy_raw_layout

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bronze parquet datasets from raw JSONL/JSON.GZ feeds.")
    parser.add_argument("--config", required=True, help="Path to configs/preprocess.yaml")
    parser.add_argument("--limit-files", type=int, default=None, help="Limit number of raw files processed")
    parser.add_argument("--start-date", help="Inclusive date filter in YYYYMMDD")
    parser.add_argument("--end-date", help="Inclusive date filter in YYYYMMDD")
    parser.add_argument("--source", choices=["binance", "polymarket"], help="Restrict to one source")
    parser.add_argument("--only", choices=["pm_orderbook", "pm_price_change", "pm_market_meta", "binance_bookticker", "binance_aggtrade", "binance_depth"], help="Write only one bronze dataset; other parsed record types are skipped")
    parser.add_argument("--force", action="store_true", help="Remove the selected bronze output directory before writing. Requires --only.")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare raw paths and log files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    bootstrap_layout(config)
    operations = link_or_copy_raw_layout(config, dry_run=args.dry_run)
    logger.info("Prepared %d raw file links under data/raw", len(operations))

    files = iter_raw_files(
        config,
        limit_files=args.limit_files,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
    )
    logger.info("Found %d raw files to process", len(files))
    if args.dry_run:
        for path in files:
            logger.info("Dry run raw file: %s", path)
        return

    writer = BronzeWriter(config)
    if args.force:
        if not args.only:
            raise ValueError("--force requires --only to avoid deleting all bronze outputs accidentally")
        target = writer.targets[args.only].root_path
        if target.exists():
            logger.info("Removing existing bronze target for --only %s: %s", args.only, target)
            shutil.rmtree(target)
    if args.only:
        writer.targets = {args.only: writer.targets[args.only]}
    totals: dict[str, int] = {}
    results = []
    for path in files:
        logger.info("Building bronze from %s", path)
        result = build_bronze_for_file(path, config, writer)
        results.append(result)
        logger.info("Finished %s -> %s", path, result.counts)
        for key, value in result.counts.items():
            totals[key] = totals.get(key, 0) + value

    logger.info("Bronze totals: %s", totals)
    quality_report = write_bronze_quality_report(config, results)
    corrupt_report = write_corrupt_raw_report(config, results)
    logger.info("Wrote %s", quality_report)
    logger.info("Wrote %s", corrupt_report)


if __name__ == "__main__":
    main()
