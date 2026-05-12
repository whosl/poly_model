from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--symbol")
    parser.add_argument("--market-id")
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--source", choices=["binance", "polymarket"])
    parser.add_argument("--skip-bronze", action="store_true")
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_step(script: str, args: argparse.Namespace) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script)), "--config", args.config]
    if args.limit_files is not None and script in {"inspect_raw.py", "build_bronze.py"}:
        command.extend(["--limit-files", str(args.limit_files)])
    if args.start_date and script in {"inspect_raw.py", "build_bronze.py", "build_silver_binance.py", "build_silver_pm.py"}:
        command.extend(["--start-date", args.start_date])
    if args.end_date and script in {"inspect_raw.py", "build_bronze.py", "build_silver_binance.py", "build_silver_pm.py"}:
        command.extend(["--end-date", args.end_date])
    if args.source and script in {"inspect_raw.py", "build_bronze.py"}:
        command.extend(["--source", args.source])
    if args.symbol and script in {"build_silver_binance.py"}:
        command.extend(["--symbol", args.symbol])
    if args.market_id and script in {"build_silver_pm.py"}:
        command.extend(["--market-id", args.market_id])
    if args.dry_run and script in {"inspect_raw.py", "build_bronze.py"}:
        command.append("--dry-run")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    run_step("inspect_raw.py", args)
    if not args.skip_bronze:
        run_step("build_bronze.py", args)
    run_step("build_pm_asset_mapping.py", args)
    if not args.skip_silver:
        run_step("build_silver_binance.py", args)
        run_step("build_silver_pm.py", args)
    if not args.skip_gold:
        run_step("build_gold_btc_direction.py", args)
        run_step("build_gold_pm_terminal.py", args)
        run_step("build_gold_pm_repricing.py", args)
    run_step("validate_datasets.py", args)


if __name__ == "__main__":
    main()
