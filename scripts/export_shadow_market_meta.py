"""Export compact PM market metadata for shadow runtime deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silver-path", default="data/silver/pm_1s")
    p.add_argument("--output-path", default="configs/shadow_market_meta.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = (
        pl.scan_parquet(str(Path(args.silver_path) / "**" / "*.parquet"), hive_partitioning=True, extra_columns="ignore")
        .select(["market_id", "yes_asset_id", "no_asset_id", "market_start_ts", "market_end_ts"])
        .unique(subset=["market_id"], keep="first")
        .collect()
    )
    df.write_parquet(out)
    print(f"Wrote {out} with {df.height} markets")


if __name__ == "__main__":
    main()
