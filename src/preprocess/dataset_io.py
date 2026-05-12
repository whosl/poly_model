from __future__ import annotations

from pathlib import Path

import polars as pl

from .io_utils import ensure_dir


def scan_parquet_dir(path: Path) -> pl.LazyFrame:
    if not path.exists() or not any(path.rglob("*.parquet")):
        return pl.LazyFrame()
    return pl.scan_parquet(str(path / "**/*.parquet"), hive_partitioning=True)


def write_partitioned_parquet(df: pl.DataFrame, root: Path, partition_cols: list[str], basename: str = "part") -> None:
    ensure_dir(root)
    if df.is_empty():
        return
    unique_parts = df.select(partition_cols).unique().iter_rows(named=True)
    for part in unique_parts:
        subset = df
        dir_parts = []
        for col in partition_cols:
            value = part[col]
            subset = subset.filter(pl.col(col) == value)
            dir_parts.append(f"{col}={value}")
        out_dir = root.joinpath(*dir_parts)
        ensure_dir(out_dir)
        subset.write_parquet(out_dir / f"{basename}.parquet")
