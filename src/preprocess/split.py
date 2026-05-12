from __future__ import annotations

import math

import polars as pl


def apply_time_split(
    df: pl.DataFrame,
    time_col: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("split"))
    if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("split ratios must sum to 1.0")
    sorted_df = df.sort(time_col)
    n = sorted_df.height
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return sorted_df.with_row_index("_row_idx").with_columns(
        pl.when(pl.col("_row_idx") < train_end)
        .then(pl.lit("train"))
        .when(pl.col("_row_idx") < valid_end)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).drop("_row_idx")


def build_market_split_map(
    df: pl.DataFrame,
    market_col: str,
    order_col: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame({market_col: [], "split": []}, schema={market_col: pl.String, "split": pl.String})
    market_order = (
        df.group_by(market_col)
        .agg(pl.min(order_col).alias(order_col))
        .sort(order_col)
        .drop_nulls(order_col)
    )
    market_order = apply_time_split(market_order, order_col, train_ratio, valid_ratio, test_ratio)
    return market_order.select([market_col, "split"])


def apply_market_split(
    df: pl.DataFrame,
    market_col: str,
    order_col: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> pl.DataFrame:
    mapping = build_market_split_map(df, market_col, order_col, train_ratio, valid_ratio, test_ratio)
    return df.join(mapping, on=market_col, how="left")
