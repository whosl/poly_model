from __future__ import annotations

from datetime import datetime
from typing import Iterable

import polars as pl

QUOTE_EVENT_SCHEMA = {
    "market_id": pl.String,
    "asset_id": pl.String,
    "ts_event": pl.Datetime(time_zone="UTC"),
    "ts_recv": pl.Datetime(time_zone="UTC"),
    "source": pl.String,
    "update_bid": pl.Float64,
    "update_ask": pl.Float64,
    "update_bid_size": pl.Float64,
    "update_ask_size": pl.Float64,
    "last_trade_price": pl.Float64,
    "event_type": pl.String,
    "source_file": pl.String,
}


def empty_quote_events() -> pl.DataFrame:
    return pl.DataFrame(schema=QUOTE_EVENT_SCHEMA)


def _ensure_cols(df: pl.DataFrame, cols: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=cols)
    exprs = []
    for col, dtype in cols.items():
        if col not in df.columns:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))
    return df.with_columns(exprs) if exprs else df


def _valid_price_expr(col: str) -> pl.Expr:
    x = pl.col(col).cast(pl.Float64, strict=False)
    return pl.when((x >= 0.0) & (x <= 1.0)).then(x).otherwise(None)


def normalize_quote_events(orderbook_df: pl.DataFrame, price_change_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Normalize orderbook snapshots and price_change rows into quote events.

    `price` is preserved only as `last_trade_price`; it is never used to build bid/ask/mid.
    """
    frames: list[pl.DataFrame] = []
    if orderbook_df is not None and not orderbook_df.is_empty():
        ob = _ensure_cols(orderbook_df, {
            "market_id": pl.String, "asset_id": pl.String, "ts_event": pl.Datetime(time_zone="UTC"), "ts_recv": pl.Datetime(time_zone="UTC"),
            "best_bid": pl.Float64, "best_ask": pl.Float64, "bid_size_1": pl.Float64, "ask_size_1": pl.Float64, "source_file": pl.String,
        })
        frames.append(ob.select(
            pl.col("market_id").cast(pl.String),
            pl.col("asset_id").cast(pl.String),
            pl.col("ts_event"),
            pl.col("ts_recv"),
            pl.lit("orderbook").alias("source"),
            _valid_price_expr("best_bid").alias("update_bid"),
            _valid_price_expr("best_ask").alias("update_ask"),
            pl.col("bid_size_1").cast(pl.Float64, strict=False).alias("update_bid_size"),
            pl.col("ask_size_1").cast(pl.Float64, strict=False).alias("update_ask_size"),
            pl.lit(None, dtype=pl.Float64).alias("last_trade_price"),
            pl.lit("orderbook").alias("event_type"),
            pl.col("source_file").cast(pl.String),
        ))
    if price_change_df is not None and not price_change_df.is_empty():
        pc = _ensure_cols(price_change_df, {
            "market_id": pl.String, "asset_id": pl.String, "ts_event": pl.Datetime(time_zone="UTC"), "ts_recv": pl.Datetime(time_zone="UTC"),
            "best_bid": pl.Float64, "best_ask": pl.Float64, "price": pl.Float64, "event_type": pl.String, "source_file": pl.String,
        })
        frames.append(pc.select(
            pl.col("market_id").cast(pl.String),
            pl.col("asset_id").cast(pl.String),
            pl.col("ts_event"),
            pl.col("ts_recv"),
            pl.lit("price_change").alias("source"),
            _valid_price_expr("best_bid").alias("update_bid"),
            _valid_price_expr("best_ask").alias("update_ask"),
            pl.lit(None, dtype=pl.Float64).alias("update_bid_size"),
            pl.lit(None, dtype=pl.Float64).alias("update_ask_size"),
            _valid_price_expr("price").alias("last_trade_price"),
            pl.col("event_type").cast(pl.String),
            pl.col("source_file").cast(pl.String),
        ))
    if not frames:
        return empty_quote_events()
    return pl.concat(frames, how="vertical").filter(pl.col("market_id").is_not_null() & pl.col("asset_id").is_not_null() & pl.col("ts_event").is_not_null())


def build_asset_quote_state(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(schema={
            "market_id": pl.String, "asset_id": pl.String, "ts_event": pl.Datetime(time_zone="UTC"),
            "current_bid": pl.Float64, "current_ask": pl.Float64, "current_bid_size": pl.Float64, "current_ask_size": pl.Float64,
            "mid": pl.Float64, "spread": pl.Float64, "crossed_quote": pl.Boolean,
            "last_quote_update_ts": pl.Datetime(time_zone="UTC"), "last_trade_price": pl.Float64,
        })
    keys = ["market_id", "asset_id"]
    ev = events.with_columns(
        pl.when(pl.col("source") == "orderbook").then(0).otherwise(1).alias("source_priority"),
        (pl.col("update_bid").is_not_null() | pl.col("update_ask").is_not_null()).alias("has_quote_update"),
    ).sort(keys + ["ts_event", "source_priority"])
    st = ev.with_columns(
        pl.col("update_bid").forward_fill().over(keys).alias("current_bid"),
        pl.col("update_ask").forward_fill().over(keys).alias("current_ask"),
        pl.col("update_bid_size").forward_fill().over(keys).alias("current_bid_size"),
        pl.col("update_ask_size").forward_fill().over(keys).alias("current_ask_size"),
        pl.when(pl.col("has_quote_update")).then(pl.col("ts_event")).otherwise(None).forward_fill().over(keys).alias("last_quote_update_ts"),
        pl.col("last_trade_price").forward_fill().over(keys).alias("last_trade_price_state"),
    ).with_columns(
        pl.when(pl.col("current_bid").is_not_null() & pl.col("current_ask").is_not_null()).then((pl.col("current_bid") + pl.col("current_ask")) / 2.0).otherwise(None).alias("mid"),
        pl.when(pl.col("current_bid").is_not_null() & pl.col("current_ask").is_not_null()).then(pl.col("current_ask") - pl.col("current_bid")).otherwise(None).alias("spread"),
        (pl.col("current_bid").is_not_null() & pl.col("current_ask").is_not_null() & (pl.col("current_bid") > pl.col("current_ask"))).alias("crossed_quote"),
    )
    return st.select(
        "market_id", "asset_id", "ts_event", "source", "current_bid", "current_ask", "current_bid_size", "current_ask_size",
        "mid", "spread", "crossed_quote", "last_quote_update_ts", pl.col("last_trade_price_state").alias("last_trade_price")
    )


def build_asset_state_on_grid(grid: pl.DataFrame, quote_state: pl.DataFrame, asset_id: str, prefix: str, ttl_seconds: int) -> pl.DataFrame:
    if quote_state.is_empty():
        return grid.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_bid"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_ask"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_mid"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_spread"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_bid_depth_5"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_ask_depth_5"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_depth_imbalance_5"),
            pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_quote_age_seconds"),
            pl.lit(True).alias(f"{prefix}_is_stale"),
            pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias(f"{prefix}_last_quote_update_ts"),
            pl.lit(False).alias(f"{prefix}_crossed_quote"),
        )
    asset_state = quote_state.filter(pl.col("asset_id") == asset_id).select(
        "ts_event", "current_bid", "current_ask", "current_bid_size", "current_ask_size",
        "mid", "spread", "crossed_quote", "last_quote_update_ts"
    ).sort("ts_event")
    if asset_state.is_empty():
        return build_asset_state_on_grid(grid, pl.DataFrame(), asset_id, prefix, ttl_seconds)
    joined = grid.join_asof(asset_state, left_on="sample_ts", right_on="ts_event", strategy="backward")
    if "ts_event" in joined.columns:
        joined = joined.drop("ts_event")
    joined = joined.rename({
        "current_bid": f"{prefix}_bid", "current_ask": f"{prefix}_ask", "mid": f"{prefix}_mid", "spread": f"{prefix}_spread",
        "current_bid_size": f"{prefix}_bid_depth_5", "current_ask_size": f"{prefix}_ask_depth_5",
        "last_quote_update_ts": f"{prefix}_last_quote_update_ts", "crossed_quote": f"{prefix}_crossed_quote",
    })
    joined = joined.with_columns(
        ((pl.col("sample_ts") - pl.col(f"{prefix}_last_quote_update_ts")).dt.total_milliseconds() / 1000.0).alias(f"{prefix}_quote_age_seconds"),
        pl.when(pl.col(f"{prefix}_bid_depth_5").is_not_null() & pl.col(f"{prefix}_ask_depth_5").is_not_null() & ((pl.col(f"{prefix}_bid_depth_5") + pl.col(f"{prefix}_ask_depth_5")) > 0))
          .then((pl.col(f"{prefix}_bid_depth_5") - pl.col(f"{prefix}_ask_depth_5")) / (pl.col(f"{prefix}_bid_depth_5") + pl.col(f"{prefix}_ask_depth_5")))
          .otherwise(None).alias(f"{prefix}_depth_imbalance_5"),
    )
    return joined.with_columns(
        (pl.col(f"{prefix}_bid").is_null() | pl.col(f"{prefix}_ask").is_null() | pl.col(f"{prefix}_last_quote_update_ts").is_null() | (pl.col(f"{prefix}_quote_age_seconds") > ttl_seconds)).alias(f"{prefix}_is_stale"),
        pl.col(f"{prefix}_crossed_quote").fill_null(False),
    )
