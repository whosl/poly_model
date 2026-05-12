from __future__ import annotations

import polars as pl

from .math_utils import clip, normal_cdf


def classify_three_way(expr: pl.Expr, threshold: float | pl.Expr, up: str = "UP", down: str = "DOWN", flat: str = "FLAT") -> pl.Expr:
    return (
        pl.when(expr > threshold)
        .then(pl.lit(up))
        .when(expr < (-threshold if isinstance(threshold, (int, float)) else -threshold))
        .then(pl.lit(down))
        .otherwise(pl.lit(flat))
    )


def compute_formula_p_yes_expr(
    current_price_col: str,
    open_price_col: str,
    sigma_col: str,
    tau_col: str,
) -> pl.Expr:
    fields = [
        pl.col(current_price_col).alias("__current_price"),
        pl.col(open_price_col).alias("__open_price"),
        pl.col(sigma_col).alias("__sigma"),
        pl.col(tau_col).alias("__tau"),
    ]
    return pl.struct(fields).map_elements(
        lambda row: _compute_formula_p_yes(
            row["__current_price"],
            row["__open_price"],
            row["__sigma"],
            row["__tau"],
        ),
        return_dtype=pl.Float64,
    )


def _compute_formula_p_yes(current_price: float | None, open_price: float | None, sigma: float | None, tau: float | None) -> float | None:
    if current_price is None or open_price is None or sigma is None or tau is None:
        return None
    if current_price <= 0 or open_price <= 0:
        return None
    sigma = max(float(sigma), 1e-6)
    tau = max(float(tau), 1.0)
    z = __import__("math").log(current_price / open_price) / (sigma * (tau ** 0.5))
    return clip(normal_cdf(z), 0.001, 0.999)


def compute_settled_yes_proxy(open_price: float | None, close_price: float | None) -> int | None:
    if open_price is None or close_price is None:
        return None
    return int(close_price > open_price)
