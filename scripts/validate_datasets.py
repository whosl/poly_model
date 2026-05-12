from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import polars as pl

from preprocess.config import load_config, resolve_path
from preprocess.dataset_io import scan_parquet_dir
from preprocess.feature_metadata import LEAKY_EXACT, LEAKY_PREFIXES
from preprocess.reporting import markdown_table, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated gold datasets.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def validate_feature_list(columns: list[str]) -> list[str]:
    bad = []
    for col in columns:
        if col in LEAKY_EXACT or col.startswith(LEAKY_PREFIXES):
            bad.append(col)
    return bad


QC_COLUMNS = {"matched_binance_sample_ts", "binance_quote_age_seconds", "binance_is_stale"}


def fmt_ts(value: object) -> str:
    return "" if value is None else str(value)


def quote_age_stats(df: pl.DataFrame) -> dict[str, object]:
    if df.is_empty() or "binance_quote_age_seconds" not in df.columns:
        return {}
    return df.select(
        pl.col("binance_quote_age_seconds").null_count().alias("null_count"),
        pl.col("binance_quote_age_seconds").min().alias("min"),
        pl.col("binance_quote_age_seconds").quantile(0.50).alias("p50"),
        pl.col("binance_quote_age_seconds").quantile(0.90).alias("p90"),
        pl.col("binance_quote_age_seconds").quantile(0.99).alias("p99"),
        pl.col("binance_quote_age_seconds").max().alias("max"),
    ).to_dicts()[0]


def dataset_time_range(df: pl.DataFrame, col: str = "sample_ts") -> dict[str, object]:
    if df.is_empty() or col not in df.columns:
        return {"min": None, "max": None}
    row = df.select(pl.col(col).min().alias("min"), pl.col(col).max().alias("max")).to_dicts()[0]
    return row


def append_pm_binance_coverage_report(
    lines: list[str],
    name: str,
    df: pl.DataFrame,
    binance_range: dict[str, object],
) -> dict[str, object]:
    pm_range = dataset_time_range(df)
    max_binance = binance_range.get("max")
    coverage: dict[str, object] = {
        "pm_sample_ts_min": pm_range["min"],
        "pm_sample_ts_max": pm_range["max"],
        "binance_sample_ts_min": binance_range.get("min"),
        "binance_sample_ts_max": max_binance,
        "rows_after_binance_max": 0,
        "markets_after_binance_max": 0,
        "ratio_after_binance_max": 0.0,
        "binance_quote_age_seconds": quote_age_stats(df),
        "binance_is_stale_rows": 0,
        "qc_columns_in_features": [],
    }
    lines.append("")
    lines.append("### Binance Coverage / Quote Age")
    lines.append("")
    lines.append(f"- pm_sample_ts_min: `{fmt_ts(pm_range['min'])}`")
    lines.append(f"- pm_sample_ts_max: `{fmt_ts(pm_range['max'])}`")
    lines.append(f"- binance_silver_sample_ts_min: `{fmt_ts(binance_range.get('min'))}`")
    lines.append(f"- binance_silver_sample_ts_max: `{fmt_ts(max_binance)}`")

    if df.is_empty() or max_binance is None:
        return coverage

    beyond = df.filter(pl.col("sample_ts") > max_binance)
    rows_after = beyond.height
    markets_after = beyond.select(pl.col("market_id").n_unique()).item() if rows_after and "market_id" in beyond.columns else 0
    ratio_after = rows_after / max(df.height, 1)
    coverage.update(
        {
            "rows_after_binance_max": rows_after,
            "markets_after_binance_max": markets_after,
            "ratio_after_binance_max": ratio_after,
        }
    )
    lines.append(f"- rows_after_binance_max: `{rows_after}`")
    lines.append(f"- markets_after_binance_max: `{markets_after}`")
    lines.append(f"- ratio_after_binance_max: `{round(float(ratio_after), 8)}`")

    if rows_after and "market_id" in beyond.columns:
        market_ranges = (
            beyond.group_by("market_id")
            .agg(
                pl.col("sample_ts").min().alias("min_sample_ts"),
                pl.col("sample_ts").max().alias("max_sample_ts"),
                pl.len().alias("rows"),
            )
            .sort("min_sample_ts")
        )
        coverage["markets_after_binance_max_detail"] = market_ranges.to_dicts()
        lines.append("")
        lines.append("#### Markets after Binance max sample_ts")
        lines.append("")
        lines.extend(markdown_table(["market_id", "min_sample_ts", "max_sample_ts", "rows"], market_ranges.rows()))

    stats = quote_age_stats(df)
    if stats:
        lines.append("")
        lines.append("#### binance_quote_age_seconds distribution")
        lines.append("")
        lines.extend(markdown_table(["metric", "value"], [[key, value] for key, value in stats.items()]))

    if "binance_is_stale" in df.columns:
        stale_rows = int(df.select(pl.col("binance_is_stale").fill_null(True).sum()).item())
        coverage["binance_is_stale_rows"] = stale_rows
        lines.append(f"- binance_is_stale_rows_in_gold: `{stale_rows}`")
    return coverage


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    reports_root = resolve_path(config, "reports")

    datasets = {
        "btc_direction": resolve_path(config, "data/gold/btc_direction_1s"),
        "pm_terminal": resolve_path(config, "data/gold/pm_terminal_1s"),
        "pm_repricing": resolve_path(config, "data/gold/pm_repricing_1s"),
    }
    binance_silver_df = scan_parquet_dir(resolve_path(config, "data/silver/binance_1s")).collect()
    binance_silver_range = dataset_time_range(binance_silver_df)
    summary: dict[str, object] = {}
    lines = ["# Dataset Validation Report", ""]

    for name, path in datasets.items():
        df = scan_parquet_dir(path).collect()
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- exists: `{path.exists()}`")
        lines.append(f"- rows: `{df.height if not df.is_empty() else 0}`")
        feature_files = {
            "btc_direction": path / "features_btc_direction.json",
            "pm_terminal": path / "features_pm_terminal.json",
            "pm_repricing": path / "features_pm_repricing.json",
        }
        features = read_json_list(feature_files[name])
        bad_features = validate_feature_list(features)
        qc_features = sorted(QC_COLUMNS.intersection(features))
        lines.append(f"- feature_leaks: `{bad_features}`")
        if qc_features:
            lines.append(f"- qc_columns_in_features: `{qc_features}`")
        if not df.is_empty():
            null_rows = [[col, round(float(df[col].null_count() / max(df.height, 1)), 6)] for col in df.columns[: min(len(df.columns), 20)]]
            lines.append("")
            lines.append("### Sample Null Ratio")
            lines.append("")
            lines.extend(markdown_table(["column", "null_ratio"], null_rows))
            lines.append("")
            if name != "btc_direction" and "market_id" in df.columns and "split" in df.columns:
                leakage = (
                    df.group_by("market_id").agg(pl.col("split").n_unique().alias("split_count")).filter(pl.col("split_count") > 1).height
                )
                lines.append(f"- market_split_leakage_count: `{leakage}`")
            if "formula_p_yes" in df.columns:
                vals = df["formula_p_yes"].drop_nulls().to_list()
                nan_inf = sum(1 for v in vals if isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                lines.append(f"- formula_p_yes_nan_inf: `{nan_inf}`")
            coverage = None
            if name in {"pm_terminal", "pm_repricing"}:
                coverage = append_pm_binance_coverage_report(lines, name, df, binance_silver_range)
                coverage["qc_columns_in_features"] = qc_features
        else:
            coverage = None
        summary[name] = {
            "rows": 0 if df.is_empty() else df.height,
            "bad_features": bad_features,
        }
        if name in {"pm_terminal", "pm_repricing"}:
            summary[name]["binance_coverage"] = coverage
            summary[name]["qc_columns_in_features"] = qc_features
        lines.append("")

    write_markdown(reports_root / "dataset_validation_report.md", lines)
    write_json(reports_root / "dataset_summary.json", summary)


if __name__ == "__main__":
    main()
