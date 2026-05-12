from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import polars as pl
import yaml

from preprocess.reporting import markdown_table, write_markdown


EDGE_BUCKETS = [
    (None, -0.10, "[-inf, -0.10)"),
    (-0.10, -0.05, "[-0.10, -0.05)"),
    (-0.05, -0.03, "[-0.05, -0.03)"),
    (-0.03, -0.01, "[-0.03, -0.01)"),
    (-0.01, 0.00, "[-0.01, 0.00)"),
    (0.00, 0.01, "[0.00, 0.01)"),
    (0.01, 0.03, "[0.01, 0.03)"),
    (0.03, 0.05, "[0.03, 0.05)"),
    (0.05, 0.10, "[0.05, 0.10)"),
    (0.10, None, "[0.10, inf)"),
]

TTE_BUCKETS = [
    (240, 300, "[240, 300]"),
    (180, 240, "[180, 240)"),
    (120, 180, "[120, 180)"),
    (60, 120, "[60, 120)"),
    (30, 60, "[30, 60)"),
    (10, 30, "[10, 30)"),
    (0, 10, "[0, 10)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 EDA for gold datasets.")
    parser.add_argument("--config", default="configs/model_stage1.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def scan_dataset(path: str | Path) -> pl.DataFrame:
    root = PROJECT_ROOT / Path(path)
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pl.DataFrame()
    return pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True).collect()


def read_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def dist_table(df: pl.DataFrame, cols: list[str]) -> list[list[Any]]:
    rows = []
    for col in cols:
        if col not in df.columns or df.is_empty():
            continue
        stats = df.select(
            pl.col(col).null_count().alias("nulls"),
            pl.col(col).min().alias("min"),
            pl.col(col).quantile(0.01).alias("p01"),
            pl.col(col).quantile(0.05).alias("p05"),
            pl.col(col).quantile(0.25).alias("p25"),
            pl.col(col).quantile(0.50).alias("p50"),
            pl.col(col).quantile(0.75).alias("p75"),
            pl.col(col).quantile(0.95).alias("p95"),
            pl.col(col).quantile(0.99).alias("p99"),
            pl.col(col).max().alias("max"),
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
        ).to_dicts()[0]
        rows.append([col] + [round_float(stats[k]) for k in ["nulls", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean", "std"]])
    return rows


def round_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value


def bucket_expr(col: str, buckets: list[tuple[float | None, float | None, str]]) -> pl.Expr:
    expr = None
    c = pl.col(col)
    for lo, hi, label in buckets:
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (c >= lo)
        if hi is not None:
            # Use closed high only for the top TTE [240,300] style bucket.
            cond = cond & ((c <= hi) if label == "[240, 300]" else (c < hi))
        arm = pl.when(cond).then(pl.lit(label))
        expr = arm if expr is None else expr.when(cond).then(pl.lit(label))
    return expr.otherwise(pl.lit("unbucketed")) if expr is not None else pl.lit("unbucketed")


def rate_table(df: pl.DataFrame, bucket_col: str, label_col: str, order: list[str], extra_rate_name: str = "settled_yes_rate") -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()
    out = (
        df.group_by(bucket_col)
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("market_count") if "market_id" in df.columns else pl.lit(None).alias("market_count"),
            pl.col(label_col).mean().alias(extra_rate_name),
        )
        .with_columns(pl.col(bucket_col).replace_strict({v: i for i, v in enumerate(order)}, default=999).alias("_order"))
        .sort("_order")
        .drop("_order")
    )
    return out


def formula_deciles(df: pl.DataFrame) -> pl.DataFrame:
    src = df.filter(pl.col("formula_p_yes").is_not_null() & pl.col("settled_yes").is_not_null()).sort("formula_p_yes")
    n = src.height
    if n == 0:
        return pl.DataFrame()
    return (
        src.with_row_index("_idx")
        .with_columns(((pl.col("_idx") * 10 / n).floor().clip(0, 9).cast(pl.Int64) + 1).alias("decile"))
        .group_by("decile")
        .agg(
            pl.len().alias("rows"),
            pl.col("market_id").n_unique().alias("market_count"),
            pl.col("formula_p_yes").min().alias("formula_min"),
            pl.col("formula_p_yes").max().alias("formula_max"),
            pl.col("formula_p_yes").mean().alias("formula_mean"),
            pl.col("settled_yes").mean().alias("settled_yes_rate"),
        )
        .sort("decile")
    )


def is_monotonic(values: list[float | None], direction: str = "increasing", tolerance: float = 0.02) -> bool:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 3:
        return False
    pairs = zip(clean, clean[1:])
    if direction == "increasing":
        return all(b + tolerance >= a for a, b in pairs)
    return all(b <= a + tolerance for a, b in pairs)


def write_basic_eda(name: str, df: pl.DataFrame, report_path: Path, label_cols: list[str]) -> None:
    lines = [f"# {name} EDA", ""]
    lines.append(f"- rows: `{df.height}`")
    if df.is_empty():
        write_markdown(report_path, lines)
        return
    if "market_id" in df.columns:
        lines.append(f"- market_count: `{df.select(pl.col('market_id').n_unique()).item()}`")
    if "date" in df.columns:
        rng = df.select(pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")).to_dicts()[0]
        lines.append(f"- date_range: `{rng['min_date']}` to `{rng['max_date']}`")
    if "split" in df.columns:
        lines.append("")
        lines.append("## Split Distribution")
        lines.extend(markdown_table(["split", "rows"], df.group_by("split").agg(pl.len().alias("rows")).sort("split").rows()))
    for label in label_cols:
        if label in df.columns:
            lines.append("")
            lines.append(f"## {label} Distribution")
            lines.extend(markdown_table([label, "rows"], df.group_by(label).agg(pl.len().alias("rows")).sort(label).rows()))
    write_markdown(report_path, lines)


def write_pm_terminal_eda(df: pl.DataFrame, out_dir: Path, dataset_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# PM Terminal EDA", ""]
    lines.append(f"- rows: `{df.height}`")
    lines.append(f"- market_count: `{df.select(pl.col('market_id').n_unique()).item() if not df.is_empty() else 0}`")
    if not df.is_empty():
        rng = df.select(pl.col("sample_ts").min().alias("min_ts"), pl.col("sample_ts").max().alias("max_ts")).to_dicts()[0]
        lines.append(f"- sample_ts_range: `{rng['min_ts']}` to `{rng['max_ts']}`")

    lines.append("")
    lines.append("## Split Distribution")
    split_dist = df.group_by("split").agg(pl.len().alias("rows"), pl.col("market_id").n_unique().alias("market_count")).sort("split")
    lines.extend(markdown_table(["split", "rows", "market_count"], split_dist.rows()))

    lines.append("")
    lines.append("## Split Date Range / Market Count")
    split_range = df.group_by("split").agg(
        pl.col("sample_ts").min().alias("min_sample_ts"),
        pl.col("sample_ts").max().alias("max_sample_ts"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("market_id").n_unique().alias("market_count"),
        pl.len().alias("rows"),
    ).sort("split")
    lines.extend(markdown_table(split_range.columns, split_range.rows()))

    for title, col in [("Settled Yes Distribution", "settled_yes"), ("Label Source Distribution", "label_source")]:
        lines.append("")
        lines.append(f"## {title}")
        lines.extend(markdown_table([col, "rows", "ratio"], df.group_by(col).agg(pl.len().alias("rows")).with_columns((pl.col("rows") / df.height).alias("ratio")).sort(col).rows()))

    dist_cols = ["formula_p_yes", "pair_mid_sum", "yes_ask", "no_ask", "yes_spread", "no_spread", "time_to_expiry_seconds"]
    lines.append("")
    lines.append("## Numeric Distributions")
    lines.extend(markdown_table(["field", "nulls", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean", "std"], dist_table(df, dist_cols)))

    deciles = formula_deciles(df)
    deciles.write_csv(out_dir / "pm_terminal_formula_deciles.csv")
    lines.append("")
    lines.append("## formula_p_yes decile vs settled_yes_rate")
    lines.extend(markdown_table(deciles.columns, deciles.rows()))

    edge_order = [b[2] for b in EDGE_BUCKETS]
    edge_df = df.with_columns(
        bucket_expr("edge_to_yes_ask", EDGE_BUCKETS).alias("edge_to_yes_ask_bucket"),
        bucket_expr("edge_to_no_ask", EDGE_BUCKETS).alias("edge_to_no_ask_bucket"),
        (1 - pl.col("settled_yes")).alias("no_settle"),
    )
    yes_edge = rate_table(edge_df, "edge_to_yes_ask_bucket", "settled_yes", edge_order, "yes_win_rate").rename({"edge_to_yes_ask_bucket": "bucket"}).with_columns(pl.lit("edge_to_yes_ask").alias("edge_field"))
    no_edge = rate_table(edge_df, "edge_to_no_ask_bucket", "no_settle", edge_order, "no_win_rate").rename({"edge_to_no_ask_bucket": "bucket"}).with_columns(pl.lit("edge_to_no_ask").alias("edge_field"))
    edge_cols = ["edge_field", "bucket", "rows", "market_count", "yes_win_rate", "no_win_rate"]
    edge_out = pl.concat([
        yes_edge.select(["edge_field", "bucket", "rows", "market_count", "yes_win_rate"]).with_columns(pl.lit(None, dtype=pl.Float64).alias("no_win_rate")).select(edge_cols),
        no_edge.select(["edge_field", "bucket", "rows", "market_count", "no_win_rate"]).with_columns(pl.lit(None, dtype=pl.Float64).alias("yes_win_rate")).select(edge_cols),
    ], how="vertical")
    edge_out.write_csv(out_dir / "pm_terminal_edge_buckets.csv")
    lines.append("")
    lines.append("## Edge buckets")
    lines.append("- YES edge bucket uses `yes_win_rate = mean(settled_yes)`.")
    lines.append("- NO edge bucket uses `no_win_rate = mean(1 - settled_yes)`.")
    lines.extend(markdown_table(edge_out.columns, edge_out.rows()))

    tte_order = [b[2] for b in TTE_BUCKETS]
    tte = df.with_columns(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
    tte_out = rate_table(tte, "time_to_expiry_bucket", "settled_yes", tte_order, "settled_yes_rate")
    tte_out.write_csv(out_dir / "pm_terminal_tte_buckets.csv")
    lines.append("")
    lines.append("## time_to_expiry buckets")
    lines.extend(markdown_table(tte_out.columns, tte_out.rows()))

    per_date = df.group_by("date").agg(pl.len().alias("rows"), pl.col("market_id").n_unique().alias("market_count"), pl.col("settled_yes").mean().alias("settled_yes_rate")).sort("date")
    lines.append("")
    lines.append("## Per-date row count / market count / settled_yes rate")
    lines.extend(markdown_table(per_date.columns, per_date.rows()))

    features = read_json(dataset_root / "features_pm_terminal.json")
    feature_cols = [c for c in features if c in df.columns]
    null_rows = []
    for col in feature_cols:
        null_rows.append([col, df[col].null_count(), round(df[col].null_count() / max(df.height, 1), 8)])
    null_rows = sorted(null_rows, key=lambda r: r[2], reverse=True)[:30]
    lines.append("")
    lines.append("## Feature null ratio top 30")
    lines.extend(markdown_table(["feature", "null_count", "null_ratio"], null_rows))

    decile_rates = deciles.get_column("settled_yes_rate").to_list() if not deciles.is_empty() else []
    yes_edge_rates = yes_edge.get_column("yes_win_rate").to_list() if not yes_edge.is_empty() else []
    no_edge_rates = no_edge.get_column("no_win_rate").to_list() if not no_edge.is_empty() else []
    formula_mono = is_monotonic(decile_rates, "increasing")
    yes_edge_mono = is_monotonic(yes_edge_rates, "increasing")
    no_edge_mono = is_monotonic(no_edge_rates, "increasing")
    recommend = formula_mono or yes_edge_mono or no_edge_mono
    lines.append("")
    lines.append("## Sanity conclusion")
    lines.append(f"- formula_p_yes roughly monotonic: `{formula_mono}`")
    lines.append(f"- edge_to_yes_ask roughly monotonic: `{yes_edge_mono}`")
    lines.append(f"- edge_to_no_ask roughly monotonic: `{no_edge_mono}`")
    lines.append(f"- recommend_continue_training: `{recommend}`")
    if recommend:
        lines.append("- conclusion: Signal sanity is sufficient to continue PM terminal baseline training.")
    else:
        lines.append("- conclusion: Monotonicity is weak; continue only as a diagnostic baseline.")

    write_markdown(out_dir / "pm_terminal_eda.md", lines)


def write_pm_repricing_label_distribution(df: pl.DataFrame, out_dir: Path) -> None:
    labels = [c for c in ["label_reprice_1s", "label_reprice_5s", "label_reprice_30s"] if c in df.columns]
    frames = []
    for label in labels:
        frames.append(df.group_by(label).agg(pl.len().alias("rows")).rename({label: "label_value"}).with_columns(pl.lit(label).alias("label")))
    out = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    out.write_csv(out_dir / "pm_repricing_label_distribution.csv")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = PROJECT_ROOT / cfg["reports"]["eda_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    btc = scan_dataset(cfg["data"]["btc_direction"])
    pm_terminal = scan_dataset(cfg["data"]["pm_terminal"])
    pm_repricing = scan_dataset(cfg["data"]["pm_repricing"])

    write_basic_eda("BTC Direction", btc, out_dir / "btc_direction_eda.md", ["label_1s", "label_5s", "label_30s"])
    write_pm_terminal_eda(pm_terminal, out_dir, PROJECT_ROOT / cfg["data"]["pm_terminal"])
    write_basic_eda("PM Repricing", pm_repricing, out_dir / "pm_repricing_eda.md", ["label_reprice_1s", "label_reprice_5s", "label_reprice_30s"])
    write_pm_repricing_label_distribution(pm_repricing, out_dir)
    print(f"Wrote EDA reports to {out_dir}")


if __name__ == "__main__":
    main()
