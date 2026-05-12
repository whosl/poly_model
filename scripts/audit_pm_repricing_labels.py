"""Audit PM repricing labels, markouts, and high-confidence signal regimes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import polars as pl


TTE_BUCKETS = [(240, 300, "[240,300]"), (180, 240, "[180,240)"), (120, 180, "[120,180)"), (60, 120, "[60,120)"), (30, 60, "[30,60)"), (10, 30, "[10,30)"), (0, 10, "[0,10)")]
SPREAD_BUCKETS = [(None, 0.005, "<0.5c"), (0.005, 0.015, "0.5-1.5c"), (0.015, 0.03, "1.5-3c"), (0.03, 0.05, "3-5c"), (0.05, None, ">=5c")]
HORIZONS = ["1s", "5s", "30s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", default="data/gold/pm_repricing_1s")
    p.add_argument("--predictions", default="reports/stage1/pm_repricing_test_predictions.parquet")
    p.add_argument("--out-dir", default="reports/audit")
    return p.parse_args()


def read_parquet_dataset(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    p = Path(path)
    lf = pl.scan_parquet(str(p / "**" / "*.parquet") if p.is_dir() else str(p), extra_columns="ignore")
    if columns:
        available = set(lf.collect_schema().names())
        lf = lf.select([c for c in columns if c in available])
    return lf.collect()


def bucket_expr(col: str, buckets: list[tuple[float | None, float | None, str]]) -> pl.Expr:
    expr = pl.lit("unbucketed")
    for lo, hi, label in reversed(buckets):
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (pl.col(col) >= lo)
        if hi is not None:
            cond = cond & ((pl.col(col) <= hi) if label.startswith("[240") else (pl.col(col) < hi))
        expr = pl.when(cond).then(pl.lit(label)).otherwise(expr)
    return expr


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def dist_rows(df: pl.DataFrame, group_col: str) -> list[dict[str, Any]]:
    rows = []
    for h in HORIZONS:
        label = f"label_reprice_{h}"
        markout = f"markout_{h}"
        if label not in df.columns:
            continue
        aggs = [
            pl.len().alias("rows"),
            (pl.col(label) == "UP").mean().alias("up_rate"),
            (pl.col(label) == "DOWN").mean().alias("down_rate"),
            (pl.col(label) == "FLAT").mean().alias("flat_rate"),
        ]
        if markout in df.columns:
            aggs += [
                pl.col(markout).mean().alias("markout_mean"),
                pl.col(markout).median().alias("markout_median"),
                pl.col(markout).quantile(0.25).alias("markout_p25"),
                pl.col(markout).quantile(0.75).alias("markout_p75"),
            ]
        for r in df.group_by(group_col).agg(aggs).to_dicts():
            rows.append({"horizon": h, "group_type": group_col, **r})
    return rows


def high_conf_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    features = [
        "yes_spread",
        "pair_mid_sum",
        "time_to_expiry_seconds",
        "formula_p_yes_minus_yes_mid",
        "btc_return_1s",
        "btc_return_5s",
        "btc_return_30s",
    ]
    for direction, prob in [("UP", "p_up_5s"), ("DOWN", "p_down_5s")]:
        if prob not in df.columns:
            continue
        for thr in [0.6, 0.7, 0.75, 0.8]:
            part = df.filter(pl.col(prob) >= thr)
            row: dict[str, Any] = {"direction": direction, "threshold": thr, "rows": part.height}
            for c in features:
                if c in part.columns and part.height:
                    row[f"{c}_mean"] = float(part[c].mean())
                    row[f"{c}_p10"] = float(part[c].quantile(0.10))
                    row[f"{c}_p90"] = float(part[c].quantile(0.90))
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "market_id",
        "sample_ts",
        "split",
        "time_to_expiry_seconds",
        "yes_spread",
        "pair_mid_sum",
        "formula_p_yes_minus_yes_mid",
        "btc_return_1s",
        "btc_return_5s",
        "btc_return_30s",
    ]
    for h in HORIZONS:
        cols += [f"label_reprice_{h}", f"markout_{h}"]
    df = read_parquet_dataset(args.gold, cols)
    df = df.with_columns(
        [
            pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"),
            bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("tte_bucket"),
            bucket_expr("yes_spread", SPREAD_BUCKETS).alias("yes_spread_bucket"),
        ]
    )
    by_date = dist_rows(df, "date")
    by_tte = dist_rows(df, "tte_bucket")
    by_spread = dist_rows(df, "yes_spread_bucket")
    write_csv(out_dir / "pm_repricing_label_by_date.csv", by_date)
    write_csv(out_dir / "pm_repricing_label_by_tte.csv", by_tte)
    write_csv(out_dir / "pm_repricing_label_by_spread.csv", by_spread)

    high_conf = []
    pred_path = Path(args.predictions)
    if pred_path.exists():
        preds = read_parquet_dataset(pred_path)
        # Join extra feature columns from gold if missing in predictions.
        need = [c for c in ["market_id", "sample_ts", "formula_p_yes_minus_yes_mid", "btc_return_1s", "btc_return_5s", "btc_return_30s"] if c in df.columns]
        extra = df.select(need).unique(subset=["market_id", "sample_ts"]) if len(need) > 2 else pl.DataFrame()
        if extra.height:
            preds = preds.join(extra, on=["market_id", "sample_ts"], how="left")
        high_conf = high_conf_rows(preds)

    overall = dist_rows(df.with_columns(pl.lit("all").alias("all")), "all")
    lines = ["# PM Repricing Label Audit\n\n", "## Overall label / markout distribution\n\n"]
    lines.append("| horizon | rows | up_rate | down_rate | flat_rate | markout_mean | markout_median |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in overall:
        lines.append(f"| {r['horizon']} | {r['rows']} | {r['up_rate']} | {r['down_rate']} | {r['flat_rate']} | {r.get('markout_mean')} | {r.get('markout_median')} |\n")
    lines.append("\n## High-confidence signal feature regimes\n\n")
    lines.append("| direction | threshold | rows | yes_spread_mean | tte_mean | formula_edge_mean | btc_return_5s_mean |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in high_conf:
        lines.append(f"| {r['direction']} | {r['threshold']} | {r['rows']} | {r.get('yes_spread_mean')} | {r.get('time_to_expiry_seconds_mean')} | {r.get('formula_p_yes_minus_yes_mid_mean')} | {r.get('btc_return_5s_mean')} |\n")
    lines.extend(
        [
            "\n## Output files\n\n",
            "- `reports/audit/pm_repricing_label_by_date.csv`\n",
            "- `reports/audit/pm_repricing_label_by_tte.csv`\n",
            "- `reports/audit/pm_repricing_label_by_spread.csv`\n",
        ]
    )
    (out_dir / "pm_repricing_label_audit.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
