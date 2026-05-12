"""Audit PM terminal label / market-price alignment.

This script is intentionally read-only.  It diagnoses whether poor market-price
baselines are caused by inverted YES/NO mapping, inverted labels, split-specific
effects, or timing/proxy-label mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for terminal alignment audit") from exc


PROB_SPECS = [
    ("yes_mid", "yes_mid"),
    ("inv_yes_mid", "1 - yes_mid"),
    ("yes_ask", "yes_ask"),
    ("inv_yes_ask", "1 - yes_ask"),
    ("yes_bid", "yes_bid"),
    ("inv_yes_bid", "1 - yes_bid"),
    ("normalized_market_mid", "yes_mid / (yes_mid + no_mid)"),
    ("inv_normalized_market_mid", "1 - normalized_market_mid"),
    ("formula_p_yes", "formula_p_yes"),
    ("inv_formula_p_yes", "1 - formula_p_yes"),
    ("lightgbm_p_model", "p_model"),
    ("inv_lightgbm_p_model", "1 - p_model"),
]

TTE_BUCKETS = [
    ("[240,300]", 240, 300),
    ("[180,240)", 180, 240),
    ("[120,180)", 120, 180),
    ("[60,120)", 60, 120),
    ("[30,60)", 30, 60),
    ("[10,30)", 10, 30),
    ("[0,10)", 0, 10),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", default="data/gold/pm_terminal_1s")
    p.add_argument("--predictions", default="reports/stage1/pm_terminal_test_predictions.parquet")
    p.add_argument("--mapping", default="configs/pm_asset_mapping.generated.yaml")
    p.add_argument("--silver-pm", default="data/silver/pm_1s")
    p.add_argument("--silver-binance", default="data/silver/binance_1s")
    p.add_argument("--pm-meta", default="data/bronze/pm_market_meta")
    p.add_argument("--out-dir", default="reports/audit")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def read_parquet_dataset(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    p = Path(path)
    if p.is_dir():
        lf = pl.scan_parquet(str(p / "**" / "*.parquet"), extra_columns="ignore")
    else:
        lf = pl.scan_parquet(str(p), extra_columns="ignore")
    if columns:
        available = set(lf.collect_schema().names())
        cols = [c for c in columns if c in available]
        lf = lf.select(cols)
    return lf.collect()


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def clip_prob(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr.astype(float), 1e-6, 1.0 - 1e-6)


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = clip_prob(p[mask])
    out: dict[str, Any] = {"rows": int(len(y))}
    if len(y) == 0:
        out.update({"auc": None, "logloss": None, "brier": None, "accuracy_0p5": None})
        return out
    if len(np.unique(y)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y, p))
    out.update(
        {
            "auc": auc,
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "accuracy_0p5": float(accuracy_score(y, p >= 0.5)),
        }
    )
    return out


def add_probability_columns(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    if {"yes_mid", "no_mid"}.issubset(df.columns):
        exprs.append(
            (pl.col("yes_mid") / (pl.col("yes_mid") + pl.col("no_mid")))
            .alias("normalized_market_mid")
        )
    for base in ["yes_mid", "yes_ask", "yes_bid", "normalized_market_mid", "formula_p_yes", "p_model"]:
        if base in df.columns:
            inv_name = {
                "yes_mid": "inv_yes_mid",
                "yes_ask": "inv_yes_ask",
                "yes_bid": "inv_yes_bid",
                "normalized_market_mid": "inv_normalized_market_mid",
                "formula_p_yes": "inv_formula_p_yes",
                "p_model": "inv_lightgbm_p_model",
            }[base]
            exprs.append((1.0 - pl.col(base)).alias(inv_name))
    if "p_model" in df.columns:
        exprs.append(pl.col("p_model").alias("lightgbm_p_model"))
    if exprs:
        df = df.with_columns(exprs)
    if "sample_ts" in df.columns:
        df = df.with_columns(pl.col("sample_ts").dt.date().cast(pl.Utf8).alias("date"))
    if "time_to_expiry_seconds" in df.columns:
        bucket_expr = pl.lit(None, dtype=pl.Utf8)
        for label, lo, hi in reversed(TTE_BUCKETS):
            if hi == 300:
                cond = (pl.col("time_to_expiry_seconds") >= lo) & (pl.col("time_to_expiry_seconds") <= hi)
            else:
                cond = (pl.col("time_to_expiry_seconds") >= lo) & (pl.col("time_to_expiry_seconds") < hi)
            bucket_expr = pl.when(cond).then(pl.lit(label)).otherwise(bucket_expr)
        df = df.with_columns(bucket_expr.alias("tte_bucket"))
    return df


def metric_table(df: pl.DataFrame, group_col: str, out_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if group_col not in df.columns:
        return rows
    groups = df.select(group_col).drop_nulls().unique().to_series().to_list()
    for g in groups:
        sub = df.filter(pl.col(group_col) == g)
        y = sub["settled_yes"].to_numpy()
        for prob_col, _desc in PROB_SPECS:
            if prob_col not in sub.columns:
                continue
            vals = sub[prob_col].to_numpy()
            m = compute_metrics(y, vals)
            rows.append({"group_type": group_col, "group": g, "probability": prob_col, **m})
    write_csv(out_path, rows)
    return rows


def decile_table(df: pl.DataFrame, out_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in df.select("split").drop_nulls().unique().to_series().to_list() if "split" in df.columns else ["all"]:
        sub_split = df.filter(pl.col("split") == split) if split != "all" else df
        for prob_col in ["yes_mid", "inv_yes_mid", "formula_p_yes"]:
            if prob_col not in sub_split.columns:
                continue
            arr = sub_split.select([prob_col, "settled_yes"]).drop_nulls().to_numpy()
            if arr.shape[0] == 0:
                continue
            p = arr[:, 0].astype(float)
            y = arr[:, 1].astype(float)
            order = np.argsort(p)
            n = len(order)
            for d in range(10):
                lo = int(d * n / 10)
                hi = int((d + 1) * n / 10)
                idx = order[lo:hi]
                if len(idx) == 0:
                    continue
                rows.append(
                    {
                        "split": split,
                        "probability": prob_col,
                        "decile": d + 1,
                        "rows": int(len(idx)),
                        "prob_min": float(np.nanmin(p[idx])),
                        "prob_max": float(np.nanmax(p[idx])),
                        "prob_mean": float(np.nanmean(p[idx])),
                        "settled_yes_rate": float(np.nanmean(y[idx])),
                    }
                )
    write_csv(out_path, rows)
    return rows


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


def load_mapping(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    found: dict[str, dict[str, Any]] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            mid = x.get("market_id") or x.get("condition_id") or x.get("conditionId")
            if mid:
                found[str(mid)] = x
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found


def maybe_json(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def infer_real_label_from_row(row: dict[str, Any], yes_asset_id: str | None, no_asset_id: str | None) -> int | None:
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in ["winning_asset_id", "winner_asset_id", "winning_token_id", "winner_token_id", "winningclobtokenid"]:
        if key in lower and lower[key] is not None:
            val = str(lower[key])
            if yes_asset_id and val == str(yes_asset_id):
                return 1
            if no_asset_id and val == str(no_asset_id):
                return 0
    for key in ["resolved_outcome", "winning_outcome", "winner", "result"]:
        if key in lower and lower[key] is not None:
            val = str(lower[key]).strip().lower()
            if val in {"yes", "true", "1", "up", "above"}:
                return 1
            if val in {"no", "false", "0", "down", "below"}:
                return 0
    # Polymarket metadata often stores outcomePrices aligned with clobTokenIds/outcomes.
    prices = maybe_json(lower.get("outcomeprices") or lower.get("outcome_prices"))
    token_ids = maybe_json(lower.get("clobtokenids") or lower.get("clob_token_ids"))
    outcomes = maybe_json(lower.get("outcomes"))
    if isinstance(prices, list) and isinstance(token_ids, list) and yes_asset_id:
        try:
            nums = [safe_float(x) for x in prices]
            if nums and max(x for x in nums if x is not None) >= 0.99:
                idx = int(np.nanargmax(np.array([np.nan if x is None else x for x in nums], dtype=float)))
                winning_token = str(token_ids[idx])
                if winning_token == str(yes_asset_id):
                    return 1
                if no_asset_id and winning_token == str(no_asset_id):
                    return 0
        except Exception:
            pass
    if isinstance(prices, list) and isinstance(outcomes, list):
        try:
            nums = [safe_float(x) for x in prices]
            idx = int(np.nanargmax(np.array([np.nan if x is None else x for x in nums], dtype=float)))
            outcome = str(outcomes[idx]).strip().lower()
            if outcome == "yes":
                return 1
            if outcome == "no":
                return 0
        except Exception:
            pass
    return None


def label_source_comparison(df: pl.DataFrame, mapping: dict[str, dict[str, Any]], meta_path: str | Path, out: Path) -> list[dict[str, Any]]:
    yes_expr = pl.first("yes_asset_id").alias("yes_asset_id") if "yes_asset_id" in df.columns else pl.lit(None).alias("yes_asset_id")
    no_expr = pl.first("no_asset_id").alias("no_asset_id") if "no_asset_id" in df.columns else pl.lit(None).alias("no_asset_id")
    markets = (
        df.group_by("market_id")
        .agg(
            [
                yes_expr,
                no_expr,
                pl.first("settled_yes").alias("settled_yes_proxy"),
            ]
        )
        .to_dicts()
    )
    by_market: dict[str, dict[str, Any]] = {}
    for m in markets:
        mid = str(m["market_id"])
        rec = dict(m)
        rec["settled_yes_real"] = infer_real_label_from_row(mapping.get(mid, {}), str(m.get("yes_asset_id")), str(m.get("no_asset_id")))
        by_market[mid] = rec

    p = Path(meta_path)
    if p.exists():
        try:
            meta = read_parquet_dataset(p)
            mid_col = next((c for c in ["market_id", "condition_id", "conditionId"] if c in meta.columns), None)
            if mid_col:
                meta = meta.unique(subset=[mid_col], keep="last")
                for r in meta.to_dicts():
                    mid = str(r.get(mid_col))
                    if mid not in by_market or by_market[mid].get("settled_yes_real") is not None:
                        continue
                    by_market[mid]["settled_yes_real"] = infer_real_label_from_row(
                        r,
                        str(by_market[mid].get("yes_asset_id")),
                        str(by_market[mid].get("no_asset_id")),
                    )
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    for mid, r in by_market.items():
        real = r.get("settled_yes_real")
        proxy = r.get("settled_yes_proxy")
        rows.append(
            {
                "market_id": mid,
                "yes_asset_id": r.get("yes_asset_id"),
                "no_asset_id": r.get("no_asset_id"),
                "settled_yes_proxy": proxy,
                "settled_yes_real": real,
                "proxy_real_match": None if real is None else int(int(proxy) == int(real)),
            }
        )
    write_csv(out, rows)
    return rows


def make_case_studies(df: pl.DataFrame, mapping: dict[str, dict[str, Any]], out: Path, seed: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    base = df.filter(pl.col("split") == "test") if "split" in df.columns else df
    market_summary = (
        base.sort("sample_ts")
        .group_by("market_id")
        .agg(
            [
                pl.first("settled_yes").alias("settled_yes"),
                pl.first("yes_asset_id").alias("yes_asset_id") if "yes_asset_id" in base.columns else pl.lit(None).alias("yes_asset_id"),
                pl.first("no_asset_id").alias("no_asset_id") if "no_asset_id" in base.columns else pl.lit(None).alias("no_asset_id"),
                pl.first("market_start_ts").alias("market_start_ts"),
                pl.first("market_end_ts").alias("market_end_ts"),
                pl.first("label_source").alias("label_source") if "label_source" in base.columns else pl.lit(None).alias("label_source"),
                pl.first("yes_mid").alias("first_yes_mid"),
                pl.last("yes_mid").alias("last_yes_mid"),
                pl.max("yes_mid").alias("max_yes_mid"),
                pl.min("yes_mid").alias("min_yes_mid"),
                pl.first("formula_p_yes").alias("first_formula_p_yes"),
                pl.last("formula_p_yes").alias("last_formula_p_yes"),
                pl.first("btc_open_price").alias("btc_open_price") if "btc_open_price" in base.columns else pl.lit(None).alias("btc_open_price"),
                pl.first("btc_close_price").alias("btc_close_price") if "btc_close_price" in base.columns else pl.lit(None).alias("btc_close_price"),
            ]
        )
    )
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for yval in [1, 0]:
        mids = market_summary.filter(pl.col("settled_yes") == yval)["market_id"].to_list()
        rng.shuffle(mids)
        chosen.extend([str(x) for x in mids[:10]])
    lines = ["# PM Terminal Alignment Case Studies\n"]
    sample_cols = [
        "sample_ts",
        "time_to_expiry_seconds",
        "btc_current_price",
        "formula_p_yes",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "no_bid",
        "no_ask",
        "no_mid",
    ]
    for mid in chosen:
        s = market_summary.filter(pl.col("market_id") == mid).to_dicts()[0]
        info = mapping.get(mid, {})
        question = info.get("question") or info.get("title") or info.get("slug") or ""
        lines.append(f"\n## market_id: `{mid}`\n")
        lines.append(f"- question: {question}\n")
        for k in [
            "yes_asset_id",
            "no_asset_id",
            "market_start_ts",
            "market_end_ts",
            "btc_open_price",
            "btc_close_price",
            "settled_yes",
            "label_source",
            "first_yes_mid",
            "last_yes_mid",
            "max_yes_mid",
            "min_yes_mid",
            "first_formula_p_yes",
            "last_formula_p_yes",
        ]:
            lines.append(f"- {k}: `{s.get(k)}`\n")
        mdf = base.filter(pl.col("market_id") == mid).sort("time_to_expiry_seconds")
        lines.append("\nLast 30s snapshots closest to requested tte:\n\n")
        cols = [c for c in sample_cols if c in mdf.columns]
        lines.append("| " + " | ".join(cols) + " |\n")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for target in [30, 25, 20, 15, 10, 5, 0]:
            sub = mdf.filter(pl.col("time_to_expiry_seconds") <= 31)
            if sub.height == 0:
                continue
            sub = sub.with_columns((pl.col("time_to_expiry_seconds") - target).abs().alias("_dist")).sort("_dist").head(1)
            r = sub.select(cols).to_dicts()[0]
            lines.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |\n")
    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_cols = [
        "market_id",
        "sample_ts",
        "split",
        "time_to_expiry_seconds",
        "settled_yes",
        "label_source",
        "yes_asset_id",
        "no_asset_id",
        "market_start_ts",
        "market_end_ts",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "no_bid",
        "no_ask",
        "no_mid",
        "formula_p_yes",
        "btc_open_price",
        "btc_close_price",
        "btc_current_price",
    ]
    df = read_parquet_dataset(args.gold, gold_cols)
    if "yes_asset_id" not in df.columns or "no_asset_id" not in df.columns:
        try:
            asset_map = (
                read_parquet_dataset(args.silver_pm, ["market_id", "yes_asset_id", "no_asset_id"])
                .drop_nulls(["market_id"])
                .unique(subset=["market_id"], keep="first")
            )
            join_cols = [c for c in ["market_id", "yes_asset_id", "no_asset_id"] if c in asset_map.columns]
            if {"market_id", "yes_asset_id", "no_asset_id"}.issubset(join_cols):
                df = df.join(asset_map.select(join_cols), on="market_id", how="left")
        except Exception:
            pass
    pred_path = Path(args.predictions)
    if pred_path.exists():
        pred = read_parquet_dataset(pred_path, ["market_id", "sample_ts", "p_model"])
        if "p_model" in pred.columns:
            df = df.join(pred, on=["market_id", "sample_ts"], how="left")
    df = add_probability_columns(df)
    mapping = load_mapping(args.mapping)

    by_split = metric_table(df, "split", out_dir / "pm_terminal_market_baseline_by_split.csv")
    by_date = metric_table(df, "date", out_dir / "pm_terminal_market_baseline_by_date.csv")
    by_tte = metric_table(df, "tte_bucket", out_dir / "pm_terminal_market_baseline_by_tte.csv")
    deciles = decile_table(df, out_dir / "pm_terminal_market_baseline_deciles.csv")

    inv_rows: list[dict[str, Any]] = []
    for rows, group_type in [(by_split, "split"), (by_tte, "tte_bucket")]:
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            by_key[(str(r["group"]), str(r["probability"]))] = r
        groups = sorted({k[0] for k in by_key})
        for g in groups:
            for a, b in [
                ("yes_mid", "inv_yes_mid"),
                ("formula_p_yes", "inv_formula_p_yes"),
                ("lightgbm_p_model", "inv_lightgbm_p_model"),
            ]:
                ra, rb = by_key.get((g, a)), by_key.get((g, b))
                if not ra or not rb:
                    continue
                auc_a, auc_b = ra.get("auc"), rb.get("auc")
                inv_rows.append(
                    {
                        "group_type": group_type,
                        "group": g,
                        "probability": a,
                        "auc": auc_a,
                        "inverted_probability": b,
                        "inverted_auc": auc_b,
                        "inverted_auc_minus_auc": None if auc_a is None or auc_b is None else float(auc_b) - float(auc_a),
                        "possible_inversion_issue": bool(auc_a is not None and auc_b is not None and float(auc_b) - float(auc_a) > 0.05),
                    }
                )
    write_csv(out_dir / "pm_terminal_inversion_check.csv", inv_rows)
    label_rows = label_source_comparison(df, mapping, args.pm_meta, out_dir / "pm_terminal_label_source_comparison.csv")
    make_case_studies(df, mapping, out_dir / "pm_terminal_case_studies.md", args.seed)

    def find_metric(rows: list[dict[str, Any]], group: str, prob: str) -> dict[str, Any] | None:
        for r in rows:
            if str(r.get("group")) == group and r.get("probability") == prob:
                return r
        return None

    test_yes = find_metric(by_split, "test", "yes_mid")
    test_inv = find_metric(by_split, "test", "inv_yes_mid")
    tte_last = find_metric(by_tte, "[0,10)", "yes_mid")
    formula_inv_test = find_metric(by_split, "test", "inv_formula_p_yes")
    real_known = [r for r in label_rows if r.get("settled_yes_real") is not None]
    real_mismatch = [r for r in real_known if r.get("proxy_real_match") == 0]

    report = [
        "# PM Terminal Label / Market Alignment Audit\n\n",
        "## Executive summary\n\n",
        f"- rows audited: `{df.height}`\n",
        f"- markets audited: `{df.select('market_id').n_unique()}`\n",
        f"- test yes_mid AUC: `{None if not test_yes else test_yes.get('auc')}`\n",
        f"- test 1-yes_mid AUC: `{None if not test_inv else test_inv.get('auc')}`\n",
        f"- [0,10) tte yes_mid AUC: `{None if not tte_last else tte_last.get('auc')}`\n",
        f"- test 1-formula_p_yes AUC: `{None if not formula_inv_test else formula_inv_test.get('auc')}`\n",
        f"- real resolved labels found: `{len(real_known)}` markets\n",
        f"- proxy vs real mismatches: `{len(real_mismatch)}` markets\n\n",
        "## Initial interpretation\n\n",
    ]
    if test_yes and test_inv and test_yes.get("auc") is not None and test_inv.get("auc") is not None:
        if float(test_inv["auc"]) - float(test_yes["auc"]) > 0.05:
            report.append("- `1 - yes_mid` materially beats `yes_mid`; this is consistent with possible inverted PM side mapping/label alignment or timing/proxy mismatch.\n")
        else:
            report.append("- `1 - yes_mid` does not materially beat `yes_mid`; no strong global inversion signal from test split alone.\n")
    if tte_last and tte_last.get("auc") is not None and float(tte_last["auc"]) < 0.5:
        report.append("- `yes_mid` remains below 0.5 AUC in the final [0,10) seconds bucket; this is a high-priority label/mapping/timing warning.\n")
    if real_known:
        mismatch_rate = len(real_mismatch) / len(real_known)
        report.append(f"- Real Polymarket resolved labels were inferred for `{len(real_known)}` markets; mismatch rate vs proxy is `{mismatch_rate:.4f}`.\n")
    else:
        report.append("- No robust real Polymarket resolved outcome field was found in mapping/meta; current settled_yes remains proxy-based.\n")
    report.extend(
        [
            "\n## Output files\n\n",
            "- `reports/audit/pm_terminal_market_baseline_by_split.csv`\n",
            "- `reports/audit/pm_terminal_market_baseline_by_tte.csv`\n",
            "- `reports/audit/pm_terminal_market_baseline_by_date.csv`\n",
            "- `reports/audit/pm_terminal_market_baseline_deciles.csv`\n",
            "- `reports/audit/pm_terminal_inversion_check.csv`\n",
            "- `reports/audit/pm_terminal_label_source_comparison.csv`\n",
            "- `reports/audit/pm_terminal_case_studies.md`\n",
        ]
    )
    (out_dir / "pm_terminal_alignment_report.md").write_text("".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
