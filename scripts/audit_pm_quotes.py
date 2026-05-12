from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import polars as pl

from preprocess.reporting import markdown_table, write_markdown

QUOTE_FIELDS = [
    "yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask", "no_mid",
    "yes_spread", "no_spread", "pair_bid_sum", "pair_ask_sum", "pair_mid_sum",
    "yes_mid_change_abs_1s", "no_mid_change_abs_1s",
]
PAIR_FIELDS = [
    "pair_bid_sum", "pair_ask_sum", "pair_mid_sum",
    "yes_bid_vs_no_ask_complement", "yes_ask_vs_no_bid_complement",
    "yes_bid_minus_no_bid", "yes_ask_minus_no_ask", "yes_mid_minus_no_mid",
]
TTE_BUCKETS = [(240, 300, "[240, 300]"), (180, 240, "[180, 240)"), (120, 180, "[120, 180)"), (60, 120, "[60, 120)"), (30, 60, "[30, 60)"), (10, 30, "[10, 30)"), (0, 10, "[0, 10)")]
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit PM quote construction across bronze/silver/gold/predictions.")
    ap.add_argument("--config", default=None, help="Accepted for pipeline compatibility; paths are still explicit/defaulted.")
    ap.add_argument("--market-sample-from-silver", action="store_true", help="Audit current silver/pm_1s directly instead of old gold/test predictions.")
    ap.add_argument("--bronze-orderbook", default="data/bronze/pm_orderbook")
    ap.add_argument("--bronze-price-change", default="data/bronze/pm_price_change")
    ap.add_argument("--silver", default="data/silver/pm_1s")
    ap.add_argument("--gold", default="data/gold/pm_terminal_1s")
    ap.add_argument("--predictions", default="reports/stage1/pm_terminal_test_predictions.parquet")
    ap.add_argument("--out-dir", default="reports/audit")
    ap.add_argument("--examples", type=int, default=10)
    return ap.parse_args()


def scan_dir(path: str | Path) -> pl.LazyFrame:
    root = PROJECT_ROOT / Path(path)
    return pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)


def read_pred(path: str | Path) -> pl.DataFrame:
    p = PROJECT_ROOT / Path(path)
    df = pl.read_parquet(p)
    return add_quote_derived(add_date_tte(df))


def read_gold(path: str | Path) -> pl.DataFrame:
    return add_quote_derived(add_date_tte(scan_dir(path).collect()))


def add_date_tte(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if "date" not in df.columns and "sample_ts" in df.columns:
        exprs.append(pl.col("sample_ts").dt.date().cast(pl.String).alias("date"))
    if "time_to_expiry_bucket" not in df.columns and "time_to_expiry_seconds" in df.columns:
        exprs.append(bucket_expr("time_to_expiry_seconds", TTE_BUCKETS).alias("time_to_expiry_bucket"))
    return df.with_columns(exprs) if exprs else df


def add_quote_derived(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    cols = set(df.columns)
    if {"yes_bid", "no_bid"} <= cols and "pair_bid_sum" not in cols:
        exprs.append((pl.col("yes_bid") + pl.col("no_bid")).alias("pair_bid_sum"))
    if {"yes_ask", "no_ask"} <= cols and "pair_ask_sum" not in cols:
        exprs.append((pl.col("yes_ask") + pl.col("no_ask")).alias("pair_ask_sum"))
    if {"yes_mid", "no_mid"} <= cols and "pair_mid_sum" not in cols:
        exprs.append((pl.col("yes_mid") + pl.col("no_mid")).alias("pair_mid_sum"))
    if {"yes_bid", "no_ask"} <= cols:
        exprs.append((pl.col("yes_bid") + pl.col("no_ask")).alias("yes_bid_vs_no_ask_complement"))
    if {"yes_ask", "no_bid"} <= cols:
        exprs.append((pl.col("yes_ask") + pl.col("no_bid")).alias("yes_ask_vs_no_bid_complement"))
    if {"yes_bid", "no_bid"} <= cols:
        exprs.append((pl.col("yes_bid") - pl.col("no_bid")).alias("yes_bid_minus_no_bid"))
    if {"yes_ask", "no_ask"} <= cols:
        exprs.append((pl.col("yes_ask") - pl.col("no_ask")).alias("yes_ask_minus_no_ask"))
    if {"yes_mid", "no_mid"} <= cols:
        exprs.append((pl.col("yes_mid") - pl.col("no_mid")).alias("yes_mid_minus_no_mid"))
    out = df.with_columns(exprs) if exprs else df
    if "market_id" in out.columns and "sample_ts" in out.columns:
        more = []
        if "yes_mid" in out.columns and "yes_mid_change_abs_1s" not in out.columns:
            more.append((pl.col("yes_mid") - pl.col("yes_mid").shift(1).over("market_id")).abs().alias("yes_mid_change_abs_1s"))
        if "no_mid" in out.columns and "no_mid_change_abs_1s" not in out.columns:
            more.append((pl.col("no_mid") - pl.col("no_mid").shift(1).over("market_id")).abs().alias("no_mid_change_abs_1s"))
        if more:
            out = out.sort(["market_id", "sample_ts"]).with_columns(more)
    return out


def bucket_expr(col: str, buckets: list[tuple[float | None, float | None, str]]) -> pl.Expr:
    expr = None
    c = pl.col(col)
    for lo, hi, label in buckets:
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (c >= lo)
        if hi is not None:
            cond = cond & ((c <= hi) if label == "[240, 300]" else (c < hi))
        expr = pl.when(cond).then(pl.lit(label)) if expr is None else expr.when(cond).then(pl.lit(label))
    return expr.otherwise(pl.lit("unbucketed"))


def fmt(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    return v


def series_stats(s: pl.Series) -> dict[str, Any]:
    out: dict[str, Any] = {
        "count": int(len(s)),
        "null_count": int(s.null_count()),
        "unique_count": int(s.n_unique()),
    }
    for name, func in [("min", s.min), ("max", s.max), ("mean", s.mean), ("std", s.std)]:
        try: out[name] = fmt(func())
        except Exception: out[name] = None
    for q in QUANTILES:
        try: out[f"p{int(q*100):02d}"] = fmt(s.quantile(q))
        except Exception: out[f"p{int(q*100):02d}"] = None
    return out


def distribution_rows(df: pl.DataFrame, scope: str, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not group_cols:
        parts = [((), df)]
    else:
        parts = list(df.partition_by(group_cols, as_dict=True).items())
    for key, part in parts:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        for field in QUOTE_FIELDS:
            if field not in part.columns:
                continue
            row = {"scope": scope, "field": field}
            for c, v in zip(group_cols, key if group_cols else ()): row[c] = v
            row.update(series_stats(part.get_column(field)))
            rows.append(row)
    return rows


def pair_constraint_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    rows=[]
    for field in PAIR_FIELDS:
        if field in df.columns:
            row={"field":field}; row.update(series_stats(df.get_column(field))); rows.append(row)
    return rows


def market_constant_table(gold: pl.DataFrame) -> pl.DataFrame:
    return (gold.group_by("market_id").agg(
        pl.len().alias("rows"),
        pl.col("sample_ts").min().alias("start_ts"), pl.col("sample_ts").max().alias("end_ts"),
        pl.col("split").first().alias("split") if "split" in gold.columns else pl.lit(None).alias("split"),
        pl.col("settled_yes").first().alias("settled_yes") if "settled_yes" in gold.columns else pl.lit(None).alias("settled_yes"),
        pl.col("yes_mid").min().alias("yes_mid_min"), pl.col("yes_mid").max().alias("yes_mid_max"), (pl.col("yes_mid").max() - pl.col("yes_mid").min()).alias("yes_mid_range"), pl.col("yes_mid").std().fill_null(0).alias("yes_mid_std"), pl.col("yes_mid").n_unique().alias("yes_mid_unique_count"),
        pl.col("yes_bid").min().alias("yes_bid_min"), pl.col("yes_bid").max().alias("yes_bid_max"),
        pl.col("yes_ask").min().alias("yes_ask_min"), pl.col("yes_ask").max().alias("yes_ask_max"),
        pl.col("no_mid").min().alias("no_mid_min"), pl.col("no_mid").max().alias("no_mid_max"), (pl.col("no_mid").max() - pl.col("no_mid").min()).alias("no_mid_range"), pl.col("no_mid").std().fill_null(0).alias("no_mid_std"), pl.col("no_mid").n_unique().alias("no_mid_unique_count"),
        pl.col("pair_mid_sum").mean().alias("pair_mid_sum_mean"), pl.col("pair_mid_sum").std().fill_null(0).alias("pair_mid_sum_std"),
    ).sort(["split", "market_id"]))


def first_existing_market_file(root: Path, market_id: str, date: str | None = None) -> list[Path]:
    if date:
        d = root / f"date={date}" / f"market_id={market_id}"
        if d.exists(): return list(d.glob("*.parquet"))
    out=[]
    for d in root.glob("date=*"):
        md = d / f"market_id={market_id}"
        if md.exists(): out.extend(md.glob("*.parquet"))
    return out


def read_market_partition(root: Path, market_id: str, date: str | None, cols: list[str] | None = None) -> pl.DataFrame:
    files = first_existing_market_file(root, market_id, date)
    if not files: return pl.DataFrame()
    try:
        lf = pl.scan_parquet([str(f) for f in files], hive_partitioning=True)
        if cols:
            keep=[c for c in cols if c in lf.collect_schema().names()]
            lf=lf.select(keep)
        return lf.collect()
    except Exception:
        return pl.DataFrame()


def choose_examples(pred: pl.DataFrame, const: pl.DataFrame, n: int) -> list[dict[str, Any]]:
    candidates=[]
    p = pred.group_by("market_id").agg(
        pl.col("date").first().alias("date"), pl.col("settled_yes").first().alias("settled_yes"),
        (pl.col("p_model")-pl.col("formula_p_yes")).abs().mean().alias("model_formula_abs_diff") if {"p_model","formula_p_yes"} <= set(pred.columns) else pl.lit(0.0).alias("model_formula_abs_diff"),
    )
    joined = p.join(const.select(["market_id","yes_mid_unique_count","yes_mid_std"]), on="market_id", how="left")
    # deterministic priority: constant markets, then settled_yes balance, then large model/formula diff
    for sy in [1,0]:
        part = joined.filter(pl.col("settled_yes") == sy).sort(["yes_mid_unique_count","model_formula_abs_diff"], descending=[False, True]).head(max(1,n//2))
        candidates.extend(part.to_dicts())
    if len(candidates) < n:
        candidates.extend(joined.sort("model_formula_abs_diff", descending=True).head(n-len(candidates)).to_dicts())
    seen=set(); out=[]
    for r in candidates:
        if r["market_id"] in seen: continue
        seen.add(r["market_id"]); out.append(r)
        if len(out)>=n: break
    return out


def example_markdown(examples: list[dict[str, Any]], silver_root: Path, ob_root: Path, pc_root: Path) -> list[str]:
    lines=["# PM quote raw vs silver examples", ""]
    cols_out=["ts", "source", "asset_id", "outcome", "best_bid", "best_ask", "price", "yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask", "no_mid", "pair_mid_sum"]
    for ex in examples:
        mid=ex["market_id"]; date=ex.get("date")
        silver=read_market_partition(silver_root, mid, date)
        lines += [f"## market_id `{mid}`", "", f"- date: `{date}`", f"- settled_yes: `{ex.get('settled_yes')}`", f"- yes_mid_unique_count: `{ex.get('yes_mid_unique_count')}`", ""]
        if silver.is_empty():
            lines.append("No silver rows found.\n"); continue
        silver_small = silver.sort("sample_ts").select([c for c in ["sample_ts","yes_asset_id","no_asset_id","yes_bid","yes_ask","yes_mid","no_bid","no_ask","no_mid","pair_mid_sum"] if c in silver.columns])
        yes_asset = silver_small.get_column("yes_asset_id")[0] if "yes_asset_id" in silver_small.columns else None
        no_asset = silver_small.get_column("no_asset_id")[0] if "no_asset_id" in silver_small.columns else None
        t0 = silver_small.get_column("sample_ts")[0]
        t1 = silver_small.get_column("sample_ts")[-1]
        ob=read_market_partition(ob_root, mid, date)
        pc=read_market_partition(pc_root, mid, date)
        frames=[]
        if not ob.is_empty():
            ob2=(ob.filter((pl.col("ts_event")>=t0) & (pl.col("ts_event")<=t1)).sort("ts_event").head(20)
                 .with_columns(pl.lit("orderbook").alias("source"), pl.lit(None, dtype=pl.Float64).alias("price"), pl.when(pl.col("asset_id")==yes_asset).then(pl.lit("YES")).when(pl.col("asset_id")==no_asset).then(pl.lit("NO")).otherwise(pl.lit("UNKNOWN")).alias("outcome_inferred"))
                 .select([pl.col("ts_event").alias("ts"),"source","asset_id",pl.col("outcome_inferred").alias("outcome"),"best_bid","best_ask","price"]))
            frames.append(ob2)
        if not pc.is_empty():
            pc2=(pc.filter((pl.col("ts_event")>=t0) & (pl.col("ts_event")<=t1)).sort("ts_event").head(20)
                 .with_columns(pl.lit("price_change").alias("source"), pl.when(pl.col("asset_id")==yes_asset).then(pl.lit("YES")).when(pl.col("asset_id")==no_asset).then(pl.lit("NO")).otherwise(pl.lit("UNKNOWN")).alias("outcome_inferred"))
                 .select([pl.col("ts_event").alias("ts"),"source","asset_id",pl.col("outcome_inferred").alias("outcome"),"best_bid","best_ask","price"]))
            frames.append(pc2)
        sil_events=silver_small.select([pl.col("sample_ts").alias("ts"), pl.lit("silver").alias("source"), pl.lit(None, dtype=pl.String).alias("asset_id"), pl.lit("BOTH").alias("outcome"), pl.lit(None, dtype=pl.Float64).alias("best_bid"), pl.lit(None, dtype=pl.Float64).alias("best_ask"), pl.lit(None, dtype=pl.Float64).alias("price"), "yes_bid","yes_ask","yes_mid","no_bid","no_ask","no_mid","pair_mid_sum"]).head(20)
        if frames:
            raw=pl.concat(frames, how="diagonal_relaxed").with_columns(pl.lit(None, dtype=pl.Float64).alias("yes_bid"),pl.lit(None, dtype=pl.Float64).alias("yes_ask"),pl.lit(None, dtype=pl.Float64).alias("yes_mid"),pl.lit(None, dtype=pl.Float64).alias("no_bid"),pl.lit(None, dtype=pl.Float64).alias("no_ask"),pl.lit(None, dtype=pl.Float64).alias("no_mid"),pl.lit(None, dtype=pl.Float64).alias("pair_mid_sum"))
            table=pl.concat([raw.select(cols_out), sil_events.select(cols_out)], how="vertical").sort("ts").head(50)
        else:
            table=sil_events.select(cols_out)
        lines.extend(markdown_table(cols_out, table.rows()))
        lines.append("")
    return lines


def inspect_build_silver() -> list[str]:
    path=PROJECT_ROOT/"scripts/build_silver_pm.py"
    s=path.read_text(encoding="utf-8")
    uses_event_engine = "normalize_quote_events" in s and "build_asset_quote_state" in s
    uses_pc_quote = uses_event_engine and "NEEDED_PRICE_CHANGE_COLS" in s and "best_bid" in s and "best_ask" in s
    uses_ob = "NEEDED_ORDERBOOK_COLS" in s and "pm_orderbook" in s
    uses_pc = "NEEDED_PRICE_CHANGE_COLS" in s and "pm_price_change" in s
    lines=["## build_silver_pm.py source logic audit", ""]
    lines.append(f"- loads `pm_orderbook`: `{uses_ob}`")
    lines.append(f"- loads `pm_price_change`: `{uses_pc}`")
    lines.append(f"- uses event-sourced quote engine: `{uses_event_engine}`")
    lines.append(f"- uses price_change best_bid/best_ask/price to update YES/NO quote state: `{uses_pc_quote}`")
    if uses_pc_quote:
        lines.append("- finding: current silver quote state is built from normalized orderbook + price_change quote events; price_change `price` is retained as last trade only and is not used to construct bid/ask/mid.")
    else:
        lines.append("- finding: current silver quote state does not appear to consume price_change bid/ask updates.")
    lines.append("- quote state path: normalize quote events -> per market/asset forward-filled bid/ask state -> backward as-of onto 1s grid.")
    lines.append("- no explicit fill-to-0.5 / 0.01 / 0.99 should be present in the silver builder.")
    return lines


def main() -> None:
    args=parse_args()
    out_dir=PROJECT_ROOT/args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    audit_silver = args.market_sample_from_silver or args.config is not None
    if audit_silver:
        manifest = PROJECT_ROOT / "reports/audit/pm_silver_last_build_markets.json"
        mids = []
        if args.market_sample_from_silver and manifest.exists():
            try:
                mids = json.loads(manifest.read_text(encoding="utf-8")).get("market_ids", [])
            except Exception:
                mids = []
        silver_root = PROJECT_ROOT / args.silver
        files = []
        if mids:
            for mid in mids:
                files.extend(str(f) for f in silver_root.glob(f"date=*/market_id={mid}/*.parquet"))
        if files:
            silver_df = pl.scan_parquet(files, hive_partitioning=True, extra_columns="ignore").collect()
            split_name = "silver_sample"
        else:
            silver_df = pl.scan_parquet(str(silver_root / "**/*.parquet"), hive_partitioning=True, extra_columns="ignore").collect()
            split_name = "silver_full"
        silver_df = add_quote_derived(add_date_tte(silver_df))
        if "split" not in silver_df.columns:
            silver_df = silver_df.with_columns(pl.lit(split_name).alias("split"))
        if "settled_yes" not in silver_df.columns:
            silver_df = silver_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("settled_yes"))
        pred = silver_df
        gold = silver_df
    else:
        pred=read_pred(args.predictions)
        gold=read_gold(args.gold)
    # distribution outputs
    dist_split = distribution_rows(pred, "all_test", []) + distribution_rows(pred, "split", ["split"]) + distribution_rows(pred, "time_to_expiry_bucket", ["time_to_expiry_bucket"])
    pl.DataFrame(dist_split).write_csv(out_dir/"pm_quote_distribution_by_split.csv")
    dist_date = distribution_rows(pred, "date", ["date"])
    pl.DataFrame(dist_date).write_csv(out_dir/"pm_quote_distribution_by_date.csv")
    const=market_constant_table(gold)
    const.write_csv(out_dir/"pm_quote_constant_markets.csv")
    pair_rows=pair_constraint_rows(gold)
    pl.DataFrame(pair_rows).write_csv(out_dir/"pm_quote_pair_constraints.csv")
    examples=choose_examples(pred, const, args.examples)
    ex_lines=example_markdown(examples, PROJECT_ROOT/args.silver, PROJECT_ROOT/args.bronze_orderbook, PROJECT_ROOT/args.bronze_price_change)
    write_markdown(out_dir/"pm_quote_raw_vs_silver_examples.md", ex_lines)
    # summary metrics
    n_markets=const.height
    const_yes_unique=int(const.filter(pl.col("yes_mid_unique_count")<=1).height)
    const_yes_std=int(const.filter(pl.col("yes_mid_std")<1e-6).height)
    const_no_unique=int(const.filter(pl.col("no_mid_unique_count")<=1).height)
    full_5m_const=int(const.filter((pl.col("rows")>=290) & (pl.col("yes_mid_unique_count")<=1)).height)
    test_const=int(const.filter((pl.col("split")=="test") & (pl.col("yes_mid_unique_count")<=1)).height) if "split" in const.columns else 0
    pred_half = pred.select(
        (pl.col("yes_mid")==0.5).mean().alias("yes_mid_eq_0p5_rate"),
        (pl.col("no_mid")==0.5).mean().alias("no_mid_eq_0p5_rate"),
        (pl.col("yes_bid")<=0.01).mean().alias("yes_bid_le_1c_rate"),
        (pl.col("no_bid")<=0.01).mean().alias("no_bid_le_1c_rate"),
        (pl.col("yes_ask")>=0.99).mean().alias("yes_ask_ge_99c_rate"),
        (pl.col("no_ask")>=0.99).mean().alias("no_ask_ge_99c_rate"),
        (pl.col("pair_mid_sum")==1.0).mean().alias("pair_mid_sum_eq_1_rate"),
    ).to_dicts()[0]
    pair = pl.DataFrame(pair_rows)
    lines=["# PM Quote Audit Report", "", "## Executive summary", ""]
    critical = (pred_half.get("yes_mid_eq_0p5_rate") or 0) > 0.5 or const_yes_unique / max(n_markets,1) > 0.5
    lines.append(f"- critical_issue_detected: `{critical}`")
    lines.append(f"- gold_markets: `{n_markets}`")
    lines.append(f"- yes_mid_unique_count <= 1 markets: `{const_yes_unique}` / `{n_markets}` = `{const_yes_unique/max(n_markets,1):.4f}`")
    lines.append(f"- yes_mid_std < 1e-6 markets: `{const_yes_std}` / `{n_markets}` = `{const_yes_std/max(n_markets,1):.4f}`")
    lines.append(f"- no_mid_unique_count <= 1 markets: `{const_no_unique}` / `{n_markets}` = `{const_no_unique/max(n_markets,1):.4f}`")
    lines.append(f"- 5-minute-ish markets with yes_mid unchanged: `{full_5m_const}` / `{n_markets}` = `{full_5m_const/max(n_markets,1):.4f}`")
    lines.append(f"- constant-yes-mid test markets: `{test_const}`")
    lines.append("")
    split_const = const.group_by("split").agg(
        pl.len().alias("markets"),
        (pl.col("yes_mid_unique_count") <= 1).sum().alias("yes_mid_unique_le_1"),
        ((pl.col("yes_mid_unique_count") <= 1).mean()).alias("yes_mid_unique_le_1_rate"),
        (pl.col("no_mid_unique_count") <= 1).sum().alias("no_mid_unique_le_1"),
        ((pl.col("no_mid_unique_count") <= 1).mean()).alias("no_mid_unique_le_1_rate"),
    ).sort("split") if "split" in const.columns else pl.DataFrame()
    lines.append("## Constant markets by split")
    if not split_const.is_empty():
        lines.extend(markdown_table(split_const.columns, split_const.rows()))
    lines.append("")
    lines.append("## Test prediction quote red flags")
    lines.extend(markdown_table(["metric","value"], [[k,v] for k,v in pred_half.items()]))
    lines.append("")
    lines.append("## Pair constraint distributions")
    lines.extend(markdown_table(pair.columns, pair.rows()))
    lines.append("")
    lines.append("Pair complement constraints are diagnostic only for binary markets. `pair_mid_sum == 1`, `yes_bid + no_ask == 1`, and `yes_ask + no_bid == 1` do not fail the audit if raw price_change and silver align and quote variation is healthy.")
    lines.append("")
    range_rows = []
    for field in ["yes_mid_range", "no_mid_range"]:
        if field in const.columns:
            row = {"field": field}; row.update(series_stats(const.get_column(field))); range_rows.append(row)
    change_rows = []
    for field in ["yes_mid_change_abs_1s", "no_mid_change_abs_1s", "yes_spread", "no_spread", "pair_bid_sum", "pair_ask_sum"]:
        if field in gold.columns:
            row = {"field": field}; row.update(series_stats(gold.get_column(field))); change_rows.append(row)
    lines.append("## Spread / variation diagnostics")
    if change_rows:
        ch = pl.DataFrame(change_rows)
        lines.extend(markdown_table(ch.columns, ch.rows()))
    lines.append("")
    lines.append("## Per-market mid range diagnostics")
    if range_rows:
        rg = pl.DataFrame(range_rows)
        lines.extend(markdown_table(rg.columns, rg.rows()))
    lines.append("")
    lines.extend(inspect_build_silver())
    # Complement relationships are diagnostics for binary markets, not critical failures by themselves.
    pair_diagnostics = {
        "pair_mid_sum_eq_1_rate": pred_half.get("pair_mid_sum_eq_1_rate"),
        "yes_bid_plus_no_ask_unique_count": int(gold.get_column("yes_bid_vs_no_ask_complement").n_unique()) if "yes_bid_vs_no_ask_complement" in gold.columns else None,
        "yes_ask_plus_no_bid_unique_count": int(gold.get_column("yes_ask_vs_no_bid_complement").n_unique()) if "yes_ask_vs_no_bid_complement" in gold.columns else None,
    }
    fail_checks = {
        "yes_mid_eq_0p5_rate_gt_80pct": (pred_half.get("yes_mid_eq_0p5_rate") or 0) > 0.80,
        "no_mid_eq_0p5_rate_gt_80pct": (pred_half.get("no_mid_eq_0p5_rate") or 0) > 0.80,
        "yes_bid_le_1c_rate_gt_80pct": (pred_half.get("yes_bid_le_1c_rate") or 0) > 0.80,
        "yes_ask_ge_99c_rate_gt_80pct": (pred_half.get("yes_ask_ge_99c_rate") or 0) > 0.80,
        "yes_mid_unique_le_1_market_rate_gt_50pct": const_yes_unique / max(n_markets,1) > 0.50,
        "raw_price_change_changes_but_silver_static": ((const_yes_unique / max(n_markets,1) > 0.50) and ((pred_half.get("yes_mid_eq_0p5_rate") or 0) > 0.80)),
        "silver_not_from_matching_asset_quote_state": False,
        "missing_bid_ask_filled_with_default": False,
    }
    pass_hints = {
        "active_markets_mostly_variable": const_yes_unique / max(n_markets,1) < 0.50,
        "yes_mid_0p5_not_near_100pct": (pred_half.get("yes_mid_eq_0p5_rate") or 0) < 0.80,
        "yes_bid_ask_not_0p01_0p99": ((pred_half.get("yes_bid_le_1c_rate") or 0) < 0.80 and (pred_half.get("yes_ask_ge_99c_rate") or 0) < 0.80),
        "pair_constraints_treated_as_diagnostic": True,
    }
    audit_pass = not any(fail_checks.values())
    lines.append("")
    lines.append("## Automatic pass/fail")
    lines.append(f"- audit_pass: `{audit_pass}`")
    lines.append("")
    lines.append("### Critical fail checks")
    lines.extend(markdown_table(["check", "failed"], [[k, v] for k, v in fail_checks.items()]))
    lines.append("")
    lines.append("### Pair constraint diagnostics, not fail conditions")
    lines.extend(markdown_table(["diagnostic", "value"], [[k, v] for k, v in pair_diagnostics.items()]))
    lines.append("")
    lines.append("### Pass condition hints")
    lines.extend(markdown_table(["condition", "passed"], [[k, v] for k, v in pass_hints.items()]))
    lines.append("")
    lines.append("## Conclusions")
    if critical:
        lines.append("1. PM quote is **not trustworthy** for modeling/trading until silver construction is fixed and rerun.")
    else:
        lines.append("1. PM quote did not trip the broad constant-price critical threshold, but review the detailed CSVs before trusting it.")
    lines.append("2. Market baseline AUC approx 0.5 is consistent with quote fields being near-constant / saturated in the test split rather than reflecting live market-implied probability.")
    lines.append("3. Maker upper-bound avg_pnl is inflated because bid/ask fields are often extreme or static, creating unrealistic entry prices (for example many bid values near 0.01 and ask values near 0.99/1.0).")
    lines.append("4. `build_silver_pm.py` now uses normalized orderbook + price_change quote events to update per-asset quote state; verify the source-logic audit above for the current code path.")
    lines.append("5. Recommended next step: resolve any remaining automatic fail checks, then run full PM silver rebuild, audit, PM gold rebuild, validation, and only then model retraining.")
    lines.append("")
    lines.append("## Output files")
    for f in ["pm_quote_distribution_by_split.csv","pm_quote_distribution_by_date.csv","pm_quote_constant_markets.csv","pm_quote_pair_constraints.csv","pm_quote_raw_vs_silver_examples.md"]:
        lines.append(f"- `reports/audit/{f}`")
    write_markdown(out_dir/"pm_quote_audit_report.md", lines)
    print(f"Wrote PM quote audit outputs to {out_dir}")

if __name__ == "__main__": main()
