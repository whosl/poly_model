from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import polars as pl
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 PM terminal workflow.")
    parser.add_argument("--config", default="configs/model_stage1.yaml")
    parser.add_argument("--only", choices=["eda", "train", "trading_eval"], default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def run_script(script: str, config: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", str(PROJECT_ROOT / config)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def fmt_metric(metrics: dict[str, Any], model: str) -> str:
    m = metrics["test"][model]
    return f"AUC={m['auc']:.6f}, logloss={m['logloss']:.6f}, Brier={m['brier']:.6f}, acc@0.5={m['accuracy_0p5']:.6f}"


def is_monotonic(vals: list[float], tol: float = 0.02) -> bool:
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 3:
        return False
    return all(b + tol >= a for a, b in zip(vals, vals[1:]))


def read_csv_or_empty(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    try:
        return pl.read_csv(path)
    except Exception:
        return pl.DataFrame()


def generate_summary(config_path: str) -> None:
    cfg = load_config(config_path)
    stage_dir = PROJECT_ROOT / cfg["reports"]["stage1_dir"]
    eda_dir = PROJECT_ROOT / cfg["reports"]["eda_dir"]
    metrics_path = stage_dir / "pm_terminal_metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    thresholds = read_csv_or_empty(stage_dir / "pm_terminal_trading_thresholds.csv")
    by_tte = read_csv_or_empty(stage_dir / "pm_terminal_trading_by_tte.csv")
    by_date = read_csv_or_empty(stage_dir / "pm_terminal_trading_by_date.csv")

    formula = metrics["test"]["formula_p_yes"]
    lgbm = metrics["test"]["lightgbm"]
    auc_delta = (lgbm["auc"] or 0) - (formula["auc"] or 0)
    logloss_delta = formula["logloss"] - lgbm["logloss"]
    brier_delta = formula["brier"] - lgbm["brier"]

    edge_mono = None
    edge_csv = eda_dir / "pm_terminal_edge_buckets.csv"
    if edge_csv.exists():
        edge_df = pl.read_csv(edge_csv)
        yes_col = "yes_win_rate" if "yes_win_rate" in edge_df.columns else "outcome_rate"
        no_col = "no_win_rate" if "no_win_rate" in edge_df.columns else "outcome_rate"
        yes = edge_df.filter(pl.col("edge_field") == "edge_to_yes_ask").get_column(yes_col).drop_nulls().to_list() if "edge_field" in edge_df.columns and yes_col in edge_df.columns else []
        no = edge_df.filter(pl.col("edge_field") == "edge_to_no_ask").get_column(no_col).drop_nulls().to_list() if "edge_field" in edge_df.columns and no_col in edge_df.columns else []
        edge_mono = {"edge_to_yes_ask": is_monotonic(yes), "edge_to_no_ask": is_monotonic(no)}

    lines = ["# PM Terminal Stage 1 Summary", ""]
    lines.append("## 1. Formula baseline metrics")
    lines.append(f"- {fmt_metric(metrics, 'formula_p_yes')}")
    lines.append("")
    lines.append("## 2. Logistic Regression metrics")
    lines.append(f"- {fmt_metric(metrics, 'logistic_regression')}")
    lines.append("")
    lines.append("## 3. LightGBM metrics")
    lines.append(f"- {fmt_metric(metrics, 'lightgbm')}")
    lines.append("")
    lines.append("## 4. LightGBM vs formula_p_yes")
    lines.append(f"- AUC delta: `{auc_delta:.6f}`")
    lines.append(f"- logloss improvement: `{logloss_delta:.6f}`")
    lines.append(f"- Brier improvement: `{brier_delta:.6f}`")
    lines.append(f"- conclusion: `{'improved' if auc_delta > 0 and logloss_delta > 0 and brier_delta > 0 else 'mixed'}`")
    lines.append("")
    lines.append("## 5. Edge bucket monotonicity")
    lines.append(f"- edge_to_yes_ask monotonic: `{None if edge_mono is None else edge_mono['edge_to_yes_ask']}`")
    lines.append(f"- edge_to_no_ask monotonic: `{None if edge_mono is None else edge_mono['edge_to_no_ask']}`")
    lines.append("")
    lines.append("## 6. Trading eval positive EV?")
    if thresholds.is_empty():
        lines.append("- No trades fired at configured thresholds `[0.01, 0.02, 0.03, 0.05, 0.08, 0.10]`; EV is not estimable from these rules.")
    else:
        first = thresholds.filter(pl.col("mode") == "first_signal_per_market_side")
        positive = first.filter(pl.col("avg_pnl") > 0) if not first.is_empty() and "avg_pnl" in first.columns else pl.DataFrame()
        lines.append(f"- positive_EV_first_signal_rows: `{positive.height}`")
        lines.append(f"- conclusion: `{'positive in at least one bucket' if positive.height else 'not positive at tested thresholds'}`")
    lines.append("")
    lines.append("## 7. all_signals vs first_signal_per_market_side")
    if thresholds.is_empty():
        lines.append("- Both modes produced `0` trades at configured thresholds.")
    else:
        mode_summary = thresholds.group_by("mode").agg(pl.col("trades").sum().alias("trades"), pl.col("total_pnl").sum().alias("total_pnl")).sort("mode")
        lines.extend(["| mode | trades | total_pnl |", "| --- | --- | --- |"])
        for row in mode_summary.rows():
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    lines.append("")
    lines.append("## 8. Steadiest threshold")
    if thresholds.is_empty():
        lines.append("- None: no trades fired.")
    else:
        first = thresholds.filter(pl.col("mode") == "first_signal_per_market_side")
        if first.is_empty():
            lines.append("- None: no first-signal trades fired.")
        else:
            best = first.sort(["sharpe_like", "avg_pnl"], descending=True).head(1).to_dicts()[0]
            lines.append(f"- threshold `{best.get('threshold')}`, side `{best.get('side')}`, sharpe_like `{best.get('sharpe_like')}`, avg_pnl `{best.get('avg_pnl')}`, trades `{best.get('trades')}`")
    lines.append("")
    lines.append("## 9. Best time_to_expiry interval")
    if by_tte.is_empty():
        lines.append("- None: no trades fired.")
    else:
        first_tte = by_tte.filter(pl.col("mode") == "first_signal_per_market_side")
        best = first_tte.sort(["avg_pnl", "trades"], descending=True).head(1).to_dicts()[0]
        lines.append(f"- bucket `{best.get('time_to_expiry_bucket')}`, threshold `{best.get('threshold')}`, side `{best.get('side')}`, avg_pnl `{best.get('avg_pnl')}`, trades `{best.get('trades')}`")
    lines.append("")
    lines.append("## 10. Profitable / losing dates")
    if by_date.is_empty():
        lines.append("- None: no trades fired.")
    else:
        first_date = by_date.filter(pl.col("mode") == "first_signal_per_market_side")
        date_summary = first_date.group_by("date").agg(pl.col("total_pnl").sum().alias("total_pnl"), pl.col("trades").sum().alias("trades")).sort("date")
        profitable = date_summary.filter(pl.col("total_pnl") > 0).get_column("date").to_list()
        losing = date_summary.filter(pl.col("total_pnl") < 0).get_column("date").to_list()
        lines.append(f"- profitable_dates: `{profitable}`")
        lines.append(f"- losing_dates: `{losing}`")
    lines.append("")
    lines.append("## 11. Continue to pm_repricing and btc_direction?")
    if auc_delta > 0 and logloss_delta > 0:
        lines.append("- Model quality improves over formula baseline, so continuing research is reasonable.")
    else:
        lines.append("- Model quality does not clearly improve; continue only after reviewing diagnostics.")
    if thresholds.is_empty():
        lines.append("- However, current PM terminal trading thresholds produce no test trades, so lower thresholds or maker/queue assumptions may be needed before relying on trading EV.")
    lines.append("- Recommendation: review PM terminal EDA/training first as requested before starting pm_repricing or btc_direction.")

    (stage_dir / "pm_terminal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.only is None:
        for script in ["run_stage1_eda.py", "train_pm_terminal.py", "evaluate_pm_terminal_trading.py"]:
            run_script(script, args.config)
    elif args.only == "eda":
        run_script("run_stage1_eda.py", args.config)
    elif args.only == "train":
        run_script("train_pm_terminal.py", args.config)
    elif args.only == "trading_eval":
        run_script("evaluate_pm_terminal_trading.py", args.config)
    generate_summary(args.config)
    print("Stage 1 PM terminal workflow complete.")


if __name__ == "__main__":
    main()
