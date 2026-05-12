"""Check historical shadow replay parity against offline predictions and silver quotes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml


PRED_PATH = "reports/stage1/pm_repricing_test_predictions.parquet"
WINDOWS_PATH = "data/shadow/replay_windows/repricing_replay_windows.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/shadow_repricing.yaml")
    p.add_argument("--predictions-path", default=PRED_PATH)
    p.add_argument("--windows-path", default=WINDOWS_PATH)
    return p.parse_args()


def read_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_latest_diag_runs(base_dir: Path) -> tuple[str | None, str | None]:
    files = sorted((base_dir / "run_diagnostics").glob("date=*/run_diagnostics-*.json"))
    non_forced = None
    forced = None
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if payload.get("mode") != "replay":
            continue
        if payload.get("force_signals_from_windows"):
            forced = payload["run_id"]
        else:
            non_forced = payload["run_id"]
    return non_forced, forced


def load_parquet(base_dir: Path, subdir: str) -> pl.DataFrame:
    files = list((base_dir / subdir).glob("date=*/*.parquet"))
    if not files:
        return pl.DataFrame()
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat(frames, how="diagonal_relaxed")


def load_windows(path: str) -> pl.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("windows", [])
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        pl.col("center_sample_ts").str.to_datetime(time_zone="UTC"),
        pl.col("start_ts").str.to_datetime(time_zone="UTC"),
        pl.col("end_ts").str.to_datetime(time_zone="UTC"),
    )


def add_expected_outcomes(df: pl.DataFrame, silver: pl.DataFrame) -> pl.DataFrame:
    pending = df.select(["run_id", "signal_id", "market_id", "direction", "signal_ts", "entry_latency_ms", "exit_horizon_seconds"])
    pending = pending.with_columns(
        (pl.col("signal_ts") + pl.duration(milliseconds=pl.col("entry_latency_ms"))).alias("entry_due_ts"),
        (pl.col("signal_ts") + pl.duration(milliseconds=pl.col("entry_latency_ms")) + pl.duration(seconds=pl.col("exit_horizon_seconds"))).alias("exit_due_ts"),
    ).sort(["market_id", "entry_due_ts"])
    silver_entry = silver.select(["market_id", pl.col("sample_ts").alias("entry_quote_ts"), pl.col("yes_ask").alias("entry_yes_ask"), pl.col("no_ask").alias("entry_no_ask")]).sort(["market_id", "entry_quote_ts"])
    silver_exit = silver.select(["market_id", pl.col("sample_ts").alias("exit_quote_ts"), pl.col("yes_bid").alias("exit_yes_bid"), pl.col("no_bid").alias("exit_no_bid")]).sort(["market_id", "exit_quote_ts"])
    enriched = pending.join_asof(silver_entry, left_on="entry_due_ts", right_on="entry_quote_ts", by="market_id", strategy="forward").sort(["market_id", "exit_due_ts"]).join_asof(silver_exit, left_on="exit_due_ts", right_on="exit_quote_ts", by="market_id", strategy="forward")
    return df.join(
        enriched.select(["run_id", "signal_id", "entry_latency_ms", "exit_horizon_seconds", "entry_quote_ts", "entry_yes_ask", "entry_no_ask", "exit_quote_ts", "exit_yes_bid", "exit_no_bid"]),
        on=["run_id", "signal_id", "entry_latency_ms", "exit_horizon_seconds"],
        how="left",
    ).with_columns(
        pl.when(pl.col("direction") == "UP").then(pl.col("entry_yes_ask")).otherwise(pl.col("entry_no_ask")).alias("expected_entry_price"),
        pl.when(pl.col("direction") == "UP").then(pl.col("exit_yes_bid")).otherwise(pl.col("exit_no_bid")).alias("expected_exit_price"),
    ).with_columns(
        (pl.col("expected_exit_price") - pl.col("expected_entry_price")).alias("expected_pnl")
    )


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    base_dir = Path(cfg["output"]["base_dir"])
    signals = load_parquet(base_dir, "repricing_signals")
    outcomes = load_parquet(base_dir, "repricing_outcomes")
    windows = load_windows(args.windows_path)
    predictions = pl.read_parquet(args.predictions_path).select(["market_id", "sample_ts", "p_up_5s", "p_down_5s"])
    silver = pl.scan_parquet(str(Path(cfg["market_meta"]["silver_pm_path"]) / "**" / "*.parquet"), hive_partitioning=True, extra_columns="ignore").select(["market_id", "sample_ts", "yes_bid", "yes_ask", "no_bid", "no_ask"]).collect().sort(["market_id", "sample_ts"])
    non_forced_run, forced_run = load_latest_diag_runs(base_dir)

    matched_signal_parts: list[pl.DataFrame] = []
    for label, run_id in [("non_forced", non_forced_run), ("forced", forced_run)]:
        if run_id is None or signals.is_empty():
            continue
        part = signals.filter(pl.col("run_id") == run_id)
        if part.is_empty():
            continue
        if "window_id" in part.columns:
            joined = windows.join(
                part.select(
                    [
                        "run_id",
                        "window_id",
                        "market_id",
                        "sample_ts",
                        "direction",
                        "config_name",
                        "p_up",
                        "p_down",
                        "is_forced_signal",
                        "signal_id",
                    ]
                ),
                on=["window_id", "market_id"],
                how="left",
            )
        else:
            joined = windows.join(
                part.select(["run_id", "market_id", "sample_ts", "direction", "config_name", "p_up", "p_down", "signal_id"]).with_columns(pl.lit(False).alias("is_forced_signal")),
                on=["market_id"],
                how="left",
            )
        joined = joined.with_columns(
            (pl.col("sample_ts") - pl.col("center_sample_ts")).dt.total_milliseconds().abs().alias("abs_ms_from_center")
        ).sort(["window_id", "abs_ms_from_center"]).group_by("window_id").first().with_columns(
            (pl.col("p_up") - pl.col("offline_p_up_5s")).abs().alias("abs_diff_p_up"),
            (pl.col("p_down") - pl.col("offline_p_down_5s")).abs().alias("abs_diff_p_down"),
            pl.col("sample_ts").is_not_null().alias("signal_found"),
            pl.lit(label).alias("parity_mode"),
        )
        matched_signal_parts.append(joined)
    signal_parity = pl.concat(matched_signal_parts, how="diagonal_relaxed") if matched_signal_parts else pl.DataFrame()

    outcome_rows: list[pl.DataFrame] = []
    for label, run_id in [("non_forced", non_forced_run), ("forced", forced_run)]:
        if run_id is None or outcomes.is_empty():
            continue
        part = outcomes.filter(pl.col("run_id") == run_id)
        if part.is_empty():
            continue
        chosen_signals = signal_parity.filter((pl.col("parity_mode") == label) & pl.col("signal_found")).select(["run_id", "signal_id", "window_id"])
        if chosen_signals.is_empty():
            continue
        part = part.join(chosen_signals, on=["run_id", "signal_id"], how="inner")
        enriched = add_expected_outcomes(part, silver).with_columns(
            (pl.col("expected_entry_price") - pl.col("entry_price")).abs().alias("abs_diff_entry_price"),
            (pl.col("expected_exit_price") - pl.col("exit_price")).abs().alias("abs_diff_exit_price"),
            (pl.col("expected_pnl") - pl.col("pnl")).abs().alias("abs_diff_pnl"),
            pl.lit(label).alias("parity_mode"),
        )
        outcome_rows.append(enriched)
    outcome_parity = pl.concat(outcome_rows, how="diagonal_relaxed") if outcome_rows else pl.DataFrame()

    out_dir = Path("reports/shadow")
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_csv = out_dir / "repricing_replay_signal_parity.csv"
    outcome_csv = out_dir / "repricing_replay_outcome_parity.csv"
    if not signal_parity.is_empty():
        signal_parity.write_csv(signal_csv)
    else:
        signal_csv.write_text("", encoding="utf-8")
    if not outcome_parity.is_empty():
        outcome_parity.write_csv(outcome_csv)
    else:
        outcome_csv.write_text("", encoding="utf-8")

    lines = ["# Repricing Replay Parity Report\n\n"]
    lines.append(f"- latest_non_forced_run: `{non_forced_run}`\n")
    lines.append(f"- latest_forced_run: `{forced_run}`\n\n")
    if not signal_parity.is_empty():
        for mode in signal_parity["parity_mode"].unique().to_list():
            part = signal_parity.filter(pl.col("parity_mode") == mode)
            lines.append(f"## Signal parity: {mode}\n\n")
            lines.append(f"- rows: `{part.height}`\n")
            lines.append(f"- signal_found_rate: `{part['signal_found'].mean()}`\n")
            lines.append(f"- abs_diff_p_up_p95: `{part['abs_diff_p_up'].drop_nulls().quantile(0.95) if part['abs_diff_p_up'].drop_nulls().len() else None}`\n")
            lines.append(f"- abs_diff_p_down_p95: `{part['abs_diff_p_down'].drop_nulls().quantile(0.95) if part['abs_diff_p_down'].drop_nulls().len() else None}`\n\n")
    if not outcome_parity.is_empty():
        for mode in outcome_parity["parity_mode"].unique().to_list():
            part = outcome_parity.filter(pl.col("parity_mode") == mode)
            lines.append(f"## Outcome parity: {mode}\n\n")
            lines.append(f"- rows: `{part.height}`\n")
            lines.append(f"- abs_diff_entry_price_p95: `{part['abs_diff_entry_price'].drop_nulls().quantile(0.95) if part['abs_diff_entry_price'].drop_nulls().len() else None}`\n")
            lines.append(f"- abs_diff_exit_price_p95: `{part['abs_diff_exit_price'].drop_nulls().quantile(0.95) if part['abs_diff_exit_price'].drop_nulls().len() else None}`\n")
            lines.append(f"- abs_diff_pnl_p95: `{part['abs_diff_pnl'].drop_nulls().quantile(0.95) if part['abs_diff_pnl'].drop_nulls().len() else None}`\n\n")
    (out_dir / "repricing_replay_parity_report.md").write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {signal_csv}")
    print(f"Wrote {outcome_csv}")
    print(f"Wrote {out_dir / 'repricing_replay_parity_report.md'}")


if __name__ == "__main__":
    main()
