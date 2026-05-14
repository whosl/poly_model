#!/usr/bin/env python3
"""Paper/live executor harness for v5 terminal Polymarket signals.

Default mode is PAPER only: it tails v5 shadow signal parquet files, applies the
same practical live-order gates (freshness, one trade per market side, TTE,
6-share depth/notional/price limits), and records would-submit decisions. It
never imports or uses trading keys in paper mode.

The live order adapter is intentionally left external; the production path can
reuse the existing poly_bot CLOB order code after this paper executor shows the
live gates are behaving correctly.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ParquetBuffer:
    def __init__(self, root: str | Path, prefix: str) -> None:
        self.root = ensure_dir(root)
        self.prefix = prefix
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        df = pl.DataFrame(self.rows)
        date_val = str(df["date"][0]) if "date" in df.columns else utc_now().date().isoformat()
        out_dir = ensure_dir(self.root / f"date={date_val}")
        out_path = out_dir / f"{self.prefix}-{utc_now().strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        df.write_parquet(out_path)
        self.rows.clear()


@dataclass
class ExecutorConfig:
    signals_dir: Path
    output_dir: Path
    state_path: Path
    market_meta_path: Path
    model_version: str = "pm_terminal_v5_hold_to_expiry"
    mode: str = "paper"
    poll_seconds: float = 5.0
    lookback_hours: float = 8.0
    max_signal_age_seconds: float = 90.0
    ignore_existing_on_start: bool = True
    target_shares: float = 6.0
    limit_price_offset: float = 0.01
    max_limit_price: float = 0.95
    min_limit_price: float = 0.01
    min_order_notional: float = 1.00
    min_tte_seconds: float = 30.0
    max_tte_seconds: float = 900.0
    max_effective_ask: float = 0.90
    max_spread: float = 0.05
    one_trade_per_market_side: bool = True
    max_decisions_per_hour: int = 12
    allow_down: bool = True
    allow_up: bool = True


def load_config(path: str | Path) -> ExecutorConfig:
    raw = read_yaml(path)
    return ExecutorConfig(
        signals_dir=Path(raw.get("signals_dir", "/opt/pm-shadow/data/shadow/repricing_signals")),
        output_dir=Path(raw.get("output_dir", "/opt/pm-shadow/data/shadow/v5_executor_decisions")),
        state_path=Path(raw.get("state_path", "/opt/pm-shadow/state/v5_paper_executor_state.json")),
        market_meta_path=Path(raw.get("market_meta_path", "/opt/pm-shadow/repo/configs/shadow_market_meta.parquet")),
        model_version=str(raw.get("model_version", "pm_terminal_v5_hold_to_expiry")),
        mode=str(raw.get("mode", "paper")),
        poll_seconds=float(raw.get("poll_seconds", 5.0)),
        lookback_hours=float(raw.get("lookback_hours", 8.0)),
        max_signal_age_seconds=float(raw.get("max_signal_age_seconds", 90.0)),
        ignore_existing_on_start=bool(raw.get("ignore_existing_on_start", True)),
        target_shares=float(raw.get("target_shares", 6.0)),
        limit_price_offset=float(raw.get("limit_price_offset", 0.01)),
        max_limit_price=float(raw.get("max_limit_price", 0.95)),
        min_limit_price=float(raw.get("min_limit_price", 0.01)),
        min_order_notional=float(raw.get("min_order_notional", 1.0)),
        min_tte_seconds=float(raw.get("min_tte_seconds", 30.0)),
        max_tte_seconds=float(raw.get("max_tte_seconds", 900.0)),
        max_effective_ask=float(raw.get("max_effective_ask", 0.90)),
        max_spread=float(raw.get("max_spread", 0.05)),
        one_trade_per_market_side=bool(raw.get("one_trade_per_market_side", True)),
        max_decisions_per_hour=int(raw.get("max_decisions_per_hour", 12)),
        allow_down=bool(raw.get("allow_down", True)),
        allow_up=bool(raw.get("allow_up", True)),
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_signal_ids": [], "accepted_market_sides": [], "decision_ts": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"processed_signal_ids": [], "accepted_market_sides": [], "decision_ts": []}


def save_state(path: Path, state: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    # Keep bounded state.
    state["processed_signal_ids"] = list(dict.fromkeys(state.get("processed_signal_ids", [])))[-5000:]
    state["accepted_market_sides"] = list(dict.fromkeys(state.get("accepted_market_sides", [])))[-2000:]
    cutoff = utc_now() - timedelta(hours=24)
    kept = []
    for s in state.get("decision_ts", [])[-5000:]:
        try:
            ts = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            if ts >= cutoff:
                kept.append(ts.isoformat())
        except Exception:
            pass
    state["decision_ts"] = kept
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def signal_files(root: Path, lookback_hours: float) -> list[Path]:
    now = utc_now()
    dates = {(now.date()).isoformat(), ((now - timedelta(days=1)).date()).isoformat()}
    files: list[Path] = []
    for d in dates:
        part = root / f"date={d}"
        if part.exists():
            files.extend(part.glob("*.parquet"))
    cutoff = now.timestamp() - lookback_hours * 3600.0
    return sorted([p for p in files if p.stat().st_mtime >= cutoff], key=lambda p: p.stat().st_mtime)


def load_recent_signals(cfg: ExecutorConfig) -> pl.DataFrame:
    files = signal_files(cfg.signals_dir, cfg.lookback_hours)
    if not files:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(str(p)) for p in files], how="diagonal_relaxed")
    if "model_version" in df.columns:
        df = df.filter(pl.col("model_version") == cfg.model_version)
    if "signal_id" in df.columns:
        df = df.unique(subset=["signal_id"], keep="last")
    return df.sort("sample_ts") if "sample_ts" in df.columns else df


def load_market_meta(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pl.read_parquet(str(path))
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dicts():
        mid = str(row.get("market_id") or "")
        if mid:
            out[mid] = row
    return out


def to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and value != value:
            return None
        return float(value)
    except Exception:
        return None


def decide(row: dict[str, Any], cfg: ExecutorConfig, meta: dict[str, dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    signal_id = str(row.get("signal_id") or "")
    market_id = str(row.get("market_id") or "")
    direction = str(row.get("direction") or "").upper()
    sample_ts = to_dt(row.get("sample_ts"))
    tte = num(row.get("time_to_expiry_seconds"))
    if tte is None:
        market_end = to_dt((meta.get(market_id) or {}).get("market_end_ts"))
        if market_end is not None:
            tte = (market_end - now).total_seconds()
    side_key = f"{market_id}:{direction}"

    reasons: list[str] = []
    if cfg.mode != "paper":
        reasons.append("live_mode_not_enabled_in_this_harness")
    if not signal_id:
        reasons.append("missing_signal_id")
    if signal_id in set(state.get("processed_signal_ids", [])):
        reasons.append("duplicate_signal")
    if direction == "UP" and not cfg.allow_up:
        reasons.append("up_disabled")
    if direction == "DOWN" and not cfg.allow_down:
        reasons.append("down_disabled")
    if direction not in {"UP", "DOWN"}:
        reasons.append("bad_direction")
    if sample_ts is None:
        reasons.append("missing_sample_ts")
    else:
        age = (now - sample_ts).total_seconds()
        if age > cfg.max_signal_age_seconds:
            reasons.append("signal_too_old")
        if age < -5:
            reasons.append("signal_from_future")
    if tte is None:
        reasons.append("missing_tte")
    else:
        if tte < cfg.min_tte_seconds:
            reasons.append("tte_too_low")
        if tte > cfg.max_tte_seconds:
            reasons.append("tte_too_high")
    if cfg.one_trade_per_market_side and side_key in set(state.get("accepted_market_sides", [])):
        reasons.append("market_side_already_accepted")

    recent_decisions = []
    cutoff = now - timedelta(hours=1)
    for s in state.get("decision_ts", []):
        try:
            ts = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            if ts >= cutoff:
                recent_decisions.append(ts)
        except Exception:
            pass
    if len(recent_decisions) >= cfg.max_decisions_per_hour:
        reasons.append("hourly_decision_limit")

    yes_ask = num(row.get("yes_effective_ask_6sh")) or num(row.get("yes_ask"))
    no_ask = num(row.get("no_effective_ask_6sh")) or num(row.get("no_ask"))
    yes_fill = num(row.get("yes_effective_fill_6sh"))
    no_fill = num(row.get("no_effective_fill_6sh"))
    yes_spread = num(row.get("yes_spread"))
    no_spread = num(row.get("no_spread"))
    edge = num(row.get("p_up" if direction == "UP" else "p_down"))

    if direction == "UP":
        effective_ask = yes_ask
        effective_fill = yes_fill
        spread = yes_spread
        asset_id = str(row.get("yes_asset_id") or (meta.get(market_id) or {}).get("yes_asset_id") or "")
        outcome = "YES"
    else:
        effective_ask = no_ask
        effective_fill = no_fill
        spread = no_spread
        asset_id = str(row.get("no_asset_id") or (meta.get(market_id) or {}).get("no_asset_id") or "")
        outcome = "NO"

    if not asset_id:
        reasons.append("missing_asset_id")
    if effective_ask is None:
        reasons.append("missing_effective_ask")
    else:
        if effective_ask > cfg.max_effective_ask:
            reasons.append("ask_too_high")
        if effective_ask <= 0:
            reasons.append("bad_ask")
    if effective_fill is not None and effective_fill + 1e-9 < cfg.target_shares:
        reasons.append("insufficient_6share_depth")
    if spread is not None and spread > cfg.max_spread:
        reasons.append("spread_too_wide")

    limit_price = None
    notional = None
    if effective_ask is not None:
        limit_price = min(cfg.max_limit_price, max(cfg.min_limit_price, effective_ask + cfg.limit_price_offset))
        notional = limit_price * cfg.target_shares
        if notional < cfg.min_order_notional:
            reasons.append("notional_below_min")
    accepted = not reasons
    return {
        "decision_id": uuid.uuid4().hex,
        "date": now.date().isoformat(),
        "decision_ts": now,
        "mode": cfg.mode,
        "accepted": accepted,
        "reject_reasons": ",".join(reasons),
        "signal_id": signal_id,
        "market_id": market_id,
        "direction": direction,
        "outcome": outcome,
        "asset_id": asset_id,
        "signal_ts": sample_ts,
        "signal_age_seconds": None if sample_ts is None else (now - sample_ts).total_seconds(),
        "time_to_expiry_seconds": tte,
        "edge": edge,
        "effective_ask_6sh": effective_ask,
        "effective_fill_6sh": effective_fill,
        "limit_price": limit_price,
        "target_shares": cfg.target_shares,
        "estimated_notional": notional,
        "yes_ask": num(row.get("yes_ask")),
        "no_ask": num(row.get("no_ask")),
        "yes_effective_ask_6sh": num(row.get("yes_effective_ask_6sh")),
        "no_effective_ask_6sh": num(row.get("no_effective_ask_6sh")),
        "p_up": num(row.get("p_up")),
        "p_down": num(row.get("p_down")),
        "terminal_p_yes": num(row.get("terminal_p_yes")),
    }


def run_once(cfg: ExecutorConfig, sink: ParquetBuffer, state: dict[str, Any], initialize_only: bool = False) -> int:
    meta = load_market_meta(cfg.market_meta_path)
    df = load_recent_signals(cfg)
    if df.height == 0:
        return 0
    processed = set(state.get("processed_signal_ids", []))
    rows = [r for r in df.to_dicts() if str(r.get("signal_id") or "") not in processed]
    if initialize_only:
        for r in rows:
            sid = str(r.get("signal_id") or "")
            if sid:
                state.setdefault("processed_signal_ids", []).append(sid)
        return len(rows)
    count = 0
    for row in rows:
        decision = decide(row, cfg, meta, state)
        sink.append(decision)
        sid = decision.get("signal_id")
        if sid:
            state.setdefault("processed_signal_ids", []).append(sid)
        if decision["accepted"]:
            state.setdefault("accepted_market_sides", []).append(f"{decision['market_id']}:{decision['direction']}")
            state.setdefault("decision_ts", []).append(utc_now().isoformat())
        count += 1
    if count:
        sink.flush()
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.mode != "paper":
        raise SystemExit("This harness currently only supports mode=paper. Refusing to run live orders.")
    ensure_dir(cfg.output_dir)
    state = load_state(cfg.state_path)
    sink = ParquetBuffer(cfg.output_dir, "v5_executor_decisions")
    if cfg.ignore_existing_on_start and not state.get("initialized"):
        n = run_once(cfg, sink, state, initialize_only=True)
        state["initialized"] = True
        save_state(cfg.state_path, state)
        print(f"initialized_state_ignored_existing_signals={n}", flush=True)
    while True:
        n = run_once(cfg, sink, state, initialize_only=False)
        if n:
            save_state(cfg.state_path, state)
            print(f"processed_new_signals={n}", flush=True)
        else:
            save_state(cfg.state_path, state)
        if args.once:
            break
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
