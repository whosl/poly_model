"""No-trade shadow logger for PM repricing.

Safety:
- never places orders
- aborts if enable_trading=true
- writes signals/outcomes/latency metrics to parquet only

Supports:
- generic live websocket mode
- historical replay-window mode
- forced-signal replay mode for outcome-tracker validation
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import time
import uuid
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import orjson
import polars as pl
import websockets
import yaml

try:
    from refresh_live_pm_market_meta import gamma_get_market_by_slug, market_to_row, slugs_for_btc_15m
except Exception:  # pragma: no cover - live refresh is best-effort.
    gamma_get_market_by_slug = None
    market_to_row = None
    slugs_for_btc_15m = None


LOGGER = logging.getLogger("pm_shadow")
ENTRY_LATENCIES_MS = [0, 250, 500, 1000, 2000]
EXIT_HORIZONS_S = [1, 5, 10, 30]
SHADOW_BANNER = "SHADOW MODE ONLY - NO ORDERS WILL BE PLACED"
UTC = timezone.utc
SENSITIVE_KEY_PATTERNS = (
    "private_key",
    "secret_key",
    "api_secret",
    "mnemonic",
    "seed_phrase",
    "wallet_key",
    "trading_key",
    "order_api",
    "order_placement",
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    warnings.filterwarnings("ignore", message="X does not have valid feature names")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/shadow_repricing.yaml")
    p.add_argument("--duration-minutes", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay-windows", default=None)
    p.add_argument("--replay-source-silver", default=None)
    p.add_argument("--replay-speed", type=float, default=0.0)
    p.add_argument("--force-signals-from-windows", action="store_true")
    return p.parse_args()


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _get_nested(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _collect_sensitive_keys(node: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if lowered in {"fail_if_order_api_configured", "disable_order_placement"}:
                hits.extend(_collect_sensitive_keys(value, path))
                continue
            if any(pattern in lowered for pattern in SENSITIVE_KEY_PATTERNS):
                hits.append(path)
            hits.extend(_collect_sensitive_keys(value, path))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            hits.extend(_collect_sensitive_keys(value, f"{prefix}[{idx}]"))
    return hits


def normalize_shadow_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw_cfg)
    model_section = cfg.get("model") or cfg.get("models") or {}
    output_section = cfg.get("output") or cfg.get("outputs") or {}
    feeds = cfg.get("feeds") or {}
    safety = cfg.get("safety") or {}
    signals = cfg.get("signals") or {}
    dry_run = cfg.get("dry_run") or {}

    model_path = model_section.get("model_path")
    features_path = model_section.get("features_path")
    if not model_path:
        model_dir = model_section.get("repricing_model_dir")
        if model_dir:
            candidate = Path(model_dir) / "lightgbm_5s.joblib"
            model_path = str(candidate)
    if not features_path:
        model_dir = model_section.get("repricing_model_dir")
        if model_dir:
            repo_root = Path(model_dir).parents[1] if len(Path(model_dir).parents) >= 2 else Path(".")
            candidate = repo_root / "models" / "pm_repricing" / "features_pm_repricing.json"
            features_path = str(candidate)

    signal_cfgs_raw = cfg.get("signal_configs")
    if signal_cfgs_raw is None:
        signal_cfgs_raw = signals.get("configs", [])
    normalized_signal_cfgs: list[dict[str, Any]] = []
    for item in signal_cfgs_raw:
        if not isinstance(item, dict):
            continue
        normalized_signal_cfgs.append(
            {
                "name": item.get("name"),
                "direction": item.get("direction", item.get("side", "BOTH")),
                "threshold_up": item.get("threshold_up", item.get("up_threshold")),
                "threshold_down": item.get("threshold_down", item.get("down_threshold")),
                "max_spread": item.get("max_spread"),
                "min_tte": item.get("min_tte", item.get("min_time_to_expiry_seconds", 0)),
                "cooldown_seconds": item.get("cooldown_seconds", 10),
            }
        )

    normalized = {
        **cfg,
        "mode": cfg.get("mode", "shadow_no_trade"),
        "enable_trading": bool(cfg.get("enable_trading", False)),
        "safety": {
            "disable_order_placement": bool(safety.get("disable_order_placement", True)),
            "fail_if_order_api_configured": bool(safety.get("fail_if_order_api_configured", True)),
        },
        "model": {
            "model_path": model_path,
            "features_path": features_path,
            "version": model_section.get("version", model_section.get("model_version", "unknown")),
            "type": model_section.get("type", "multiclass_repricing"),
            "up_model_path": model_section.get("up_model_path"),
            "down_model_path": model_section.get("down_model_path"),
            "fee_rate": float(model_section.get("fee_rate", 0.07)),
            "slippage_buffer": float(model_section.get("slippage_buffer", 0.0025)),
        },
        "market_meta": {
            "silver_pm_path": _get_nested(cfg, "market_meta", "silver_pm_path", default="data/silver/pm_1s"),
        },
        "output": {
            "base_dir": output_section.get("base_dir", "data/shadow"),
            "report_dir": output_section.get("report_dir", "reports/shadow"),
            "flush_interval_seconds": int(output_section.get("flush_interval_seconds", 10)),
        },
        "binance": {
            "symbol": _get_nested(cfg, "binance", "symbol", default=feeds.get("binance", {}).get("symbol", "BTCUSDT")),
            "ws_url": _get_nested(cfg, "binance", "ws_url", default=None)
            or (
                f"wss://stream.binance.com:9443/stream?streams="
                f"{str(feeds.get('binance', {}).get('symbol', 'BTCUSDT')).lower()}@bookTicker/"
                f"{str(feeds.get('binance', {}).get('symbol', 'BTCUSDT')).lower()}@aggTrade"
            ),
        },
        "polymarket": {
            "ws_url": _get_nested(cfg, "polymarket", "ws_url", default=None) or feeds.get("polymarket", {}).get("websocket_url", ""),
            "subscribe_messages": _get_nested(cfg, "polymarket", "subscribe_messages", default=[]),
            "quote_ttl_seconds": _get_nested(cfg, "polymarket", "quote_ttl_seconds", default=feeds.get("polymarket", {}).get("quote_ttl_seconds", 5.0)),
            "user_channel_enabled": bool(_get_nested(cfg, "polymarket", "user_channel_enabled", default=feeds.get("polymarket", {}).get("user_channel_enabled", False))),
        },
        "signal_configs": normalized_signal_cfgs,
        "dry_run": {
            "historical_gold_path": dry_run.get("historical_gold_path", "data/gold/pm_repricing_1s"),
            "max_rows": int(dry_run.get("max_rows", 5000)),
            "reset_cooldown_per_row": bool(dry_run.get("reset_cooldown_per_row", False)),
        },
    }
    return normalized


def validate_shadow_config(cfg: dict[str, Any]) -> None:
    if cfg.get("mode") != "shadow_no_trade":
        raise RuntimeError("shadow logger requires mode=shadow_no_trade")
    if cfg.get("enable_trading") is True:
        raise RuntimeError("enable_trading=true is forbidden in shadow logger")
    safety = cfg.get("safety") or {}
    if not safety.get("disable_order_placement", False):
        raise RuntimeError("disable_order_placement must be true")
    if not safety.get("fail_if_order_api_configured", False):
        raise RuntimeError("fail_if_order_api_configured must be true")
    if _get_nested(cfg, "polymarket", "user_channel_enabled", default=False):
        raise RuntimeError("polymarket user channel must stay disabled in shadow mode")
    sensitive_hits = _collect_sensitive_keys(cfg)
    if sensitive_hits and safety.get("fail_if_order_api_configured", False):
        raise RuntimeError(f"potential trading/order secret config detected: {sensitive_hits}")
    model = cfg.get("model") or {}
    if not model.get("features_path"):
        raise RuntimeError("shadow config missing features_path")
    if model.get("type") == "executable_net_binary":
        if not model.get("up_model_path") or not model.get("down_model_path"):
            raise RuntimeError("executable_net_binary requires up_model_path/down_model_path")
    elif not model.get("model_path"):
        raise RuntimeError("shadow config missing model_path")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scan_dataset(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    p = Path(path)
    if p.is_dir():
        glob = str(p / "**" / "*.parquet")
    else:
        glob = str(p)
    lf = pl.scan_parquet(glob, hive_partitioning=True, extra_columns="ignore")
    if columns:
        names = set(lf.collect_schema().names())
        lf = lf.select([c for c in columns if c in names])
    return lf.collect()


def _parse_levels(levels: Any) -> list[tuple[float, float]]:
    vals: list[tuple[float, float]] = []
    if not isinstance(levels, list):
        return vals
    for raw in levels:
        price = None
        size = None
        if isinstance(raw, dict):
            price = raw.get("price") or raw.get("p")
            size = raw.get("size") or raw.get("quantity") or raw.get("q")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 1:
            price = raw[0]
            size = raw[1] if len(raw) > 1 else None
        try:
            if price is None:
                continue
            vals.append((float(price), 0.0 if size is None else float(size)))
        except Exception:
            continue
    return vals


def best_levels(levels: Any, side: str) -> tuple[float | None, float | None]:
    vals = _parse_levels(levels)
    if not vals:
        return None, None
    vals.sort(key=lambda x: x[0], reverse=(side == "bid"))
    price, size = vals[0]
    return price, size


def depth_sum(levels: Any, side: str, n: int = 5) -> float | None:
    vals = _parse_levels(levels)
    if not vals:
        return None
    vals.sort(key=lambda x: x[0], reverse=(side == "bid"))
    return float(sum(size for _price, size in vals[:n]))


def to_utc_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        num = float(value)
        if num > 10_000_000_000:
            return datetime.fromtimestamp(num / 1000.0, tz=UTC)
        return datetime.fromtimestamp(num, tz=UTC)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


@dataclass
class ShadowSignalConfig:
    name: str
    direction: str
    threshold_up: float | None = None
    threshold_down: float | None = None
    max_spread: float | None = None
    min_tte: float = 0.0
    cooldown_seconds: float = 10.0


@dataclass
class MarketMeta:
    market_id: str
    yes_asset_id: str
    no_asset_id: str
    market_start_ts: datetime | None
    market_end_ts: datetime | None


@dataclass
class QuoteSnapshot:
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    bid_depth_5: float | None = None
    ask_depth_5: float | None = None
    ts_event: datetime | None = None
    ts_recv: datetime | None = None
    crossed: bool = False
    quote_age_seconds: float | None = None
    is_stale: bool = False

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass
class MarketState:
    meta: MarketMeta
    yes: QuoteSnapshot = field(default_factory=QuoteSnapshot)
    no: QuoteSnapshot = field(default_factory=QuoteSnapshot)
    pm_mid_hist: deque[tuple[datetime, float]] = field(default_factory=deque)
    last_pm_update_ts: datetime | None = None
    last_sample_ts: datetime | None = None


@dataclass
class BinanceState:
    bid: float | None = None
    ask: float | None = None
    ts_event: datetime | None = None
    ts_recv: datetime | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    mid_hist: deque[tuple[datetime, float]] = field(default_factory=deque)
    trade_hist: deque[tuple[datetime, float, float, bool]] = field(default_factory=deque)

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass
class ReplayWindow:
    window_id: str
    market_id: str
    center_sample_ts: datetime
    start_ts: datetime
    end_ts: datetime
    expected_direction: str
    expected_config_name: str
    expected_threshold: float
    offline_p_up_5s: float
    offline_p_down_5s: float
    offline_yes_bid: float | None
    offline_yes_ask: float | None
    offline_no_bid: float | None
    offline_no_ask: float | None


@dataclass
class PendingOutcome:
    signal_id: str
    run_id: str
    window_id: str | None
    config_name: str
    direction: str
    market_id: str
    signal_ts: datetime
    local_signal_ts: datetime
    latency_ms: int
    exit_horizon_seconds: int
    threshold: float
    entry_due_ts: datetime
    exit_due_ts: datetime
    is_forced_signal: bool = False
    p_up: float | None = None
    p_down: float | None = None
    p_flat: float | None = None
    simulated_entry_ts: datetime | None = None
    simulated_exit_ts: datetime | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    entry_quote_ts: datetime | None = None
    exit_quote_ts: datetime | None = None
    entry_quote_available: bool = False
    exit_quote_available: bool = False
    entry_quote_stale: bool = False
    exit_quote_stale: bool = False
    entry_crossed_quote: bool = False
    exit_crossed_quote: bool = False


class ParquetBuffer:
    def __init__(self, root: str | Path, prefix: str) -> None:
        self.root = ensure_dir(root)
        self.prefix = prefix
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self, date_override: str | None = None) -> None:
        if not self.rows:
            return
        df = pl.DataFrame(self.rows)
        if "date" not in df.columns:
            date_val = date_override or utc_now().date().isoformat()
            df = df.with_columns(pl.lit(date_val).alias("date"))
        date_val = str(df["date"][0])
        out_dir = ensure_dir(self.root / f"date={date_val}")
        out_path = out_dir / f"{self.prefix}-{utc_now().strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        df.write_parquet(out_path)
        self.rows.clear()


class ShadowRuntime:
    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.run_id = f"{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.start_wall = utc_now()
        self.stop_wall = self.start_wall + timedelta(minutes=args.duration_minutes)
        self.replay_mode = bool(args.replay_windows)
        self.feature_names = read_json(cfg["model"]["features_path"])
        self.model_type = cfg["model"].get("type", "multiclass_repricing")
        if self.model_type == "executable_net_binary":
            self.model = None
            self.model_up = joblib.load(cfg["model"]["up_model_path"])
            self.model_down = joblib.load(cfg["model"]["down_model_path"])
        else:
            self.model = joblib.load(cfg["model"]["model_path"])
            self.model_up = None
            self.model_down = None
        self.signal_configs = [ShadowSignalConfig(**x) for x in cfg["signal_configs"]]
        self.signal_cfg_by_name = {x.name: x for x in self.signal_configs}
        self.market_meta, self.asset_to_market = self._load_market_meta(cfg["market_meta"]["silver_pm_path"])
        self.market_states: dict[str, MarketState] = {mid: MarketState(meta=m) for mid, m in self.market_meta.items()}
        self.binance = BinanceState()
        self.last_signal_ts: dict[tuple[str, str, str], datetime] = {}
        self.pending: list[PendingOutcome] = []
        self.replay_windows: list[ReplayWindow] = []
        self.replay_windows_by_market: dict[str, list[ReplayWindow]] = {}
        self.historical_gold_lookup: dict[tuple[str, datetime], dict[str, Any]] = {}
        self.report_dates: set[str] = set()

        base_shadow = ensure_dir(cfg["output"]["base_dir"])
        self.pm_quote_sink = ParquetBuffer(base_shadow / "pm_quote_state", "pm_quote_state")
        self.binance_sink = ParquetBuffer(base_shadow / "binance_ticks", "binance_ticks")
        self.signals_sink = ParquetBuffer(base_shadow / "repricing_signals", "repricing_signals")
        self.outcomes_sink = ParquetBuffer(base_shadow / "repricing_outcomes", "repricing_outcomes")
        self.latency_sink = ParquetBuffer(base_shadow / "latency_metrics", "latency_metrics")
        self.diagnostics_dir = ensure_dir(base_shadow / "run_diagnostics")

        self.stats: dict[str, Any] = {
            "quote_snapshot_count": 0,
            "feature_vector_count": 0,
            "feature_ready_count": 0,
            "model_inference_count": 0,
            "signal_count": 0,
            "outcome_count": 0,
            "max_p_up": None,
            "max_p_down": None,
            "p_up_values": [],
            "p_down_values": [],
            "filtered_by_reason": Counter(),
            "mode": "replay" if self.replay_mode else ("dry_run" if args.dry_run else "live"),
            "force_signals_from_windows": bool(args.force_signals_from_windows),
            "replay_windows_path": args.replay_windows,
        }

        if self.replay_mode:
            self._load_replay_windows(args.replay_windows)
            self._load_historical_gold(cfg["dry_run"]["historical_gold_path"])

    def polymarket_asset_ids(self) -> list[str]:
        asset_ids: list[str] = []
        for meta in self.market_meta.values():
            if meta.yes_asset_id:
                asset_ids.append(str(meta.yes_asset_id))
            if meta.no_asset_id:
                asset_ids.append(str(meta.no_asset_id))
        return sorted(set(asset_ids))

    def _load_market_meta(self, silver_pm_path: str) -> tuple[dict[str, MarketMeta], dict[str, tuple[str, str]]]:
        cols = ["market_id", "yes_asset_id", "no_asset_id", "market_start_ts", "market_end_ts"]
        df = scan_dataset(silver_pm_path, cols).unique(subset=["market_id"], keep="first")
        metas: dict[str, MarketMeta] = {}
        asset_map: dict[str, tuple[str, str]] = {}
        for r in df.to_dicts():
            if not r.get("market_id") or not r.get("yes_asset_id") or not r.get("no_asset_id"):
                continue
            meta = MarketMeta(
                market_id=str(r["market_id"]),
                yes_asset_id=str(r["yes_asset_id"]),
                no_asset_id=str(r["no_asset_id"]),
                market_start_ts=r.get("market_start_ts"),
                market_end_ts=r.get("market_end_ts"),
            )
            metas[meta.market_id] = meta
            asset_map[meta.yes_asset_id] = (meta.market_id, "UP")
            asset_map[meta.no_asset_id] = (meta.market_id, "DOWN")
        return metas, asset_map

    def refresh_live_market_meta(self, cfg: dict[str, Any]) -> int:
        """Refresh BTC 15m market metadata in-process and merge it into runtime state.

        The BTC 15m market slug is deterministic, so the process does not need to be
        restarted just to discover new markets. This best-effort refresh queries Gamma
        for the rolling slug window, rewrites the configured metadata parquet, and
        updates the in-memory market/asset maps used by the Polymarket websocket parser.
        """
        if gamma_get_market_by_slug is None or market_to_row is None or slugs_for_btc_15m is None:
            LOGGER.warning("Live market metadata refresh unavailable; refresh_live_pm_market_meta import failed")
            return 0
        pm_feed_cfg = _get_nested(cfg, "feeds", "polymarket", default={}) or {}
        lookback_minutes = float(pm_feed_cfg.get("metadata_refresh_lookback_minutes", 30))
        lookahead_hours = float(pm_feed_cfg.get("metadata_refresh_lookahead_hours", 4))
        timeout_seconds = float(pm_feed_cfg.get("metadata_refresh_timeout_seconds", 10))
        sleep_seconds = float(pm_feed_cfg.get("metadata_refresh_sleep_seconds", 0.02))
        started = utc_now()
        rows: list[dict[str, Any]] = []
        misses = 0
        for slug, start_ts, end_ts in slugs_for_btc_15m(started, int(lookback_minutes), lookahead_hours):
            try:
                market = gamma_get_market_by_slug(slug, timeout_seconds)
                row = market_to_row(market, start_ts, end_ts) if market else None
            except Exception:
                row = None
            if row is None:
                misses += 1
            elif row["market_end_ts"] >= started - timedelta(minutes=lookback_minutes):
                rows.append(row)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if not rows:
            LOGGER.warning("Live market metadata refresh found no rows (misses=%s)", misses)
            return 0

        df = pl.DataFrame(rows).unique(subset=["market_id"], keep="first").sort("market_start_ts")
        out = Path(cfg["market_meta"]["silver_pm_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)

        new_metas, new_asset_map = self._load_market_meta(str(out))
        added = 0
        for market_id, meta in new_metas.items():
            if market_id not in self.market_meta:
                added += 1
                self.market_states[market_id] = MarketState(meta=meta)
            elif market_id in self.market_states:
                self.market_states[market_id].meta = meta
            self.market_meta[market_id] = meta
        self.asset_to_market.update(new_asset_map)
        LOGGER.info(
            "Live market metadata refreshed: rows=%s added=%s total=%s misses=%s max_end=%s",
            len(new_metas),
            added,
            len(self.market_meta),
            misses,
            df["market_end_ts"].max(),
        )
        return added

    def _load_replay_windows(self, path: str | None) -> None:
        if not path:
            return
        payload = read_json(path)
        windows: list[ReplayWindow] = []
        for raw in payload.get("windows", []):
            windows.append(
                ReplayWindow(
                    window_id=str(raw["window_id"]),
                    market_id=str(raw["market_id"]),
                    center_sample_ts=to_utc_ts(raw["center_sample_ts"]) or utc_now(),
                    start_ts=to_utc_ts(raw["start_ts"]) or utc_now(),
                    end_ts=to_utc_ts(raw["end_ts"]) or utc_now(),
                    expected_direction=str(raw["expected_direction"]),
                    expected_config_name=str(raw["expected_config_name"]),
                    expected_threshold=float(raw["expected_threshold"]),
                    offline_p_up_5s=float(raw["offline_p_up_5s"]),
                    offline_p_down_5s=float(raw["offline_p_down_5s"]),
                    offline_yes_bid=None if raw.get("offline_yes_bid") is None else float(raw["offline_yes_bid"]),
                    offline_yes_ask=None if raw.get("offline_yes_ask") is None else float(raw["offline_yes_ask"]),
                    offline_no_bid=None if raw.get("offline_no_bid") is None else float(raw["offline_no_bid"]),
                    offline_no_ask=None if raw.get("offline_no_ask") is None else float(raw["offline_no_ask"]),
                )
            )
        self.replay_windows = windows
        by_market: dict[str, list[ReplayWindow]] = {}
        for w in windows:
            by_market.setdefault(w.market_id, []).append(w)
        self.replay_windows_by_market = by_market
        LOGGER.info("Loaded %s replay windows across %s markets", len(windows), len(by_market))

    def _load_historical_gold(self, gold_path: str) -> None:
        cols = list(dict.fromkeys(self.feature_names + ["market_id", "sample_ts", "split"]))
        df = scan_dataset(gold_path, cols)
        lookup: dict[tuple[str, datetime], dict[str, Any]] = {}
        for row in df.to_dicts():
            market_id = row.get("market_id")
            ts = row.get("sample_ts")
            if market_id is None or ts is None:
                continue
            lookup[(str(market_id), ts)] = row
        self.historical_gold_lookup = lookup
        LOGGER.info("Loaded %s historical gold feature rows for replay parity", len(lookup))

    def _record_prob_stats(self, p_up: float, p_down: float) -> None:
        self.stats["p_up_values"].append(float(p_up))
        self.stats["p_down_values"].append(float(p_down))
        self.stats["max_p_up"] = p_up if self.stats["max_p_up"] is None else max(self.stats["max_p_up"], p_up)
        self.stats["max_p_down"] = p_down if self.stats["max_p_down"] is None else max(self.stats["max_p_down"], p_down)

    def _prune_histories(self, now_ts: datetime) -> None:
        cutoff_60 = now_ts - timedelta(seconds=65)
        while self.binance.mid_hist and self.binance.mid_hist[0][0] < cutoff_60:
            self.binance.mid_hist.popleft()
        for st in self.market_states.values():
            while st.pm_mid_hist and st.pm_mid_hist[0][0] < cutoff_60:
                st.pm_mid_hist.popleft()

    def _lag_value(self, hist: deque[tuple[datetime, float]], now_ts: datetime, seconds: int) -> float | None:
        target = now_ts - timedelta(seconds=seconds)
        candidate = None
        for ts, value in hist:
            if ts <= target:
                candidate = value
            else:
                break
        return candidate

    def _rolling_std_logret(self, hist: deque[tuple[datetime, float]], window_seconds: int) -> float | None:
        vals = [v for _, v in hist if v and v > 0]
        if len(vals) < 2:
            return None
        logrets = np.diff(np.log(np.asarray(vals[-window_seconds:], dtype=float)))
        if len(logrets) < 2:
            return None
        return float(np.nanstd(logrets, ddof=1))

    def update_binance_bookticker(self, payload: dict[str, Any], recv_ts: datetime) -> None:
        try:
            bid = float(payload.get("b") or payload.get("bidPrice"))
            ask = float(payload.get("a") or payload.get("askPrice"))
        except Exception:
            return
        event_ms = payload.get("E") or payload.get("eventTime") or payload.get("T")
        event_ts = to_utc_ts(event_ms) or recv_ts
        self.binance.bid = bid
        self.binance.ask = ask
        try:
            self.binance.bid_qty = float(payload.get("B") or payload.get("bidQty") or 0.0)
            self.binance.ask_qty = float(payload.get("A") or payload.get("askQty") or 0.0)
        except Exception:
            pass
        self.binance.ts_event = event_ts
        self.binance.ts_recv = recv_ts
        if self.binance.mid is not None:
            self.binance.mid_hist.append((event_ts, self.binance.mid))
        self._prune_histories(event_ts)
        self.binance_sink.append(
            {
                "run_id": self.run_id,
                "date": recv_ts.date().isoformat(),
                "sample_ts": event_ts,
                "ts_recv": recv_ts,
                "stream": "bookTicker",
                "bid": bid,
                "ask": ask,
                "mid": self.binance.mid,
                "bid_qty": self.binance.bid_qty,
                "ask_qty": self.binance.ask_qty,
            }
        )

    def update_binance_aggtrade(self, payload: dict[str, Any], recv_ts: datetime) -> None:
        try:
            price = float(payload.get("p") or payload.get("price"))
            qty = float(payload.get("q") or payload.get("qty") or 0.0)
        except Exception:
            return
        event_ms = payload.get("E") or payload.get("eventTime") or payload.get("T")
        event_ts = to_utc_ts(event_ms) or recv_ts
        is_buyer_maker = bool(payload.get("m"))
        self.binance.trade_hist.append((event_ts, price, qty, is_buyer_maker))
        self._prune_histories(event_ts)
        self.binance_sink.append(
            {
                "run_id": self.run_id,
                "date": recv_ts.date().isoformat(),
                "sample_ts": event_ts,
                "ts_recv": recv_ts,
                "stream": "aggTrade",
                "trade_price": price,
                "trade_qty": qty,
                "is_buyer_maker": is_buyer_maker,
            }
        )

    def _update_market_state_from_row(self, row: dict[str, Any]) -> None:
        market_id = str(row["market_id"])
        if market_id not in self.market_states:
            return
        st = self.market_states[market_id]
        ts = row["sample_ts"]
        st.last_sample_ts = ts
        st.yes.bid = row.get("yes_bid")
        st.yes.ask = row.get("yes_ask")
        st.yes.ts_event = row.get("yes_last_quote_update_ts") or ts
        st.yes.ts_recv = ts
        st.yes.crossed = bool(row.get("yes_crossed_quote") or (st.yes.bid is not None and st.yes.ask is not None and st.yes.bid > st.yes.ask))
        st.yes.quote_age_seconds = row.get("yes_quote_age_seconds")
        st.yes.is_stale = bool(row.get("yes_is_stale", False))

        st.no.bid = row.get("no_bid")
        st.no.ask = row.get("no_ask")
        st.no.ts_event = row.get("no_last_quote_update_ts") or ts
        st.no.ts_recv = ts
        st.no.crossed = bool(row.get("no_crossed_quote") or (st.no.bid is not None and st.no.ask is not None and st.no.bid > st.no.ask))
        st.no.quote_age_seconds = row.get("no_quote_age_seconds")
        st.no.is_stale = bool(row.get("no_is_stale", False))

        st.last_pm_update_ts = ts if row.get("seconds_since_last_pm_update") is not None else st.last_pm_update_ts
        if row.get("yes_mid") is not None:
            st.pm_mid_hist.append((ts, float(row["yes_mid"])))
        self._prune_histories(ts)
        self.stats["quote_snapshot_count"] += 1
        date_str = str(row.get("date") or ts.date().isoformat())
        self.report_dates.add(date_str)
        self.pm_quote_sink.append(
            {
                "run_id": self.run_id,
                "date": date_str,
                "market_id": market_id,
                "side": "BOTH",
                "sample_ts": ts,
                "yes_bid": st.yes.bid,
                "yes_ask": st.yes.ask,
                "no_bid": st.no.bid,
                "no_ask": st.no.ask,
                "yes_mid": row.get("yes_mid"),
                "no_mid": row.get("no_mid"),
                "yes_quote_age_seconds": st.yes.quote_age_seconds,
                "no_quote_age_seconds": st.no.quote_age_seconds,
                "yes_is_stale": st.yes.is_stale,
                "no_is_stale": st.no.is_stale,
            }
        )

    def update_pm_quote(self, event: dict[str, Any], recv_ts: datetime) -> None:
        asset_id = str(event.get("asset_id") or event.get("token_id") or "")
        market_id = str(event.get("market_id") or event.get("market") or "")
        if not market_id and asset_id in self.asset_to_market:
            market_id = self.asset_to_market[asset_id][0]
        if not market_id or market_id not in self.market_states:
            return
        st = self.market_states[market_id]
        side = "UP"
        if asset_id:
            if asset_id == st.meta.yes_asset_id:
                side = "UP"
            elif asset_id == st.meta.no_asset_id:
                side = "DOWN"
        q = st.yes if side == "UP" else st.no
        event_ts = to_utc_ts(event.get("ts_event") or event.get("timestamp")) or recv_ts
        bid = event.get("best_bid")
        ask = event.get("best_ask")
        if "bids" in event:
            q.bid_depth_5 = depth_sum(event.get("bids"), "bid", 5)
        if "asks" in event:
            q.ask_depth_5 = depth_sum(event.get("asks"), "ask", 5)
        if bid is None and "bids" in event:
            bid, q.bid_size = best_levels(event.get("bids"), "bid")
        if ask is None and "asks" in event:
            ask, q.ask_size = best_levels(event.get("asks"), "ask")
        try:
            q.bid = None if bid is None else float(bid)
            q.ask = None if ask is None else float(ask)
        except Exception:
            return
        q.ts_event = event_ts
        q.ts_recv = recv_ts
        q.crossed = bool(q.bid is not None and q.ask is not None and q.bid > q.ask)
        q.quote_age_seconds = 0.0
        q.is_stale = False
        st.last_pm_update_ts = event_ts
        st.last_sample_ts = recv_ts
        if st.yes.mid is not None:
            st.pm_mid_hist.append((event_ts, st.yes.mid))
        self._prune_histories(event_ts)
        date_str = recv_ts.date().isoformat()
        self.report_dates.add(date_str)
        self.stats["quote_snapshot_count"] += 1
        self.pm_quote_sink.append(
            {
                "run_id": self.run_id,
                "date": date_str,
                "market_id": market_id,
                "asset_id": asset_id,
                "side": side,
                "sample_ts": event_ts,
                "ts_recv": recv_ts,
                "best_bid": q.bid,
                "best_ask": q.ask,
                "mid": q.mid,
                "spread": q.spread,
                "bid_depth_5": q.bid_depth_5,
                "ask_depth_5": q.ask_depth_5,
                "crossed_quote": q.crossed,
            }
        )
        self._maybe_emit_signal(market_id, recv_ts)
        self._advance_pending(market_id, recv_ts)

    def _trade_flow(self, recv_ts: datetime, window_seconds: int) -> dict[str, float | int | None]:
        rows = [(ts, price, qty, is_buyer_maker) for ts, price, qty, is_buyer_maker in self.binance.trade_hist if 0 <= (recv_ts - ts).total_seconds() <= window_seconds]
        if not rows:
            return {"buy": 0.0, "sell": 0.0, "net": 0.0, "total": 0.0, "count": 0, "imbalance": None}
        buy = sum(qty for _ts, _p, qty, maker in rows if not maker)
        sell = sum(qty for _ts, _p, qty, maker in rows if maker)
        total = buy + sell
        return {"buy": float(buy), "sell": float(sell), "net": float(buy - sell), "total": float(total), "count": len(rows), "imbalance": None if total <= 0 else float((buy - sell) / total)}

    def _depth_imbalance(self, bid_depth: float | None, ask_depth: float | None) -> float | None:
        if bid_depth is None or ask_depth is None or bid_depth + ask_depth <= 0:
            return None
        return float((bid_depth - ask_depth) / (bid_depth + ask_depth))

    def _feature_map_from_state(self, market_id: str, recv_ts: datetime) -> dict[str, float | None] | None:
        st = self.market_states[market_id]
        if self.binance.mid is None or st.yes.mid is None or st.no.mid is None:
            return None
        if st.meta.market_end_ts is None or st.meta.market_start_ts is None:
            return None
        tte = (st.meta.market_end_ts - recv_ts).total_seconds()
        if tte <= 0:
            return None
        elapsed = (recv_ts - st.meta.market_start_ts).total_seconds()
        lag1 = self._lag_value(self.binance.mid_hist, recv_ts, 1)
        lag5 = self._lag_value(self.binance.mid_hist, recv_ts, 5)
        lag10 = self._lag_value(self.binance.mid_hist, recv_ts, 10)
        lag30 = self._lag_value(self.binance.mid_hist, recv_ts, 30)
        yes_mid_1 = self._lag_value(st.pm_mid_hist, recv_ts, 1)
        yes_mid_5 = self._lag_value(st.pm_mid_hist, recv_ts, 5)
        # Live shadow currently tracks YES mid history; NO momentum is set null and imputed.
        f1 = self._trade_flow(recv_ts, 1)
        f5 = self._trade_flow(recv_ts, 5)
        f30 = self._trade_flow(recv_ts, 30)
        btc_bid_depth = self.binance.bid_qty
        btc_ask_depth = self.binance.ask_qty
        yes_di = self._depth_imbalance(st.yes.bid_depth_5, st.yes.ask_depth_5)
        no_di = self._depth_imbalance(st.no.bid_depth_5, st.no.ask_depth_5)
        btc_di = self._depth_imbalance(btc_bid_depth, btc_ask_depth)
        btc_spread = None if self.binance.bid is None or self.binance.ask is None else self.binance.ask - self.binance.bid
        btc_micro = None
        if self.binance.bid is not None and self.binance.ask is not None and self.binance.bid_qty is not None and self.binance.ask_qty is not None and self.binance.bid_qty + self.binance.ask_qty > 0:
            micro = (self.binance.ask * self.binance.bid_qty + self.binance.bid * self.binance.ask_qty) / (self.binance.bid_qty + self.binance.ask_qty)
            btc_micro = micro - self.binance.mid
        formula_p_yes = 0.5
        feature_map = {
            "time_to_expiry_seconds": tte,
            "time_elapsed_seconds": elapsed,
            "yes_bid": st.yes.bid,
            "yes_ask": st.yes.ask,
            "yes_mid": st.yes.mid,
            "no_bid": st.no.bid,
            "no_ask": st.no.ask,
            "no_mid": st.no.mid,
            "yes_spread": st.yes.spread,
            "no_spread": st.no.spread,
            "yes_bid_depth_5": st.yes.bid_depth_5,
            "yes_ask_depth_5": st.yes.ask_depth_5,
            "yes_depth_imbalance_5": yes_di,
            "yes_depth_imbalance_5_live": yes_di,
            "no_bid_depth_5": st.no.bid_depth_5,
            "no_ask_depth_5": st.no.ask_depth_5,
            "no_depth_imbalance_5": no_di,
            "pair_bid_sum": None if st.yes.bid is None or st.no.bid is None else st.yes.bid + st.no.bid,
            "pair_ask_sum": None if st.yes.ask is None or st.no.ask is None else st.yes.ask + st.no.ask,
            "pair_mid_sum": st.yes.mid + st.no.mid,
            "pair_mid_sum_live": st.yes.mid + st.no.mid,
            "btc_return_1s": None if lag1 in (None, 0) else (self.binance.mid / lag1 - 1.0),
            "btc_return_5s": None if lag5 in (None, 0) else (self.binance.mid / lag5 - 1.0),
            "btc_return_10s": None if lag10 in (None, 0) else (self.binance.mid / lag10 - 1.0),
            "btc_return_30s": None if lag30 in (None, 0) else (self.binance.mid / lag30 - 1.0),
            "realized_vol_10s": self._rolling_std_logret(self.binance.mid_hist, 10),
            "realized_vol_30s": self._rolling_std_logret(self.binance.mid_hist, 30),
            "btc_realized_vol_60s": self._rolling_std_logret(self.binance.mid_hist, 60),
            "formula_p_yes": formula_p_yes,
            "formula_p_yes_minus_yes_mid": formula_p_yes - st.yes.mid,
            "formula_p_yes_minus_yes_ask": None if st.yes.ask is None else formula_p_yes - st.yes.ask,
            "formula_p_no_minus_no_ask": None if st.no.ask is None else (1.0 - formula_p_yes) - st.no.ask,
            "pm_yes_mid_change_1s_past": None if yes_mid_1 is None else st.yes.mid - yes_mid_1,
            "pm_yes_mid_change_5s_past": None if yes_mid_5 is None else st.yes.mid - yes_mid_5,
            "pm_no_mid_change_1s_past": None,
            "pm_no_mid_change_5s_past": None,
            "seconds_since_last_pm_update": None if st.last_pm_update_ts is None else (recv_ts - st.last_pm_update_ts).total_seconds(),
            "yes_quote_age_seconds": None if st.yes.ts_event is None else max(0.0, (recv_ts - st.yes.ts_event).total_seconds()),
            "no_quote_age_seconds": None if st.no.ts_event is None else max(0.0, (recv_ts - st.no.ts_event).total_seconds()),
            "btc_spread": btc_spread,
            "btc_microprice_minus_mid": btc_micro,
            "btc_buy_volume_1s": f1["buy"], "btc_sell_volume_1s": f1["sell"], "btc_net_volume_1s": f1["net"], "btc_total_volume_1s": f1["total"], "btc_trade_count_1s": f1["count"], "btc_trade_imbalance_1s": f1["imbalance"],
            "btc_buy_volume_5s": f5["buy"], "btc_sell_volume_5s": f5["sell"], "btc_net_volume_5s": f5["net"], "btc_total_volume_5s": f5["total"], "btc_trade_count_5s": f5["count"], "btc_trade_imbalance_5s": f5["imbalance"],
            "btc_buy_volume_30s": f30["buy"], "btc_sell_volume_30s": f30["sell"], "btc_net_volume_30s": f30["net"], "btc_total_volume_30s": f30["total"], "btc_trade_count_30s": f30["count"], "btc_trade_imbalance_30s": f30["imbalance"],
            "btc_bid_depth_5": btc_bid_depth,
            "btc_ask_depth_5": btc_ask_depth,
            "btc_depth_imbalance_5": btc_di,
            "latency_seconds": 0,
        }
        return feature_map

    def _feature_map_from_historical_gold(self, market_id: str, recv_ts: datetime) -> dict[str, float | None] | None:
        row = self.historical_gold_lookup.get((market_id, recv_ts))
        if row is None:
            return None
        return {name: row.get(name) for name in self.feature_names}

    def _maybe_emit_signal(self, market_id: str, recv_ts: datetime, window_id: str | None = None) -> None:
        self.stats["feature_vector_count"] += 1
        if self.replay_mode:
            feature_map = self._feature_map_from_historical_gold(market_id, recv_ts)
        else:
            feature_map = self._feature_map_from_state(market_id, recv_ts)
        if feature_map is None:
            self.stats["filtered_by_reason"]["feature_not_ready"] += 1
            return
        self.stats["feature_ready_count"] += 1
        x = np.asarray([[np.nan if feature_map.get(f) is None else feature_map.get(f) for f in self.feature_names]], dtype=float)
        infer_start = utc_now() if not self.replay_mode else recv_ts
        if self.model_type == "executable_net_binary":
            p_up = float(self.model_up.predict_proba(x)[0][1])
            p_down = float(self.model_down.predict_proba(x)[0][1])
            p_flat = 0.0
        else:
            probs = self.model.predict_proba(x)[0]
            p_down, p_flat, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        infer_end = utc_now() if not self.replay_mode else recv_ts
        self.stats["model_inference_count"] += 1
        self._record_prob_stats(p_up, p_down)
        for cfg in self.signal_configs:
            self._evaluate_config(cfg, market_id, recv_ts, feature_map, p_up, p_down, p_flat, infer_start, infer_end, is_forced_signal=False, window_id=window_id)

    def _emit_signal(
        self,
        cfg: ShadowSignalConfig,
        market_id: str,
        recv_ts: datetime,
        feature_map: dict[str, float | None],
        p_up: float,
        p_down: float,
        p_flat: float,
        infer_start: datetime,
        infer_end: datetime,
        direction: str,
        threshold: float,
        is_forced_signal: bool,
        window_id: str | None,
    ) -> None:
        signal_id = uuid.uuid4().hex
        local_signal_ts = recv_ts if self.replay_mode else utc_now()
        date_str = recv_ts.date().isoformat()
        self.report_dates.add(date_str)
        self.stats["signal_count"] += 1
        signal_row = {
            "run_id": self.run_id,
            "date": date_str,
            "signal_id": signal_id,
            "window_id": window_id,
            "config_name": cfg.name,
            "market_id": market_id,
            "sample_ts": recv_ts,
            "local_signal_ts": local_signal_ts,
            "model_version": self.cfg["model"].get("version", "unknown"),
            "p_up": p_up,
            "p_down": p_down,
            "p_flat": p_flat,
            "direction": direction,
            "threshold": threshold,
            "yes_bid": feature_map.get("yes_bid"),
            "yes_ask": feature_map.get("yes_ask"),
            "no_bid": feature_map.get("no_bid"),
            "no_ask": feature_map.get("no_ask"),
            "yes_spread": feature_map.get("yes_spread"),
            "no_spread": None if self.market_states[market_id].no.spread is None else self.market_states[market_id].no.spread,
            "quote_age": feature_map.get("seconds_since_last_pm_update"),
            "formula_p_yes": feature_map.get("formula_p_yes"),
            "yes_mid": feature_map.get("yes_mid"),
            "no_mid": feature_map.get("no_mid"),
            "btc_mid": self.binance.mid,
            "btc_return_1s": feature_map.get("btc_return_1s"),
            "btc_return_5s": feature_map.get("btc_return_5s"),
            "is_forced_signal": is_forced_signal,
        }
        self.signals_sink.append(signal_row)
        self.latency_sink.append(
            {
                "run_id": self.run_id,
                "date": date_str,
                "signal_id": signal_id,
                "window_id": window_id,
                "market_id": market_id,
                "config_name": cfg.name,
                "is_forced_signal": is_forced_signal,
                "data_receive_ts": recv_ts,
                "feature_ready_ts": infer_start,
                "model_infer_start_ts": infer_start,
                "model_infer_end_ts": infer_end,
                "signal_emit_ts": local_signal_ts,
                "quote_update_ts": self.market_states[market_id].last_pm_update_ts,
                "feature_latency_ms": (infer_start - recv_ts).total_seconds() * 1000.0,
                "model_latency_ms": (infer_end - infer_start).total_seconds() * 1000.0,
                "emit_latency_ms": (local_signal_ts - recv_ts).total_seconds() * 1000.0,
                "quote_age_ms": None if self.market_states[market_id].last_pm_update_ts is None else (local_signal_ts - self.market_states[market_id].last_pm_update_ts).total_seconds() * 1000.0,
            }
        )
        for latency_ms in ENTRY_LATENCIES_MS:
            for horizon_s in EXIT_HORIZONS_S:
                self.pending.append(
                    PendingOutcome(
                        signal_id=signal_id,
                        run_id=self.run_id,
                        window_id=window_id,
                        config_name=cfg.name,
                        direction=direction,
                        market_id=market_id,
                        signal_ts=recv_ts,
                        local_signal_ts=local_signal_ts,
                        latency_ms=latency_ms,
                        exit_horizon_seconds=horizon_s,
                        threshold=threshold,
                        entry_due_ts=recv_ts + timedelta(milliseconds=latency_ms),
                        exit_due_ts=recv_ts + timedelta(milliseconds=latency_ms, seconds=horizon_s),
                        is_forced_signal=is_forced_signal,
                        p_up=p_up,
                        p_down=p_down,
                        p_flat=p_flat,
                    )
                )

    def _evaluate_config(
        self,
        cfg: ShadowSignalConfig,
        market_id: str,
        recv_ts: datetime,
        feature_map: dict[str, float | None],
        p_up: float,
        p_down: float,
        p_flat: float,
        infer_start: datetime,
        infer_end: datetime,
        is_forced_signal: bool,
        window_id: str | None,
    ) -> None:
        tte = float(feature_map.get("time_to_expiry_seconds") or 0.0)
        yes_spread = feature_map.get("yes_spread")
        no_spread = None if self.market_states[market_id].no.spread is None else self.market_states[market_id].no.spread
        st = self.market_states[market_id]
        if st.yes.is_stale or st.no.is_stale or st.yes.crossed or st.no.crossed:
            self.stats["filtered_by_reason"]["quote_stale"] += 1
            return
        if tte < cfg.min_tte:
            self.stats["filtered_by_reason"]["tte_too_low"] += 1
            return
        candidates: list[tuple[str, float]] = []
        if cfg.direction in {"UP", "BOTH"} and cfg.threshold_up is not None:
            if p_up >= cfg.threshold_up or is_forced_signal:
                candidates.append(("UP", float(cfg.threshold_up)))
            else:
                self.stats["filtered_by_reason"]["probability_below_threshold"] += 1
        if cfg.direction in {"DOWN", "BOTH"} and cfg.threshold_down is not None:
            if p_down >= cfg.threshold_down or is_forced_signal:
                candidates.append(("DOWN", float(cfg.threshold_down)))
            else:
                self.stats["filtered_by_reason"]["probability_below_threshold"] += 1
        for direction, threshold in candidates:
            spread = yes_spread if direction == "UP" else no_spread
            if spread is None or (cfg.max_spread is not None and spread > cfg.max_spread):
                self.stats["filtered_by_reason"]["spread_too_wide"] += 1
                continue
            cool_key = (cfg.name, market_id, direction)
            last = self.last_signal_ts.get(cool_key)
            if last is not None and (recv_ts - last).total_seconds() < cfg.cooldown_seconds:
                self.stats["filtered_by_reason"]["cooldown"] += 1
                continue
            self.last_signal_ts[cool_key] = recv_ts
            self._emit_signal(
                cfg=cfg,
                market_id=market_id,
                recv_ts=recv_ts,
                feature_map=feature_map,
                p_up=p_up,
                p_down=p_down,
                p_flat=p_flat,
                infer_start=infer_start,
                infer_end=infer_end,
                direction=direction,
                threshold=threshold,
                is_forced_signal=is_forced_signal,
                window_id=window_id,
            )

    def _force_window_signal(self, window: ReplayWindow, recv_ts: datetime) -> None:
        cfg = self.signal_cfg_by_name.get(window.expected_config_name)
        if cfg is None:
            return
        feature_map = self._feature_map_from_historical_gold(window.market_id, recv_ts)
        if feature_map is None:
            self.stats["filtered_by_reason"]["feature_not_ready"] += 1
            return
        x = np.asarray([[np.nan if feature_map.get(f) is None else feature_map.get(f) for f in self.feature_names]], dtype=float)
        probs = self.model.predict_proba(x)[0]
        p_down, p_flat, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        self._record_prob_stats(p_up, p_down)
        self._emit_signal(
            cfg=cfg,
            market_id=window.market_id,
            recv_ts=recv_ts,
            feature_map=feature_map,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            infer_start=recv_ts,
            infer_end=recv_ts,
            direction=window.expected_direction,
            threshold=window.expected_threshold,
            is_forced_signal=True,
            window_id=window.window_id,
        )

    def _advance_pending(self, market_id: str, recv_ts: datetime) -> None:
        st = self.market_states[market_id]
        remaining: list[PendingOutcome] = []
        for po in self.pending:
            if po.market_id != market_id:
                remaining.append(po)
                continue
            quote = st.yes if po.direction == "UP" else st.no
            if po.entry_price is None and recv_ts >= po.entry_due_ts:
                po.simulated_entry_ts = recv_ts
                po.entry_quote_ts = quote.ts_event
                po.entry_quote_available = quote.ask is not None
                po.entry_quote_stale = bool(quote.is_stale)
                po.entry_crossed_quote = quote.crossed
                po.entry_price = quote.ask
            if po.entry_price is not None and po.exit_price is None and recv_ts >= po.exit_due_ts:
                po.simulated_exit_ts = recv_ts
                po.exit_quote_ts = quote.ts_event
                po.exit_quote_available = quote.bid is not None
                po.exit_quote_stale = bool(quote.is_stale)
                po.exit_crossed_quote = quote.crossed
                po.exit_price = quote.bid
            if po.exit_price is not None or recv_ts > po.exit_due_ts + timedelta(seconds=2):
                pnl = None
                roi = None
                net_pnl = None
                entry_fee = None
                exit_fee = None
                if po.entry_price is not None and po.exit_price is not None:
                    pnl = float(po.exit_price - po.entry_price)
                    roi = None if po.entry_price <= 0 else float(pnl / po.entry_price)
                    fee_rate = float(self.cfg["model"].get("fee_rate", 0.07))
                    slip = float(self.cfg["model"].get("slippage_buffer", 0.0025))
                    entry_fee = fee_rate * po.entry_price * (1.0 - po.entry_price)
                    exit_fee = fee_rate * po.exit_price * (1.0 - po.exit_price)
                    net_pnl = float(pnl - entry_fee - exit_fee - slip)
                self.stats["outcome_count"] += 1
                self.outcomes_sink.append(
                    {
                        "run_id": self.run_id,
                        "date": recv_ts.date().isoformat(),
                        "signal_id": po.signal_id,
                        "window_id": po.window_id,
                        "config_name": po.config_name,
                        "market_id": po.market_id,
                        "direction": po.direction,
                        "threshold": po.threshold,
                        "signal_ts": po.signal_ts,
                        "entry_latency_ms": po.latency_ms,
                        "exit_horizon_seconds": po.exit_horizon_seconds,
                        "simulated_entry_ts": po.simulated_entry_ts,
                        "simulated_exit_ts": po.simulated_exit_ts,
                        "entry_quote_ts": po.entry_quote_ts,
                        "entry_price": po.entry_price,
                        "exit_quote_ts": po.exit_quote_ts,
                        "exit_price": po.exit_price,
                        "pnl": pnl,
                        "roi": roi,
                        "entry_fee": entry_fee,
                        "exit_fee": exit_fee,
                        "net_pnl": net_pnl,
                        "entry_quote_available": po.entry_quote_available,
                        "exit_quote_available": po.exit_quote_available,
                        "entry_quote_stale": po.entry_quote_stale,
                        "exit_quote_stale": po.exit_quote_stale,
                        "entry_crossed_quote": po.entry_crossed_quote,
                        "exit_crossed_quote": po.exit_crossed_quote,
                        "is_forced_signal": po.is_forced_signal,
                        "p_up": po.p_up,
                        "p_down": po.p_down,
                        "p_flat": po.p_flat,
                    }
                )
            else:
                remaining.append(po)
        self.pending = remaining

    def finalize(self) -> None:
        self.pm_quote_sink.flush()
        self.binance_sink.flush()
        self.signals_sink.flush()
        self.outcomes_sink.flush()
        self.latency_sink.flush()
        self._write_diagnostics()

    def _write_diagnostics(self) -> None:
        if not self.report_dates:
            self.report_dates.add(self.start_wall.date().isoformat())
        filtered = dict(self.stats["filtered_by_reason"])
        p_up = np.asarray(self.stats["p_up_values"], dtype=float) if self.stats["p_up_values"] else np.asarray([], dtype=float)
        p_down = np.asarray(self.stats["p_down_values"], dtype=float) if self.stats["p_down_values"] else np.asarray([], dtype=float)
        summary = {
            "run_id": self.run_id,
            "created_at": utc_now().isoformat(),
            "mode": self.stats["mode"],
            "force_signals_from_windows": self.stats["force_signals_from_windows"],
            "report_dates": sorted(self.report_dates),
            "quote_snapshot_count": self.stats["quote_snapshot_count"],
            "feature_vector_count": self.stats["feature_vector_count"],
            "feature_ready_count": self.stats["feature_ready_count"],
            "model_inference_count": self.stats["model_inference_count"],
            "signal_count": self.stats["signal_count"],
            "outcome_count": self.stats["outcome_count"],
            "filtered_by_reason": filtered,
            "max_p_up": self.stats["max_p_up"],
            "max_p_down": self.stats["max_p_down"],
            "p_up_quantiles": {} if p_up.size == 0 else {k: float(np.quantile(p_up, q)) for k, q in [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)]},
            "p_down_quantiles": {} if p_down.size == 0 else {k: float(np.quantile(p_down, q)) for k, q in [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)]},
        }
        for date_str in summary["report_dates"]:
            out = ensure_dir(self.diagnostics_dir / f"date={date_str}") / f"run_diagnostics-{self.run_id}.json"
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


async def binance_task(rt: ShadowRuntime, cfg: dict[str, Any]) -> None:
    url = cfg["binance"]["ws_url"]
    # Binance already sends enough traffic on bookTicker/aggTrade for BTCUSDT; client-side
    # websocket pings have been a common source of false-positive keepalive timeouts on
    # Lightsail. Disable protocol pings and rely on the outer resilient task to reconnect
    # if messages stop or the TCP connection closes. Prefer port 443 in config; it is less
    # likely to be filtered/throttled than 9443 on some networks.
    async with websockets.connect(
        url,
        ping_interval=None,
        ping_timeout=None,
        open_timeout=15,
        close_timeout=5,
        max_size=2**22,
    ) as ws:
        async for msg in ws:
            recv_ts = utc_now()
            try:
                raw = orjson.loads(msg)
            except Exception:
                continue
            stream = (raw.get("stream") or "").lower()
            payload = raw.get("data", raw)
            if stream.endswith("@bookticker") or payload.get("e") == "bookTicker":
                rt.update_binance_bookticker(payload, recv_ts)
            elif stream.endswith("@aggtrade") or payload.get("e") == "aggTrade":
                rt.update_binance_aggtrade(payload, recv_ts)
            if utc_now() >= rt.stop_wall:
                break


def parse_pm_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "best_bid" in payload or "best_ask" in payload:
        return {
            "market_id": payload.get("market_id") or payload.get("market"),
            "asset_id": payload.get("asset_id") or payload.get("token_id"),
            "ts_event": payload.get("timestamp") or payload.get("ts_event"),
            "best_bid": payload.get("best_bid"),
            "best_ask": payload.get("best_ask"),
        }
    if "bids" in payload or "asks" in payload:
        return {
            "market_id": payload.get("market_id") or payload.get("market"),
            "asset_id": payload.get("asset_id") or payload.get("token_id"),
            "ts_event": payload.get("timestamp") or payload.get("ts_event"),
            "bids": payload.get("bids"),
            "asks": payload.get("asks"),
        }
    if isinstance(payload.get("price_changes"), list):
        return None
    return None


def _active_pm_subscription_metas(rt: ShadowRuntime, cfg: dict[str, Any]) -> list[MarketMeta]:
    pm_feed_cfg = _get_nested(cfg, "feeds", "polymarket", default={}) or {}
    lookback_s = float(pm_feed_cfg.get("subscribe_lookback_seconds", 300))
    lookahead_s = float(pm_feed_cfg.get("subscribe_lookahead_seconds", 1800))
    max_markets = int(pm_feed_cfg.get("max_subscribed_markets", 12))
    now = utc_now()
    metas: list[MarketMeta] = []
    for meta in rt.market_meta.values():
        start_ts = to_utc_ts(meta.market_start_ts)
        end_ts = to_utc_ts(meta.market_end_ts)
        if start_ts is None or end_ts is None:
            continue
        if end_ts < now - timedelta(seconds=lookback_s):
            continue
        if start_ts > now + timedelta(seconds=lookahead_s):
            continue
        metas.append(meta)
    metas.sort(key=lambda m: (to_utc_ts(m.market_start_ts) or now, to_utc_ts(m.market_end_ts) or now, m.market_id))
    if max_markets > 0:
        metas = metas[:max_markets]
    return metas


def build_pm_subscribe_messages(rt: ShadowRuntime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = cfg["polymarket"].get("subscribe_messages") or []
    if explicit:
        LOGGER.info("Using %s explicit Polymarket subscribe message(s)", len(explicit))
        return explicit
    channel = (
        _get_nested(cfg, "feeds", "polymarket", "market_channel", default=None)
        or cfg["polymarket"].get("market_channel")
        or "market"
    )
    metas = _active_pm_subscription_metas(rt, cfg)
    asset_ids: list[str] = []
    for meta in metas:
        if meta.yes_asset_id:
            asset_ids.append(str(meta.yes_asset_id))
        if meta.no_asset_id:
            asset_ids.append(str(meta.no_asset_id))
    asset_ids = sorted(set(asset_ids))
    if not asset_ids:
        LOGGER.warning("No active Polymarket assets selected for subscription from %s metadata markets", len(rt.market_meta))
        return []
    pm_feed_cfg = _get_nested(cfg, "feeds", "polymarket", default={}) or {}
    batch_size = int(pm_feed_cfg.get("subscribe_batch_size", 20))
    batch_size = max(1, min(batch_size, 200))
    messages: list[dict[str, Any]] = []
    for idx in range(0, len(asset_ids), batch_size):
        messages.append({"type": channel, "assets_ids": asset_ids[idx : idx + batch_size], "custom_feature_enabled": True})
    LOGGER.info(
        "Polymarket subscribe: %s market(s), %s asset(s), %s message(s), batch_size=%s",
        len(metas),
        len(asset_ids),
        len(messages),
        batch_size,
    )
    for meta in metas[:5]:
        LOGGER.info(
            "Polymarket subscribe market: %s start=%s end=%s yes=%s no=%s",
            meta.market_id,
            meta.market_start_ts,
            meta.market_end_ts,
            meta.yes_asset_id,
            meta.no_asset_id,
        )
    return messages


def pm_active_asset_ids(rt: ShadowRuntime, cfg: dict[str, Any]) -> set[str]:
    asset_ids: set[str] = set()
    for meta in _active_pm_subscription_metas(rt, cfg):
        if meta.yes_asset_id:
            asset_ids.add(str(meta.yes_asset_id))
        if meta.no_asset_id:
            asset_ids.add(str(meta.no_asset_id))
    return asset_ids


async def _polymarket_app_heartbeat(ws: Any, interval_seconds: float = 10.0) -> None:
    """Polymarket market/user channels require an application-level PING text.

    This is separate from the websocket protocol ping used by the websockets
    library. Polymarket documents that clients should send the string ``PING``
    every 10 seconds and the server responds with ``PONG``.
    """
    while True:
        await ws.send("PING")
        await asyncio.sleep(interval_seconds)


async def polymarket_task(rt: ShadowRuntime, cfg: dict[str, Any]) -> None:
    pm_cfg = cfg["polymarket"]
    url = pm_cfg.get("ws_url")
    if not url:
        raise RuntimeError("Missing polymarket.ws_url in shadow config")
    # Use Polymarket's documented application-level heartbeat instead of
    # websocket protocol pings, which the CLOB endpoint doesn't consistently
    # answer with protocol pongs.
    async with websockets.connect(url, ping_interval=None, ping_timeout=None, max_size=2**23) as ws:
        heartbeat_task = asyncio.create_task(_polymarket_app_heartbeat(ws))
        try:
            pm_feed_cfg = _get_nested(cfg, "feeds", "polymarket", default={}) or {}
            online_refresh = bool(pm_feed_cfg.get("online_metadata_refresh_enabled", True))
            refresh_interval_s = float(pm_feed_cfg.get("online_metadata_refresh_seconds", 300))
            next_refresh = utc_now() + timedelta(seconds=max(30.0, refresh_interval_s))
            subscribed_assets: set[str] = set()

            subs = build_pm_subscribe_messages(rt, cfg)
            if not subs:
                raise RuntimeError("No Polymarket subscribe messages available; refresh market metadata or provide explicit subscribe_messages")
            for sub in subs:
                await ws.send(orjson.dumps(sub).decode("utf-8"))
                LOGGER.info("Sent Polymarket subscription with %s asset(s)", len(sub.get("assets_ids", [])))
                subscribed_assets.update(str(x) for x in sub.get("assets_ids", []))
                await asyncio.sleep(0.1)
            first_logged = False
            async for msg in ws:
                recv_ts = utc_now()
                if online_refresh and recv_ts >= next_refresh:
                    try:
                        await asyncio.to_thread(rt.refresh_live_market_meta, cfg)
                        wanted_assets = pm_active_asset_ids(rt, cfg)
                        new_assets = sorted(wanted_assets - subscribed_assets)
                        if new_assets:
                            channel = (
                                _get_nested(cfg, "feeds", "polymarket", "market_channel", default=None)
                                or cfg["polymarket"].get("market_channel")
                                or "market"
                            )
                            batch_size = int(pm_feed_cfg.get("subscribe_batch_size", 20))
                            batch_size = max(1, min(batch_size, 200))
                            for idx in range(0, len(new_assets), batch_size):
                                sub = {"type": channel, "assets_ids": new_assets[idx : idx + batch_size], "custom_feature_enabled": True}
                                await ws.send(orjson.dumps(sub).decode("utf-8"))
                                LOGGER.info("Sent Polymarket incremental subscription with %s new asset(s)", len(sub["assets_ids"]))
                                await asyncio.sleep(0.1)
                            subscribed_assets.update(new_assets)
                    except Exception as exc:
                        LOGGER.warning("Online Polymarket metadata refresh failed: %s", exc)
                    finally:
                        next_refresh = utc_now() + timedelta(seconds=max(30.0, refresh_interval_s))
                if isinstance(msg, str) and msg.strip().upper() == "PONG":
                    continue
                try:
                    raw = orjson.loads(msg)
                except Exception as exc:
                    LOGGER.warning("Failed to decode Polymarket websocket message: %s", exc)
                    continue
                if not first_logged:
                    preview = msg[:500] if isinstance(msg, str) else str(msg)[:500]
                    LOGGER.info("First Polymarket websocket message preview: %s", preview)
                    first_logged = True
                payload = raw.get("data", raw) if isinstance(raw, dict) else raw
                before_count = rt.stats["quote_snapshot_count"]
                if isinstance(payload, dict) and isinstance(payload.get("price_changes"), list):
                    base = {
                        "market_id": payload.get("market_id") or payload.get("market"),
                        "timestamp": payload.get("timestamp") or payload.get("ts_event"),
                    }
                    for change in payload["price_changes"]:
                        event = {**base, **change}
                        parsed = parse_pm_event(event)
                        if parsed:
                            rt.update_pm_quote(parsed, recv_ts)
                elif isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        parsed = parse_pm_event(item)
                        if parsed:
                            rt.update_pm_quote(parsed, recv_ts)
                elif isinstance(payload, dict):
                    parsed = parse_pm_event(payload)
                    if parsed:
                        rt.update_pm_quote(parsed, recv_ts)
                if rt.stats["quote_snapshot_count"] > before_count and rt.stats["quote_snapshot_count"] <= 5:
                    LOGGER.info("Parsed Polymarket quote snapshot count=%s", rt.stats["quote_snapshot_count"])
                if utc_now() >= rt.stop_wall:
                    break
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)


def run_replay_windows(rt: ShadowRuntime, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if not args.replay_windows:
        raise RuntimeError("replay mode requires --replay-windows")
    silver_path = args.replay_source_silver or cfg["market_meta"]["silver_pm_path"]
    cols = [
        "market_id",
        "sample_ts",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "yes_spread",
        "yes_quote_age_seconds",
        "yes_is_stale",
        "yes_last_quote_update_ts",
        "yes_crossed_quote",
        "no_bid",
        "no_ask",
        "no_mid",
        "no_spread",
        "no_quote_age_seconds",
        "no_is_stale",
        "no_last_quote_update_ts",
        "no_crossed_quote",
        "seconds_since_last_pm_update",
        "time_to_expiry_seconds",
        "date",
    ]
    silver = scan_dataset(silver_path, cols)
    silver_by_market: dict[str, pl.DataFrame] = {}
    for market_id, part in silver.partition_by("market_id", as_dict=True).items():
        silver_by_market[str(market_id[0] if isinstance(market_id, tuple) else market_id)] = part.sort("sample_ts")

    for window in rt.replay_windows:
        part = silver_by_market.get(window.market_id)
        if part is None or part.is_empty():
            LOGGER.warning("replay window %s market %s missing in silver", window.window_id, window.market_id)
            continue
        rt.last_signal_ts = {k: v for k, v in rt.last_signal_ts.items() if k[1] != window.market_id}
        st = rt.market_states[window.market_id]
        st.pm_mid_hist.clear()
        st.last_pm_update_ts = None
        st.last_sample_ts = None
        rows = part.filter((pl.col("sample_ts") >= window.start_ts) & (pl.col("sample_ts") <= window.end_ts)).sort("sample_ts")
        if rows.is_empty():
            continue
        forced_done = False
        for row in rows.to_dicts():
            ts = row["sample_ts"]
            rt._update_market_state_from_row(row)
            if ts >= window.center_sample_ts:
                rt._maybe_emit_signal(window.market_id, ts, window.window_id)
            if args.force_signals_from_windows and not forced_done and ts == window.center_sample_ts:
                rt._force_window_signal(window, ts)
                forced_done = True
            rt._advance_pending(window.market_id, ts)
            if args.replay_speed and args.replay_speed > 0:
                time.sleep(args.replay_speed)
        # Flush any overdue pending using final quote snapshot timestamp.
        if rows.height:
            end_ts = rows[-1, "sample_ts"]
            rt._advance_pending(window.market_id, end_ts + timedelta(seconds=35))


def run_dry_replay(rt: ShadowRuntime, cfg: dict[str, Any]) -> None:
    """Fallback smoke test when replay windows are not provided."""
    gold_cols = list(dict.fromkeys(rt.feature_names + ["market_id", "sample_ts", "yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask", "no_mid", "yes_spread", "time_to_expiry_seconds", "seconds_since_last_pm_update", "split", "date"]))
    gold = scan_dataset(cfg["dry_run"]["historical_gold_path"], gold_cols).sort("sample_ts").tail(int(cfg["dry_run"].get("max_rows", 2000)))
    if gold.is_empty():
        LOGGER.warning("dry-run: no historical gold rows found")
        return
    for row in gold.to_dicts():
        market_id = str(row["market_id"])
        if market_id not in rt.market_states:
            continue
        recv_ts = row["sample_ts"]
        rt._update_market_state_from_row(
            {
                "market_id": market_id,
                "sample_ts": recv_ts,
                "yes_bid": row.get("yes_bid"),
                "yes_ask": row.get("yes_ask"),
                "yes_mid": row.get("yes_mid"),
                "yes_quote_age_seconds": 0.0,
                "yes_is_stale": False,
                "no_bid": row.get("no_bid"),
                "no_ask": row.get("no_ask"),
                "no_mid": row.get("no_mid"),
                "no_quote_age_seconds": 0.0,
                "no_is_stale": False,
                "seconds_since_last_pm_update": row.get("seconds_since_last_pm_update"),
                "date": row.get("date") or str(recv_ts.date()),
            }
        )
        rt._maybe_emit_signal(market_id, recv_ts)
        rt._advance_pending(market_id, recv_ts)


async def _resilient_feed_task(name: str, coro_factory: Any, rt: ShadowRuntime, cfg: dict[str, Any]) -> None:
    """Run a live feed until stop_wall, reconnecting if the websocket task exits early.

    In shadow mode we prefer data continuity over failing the whole process. A feed
    task can return if the remote websocket closes cleanly; without this wrapper the
    other feed keeps running and systemd/health checks see an active process while
    one side of the market data is stale.
    """
    attempt = 0
    while utc_now() < rt.stop_wall:
        try:
            attempt += 1
            if attempt > 1:
                LOGGER.warning("%s feed restarting, attempt=%s", name, attempt)
            await coro_factory(rt, cfg)
            if utc_now() < rt.stop_wall:
                LOGGER.warning("%s feed exited before stop_wall; reconnecting", name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.warning("%s feed failed (%s: %s); reconnecting", name, type(exc).__name__, exc)
        await asyncio.sleep(min(30.0, 2.0 * attempt))


async def live_main(rt: ShadowRuntime, cfg: dict[str, Any]) -> None:
    tasks = [
        asyncio.create_task(_resilient_feed_task("binance", binance_task, rt, cfg)),
        asyncio.create_task(_resilient_feed_task("polymarket", polymarket_task, rt, cfg)),
    ]
    flush_interval = max(1, int(cfg["output"].get("flush_interval_seconds", 10)))
    try:
        while utc_now() < rt.stop_wall:
            await asyncio.sleep(1.0)
            if int(time.time()) % flush_interval == 0:
                rt.pm_quote_sink.flush()
                rt.binance_sink.flush()
                rt.signals_sink.flush()
                rt.outcomes_sink.flush()
                rt.latency_sink.flush()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = normalize_shadow_config(read_yaml(args.config))
    validate_shadow_config(cfg)
    print(SHADOW_BANNER)
    rt = ShadowRuntime(cfg, args)
    if args.replay_windows:
        LOGGER.info("Running replay-window shadow mode")
        run_replay_windows(rt, args, cfg)
    elif args.dry_run:
        LOGGER.info("Running shadow logger in DRY RUN mode for smoke test only")
        run_dry_replay(rt, cfg)
    else:
        asyncio.run(live_main(rt, cfg))
    rt.finalize()


if __name__ == "__main__":
    main()
