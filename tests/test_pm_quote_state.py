from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from preprocess.pm_quote_state import build_asset_quote_state, build_asset_state_on_grid, normalize_quote_events
from preprocess.bronze import sort_pm_orderbook_levels


def dt(s: int) -> datetime:
    return datetime(2026, 1, 1, 0, 0, s, tzinfo=UTC)


def grid_at(sec: int) -> pl.DataFrame:
    return pl.DataFrame({"sample_ts": [dt(sec)]})


def test_price_change_updates_quote_state() -> None:
    ob = pl.DataFrame({"market_id": ["m"], "asset_id": ["YES"], "ts_event": [dt(0)], "ts_recv": [dt(0)], "best_bid": [0.01], "best_ask": [0.99], "bid_size_1": [1.0], "ask_size_1": [1.0], "source_file": ["ob"]})
    pc = pl.DataFrame({"market_id": ["m"], "asset_id": ["YES"], "ts_event": [dt(1)], "ts_recv": [dt(1)], "best_bid": [0.48], "best_ask": [0.49], "price": [0.48], "event_type": ["price_change"], "source_file": ["pc"]})
    state = build_asset_quote_state(normalize_quote_events(ob, pc))
    out = build_asset_state_on_grid(grid_at(2), state, "YES", "yes", 60)
    assert out["yes_bid"][0] == 0.48
    assert out["yes_ask"][0] == 0.49
    assert out["yes_mid"][0] == 0.485


def test_no_mechanical_complement_when_only_yes_updates() -> None:
    pc = pl.DataFrame({"market_id": ["m"], "asset_id": ["YES"], "ts_event": [dt(1)], "ts_recv": [dt(1)], "best_bid": [0.48], "best_ask": [0.49], "price": [None], "event_type": ["price_change"], "source_file": ["pc"]})
    state = build_asset_quote_state(normalize_quote_events(pl.DataFrame(), pc))
    out = build_asset_state_on_grid(grid_at(2), state, "NO", "no", 60)
    assert out["no_bid"][0] is None
    assert out["no_ask"][0] is None
    assert out["no_mid"][0] is None


def test_missing_bid_ask_does_not_fill_mid_0p5() -> None:
    pc = pl.DataFrame({"market_id": ["m"], "asset_id": ["YES"], "ts_event": [dt(1)], "ts_recv": [dt(1)], "price": [0.5], "event_type": ["price_change"], "source_file": ["pc"]})
    state = build_asset_quote_state(normalize_quote_events(pl.DataFrame(), pc))
    out = build_asset_state_on_grid(grid_at(2), state, "YES", "yes", 60)
    assert out["yes_bid"][0] is None
    assert out["yes_ask"][0] is None
    assert out["yes_mid"][0] is None


def test_asof_does_not_use_future_quote() -> None:
    pc = pl.DataFrame({"market_id": ["m"], "asset_id": ["YES"], "ts_event": [dt(3)], "ts_recv": [dt(3)], "best_bid": [0.48], "best_ask": [0.49], "price": [None], "event_type": ["price_change"], "source_file": ["pc"]})
    state = build_asset_quote_state(normalize_quote_events(pl.DataFrame(), pc))
    out = build_asset_state_on_grid(grid_at(2), state, "YES", "yes", 60)
    assert out["yes_bid"][0] is None
    assert out["yes_mid"][0] is None


def test_unsorted_orderbook_levels_helper() -> None:
    bids = [["0.10", "1"], ["0.40", "4"], ["0.25", "2"]]
    asks = [["0.90", "1"], ["0.45", "3"], ["0.50", "2"]]
    assert sort_pm_orderbook_levels(bids, "bid", 5)[0] == (0.40, 4.0)
    assert sort_pm_orderbook_levels(asks, "ask", 5)[0] == (0.45, 3.0)
