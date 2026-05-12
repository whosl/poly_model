from __future__ import annotations

from datetime import UTC, datetime

from preprocess.bronze import compute_book_metrics, sort_pm_orderbook_levels


def test_pm_orderbook_best_quote_from_unsorted_levels() -> None:
    bids = [{"price": "0.20", "size": "1"}, {"price": "0.48", "size": "7"}, {"price": "0.31", "size": "3"}]
    asks = [{"price": "0.70", "size": "5"}, {"price": "0.49", "size": "11"}, {"price": "0.55", "size": "2"}]
    bid_levels = sort_pm_orderbook_levels(bids, "bid", 5)
    ask_levels = sort_pm_orderbook_levels(asks, "ask", 5)
    best_bid, bid_size_1 = bid_levels[0]
    best_ask, ask_size_1 = ask_levels[0]
    mid, spread, _ = compute_book_metrics(best_bid, best_ask, bid_size_1, ask_size_1)
    assert best_bid == 0.48
    assert bid_size_1 == 7.0
    assert best_ask == 0.49
    assert ask_size_1 == 11.0
    assert mid == 0.485
    assert spread == 0.010000000000000009
