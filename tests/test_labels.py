from __future__ import annotations

from preprocess.gold_utils import compute_settled_yes_proxy


def classify(ret: float, threshold: float = 0.0001) -> str:
    if ret > threshold:
        return "UP"
    if ret < -threshold:
        return "DOWN"
    return "FLAT"


def test_btc_direction_label_logic() -> None:
    assert classify(0.001) == "UP"
    assert classify(-0.001) == "DOWN"
    assert classify(0.0) == "FLAT"


def test_pm_markout_label_logic() -> None:
    assert classify(0.02, threshold=0.01) == "UP"
    assert classify(-0.02, threshold=0.01) == "DOWN"
    assert classify(0.005, threshold=0.01) == "FLAT"


def test_terminal_settled_yes_proxy() -> None:
    assert compute_settled_yes_proxy(100.0, 101.0) == 1
    assert compute_settled_yes_proxy(100.0, 99.0) == 0

