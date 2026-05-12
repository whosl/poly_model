from __future__ import annotations

from preprocess.feature_metadata import is_feature_column


def test_feature_list_excludes_future_and_labels() -> None:
    assert not is_feature_column("future_mid_1s")
    assert not is_feature_column("markout_1s")
    assert not is_feature_column("label_1s")
    assert not is_feature_column("settled_yes")
    assert not is_feature_column("btc_close_price")
    assert is_feature_column("mid_price")

