from __future__ import annotations

from datetime import datetime, UTC

import polars as pl


def test_asof_join_uses_past_only() -> None:
    left = pl.DataFrame(
        {
            "sample_ts": pl.datetime_range(
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                interval="1s",
                eager=True,
            )
        }
    )
    right = pl.DataFrame(
        {
            "ts_event": pl.Series(
                [datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC), datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)]
            ),
            "value": [10, 20],
        }
    )
    joined = left.join_asof(right, left_on="sample_ts", right_on="ts_event", strategy="backward")
    assert joined["value"].to_list() == [10, 20, 20]
