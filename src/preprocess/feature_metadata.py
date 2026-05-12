from __future__ import annotations

from pathlib import Path

from .reporting import write_json


LEAKY_PREFIXES = ("future_", "markout_", "label_")
LEAKY_EXACT = {"settled_yes", "btc_close_price", "label_source", "split"}


def is_feature_column(column: str) -> bool:
    if column in LEAKY_EXACT:
        return False
    if column.startswith(LEAKY_PREFIXES):
        return False
    return True


def write_feature_and_label_lists(
    root: Path,
    feature_file: str,
    label_file: str,
    columns: list[str],
    label_columns: list[str],
    extra_exclude: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    excluded = set(label_columns)
    if extra_exclude:
        excluded.update(extra_exclude)
    features = [col for col in columns if is_feature_column(col) and col not in excluded]
    labels = [col for col in columns if col in label_columns]
    write_json(root / feature_file, features)
    write_json(root / label_file, labels)
    return features, labels
