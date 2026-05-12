from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


def flatten_keys(obj: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(flatten_keys(value, path))
    elif isinstance(obj, list):
        if obj:
            keys.extend(flatten_keys(obj[0], f"{prefix}[]"))
    return keys


def get_nested(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if current is None:
            return None
        if part.endswith("[]"):
            base = part[:-2]
            if not isinstance(current, dict):
                return None
            current = current.get(base)
            if not isinstance(current, list) or not current:
                return None
            current = current[0]
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def first_non_null(obj: dict, candidates: Iterable[str]) -> Any:
    for candidate in candidates:
        value = get_nested(obj, candidate)
        if value is not None:
            return value
    return None


def infer_record_type(record: dict) -> str:
    raw = record.get("raw", record)
    if isinstance(raw, dict) and "stream" in raw:
        stream = str(raw["stream"]).lower()
        if "bookticker" in stream:
            return "binance_bookticker"
        if "aggtrade" in stream:
            return "binance_aggtrade"
        if "depth" in stream:
            return "binance_depth"
    if isinstance(raw, list):
        first = raw[0] if raw else {}
        if isinstance(first, dict) and {"bids", "asks"}.issubset(first.keys()):
            return "pm_orderbook"
    if isinstance(raw, dict):
        event_type = raw.get("event_type")
        if raw.get("price_changes") or event_type == "price_change":
            return "pm_price_change"
        if {"bids", "asks"}.issubset(raw.keys()):
            return "pm_orderbook"
        if event_type:
            return f"pm_{event_type}"
    return "unknown"


@dataclass
class FileSchemaSummary:
    path: str
    record_types: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    top_level_keys: set[str] = field(default_factory=set)
    nested_keys: set[str] = field(default_factory=set)
    timestamp_candidates: set[str] = field(default_factory=set)
    id_candidates: set[str] = field(default_factory=set)
    sample_values: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_record(self, record: dict) -> None:
        record_type = infer_record_type(record)
        self.record_types[record_type] += 1
        self.top_level_keys.update(record.keys())
        flattened = flatten_keys(record)
        self.nested_keys.update(flattened)
        for key in flattened:
            lowered = key.lower()
            value = get_nested(record, key)
            if any(token in lowered for token in ("time", "ts", "recv_ns", "timestamp")) or key.endswith((".E", ".T")):
                self.timestamp_candidates.add(key)
            if any(token in lowered for token in ("asset", "market", "symbol", "stream")) or key.endswith((".s", ".asset_id", ".market")):
                self.id_candidates.add(key)
            if len(self.sample_values[key]) < 3 and value is not None:
                self.sample_values[key].append(repr(value)[:120])
