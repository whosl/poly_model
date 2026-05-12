from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import pyarrow as pa
import pyarrow.dataset as ds

from .config import resolve_path
from .io_utils import JsonRecordReader, RawReadStatus, ensure_dir
from .reporting import markdown_table, write_markdown
from .schema_utils import first_non_null, infer_record_type

logger = logging.getLogger(__name__)


def parse_timestamp_to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdigit():
            value = int(value)
        else:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.astimezone(UTC)
            except ValueError:
                return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        abs_numeric = abs(numeric)
        if abs_numeric > 10**18:
            return datetime.fromtimestamp(numeric / 1_000_000_000, tz=UTC)
        if abs_numeric > 10**15:
            return datetime.fromtimestamp(numeric / 1_000_000, tz=UTC)
        if abs_numeric > 10**12:
            return datetime.fromtimestamp(numeric / 1_000, tz=UTC)
        return datetime.fromtimestamp(numeric, tz=UTC)
    return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def infer_symbol_from_stream(stream: str | None) -> str | None:
    if not stream:
        return None
    return stream.split("@", 1)[0].upper()


def normalize_outcome(value: Any, asset_id: str | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"YES", "NO"}:
        return text
    if asset_id:
        return None
    return None


def compute_book_metrics(best_bid: float | None, best_ask: float | None, bid_qty: float | None, ask_qty: float | None) -> tuple[float | None, float | None, float | None]:
    mid = None
    spread = None
    microprice = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
    if None not in {best_bid, best_ask, bid_qty, ask_qty} and (bid_qty or 0) + (ask_qty or 0) > 0:
        microprice = ((best_ask * bid_qty) + (best_bid * ask_qty)) / (bid_qty + ask_qty)
    return mid, spread, microprice


def compute_depth_metrics(levels: list[tuple[float | None, float | None]], top_n: int) -> tuple[float | None, float | None, float | None]:
    bids = [qty for _, qty in levels[:top_n] if qty is not None]
    asks = [qty for _, qty in levels[top_n : top_n * 2] if qty is not None]
    bid_sum = sum(bids) if bids else None
    ask_sum = sum(asks) if asks else None
    imbalance = None
    if bid_sum is not None and ask_sum is not None and (bid_sum + ask_sum) > 0:
        imbalance = (bid_sum - ask_sum) / (bid_sum + ask_sum)
    return bid_sum, ask_sum, imbalance


def _date_from_path(path: Path) -> str | None:
    for part in path.parts:
        if len(part) == 8 and part.isdigit():
            return part
    return None


def iter_raw_files(
    config: dict,
    limit_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = None,
) -> list[Path]:
    roots = [
        resolve_path(config, config["raw_paths"]["binance"]),
        resolve_path(config, config["raw_paths"]["polymarket"]),
    ]
    if source == "binance":
        roots = [roots[0]]
    elif source == "polymarket":
        roots = [roots[1]]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.jsonl.gz")))
        files.extend(sorted(root.rglob("*.json.gz")))
        files.extend(sorted(root.rglob("*.jsonl")))
    files = sorted(set(files))
    if start_date or end_date:
        filtered = []
        for path in files:
            date_part = _date_from_path(path)
            if date_part is None:
                continue
            if start_date and date_part < start_date:
                continue
            if end_date and date_part > end_date:
                continue
            filtered.append(path)
        files = filtered
    if limit_files is not None:
        return files[:limit_files]
    return files


def extract_payload(record: dict, mapping: dict[str, Any], candidates_key: str) -> Any:
    return first_non_null(record, mapping.get(candidates_key, []))


def normalize_binance_bookticker(record: dict, config: dict, source_file: str) -> dict[str, Any] | None:
    mapping = config["schema_mapping"]["binance"]
    raw = extract_payload(record, mapping, "payload_field_candidates")
    if not isinstance(raw, dict):
        return None
    symbol = first_non_null(raw, mapping["symbol_candidates"]) or infer_symbol_from_stream(first_non_null(record, mapping["stream_field_candidates"]))
    bid_price = to_float(first_non_null(raw, mapping["bookticker"]["bid_price_candidates"]))
    bid_qty = to_float(first_non_null(raw, mapping["bookticker"]["bid_qty_candidates"]))
    ask_price = to_float(first_non_null(raw, mapping["bookticker"]["ask_price_candidates"]))
    ask_qty = to_float(first_non_null(raw, mapping["bookticker"]["ask_qty_candidates"]))
    ts_recv = parse_timestamp_to_utc(first_non_null(record, mapping["recv_ts_candidates"]))
    ts_event = parse_timestamp_to_utc(first_non_null(raw, mapping["event_ts_candidates"])) or ts_recv
    mid, spread, microprice = compute_book_metrics(bid_price, ask_price, bid_qty, ask_qty)
    return {
        "ts_event": ts_event,
        "ts_recv": ts_recv or ts_event,
        "symbol": symbol,
        "bid_price": bid_price,
        "bid_qty": bid_qty,
        "ask_price": ask_price,
        "ask_qty": ask_qty,
        "mid_price": mid,
        "spread": spread,
        "microprice": microprice,
        "source_file": source_file,
    }


def normalize_binance_aggtrade(record: dict, config: dict, source_file: str) -> dict[str, Any] | None:
    mapping = config["schema_mapping"]["binance"]
    raw = extract_payload(record, mapping, "payload_field_candidates")
    if not isinstance(raw, dict):
        return None
    symbol = first_non_null(raw, mapping["symbol_candidates"]) or infer_symbol_from_stream(first_non_null(record, mapping["stream_field_candidates"]))
    price = to_float(first_non_null(raw, mapping["aggtrade"]["price_candidates"]))
    qty = to_float(first_non_null(raw, mapping["aggtrade"]["qty_candidates"]))
    is_buyer_maker = to_bool(first_non_null(raw, mapping["aggtrade"]["buyer_maker_candidates"]))
    ts_recv = parse_timestamp_to_utc(first_non_null(record, mapping["recv_ts_candidates"]))
    ts_event = parse_timestamp_to_utc(first_non_null(raw, mapping["event_ts_candidates"])) or ts_recv
    side = None
    if is_buyer_maker is not None:
        side = "sell_aggressor" if is_buyer_maker else "buy_aggressor"
    return {
        "ts_event": ts_event,
        "ts_recv": ts_recv or ts_event,
        "symbol": symbol,
        "price": price,
        "qty": qty,
        "is_buyer_maker": is_buyer_maker,
        "side": side,
        "notional": price * qty if price is not None and qty is not None else None,
        "source_file": source_file,
    }


def _extract_price_levels(raw_levels: Any, depth_levels: int | None) -> list[tuple[float | None, float | None]]:
    rows: list[tuple[float | None, float | None]] = []
    if not isinstance(raw_levels, list):
        return rows
    iterable = raw_levels if depth_levels is None else raw_levels[:depth_levels]
    for entry in iterable:
        if isinstance(entry, dict):
            rows.append((to_float(entry.get("price")), to_float(entry.get("size"))))
        elif isinstance(entry, list) and len(entry) >= 2:
            rows.append((to_float(entry[0]), to_float(entry[1])))
    return rows


def sort_pm_orderbook_levels(raw_levels: Any, side: str, depth_levels: int = 5) -> list[tuple[float | None, float | None]]:
    """Return Polymarket levels sorted to true best-first order.

    Raw Polymarket orderbook arrays are not trusted to be sorted.  We parse all
    supplied prices/sizes as floats, drop levels without prices, then sort bids
    high-to-low and asks low-to-high before taking top N.
    """
    levels = [(px, qty) for px, qty in _extract_price_levels(raw_levels, None) if px is not None]
    reverse = side.lower() == "bid"
    return sorted(levels, key=lambda item: item[0], reverse=reverse)[:depth_levels]


def normalize_binance_depth(record: dict, config: dict, source_file: str) -> dict[str, Any] | None:
    mapping = config["schema_mapping"]["binance"]
    raw = extract_payload(record, mapping, "payload_field_candidates")
    if not isinstance(raw, dict):
        return None
    depth_levels = int(config["features"]["depth_levels"])
    bids = _extract_price_levels(first_non_null(raw, mapping["depth"]["bids_candidates"]), depth_levels)
    asks = _extract_price_levels(first_non_null(raw, mapping["depth"]["asks_candidates"]), depth_levels)
    symbol = first_non_null(raw, mapping["symbol_candidates"]) or infer_symbol_from_stream(first_non_null(record, mapping["stream_field_candidates"]))
    ts_recv = parse_timestamp_to_utc(first_non_null(record, mapping["recv_ts_candidates"]))
    ts_event = parse_timestamp_to_utc(first_non_null(raw, mapping["event_ts_candidates"])) or ts_recv
    row: dict[str, Any] = {
        "ts_event": ts_event,
        "ts_recv": ts_recv or ts_event,
        "symbol": symbol,
        "source_file": source_file,
    }
    bid_sum = 0.0
    ask_sum = 0.0
    bid_seen = False
    ask_seen = False
    for idx in range(depth_levels):
        bid_px, bid_qty = bids[idx] if idx < len(bids) else (None, None)
        ask_px, ask_qty = asks[idx] if idx < len(asks) else (None, None)
        row[f"bid_px_{idx + 1}"] = bid_px
        row[f"bid_qty_{idx + 1}"] = bid_qty
        row[f"ask_px_{idx + 1}"] = ask_px
        row[f"ask_qty_{idx + 1}"] = ask_qty
        if bid_qty is not None:
            bid_sum += bid_qty
            bid_seen = True
        if ask_qty is not None:
            ask_sum += ask_qty
            ask_seen = True
    row[f"bid_depth_{depth_levels}"] = bid_sum if bid_seen else None
    row[f"ask_depth_{depth_levels}"] = ask_sum if ask_seen else None
    if bid_seen and ask_seen and (bid_sum + ask_sum) > 0:
        row[f"depth_imbalance_{depth_levels}"] = (bid_sum - ask_sum) / (bid_sum + ask_sum)
    else:
        row[f"depth_imbalance_{depth_levels}"] = None
    return row


def _normalize_pm_book_row(raw: dict, record: dict, config: dict, source_file: str) -> dict[str, Any] | None:
    mapping = config["schema_mapping"]["polymarket"]
    market_id = first_non_null(raw, mapping["market_id_candidates"])
    asset_id = first_non_null(raw, mapping["asset_id_candidates"])
    outcome = normalize_outcome(first_non_null(raw, mapping["outcome_candidates"]), asset_id=asset_id)
    ts_event = parse_timestamp_to_utc(first_non_null(raw, mapping["event_ts_candidates"]))
    ts_recv = parse_timestamp_to_utc(first_non_null(record, mapping["recv_ts_candidates"])) or ts_event
    bids = sort_pm_orderbook_levels(first_non_null(raw, mapping["orderbook_bids_candidates"]), "bid", 5)
    asks = sort_pm_orderbook_levels(first_non_null(raw, mapping["orderbook_asks_candidates"]), "ask", 5)
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    bid_size_1 = bids[0][1] if bids else None
    ask_size_1 = asks[0][1] if asks else None
    mid, spread, _ = compute_book_metrics(best_bid, best_ask, bid_size_1, ask_size_1)
    bid_depth_3 = sum(qty for _, qty in bids[:3] if qty is not None) if bids else None
    ask_depth_3 = sum(qty for _, qty in asks[:3] if qty is not None) if asks else None
    bid_depth_5 = sum(qty for _, qty in bids[:5] if qty is not None) if bids else None
    ask_depth_5 = sum(qty for _, qty in asks[:5] if qty is not None) if asks else None

    def imbalance(bid_depth: float | None, ask_depth: float | None) -> float | None:
        if bid_depth is None or ask_depth is None or (bid_depth + ask_depth) <= 0:
            return None
        return (bid_depth - ask_depth) / (bid_depth + ask_depth)

    return {
        "ts_event": ts_event,
        "ts_recv": ts_recv,
        "market_id": market_id,
        "asset_id": asset_id,
        "outcome": outcome,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "bid_size_1": bid_size_1,
        "ask_size_1": ask_size_1,
        "bid_depth_3": bid_depth_3,
        "ask_depth_3": ask_depth_3,
        "bid_depth_5": bid_depth_5,
        "ask_depth_5": ask_depth_5,
        "depth_imbalance_3": imbalance(bid_depth_3, ask_depth_3),
        "depth_imbalance_5": imbalance(bid_depth_5, ask_depth_5),
        "source_file": source_file,
    }


def normalize_pm_orderbook(record: dict, config: dict, source_file: str) -> list[dict[str, Any]]:
    raw = record.get("raw", record)
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = [raw]
    else:
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = _normalize_pm_book_row(row, record, config, source_file)
        if item:
            normalized.append(item)
    return normalized


def normalize_pm_price_change(record: dict, config: dict, source_file: str) -> list[dict[str, Any]]:
    mapping = config["schema_mapping"]["polymarket"]
    raw = record.get("raw", record)
    if not isinstance(raw, dict):
        return []
    market_id = first_non_null(raw, mapping["market_id_candidates"])
    ts_event = parse_timestamp_to_utc(first_non_null(raw, mapping["event_ts_candidates"]))
    ts_recv = parse_timestamp_to_utc(first_non_null(record, mapping["recv_ts_candidates"])) or ts_event
    event_type = raw.get("event_type")
    changes = first_non_null(raw, mapping["price_change_array_candidates"])
    if isinstance(changes, list):
        rows = changes
    else:
        rows = [raw]
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset_id = first_non_null(row, mapping["asset_id_candidates"])
        outcome = normalize_outcome(first_non_null(row, mapping["outcome_candidates"]), asset_id=asset_id)
        normalized.append(
            {
                "ts_event": ts_event or parse_timestamp_to_utc(first_non_null(row, mapping["event_ts_candidates"])),
                "ts_recv": ts_recv,
                "market_id": market_id or first_non_null(row, mapping["market_id_candidates"]),
                "asset_id": asset_id,
                "outcome": outcome,
                "price": to_float(row.get("price")),
                "best_bid": to_float(row.get("best_bid")),
                "best_ask": to_float(row.get("best_ask")),
                "side": row.get("side"),
                "event_type": event_type or row.get("event_type"),
                "source_file": source_file,
            }
        )
    return normalized


@dataclass
class BronzeTarget:
    dataset_name: str
    root_path: Path
    partition_cols: list[str]


@dataclass
class BronzeFileResult:
    source_file: str
    counts: dict[str, int]
    read_status: RawReadStatus


class BronzeWriter:
    def __init__(self, config: dict) -> None:
        self.config = config
        bronze_root = resolve_path(config, config["output_paths"]["bronze"])
        ensure_dir(bronze_root)
        self.targets = {
            "binance_bookticker": BronzeTarget("binance_bookticker", bronze_root / "binance_bookticker", ["date", "symbol"]),
            "binance_aggtrade": BronzeTarget("binance_aggtrade", bronze_root / "binance_aggtrade", ["date", "symbol"]),
            "binance_depth": BronzeTarget("binance_depth", bronze_root / "binance_depth", ["date", "symbol"]),
            "pm_orderbook": BronzeTarget("pm_orderbook", bronze_root / "pm_orderbook", ["date", "market_id"]),
            "pm_price_change": BronzeTarget("pm_price_change", bronze_root / "pm_price_change", ["date", "market_id"]),
            "pm_market_meta": BronzeTarget("pm_market_meta", bronze_root / "pm_market_meta", ["date", "market_id"]),
        }

    def write_rows(self, dataset_name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        target = self.targets[dataset_name]
        enriched = []
        for row in rows:
            ts_event = row.get("ts_event")
            if ts_event is None:
                continue
            row = dict(row)
            row["date"] = ts_event.date().isoformat()
            enriched.append(row)
        if not enriched:
            return
        table = pa.Table.from_pylist(enriched)
        ds.write_dataset(
            data=table,
            base_dir=str(target.root_path),
            format="parquet",
            partitioning=ds.partitioning(
                schema=pa.schema([pa.field(col, table.schema.field(col).type) for col in target.partition_cols]),
                flavor="hive",
            ),
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"{uuid4().hex}-{{i}}.parquet",
            file_options=ds.ParquetFileFormat().make_write_options(
                compression=self.config["io"]["parquet_compression"],
            ),
        )


def build_bronze_for_file(path: Path, config: dict, writer: BronzeWriter) -> BronzeFileResult:
    counters: dict[str, int] = defaultdict(int)
    dedupe_keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch_size = int(config["io"]["bronze_batch_size"])
    source_file = str(path)
    reader = JsonRecordReader(path)

    def flush(dataset_name: str) -> None:
        batch = batches[dataset_name]
        if not batch:
            return
        batch.sort(key=lambda row: row.get("ts_event") or datetime.min.replace(tzinfo=UTC))
        writer.write_rows(dataset_name, batch)
        batch.clear()
        dedupe_keys[dataset_name].clear()

    for record in reader:
        record_type = infer_record_type(record)
        normalized_rows: list[dict[str, Any]] = []
        if record_type == "binance_bookticker":
            row = normalize_binance_bookticker(record, config, source_file)
            normalized_rows = [row] if row else []
        elif record_type == "binance_aggtrade":
            row = normalize_binance_aggtrade(record, config, source_file)
            normalized_rows = [row] if row else []
        elif record_type == "binance_depth":
            row = normalize_binance_depth(record, config, source_file)
            normalized_rows = [row] if row else []
        elif record_type == "pm_orderbook":
            normalized_rows = normalize_pm_orderbook(record, config, source_file)
        elif record_type.startswith("pm_"):
            normalized_rows = normalize_pm_price_change(record, config, source_file)
            record_type = "pm_price_change"
        else:
            logger.warning("Skipping unknown record type in %s", path)
            continue

        if record_type not in writer.targets:
            continue
        for row in normalized_rows:
            row_key = tuple(sorted((key, str(value)) for key, value in row.items() if key != "source_file"))
            if row_key in dedupe_keys[record_type]:
                continue
            dedupe_keys[record_type].add(row_key)
            batches[record_type].append(row)
            counters[record_type] += 1
            if len(batches[record_type]) >= batch_size:
                flush(record_type)

    for dataset_name in list(batches):
        flush(dataset_name)
    return BronzeFileResult(source_file=source_file, counts=dict(counters), read_status=reader.status)


def write_corrupt_raw_report(config: dict, results: list[BronzeFileResult]) -> Path:
    reports_root = resolve_path(config, config["output_paths"]["reports"])
    path = reports_root / "corrupt_raw_files.md"
    corrupt = [result for result in results if result.read_status.is_corrupt]
    lines = ["# Corrupt Raw Files", ""]
    if not corrupt:
        lines.append("No corrupted or truncated raw files were encountered in the current run.")
        write_markdown(path, lines)
        return path

    rows: list[list[Any]] = []
    for result in corrupt:
        rows.append(
            [
                result.source_file,
                result.read_status.error_type,
                result.read_status.readable_rows,
                result.read_status.failed_at_line,
                result.read_status.prefix_retained,
                sum(result.counts.values()),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "source_file",
                "error_type",
                "readable_rows",
                "failed_at_line",
                "prefix_retained",
                "output_parquet_rows",
            ],
            rows,
        )
    )
    write_markdown(path, lines)
    return path


def write_bronze_quality_report(config: dict, results: list[BronzeFileResult]) -> Path:
    reports_root = resolve_path(config, config["output_paths"]["reports"])
    path = reports_root / "bronze_quality_report.md"
    lines = ["# Bronze Quality Report", ""]
    if not results:
        lines.append("No raw files were processed.")
        write_markdown(path, lines)
        return path

    total_counts: dict[str, int] = defaultdict(int)
    corrupt_files = 0
    processed_rows = 0
    for result in results:
        processed_rows += sum(result.counts.values())
        if result.read_status.is_corrupt:
            corrupt_files += 1
        for key, value in result.counts.items():
            total_counts[key] += value

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Processed files: `{len(results)}`")
    lines.append(f"- Corrupt/truncated files in current run: `{corrupt_files}`")
    lines.append(f"- Total output rows: `{processed_rows}`")
    lines.append("")
    lines.append("## Dataset Row Counts")
    lines.append("")
    for key, value in sorted(total_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Per File Output")
    lines.append("")
    table_rows = []
    for result in results:
        table_rows.append(
            [
                result.source_file,
                result.read_status.readable_rows,
                result.read_status.error_type or "",
                result.counts.get("binance_bookticker", 0),
                result.counts.get("binance_aggtrade", 0),
                result.counts.get("binance_depth", 0),
                result.counts.get("pm_orderbook", 0),
                result.counts.get("pm_price_change", 0),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "source_file",
                "readable_rows",
                "error_type",
                "binance_bookticker",
                "binance_aggtrade",
                "binance_depth",
                "pm_orderbook",
                "pm_price_change",
            ],
            table_rows,
        )
    )
    write_markdown(path, lines)
    return path
