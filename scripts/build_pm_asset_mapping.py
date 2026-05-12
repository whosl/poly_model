from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pyarrow.parquet as pq

from preprocess.asset_mapping import (
    build_mapping_record,
    generate_mapping_reports,
    resolve_mapping_paths,
    save_mapping_file,
)
from preprocess.config import load_config, resolve_path
from preprocess.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Polymarket YES/NO asset mapping without loading full bronze tables.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--scan-outcomes", action="store_true", help="Read parquet row data to look for explicit YES/NO outcomes; slow on large bronze outputs.")
    parser.add_argument("--no-gamma-api", action="store_true", help="Disable Polymarket Gamma API metadata lookup.")
    parser.add_argument("--no-clob-token-api", action="store_true", help="Disable Polymarket CLOB markets-by-token fallback lookup.")
    parser.add_argument("--gamma-batch-size", type=int, default=50, help="Number of condition_ids per Gamma API request.")
    parser.add_argument("--gamma-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--gamma-retries", type=int, default=0)
    parser.add_argument("--gamma-max-consecutive-errors", type=int, default=3)
    parser.add_argument("--clob-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--clob-retries", type=int, default=0)
    parser.add_argument("--clob-max-consecutive-errors", type=int, default=5)
    parser.add_argument("--clob-workers", type=int, default=8)
    parser.add_argument("--limit-markets", type=int, default=None, help="Limit discovered markets after sorting; useful for API smoke tests.")
    return parser.parse_args()


def hive_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def safe_read_columns(path: Path, columns: list[str]):
    try:
        meta = pq.read_metadata(path)
        available = set(meta.schema.names)
        wanted = [col for col in columns if col in available]
        if not wanted:
            return []
        table = pq.read_table(path, columns=wanted)
        return table.to_pylist()
    except Exception as exc:  # noqa: BLE001 - report and continue; one bad parquet should not kill mapping
        logger.warning("Failed to read %s: %s", path, exc)
        return []


def sample_asset_ids(path: Path, max_ids: int = 4) -> set[str]:
    try:
        pf = pq.ParquetFile(path)
        if "asset_id" not in pf.schema_arrow.names:
            return set()
        asset_ids: set[str] = set()
        for batch in pf.iter_batches(batch_size=256, columns=["asset_id"]):
            values = batch.column(0).to_pylist()
            asset_ids.update(str(value) for value in values if value is not None)
            if len(asset_ids) >= max_ids:
                break
        return asset_ids
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to sample asset_id from %s: %s", path, exc)
        return set()


def parquet_column_index(meta: pq.FileMetaData, column_name: str) -> int | None:
    names = meta.schema.names
    try:
        return names.index(column_name)
    except ValueError:
        return None


def file_has_non_null_outcome(meta: pq.FileMetaData) -> bool:
    idx = parquet_column_index(meta, "outcome")
    if idx is None:
        return False
    try:
        if str(meta.schema.column(idx).logical_type).lower() == "null":
            return False
    except Exception:
        pass
    total_rows = meta.num_rows
    if total_rows == 0:
        return False
    known_nulls = 0
    saw_stats = False
    for rg_idx in range(meta.num_row_groups):
        stats = meta.row_group(rg_idx).column(idx).statistics
        if stats is None or stats.null_count is None:
            return True
        saw_stats = True
        known_nulls += stats.null_count
    return (not saw_stats) or known_nulls < total_rows


def metadata_min_ts(meta: pq.FileMetaData) -> Any:
    idx = parquet_column_index(meta, "ts_event")
    if idx is None:
        return None
    min_ts = None
    for rg_idx in range(meta.num_row_groups):
        stats = meta.row_group(rg_idx).column(idx).statistics
        if stats is None or stats.min is None:
            continue
        value = stats.min
        if min_ts is None or value < min_ts:
            min_ts = value
    return min_ts


def update_from_file(
    path: Path,
    known_market_ids: set[str],
    votes: dict[str, dict[str, set[str]]],
    first_ts_by_market: dict[str, Any],
    asset_ids_by_market: dict[str, set[str]],
) -> None:
    market_id = hive_value(path, "market_id")
    if market_id:
        known_market_ids.add(market_id)

    try:
        meta = pq.read_metadata(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read metadata %s: %s", path, exc)
        return

    if market_id:
        ts_min = metadata_min_ts(meta)
        if ts_min is not None:
            old = first_ts_by_market.get(market_id)
            if old is None or ts_min < old:
                first_ts_by_market[market_id] = ts_min
        if len(asset_ids_by_market.setdefault(market_id, set())) < 2:
            asset_ids_by_market[market_id].update(sample_asset_ids(path))

    if not file_has_non_null_outcome(meta):
        return

    rows = safe_read_columns(path, ["ts_event", "asset_id", "outcome"])
    for row in rows:
        row_market_id = market_id or row.get("market_id")
        if not row_market_id:
            continue
        known_market_ids.add(row_market_id)

        ts_event = row.get("ts_event")
        if ts_event is not None:
            old = first_ts_by_market.get(row_market_id)
            if old is None or ts_event < old:
                first_ts_by_market[row_market_id] = ts_event

        outcome = row.get("outcome")
        asset_id = row.get("asset_id")
        if asset_id is not None:
            asset_ids_by_market.setdefault(row_market_id, set()).add(str(asset_id))
        if outcome is None or asset_id is None:
            continue
        outcome_text = str(outcome).upper()
        if outcome_text not in {"YES", "NO"}:
            continue
        market_votes = votes.setdefault(row_market_id, {"YES": set(), "NO": set()})
        market_votes[outcome_text].add(str(asset_id))


def discover_markets_from_hive_dirs(roots: list[Path]) -> set[str]:
    market_ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for date_dir in root.glob("date=*"):
            if not date_dir.is_dir():
                continue
            for market_dir in date_dir.glob("market_id=*"):
                if market_dir.is_dir():
                    market_ids.add(market_dir.name.split("=", 1)[1])
    return market_ids


def list_pm_parquet_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            for date_dir in sorted(root.glob("date=*")):
                files.extend(sorted(date_dir.glob("market_id=*/*.parquet")))
    return files


def sample_bronze_metadata_and_assets(
    files: list[Path],
    known_market_ids: set[str],
    first_ts_by_market: dict[str, Any],
    asset_ids_by_market: dict[str, set[str]],
) -> int:
    scanned = 0
    remaining_for_assets = set(known_market_ids)
    completed_markets: set[str] = set()
    for idx, path in enumerate(files, start=1):
        market_id = hive_value(path, "market_id")
        if not market_id:
            continue
        if market_id in completed_markets:
            continue
        scanned += 1
        if idx % 5000 == 0:
            logger.info("Scanned metadata/sample asset_id from %d/%d PM parquet files", idx, len(files))
        try:
            meta = pq.read_metadata(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read metadata %s: %s", path, exc)
            continue
        ts_min = metadata_min_ts(meta)
        if ts_min is not None:
            old = first_ts_by_market.get(market_id)
            if old is None or ts_min < old:
                first_ts_by_market[market_id] = ts_min
        if market_id in remaining_for_assets:
            asset_ids_by_market.setdefault(market_id, set()).update(sample_asset_ids(path))
            if len(asset_ids_by_market[market_id]) >= 2:
                remaining_for_assets.discard(market_id)
        if market_id in first_ts_by_market and len(asset_ids_by_market.get(market_id, set())) >= 2:
            completed_markets.add(market_id)
        if len(completed_markets) >= len(known_market_ids):
            # We have at least one timestamp and two token IDs per market.
            break
    return scanned


def first_outcome_type_is_null(roots: list[Path]) -> bool:
    for root in roots:
        if not root.exists():
            continue
        first = None
        for date_dir in root.glob("date=*"):
            first = next(date_dir.glob("market_id=*/*.parquet"), None)
            if first is not None:
                break
        if first is None:
            continue
        try:
            meta = pq.read_metadata(first)
            idx = parquet_column_index(meta, "outcome")
            if idx is None:
                continue
            # Current bronze writes all-null outcomes as Arrow Null -> Parquet int32 Null.
            # If the first representative file has null-only outcome, scanning all row
            # data is not useful unless the user explicitly asks with --scan-outcomes.
            return not file_has_non_null_outcome(meta)
        except Exception:
            continue
    return False


def normalize_iso_ts(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        return text
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def parse_array_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


YES_OUTCOME_ALIASES = {"YES", "Y", "UP", "HIGHER", "ABOVE"}
NO_OUTCOME_ALIASES = {"NO", "N", "DOWN", "LOWER", "BELOW"}


def classify_outcome_label(value: Any) -> str | None:
    text = str(value).strip().upper()
    if text in YES_OUTCOME_ALIASES:
        return "YES"
    if text in NO_OUTCOME_ALIASES:
        return "NO"
    return None


def extract_gamma_token_pairs(market: dict[str, Any]) -> tuple[set[str], set[str], str | None]:
    yes_ids: set[str] = set()
    no_ids: set[str] = set()
    issue: str | None = None

    tokens = parse_array_field(market.get("tokens"))
    if tokens:
        for token in tokens:
            if not isinstance(token, dict):
                continue
            label = classify_outcome_label(token.get("outcome") or token.get("name"))
            token_id = token.get("token_id") or token.get("tokenId") or token.get("clobTokenId")
            if label == "YES" and token_id is not None:
                yes_ids.add(str(token_id))
            elif label == "NO" and token_id is not None:
                no_ids.add(str(token_id))

    outcomes = parse_array_field(market.get("outcomes"))
    clob_token_ids = parse_array_field(market.get("clobTokenIds") or market.get("clob_token_ids"))
    if outcomes or clob_token_ids:
        if len(outcomes) != len(clob_token_ids):
            issue = f"outcomes/clobTokenIds length mismatch: {len(outcomes)} vs {len(clob_token_ids)}"
        for outcome, token_id in zip(outcomes, clob_token_ids):
            label = classify_outcome_label(outcome)
            if label == "YES" and token_id is not None:
                yes_ids.add(str(token_id))
            elif label == "NO" and token_id is not None:
                no_ids.add(str(token_id))

    if (outcomes or tokens) and not (yes_ids and no_ids) and issue is None:
        issue = "could not classify binary YES/NO or UP/DOWN outcomes"
    return yes_ids, no_ids, issue


def gamma_market_id(market: dict[str, Any]) -> str | None:
    for key in ("conditionId", "condition_id", "conditionID"):
        value = market.get(key)
        if value:
            return str(value).lower()
    return None


def gamma_query_variants(condition_ids: list[str]) -> list[str]:
    # Different public examples of Gamma use either repeated params or a
    # comma-separated condition_ids value; try both so the script survives minor
    # gateway/parser differences.
    variants = [
        urllib.parse.urlencode({"condition_ids": condition_ids}, doseq=True),
        urllib.parse.urlencode({"condition_ids": ",".join(condition_ids)}),
    ]
    if len(condition_ids) == 1:
        variants.append(urllib.parse.urlencode({"condition_id": condition_ids[0]}))
    return list(dict.fromkeys(variants))


def fetch_gamma_batch(condition_ids: list[str], timeout_seconds: float, retries: int) -> tuple[list[dict[str, Any]], bool]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "poly_model_preprocess/1.0 (+https://polymarket.com)",
    }
    last_exc: Exception | None = None
    for query in gamma_query_variants(condition_ids):
        url = f"https://gamma-api.polymarket.com/markets?{query}"
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, list):
                    rows = [item for item in payload if isinstance(item, dict)]
                elif isinstance(payload, dict):
                    rows = []
                    for key in ("markets", "data"):
                        data = payload.get(key)
                        if isinstance(data, list):
                            rows = [item for item in data if isinstance(item, dict)]
                            break
                else:
                    rows = []
                if rows:
                    return rows, False
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        logger.warning("Gamma API request failed for %d condition_ids: %s", len(condition_ids), last_exc)
    return [], last_exc is not None


def fetch_gamma_mappings(
    market_ids: set[str],
    batch_size: int,
    timeout_seconds: float,
    retries: int,
    max_consecutive_errors: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    if not market_ids:
        return {}, 0
    batch_size = max(1, batch_size)
    fetched: dict[str, dict[str, Any]] = {}
    requested = sorted(market_ids)
    request_count = 0
    consecutive_errors = 0
    for start in range(0, len(requested), batch_size):
        batch = requested[start : start + batch_size]
        request_count += 1
        markets, had_error = fetch_gamma_batch(batch, timeout_seconds=timeout_seconds, retries=retries)
        if had_error:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    "Stopping Gamma API lookup after %d consecutive request errors.",
                    consecutive_errors,
                )
                break
        else:
            consecutive_errors = 0
        for market in markets:
            market_id = gamma_market_id(market)
            if market_id:
                fetched[market_id] = market
        logger.info(
            "Gamma API batch %d: requested=%d cumulative_matches=%d",
            request_count,
            len(batch),
            len(fetched),
        )
    return fetched, request_count


def fetch_clob_market_by_token(token_id: str, timeout_seconds: float, retries: int) -> tuple[dict[str, Any] | None, bool]:
    url = f"https://clob.polymarket.com/markets-by-token/{urllib.parse.quote(str(token_id), safe='')}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "poly_model_preprocess/1.0 (+https://polymarket.com)",
    }
    request = urllib.request.Request(url, headers=headers)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else None, False
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        logger.warning("CLOB markets-by-token request failed for token %s: %s", token_id, last_exc)
    return None, last_exc is not None


def fetch_clob_mappings(
    asset_ids_by_market: dict[str, set[str]],
    timeout_seconds: float,
    retries: int,
    max_consecutive_errors: int,
    workers: int,
) -> tuple[dict[str, dict[str, str]], int]:
    mappings: dict[str, dict[str, str]] = {}
    jobs: list[tuple[str, str]] = []
    seen_tokens: set[str] = set()
    for market_id in sorted(asset_ids_by_market):
        token_ids = sorted(asset_ids_by_market.get(market_id, set()))
        if not token_ids:
            continue
        # Any token in a binary CLOB market maps back to both primary/secondary
        # token IDs, so one request per market is sufficient.
        token_id = token_ids[0]
        if token_id in seen_tokens:
            continue
        seen_tokens.add(token_id)
        jobs.append((market_id, token_id))

    request_count = len(jobs)
    consecutive_errors = 0
    completed = 0
    workers = max(1, workers)

    def fetch_job(job: tuple[str, str]) -> tuple[str, dict[str, Any] | None, bool]:
        market_id, token_id = job
        payload, had_error = fetch_clob_market_by_token(token_id, timeout_seconds=timeout_seconds, retries=retries)
        return market_id, payload, had_error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_job, job) for job in jobs]
        for future in as_completed(futures):
            market_id, payload, had_error = future.result()
            completed += 1
            if had_error:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning("Observed %d consecutive CLOB token lookup errors.", consecutive_errors)
                continue
            consecutive_errors = 0
            if payload:
                condition_id = str(payload.get("condition_id") or payload.get("conditionId") or "").lower()
                primary = payload.get("primary_token_id") or payload.get("primaryTokenId")
                secondary = payload.get("secondary_token_id") or payload.get("secondaryTokenId")
                if condition_id == market_id.lower() and primary and secondary:
                    mappings[market_id.lower()] = {
                        "yes_asset_id": str(primary),
                        "no_asset_id": str(secondary),
                        "mapping_source": "clob_markets_by_token",
                    }
            if completed % 100 == 0:
                logger.info("CLOB token lookup completed=%d/%d cumulative_matches=%d", completed, request_count, len(mappings))
    return mappings, request_count


def scan_pm_bronze(
    config: dict,
    limit_files: int | None = None,
    scan_outcomes: bool = False,
    limit_markets: int | None = None,
) -> tuple[set[str], dict[str, dict[str, set[str]]], dict[str, Any], dict[str, set[str]], int]:
    roots = [
        resolve_path(config, "data/bronze/pm_orderbook"),
        resolve_path(config, "data/bronze/pm_price_change"),
        resolve_path(config, "data/bronze/pm_market_meta"),
    ]
    known_market_ids = discover_markets_from_hive_dirs(roots)
    if limit_markets is not None:
        known_market_ids = set(sorted(known_market_ids)[:limit_markets])
    votes: dict[str, dict[str, set[str]]] = {}
    first_ts_by_market: dict[str, Any] = {}
    asset_ids_by_market: dict[str, set[str]] = {}

    files = list_pm_parquet_files(roots)
    if limit_markets is not None:
        files = [path for path in files if hive_value(path, "market_id") in known_market_ids]
    if limit_files is not None:
        files = files[:limit_files]

    if not scan_outcomes and first_outcome_type_is_null(roots):
        logger.info("Representative PM bronze outcome column is all-null; sampling metadata and token IDs only.")
        files_scanned = sample_bronze_metadata_and_assets(files, known_market_ids, first_ts_by_market, asset_ids_by_market)
        return known_market_ids, votes, first_ts_by_market, asset_ids_by_market, files_scanned

    for idx, path in enumerate(files, start=1):
        if idx % 1000 == 0:
            logger.info("Scanned %d/%d PM bronze parquet files", idx, len(files))
        update_from_file(path, known_market_ids, votes, first_ts_by_market, asset_ids_by_market)
    return known_market_ids, votes, first_ts_by_market, asset_ids_by_market, len(files)


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    known_market_ids, votes, first_ts_by_market, asset_ids_by_market, files_scanned = scan_pm_bronze(
        config,
        limit_files=args.limit_files,
        scan_outcomes=args.scan_outcomes,
        limit_markets=args.limit_markets,
    )
    logger.info(
        "Scanned %d files, found %d markets with %d markets having explicit outcome votes and %d markets having sampled token IDs",
        files_scanned,
        len(known_market_ids),
        len(votes),
        sum(1 for ids in asset_ids_by_market.values() if ids),
    )

    gamma_markets: dict[str, dict[str, Any]] = {}
    if not args.no_gamma_api:
        gamma_markets, gamma_requests = fetch_gamma_mappings(
            known_market_ids,
            batch_size=args.gamma_batch_size,
            timeout_seconds=args.gamma_timeout_seconds,
            retries=args.gamma_retries,
            max_consecutive_errors=args.gamma_max_consecutive_errors,
        )
        logger.info(
            "Gamma API lookup finished: requested %d markets in %d batches, matched %d markets",
            len(known_market_ids),
            gamma_requests,
            len(gamma_markets),
        )

    clob_mappings: dict[str, dict[str, str]] = {}
    if not args.no_clob_token_api:
        clob_mappings, clob_requests = fetch_clob_mappings(
            asset_ids_by_market,
            timeout_seconds=args.clob_timeout_seconds,
            retries=args.clob_retries,
            max_consecutive_errors=args.clob_max_consecutive_errors,
            workers=args.clob_workers,
        )
        logger.info(
            "CLOB token lookup finished: requested %d tokens, matched %d markets",
            clob_requests,
            len(clob_mappings),
        )

    duration_seconds = int(config["polymarket"]["market_duration_seconds"])
    mapping_payload = {"markets": {}}
    for market_id in sorted(known_market_ids):
        outcome_map = votes.get(market_id, {"YES": set(), "NO": set()})
        first_ts = first_ts_by_market.get(market_id)
        start_ts = first_ts.isoformat().replace("+00:00", "Z") if first_ts is not None else None
        end_ts = (first_ts + timedelta(seconds=duration_seconds)).isoformat().replace("+00:00", "Z") if first_ts is not None else None
        question = None
        gamma_issue = None
        gamma_market = gamma_markets.get(market_id.lower())
        if gamma_market is not None:
            gamma_yes_ids, gamma_no_ids, gamma_issue = extract_gamma_token_pairs(gamma_market)
            outcome_map["YES"].update(gamma_yes_ids)
            outcome_map["NO"].update(gamma_no_ids)
            question = gamma_market.get("question") or gamma_market.get("title") or gamma_market.get("slug")
            start_ts = normalize_iso_ts(gamma_market.get("startDateIso") or gamma_market.get("startDate")) or start_ts
            end_ts = normalize_iso_ts(gamma_market.get("endDateIso") or gamma_market.get("endDate")) or end_ts
        clob_mapping = clob_mappings.get(market_id.lower())
        if clob_mapping is not None and not (outcome_map["YES"] and outcome_map["NO"]):
            outcome_map["YES"].add(clob_mapping["yes_asset_id"])
            outcome_map["NO"].add(clob_mapping["no_asset_id"])
        mapping_payload["markets"][market_id] = build_mapping_record(
            market_id=market_id,
            yes_asset_ids=outcome_map["YES"],
            no_asset_ids=outcome_map["NO"],
            question=question,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if gamma_market is not None:
            mapping_payload["markets"][market_id]["mapping_source"] = "gamma_api+bronze"
        elif clob_mapping is not None:
            mapping_payload["markets"][market_id]["mapping_source"] = clob_mapping["mapping_source"]
        if gamma_issue and mapping_payload["markets"][market_id].get("mapping_status") != "ok":
            mapping_payload["markets"][market_id]["mapping_issue"] = gamma_issue

    _, generated_path = resolve_mapping_paths(config)
    save_mapping_file(generated_path, mapping_payload)
    mapping_report, unmapped_report = generate_mapping_reports(config, mapping_payload)
    logger.info("Wrote %s", generated_path)
    logger.info("Wrote %s", mapping_report)
    logger.info("Wrote %s", unmapped_report)


if __name__ == "__main__":
    main()
