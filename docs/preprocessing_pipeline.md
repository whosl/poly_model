# Preprocessing Pipeline

## Overview

This repository builds an offline preprocessing pipeline for BTC / Polymarket BTC 5m prediction datasets. The pipeline is organized into four layers:

- `raw`: original gzip JSON feeds.
- `bronze`: normalized event-level parquet tables.
- `silver`: aligned 1s feature tables.
- `gold`: model-ready training datasets.

The current initial implementation covers raw inspection and `raw -> bronze parquet`.

## Directory Layout

- `data/raw/binance`: normalized raw input location for Binance files.
- `data/raw/polymarket`: normalized raw input location for Polymarket files.
- `data/bronze/*`: standardized parquet event tables.
- `data/silver/*`: reserved for 1s feature tables.
- `data/gold/*`: reserved for model-ready training datasets.
- `reports/raw_schema_report.md`: generated raw schema summary.

If raw files already exist under `data/YYYYMMDD/*.jsonl.gz`, the scripts create links under `data/raw/...` so the rest of the pipeline can use a stable layout.

## Scripts

- `scripts/inspect_raw.py`: scans raw files, inspects top-level and nested schema, infers likely data type, and writes `reports/raw_schema_report.md`.
- `scripts/build_bronze.py`: standardizes raw records into event-level parquet partitions.
- `scripts/run_preprocess_pipeline.py`: initial orchestrator for `inspect_raw -> bronze`.

## Bronze Datasets

### Binance bookTicker

Output root: `data/bronze/binance_bookticker`

Fields:

- `ts_event`
- `ts_recv`
- `symbol`
- `bid_price`
- `bid_qty`
- `ask_price`
- `ask_qty`
- `mid_price`
- `spread`
- `microprice`
- `source_file`

### Binance aggTrade

Output root: `data/bronze/binance_aggtrade`

Fields:

- `ts_event`
- `ts_recv`
- `symbol`
- `price`
- `qty`
- `is_buyer_maker`
- `side`
- `notional`
- `source_file`

### Binance depth

Output root: `data/bronze/binance_depth`

Fields include:

- `ts_event`
- `ts_recv`
- `symbol`
- `bid_px_1 ... bid_px_N`
- `bid_qty_1 ... bid_qty_N`
- `ask_px_1 ... ask_px_N`
- `ask_qty_1 ... ask_qty_N`
- `bid_depth_N`
- `ask_depth_N`
- `depth_imbalance_N`
- `source_file`

### Polymarket orderbook

Output root: `data/bronze/pm_orderbook`

Fields:

- `ts_event`
- `ts_recv`
- `market_id`
- `asset_id`
- `outcome`
- `best_bid`
- `best_ask`
- `mid`
- `spread`
- `bid_size_1`
- `ask_size_1`
- `bid_depth_3`
- `ask_depth_3`
- `bid_depth_5`
- `ask_depth_5`
- `depth_imbalance_3`
- `depth_imbalance_5`
- `source_file`

### Polymarket price change

Output root: `data/bronze/pm_price_change`

Fields:

- `ts_event`
- `ts_recv`
- `market_id`
- `asset_id`
- `outcome`
- `price`
- `best_bid`
- `best_ask`
- `side`
- `event_type`
- `source_file`

## Time Alignment Principles

- All timestamps are normalized to UTC.
- `ts_event` uses exchange or payload event time when available.
- `ts_recv` uses local receive time from raw records when available.
- If `ts_event` is missing, the current bronze implementation falls back to `ts_recv`.
- Future silver and gold layers must only use observations with timestamp `<= sample_ts`.

## Mapping and Schema Variance

Raw schema is not assumed stable. Field mappings are centralized in `configs/preprocess.yaml` under `schema_mapping`.

Current observed patterns:

- Binance records use websocket format with `raw.stream` and `raw.data`.
- Binance receive time is available as `recv_ns`.
- Polymarket records mix:
  - orderbook snapshots as `raw: [ ... ]`
  - price changes as `raw.price_changes`
  - event types such as `book`, `price_change`, `best_bid_ask`, `last_trade_price`

Known limitation in the initial bronze pass:

- `outcome` is not reliably present in current Polymarket raw data. The bronze layer preserves `asset_id` and leaves `outcome` null when YES/NO is not explicit. Later stages will need explicit YES/NO asset mapping logic or metadata enrichment.
- Some current raw gzip files appear partially corrupted or truncated. The reader keeps the readable prefix and emits an explicit warning so the pipeline can continue without silently hiding the issue.

## Usage

```bash
python scripts/inspect_raw.py --config configs/preprocess.yaml
python scripts/build_bronze.py --config configs/preprocess.yaml
python scripts/run_preprocess_pipeline.py --config configs/preprocess.yaml
```

Useful flags:

- `--limit-files`: process only a subset of files during development.
- `--dry-run`: show planned work without writing outputs.

## Next Steps

The next implementation stage will add:

- `silver/binance_1s`
- `silver/pm_1s`
- `gold/btc_direction_1s`
- `gold/pm_terminal_1s`
- `gold/pm_repricing_1s`

Those stages must preserve strict as-of semantics, quote TTL logic, and leak-proof label generation.
