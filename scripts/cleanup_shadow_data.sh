#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/opt/pm-shadow/data/shadow"
DRY_RUN=false
DAYS_RAW=3
DAYS_QUOTES=7
DAYS_PARQUET=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --days-raw) DAYS_RAW="$2"; shift 2 ;;
    --days-quotes) DAYS_QUOTES="$2"; shift 2 ;;
    --days-parquet) DAYS_PARQUET="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

run_find_delete() {
  local target="$1"
  local days="$2"
  if [[ ! -d "${target}" ]]; then
    return
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    find "${target}" -type f -mtime +"${days}" -print
  else
    find "${target}" -type f -mtime +"${days}" -print -delete
  fi
}

echo "cleanup shadow data root=${ROOT_DIR} dry_run=${DRY_RUN}"

# raw websocket or feed captures
run_find_delete "${ROOT_DIR}/raw_ws" "${DAYS_RAW}"
run_find_delete "${ROOT_DIR}/binance_ticks" "${DAYS_RAW}"

# quote snapshots
run_find_delete "${ROOT_DIR}/pm_quote_state" "${DAYS_QUOTES}"

# core parquet outputs
run_find_delete "${ROOT_DIR}/repricing_signals" "${DAYS_PARQUET}"
run_find_delete "${ROOT_DIR}/repricing_outcomes" "${DAYS_PARQUET}"
run_find_delete "${ROOT_DIR}/latency_metrics" "${DAYS_PARQUET}"

echo "cleanup complete"
