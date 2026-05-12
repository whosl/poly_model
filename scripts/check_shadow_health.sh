#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-pm-repricing-shadow}"
ROOT_DIR="${2:-/opt/pm-shadow}"
STDOUT_LOG="${ROOT_DIR}/logs/shadow_stdout.log"
STDERR_LOG="${ROOT_DIR}/logs/shadow_stderr.log"
DATA_DIR="${ROOT_DIR}/data/shadow"

pass() { echo "PASS: $*"; }
warn() { echo "WARN: $*"; }
fail() { echo "FAIL: $*"; }

status_rc=0

if systemctl is-active --quiet "${SERVICE_NAME}"; then
  pass "systemd service ${SERVICE_NAME} is active"
else
  fail "systemd service ${SERVICE_NAME} is not active"
  status_rc=1
fi

now_epoch=$(date +%s)

check_recent_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    fail "${label} missing: ${path}"
    status_rc=1
    return
  fi
  local mtime
  mtime=$(stat -c %Y "${path}")
  if (( now_epoch - mtime <= 300 )); then
    pass "${label} updated within last 5 minutes"
  else
    warn "${label} not updated within last 5 minutes"
  fi
}

check_recent_file "${STDOUT_LOG}" "shadow stdout log"
check_recent_file "${STDERR_LOG}" "shadow stderr log"

latest_parquet=$(find "${DATA_DIR}" -type f -name '*.parquet' -mmin -5 2>/dev/null | head -n 1 || true)
if [[ -n "${latest_parquet}" ]]; then
  pass "parquet output written within last 5 minutes"
else
  warn "no parquet output in last 5 minutes"
fi

if [[ -f "${STDOUT_LOG}" ]] && grep -q "SHADOW MODE ONLY - NO ORDERS WILL BE PLACED" "${STDOUT_LOG}"; then
  pass "stdout contains shadow-only startup banner"
else
  fail "shadow-only startup banner not found in stdout log"
  status_rc=1
fi

if [[ -f "${STDERR_LOG}" ]] && grep -qi "traceback" "${STDERR_LOG}"; then
  fail "traceback detected in stderr log"
  status_rc=1
else
  pass "no traceback detected in stderr log"
fi

disk_use=$(df -h "${ROOT_DIR}" | awk 'NR==2 {print $5}')
mem_line=$(free -m | awk 'NR==2 {printf "%s/%sMB (%.1f%%)", $3, $2, ($3/$2)*100}')
echo "INFO: disk usage on ${ROOT_DIR}: ${disk_use}"
echo "INFO: memory usage: ${mem_line}"

exit "${status_rc}"
