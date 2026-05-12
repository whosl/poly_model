#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/opt/pm-shadow"
REPO_DIR="${ROOT_DIR}/repo"
VENV_DIR="${ROOT_DIR}/venv"

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  tmux \
  htop \
  jq \
  unzip

sudo mkdir -p \
  "${ROOT_DIR}/repo" \
  "${ROOT_DIR}/venv" \
  "${ROOT_DIR}/data" \
  "${ROOT_DIR}/logs" \
  "${ROOT_DIR}/reports" \
  "${ROOT_DIR}/scripts"

sudo chown -R "${USER}:${USER}" "${ROOT_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

cd "${REPO_DIR}"
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
elif [[ -f pyproject.toml ]]; then
  pip install .
else
  echo "ERROR: neither requirements.txt nor pyproject.toml found" >&2
  exit 1
fi

pytest tests/test_shadow_safety.py

echo "Shadow server setup complete."
