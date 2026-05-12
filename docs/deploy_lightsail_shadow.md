# Deploy PM Repricing Shadow Logger to AWS Lightsail (Ireland)

## Scope

This system is **SHADOW ONLY**.

- It **does not place orders**
- It **must not** receive any trading private key
- It **must not** enable trading config
- It **must not** enable order placement APIs

At startup, the service must print:

`SHADOW MODE ONLY - NO ORDERS WILL BE PLACED`

## 1. Local preparation

1. Confirm replay parity already passed locally
2. Confirm no-trade safety test passes:

```bash
pytest tests/test_shadow_safety.py
```

3. Export compact read-only market metadata required by the runtime:

```bash
python scripts/export_shadow_market_meta.py \
  --silver-path data/silver/pm_1s \
  --output-path configs/shadow_market_meta.parquet
```

4. Confirm `configs/shadow_repricing.lightsail.yaml` still contains:

- `mode: shadow_no_trade`
- `enable_trading: false`
- `disable_order_placement: true`
- `fail_if_order_api_configured: true`

5. Confirm project tree does **not** contain:

- `.pem`
- `.key`
- trading `.env`
- private wallet / trading secret material

## 2. SSH key permissions

Do **not** copy your PEM into the repo.

Linux/macOS / WSL:

```bash
chmod 400 /path/to/EuKey.pem
```

Test connectivity:

```bash
ssh -i /path/to/EuKey.pem ubuntu@<LIGHTSAIL_PUBLIC_IP>
```

## 3. Recommended server layout

```text
/opt/pm-shadow/
  repo/
  venv/
  data/
  models/
  logs/
  reports/
  scripts/
```

Create it:

```bash
sudo mkdir -p /opt/pm-shadow/{repo,venv,data,models,logs,reports,scripts}
sudo chown -R ubuntu:ubuntu /opt/pm-shadow
```

## 4. rsync deployment

Recommended rsync:

```bash
rsync -avz --progress \
  -e "ssh -i /path/to/EuKey.pem" \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".venv/" \
  --exclude "venv/" \
  --exclude "*.pem" \
  --exclude "*.key" \
  --exclude ".env" \
  --exclude "data/raw/" \
  --exclude "data/bronze/" \
  --exclude "data/silver/" \
  --exclude "data/gold/" \
  --exclude "reports/stage1/" \
  --exclude "reports/audit/" \
  ./ ubuntu@<LIGHTSAIL_PUBLIC_IP>:/opt/pm-shadow/repo/
```

Important:

- shadow output directories should be created on server
- do not sync local `data/shadow/` over live server outputs

## 5. Server initialization

Run:

```bash
cd /opt/pm-shadow/repo
bash scripts/server_setup_shadow.sh
```

This will:

- install Python and utilities
- create `/opt/pm-shadow/venv`
- install dependencies
- run `pytest tests/test_shadow_safety.py`

## 6. Configure websocket endpoints

Edit:

```text
configs/shadow_repricing.lightsail.yaml
```

Fill:

- `feeds.polymarket.websocket_url`
- `polymarket.ws_url`
- `feeds.polymarket.market_channel`
- `polymarket.subscribe_messages`

If Polymarket read-only market data needs a token, use **read-only market-data credentials only**.
Do **not** place any trading key, wallet key, mnemonic, or order API secret on the server.

## 7. Dry-run before live

Required gate:

```bash
source /opt/pm-shadow/venv/bin/activate
cd /opt/pm-shadow/repo

pytest tests/test_shadow_safety.py

python scripts/run_pm_repricing_shadow.py \
  --config configs/shadow_repricing.lightsail.yaml \
  --dry-run \
  --duration-minutes 5
```

If replay data exists on the server, also run:

```bash
python scripts/run_pm_repricing_shadow.py \
  --config configs/shadow_repricing.lightsail.yaml \
  --replay-windows data/shadow/replay_windows/repricing_replay_windows.json \
  --replay-source-silver data/silver/pm_1s \
  --dry-run \
  --duration-minutes 60
```

Only after these pass should live shadow be started.

## 8. Install and start systemd

```bash
sudo cp deploy/pm-repricing-shadow.service /etc/systemd/system/
sudo cp deploy/pm-repricing-shadow-summary.service /etc/systemd/system/
sudo cp deploy/pm-repricing-shadow-summary.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pm-repricing-shadow
sudo systemctl enable pm-repricing-shadow-summary.timer
sudo systemctl start pm-repricing-shadow
sudo systemctl start pm-repricing-shadow-summary.timer
sudo systemctl status pm-repricing-shadow
sudo systemctl status pm-repricing-shadow-summary.timer
```

## 9. View logs

```bash
tail -f /opt/pm-shadow/logs/shadow_stdout.log
tail -f /opt/pm-shadow/logs/shadow_stderr.log
journalctl -u pm-repricing-shadow -f
```

## 10. Generate / inspect daily reports

Manual summary:

```bash
/opt/pm-shadow/venv/bin/python scripts/summarize_repricing_shadow.py \
  --config configs/shadow_repricing.lightsail.yaml \
  --date $(date -u +%F)
```

Output:

```text
/opt/pm-shadow/reports/shadow/
```

## 11. Pull data back to local

Signals / outcomes / latency metrics:

```bash
rsync -avz --progress \
  -e "ssh -i /path/to/EuKey.pem" \
  ubuntu@<LIGHTSAIL_PUBLIC_IP>:/opt/pm-shadow/data/shadow/ \
  ./data/shadow_lightsail/
```

Reports:

```bash
rsync -avz --progress \
  -e "ssh -i /path/to/EuKey.pem" \
  ubuntu@<LIGHTSAIL_PUBLIC_IP>:/opt/pm-shadow/reports/shadow/ \
  ./reports/shadow_lightsail/
```

## 12. Health checks

Run:

```bash
bash scripts/check_shadow_health.sh
```

It checks:

- systemd active state
- recent log updates
- recent parquet output
- presence of shadow-only banner
- traceback absence
- disk usage
- memory usage

## 13. Stop service

```bash
sudo systemctl stop pm-repricing-shadow
sudo systemctl disable pm-repricing-shadow
```

## 14. Data retention and cleanup

Recommended retention:

- raw websocket logs: 3 days
- quote snapshots: 7 days
- signals/outcomes/latency metrics: 30 days or longer
- reports: keep long-term

Dry-run cleanup preview:

```bash
bash scripts/cleanup_shadow_data.sh --dry-run --days-raw 3 --days-quotes 7 --days-parquet 30
```

Apply cleanup:

```bash
bash scripts/cleanup_shadow_data.sh --days-raw 3 --days-quotes 7 --days-parquet 30
```

## Final safety reminders

- This deployment is **SHADOW ONLY**
- It must **never** place orders
- Do **not** upload trading private keys
- Do **not** enable trading config
- Do **not** enable user order channels
