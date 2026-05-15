# Lightsail model/session isolation map

Created: 2026-05-15 Asia/Shanghai

This repo now has two local deployment roots.  Do not edit/copy one root into the other.

## 1) This Codex session / Ireland v5 terminal hold-to-expiry

Local root:

- `deployments/ireland_v5_terminal_6share/`

Purpose:

- BTC 15m Polymarket terminal hold-to-expiry shadow/live-prep model.
- 6-share execution simulation.
- Uses normal v5 edge threshold 0.10, not tail-chase.

Key local files:

- `deployments/ireland_v5_terminal_6share/configs/shadow_repricing.v5_terminal_6share.lightsail.yaml`
- `deployments/ireland_v5_terminal_6share/configs/v5_paper_executor.lightsail.yaml`
- `deployments/ireland_v5_terminal_6share/models/pm_terminal/lightgbm.joblib`
- `deployments/ireland_v5_terminal_6share/models/pm_terminal/features_pm_terminal.json`
- `deployments/ireland_v5_terminal_6share/scripts/run_pm_repricing_shadow.py`
- `deployments/ireland_v5_terminal_6share/scripts/run_v5_paper_executor.py`

Expected v5 identifiers:

- model version: `pm_terminal_v5_hold_to_expiry`
- signal config name: `v5_terminal_hold_edge10`
- target shares: `6.0`
- signal queue on its Lightsail: `/opt/pm-shadow/data/shadow/v5_signal_queue/signals.jsonl`
- executor output on its Lightsail: `/opt/pm-shadow/data/shadow/v5_executor_decisions`
- executor state on its Lightsail: `/opt/pm-shadow/state/v5_paper_executor_state.json`

Expected deployed model paths on the v5 Lightsail if using isolated deployment:

- `/opt/pm-shadow/repo/deployments/ireland_v5_terminal_6share/models/pm_terminal/lightgbm.joblib`
- `/opt/pm-shadow/repo/deployments/ireland_v5_terminal_6share/models/pm_terminal/features_pm_terminal.json`

## 2) Other Codex session / other Lightsail v6 tail-chase

Local root:

- `deployments/other_lightsail_v6_tail_chase/`

Purpose:

- Other session's tail-chase strict experiment.
- Kept separate from v5 so it does not overwrite Ireland v5 config/model paths.

Key local files:

- `deployments/other_lightsail_v6_tail_chase/configs/shadow_repricing.v6_tail_chase.lightsail.yaml`
- `deployments/other_lightsail_v6_tail_chase/configs/v6_tail_chase_paper_executor.lightsail.yaml`
- `deployments/other_lightsail_v6_tail_chase/models/pm_terminal/lightgbm.joblib`
- `deployments/other_lightsail_v6_tail_chase/models/pm_terminal/features_pm_terminal.json`
- `deployments/other_lightsail_v6_tail_chase/scripts/run_pm_repricing_shadow.py`
- `deployments/other_lightsail_v6_tail_chase/scripts/run_v5_paper_executor.py`

Expected v6 identifiers:

- model version: `pm_terminal_v6_tail_chase_strict_shadow`
- signal config name: `v6_terminal_tail_chase_strict`
- target shares: `25.0`
- signal queue on its Lightsail: `/opt/pm-shadow/data/shadow/v6_tail_chase_signal_queue/signals.jsonl`
- executor output on its Lightsail: `/opt/pm-shadow/data/shadow/v6_tail_chase_executor_decisions`
- executor state on its Lightsail: `/opt/pm-shadow/state/v6_tail_chase_paper_executor_state.json`

Expected deployed model paths on the v6 Lightsail if using isolated deployment:

- `/opt/pm-shadow/repo/deployments/other_lightsail_v6_tail_chase/models/pm_terminal/lightgbm.joblib`
- `/opt/pm-shadow/repo/deployments/other_lightsail_v6_tail_chase/models/pm_terminal/features_pm_terminal.json`

## Rules for both sessions

- Do not deploy from repo-root `configs/shadow_repricing.v5_terminal.lightsail.yaml` unless it has been checked immediately before deploy.
- v5 should only deploy from `deployments/ireland_v5_terminal_6share/...`.
- v6 should only deploy from `deployments/other_lightsail_v6_tail_chase/...`.
- The two sessions must use different queue/output/state paths.
- If a shared script is changed, copy it deliberately into the relevant deployment root only after review.
