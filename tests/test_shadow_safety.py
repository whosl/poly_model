from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

MODULE_PATH = Path("scripts/run_pm_repricing_shadow.py")
SPEC = importlib.util.spec_from_file_location("run_pm_repricing_shadow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
normalize_shadow_config = MODULE.normalize_shadow_config
validate_shadow_config = MODULE.validate_shadow_config


def test_local_shadow_config_is_no_trade() -> None:
    cfg = normalize_shadow_config(yaml.safe_load(Path("configs/shadow_repricing.yaml").read_text(encoding="utf-8")))
    validate_shadow_config(cfg)
    assert cfg["mode"] == "shadow_no_trade"
    assert cfg["enable_trading"] is False
    assert cfg["safety"]["disable_order_placement"] is True
    assert cfg["safety"]["fail_if_order_api_configured"] is True


def test_lightsail_shadow_config_is_no_trade() -> None:
    cfg = normalize_shadow_config(yaml.safe_load(Path("configs/shadow_repricing.lightsail.yaml").read_text(encoding="utf-8")))
    validate_shadow_config(cfg)
    assert cfg["mode"] == "shadow_no_trade"
    assert cfg["enable_trading"] is False
    assert cfg["polymarket"]["user_channel_enabled"] is False


def test_rejects_trading_secret_keys() -> None:
    cfg = normalize_shadow_config(
        {
            "mode": "shadow_no_trade",
            "enable_trading": False,
            "model": {"model_path": "m.joblib", "features_path": "f.json"},
            "market_meta": {"silver_pm_path": "meta.parquet"},
            "output": {"base_dir": "data/shadow", "report_dir": "reports/shadow"},
            "safety": {"disable_order_placement": True, "fail_if_order_api_configured": True},
            "signal_configs": [{"name": "x", "direction": "UP", "threshold_up": 0.8, "max_spread": 0.1, "min_tte": 10, "cooldown_seconds": 10}],
            "trading_private_key": "should-not-exist",
        }
    )
    with pytest.raises(RuntimeError):
        validate_shadow_config(cfg)


def test_rejects_enable_trading_true() -> None:
    cfg = normalize_shadow_config(
        {
            "mode": "shadow_no_trade",
            "enable_trading": True,
            "model": {"model_path": "m.joblib", "features_path": "f.json"},
            "market_meta": {"silver_pm_path": "meta.parquet"},
            "output": {"base_dir": "data/shadow", "report_dir": "reports/shadow"},
            "safety": {"disable_order_placement": True, "fail_if_order_api_configured": True},
            "signal_configs": [{"name": "x", "direction": "UP", "threshold_up": 0.8, "max_spread": 0.1, "min_tte": 10, "cooldown_seconds": 10}],
        }
    )
    with pytest.raises(RuntimeError):
        validate_shadow_config(cfg)
