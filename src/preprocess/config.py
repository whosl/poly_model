from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    project_root = path.resolve().parents[1]
    config["_config_path"] = str(path.resolve())
    config["_project_root"] = str(project_root)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path
