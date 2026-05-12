from __future__ import annotations

from pathlib import Path

from .config import resolve_path
from .io_utils import ensure_dir


def bootstrap_layout(config: dict) -> None:
    directories = [
        "data/raw/binance",
        "data/raw/polymarket",
        "data/bronze/binance_bookticker",
        "data/bronze/binance_aggtrade",
        "data/bronze/binance_depth",
        "data/bronze/pm_orderbook",
        "data/bronze/pm_price_change",
        "data/bronze/pm_market_meta",
        "data/silver/binance_1s",
        "data/silver/pm_1s",
        "data/silver/joined_1s",
        "data/gold/btc_direction_1s",
        "data/gold/pm_terminal_1s",
        "data/gold/pm_repricing_1s",
        "reports",
        "docs",
        "tests",
        "scripts",
    ]
    for rel in directories:
        ensure_dir(resolve_path(config, rel))


def link_or_copy_raw_layout(config: dict, dry_run: bool = False) -> list[tuple[Path, Path]]:
    project_root = Path(config["_project_root"])
    raw_binance = resolve_path(config, config["raw_paths"]["binance"])
    raw_polymarket = resolve_path(config, config["raw_paths"]["polymarket"])
    source_root = project_root / "data"

    operations: list[tuple[Path, Path]] = []
    for source in sorted(source_root.glob("20*/*.jsonl.gz")):
        name = source.name.lower()
        if "binance" in name:
            target = raw_binance / source.parent.name / source.name
        elif "poly" in name:
            target = raw_polymarket / source.parent.name / source.name
        else:
            continue
        operations.append((source, target))

    for source, target in operations:
        if dry_run or target.exists():
            continue
        ensure_dir(target.parent)
        try:
            target.hardlink_to(source)
        except OSError:
            target.symlink_to(source)
    return operations
