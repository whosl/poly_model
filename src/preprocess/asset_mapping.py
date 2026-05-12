from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from .config import resolve_path
from .io_utils import ensure_dir
from .reporting import markdown_table, write_markdown


def load_mapping_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"markets": {}}
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    payload.setdefault("markets", {})
    return payload


def save_mapping_file(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=True)


def resolve_mapping_paths(config: dict) -> tuple[Path, Path]:
    manual = resolve_path(config, "configs/pm_asset_mapping.yaml")
    generated = resolve_path(config, "configs/pm_asset_mapping.generated.yaml")
    return manual, generated


def load_combined_mapping(config: dict) -> dict[str, Any]:
    manual_path, generated_path = resolve_mapping_paths(config)
    generated = load_mapping_file(generated_path)
    manual = load_mapping_file(manual_path)
    merged = {"markets": {}}
    merged["markets"].update(generated.get("markets", {}))
    merged["markets"].update(manual.get("markets", {}))
    return merged


def build_mapping_record(
    market_id: str,
    yes_asset_ids: set[str],
    no_asset_ids: set[str],
    question: str | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> dict[str, Any]:
    if len(yes_asset_ids) == 1 and len(no_asset_ids) == 1:
        return {
            "yes_asset_id": next(iter(yes_asset_ids)),
            "no_asset_id": next(iter(no_asset_ids)),
            "question": question,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "mapping_status": "ok",
        }
    if yes_asset_ids or no_asset_ids:
        return {
            "yes_asset_ids": sorted(yes_asset_ids),
            "no_asset_ids": sorted(no_asset_ids),
            "question": question,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "mapping_status": "conflict",
        }
    return {
        "question": question,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "mapping_status": "unmapped",
    }


def generate_mapping_reports(config: dict, mapping_payload: dict[str, Any]) -> tuple[Path, Path]:
    reports_root = resolve_path(config, config["output_paths"]["reports"])
    mapping_report = reports_root / "pm_asset_mapping_report.md"
    unmapped_report = reports_root / "pm_unmapped_markets.md"
    markets = mapping_payload.get("markets", {})
    ok_rows = []
    unmapped_rows = []
    conflict_rows = []
    for market_id, info in sorted(markets.items()):
        status = info.get("mapping_status", "unmapped")
        row = [market_id, status, info.get("yes_asset_id"), info.get("no_asset_id"), info.get("question"), info.get("start_ts"), info.get("end_ts")]
        if status == "ok":
            ok_rows.append(row)
        elif status == "conflict":
            conflict_rows.append(row)
        else:
            unmapped_rows.append(row)

    lines = ["# Polymarket Asset Mapping Report", ""]
    lines.append(f"- total_markets: `{len(markets)}`")
    lines.append(f"- ok: `{len(ok_rows)}`")
    lines.append(f"- conflict: `{len(conflict_rows)}`")
    lines.append(f"- unmapped: `{len(unmapped_rows)}`")
    lines.append("")
    if ok_rows:
        lines.append("## OK Mappings")
        lines.append("")
        lines.extend(markdown_table(["market_id", "status", "yes_asset_id", "no_asset_id", "question", "start_ts", "end_ts"], ok_rows))
        lines.append("")
    if conflict_rows:
        lines.append("## Conflicts")
        lines.append("")
        lines.extend(markdown_table(["market_id", "status", "yes_asset_id", "no_asset_id", "question", "start_ts", "end_ts"], conflict_rows))
        lines.append("")
    if unmapped_rows:
        lines.append("## Unmapped")
        lines.append("")
        lines.extend(markdown_table(["market_id", "status", "yes_asset_id", "no_asset_id", "question", "start_ts", "end_ts"], unmapped_rows))
    write_markdown(mapping_report, lines)

    unmapped_lines = ["# Polymarket Unmapped Markets", ""]
    if not unmapped_rows:
        unmapped_lines.append("No unmapped markets.")
    else:
        unmapped_lines.extend(markdown_table(["market_id", "status", "yes_asset_id", "no_asset_id", "question", "start_ts", "end_ts"], unmapped_rows))
    write_markdown(unmapped_report, unmapped_lines)
    return mapping_report, unmapped_report


def mapping_to_frame(mapping_payload: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for market_id, info in mapping_payload.get("markets", {}).items():
        rows.append(
            {
                "market_id": market_id,
                "yes_asset_id": info.get("yes_asset_id"),
                "no_asset_id": info.get("no_asset_id"),
                "question": info.get("question"),
                "market_start_ts": info.get("start_ts"),
                "market_end_ts": info.get("end_ts"),
                "mapping_status": info.get("mapping_status", "unmapped"),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "market_id": pl.String,
            "yes_asset_id": pl.String,
            "no_asset_id": pl.String,
            "question": pl.String,
            "market_start_ts": pl.String,
            "market_end_ts": pl.String,
            "mapping_status": pl.String,
        }
    )
