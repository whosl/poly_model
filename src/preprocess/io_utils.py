from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import zlib

logger = logging.getLogger(__name__)


@dataclass
class RawReadStatus:
    source_file: str
    readable_rows: int = 0
    failed_at_line: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    prefix_retained: bool = False

    @property
    def is_corrupt(self) -> bool:
        return self.error_type is not None


class JsonRecordReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.status = RawReadStatus(source_file=str(path))

    def __iter__(self) -> Iterator[dict]:
        yield from self._iter_records()

    def _iter_records(self) -> Iterator[dict]:
        suffixes = self.path.suffixes
        if suffixes[-2:] == [".jsonl", ".gz"] or suffixes[-2:] == [".json", ".gz"]:
            opener = gzip.open
        elif self.path.suffix in {".jsonl", ".json"}:
            opener = open
        else:
            raise ValueError(f"Unsupported file extension for {self.path}")

        with opener(self.path, "rt", encoding="utf-8") as fh:
            if self.path.suffixes[-2:] == [".json", ".gz"] or self.path.suffix == ".json":
                try:
                    payload = json.load(fh)
                except (OSError, EOFError, zlib.error, json.JSONDecodeError) as exc:
                    self._mark_error(exc)
                    return
                rows: Iterable[Any]
                if isinstance(payload, list):
                    rows = payload
                elif isinstance(payload, dict):
                    rows = [payload]
                else:
                    rows = []
                for row in rows:
                    if isinstance(row, dict):
                        self.status.readable_rows += 1
                        yield row
                return

            try:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._mark_error(exc)
                        raise ValueError(f"Failed to parse JSON line in {self.path}: {exc}") from exc
                    if isinstance(record, dict):
                        self.status.readable_rows += 1
                        yield record
            except (OSError, EOFError, zlib.error) as exc:
                self._mark_error(exc)
                logger.warning("Stopped early while reading %s: %s", self.path, exc)

    def _mark_error(self, exc: Exception) -> None:
        self.status.failed_at_line = self.status.readable_rows + 1
        self.status.error_type = type(exc).__name__
        self.status.error_message = str(exc)
        self.status.prefix_retained = self.status.readable_rows > 0


def iter_json_records(path: Path) -> Iterator[dict]:
    yield from JsonRecordReader(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
