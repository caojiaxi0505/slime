"""Append-only run store + aggregate writers for swegym_filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    # Iterate by file lines (\n), not str.splitlines(): patches may contain
    # U+2028/U+2029/\x85 inside JSON strings, which splitlines() would break.
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class RunStore:
    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self.runs_path = self.out_dir / "runs.jsonl"

    def load_runs(self, *, phase: str | None = None) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.runs_path)
        if phase is None:
            return rows
        return [r for r in rows if r.get("phase") == phase]

    def completed_keys(self, phase: str) -> set[tuple[str, int]]:
        keys: set[tuple[str, int]] = set()
        for r in self.load_runs(phase=phase):
            iid = str(r.get("instance_id") or "")
            if not iid:
                continue
            try:
                idx = int(r.get("repeat_idx"))
            except (TypeError, ValueError):
                continue
            keys.add((iid, idx))
        return keys

    def append_run(self, row: dict[str, Any]) -> None:
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_meta(self, meta: dict[str, Any]) -> None:
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def write_named_jsonl(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self.out_dir / name
        write_jsonl(path, rows)
        return path

    def artifact_dir(self, instance_id: str, repeat_idx: int) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        d = self.out_dir / "artifacts" / safe / f"r{repeat_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d
