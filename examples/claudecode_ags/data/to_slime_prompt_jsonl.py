"""Convert swegym_filter JSONL rows into slime ``PROMPT_DATA`` JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REQUIRED_META = ("instance_id", "image", "problem_statement", "FAIL_TO_PASS")


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    md = dict(row.get("metadata") or {})
    if not md.get("instance_id"):
        md["instance_id"] = row.get("instance_id")
    problem = str(md.get("problem_statement") or "").strip()
    if not problem:
        raise ValueError(f"missing problem_statement for {md.get('instance_id')!r}")
    for key in _REQUIRED_META:
        if key == "problem_statement":
            continue
        if key not in md or md[key] in (None, ""):
            raise ValueError(f"missing {key} for {md.get('instance_id')!r}")
    extra = dict(md)
    for k in ("n_resolved", "n_repeats", "n_runs", "pass_rate"):
        if k in row:
            extra[k] = row[k]
    # Match existing train_grpo_*.slime.jsonl chat-prompt shape.
    return {
        "prompt": [{"role": "user", "content": problem}],
        "extra_info": extra,
    }


def convert_file(src: Path, dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(convert_row(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--expect-rows", type=int, default=None)
    args = p.parse_args(argv)
    n = convert_file(args.src, args.dst)
    if args.expect_rows is not None and n != args.expect_rows:
        raise SystemExit(f"row count {n} != expect {args.expect_rows}")
    print(f"wrote {n} rows -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
