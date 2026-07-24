"""Convert Claude Code SFT trial logs to slime SFT JSONL.

``sft_remote_smoke.py`` writes audit-friendly trial rows with fields such as
``resolved``, ``diff_path`` and ``turns``.  Slime SFT only needs a conversation
column, normally ``messages``, plus optional metadata.  This converter keeps the
training file small and prevents unresolved or malformed trials from entering
SFT by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    if not all(isinstance(message, dict) for message in messages):
        return False
    if not any(message.get("role") == "assistant" for message in messages):
        return False
    return messages[-1].get("role") == "assistant"


def _metadata_from_trial(trial: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "instance_id": trial.get("instance_id"),
        "index": trial.get("index"),
        "session_id_sha256": trial.get("session_id_sha256"),
        "ok": bool(trial.get("ok")),
        "resolved": bool(trial.get("resolved")),
        "agent_exit_code": trial.get("agent_exit_code"),
        "applied_cleanly": bool(trial.get("applied_cleanly")),
        "diff_path": trial.get("diff_path"),
        "diff_chars": trial.get("diff_chars"),
        "trajectory_path": trial.get("trajectory_path"),
        "turn_count": trial.get("turn_count"),
    }


def convert(args: argparse.Namespace) -> dict[str, int]:
    counts = {
        "read": 0,
        "written": 0,
        "skip_not_ok": 0,
        "skip_unresolved": 0,
        "skip_bad_messages": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as out:
        for input_path in [Path(p) for p in args.input]:
            for line_no, trial in _iter_jsonl(input_path):
                counts["read"] += 1
                if args.ok_only and not bool(trial.get("ok")):
                    counts["skip_not_ok"] += 1
                    continue
                if args.resolved_only and not bool(trial.get("resolved")):
                    counts["skip_unresolved"] += 1
                    continue
                messages = trial.get("messages")
                if not _valid_messages(messages):
                    counts["skip_bad_messages"] += 1
                    if not args.quiet:
                        print(
                            f"warning: skip malformed messages at {input_path}:{line_no} "
                            f"instance_id={trial.get('instance_id')!r}"
                        )
                    continue
                row = {
                    args.messages_key: messages,
                    args.metadata_key: _metadata_from_trial(trial, source=args.source),
                }
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                out.write("\n")
                counts["written"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Input sft_trials.jsonl file(s).")
    parser.add_argument("--output", required=True, help="Output slime SFT JSONL path.")
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--metadata-key", default="metadata")
    parser.add_argument("--source", default="deepseek_v4_pro_cc_ags_swegym")
    parser.add_argument("--include-unresolved", action="store_true", help="Keep unresolved trials too.")
    parser.add_argument("--include-not-ok", action="store_true", help="Keep trials whose task run failed.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.resolved_only = not args.include_unresolved
    args.ok_only = not args.include_not_ok

    counts = convert(args)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
