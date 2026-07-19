#!/usr/bin/env python3
"""Audit consecutive repeated reasoning/content across GRPO assistant turns."""

from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch

from examples.claudecode_ags.rewards.tool_loop_penalty import assistant_turns


RUNS = {
    "8 GPU": [
        (range(22), Path("/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_1node_grpo_debug/rollout_dumps")),
    ],
    "16 GPU": [
        (range(10), Path("/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_2node_grpo_c64_t45/rollout_dumps")),
        (
            range(10, 22),
            Path("/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_2node_grpo_c64_t45_resume10_adapterfix/rollout_dumps"),
        ),
    ],
}

TOOL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TAG_RE = re.compile(r"<\|[^>]+\|>|</?think>")
WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize(text: str) -> str:
    text = TOOL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    return " ".join(text.split()).strip().lower()


def split_turn(turn: str) -> tuple[str, str]:
    """Qwen emits reasoning before </think> and visible content after it."""
    if "</think>" not in turn:
        return "", normalize(turn)
    reasoning, content = turn.split("</think>", 1)
    return normalize(reasoning), normalize(content)


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = WORD_RE.findall(text)
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    aa, bb = shingles(a), shingles(b)
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def longest_run(values: list[str], threshold: float) -> int:
    longest = current = 0
    previous = ""
    for value in values:
        # Ignore tiny boilerplate; an empty/short text breaks the run.
        if len(value) >= 20 and len(previous) >= 20 and similarity(previous, value) >= threshold:
            current += 1
        elif len(value) >= 20:
            current = 1
        else:
            current = 0
        previous = value
        longest = max(longest, current)
    return longest


def episode_key(sample: dict[str, Any]) -> str:
    return str(sample.get("session_id") or repr((sample.get("index"), sample.get("group_index"))))


def audit_episode(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning: list[str] = []
    content: list[str] = []
    for sample in sorted(samples, key=lambda s: int((s.get("metadata") or {}).get("segment_idx", -1))):
        for turn in assistant_turns(str(sample.get("response") or "")):
            r, c = split_turn(turn)
            reasoning.append(r)
            content.append(c)
    return {
        "turns": len(reasoning),
        "reasoning_nonempty": sum(bool(x) for x in reasoning),
        "content_nonempty": sum(bool(x) for x in content),
        "reasoning_exact": longest_run(reasoning, 1.0),
        "reasoning_near80": longest_run(reasoning, 0.8),
        "reasoning_near60": longest_run(reasoning, 0.6),
        "content_exact": longest_run(content, 1.0),
        "content_near80": longest_run(content, 0.8),
        "content_near60": longest_run(content, 0.6),
    }


def main() -> None:
    out_dir = Path("docs/superpowers/notes/assets/2026-07-16-reasoning-content-repetition")
    out_dir.mkdir(parents=True, exist_ok=True)
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for config, sources in RUNS.items():
        for steps, root in sources:
            for step in steps:
                path = root / f"rollout_{step}.pt"
                if not path.exists():
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
                for sample in payload["samples"]:
                    groups[episode_key(sample)].append(sample)
                audits = []
                for sid, members in groups.items():
                    row = {"config": config, "step": step, "session_id": sid, **audit_episode(members)}
                    audits.append(row)
                    episode_rows.append(row)
                summary: dict[str, Any] = {
                    "config": config,
                    "step": step,
                    "episodes": len(audits),
                    "turns": sum(x["turns"] for x in audits),
                }
                for field in (
                    "reasoning_exact", "reasoning_near80", "reasoning_near60",
                    "content_exact", "content_near80", "content_near60",
                ):
                    for threshold in (3, 4, 5):
                        summary[f"{field}_ge{threshold}"] = sum(x[field] >= threshold for x in audits)
                step_rows.append(summary)
                print(json.dumps(summary, sort_keys=True), flush=True)
                del payload

    for name, rows in (("steps.csv", step_rows), ("episodes.csv", episode_rows)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
