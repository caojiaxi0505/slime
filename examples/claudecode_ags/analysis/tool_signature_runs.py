#!/usr/bin/env python3
"""Plot consecutive identical tool-signature runs in GRPO rollout dumps."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch


RUNS = {
    "8 GPU": [
        (
            range(0, 22),
            Path("/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_1node_grpo_debug/rollout_dumps"),
        )
    ],
    "16 GPU": [
        (
            range(0, 10),
            Path("/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_2node_grpo_c64_t45/rollout_dumps"),
        ),
        (
            range(10, 21),
            Path(
                "/mnt/sn-007/jiaxicao/checkpoints/cc-ags/"
                "qwen35_9b_cc_ags_2node_grpo_c64_t45_resume10_adapterfix/rollout_dumps"
            ),
        ),
    ],
}

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", re.DOTALL)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _episode_key(sample: dict[str, Any]) -> str:
    session_id = sample.get("session_id")
    if session_id:
        return str(session_id)
    metadata = sample.get("metadata") or {}
    return repr(
        (
            metadata.get("instance_id"),
            sample.get("group_index"),
            sample.get("index"),
            sample.get("rollout_id"),
        )
    )


def _segment_index(sample: dict[str, Any]) -> int:
    value = (sample.get("metadata") or {}).get("segment_idx")
    return -1 if value is None else int(value)


def _assistant_turns(response: str) -> Iterable[str]:
    """Yield generated assistant turns from one decoded adapter segment.

    A segment starts directly with assistant content. Later turns are rendered
    with explicit chat-template markers after tool results.
    """

    if not response:
        return
    first_marker = response.find("<|im_start|>")
    first_end = response.find("<|im_end|>")
    if first_end >= 0 and (first_marker < 0 or first_end < first_marker):
        yield response[:first_end]
    elif first_marker < 0:
        yield response
    for match in ASSISTANT_RE.finditer(response):
        yield match.group(1)


def _turn_signatures(turn: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    exact_calls: list[str] = []
    tool_names: list[str] = []
    for function, body in TOOL_CALL_RE.findall(turn):
        function = function.strip()
        parameters = sorted((name.strip(), _normalize(value)) for name, value in PARAM_RE.findall(body))
        exact_calls.append(json.dumps([function, parameters], ensure_ascii=False, separators=(",", ":")))
        tool_names.append(function)
    if not exact_calls:
        return None
    # Preserve multiplicity but ignore the order of parallel calls.
    return tuple(sorted(exact_calls)), tuple(sorted(tool_names))


def _max_identical_run(signatures: list[tuple[str, ...] | None]) -> int:
    longest = 0
    current = 0
    previous: tuple[str, ...] | None = None
    for signature in signatures:
        if signature is not None and signature == previous:
            current += 1
        elif signature is not None:
            current = 1
        else:
            current = 0
        previous = signature
        longest = max(longest, current)
    return longest


def _audit_episode(samples: list[dict[str, Any]]) -> tuple[int, int, int]:
    exact: list[tuple[str, ...] | None] = []
    names: list[tuple[str, ...] | None] = []
    n_turns = 0
    for sample in sorted(samples, key=_segment_index):
        for turn in _assistant_turns(str(sample.get("response") or "")):
            n_turns += 1
            signatures = _turn_signatures(turn)
            if signatures is None:
                exact.append(None)
                names.append(None)
            else:
                exact.append(signatures[0])
                names.append(signatures[1])
    return _max_identical_run(exact), _max_identical_run(names), n_turns


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config, sources in RUNS.items():
        for steps, root in sources:
            for step in steps:
                path = root / f"rollout_{step}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
                for sample in payload["samples"]:
                    groups[_episode_key(sample)].append(sample)
                audits = [_audit_episode(members) for members in groups.values()]
                if len(audits) != 128:
                    raise RuntimeError(f"{config} step {step}: expected 128 episodes, found {len(audits)}")
                row: dict[str, Any] = {
                    "config": config,
                    "step": step,
                    "episodes": len(audits),
                    "assistant_turns": sum(item[2] for item in audits),
                }
                for threshold in (3, 4, 5):
                    row[f"exact_ge{threshold}"] = sum(item[0] >= threshold for item in audits)
                    row[f"names_ge{threshold}"] = sum(item[1] >= threshold for item in audits)
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del payload, groups, audits
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], *, prefix: str, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    colors = {"8 GPU": "#2563eb", "16 GPU": "#dc2626"}
    for axis, threshold in zip(axes, (3, 4, 5), strict=True):
        for config in RUNS:
            selected = sorted((row for row in rows if row["config"] == config), key=lambda row: row["step"])
            axis.plot(
                [row["step"] for row in selected],
                [row[f"{prefix}_ge{threshold}"] for row in selected],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=colors[config],
                label=config,
            )
        axis.axvline(10, color="#6b7280", linestyle="--", linewidth=1, alpha=0.7)
        axis.set_title(f"Run length >= {threshold}")
        axis.set_xlabel("Training step")
        axis.set_xticks(range(0, 22, 2))
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Episodes with at least one run (out of 128)")
    axes[-1].legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect()
    write_csv(rows, args.output_dir / "tool_signature_runs.csv")
    plot(
        rows,
        prefix="exact",
        output=args.output_dir / "tool_signature_runs_exact.png",
        title="Consecutive identical exact tool-call signatures",
    )
    plot(
        rows,
        prefix="names",
        output=args.output_dir / "tool_signature_runs_names.png",
        title="Consecutive identical tool-name signatures",
    )


if __name__ == "__main__":
    main()
