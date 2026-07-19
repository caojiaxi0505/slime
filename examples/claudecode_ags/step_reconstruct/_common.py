"""Shared helpers for step-level GRPO capture / workspace rebuild."""

from __future__ import annotations

import re

# Snapshots live outside the repo workdir so they never pollute ``git diff``.
SNAP_DIR = "/home/agent/.cagent_snapshots"
SNAP_SCRIPT = "/home/agent/.cagent_snapshot.sh"
METADATA_SCRIPT = "/home/agent/.cagent_workspace_metadata.py"
TRACKED_METADATA_PATHS = "/home/agent/.cagent_snapshots/.metadata_paths.json"
CC_PROJECTS_DIR = "/home/agent/.claude/projects"
CC_SETTINGS_PATH = "/home/agent/.claude/settings.json"

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")


async def workspace_diff(sb, workdir: str) -> str:
    """Cumulative diff vs HEAD (staged + unstaged), Path A exclude set."""
    cmd = (
        f"cd {workdir} && git add -N . && "
        f"git diff HEAD --binary -- . "
        f"':(exclude)PROBLEM_STATEMENT.md' "
        f"':(exclude)claude_code_trajectory.jsonl' "
        f"':(exclude).cagent_done' ':(exclude).cagent_run.sh' "
        f"':(exclude).harness/**'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


def normalize_diff(text: str) -> str:
    """Drop ``index <sha>..<sha>`` lines and trailing whitespace."""
    out: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("index "):
            continue
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return "\n".join(out).strip()


def changed_paths(text: str) -> set[str]:
    """Set of ``b/`` paths touched by a unified diff."""
    paths: set[str] = set()
    for line in (text or "").splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            paths.add(m.group(2).strip())
    return paths
