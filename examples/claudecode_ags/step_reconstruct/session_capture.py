"""Stage-1 session capture: SessionBundle + PostToolUse snapshot helpers.

Full live AGS Stage-1 runner is wired from ``hybrid_generate`` using Path A
``agent_runtime``; this module owns the on-disk bundle format and hook install.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import tarfile
from dataclasses import asdict, dataclass, field
from typing import Any

from examples.claudecode_ags.step_reconstruct import _common

logger = logging.getLogger(__name__)

_SNAPSHOT_SCRIPT = r"""#!/bin/bash
set +e
WORKDIR="${1:-/testbed}"
SOURCE="${2:-hook}"
SNAP_DIR="/home/agent/.cagent_snapshots"
mkdir -p "$SNAP_DIR" 2>/dev/null
CNT="$SNAP_DIR/.counter"
LOCK="$SNAP_DIR/.lock"

PAYLOAD=""
if [ ! -t 0 ]; then PAYLOAD="$(cat 2>/dev/null)"; fi

_snapshot() {
  local seq sid tool
  seq=$(( $(cat "$CNT" 2>/dev/null || echo 0) + 1 ))
  echo "$seq" > "$CNT"
  sid=$(printf '%04d' "$seq")
  cd "$WORKDIR" 2>/dev/null || return 0
  git add -N . >/dev/null 2>&1
  git diff HEAD --binary -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude)claude_code_trajectory.jsonl' ':(exclude).cagent_done' ':(exclude).cagent_run.sh' > "$SNAP_DIR/step_${sid}.diff" 2>/dev/null
  printf '%s' "$PAYLOAD" > "$SNAP_DIR/step_${sid}.payload.json" 2>/dev/null
  tool=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
  printf '{"seq": %s, "ts": %s, "source": "%s", "tool_name": "%s"}\n' "$seq" "$(date +%s)" "$SOURCE" "${tool}" >> "$SNAP_DIR/index.jsonl"
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK"
  flock 9
  _snapshot
  flock -u 9
else
  _snapshot
fi
exit 0
"""


@dataclass
class StepRecord:
    seq: int
    ts: int = 0
    source: str = "hook"
    tool_name: str = ""
    tool_use_id: str = ""
    diff_file: str = ""


@dataclass
class SessionBundle:
    """Everything needed to rebuild any prefix state of a Stage-1 trajectory."""

    instance_id: str
    session_id: str
    cc_session_id: str
    task_metadata: dict[str, Any]
    steps: list[StepRecord] = field(default_factory=list)
    cc_session_files: list[str] = field(default_factory=list)
    segments_summary: list[dict[str, Any]] = field(default_factory=list)
    claude_exit_code: int | None = None
    snapshots_rel: str = ".cagent_snapshots"
    projects_rel: str = "cc_projects"
    transcript_rel: str = "transcript.jsonl"
    final_diff_rel: str = "final.diff"
    dir: str = ""

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        payload = asdict(self)
        payload.pop("dir", None)
        with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def load(cls, out_dir: str) -> "SessionBundle":
        with open(os.path.join(out_dir, "bundle.json"), encoding="utf-8") as f:
            data = json.load(f)
        steps = [StepRecord(**s) for s in data.pop("steps", [])]
        bundle = cls(steps=steps, **data)
        bundle.dir = out_dir
        return bundle

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def step_diff(self, step_t: int) -> str:
        rec = self.steps[step_t]
        path = os.path.join(self.dir, rec.diff_file)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def final_diff(self) -> str:
        path = os.path.join(self.dir, self.final_diff_rel)
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()


async def install_snapshot_hook(sb, workdir: str) -> None:
    """Install snapshot script + Claude Code PostToolUse hook."""
    await sb.write_file(_common.SNAP_SCRIPT, _SNAPSHOT_SCRIPT, user="agent")
    await sb.exec(
        f"chmod +x {shlex.quote(_common.SNAP_SCRIPT)} && "
        f"mkdir -p {shlex.quote(_common.SNAP_DIR)} && "
        f"chown -R agent:agent {shlex.quote(_common.SNAP_DIR)} {shlex.quote(_common.SNAP_SCRIPT)}",
        user="root",
        timeout=60,
        check=False,
    )

    settings: dict[str, Any] = {}
    existing = await sb.read_file(_common.CC_SETTINGS_PATH, user="agent")
    if existing and str(existing).strip():
        try:
            settings = json.loads(existing)
        except json.JSONDecodeError:
            settings = {}
    settings.setdefault("hasCompletedOnboarding", True)
    settings.setdefault("bypassPermissionsModeAccepted", True)
    hook_cmd = f"bash {_common.SNAP_SCRIPT} {shlex.quote(workdir)} hook"
    settings["hooks"] = {
        "PostToolUse": [
            {"matcher": "*", "hooks": [{"type": "command", "command": hook_cmd, "timeout": 60}]}
        ]
    }
    await sb.write_file(_common.CC_SETTINGS_PATH, json.dumps(settings, indent=2), user="agent")
    logger.info("[step_reconstruct] installed PostToolUse snapshot hook for %s", workdir)


def steps_from_diff_files(bundle_dir: str, diffs: list[str], *, snapshots_rel: str = ".cagent_snapshots") -> list[StepRecord]:
    """Write diffs under snapshots_rel and return StepRecords (test helper)."""
    snap = os.path.join(bundle_dir, snapshots_rel)
    os.makedirs(snap, exist_ok=True)
    steps: list[StepRecord] = []
    for i, text in enumerate(diffs):
        rel = os.path.join(snapshots_rel, f"step_{i:04d}.diff")
        with open(os.path.join(bundle_dir, rel), "w", encoding="utf-8") as f:
            f.write(text)
        steps.append(StepRecord(seq=i + 1, source="hook", diff_file=rel))
    return steps


async def pull_remote_dir(sb, remote_dir: str, local_parent: str) -> str | None:
    """tar+base64 a sandbox dir to ``local_parent/<basename>``."""
    parent = os.path.dirname(remote_dir.rstrip("/")) or "/"
    name = os.path.basename(remote_dir.rstrip("/"))
    cmd = (
        f"test -d {shlex.quote(remote_dir)} && cd {shlex.quote(parent)} && "
        f"tar czf - {shlex.quote(name)} | base64"
    )
    ec, out, _ = await sb.exec(cmd, user="agent", timeout=240, check=False)
    if ec != 0 or not (out or "").strip():
        return None
    raw = base64.b64decode("".join(str(out).split()))
    os.makedirs(local_parent, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(local_parent, filter="data")
    return os.path.join(local_parent, name)


def _tool_use_id_from_payload(snap_dir: str, sid: str) -> str:
    path = os.path.join(snap_dir, f"step_{sid}.payload.json")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    return str(obj.get("tool_use_id") or "")


def build_hook_steps_from_snap_dir(snap_dir: str, bundle_dir: str) -> list[StepRecord]:
    """Build StepRecords from pulled snapshots; keep PostToolUse hook only.

    Prefer ``index.jsonl`` source==hook ordering; fall back to sorted
    ``step_*.diff`` files when the index is missing.
    """
    steps: list[StepRecord] = []
    index_path = os.path.join(snap_dir, "index.jsonl")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(obj.get("source") or "hook") != "hook":
                    continue
                seq = int(obj.get("seq") or 0)
                sid = f"{seq:04d}"
                rel = os.path.join(os.path.basename(snap_dir), f"step_{sid}.diff")
                # snap_dir may be bundle_dir/.cagent_snapshots
                if os.path.basename(snap_dir) != ".cagent_snapshots":
                    rel = os.path.relpath(os.path.join(snap_dir, f"step_{sid}.diff"), bundle_dir)
                else:
                    rel = os.path.join(".cagent_snapshots", f"step_{sid}.diff")
                abs_diff = os.path.join(bundle_dir, rel)
                if not os.path.isfile(abs_diff):
                    # try local snap_dir directly
                    candidate = os.path.join(snap_dir, f"step_{sid}.diff")
                    if os.path.isfile(candidate):
                        rel = os.path.relpath(candidate, bundle_dir)
                    else:
                        continue
                steps.append(
                    StepRecord(
                        seq=seq,
                        ts=int(obj.get("ts") or 0),
                        source="hook",
                        tool_name=str(obj.get("tool_name") or ""),
                        tool_use_id=_tool_use_id_from_payload(snap_dir, sid),
                        diff_file=rel,
                    )
                )
        steps.sort(key=lambda s: s.seq)
        return steps

    # Fallback: sorted step_*.diff
    names = sorted(n for n in os.listdir(snap_dir) if n.startswith("step_") and n.endswith(".diff"))
    for i, name in enumerate(names):
        rel = os.path.relpath(os.path.join(snap_dir, name), bundle_dir)
        sid = name[len("step_") : -len(".diff")]
        steps.append(
            StepRecord(
                seq=i + 1,
                source="hook",
                tool_use_id=_tool_use_id_from_payload(snap_dir, sid),
                diff_file=rel,
            )
        )
    return steps


async def capture_snapshots_to_bundle(
    sb,
    *,
    out_dir: str,
    workdir: str,
    instance_id: str,
    session_id: str,
    task_metadata: dict[str, Any],
    final_diff: str,
    claude_exit_code: int | None = None,
) -> SessionBundle:
    """Pull ``.cagent_snapshots`` and write a SessionBundle under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    snap_local = await pull_remote_dir(sb, _common.SNAP_DIR, out_dir)
    steps: list[StepRecord] = []
    if snap_local and os.path.isdir(snap_local):
        # Normalize dirname to .cagent_snapshots for stable relative paths
        dest = os.path.join(out_dir, ".cagent_snapshots")
        if os.path.abspath(snap_local) != os.path.abspath(dest):
            if os.path.isdir(dest):
                # already extracted under expected name
                pass
            else:
                os.rename(snap_local, dest)
            snap_local = dest
        steps = build_hook_steps_from_snap_dir(snap_local, out_dir)

    final_path = os.path.join(out_dir, "final.diff")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(final_diff or "")

    bundle = SessionBundle(
        instance_id=instance_id,
        session_id=session_id,
        cc_session_id="",
        task_metadata=dict(task_metadata),
        steps=steps,
        claude_exit_code=claude_exit_code,
        dir=out_dir,
    )
    bundle.save(out_dir)
    logger.info(
        "[step_reconstruct] captured bundle %s steps=%d final_diff=%dB",
        instance_id,
        len(steps),
        len(final_diff or ""),
    )
    return bundle
