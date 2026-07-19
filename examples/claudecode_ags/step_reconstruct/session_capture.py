"""Stage-1 session capture: SessionBundle + PostToolUse snapshot helpers.

Full live AGS Stage-1 runner is wired from ``hybrid_generate`` using Path A
``agent_runtime``; this module owns the on-disk bundle format and hook install.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import os
import shlex
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

from examples.claudecode_ags.step_reconstruct import _common
from examples.claudecode_ags.step_reconstruct.native_session import parse_native_session
from examples.claudecode_ags.step_reconstruct.transcript import validate_transcript
from slime.agent.adapters.anthropic_segmented import (
    PromptCheckpoint,
    canonical_sha256,
    prompt_ids_sha256,
)

logger = logging.getLogger(__name__)

_WORKSPACE_METADATA_SCRIPT = r'''#!/usr/bin/env python3
import json
import hashlib
import os
import stat
import subprocess
import sys

VERSION = 1


def _inside(workdir, value):
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value if os.path.isabs(value) else os.path.join(workdir, value)
    candidate = os.path.abspath(candidate)
    try:
        if os.path.commonpath([workdir, candidate]) != workdir:
            return None
    except ValueError:
        return None
    rel = os.path.relpath(candidate, workdir)
    if rel in {".git", ".harness"} or rel.startswith(".git/") or rel.startswith(".harness/"):
        return None
    return rel


def _ignored_inside(workdir, value):
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value if os.path.isabs(value) else os.path.join(workdir, value)
    candidate = os.path.abspath(candidate)
    try:
        if os.path.commonpath([workdir, candidate]) != workdir:
            return False
    except ValueError:
        return False
    rel = os.path.relpath(candidate, workdir)
    return rel in {".git", ".harness"} or rel.startswith(".git/") or rel.startswith(".harness/")


def _payload_paths(workdir, payload_path):
    try:
        with open(payload_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return set(), set()
    found = set()
    unsupported = set()

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"file_path", "path", "notebook_path"}:
                    rel = _inside(workdir, child)
                    if rel:
                        found.add(rel)
                    elif (
                        isinstance(child, str)
                        and child.strip()
                        and not _ignored_inside(workdir, child)
                    ):
                        unsupported.add(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found, unsupported


def _git_paths(workdir):
    paths = set()
    commands = [
        ["git", "-C", workdir, "diff", "--name-only", "-z", "HEAD", "--"],
        ["git", "-C", workdir, "ls-files", "--others", "--exclude-standard", "-z"],
    ]
    for command in commands:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        for raw in proc.stdout.split(b"\0"):
            if not raw:
                continue
            rel = _inside(workdir, os.path.join(workdir, os.fsdecode(raw)))
            if rel:
                paths.add(rel)
    return paths


def _kind(mode):
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "dir"
    return "other"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def capture(workdir, output_path, payload_path, tracked_path):
    workdir = os.path.abspath(workdir)
    tracked = {"PROBLEM_STATEMENT.md"}
    unsupported_tracked_path = tracked_path + ".unsupported"
    unsupported_tracked = set()
    try:
        with open(tracked_path, encoding="utf-8") as handle:
            tracked.update(json.load(handle))
    except Exception:
        pass
    try:
        with open(unsupported_tracked_path, encoding="utf-8") as handle:
            unsupported_tracked.update(json.load(handle))
    except Exception:
        pass
    payload_paths, unsupported_paths = _payload_paths(workdir, payload_path)
    tracked.update(payload_paths)
    unsupported_tracked.update(unsupported_paths)
    tracked.update(_git_paths(workdir))
    os.makedirs(os.path.dirname(tracked_path), exist_ok=True)
    with open(tracked_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(tracked), handle, separators=(",", ":"))
    with open(unsupported_tracked_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(unsupported_tracked), handle, separators=(",", ":"))

    records = []
    for rel in sorted(tracked):
        path = os.path.join(workdir, rel)
        try:
            value = os.lstat(path)
        except FileNotFoundError:
            records.append({"path": rel, "kind": "missing"})
            continue
        kind = _kind(value.st_mode)
        record = {
            "path": rel,
            "kind": kind,
            "mode": stat.S_IMODE(value.st_mode),
            "uid": value.st_uid,
            "gid": value.st_gid,
            "atime_ns": value.st_atime_ns,
            "mtime_ns": value.st_mtime_ns,
        }
        if kind == "symlink":
            record["link_target"] = os.readlink(path)
        elif kind == "file":
            record["sha256"] = _sha256(path)
        records.append(record)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": VERSION,
                "workdir": workdir,
                "records": records,
                "unsupported_paths": sorted(unsupported_tracked),
            },
            handle,
            separators=(",", ":"),
        )


def _actual_kind(path):
    try:
        return _kind(os.lstat(path).st_mode)
    except FileNotFoundError:
        return "missing"


def restore(workdir, manifest_path, verify_only=False):
    workdir = os.path.abspath(workdir)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != VERSION:
        raise RuntimeError("unsupported workspace metadata version")
    if manifest.get("unsupported_paths"):
        raise RuntimeError("workspace metadata contains paths outside workdir")
    records = manifest.get("records") or []
    if not verify_only:
        # Files first and directories last so child operations cannot perturb a
        # restored directory timestamp.
        ordered = sorted(records, key=lambda item: item.get("kind") == "dir")
        for record in ordered:
            rel = _inside(workdir, os.path.join(workdir, record.get("path", "")))
            if not rel:
                raise RuntimeError("unsafe metadata path")
            path = os.path.join(workdir, rel)
            if record.get("kind") == "missing":
                if os.path.lexists(path):
                    raise RuntimeError("expected missing path: " + rel)
                continue
            if _actual_kind(path) != record.get("kind"):
                raise RuntimeError("path kind mismatch: " + rel)
            if record.get("kind") != "symlink":
                os.chmod(path, int(record["mode"]), follow_symlinks=False)
            os.utime(
                path,
                ns=(int(record["atime_ns"]), int(record["mtime_ns"])),
                follow_symlinks=False,
            )

    errors = []
    for record in records:
        rel = _inside(workdir, os.path.join(workdir, record.get("path", "")))
        if not rel:
            errors.append("unsafe:" + str(record.get("path")))
            continue
        path = os.path.join(workdir, rel)
        actual_kind = _actual_kind(path)
        if actual_kind != record.get("kind"):
            errors.append("kind:" + rel)
            continue
        if actual_kind == "missing":
            continue
        value = os.lstat(path)
        if stat.S_IMODE(value.st_mode) != int(record["mode"]):
            errors.append("mode:" + rel)
        if value.st_mtime_ns != int(record["mtime_ns"]):
            errors.append("mtime:" + rel)
        if actual_kind == "symlink" and os.readlink(path) != record.get("link_target"):
            errors.append("link:" + rel)
        if actual_kind == "file" and _sha256(path) != record.get("sha256"):
            errors.append("content:" + rel)
    if errors:
        raise RuntimeError("workspace metadata verification failed: " + ",".join(errors[:20]))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("usage: metadata.py capture|restore|verify WORKDIR MANIFEST [PAYLOAD TRACKED]")
    command, workdir, manifest = sys.argv[1:4]
    if command == "capture":
        capture(workdir, manifest, sys.argv[4], sys.argv[5])
    elif command == "restore":
        restore(workdir, manifest, verify_only=False)
    elif command == "verify":
        restore(workdir, manifest, verify_only=True)
    else:
        raise SystemExit("unknown command: " + command)
'''

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
  git diff HEAD --binary -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude)claude_code_trajectory.jsonl' ':(exclude).cagent_done' ':(exclude).cagent_run.sh' ':(exclude).harness/**' > "$SNAP_DIR/step_${sid}.diff" 2>/dev/null
  printf '%s' "$PAYLOAD" > "$SNAP_DIR/step_${sid}.payload.json" 2>/dev/null
  python3 /home/agent/.cagent_workspace_metadata.py capture "$WORKDIR" \
    "$SNAP_DIR/step_${sid}.metadata.json" "$SNAP_DIR/step_${sid}.payload.json" \
    "$SNAP_DIR/.metadata_paths.json" 2>&1 || \
    printf 'WARNING: workspace metadata capture failed for step %s\n' "$sid" >&2
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
    metadata_file: str = ""


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
    checkpoints_rel: str = "prompt_checkpoints.json.gz"
    initial_diff_rel: str = "initial.diff"
    final_diff_rel: str = "final.diff"
    initial_metadata_rel: str = ".cagent_snapshots/initial.metadata.json"
    initial_state_captured: bool = False
    prompt_checkpoints_valid: bool = False
    prompt_checkpoints_error: str = ""
    prompt_checkpoint_count: int = 0
    native_session_valid: bool = False
    native_session_error: str = ""
    workspace_metadata_valid: bool = False
    workspace_metadata_error: str = ""
    transcript_valid: bool = False
    transcript_error: str = ""
    transcript_bytes: int = 0
    transcript_events: int = 0
    transcript_conversation_events: int = 0
    transcript_tool_uses: int = 0
    transcript_tool_results: int = 0
    transcript_last_aligned_tool_use_id: str = ""
    dir: str = ""

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        payload = asdict(self)
        payload.pop("dir", None)
        with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def load(cls, out_dir: str) -> SessionBundle:
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
        if step_t < 0:
            raise IndexError(f"step_t must be >= 0; got {step_t}")
        rec = self.steps[step_t]
        path = os.path.join(self.dir, rec.diff_file)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def initial_diff(self) -> str:
        path = os.path.join(self.dir, self.initial_diff_rel)
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def final_diff(self) -> str:
        path = os.path.join(self.dir, self.final_diff_rel)
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def transcript(self) -> str:
        path = os.path.join(self.dir, self.transcript_rel)
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def prompt_checkpoints(self) -> list[dict[str, Any]]:
        path = os.path.join(self.dir, self.checkpoints_rel)
        if not os.path.isfile(path):
            return []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("prompt checkpoint file must contain a list")
        return payload

    def checkpoint_for_tool_use_id(self, tool_use_id: str) -> dict[str, Any] | None:
        matches = [
            checkpoint
            for checkpoint in self.prompt_checkpoints()
            if str(tool_use_id) in [str(x) for x in checkpoint.get("generated_tool_use_ids") or []]
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(f"tool_use id {tool_use_id!r} maps to {len(matches)} checkpoints")
        return matches[0]

    def native_session(self) -> str:
        candidates = [
            rel
            for rel in self.cc_session_files
            if os.path.basename(rel) == f"{self.cc_session_id}.jsonl"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one native session for {self.cc_session_id!r}, got {len(candidates)}"
            )
        path = os.path.join(self.dir, candidates[0])
        with open(path, encoding="utf-8", errors="strict") as f:
            return f.read()

    def metadata_path(self, step_t: int) -> str:
        if step_t == -1:
            rel = self.initial_metadata_rel
        else:
            if step_t < 0 or step_t >= len(self.steps):
                raise IndexError(f"step_t out of range: {step_t}")
            rel = self.steps[step_t].metadata_file
        return os.path.join(self.dir, rel) if rel else ""

    def token_exact_readiness(self, step_t: int | None = None) -> tuple[bool, str]:
        """Check global state and, when supplied, one pre-turn workspace state."""
        failures: list[str] = []
        if not self.prompt_checkpoints_valid:
            failures.append(self.prompt_checkpoints_error or "prompt_checkpoints_not_validated")
        if not self.native_session_valid:
            failures.append(self.native_session_error or "native_session_not_validated")
        if step_t is not None:
            metadata_valid, metadata_error = validate_workspace_metadata_files(
                [self.metadata_path(step_t)]
            )
            if not metadata_valid:
                failures.append(metadata_error)
        return not failures, ";".join(failures)


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_gzip_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, separators=(",", ":"), default=str)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def validate_prompt_checkpoints(checkpoints: list[dict[str, Any]]) -> tuple[bool, str]:
    """Validate checkpoint structure, hashes, and unique tool-id mapping."""
    if not checkpoints:
        return False, "missing_prompt_checkpoints"
    checkpoint_ids: set[str] = set()
    tool_ids: set[str] = set()
    try:
        for raw in checkpoints:
            checkpoint = PromptCheckpoint.from_dict(raw)
            if not checkpoint.checkpoint_id or checkpoint.checkpoint_id in checkpoint_ids:
                return False, f"duplicate_or_empty_checkpoint_id:{checkpoint.checkpoint_id}"
            checkpoint_ids.add(checkpoint.checkpoint_id)
            if prompt_ids_sha256(checkpoint.prompt_ids) != checkpoint.prompt_sha256:
                return False, f"prompt_hash_mismatch:{checkpoint.checkpoint_id}"
            if canonical_sha256(checkpoint.tools_schema) != checkpoint.tools_sha256:
                return False, f"tools_hash_mismatch:{checkpoint.checkpoint_id}"
            generated_names = {
                str(tool_use_id): str(name)
                for tool_use_id, name in checkpoint.generated_tool_use_names.items()
            }
            if set(generated_names) != set(checkpoint.generated_tool_use_ids):
                return False, f"generated_tool_name_ids_mismatch:{checkpoint.checkpoint_id}"
            if any(not name for name in generated_names.values()):
                return False, f"generated_tool_name_empty:{checkpoint.checkpoint_id}"
            for tool_use_id in checkpoint.generated_tool_use_ids:
                tool_use_id = str(tool_use_id)
                if not tool_use_id or tool_use_id in tool_ids:
                    return False, f"duplicate_or_empty_generated_tool_id:{tool_use_id}"
                tool_ids.add(tool_use_id)
    except (TypeError, ValueError, KeyError) as exc:
        return False, f"invalid_prompt_checkpoint:{exc}"
    return True, ""


def validate_workspace_metadata_files(paths: list[str]) -> tuple[bool, str]:
    if not paths:
        return False, "missing_workspace_metadata"
    for path in paths:
        if not path or not os.path.isfile(path):
            return False, f"missing_workspace_metadata:{path}"
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid_workspace_metadata:{path}:{exc}"
        if manifest.get("version") != 1 or not isinstance(manifest.get("records"), list):
            return False, f"unsupported_workspace_metadata:{path}"
        if manifest.get("unsupported_paths"):
            return False, f"workspace_metadata_paths_outside_workdir:{path}"
        for record in manifest["records"]:
            if not isinstance(record, dict) or not record.get("path") or not record.get("kind"):
                return False, f"invalid_workspace_metadata_record:{path}"
            if record.get("kind") == "file" and not record.get("sha256"):
                return False, f"workspace_metadata_file_hash_missing:{path}:{record.get('path')}"
    return True, ""


async def install_snapshot_hook(sb, workdir: str) -> None:
    """Install snapshot script + Claude Code success/failure hooks."""
    # ``write_file`` may create parents as root in AGS.  Claude Code itself runs
    # as ``agent`` and silently skips native session persistence when .claude is
    # not writable.
    await sb.exec(
        "mkdir -p /home/agent/.claude/projects /home/agent/.cagent_snapshots && "
        "chown -R agent:agent /home/agent/.claude /home/agent/.cagent_snapshots",
        user="root",
        timeout=60,
        check=True,
    )
    await sb.write_file(_common.SNAP_SCRIPT, _SNAPSHOT_SCRIPT, user="agent")
    await sb.write_file(_common.METADATA_SCRIPT, _WORKSPACE_METADATA_SCRIPT, user="agent")
    await sb.exec(
        f"chmod +x {shlex.quote(_common.SNAP_SCRIPT)} && "
        f"chmod +x {shlex.quote(_common.METADATA_SCRIPT)} && "
        f"mkdir -p {shlex.quote(_common.SNAP_DIR)} && "
        f"chown -R agent:agent {shlex.quote(_common.SNAP_DIR)} "
        f"{shlex.quote(_common.SNAP_SCRIPT)} {shlex.quote(_common.METADATA_SCRIPT)}",
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
    snapshot_hook = [
        {"matcher": "*", "hooks": [{"type": "command", "command": hook_cmd, "timeout": 60}]}
    ]
    settings["hooks"] = {
        # Failed tools also produce a tool_result in the transcript. Capture
        # their unchanged workspace so transcript and snapshots remain 1:1.
        "PostToolUse": snapshot_hook,
        "PostToolUseFailure": snapshot_hook,
    }
    await sb.write_file(_common.CC_SETTINGS_PATH, json.dumps(settings, indent=2), user="agent")
    logger.info("[step_reconstruct] installed PostToolUse snapshot hook for %s", workdir)


async def capture_initial_workspace_metadata(sb, workdir: str) -> bool:
    """Capture metadata for logical snapshot ``-1`` before Claude starts."""
    empty_payload = f"{_common.SNAP_DIR}/initial.payload.json"
    output = f"{_common.SNAP_DIR}/initial.metadata.json"
    await sb.write_file(empty_payload, "{}", user="agent")
    command = (
        f"python3 {shlex.quote(_common.METADATA_SCRIPT)} capture {shlex.quote(workdir)} "
        f"{shlex.quote(output)} {shlex.quote(empty_payload)} "
        f"{shlex.quote(_common.TRACKED_METADATA_PATHS)}"
    )
    ec, _out, err = await sb.exec(command, user="agent", timeout=120, check=False)
    if ec != 0:
        logger.warning("[step_reconstruct] initial workspace metadata capture failed: %s", (err or "")[:400])
        return False
    return True


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
    """Build StepRecords from pulled snapshots; keep tool-boundary hooks only.

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
                        metadata_file=(
                            os.path.relpath(
                                os.path.join(snap_dir, f"step_{sid}.metadata.json"),
                                bundle_dir,
                            )
                            if os.path.isfile(os.path.join(snap_dir, f"step_{sid}.metadata.json"))
                            else ""
                        ),
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
                metadata_file=(
                    os.path.relpath(
                        os.path.join(snap_dir, f"step_{sid}.metadata.json"),
                        bundle_dir,
                    )
                    if os.path.isfile(os.path.join(snap_dir, f"step_{sid}.metadata.json"))
                    else ""
                ),
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
    initial_diff: str,
    final_diff: str,
    transcript_text: str | None = None,
    claude_exit_code: int | None = None,
    cc_session_id: str = "",
    prompt_checkpoints: list[dict[str, Any]] | None = None,
) -> SessionBundle:
    """Pull snapshots, prompt checkpoints, and Claude Code native state."""
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

    initial_metadata_path = os.path.join(out_dir, ".cagent_snapshots", "initial.metadata.json")
    missing_step_metadata = [step.seq for step in steps if not step.metadata_file]
    if not os.path.isfile(initial_metadata_path):
        workspace_metadata_valid = False
        workspace_metadata_error = "missing_initial_workspace_metadata"
    elif missing_step_metadata:
        workspace_metadata_valid = False
        workspace_metadata_error = f"missing_step_workspace_metadata:{missing_step_metadata[:20]}"
    else:
        workspace_metadata_valid, workspace_metadata_error = validate_workspace_metadata_files(
            [initial_metadata_path]
            + [os.path.join(out_dir, step.metadata_file) for step in steps]
        )

    if transcript_text is None:
        transcript_text = await sb.read_file(f"{workdir}/.harness/trajectory.jsonl", user="agent")
    transcript_text = str(transcript_text or "")
    validation = validate_transcript(
        transcript_text,
        [str(step.tool_use_id or "") for step in steps],
    )

    atomic_write_text(os.path.join(out_dir, "initial.diff"), initial_diff or "")
    atomic_write_text(os.path.join(out_dir, "final.diff"), final_diff or "")
    atomic_write_text(os.path.join(out_dir, "transcript.jsonl"), transcript_text)

    checkpoints = json.loads(json.dumps(prompt_checkpoints or [], ensure_ascii=False, default=str))
    checkpoints_valid, checkpoints_error = validate_prompt_checkpoints(checkpoints)
    atomic_write_gzip_json(os.path.join(out_dir, "prompt_checkpoints.json.gz"), checkpoints)

    cc_session_files: list[str] = []
    native_session_valid = False
    native_session_error = "missing_cc_session_id"
    if cc_session_id:
        native_local = await pull_remote_dir(sb, "/home/agent/.claude/projects", out_dir)
        if native_local and os.path.isdir(native_local):
            dest = os.path.join(out_dir, "cc_projects")
            if os.path.abspath(native_local) != os.path.abspath(dest):
                if not os.path.exists(dest):
                    os.rename(native_local, dest)
                native_local = dest
            for root, _dirs, files in os.walk(native_local):
                for name in files:
                    if name.endswith(".jsonl"):
                        cc_session_files.append(os.path.relpath(os.path.join(root, name), out_dir))
            cc_session_files.sort()
            target_files = [
                rel for rel in cc_session_files if os.path.basename(rel) == f"{cc_session_id}.jsonl"
            ]
            if len(target_files) != 1:
                native_session_error = f"expected_one_native_session:{cc_session_id}:found={len(target_files)}"
            else:
                try:
                    target_path = os.path.join(out_dir, target_files[0])
                    with open(target_path, encoding="utf-8", errors="strict") as handle:
                        native_rows = parse_native_session(handle.read())
                    session_ids = {
                        str(row.get("sessionId")) for row in native_rows if row.get("sessionId")
                    }
                    if session_ids != {cc_session_id}:
                        native_session_error = f"native_session_id_mismatch:{sorted(session_ids)}"
                    else:
                        native_session_valid = True
                        native_session_error = ""
                except (OSError, UnicodeError, ValueError) as exc:
                    native_session_error = f"invalid_native_session:{exc}"
        else:
            native_session_error = "claude_projects_directory_missing"

    bundle = SessionBundle(
        instance_id=instance_id,
        session_id=session_id,
        cc_session_id=cc_session_id,
        task_metadata=dict(task_metadata),
        steps=steps,
        cc_session_files=cc_session_files,
        claude_exit_code=claude_exit_code,
        initial_state_captured=True,
        prompt_checkpoints_valid=checkpoints_valid,
        prompt_checkpoints_error=checkpoints_error,
        prompt_checkpoint_count=len(checkpoints),
        native_session_valid=native_session_valid,
        native_session_error=native_session_error,
        workspace_metadata_valid=workspace_metadata_valid,
        workspace_metadata_error=workspace_metadata_error,
        transcript_valid=validation.valid,
        transcript_error=validation.error,
        transcript_bytes=len(transcript_text.encode("utf-8")),
        transcript_events=validation.event_count,
        transcript_conversation_events=validation.conversation_event_count,
        transcript_tool_uses=validation.tool_use_count,
        transcript_tool_results=validation.tool_result_count,
        transcript_last_aligned_tool_use_id=validation.last_aligned_tool_use_id,
        dir=out_dir,
    )
    bundle.save(out_dir)
    if not validation.valid:
        logger.warning(
            "[step_reconstruct] transcript invalid instance=%s session=%s bundle=%s reason=%s bytes=%d",
            instance_id,
            session_id,
            out_dir,
            validation.error,
            bundle.transcript_bytes,
        )
    if not checkpoints_valid:
        logger.warning(
            "[step_reconstruct] prompt checkpoints invalid instance=%s session=%s bundle=%s reason=%s",
            instance_id,
            session_id,
            out_dir,
            checkpoints_error,
        )
    if not native_session_valid:
        logger.warning(
            "[step_reconstruct] native Claude session invalid instance=%s session=%s cc_session=%s "
            "bundle=%s reason=%s",
            instance_id,
            session_id,
            cc_session_id,
            out_dir,
            native_session_error,
        )
    if not workspace_metadata_valid:
        logger.warning(
            "[step_reconstruct] workspace metadata invalid instance=%s session=%s bundle=%s reason=%s",
            instance_id,
            session_id,
            out_dir,
            workspace_metadata_error,
        )
    else:
        logger.info(
            "[step_reconstruct] captured bundle %s steps=%d transcript=%dB events=%d final_diff=%dB",
            instance_id,
            len(steps),
            bundle.transcript_bytes,
            bundle.transcript_events,
            len(final_diff or ""),
        )
    return bundle
