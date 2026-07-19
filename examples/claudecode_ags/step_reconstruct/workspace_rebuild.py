"""Rebuild prefix state ``s_t`` and prefix-reseed continuation bridge."""

from __future__ import annotations

import logging
import os
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from examples.claudecode_ags.step_reconstruct import _common
from examples.claudecode_ags.step_reconstruct.session_capture import (
    SessionBundle,
    _WORKSPACE_METADATA_SCRIPT,
)
from examples.claudecode_ags.step_reconstruct.transcript import build_transcript_prefix

logger = logging.getLogger(__name__)


async def restore_workspace_metadata(sb, *, workdir: str, metadata_path: str) -> bool:
    """Restore and verify mode/symlink/mtime after the cumulative diff."""
    if not metadata_path or not os.path.isfile(metadata_path):
        logger.warning("[step_reconstruct] workspace metadata file missing: %s", metadata_path)
        return False
    with open(metadata_path, encoding="utf-8") as handle:
        manifest = handle.read()
    remote_manifest = "/tmp/slime_workspace_metadata.json"
    await sb.write_file(_common.METADATA_SCRIPT, _WORKSPACE_METADATA_SCRIPT, user="agent")
    await sb.write_file(remote_manifest, manifest, user="agent")
    await sb.exec(
        f"chmod +x {shlex.quote(_common.METADATA_SCRIPT)}",
        user="root",
        timeout=60,
        check=True,
    )
    command = (
        f"python3 {shlex.quote(_common.METADATA_SCRIPT)} restore "
        f"{shlex.quote(workdir)} {shlex.quote(remote_manifest)}"
    )
    ec, _out, err = await sb.exec(command, user="agent", timeout=120, check=False)
    if ec != 0:
        logger.warning("[step_reconstruct] workspace metadata restore failed: %s", (err or "")[:500])
        return False
    return True


async def apply_diff(sb, workdir: str, diff_text: str, *, reverse: bool = False) -> bool:
    """Apply a unified diff inside the sandbox workdir. Returns success."""
    if not (diff_text or "").strip():
        return True
    remote = "/tmp/_step_reconstruct_apply.diff"
    await sb.write_file(remote, diff_text, user="agent")
    reverse_flag = "--reverse " if reverse else ""
    cmd = (
        f"cd {shlex.quote(workdir)} && "
        f"git apply {reverse_flag}--whitespace=nowarn {shlex.quote(remote)} "
        f"|| git apply --3way {reverse_flag}--whitespace=nowarn {shlex.quote(remote)}"
    )
    ec, _, err = await sb.exec(cmd, user="agent", timeout=120, check=False)
    if ec != 0:
        logger.warning("[step_reconstruct] git apply failed: %s", (err or "")[:300])
        return False
    return True


async def restore_cumulative_diff(
    sb,
    *,
    workdir: str,
    initial_diff: str,
    target_diff: str,
) -> bool:
    """Move a freshly prepared baseline to one captured cumulative state.

    ``prepare_workspace`` already recreates the Stage-1 baseline. Therefore
    ``initial.diff`` is first used as a fingerprint and must not be blindly
    applied a second time. If it is non-empty, reverse just those recorded
    baseline changes before applying the target cumulative diff (which is
    itself relative to ``HEAD``).
    """
    current = await _common.workspace_diff(sb, workdir)
    if _common.normalize_diff(current) != _common.normalize_diff(initial_diff):
        logger.warning(
            "[step_reconstruct] prepared baseline mismatch expected_paths=%s actual_paths=%s",
            sorted(_common.changed_paths(initial_diff)),
            sorted(_common.changed_paths(current)),
        )
        return False

    if _common.normalize_diff(target_diff) == _common.normalize_diff(initial_diff):
        return True

    if (initial_diff or "").strip():
        if not await apply_diff(sb, workdir, initial_diff, reverse=True):
            logger.warning("[step_reconstruct] failed to reverse captured initial.diff")
            return False
        clean = await _common.workspace_diff(sb, workdir)
        if _common.normalize_diff(clean):
            logger.warning(
                "[step_reconstruct] initial.diff reverse did not restore HEAD; remaining_paths=%s",
                sorted(_common.changed_paths(clean)),
            )
            return False

    if not await apply_diff(sb, workdir, target_diff):
        return False
    actual = await _common.workspace_diff(sb, workdir)
    return _common.normalize_diff(actual) == _common.normalize_diff(target_diff)


@asynccontextmanager
async def rebuilt_workspace(bundle: SessionBundle, step_t: int) -> AsyncIterator[tuple[Any, bool]]:
    """Yield ``(sb, applied)`` rebuilt to prefix ``s_t`` using Path A sandbox."""
    from examples.claudecode_ags import agent_runtime
    from slime.agent.sandbox import make_sandbox

    md = bundle.task_metadata
    image = md["image"]
    workdir = md["workdir"]
    if step_t < -1 or step_t >= bundle.num_steps:
        raise IndexError(f"step_t out of range: {step_t} for {bundle.num_steps} steps")
    initial_diff = bundle.initial_diff()
    diff_text = initial_diff if step_t == -1 else bundle.step_diff(step_t)

    async with make_sandbox(image) as sb:
        await agent_runtime.prepare_workspace(
            sb,
            workdir=workdir,
            problem_statement=str(md.get("problem_statement") or ""),
            instance_id=str(md.get("instance_id") or bundle.instance_id or ""),
            data_source=str(md.get("data_source") or ""),
            base_commit=str(md.get("base_commit") or ""),
            swe_smith_bug_patch=md.get("swe_smith_bug_patch"),
            pre_commands=md.get("pre_commands") or "",
            install_config=md.get("install_config") or {},
            rollout_side=True,
        )
        verified = await restore_cumulative_diff(
            sb,
            workdir=workdir,
            initial_diff=initial_diff,
            target_diff=diff_text,
        )
        if verified:
            verified = await restore_workspace_metadata(
                sb,
                workdir=workdir,
                metadata_path=bundle.metadata_path(step_t),
            )
        actual = await _common.workspace_diff(sb, workdir) if verified else ""
        if not verified:
            logger.warning(
                "[step_reconstruct] rebuild verification FAILED at step %s for %s expected_paths=%s actual_paths=%s",
                step_t,
                bundle.instance_id,
                sorted(_common.changed_paths(diff_text)),
                sorted(_common.changed_paths(actual)),
            )
        yield sb, verified


async def verify_rebuild(sb, bundle: SessionBundle, step_t: int) -> tuple[bool, dict[str, Any]]:
    """Path-set equality is the primary gate; normalized text is secondary."""
    md = bundle.task_metadata
    expected = bundle.initial_diff() if step_t == -1 else bundle.step_diff(step_t)
    actual = await _common.workspace_diff(sb, md["workdir"])

    exp_paths = _common.changed_paths(expected)
    act_paths = _common.changed_paths(actual)
    paths_match = exp_paths == act_paths
    text_match = _common.normalize_diff(expected) == _common.normalize_diff(actual)
    details = {
        "step_t": step_t,
        "paths_match": paths_match,
        "text_match": text_match,
        "expected_paths": sorted(exp_paths),
        "actual_paths": sorted(act_paths),
        "missing_paths": sorted(exp_paths - act_paths),
        "extra_paths": sorted(act_paths - exp_paths),
    }
    return paths_match and text_match, details


def truncate_transcript_prefix(
    transcript_jsonl: str,
    step_t: int,
    *,
    snapshot_tool_use_ids: list[str],
    initial_prompt: str,
) -> str:
    """Backward-compatible name for the structured prefix builder."""
    return build_transcript_prefix(
        transcript_jsonl,
        snapshot_tool_use_ids,
        step_t,
        initial_prompt=initial_prompt,
    )


async def write_prefix_reseed_file(sb, *, prefix_text: str, remote_path: str = "/tmp/cc_prefix.jsonl") -> str:
    await sb.write_file(remote_path, prefix_text, user="agent")
    return remote_path


async def resume_and_run(
    sb,
    *,
    workdir: str,
    prefix_text: str,
    env: dict[str, str],
    time_budget_sec: int,
    runner=None,
) -> dict[str, Any]:
    """Prefix re-seed then continue CC via stream-json stdin.

    ``runner`` is an injectable async callable ``(sb, **kwargs) -> dict`` for
    tests. Default uses ``agent_runtime.run_claude_with_prefix``.
    """
    if not (prefix_text or "").strip():
        raise RuntimeError("prefix transcript is empty; refusing fresh-agent fallback")
    prefix_path = await write_prefix_reseed_file(sb, prefix_text=prefix_text)
    if runner is not None:
        return await runner(
            sb,
            workdir=workdir,
            env=env,
            time_budget_sec=time_budget_sec,
            prefix_path=prefix_path,
        )

    from examples.claudecode_ags import agent_runtime

    return await agent_runtime.run_claude_with_prefix(
        sb,
        workdir=workdir,
        env=env,
        time_budget_sec=time_budget_sec,
        prefix_path=prefix_path,
    )
