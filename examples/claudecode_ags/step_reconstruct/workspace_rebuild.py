"""Rebuild prefix state ``s_t`` and prefix-reseed continuation bridge."""

from __future__ import annotations

import logging
import os
import shlex
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from examples.claudecode_ags.step_reconstruct import _common
from examples.claudecode_ags.step_reconstruct.session_capture import SessionBundle

logger = logging.getLogger(__name__)


async def apply_diff(sb, workdir: str, diff_text: str) -> bool:
    """Apply a unified diff inside the sandbox workdir. Returns success."""
    if not (diff_text or "").strip():
        return True
    remote = "/tmp/_step_reconstruct_apply.diff"
    await sb.write_file(remote, diff_text, user="agent")
    cmd = (
        f"cd {shlex.quote(workdir)} && "
        f"git apply --whitespace=nowarn {shlex.quote(remote)} "
        f"|| git apply --3way --whitespace=nowarn {shlex.quote(remote)}"
    )
    ec, _, err = await sb.exec(cmd, user="agent", timeout=120, check=False)
    if ec != 0:
        logger.warning("[step_reconstruct] git apply failed: %s", (err or "")[:300])
        return False
    return True


@asynccontextmanager
async def rebuilt_workspace(bundle: SessionBundle, step_t: int) -> AsyncIterator[tuple[Any, bool]]:
    """Yield ``(sb, applied)`` rebuilt to prefix ``s_t`` using Path A sandbox."""
    from examples.claudecode_ags import agent_runtime
    from slime.agent.sandbox import make_sandbox

    md = bundle.task_metadata
    image = md["image"]
    workdir = md["workdir"]
    if step_t == -1:
        diff_text = bundle.final_diff()
    else:
        diff_text = bundle.step_diff(step_t)

    async with make_sandbox(image) as sb:
        await agent_runtime.prepare_workspace(
            sb,
            workdir=workdir,
            problem_statement=str(md.get("problem_statement") or ""),
        )
        applied = await apply_diff(sb, workdir, diff_text)
        if not applied:
            logger.warning(
                "[step_reconstruct] git apply FAILED at step %s for %s",
                step_t,
                bundle.instance_id,
            )
        yield sb, applied


async def verify_rebuild(sb, bundle: SessionBundle, step_t: int) -> tuple[bool, dict[str, Any]]:
    """Path-set equality is the primary gate; normalized text is secondary."""
    md = bundle.task_metadata
    expected = bundle.final_diff() if step_t == -1 else bundle.step_diff(step_t)
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
    return paths_match, details


def truncate_transcript_prefix(transcript_jsonl: str, step_t: int) -> str:
    """Keep transcript lines up through the ``step_t``-th tool_result (0-based).

    Simple heuristic for tests / re-seed: count ``tool_result`` events; keep
    lines until count > step_t. If fewer events exist, return the full transcript.
    """
    lines = (transcript_jsonl or "").splitlines()
    if step_t < 0:
        return transcript_jsonl or ""
    kept: list[str] = []
    tool_results = 0
    for line in lines:
        kept.append(line)
        if '"tool_result"' in line or '"type":"tool_result"' in line.replace(" ", ""):
            if tool_results >= step_t:
                break
            tool_results += 1
    return "\n".join(kept) + ("\n" if kept else "")


async def write_prefix_reseed_file(sb, *, prefix_text: str, remote_path: str = "/tmp/cc_prefix.jsonl") -> str:
    await sb.write_file(remote_path, prefix_text, user="agent")
    return remote_path


async def resume_and_run(
    sb,
    *,
    workdir: str,
    prefix_text: str,
    prompt: str,
    env: dict[str, str],
    time_budget_sec: int,
    runner=None,
) -> dict[str, Any]:
    """Prefix re-seed then continue CC via stream-json stdin.

    ``runner`` is an injectable async callable ``(sb, **kwargs) -> dict`` for
    tests. Default uses ``agent_runtime.run_claude_with_prefix``.
    """
    prefix_path = await write_prefix_reseed_file(sb, prefix_text=prefix_text)
    if runner is not None:
        return await runner(
            sb,
            workdir=workdir,
            prompt=prompt,
            env=env,
            time_budget_sec=time_budget_sec,
            prefix_path=prefix_path,
        )

    from examples.claudecode_ags import agent_runtime

    return await agent_runtime.run_claude_with_prefix(
        sb,
        workdir=workdir,
        prompt=prompt,
        env=env,
        time_budget_sec=time_budget_sec,
        prefix_path=prefix_path,
    )
