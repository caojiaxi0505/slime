"""Live AGS runners for Path A hybrid_generate (Stage-1 capture + Stage-2 branch)."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags import generate as gen
from examples.claudecode_ags.step_reconstruct import _common
from examples.claudecode_ags.step_reconstruct.edit_ppl import (
    StepTurnAlignmentError,
    align_logprobs_to_steps,
)
from examples.claudecode_ags.step_reconstruct.session_capture import (
    SessionBundle,
    capture_snapshots_to_bundle,
    install_snapshot_hook,
)
from examples.claudecode_ags.step_reconstruct.workspace_rebuild import (
    rebuilt_workspace,
    resume_and_run,
    truncate_transcript_prefix,
)
from slime.agent.sandbox import make_sandbox
from slime.agent.segment_trajectory import fan_out_sample_segments
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _bundle_root(args: Any) -> str:
    root = (
        os.environ.get("STEP_GRPO_BUNDLE_DIR")
        or getattr(args, "save", None)
        or getattr(args, "load", None)
        or "/tmp/cc_ags_step_reconstruct"
    )
    path = os.path.join(str(root), "step_reconstruct_bundles")
    os.makedirs(path, exist_ok=True)
    return path


def _collect_aligned_turn_logprobs(
    adapter,
    session_id: str,
    *,
    step_tool_use_ids: list[str],
) -> list[list[float]]:
    """Align turns to PostToolUse snapshots by minted ``tool_use_id``.

    Uses ``session.turn_log`` (survives wipe clears). Extra emitted tools with
    no snapshot are ignored; every snapshot id must resolve or we raise
    :class:`StepTurnAlignmentError`.
    """
    session = getattr(adapter, "store", {}).get(session_id)
    if session is None:
        return align_logprobs_to_steps([], step_tool_use_ids)
    entries: list[tuple[list[float], list[str]]] = []
    for turn, tool_ids in getattr(session, "turn_log", None) or []:
        lps = list(getattr(turn, "output_log_probs", []) or [])
        if isinstance(tool_ids, int):
            # Backward-compatible with older test doubles.
            ids = [f"toolu_test_{i}" for i in range(int(tool_ids))]
        else:
            ids = [str(x) for x in (tool_ids or [])]
        entries.append((lps, ids))
    aligned = align_logprobs_to_steps(entries, step_tool_use_ids)
    n_emitted = sum(len(ids) for _, ids in entries)
    if n_emitted > len(step_tool_use_ids):
        logger.info(
            "[hybrid-live] align by tool_use_id: emitted=%d snapshots=%d dropped_no_hook=%d",
            n_emitted,
            len(step_tool_use_ids),
            n_emitted - len(step_tool_use_ids),
        )
    return aligned


def _shared_rollout_id(base: Sample) -> int:
    """Rollout id shared by all sibling samples from one hybrid_generate call.

    slime's compact-rollout validator requires every Sample in the leaf
    ``list[Sample]`` to carry the same non-None ``rollout_id``. Unique
    ``Sample.index`` values are fine for session identity; do **not** mirror
    them into ``rollout_id``.
    """
    if base.rollout_id is not None:
        return int(base.rollout_id)
    return int(base.index or 0)


def _trial_sample(base: Sample, *, trial_idx: int, base_index: int, group_index: int) -> Sample:
    s = copy.copy(base)
    s.index = base_index * 4096 + trial_idx
    s.group_index = group_index
    s.rollout_id = _shared_rollout_id(base)
    s.session_id = None
    s.metadata = dict(base.metadata or {})
    return s


async def live_vanilla_runner(
    *,
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    trial_idx: int,
    group_index: int,
    base_index: int,
) -> tuple[SessionBundle, list[Sample], bool, list[list[float]]]:
    """One Stage-1 trial with PostToolUse snapshots + Path A reward/fan-out."""
    md = gen._parse_metadata(sample)
    instance_id = md["instance_id"]
    if not md["image"] or not md["workdir"]:
        raise ValueError(f"missing image/workdir for {instance_id}")

    time_budget, eval_timeout, guard = gen._timeouts()
    state = gen._AdapterService(args)
    trial = _trial_sample(sample, trial_idx=trial_idx, base_index=base_index, group_index=group_index)
    session_id = trial.session_id = gen._session_id(trial, f"{instance_id}-t{trial_idx}")
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
    )

    out_dir = os.path.join(_bundle_root(args), instance_id, f"trial_{trial_idx}_{secrets.token_hex(4)}")
    turn_logprobs: list[list[float]] = []
    bundle: SessionBundle | None = None
    samples: list[Sample] = []
    is_solved = False
    t0 = time.time()

    try:
        claude_env = gen._build_claude_env(adapter_url=state.adapter_url, session_id=session_id)
        # Acquire slot before the per-agent guard so queue wait does not burn
        # the Claude/eval budget (Stage-2 has large fan-out behind concurrency=64).
        t_queue = time.time()
        async with gen.agent_concurrency_cm():
            queue_wait = time.time() - t_queue
            t_agent = time.time()
            async with asyncio.timeout(guard):
                async with make_sandbox(md["image"]) as sb:
                    await agent_runtime.prepare_workspace(
                        sb,
                        workdir=md["workdir"],
                        problem_statement=md["problem_statement"],
                        instance_id=md["instance_id"],
                        data_source=md["data_source"],
                        base_commit=md["base_commit"],
                        swe_smith_bug_patch=md.get("swe_smith_bug_patch"),
                        pre_commands=md.get("pre_commands") or "",
                        install_config=md.get("install_config") or {},
                        rollout_side=True,
                    )
                    await agent_runtime.install_toolchain(sb)
                    await install_snapshot_hook(sb, md["workdir"])
                    agent_result = await agent_runtime.run_claude(
                        sb,
                        workdir=md["workdir"],
                        prompt=md["agent_prompt"],
                        env=claude_env,
                        time_budget_sec=time_budget,
                    )
                    # Prefer HEAD-aligned diff used by snapshot script.
                    final_diff = await _common.workspace_diff(sb, md["workdir"])
                    if not (final_diff or "").strip():
                        final_diff = await agent_runtime.git_diff(sb, workdir=md["workdir"])

                    # Preserve official grading fields (FAIL_TO_PASS, repo, …) —
                    # same merge as generate.py. _parse_metadata alone strips them.
                    task_md = {
                        **(sample.metadata or {}),
                        **md,
                        "image": md["image"],
                        "workdir": md["workdir"],
                        "problem_statement": md["problem_statement"],
                        "eval_cmd": md["eval_cmd"],
                        "agent_prompt": md["agent_prompt"],
                    }
                    bundle = await capture_snapshots_to_bundle(
                        sb,
                        out_dir=out_dir,
                        workdir=md["workdir"],
                        instance_id=instance_id,
                        session_id=session_id,
                        task_metadata=task_md,
                        final_diff=final_diff or "",
                        claude_exit_code=int(agent_result.get("exit_code") or 0),
                    )
            agent_elapsed = time.time() - t_agent

        # Eval / fan-out outside the agent concurrency slot.
        t_eval = time.time()
        eval_result = await gen._evaluate_diff(
            image=md["image"],
            workdir=md["workdir"],
            eval_cmd=md["eval_cmd"],
            diff_text=bundle.final_diff() if bundle else "",
            timeout_sec=eval_timeout,
            metadata={**(sample.metadata or {}), **md},
        )
        eval_elapsed = time.time() - t_eval
        reward_path = getattr(
            args,
            "custom_cc_reward_function_path",
            "examples.claudecode_ags.rewards.default.compose",
        )
        reward_fn = load_function(reward_path)
        base_eval = {"resolved": eval_result.resolved, **eval_result.details}
        reward, reward_details = reward_fn(base_eval=base_eval, sample=trial, args=args)
        is_solved = bool(eval_result.resolved) or float(reward) == 1.0
        f2p_p2p = gen._f2p_p2p_metrics(base_eval)

        turn_logprobs = _collect_aligned_turn_logprobs(
            state.adapter,
            session_id,
            step_tool_use_ids=[str(s.tool_use_id or "") for s in (bundle.steps if bundle else [])],
        )
        segments = await state.adapter.finish_session(session_id)
        samples = fan_out_sample_segments(
            trial,
            segments,
            reward=float(reward),
            tokenizer=state.tokenizer,
            # fan_out defaults to sample.index; force parent rollout_id.
            rollout_id=int(trial.rollout_id),
            metadata={
                "instance_id": instance_id,
                "grading_solved": is_solved,
                "applied_cleanly": bool(eval_result.applied_cleanly),
                "agent_exit_code": agent_result.get("exit_code"),
                "reward_details": reward_details,
                "base_eval": base_eval,
                "sample_kind": "vanilla",
                "trial_idx": trial_idx,
                # Distinct from shared rollout_id so GRPO treats K trials
                # as separate episodes (see 2026-07-14 vanilla episode-key note).
                "branch_uid": f"v:{group_index}:t{trial_idx}",
                "bundle_dir": out_dir,
                "agent_elapsed_sec": agent_elapsed,
                "agent_queue_wait_sec": queue_wait,
                "eval_elapsed_sec": eval_elapsed,
                "total_elapsed_sec": time.time() - t0,
                **f2p_p2p,
            },
        )
        if not samples:
            raise RuntimeError("adapter_session_empty")

        # Persist transcript placeholder for prefix reseed (best-effort).
        transcript_path = os.path.join(out_dir, "transcript.jsonl")
        if not os.path.isfile(transcript_path):
            os.makedirs(out_dir, exist_ok=True)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("")

        logger.info(
            "[hybrid-live] vanilla trial=%d %s reward=%.2f solved=%s steps=%d aligned_turns=%d",
            trial_idx,
            instance_id,
            float(reward),
            is_solved,
            bundle.num_steps if bundle else 0,
            len(turn_logprobs),
        )
        assert bundle is not None
        return bundle, samples, is_solved, turn_logprobs
    except Exception:
        logger.warning(
            "[hybrid-live] vanilla trial=%d failed:\n%s",
            trial_idx,
            traceback.format_exc(),
        )
        raise
    finally:
        try:
            await state.adapter.shutdown_session(session_id, wait_timeout=5.0)
        except Exception:
            pass


_SUBMIT_SEM: asyncio.Semaphore | None = None
_SUBMIT_SEM_LIMIT: int | None = None


def _branch_submit_batch_size() -> int:
    """How many Stage-2 sandboxes may be *submitted/started* at once.

    This is **not** a cap on how many AGS sandboxes may be running CC at once.
    After StartSandbox succeeds, the slot is released so more can be submitted
    while earlier branches keep running.

    Env (first hit wins):
      - ``STEP_GRPO_BRANCH_SUBMIT_BATCH`` (preferred)
      - ``STEP_GRPO_BRANCH_CONCURRENCY`` (legacy alias)
    ``<=0`` disables the submit gate (start all at once).
    Default: **64**.
    """
    if os.environ.get("STEP_GRPO_BRANCH_SUBMIT_BATCH") is not None:
        return _env_int("STEP_GRPO_BRANCH_SUBMIT_BATCH", 64)
    return _env_int("STEP_GRPO_BRANCH_CONCURRENCY", 64)


def _branch_submit_cm():
    """Gate only the sandbox-create / submit phase."""
    global _SUBMIT_SEM, _SUBMIT_SEM_LIMIT
    from contextlib import nullcontext

    n = _branch_submit_batch_size()
    if n <= 0:
        return nullcontext()
    if _SUBMIT_SEM is None or _SUBMIT_SEM_LIMIT != n:
        _SUBMIT_SEM = asyncio.Semaphore(n)
        _SUBMIT_SEM_LIMIT = n
    return _SUBMIT_SEM


@asynccontextmanager
async def _workspace_after_submit(bundle: SessionBundle, step_t: int):
    """Acquire submit slot until sandbox is up, then release while work continues."""
    ws_cm = rebuilt_workspace(bundle, step_t)
    async with _branch_submit_cm():
        sb, applied = await ws_cm.__aenter__()
    try:
        yield sb, applied
    except BaseException as exc:
        await ws_cm.__aexit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        await ws_cm.__aexit__(None, None, None)


async def live_branch_runner(
    *,
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    bundle: SessionBundle,
    source_trial_idx: int,
    step_t: int,
    branch_idx: int,
    group_index: int,
    edit_ppl: float,
) -> list[Sample]:
    """Rebuild ``s_t``, prefix-reseed, continue CC, eval, fan-out branch samples."""
    md = dict(bundle.task_metadata)
    instance_id = str(md.get("instance_id") or bundle.instance_id)
    workdir = str(md.get("workdir") or "/testbed")
    image = str(md.get("image") or "")
    if not image:
        raise ValueError("bundle missing image")

    time_budget, eval_timeout, guard = gen._timeouts()
    branch_budget = _env_int("STEP_GRPO_BRANCH_BUDGET_SEC", time_budget)
    state = gen._AdapterService(args)

    branch_sample = copy.copy(sample)
    branch_sample.index = (int(sample.index or 0) * 4096) + 1000 + source_trial_idx * 64 + branch_idx
    branch_sample.group_index = group_index
    branch_sample.rollout_id = _shared_rollout_id(sample)
    branch_sample.session_id = None
    session_id = branch_sample.session_id = gen._session_id(
        branch_sample, f"{instance_id}-b{source_trial_idx}-{step_t}-{branch_idx}"
    )
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
    )

    t0 = time.time()
    try:
        # Acquire slot before the per-agent guard so queue wait does not burn
        # the Claude/eval budget (Stage-2 has large fan-out behind concurrency=64).
        t_queue = time.time()
        async with gen.agent_concurrency_cm():
            queue_wait = time.time() - t_queue
            t_agent = time.time()
            async with asyncio.timeout(guard):
                async with _workspace_after_submit(bundle, step_t) as (sb, applied):
                    if not applied:
                        raise RuntimeError(f"rebuild apply failed t={step_t}")

                    transcript_path = os.path.join(bundle.dir, bundle.transcript_rel)
                    prefix_text = ""
                    if os.path.isfile(transcript_path):
                        with open(transcript_path, encoding="utf-8", errors="replace") as f:
                            prefix_text = truncate_transcript_prefix(f.read(), step_t)

                    claude_env = gen._build_claude_env(
                        adapter_url=state.adapter_url, session_id=session_id
                    )
                    await agent_runtime.install_toolchain(sb)
                    prompt = str(md.get("agent_prompt") or gen._DEFAULT_AGENT_PROMPT)
                    if prefix_text.strip():
                        agent_result = await resume_and_run(
                            sb,
                            workdir=workdir,
                            prefix_text=prefix_text,
                            prompt=prompt,
                            env=claude_env,
                            time_budget_sec=branch_budget,
                        )
                    else:
                        # No transcript: continue from rebuilt workspace only.
                        agent_result = await agent_runtime.run_claude(
                            sb,
                            workdir=workdir,
                            prompt=prompt,
                            env=claude_env,
                            time_budget_sec=branch_budget,
                        )

                    cont_diff = await _common.workspace_diff(sb, workdir)
                    if not (cont_diff or "").strip():
                        cont_diff = await agent_runtime.git_diff(sb, workdir=workdir)
            agent_elapsed = time.time() - t_agent

        # Eval / fan-out outside the agent concurrency slot.
        t_eval = time.time()
        eval_result = await gen._evaluate_diff(
            image=image,
            workdir=workdir,
            eval_cmd=str(md.get("eval_cmd") or ""),
            diff_text=cont_diff or "",
            timeout_sec=eval_timeout,
            # Prefer sample.metadata grading fields; bundle.task_metadata
            # may be incomplete on older bundles.
            metadata={**(sample.metadata or {}), **md},
        )
        eval_elapsed = time.time() - t_eval
        reward_path = getattr(
            args,
            "custom_cc_reward_function_path",
            "examples.claudecode_ags.rewards.default.compose",
        )
        reward_fn = load_function(reward_path)
        base_eval = {"resolved": eval_result.resolved, **eval_result.details}
        reward, reward_details = reward_fn(
            base_eval=base_eval, sample=branch_sample, args=args
        )
        f2p_p2p = gen._f2p_p2p_metrics(base_eval)

        segments = await state.adapter.finish_session(session_id)
        step_group_key = f"{group_index}:{source_trial_idx}:{step_t}"
        samples = fan_out_sample_segments(
            branch_sample,
            segments,
            reward=float(reward),
            tokenizer=state.tokenizer,
            rollout_id=int(branch_sample.rollout_id),
            metadata={
                "instance_id": instance_id,
                "sample_kind": "branch",
                "source_trial_idx": source_trial_idx,
                "step_t": step_t,
                "branch_idx": branch_idx,
                "step_group_key": step_group_key,
                "edit_ppl": edit_ppl,
                "branch_uid": f"{step_group_key}:{branch_idx}",
                "grading_solved": bool(eval_result.resolved),
                "applied_cleanly": bool(eval_result.applied_cleanly),
                "agent_exit_code": agent_result.get("exit_code"),
                "reward_details": reward_details,
                "base_eval": base_eval,
                "agent_elapsed_sec": agent_elapsed,
                "agent_queue_wait_sec": queue_wait,
                "eval_elapsed_sec": eval_elapsed,
                "total_elapsed_sec": time.time() - t0,
                **f2p_p2p,
            },
        )
        # Train scope = full continuation of *model* tokens only.
        # Keep fan_out / merge_turns loss_mask (1 on assistant outputs, 0 on
        # tool/context tails). Forcing all-1s would train on placeholder
        # rollout_log_probs=0.0 and break TIS/RS (see notes 2026-07-14).
        if not samples:
            raise RuntimeError("branch adapter_session_empty")
        logger.info(
            "[hybrid-live] branch src=%d t=%d b=%d reward=%.2f segs=%d exit=%s queue=%.1fs agent=%.1fs",
            source_trial_idx,
            step_t,
            branch_idx,
            float(reward),
            len(samples),
            agent_result.get("exit_code"),
            queue_wait,
            agent_elapsed,
        )
        return samples
    finally:
        try:
            await state.adapter.shutdown_session(session_id, wait_timeout=5.0)
        except Exception:
            pass
