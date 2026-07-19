"""Live AGS runners for Path A hybrid_generate (Stage-1 capture + Stage-2 branch)."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import secrets
import time
import traceback
import uuid
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
    atomic_write_text,
    capture_initial_workspace_metadata,
    capture_snapshots_to_bundle,
    install_snapshot_hook,
)
from examples.claudecode_ags.step_reconstruct.native_session import (
    truncate_native_session_before_tools,
)
from examples.claudecode_ags.step_reconstruct.workspace_rebuild import (
    rebuilt_workspace,
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
    explicit = os.environ.get("STEP_GRPO_BUNDLE_DIR")
    if explicit:
        path = str(explicit)
    else:
        root = getattr(args, "save", None) or getattr(args, "load", None) or "/tmp/cc_ags_step_reconstruct"
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


def _collect_stage2_alignment_or_disable(adapter, session_id: str, bundle: SessionBundle) -> list[list[float]]:
    """Keep Stage-1 trainable when reconstruction-only alignment is invalid."""
    step_ids = [str(step.tool_use_id or "") for step in bundle.steps]
    try:
        return _collect_aligned_turn_logprobs(
            adapter,
            session_id,
            step_tool_use_ids=step_ids,
        )
    except StepTurnAlignmentError as exc:
        bundle.transcript_valid = False
        bundle.transcript_error = str(exc)
        bundle.save(bundle.dir)
        logger.warning(
            "[hybrid-live] Stage-2 disabled but Stage-1 retained: turn/snapshot alignment invalid "
            "instance=%s session=%s bundle=%s reason=%s",
            bundle.instance_id,
            session_id,
            bundle.dir,
            exc,
        )
        # Preserve the one-entry-per-snapshot shape. Empty logprob rows produce
        # no edit-PPL candidates even if a caller overlooks transcript_valid.
        return [[] for _ in bundle.steps]


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
    cc_session_id = str(uuid.uuid4())
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
        capture_prompt_checkpoints=True,
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
                    initial_diff = await _common.workspace_diff(sb, md["workdir"])
                    await capture_initial_workspace_metadata(sb, md["workdir"])
                    agent_result = await agent_runtime.run_claude(
                        sb,
                        workdir=md["workdir"],
                        prompt=md["agent_prompt"],
                        env=claude_env,
                        time_budget_sec=time_budget,
                        claude_session_id=cc_session_id,
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
                    exporter = getattr(state.adapter, "export_prompt_checkpoints_async", None)
                    prompt_checkpoints = (
                        await exporter(session_id, clear=True) if callable(exporter) else []
                    )
                    if not isinstance(prompt_checkpoints, list):
                        prompt_checkpoints = []
                    bundle = await capture_snapshots_to_bundle(
                        sb,
                        out_dir=out_dir,
                        workdir=md["workdir"],
                        instance_id=instance_id,
                        session_id=session_id,
                        task_metadata=task_md,
                        initial_diff=initial_diff or "",
                        final_diff=final_diff or "",
                        transcript_text=str(agent_result.get("trajectory_jsonl") or ""),
                        claude_exit_code=int(agent_result.get("exit_code") or 0),
                        cc_session_id=cc_session_id,
                        prompt_checkpoints=prompt_checkpoints,
                    )
                    del prompt_checkpoints
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
        f2p_p2p = gen._f2p_p2p_metrics(base_eval)

        turn_logprobs = _collect_stage2_alignment_or_disable(
            state.adapter,
            session_id,
            bundle,
        )
        segments = await state.adapter.finish_session(session_id)
        from examples.claudecode_ags.rewards.tool_loop_penalty import adjust_episode_reward

        reward, reward_details, reward_audit = adjust_episode_reward(
            base_reward=float(reward),
            reward_details=reward_details,
            resolved=bool(eval_result.resolved),
            exit_code=agent_result.get("exit_code"),
            responses=(
                state.tokenizer.decode(segment.response_ids, skip_special_tokens=False)
                for segment in segments
            ),
            timeout_outcome_enabled=gen._env_int("SLIME_CC_TIMEOUT_OUTCOME_REWARD", 0) > 0,
            tool_loop_enabled=gen._env_int("SLIME_CC_TOOL_LOOP_PENALTY", 0) > 0,
        )
        is_solved = bool(eval_result.resolved) or float(reward) == 1.0
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
                **reward_audit,
                "reward_details": reward_details,
                "base_eval": base_eval,
                "sample_kind": "vanilla",
                "trial_idx": trial_idx,
                # Distinct from shared rollout_id so GRPO treats K trials
                # as separate episodes (see 2026-07-14 vanilla episode-key note).
                "branch_uid": f"v:{group_index}:t{trial_idx}",
                "bundle_dir": out_dir,
                "transcript_valid": bool(bundle.transcript_valid),
                "transcript_error": str(bundle.transcript_error or ""),
                "transcript_bytes": int(bundle.transcript_bytes),
                "transcript_events": int(bundle.transcript_events),
                "agent_elapsed_sec": agent_elapsed,
                "agent_queue_wait_sec": queue_wait,
                "eval_elapsed_sec": eval_elapsed,
                "total_elapsed_sec": time.time() - t0,
                **f2p_p2p,
            },
        )
        if not samples:
            raise RuntimeError("adapter_session_empty")

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
    edit_step_i: int,
    branch_step_t: int,
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
    branch_sample.index = (
        int(sample.index or 0) * 1_000_000_000
        + 100_000_000
        + source_trial_idx * 10_000_000
        + edit_step_i * 1_000
        + branch_idx
    )
    branch_sample.group_index = group_index
    branch_sample.rollout_id = _shared_rollout_id(sample)
    branch_sample.session_id = None
    session_id = branch_sample.session_id = gen._session_id(
        branch_sample, f"{instance_id}-b{source_trial_idx}-e{edit_step_i}-s{branch_step_t}-{branch_idx}"
    )
    branch_transcript_rel = os.path.join(
        "branch_runs",
        f"trial_{source_trial_idx}_edit_{edit_step_i}_branch_{branch_idx}",
        "transcript.jsonl",
    )
    t0 = time.time()
    try:
        # Acquire slot before the per-agent guard so queue wait does not burn
        # the Claude/eval budget. Checkpoint/native state is loaded only after
        # admission so queued fan-out does not retain hundreds of prompt copies.
        t_queue = time.time()
        async with gen.agent_concurrency_cm():
            queue_wait = time.time() - t_queue
            t_agent = time.time()
            async with asyncio.timeout(guard):
                exact_ready, exact_error = bundle.token_exact_readiness(branch_step_t)
                if not exact_ready:
                    raise RuntimeError(f"token_exact_bundle_not_ready:{exact_error}")
                snapshot_ids = [str(step.tool_use_id or "") for step in bundle.steps]
                if edit_step_i < 0 or edit_step_i >= len(snapshot_ids):
                    raise IndexError(f"edit_step_i out of range: {edit_step_i}")
                target_tool_use_id = snapshot_ids[edit_step_i]
                checkpoint = bundle.checkpoint_for_tool_use_id(target_tool_use_id)
                if checkpoint is None:
                    raise RuntimeError(f"missing_prompt_checkpoint_for_tool:{target_tool_use_id}")
                checkpoint_tool_ids = [
                    str(value) for value in checkpoint.get("generated_tool_use_ids") or []
                ]
                if target_tool_use_id not in checkpoint_tool_ids:
                    raise RuntimeError(
                        f"checkpoint_does_not_generate_target_tool:{target_tool_use_id}"
                    )
                native_prefix = truncate_native_session_before_tools(
                    bundle.native_session(),
                    checkpoint_tool_ids,
                )
                await state.adapter.open_session_async(
                    session_id,
                    sampling_defaults=dict(sampling_params or {}),
                    max_context_tokens=state.max_context_len,
                    resume_checkpoint=checkpoint,
                )

                async with _workspace_after_submit(bundle, branch_step_t) as (sb, applied):
                    if not applied:
                        raise RuntimeError(f"rebuild apply/verify failed t={branch_step_t}")

                    claude_env = gen._build_claude_env(
                        adapter_url=state.adapter_url, session_id=session_id
                    )
                    await agent_runtime.install_toolchain(sb)
                    await agent_runtime.ensure_claude_home_writable(sb)
                    await install_snapshot_hook(sb, workdir)
                    agent_result = await agent_runtime.run_claude_native_resume(
                        sb,
                        workdir=workdir,
                        env=claude_env,
                        time_budget_sec=branch_budget,
                        session_jsonl=native_prefix.jsonl,
                    )
                    # Persist the wire transcript before validating resume status.
                    # Protocol failures are exactly the cases where this artifact
                    # is most useful; previously it was written only for survivors.
                    atomic_write_text(
                        os.path.join(bundle.dir, branch_transcript_rel),
                        str(agent_result.get("trajectory_jsonl") or ""),
                    )

                    cont_diff = await _common.workspace_diff(sb, workdir)
                    if not (cont_diff or "").strip():
                        cont_diff = await agent_runtime.git_diff(sb, workdir=workdir)

                resume_status = state.adapter.resume_status(session_id)
                if (
                    resume_status.get("mode") != "token_exact"
                    or not resume_status.get("handshake_validated")
                    or not resume_status.get("first_prompt_exact")
                    or int(resume_status.get("exact_request_count") or 0) <= 0
                    or resume_status.get("error")
                ):
                    raise RuntimeError(f"token_exact_resume_not_verified:{resume_status}")
                segments = await state.adapter.finish_session(session_id)
                if not segments:
                    logger.warning(
                        "[hybrid-live] Stage-2 skipped: native token-exact resume produced no new "
                        "adapter segments instance=%s trial=%d edit=%d branch=%d bundle=%s exit=%s",
                        instance_id,
                        source_trial_idx,
                        edit_step_i,
                        branch_idx,
                        bundle.dir,
                        agent_result.get("exit_code"),
                    )
                    raise RuntimeError("token_exact_resume_no_new_adapter_segments")
                native_target_row_index = native_prefix.target_row_index
                del checkpoint, native_prefix
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

        from examples.claudecode_ags.rewards.tool_loop_penalty import adjust_episode_reward

        reward, reward_details, reward_audit = adjust_episode_reward(
            base_reward=float(reward),
            reward_details=reward_details,
            resolved=bool(eval_result.resolved),
            exit_code=agent_result.get("exit_code"),
            # Stage-2 is responsible only for its newly generated continuation;
            # the Stage-1 prefix is intentionally absent from these segments.
            responses=(
                state.tokenizer.decode(segment.response_ids, skip_special_tokens=False)
                for segment in segments
            ),
            timeout_outcome_enabled=gen._env_int("SLIME_CC_TIMEOUT_OUTCOME_REWARD", 0) > 0,
            tool_loop_enabled=gen._env_int("SLIME_CC_TOOL_LOOP_PENALTY", 0) > 0,
        )

        step_group_key = f"{group_index}:{source_trial_idx}:edit:{edit_step_i}"
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
                "edit_step_i": edit_step_i,
                "branch_step_t": branch_step_t,
                # Compatibility for older metric readers: this is the target edit.
                "step_t": edit_step_i,
                "branch_idx": branch_idx,
                "step_group_key": step_group_key,
                "edit_ppl": edit_ppl,
                "branch_uid": f"{step_group_key}:{branch_idx}",
                "branch_transcript_rel": branch_transcript_rel,
                "prefix_reseed_mode": "native-session-checkpoint",
                "prefix_reseed_verified": True,
                "prompt_checkpoint_id": resume_status["checkpoint_id"],
                "prompt_sha256": resume_status["first_prompt_sha256"],
                "prompt_exact": resume_status["first_prompt_exact"],
                "token_exact_request_count": resume_status["exact_request_count"],
                "tool_use_echo_mismatch_count": int(
                    resume_status.get("tool_use_echo_mismatch_count") or 0
                ),
                "tool_use_echo_missing_count": int(
                    resume_status.get("tool_use_echo_missing_count") or 0
                ),
                "tool_use_echo_payload_mismatch_count": int(
                    resume_status.get("tool_use_echo_payload_mismatch_count") or 0
                ),
                "runtime_tool_schema_mismatch_count": int(
                    resume_status.get("runtime_tool_schema_mismatch_count") or 0
                ),
                "generated_runtime_tool_unavailable_count": int(
                    resume_status.get("generated_runtime_tool_unavailable_count") or 0
                ),
                "generated_runtime_tool_input_invalid_count": int(
                    resume_status.get("generated_runtime_tool_input_invalid_count") or 0
                ),
                "max_tokens_continuation_count": int(
                    resume_status.get("max_tokens_continuation_count") or 0
                ),
                "post_end_turn_ack_count": int(
                    resume_status.get("post_end_turn_ack_count") or 0
                ),
                "resume_request_replay_count": int(
                    resume_status.get("request_replay_count") or 0
                ),
                "resume_last_stop_reason": str(resume_status.get("last_stop_reason") or ""),
                "native_session_target_row_index": native_target_row_index,
                "source_transcript_bytes": int(bundle.transcript_bytes),
                "grading_solved": bool(eval_result.resolved),
                "applied_cleanly": bool(eval_result.applied_cleanly),
                "agent_exit_code": agent_result.get("exit_code"),
                **reward_audit,
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
            "[hybrid-live] branch src=%d edit=%d pre=%d b=%d reward=%.2f segs=%d exit=%s queue=%.1fs agent=%.1fs",
            source_trial_idx,
            edit_step_i,
            branch_step_t,
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
