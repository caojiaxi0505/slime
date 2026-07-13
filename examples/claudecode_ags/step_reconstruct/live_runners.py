"""Live AGS runners for Path A hybrid_generate (Stage-1 capture + Stage-2 branch)."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import secrets
import traceback
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags import generate as gen
from examples.claudecode_ags.step_reconstruct import _common
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


def _collect_turn_logprobs(adapter, session_id: str) -> list[list[float]]:
    """Peek adapter session turns before ``finish_session`` drains them."""
    session = getattr(adapter, "store", {}).get(session_id)
    if session is None:
        return []
    turns = []
    for seg in getattr(session, "segments", []) or []:
        turns.extend(getattr(seg, "turns", []) or [])
    active_sub = getattr(session, "active_sub", None)
    if active_sub is not None:
        turns.extend(getattr(active_sub, "turns", []) or [])
    turns.extend(getattr(session.main, "turns", []) or [])
    return [list(getattr(t, "output_log_probs", []) or []) for t in turns]


def _trial_sample(base: Sample, *, trial_idx: int, base_index: int, group_index: int) -> Sample:
    s = copy.copy(base)
    s.index = base_index * 4096 + trial_idx
    s.group_index = group_index
    s.rollout_id = s.index
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

    try:
        async with asyncio.timeout(guard):
            claude_env = gen._build_claude_env(adapter_url=state.adapter_url, session_id=session_id)
            async with make_sandbox(md["image"]) as sb:
                await agent_runtime.prepare_workspace(
                    sb,
                    workdir=md["workdir"],
                    problem_statement=md["problem_statement"],
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

                task_md = {
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

            eval_result = await gen._evaluate_diff(
                image=md["image"],
                workdir=md["workdir"],
                eval_cmd=md["eval_cmd"],
                diff_text=bundle.final_diff() if bundle else "",
                timeout_sec=eval_timeout,
            )
            reward_path = getattr(
                args,
                "custom_cc_reward_function_path",
                "examples.claudecode_ags.rewards.default.compose",
            )
            reward_fn = load_function(reward_path)
            base_eval = {"resolved": eval_result.resolved, **eval_result.details}
            reward, reward_details = reward_fn(base_eval=base_eval, sample=trial, args=args)
            is_solved = bool(eval_result.resolved) or float(reward) == 1.0

            turn_logprobs = _collect_turn_logprobs(state.adapter, session_id)
            segments = await state.adapter.finish_session(session_id)
            samples = fan_out_sample_segments(
                trial,
                segments,
                reward=float(reward),
                tokenizer=state.tokenizer,
                metadata={
                    "instance_id": instance_id,
                    "grading_solved": is_solved,
                    "applied_cleanly": bool(eval_result.applied_cleanly),
                    "agent_exit_code": agent_result.get("exit_code"),
                    "reward_details": reward_details,
                    "base_eval": base_eval,
                    "sample_kind": "vanilla",
                    "trial_idx": trial_idx,
                    "bundle_dir": out_dir,
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
                "[hybrid-live] vanilla trial=%d %s reward=%.2f solved=%s steps=%d turns=%d",
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


_BRANCH_SEM: asyncio.Semaphore | None = None


def _branch_semaphore() -> asyncio.Semaphore:
    global _BRANCH_SEM
    if _BRANCH_SEM is None:
        _BRANCH_SEM = asyncio.Semaphore(max(1, _env_int("STEP_GRPO_BRANCH_CONCURRENCY", 8)))
    return _BRANCH_SEM


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
    branch_sample.rollout_id = branch_sample.index
    branch_sample.session_id = None
    session_id = branch_sample.session_id = gen._session_id(
        branch_sample, f"{instance_id}-b{source_trial_idx}-{step_t}-{branch_idx}"
    )
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
    )

    async with _branch_semaphore():
        try:
            async with asyncio.timeout(guard):
                async with rebuilt_workspace(bundle, step_t) as (sb, applied):
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
                        await resume_and_run(
                            sb,
                            workdir=workdir,
                            prefix_text=prefix_text,
                            prompt=prompt,
                            env=claude_env,
                            time_budget_sec=branch_budget,
                        )
                    else:
                        # No transcript: continue from rebuilt workspace only.
                        await agent_runtime.run_claude(
                            sb,
                            workdir=workdir,
                            prompt=prompt,
                            env=claude_env,
                            time_budget_sec=branch_budget,
                        )

                    cont_diff = await _common.workspace_diff(sb, workdir)
                    if not (cont_diff or "").strip():
                        cont_diff = await agent_runtime.git_diff(sb, workdir=workdir)

                eval_result = await gen._evaluate_diff(
                    image=image,
                    workdir=workdir,
                    eval_cmd=str(md.get("eval_cmd") or ""),
                    diff_text=cont_diff or "",
                    timeout_sec=eval_timeout,
                )
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

                segments = await state.adapter.finish_session(session_id)
                step_group_key = f"{group_index}:{source_trial_idx}:{step_t}"
                samples = fan_out_sample_segments(
                    branch_sample,
                    segments,
                    reward=float(reward),
                    tokenizer=state.tokenizer,
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
                        "reward_details": reward_details,
                        "base_eval": base_eval,
                    },
                )
                # Full continuation trainable.
                for s in samples:
                    if s.loss_mask is None and s.response_length:
                        s.loss_mask = [1] * int(s.response_length)
                    elif s.loss_mask is not None:
                        s.loss_mask = [1] * len(s.loss_mask)
                if not samples:
                    raise RuntimeError("branch adapter_session_empty")
                logger.info(
                    "[hybrid-live] branch src=%d t=%d b=%d reward=%.2f segs=%d",
                    source_trial_idx,
                    step_t,
                    branch_idx,
                    float(reward),
                    len(samples),
                )
                return samples
        finally:
            try:
                await state.adapter.shutdown_session(session_id, wait_timeout=5.0)
            except Exception:
                pass
