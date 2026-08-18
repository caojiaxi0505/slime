"""Stage-2 runner: native resume + remote teacher continuation → student SFT sample."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags import generate as gen
from examples.claudecode_ags.sft_remote_openai_adapter import RemoteOpenAISFTAdapter
from examples.claudecode_ags.step_reconstruct import _common
from examples.claudecode_ags.step_reconstruct.live_runners import (
    HybridPipelineTimeoutError,
    _shared_rollout_id,
    _workspace_after_submit,
)
from examples.claudecode_ags.step_reconstruct.native_session import truncate_native_session_before_tools
from examples.claudecode_ags.step_reconstruct.session_capture import (
    SessionBundle,
    atomic_write_text,
    install_snapshot_hook,
)
from examples.claudecode_ags.step_reconstruct.teacher_segments import (
    encode_teacher_continuation_sample,
    load_sft_turn_rows,
)

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _teacher_bind() -> tuple[str, int]:
    host = (os.environ.get("SLIME_TEACHER_ADAPTER_BIND_HOST") or "127.0.0.1").strip()
    port = _env_int("SLIME_TEACHER_ADAPTER_PORT", 18002)
    return host, port


def teacher_max_steps() -> int:
    """Teacher turns to take per relabel; ``0`` runs the task to the end.

    The default takes two turns so the SFT target contains a tool call plus the
    teacher's reaction to its result, at the cost of one sandbox per relabel.
    """
    value = _env_int("STEP_GRPO_TEACHER_MAX_STEPS", 2)
    if value < 0:
        raise ValueError(f"STEP_GRPO_TEACHER_MAX_STEPS must be >= 0, got {value}")
    return value


def teacher_grades_continuation() -> bool:
    """Whether Stage-2 runs the grader on the teacher's workspace.

    Grading only makes sense when the teacher runs to the end; a capped teacher
    stops mid-task, so the resolved gate is skipped along with the eval sandbox.
    """
    return teacher_max_steps() == 0 and _env_bool("STEP_GRPO_TEACHER_SFT_RESOLVED_ONLY", True)


def _teacher_sft_log_dir(args: Any) -> str:
    log_dir = (os.environ.get("SLIME_TEACHER_SFT_LOG_DIR") or "").strip()
    if not log_dir:
        root = getattr(args, "save", None) or getattr(args, "load", None) or "/tmp"
        log_dir = os.path.join(str(root), "teacher_sft_turns")
    student_log_dir = (os.environ.get("SLIME_AGENT_SFT_LOG_DIR") or "").strip()
    if student_log_dir and os.path.realpath(student_log_dir) == os.path.realpath(log_dir):
        raise RuntimeError(
            "teacher_sft_log_isolation: SLIME_AGENT_SFT_LOG_DIR and "
            "SLIME_TEACHER_SFT_LOG_DIR must differ"
        )
    return log_dir


class _TeacherAdapterService(metaclass=SingletonMeta):
    """One remote-OpenAI Anthropic shim per process for teacher continuations."""

    def __init__(self, args: Any) -> None:
        log_dir = _teacher_sft_log_dir(args)
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.loss_mask_type = str(getattr(args, "loss_mask_type", None) or "qwen3")
        os.makedirs(log_dir, exist_ok=True)
        self.sft_log_dir = log_dir

        base_url = (os.environ.get("SLIME_REMOTE_OPENAI_BASE_URL") or "").strip()
        api_key = (os.environ.get("SLIME_REMOTE_OPENAI_API_KEY") or "").strip()
        model = (os.environ.get("SLIME_REMOTE_OPENAI_MODEL") or "").strip()
        if not base_url or not api_key or not model:
            raise RuntimeError(
                "teacher continuation requires SLIME_REMOTE_OPENAI_BASE_URL, "
                "SLIME_REMOTE_OPENAI_API_KEY, and SLIME_REMOTE_OPENAI_MODEL"
            )

        # Construct directly so Teacher logs stay isolated from Stage-1 logs.
        self.adapter = RemoteOpenAISFTAdapter(
            base_url=base_url,
            api_key=api_key,
            model=model,
            sft_log_dir=log_dir,
            temperature=float(os.environ.get("SLIME_REMOTE_OPENAI_TEMPERATURE") or "1"),
            top_p=float(os.environ.get("SLIME_REMOTE_OPENAI_TOP_P") or "0.95"),
            top_k=int(os.environ.get("SLIME_REMOTE_OPENAI_TOP_K") or "20"),
            reasoning_effort=str(os.environ.get("SLIME_REMOTE_OPENAI_REASONING_EFFORT") or "max"),
            thinking_type=os.environ.get("SLIME_REMOTE_OPENAI_THINKING_TYPE"),
            max_turns_per_sid=teacher_max_steps() or None,
            require_registered_sessions=True,
        )
        bind_host, bind_port = _teacher_bind()
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=bind_host,
            port=bind_port,
            thread_name="cc-ags-teacher-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        public = (os.environ.get("SLIME_TEACHER_ADAPTER_PUBLIC_URL") or "").strip()
        if public:
            self.adapter_url = public.rstrip("/")
        else:
            self.adapter_url = f"http://127.0.0.1:{self.app_handle.port}"
        logger.info(
            "[teacher-sft] adapter=%s model=%s thinking=%s effort=%s log_dir=%s",
            self.adapter_url,
            model,
            self.adapter.thinking_type,
            self.adapter.reasoning_effort,
            log_dir,
        )


def ensure_teacher_adapter(args: Any) -> _TeacherAdapterService:
    """Start the teacher Anthropic shim before Stage-2 sandboxes need it.

    Bind ``0.0.0.0`` and set ``SLIME_TEACHER_ADAPTER_PUBLIC_URL`` in cluster
    jobs so AGS sandboxes can reach this process through the teacher ALB.
    """
    return _TeacherAdapterService(args)


async def teacher_branch_runner(
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
    """Resume workspace/session, let teacher continue, encode student SFT sample."""
    del sampling_params  # teacher sampling comes from remote OpenAI env

    md = dict(bundle.task_metadata)
    instance_id = str(md.get("instance_id") or bundle.instance_id)
    workdir = str(md.get("workdir") or "/testbed")
    image = str(md.get("image") or "")
    if not image:
        raise ValueError("bundle missing image")

    time_budget, eval_timeout, guard = gen._timeouts()
    branch_budget = _env_int("STEP_GRPO_BRANCH_BUDGET_SEC", time_budget)
    max_steps = teacher_max_steps()
    grade = teacher_grades_continuation()
    teacher = _TeacherAdapterService(args)

    branch_sample = copy.copy(sample)
    branch_sample.index = (
        int(sample.index or 0) * 1_000_000_000
        + 200_000_000
        + source_trial_idx * 10_000_000
        + edit_step_i * 1_000
        + branch_idx
    )
    branch_sample.group_index = group_index
    branch_sample.rollout_id = _shared_rollout_id(sample)
    branch_sample.session_id = None
    session_id = branch_sample.session_id = gen._session_id(
        branch_sample,
        f"{instance_id}-teach{source_trial_idx}-e{edit_step_i}-s{branch_step_t}-{branch_idx}",
    )
    sft_path: str | None = None
    resume_session_registered = False
    branch_transcript_rel = os.path.join(
        "teacher_branch_runs",
        f"trial_{source_trial_idx}_edit_{edit_step_i}_branch_{branch_idx}",
        "transcript.jsonl",
    )
    t0 = time.time()
    try:
        t_queue = time.time()
        async with gen.agent_concurrency_cm():
            queue_wait = time.time() - t_queue
            pipeline_deadline = asyncio.get_running_loop().time() + guard
            t_agent = time.time()
            try:
                async with asyncio.timeout_at(pipeline_deadline):
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
                    checkpoint_tool_ids = [str(value) for value in (checkpoint.get("generated_tool_use_ids") or [])]
                    if target_tool_use_id not in checkpoint_tool_ids:
                        raise RuntimeError(f"checkpoint_does_not_generate_target_tool:{target_tool_use_id}")
                    native_prefix = truncate_native_session_before_tools(
                        bundle.native_session(),
                        checkpoint_tool_ids,
                    )

                    async with _workspace_after_submit(bundle, branch_step_t) as (sb, applied):
                        if not applied:
                            raise RuntimeError(f"rebuild apply/verify failed t={branch_step_t}")

                        claude_env = gen._build_claude_env(adapter_url=teacher.adapter_url, session_id=session_id)
                        await agent_runtime.install_toolchain(sb)
                        await agent_runtime.ensure_claude_home_writable(sb)
                        await install_snapshot_hook(sb, workdir)
                        # Claude Code restores the harness and executes tools;
                        # the adapter owns the Teacher-visible model history.
                        sft_path = teacher.adapter.register_resume_session(
                            session_id,
                            checkpoint,
                        )
                        resume_session_registered = True
                        agent_result = await agent_runtime.run_claude_native_resume(
                            sb,
                            workdir=workdir,
                            env=claude_env,
                            time_budget_sec=branch_budget,
                            session_jsonl=native_prefix.jsonl,
                        )
                        atomic_write_text(
                            os.path.join(bundle.dir, branch_transcript_rel),
                            str(agent_result.get("trajectory_jsonl") or ""),
                        )
                        if grade:
                            cont_diff = await _common.workspace_diff(sb, workdir)
                            if not (cont_diff or "").strip():
                                cont_diff = await agent_runtime.git_diff(sb, workdir=workdir)
            except asyncio.TimeoutError as exc:
                raise HybridPipelineTimeoutError(
                    stage="stage2",
                    phase="agent_pipeline",
                    instance_id=instance_id,
                    guard_sec=guard,
                    elapsed_sec=time.time() - t_agent,
                    detail=(
                        f"teacher trial={source_trial_idx} edit={edit_step_i} "
                        f"branch={branch_idx} branch_step_t={branch_step_t}"
                    ),
                ) from exc
            agent_elapsed = time.time() - t_agent

        if not resume_session_registered or not sft_path:
            raise RuntimeError("teacher_context_integrity:resume_session_not_registered")
        resume_status = teacher.adapter.resume_session_status(session_id)
        if resume_status.get("error"):
            raise RuntimeError(
                "teacher_context_integrity:adapter_rejected_context:"
                f"{resume_status['error']}"
            )
        if str(resume_status.get("log_path") or "") != sft_path:
            raise RuntimeError("teacher_context_integrity:attempt_log_path_changed")
        if str(resume_status.get("checkpoint_id") or "") != str(
            checkpoint.get("checkpoint_id") or ""
        ):
            raise RuntimeError("teacher_context_integrity:checkpoint_binding_changed")

        teacher_resolved: bool | None = None
        eval_elapsed = 0.0
        if grade:
            try:
                async with asyncio.timeout_at(pipeline_deadline):
                    t_eval = time.time()
                    eval_result = await gen._evaluate_diff(
                        image=image,
                        workdir=workdir,
                        eval_cmd=str(md.get("eval_cmd") or ""),
                        diff_text=cont_diff or "",
                        timeout_sec=eval_timeout,
                        metadata={**(sample.metadata or {}), **md},
                    )
                    eval_elapsed = time.time() - t_eval
            except asyncio.TimeoutError as exc:
                raise HybridPipelineTimeoutError(
                    stage="stage2",
                    phase="eval_pipeline",
                    instance_id=instance_id,
                    guard_sec=guard,
                    elapsed_sec=time.time() - t0,
                    detail=(
                        f"teacher trial={source_trial_idx} edit={edit_step_i} "
                        f"branch={branch_idx} agent_elapsed_sec={agent_elapsed:.1f}"
                    ),
                ) from exc
            teacher_resolved = bool(eval_result.resolved)
            if not teacher_resolved:
                logger.info(
                    "[teacher-sft] drop unresolved continuation instance=%s trial=%d edit=%d branch=%d",
                    instance_id,
                    source_trial_idx,
                    edit_step_i,
                    branch_idx,
                )
                return []

        if not os.path.isfile(sft_path):
            raise RuntimeError(f"teacher_sft_turns_missing:{sft_path}")
        sft_rows = load_sft_turn_rows(sft_path)
        if not sft_rows:
            raise RuntimeError(f"teacher_sft_turns_empty:{sft_path}")

        # Teacher rows train through the SFT loss with advantage 0, so the reward
        # is bookkeeping only and stays 0 when there is nothing to grade.
        reward: float = 0.0
        reward_details: dict[str, Any] = {}
        if grade:
            reward_path = getattr(
                args,
                "custom_cc_reward_function_path",
                "examples.claudecode_ags.rewards.default.compose",
            )
            reward_fn = load_function(reward_path)
            base_eval = {"resolved": eval_result.resolved, **eval_result.details}
            reward, reward_details = reward_fn(base_eval=base_eval, sample=branch_sample, args=args)

        step_group_key = f"{group_index}:{source_trial_idx}:edit:{edit_step_i}"
        encoded = encode_teacher_continuation_sample(
            sample=branch_sample,
            tokenizer=teacher.tokenizer,
            checkpoint=checkpoint,
            sft_rows=sft_rows,
            loss_mask_type=teacher.loss_mask_type,
            metadata={
                "instance_id": instance_id,
                "sample_kind": "teacher_sft",
                "source_trial_idx": source_trial_idx,
                "edit_step_i": edit_step_i,
                "branch_step_t": branch_step_t,
                "step_t": edit_step_i,
                "branch_idx": branch_idx,
                "step_group_key": step_group_key,
                "edit_ppl": edit_ppl,
                "branch_uid": f"teach:{step_group_key}:{branch_idx}",
                "reward_details": reward_details,
                "teacher_resolved": teacher_resolved,
                "teacher_max_steps": max_steps,
                "teacher_num_turns": len(sft_rows),
                "teacher_sft_log": sft_path,
                "teacher_queue_wait_sec": queue_wait,
                "teacher_agent_elapsed_sec": agent_elapsed,
                "teacher_eval_elapsed_sec": eval_elapsed,
                "teacher_episode_reward": float(reward),
            },
        )
        encoded.reward = float(reward)
        encoded.rollout_id = int(branch_sample.rollout_id)
        return [encoded]
    finally:
        if resume_session_registered:
            teacher.adapter.close_resume_session(session_id)
