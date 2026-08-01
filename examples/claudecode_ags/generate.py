"""CC + AGS per-sample generate() for slime rollouts.

    --custom-generate-function-path examples.claudecode_ags.generate.generate
    --custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags.swe_eval import dispatch as swe_eval_dispatch
from examples.claudecode_ags.swe_eval.base import EvalResult
from slime.agent.adapters.anthropic_segmented import SegmentedAnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.sandbox import make_sandbox
from slime.agent.segment_trajectory import fan_out_sample_segments
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_CC_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_", "BASH_")
_CC_ENV_EXACT = frozenset(
    {
        "API_TIMEOUT_MS",
        "API_FORCE_IDLE_TIMEOUT",
        "TASK_MAX_OUTPUT_LENGTH",
        "MAX_MCP_OUTPUT_TOKENS",
        "MAX_THINKING_TOKENS",
        "IS_SANDBOX",
    }
)
_DEFAULT_AGENT_PROMPT = "Read PROBLEM_STATEMENT.md and fix the issue. Keep changes minimal."
_CLIENT_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _session_sampling_overrides(
    sampling_params: dict[str, Any] | None,
    *,
    evaluation: bool,
) -> dict[str, Any]:
    """Return framework parameters that must win over the Claude request body.

    Claude Code currently sends its own sampling knobs (notably
    ``temperature=1``). Evaluation must instead use slime's explicit
    ``--eval-*`` settings. Training keeps its existing request-body behavior.
    """
    if not evaluation:
        return {}
    params = sampling_params or {}
    return {key: params[key] for key in _CLIENT_SAMPLING_KEYS if key in params}


def _artifact_component(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return (text or fallback)[:160]


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


async def _save_eval_artifacts(
    *,
    sample: Sample,
    instance_id: str,
    session_id: str,
    agent_result: dict[str, Any],
    diff_text: str,
    agent_prompt: str,
    agent_time_budget_sec: int,
    eval_result: EvalResult | None = None,
) -> None:
    """Persist audit-grade per-trial artifacts when explicitly enabled.

    The adapter's ``SLIME_AGENT_SFT_LOG_DIR`` stores the exact model-visible
    request/response for every turn. This complementary export preserves the
    Claude Code trajectory, model patch, and grading result after AGS sandboxes
    are destroyed.
    """
    root = (os.environ.get("SLIME_CC_EVAL_ARTIFACT_DIR") or "").strip()
    if not root:
        return

    instance_component = _artifact_component(instance_id, fallback="unknown")
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    trial_dir = Path(root) / instance_component / session_hash
    manifest = {
        "version": 1,
        "instance_id": instance_id,
        "session_id": session_id,
        "session_id_sha256": hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
        "sample_index": sample.index,
        "sample_group_index": sample.group_index,
        "agent_exit_code": agent_result.get("exit_code"),
        "trajectory_path_in_sandbox": agent_result.get("trajectory_path"),
        "model_patch_chars": len(diff_text),
        "evaluation_complete": eval_result is not None,
        "agent_runtime": {
            "time_budget_sec": agent_time_budget_sec,
            "initial_input_mode": (
                os.environ.get("SLIME_CC_INITIAL_INPUT_MODE") or "stream-json"
            ),
            "max_output_tokens": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
            "auto_compact_window": os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"),
            "agent_prompt": agent_prompt,
            "agent_prompt_sha256": hashlib.sha256(
                agent_prompt.encode("utf-8")
            ).hexdigest(),
        },
    }
    writes: dict[Path, str] = {
        trial_dir / "manifest.json": json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        trial_dir / "trajectory.jsonl": str(agent_result.get("trajectory_jsonl") or ""),
        trial_dir / "model.patch": diff_text,
    }
    if eval_result is not None:
        writes[trial_dir / "eval_result.json"] = (
            json.dumps(
                {
                    "resolved": bool(eval_result.resolved),
                    "applied_cleanly": bool(eval_result.applied_cleanly),
                    "details": eval_result.details,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n"
        )

    try:
        await asyncio.gather(
            *(asyncio.to_thread(_atomic_write, path, data) for path, data in writes.items())
        )
    except Exception:
        logger.exception(
            "[claudecode_ags] %s: failed to persist eval artifacts under %s",
            instance_id,
            trial_dir,
        )


# Global in-flight cap for CC agents (sandbox + claude). Eval sandboxes are outside.
_AGENT_SEM: asyncio.Semaphore | None = None
_AGENT_SEM_LIMIT: int | None = None
_EVAL_SEM: asyncio.Semaphore | None = None
_EVAL_SEM_LIMIT: int | None = None


def agent_concurrency_cm():
    """Limit concurrent agent runs cluster-wide in this process.

    Env ``SLIME_CC_AGENT_CONCURRENCY`` (default 64). ``<=0`` disables the gate.
    Hold for the whole agent sandbox lifetime (create → claude → diff), not just
    StartSandbox submit.
    """
    from contextlib import nullcontext

    global _AGENT_SEM, _AGENT_SEM_LIMIT
    n = _env_int("SLIME_CC_AGENT_CONCURRENCY", 64)
    if n <= 0:
        return nullcontext()
    if _AGENT_SEM is None or _AGENT_SEM_LIMIT != n:
        _AGENT_SEM = asyncio.Semaphore(n)
        _AGENT_SEM_LIMIT = n
        logger.info("[claudecode_ags] agent concurrency limit=%d", n)
    return _AGENT_SEM


@asynccontextmanager
async def eval_concurrency_cm():
    """Limit concurrent clean-sandbox evaluators in this process.

    Env ``SLIME_CC_EVAL_CONCURRENCY`` defaults to 64. ``<=0`` disables the
    gate. The slot covers eval sandbox creation, workspace setup, patch apply,
    test execution, parsing, and any fresh-sandbox retry.
    """
    global _EVAL_SEM, _EVAL_SEM_LIMIT
    n = _env_int("SLIME_CC_EVAL_CONCURRENCY", 64)
    if n <= 0:
        yield 0.0
        return
    if _EVAL_SEM is None or _EVAL_SEM_LIMIT != n:
        _EVAL_SEM = asyncio.Semaphore(n)
        _EVAL_SEM_LIMIT = n
        logger.info("[claudecode_ags] eval concurrency limit=%d", n)
    queued_at = time.time()
    async with _EVAL_SEM:
        yield time.time() - queued_at


def _timeouts() -> tuple[int, int, int]:
    budget = _env_int("SLIME_CC_TIME_BUDGET_SEC", 1800)
    eval_timeout = _env_int("SLIME_CC_EVAL_TIMEOUT_SEC", 600)
    # One absolute deadline covers agent sandbox boot through final evaluation.
    # Agent-concurrency queueing remains outside this budget.
    guard = _env_int("SLIME_CC_GENERATE_GUARD_SEC", 0) or 2700
    return budget, eval_timeout, guard


def _adapter_bind() -> tuple[str, int]:
    return os.environ.get("SLIME_ADAPTER_BIND_HOST", "0.0.0.0"), _env_int("SLIME_ADAPTER_PORT", 18001)


def _adapter_public_url(bound_port: int) -> str:
    public = (os.environ.get("SLIME_ADAPTER_PUBLIC_URL") or "").strip().rstrip("/")
    if public:
        return public
    host = os.environ.get("SLIME_ADAPTER_BIND_HOST", "0.0.0.0")
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{bound_port}"


def _filter_cc_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Pass through Claude Code / Anthropic / Bash knobs only — no SWE_* aliases."""
    src = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for key, value in src.items():
        if key.startswith(_CC_ENV_PREFIXES) or key in _CC_ENV_EXACT:
            out[key] = value
    return out


def _build_claude_env(*, adapter_url: str, session_id: str, model_label: str = "claude-sonnet") -> dict[str, str]:
    env = _filter_cc_env()
    env.update(
        {
            "ANTHROPIC_BASE_URL": adapter_url,
            "ANTHROPIC_AUTH_TOKEN": session_id,
            "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL") or model_label,
            "IS_SANDBOX": env.get("IS_SANDBOX") or "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") or "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": env.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") or "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": env.get("CLAUDE_CODE_ATTRIBUTION_HEADER") or "0",
        }
    )
    return env


def _parse_metadata(sample: Sample) -> dict[str, Any]:
    from examples.claudecode_ags.bug_patch_source import resolve_swe_smith_bug_patch
    from examples.claudecode_ags.dataset_normalize import normalize_official_row
    from examples.claudecode_ags.workspace_init import coerce_install_config, normalize_pre_commands

    md = dict(sample.metadata or {})
    dataset_type = str(md.get("dataset_type") or "").strip()
    # Verified parquet nests the docker image under sandbox_overrides; train
    # jsonl stores it as top-level ``image``. Normalize either shape.
    if not str(md.get("image") or "").strip():
        so = md.get("sandbox_overrides") if isinstance(md.get("sandbox_overrides"), dict) else {}
        docker_image = str(so.get("docker_image") or "").strip()
        if docker_image:
            md["image"] = docker_image
        elif not dataset_type and str(md.get("instance_id") or "").strip():
            # Official SWE-bench Verified rows without dataset_type still have
            # instance_id; derive the registry image the same way as normalize.
            dataset_type = "swebench_verified"
    if dataset_type:
        md = normalize_official_row(md, dataset_type=dataset_type)

    prompt = sample.prompt if isinstance(sample.prompt, str) else ""
    problem = md.get("problem_statement") or prompt
    swebench = md.get("swebench") if isinstance(md.get("swebench"), dict) else {}
    base_commit = str(md.get("base_commit") or swebench.get("base_commit") or "").strip()
    patch = resolve_swe_smith_bug_patch(md)
    return {
        "image": (md.get("image") or "").strip(),
        "workdir": (md.get("workdir") or "/testbed").strip(),
        "problem_statement": problem,
        "eval_cmd": (md.get("eval_cmd") or "").strip(),
        "instance_id": str(md.get("instance_id") or "unknown"),
        "agent_prompt": (
            md.get("agent_prompt")
            or os.environ.get("SLIME_CC_AGENT_PROMPT")
            or _DEFAULT_AGENT_PROMPT
        ).strip(),
        "data_source": str(md.get("data_source") or ""),
        "base_commit": base_commit,
        "swe_smith_bug_patch": patch or md.get("swe_smith_bug_patch"),
        "pre_commands": normalize_pre_commands(md.get("pre_commands")),
        "install_config": coerce_install_config(md.get("install_config")),
        "cc_source": md.get("cc_source") if isinstance(md.get("cc_source"), dict) else {},
        "data_path": md.get("data_path"),
        "dataset_type": str(md.get("dataset_type") or ""),
    }


def _f2p_p2p_metrics(base_eval: dict[str, Any]) -> dict[str, Any]:
    """Flatten F2P/P2P pass ratios from SWE grading details for Wandb."""
    report = base_eval.get("reward_tests_status") or {}
    f2p = report.get("FAIL_TO_PASS") or {}
    p2p = report.get("PASS_TO_PASS") or {}
    out: dict[str, Any] = {}
    if "pass_ratio" in f2p:
        out["test_f2p_ratio"] = float(f2p["pass_ratio"])
        out["test_f2p_passed"] = int(f2p.get("pass_count") or 0)
        out["test_f2p_total"] = int(f2p.get("total") or 0)
    if "pass_ratio" in p2p:
        out["test_p2p_ratio"] = float(p2p["pass_ratio"])
        out["test_p2p_passed"] = int(p2p.get("pass_count") or 0)
        out["test_p2p_total"] = int(p2p.get("total") or 0)
    return out


def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"ccags-{instance_id}-{sample.index}-{sample.group_index}"
    return f"ccags-{instance_id}-{secrets.token_hex(8)}"


def _abort(sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason, "instance_id": instance_id}
    logger.warning("[claudecode_ags] %s aborted: %s", instance_id, reason)
    return [sample]


def _eval_only(sample: Sample, *, reward: float, details: dict[str, Any], instance_id: str) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = float(reward)
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "grading_solved": float(reward) == 1.0,
        "reward_details": details,
    }
    return [sample]


class _AdapterService(metaclass=SingletonMeta):
    """One threaded Anthropic adapter HTTP server per process."""

    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        bind_host, bind_port = _adapter_bind()
        self.adapter = SegmentedAnthropicAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=bind_host,
            port=bind_port,
            thread_name="cc-ags-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = _adapter_public_url(self.app_handle.port)
        logger.info(
            "[claudecode_ags] adapter=%s tokenizer=%s cpu_workers=%d",
            self.adapter_url,
            args.hf_checkpoint,
            self.adapter.cpu_workers,
        )


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Boot sandbox → run Claude Code → eval → pluggable reward → fan-out segments."""
    md = _parse_metadata(sample)
    instance_id = md["instance_id"]
    if not md["image"] or not md["workdir"]:
        return _abort(sample, "missing_image_or_workdir", instance_id)

    time_budget, eval_timeout, guard = _timeouts()
    state = _AdapterService(args)
    session_id = sample.session_id = _session_id(sample, instance_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        sampling_overrides=_session_sampling_overrides(
            sampling_params,
            evaluation=evaluation,
        ),
        max_context_tokens=state.max_context_len,
    )

    t0 = time.time()
    pipeline_started: float | None = None
    try:
        claude_env = _build_claude_env(adapter_url=state.adapter_url, session_id=session_id)
        # Acquire the agent slot before starting the shared Agent→eval deadline,
        # so agent queue wait does not burn the 2700-second pipeline budget.
        t_queue = time.time()
        async with agent_concurrency_cm():
            queue_wait = time.time() - t_queue
            pipeline_started = time.time()
            pipeline_deadline = asyncio.get_running_loop().time() + guard
            t_agent = time.time()
            async with asyncio.timeout_at(pipeline_deadline):
                async with make_sandbox(md["image"]) as sb:
                    try:
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
                    except RuntimeError as e:
                        return _abort(sample, f"workspace_init:{e}", instance_id)
                    await agent_runtime.install_toolchain(sb)
                    agent_result = await agent_runtime.run_claude(
                        sb,
                        workdir=md["workdir"],
                        prompt=md["agent_prompt"],
                        env=claude_env,
                        time_budget_sec=time_budget,
                    )
                    diff_text = await agent_runtime.git_diff(sb, workdir=md["workdir"])
            agent_elapsed = time.time() - t_agent

        # Re-enter the same absolute deadline after releasing the agent slot.
        # This covers artifact export, eval-slot queueing, fresh sandbox setup,
        # test execution, parsing, and retry without holding an agent slot.
        async with asyncio.timeout_at(pipeline_deadline):
            # Export the sandbox-resident trajectory before grading so agent
            # artifacts survive even if eval setup or test execution later fails.
            if evaluation:
                await _save_eval_artifacts(
                    sample=sample,
                    instance_id=instance_id,
                    session_id=session_id,
                    agent_result=agent_result,
                    diff_text=diff_text,
                    agent_prompt=md["agent_prompt"],
                    agent_time_budget_sec=time_budget,
                )

            # Keep official grading fields (FAIL_TO_PASS, repo, …) from sample.metadata.
            t_eval = time.time()
            evaluate = _evaluate_diff_with_infra_retry if evaluation else _evaluate_diff
            eval_result = await evaluate(
                image=md["image"],
                workdir=md["workdir"],
                eval_cmd=md["eval_cmd"],
                diff_text=diff_text,
                timeout_sec=eval_timeout,
                metadata={**(sample.metadata or {}), **md},
            )
            eval_elapsed = time.time() - t_eval
        eval_queue_wait = float(eval_result.details.get("eval_queue_wait_sec") or 0.0)

        reward_path = getattr(
            args,
            "custom_cc_reward_function_path",
            "examples.claudecode_ags.rewards.default.compose",
        )
        reward_fn = load_function(reward_path)
        base_eval = {"resolved": eval_result.resolved, **eval_result.details}
        reward, reward_details = reward_fn(base_eval=base_eval, sample=sample, args=args)
        f2p_p2p = _f2p_p2p_metrics(base_eval)

        if evaluation:
            await _save_eval_artifacts(
                sample=sample,
                instance_id=instance_id,
                session_id=session_id,
                agent_result=agent_result,
                diff_text=diff_text,
                agent_prompt=md["agent_prompt"],
                agent_time_budget_sec=time_budget,
                eval_result=eval_result,
            )
            return _eval_only(
                sample,
                reward=float(reward),
                details={
                    **reward_details,
                    **f2p_p2p,
                    "eval_queue_wait_sec": eval_queue_wait,
                },
                instance_id=instance_id,
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
            timeout_outcome_enabled=_env_int("SLIME_CC_TIMEOUT_OUTCOME_REWARD", 0) > 0,
            tool_loop_enabled=_env_int("SLIME_CC_TOOL_LOOP_PENALTY", 0) > 0,
        )

        total_elapsed = time.time() - t0
        samples = fan_out_sample_segments(
            sample,
            segments,
            reward=float(reward),
            tokenizer=state.tokenizer,
            metadata={
                "instance_id": instance_id,
                "grading_solved": bool(eval_result.resolved),
                "applied_cleanly": bool(eval_result.applied_cleanly),
                "agent_exit_code": agent_result.get("exit_code"),
                **reward_audit,
                "reward_details": reward_details,
                "base_eval": base_eval,
                "agent_elapsed_sec": agent_elapsed,
                "agent_queue_wait_sec": queue_wait,
                "eval_elapsed_sec": eval_elapsed,
                "eval_queue_wait_sec": eval_queue_wait,
                "total_elapsed_sec": total_elapsed,
                **f2p_p2p,
            },
        )
        if not samples:
            return _abort(sample, "adapter_session_empty", instance_id)

        logger.info(
            "[claudecode_ags] %s: reward=%.2f resolved=%s f2p=%s p2p=%s "
            "segments=%d agent=%.1fs eval=%.1fs eval_queue=%.1fs elapsed=%.1fs",
            instance_id,
            float(reward),
            bool(eval_result.resolved),
            f2p_p2p.get("test_f2p_ratio"),
            f2p_p2p.get("test_p2p_ratio"),
            len(samples),
            agent_elapsed,
            eval_elapsed,
            eval_queue_wait,
            total_elapsed,
        )
        return samples

    except asyncio.TimeoutError:
        pipeline_elapsed = time.time() - (pipeline_started or t0)
        logger.warning(
            "[claudecode_ags] %s: Agent→eval pipeline guard timed out "
            "after %.1fs (limit=%ds)",
            instance_id,
            pipeline_elapsed,
            guard,
        )
        return _abort(sample, "wall_clock_timeout", instance_id)
    except Exception as e:
        from examples.claudecode_ags.swe_eval.official import SwebenchHarnessError

        if evaluation and isinstance(e, SwebenchHarnessError):
            logger.exception(
                "[claudecode_ags] %s: fatal SWE-bench harness failure",
                instance_id,
            )
            raise
        logger.warning(
            "[claudecode_ags] %s: rollout failed: %s\n%s",
            instance_id,
            e,
            traceback.format_exc(),
        )
        return _abort(sample, f"exception:{type(e).__name__}", instance_id)
    finally:
        try:
            await state.adapter.shutdown_session(session_id, wait_timeout=5.0)
        except Exception:
            pass


async def _evaluate_diff_once(
    *,
    image: str,
    workdir: str,
    eval_cmd: str,
    diff_text: str,
    timeout_sec: int,
    metadata: dict[str, Any] | None = None,
) -> EvalResult:
    from examples.claudecode_ags.workspace_init import initialize_task_workspace, task_fields_from_metadata

    md = dict(metadata or {})
    if eval_cmd and not md.get("eval_cmd"):
        md["eval_cmd"] = eval_cmd
    if workdir:
        md["workdir"] = workdir

    fields = task_fields_from_metadata(md)
    fields.workdir = workdir or fields.workdir
    async with make_sandbox(image) as sb:
        ok = await initialize_task_workspace(sb, fields, rollout_side=False)
        if not ok:
            return EvalResult(
                resolved=False,
                applied_cleanly=False,
                details={"reason": "eval_workspace_init_failed"},
            )
        return await swe_eval_dispatch.evaluate(
            sb,
            metadata=md,
            diff_text=diff_text,
            timeout_sec=timeout_sec,
        )


def _with_eval_queue_wait(result: EvalResult, queue_wait: float) -> EvalResult:
    result.details["eval_queue_wait_sec"] = float(queue_wait)
    return result


async def _evaluate_diff(
    *,
    image: str,
    workdir: str,
    eval_cmd: str,
    diff_text: str,
    timeout_sec: int,
    metadata: dict[str, Any] | None = None,
) -> EvalResult:
    async with eval_concurrency_cm() as queue_wait:
        result = await _evaluate_diff_once(
            image=image,
            workdir=workdir,
            eval_cmd=eval_cmd,
            diff_text=diff_text,
            timeout_sec=timeout_sec,
            metadata=metadata,
        )
    return _with_eval_queue_wait(result, queue_wait)


async def _evaluate_diff_with_infra_retry(
    *,
    image: str,
    workdir: str,
    eval_cmd: str,
    diff_text: str,
    timeout_sec: int,
    metadata: dict[str, Any] | None = None,
) -> EvalResult:
    """Bound the whole evaluator attempt and retry infra failures in a fresh sandbox."""
    retries = max(0, _env_int("SLIME_CC_EVAL_INFRA_RETRIES", 1))
    guard_sec = max(
        timeout_sec,
        _env_int("SLIME_CC_EVAL_GUARD_SEC", timeout_sec + 180),
    )
    from examples.claudecode_ags.swe_eval.official import SwebenchHarnessError
    from slime.agent.sandbox_ags import _is_transient_request_error

    async with eval_concurrency_cm() as queue_wait:
        for attempt in range(retries + 1):
            try:
                # Eval concurrency queueing is bounded by the outer pipeline
                # deadline, but does not consume this per-attempt guard.
                async with asyncio.timeout(guard_sec):
                    result = await _evaluate_diff_once(
                        image=image,
                        workdir=workdir,
                        eval_cmd=eval_cmd,
                        diff_text=diff_text,
                        timeout_sec=timeout_sec,
                        metadata=metadata,
                    )
                return _with_eval_queue_wait(result, queue_wait)
            except SwebenchHarnessError:
                raise
            except Exception as exc:
                retryable = isinstance(exc, asyncio.TimeoutError) or _is_transient_request_error(exc)
                if not retryable or attempt >= retries:
                    raise
                logger.warning(
                    "[claudecode_ags] eval infra failure; retry %d/%d in a fresh sandbox: %s",
                    attempt + 1,
                    retries,
                    exc,
                )
    raise AssertionError("unreachable")
