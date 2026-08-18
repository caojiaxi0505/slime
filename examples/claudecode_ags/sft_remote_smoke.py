"""Run a small SWE-Gym Claude Code smoke with a remote OpenAI-compatible model.

This is for SFT data collection validation, not RL training.  It runs real
Claude Code in AGS sandboxes and uses ``RemoteOpenAISFTAdapter`` to log one
SFT-grade request/response row per model turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags.generate import _evaluate_diff, _parse_metadata
from examples.claudecode_ags.sft_remote_openai_adapter import RemoteOpenAISFTAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.sandbox import make_sandbox
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _load_samples(path: str, *, limit: int, offset: int = 0) -> list[Sample]:
    out: list[Sample] = []
    with open(path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < offset:
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            sample = Sample(
                index=line_idx,
                group_index=0,
                prompt=row.get("prompt") or "",
                metadata=dict(row.get("extra_info") or row.get("metadata") or {}),
            )
            out.append(sample)
            if len(out) >= limit:
                break
    return out


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def _load_session_turns(out_dir: Path, session_id_sha256: str) -> list[dict[str, Any]]:
    """Load turn-level SFT records for the main Claude Code session."""
    prefix = (session_id_sha256 or "")[:16]
    if not prefix:
        return []
    candidates = sorted((out_dir / "sft_turns").glob(f"{prefix}*.sft_turns.jsonl"))
    if not candidates:
        return []
    turns: list[dict[str, Any]] = []
    with candidates[0].open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
    turns.sort(key=lambda row: int(row.get("turn_index") or 0))
    return turns


def _assistant_message_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    response = turn.get("response") if isinstance(turn.get("response"), dict) else {}
    raw = response.get("raw_openai_message")
    if isinstance(raw, dict):
        msg = dict(raw)
        msg.setdefault("role", "assistant")
        return msg
    message = response.get("message")
    if isinstance(message, dict):
        msg = dict(message)
        msg.setdefault("role", "assistant")
        return msg
    return {"role": "assistant", "content": ""}


def _trial_messages_from_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one multi-turn SFT conversation from cumulative per-turn prompts.

    Each turn record stores the exact model-visible prompt for that request.
    The last prompt is the most complete context Claude Code sent before the
    final model response, so appending the last response yields a compact
    trial-level conversation while the original per-turn prompts remain
    available in ``sft_turns/`` for audit.
    """
    if not turns:
        return []
    prompt = turns[-1].get("prompt") if isinstance(turns[-1].get("prompt"), dict) else {}
    messages = prompt.get("messages") if isinstance(prompt.get("messages"), list) else []
    return [dict(m) for m in messages if isinstance(m, dict)] + [_assistant_message_from_turn(turns[-1])]


def _valid_sft_messages(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(isinstance(message, dict) for message in messages)
        and any(message.get("role") == "assistant" for message in messages)
        and messages[-1].get("role") == "assistant"
    )


def _slime_sft_row_from_trial(
    trial: dict[str, Any],
    *,
    resolved_only: bool,
) -> dict[str, Any] | None:
    messages = trial.get("messages")
    if not (trial.get("ok") and _valid_sft_messages(messages)):
        return None
    if resolved_only and not trial.get("resolved"):
        return None
    return {
        "messages": messages,
        "metadata": {
            "source": "deepseek_v4_pro_cc_ags_swegym",
            "instance_id": trial.get("instance_id"),
            "index": trial.get("index"),
            "session_id_sha256": trial.get("session_id_sha256"),
            "ok": bool(trial.get("ok")),
            "resolved": bool(trial.get("resolved")),
            "agent_exit_code": trial.get("agent_exit_code"),
            "applied_cleanly": bool(trial.get("applied_cleanly")),
            "diff_path": trial.get("diff_path"),
            "diff_chars": trial.get("diff_chars"),
            "trajectory_path": trial.get("trajectory_path"),
            "turn_count": trial.get("turn_count"),
        },
    }


def _eval_summary(details: Any) -> dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    keys = ("mode", "exit_code", "resolved", "reward_tests_status", "parser")
    return {k: details.get(k) for k in keys if k in details}


def _write_trial_record(out_dir: Path, result: dict[str, Any]) -> None:
    """Append one trial-level SFT row for slime training."""
    turns = _load_session_turns(out_dir, str(result.get("session_id_sha256") or ""))
    trial = {
        "version": 1,
        "format": "slime_sft_trial",
        "index": result.get("index"),
        "instance_id": result.get("instance_id"),
        "session_id_sha256": result.get("session_id_sha256"),
        "ok": result.get("ok"),
        "resolved": result.get("resolved"),
        "agent_exit_code": result.get("agent_exit_code"),
        "applied_cleanly": result.get("applied_cleanly"),
        "diff_path": result.get("diff_path"),
        "diff_chars": result.get("diff_chars"),
        "trajectory_path": result.get("trajectory_path"),
        "elapsed_sec": result.get("elapsed_sec"),
        "eval_summary": _eval_summary(result.get("eval_details")),
        "messages": _trial_messages_from_turns(turns),
        "turns": [
            {
                "turn_index": turn.get("turn_index"),
                "request_sha256": turn.get("request_sha256"),
                "request_wire_sha256": turn.get("request_wire_sha256"),
                "prompt_token_count": (turn.get("prompt") or {}).get("prompt_token_count")
                if isinstance(turn.get("prompt"), dict)
                else None,
                "output_token_count": (turn.get("response") or {}).get("output_token_count")
                if isinstance(turn.get("response"), dict)
                else None,
                "finish_reason": (turn.get("response") or {}).get("finish_reason")
                if isinstance(turn.get("response"), dict)
                else None,
                "stop_reason": (turn.get("response") or {}).get("stop_reason")
                if isinstance(turn.get("response"), dict)
                else None,
                "latency_sec": turn.get("latency_sec"),
                "usage": turn.get("usage"),
            }
            for turn in turns
        ],
        "turn_count": len(turns),
    }
    if not trial["messages"]:
        trial["warning"] = "main_session_turns_missing"
    _append_jsonl(out_dir / "sft_trials.jsonl", trial)
    slime_sft_all_row = _slime_sft_row_from_trial(trial, resolved_only=False)
    if slime_sft_all_row is not None:
        _append_jsonl(out_dir / "slime_sft_all.jsonl", slime_sft_all_row)
    slime_sft_resolved_row = _slime_sft_row_from_trial(trial, resolved_only=True)
    if slime_sft_resolved_row is not None:
        _append_jsonl(out_dir / "slime_sft_resolved.jsonl", slime_sft_resolved_row)


def _build_adapter(
    out_dir: Path,
    *,
    host: str,
    port: int,
    public_url: str | None = None,
) -> tuple[RemoteOpenAISFTAdapter, str, Any]:
    adapter = RemoteOpenAISFTAdapter(
        base_url=os.environ["SLIME_REMOTE_OPENAI_BASE_URL"],
        api_key=os.environ["SLIME_REMOTE_OPENAI_API_KEY"],
        model=os.environ["SLIME_REMOTE_OPENAI_MODEL"],
        sft_log_dir=str(out_dir / "sft_turns"),
        temperature=float(os.environ.get("SLIME_REMOTE_OPENAI_TEMPERATURE") or "1"),
        top_p=float(os.environ.get("SLIME_REMOTE_OPENAI_TOP_P") or "0.95"),
        top_k=int(os.environ.get("SLIME_REMOTE_OPENAI_TOP_K") or "20"),
        reasoning_effort=os.environ.get("SLIME_REMOTE_OPENAI_REASONING_EFFORT") or "max",
        thinking_type=os.environ.get("SLIME_REMOTE_OPENAI_THINKING_TYPE"),
    )
    handle = run_app_in_thread(
        adapter.app,
        host=host,
        port=port,
        thread_name="remote-openai-sft-smoke-adapter",
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )
    return adapter, (public_url or f"http://127.0.0.1:{handle.port}").rstrip("/"), handle


def _claude_env(*, adapter_url: str, session_id: str) -> dict[str, str]:
    prefixes = ("ANTHROPIC_", "CLAUDE_", "BASH_")
    exact = {
        "API_TIMEOUT_MS",
        "API_FORCE_IDLE_TIMEOUT",
        "TASK_MAX_OUTPUT_LENGTH",
        "MAX_MCP_OUTPUT_TOKENS",
        "MAX_THINKING_TOKENS",
        "IS_SANDBOX",
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if any(k.startswith(prefix) for prefix in prefixes) or k in exact
    }
    env.update(
        {
            "ANTHROPIC_BASE_URL": adapter_url,
            "ANTHROPIC_AUTH_TOKEN": session_id,
            "ANTHROPIC_MODEL": "slime-remote-openai",
            "ANTHROPIC_SMALL_FAST_MODEL": "slime-remote-openai",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") or "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": env.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") or "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": env.get("CLAUDE_CODE_ATTRIBUTION_HEADER") or "0",
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") or "101000",
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": env.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") or "95",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") or "16384",
            "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS": env.get("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS") or "16000",
            "BASH_MAX_OUTPUT_LENGTH": env.get("BASH_MAX_OUTPUT_LENGTH") or "65536",
            "TASK_MAX_OUTPUT_LENGTH": env.get("TASK_MAX_OUTPUT_LENGTH") or "32000",
            "MAX_MCP_OUTPUT_TOKENS": env.get("MAX_MCP_OUTPUT_TOKENS") or "25000",
            "MAX_THINKING_TOKENS": env.get("MAX_THINKING_TOKENS") or "0",
            "BASH_DEFAULT_TIMEOUT_MS": env.get("BASH_DEFAULT_TIMEOUT_MS") or "300000",
            "BASH_MAX_TIMEOUT_MS": env.get("BASH_MAX_TIMEOUT_MS") or "900000",
            "API_TIMEOUT_MS": env.get("API_TIMEOUT_MS") or "1200000",
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": env.get("CLAUDE_STREAM_IDLE_TIMEOUT_MS") or "1200000",
            "CLAUDE_ENABLE_STREAM_WATCHDOG": env.get("CLAUDE_ENABLE_STREAM_WATCHDOG") or "0",
            "CLAUDE_ENABLE_BYTE_WATCHDOG": env.get("CLAUDE_ENABLE_BYTE_WATCHDOG") or "0",
            "IS_SANDBOX": env.get("IS_SANDBOX") or "1",
        }
    )
    return env


async def _run_one(
    sample: Sample,
    *,
    adapter_url: str,
    out_dir: Path,
    agent_timeout: int,
    eval_timeout: int,
) -> dict[str, Any]:
    md = _parse_metadata(sample)
    session_id = f"deepseek-sft-{md['instance_id']}-{sample.index}-{secrets.token_hex(4)}"
    started = time.time()
    result: dict[str, Any] = {
        "index": sample.index,
        "instance_id": md["instance_id"],
        "repo": md.get("repo"),
        "session_id_sha256": __import__("hashlib").sha256(session_id.encode()).hexdigest(),
    }
    try:
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
            agent_result = await agent_runtime.run_claude(
                sb,
                workdir=md["workdir"],
                prompt=md["agent_prompt"],
                env=_claude_env(adapter_url=adapter_url, session_id=session_id),
                time_budget_sec=agent_timeout,
            )
            diff_text = await agent_runtime.git_diff(sb, workdir=md["workdir"])
        eval_result = await _evaluate_diff(
            image=md["image"],
            workdir=md["workdir"],
            eval_cmd=md["eval_cmd"],
            diff_text=diff_text,
            timeout_sec=eval_timeout,
            metadata=sample.metadata,
        )
        result.update(
            {
                "ok": True,
                "agent_exit_code": agent_result.get("exit_code"),
                "trajectory_path": agent_result.get("trajectory_path"),
                "diff_chars": len(diff_text or ""),
                "diff_path": str(out_dir / "diffs" / f"{sample.index}_{md['instance_id']}.diff"),
                "resolved": bool(eval_result.resolved),
                "applied_cleanly": bool(eval_result.applied_cleanly),
                "eval_details": eval_result.details,
            }
        )
        diff_path = Path(result["diff_path"])
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(diff_text or "", encoding="utf-8")
    except Exception as exc:
        logger.exception("task failed instance_id=%s", md["instance_id"])
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_sec"] = time.time() - started
    return result


async def amain() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--adapter-host", default="127.0.0.1")
    parser.add_argument("--adapter-port", type=int, default=18081)
    parser.add_argument("--adapter-public-url", default=os.environ.get("SLIME_ADAPTER_PUBLIC_URL") or "")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--agent-timeout", type=int, default=int(os.environ.get("SLIME_CC_TIME_BUDGET_SEC") or "1800"))
    parser.add_argument("--eval-timeout", type=int, default=int(os.environ.get("SLIME_CC_EVAL_TIMEOUT_SEC") or "600"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter, adapter_url, adapter_handle = _build_adapter(
        out_dir,
        host=args.adapter_host,
        port=args.adapter_port,
        public_url=args.adapter_public_url or None,
    )
    samples = _load_samples(args.data, limit=args.limit, offset=args.offset)
    concurrency = max(1, int(args.concurrency or 1))
    logger.info(
        "loaded %d samples adapter_url=%s out_dir=%s concurrency=%d",
        len(samples),
        adapter_url,
        out_dir,
        concurrency,
    )
    try:
        sem = asyncio.Semaphore(concurrency)

        async def _run_guarded(sample: Sample) -> dict[str, Any]:
            async with sem:
                return await _run_one(
                    sample,
                    adapter_url=adapter_url,
                    out_dir=out_dir,
                    agent_timeout=args.agent_timeout,
                    eval_timeout=args.eval_timeout,
                )

        tasks = [asyncio.create_task(_run_guarded(sample)) for sample in samples]
        for fut in asyncio.as_completed(tasks):
            row = await fut
            _append_jsonl(out_dir / "results.jsonl", row)
            _write_trial_record(out_dir, row)
            done = sum(1 for t in tasks if t.done())
            logger.info(
                "result %s resolved=%s ok=%s elapsed=%.1fs progress=%d/%d",
                row["instance_id"],
                row.get("resolved"),
                row.get("ok"),
                row["elapsed_sec"],
                done,
                len(tasks),
            )
    finally:
        adapter_handle.stop()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
