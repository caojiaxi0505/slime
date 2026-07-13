"""Single-attempt runners for gold eval and live Claude Code (L1/L2 primitives)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags.smoke import ags_smoke
from examples.claudecode_ags.swe_eval import dispatch as swe_eval_dispatch
from examples.claudecode_ags.workspace_init import initialize_task_workspace, task_fields_from_metadata
from slime.agent.sandbox import EXIT_TIME_BUDGET_EXCEEDED, make_sandbox


async def run_gold_attempt(md: dict[str, Any], *, eval_timeout: int) -> dict[str, Any]:
    """One gold-patch evaluate in a fresh sandbox (L1 path)."""
    t0 = time.monotonic()
    image = md["image"]
    gold = str(md.get("patch") or "").strip()
    if not gold:
        return {
            "resolved": False,
            "applied_cleanly": False,
            "infra_ok": False,
            "timeout_hit": False,
            "elapsed_sec": 0.0,
            "details": {},
            "error": "empty gold patch",
        }

    fields = task_fields_from_metadata(md)
    try:
        async with make_sandbox(image) as sb:
            ok = await initialize_task_workspace(sb, fields, rollout_side=False)
            if not ok:
                return {
                    "resolved": False,
                    "applied_cleanly": False,
                    "infra_ok": False,
                    "timeout_hit": False,
                    "elapsed_sec": time.monotonic() - t0,
                    "details": {},
                    "error": "workspace_init_failed",
                    "sandbox_id": getattr(sb, "sandbox_id", ""),
                }
            result = await swe_eval_dispatch.evaluate(
                sb,
                metadata=md,
                diff_text=gold,
                timeout_sec=eval_timeout,
            )
            details = dict(result.details or {})
            exit_code = details.get("exit_code")
            timeout_hit = bool(details.get("timeout")) or exit_code in (-1, "timeout")
            return {
                "resolved": bool(result.resolved),
                "applied_cleanly": bool(result.applied_cleanly),
                "infra_ok": True,
                "timeout_hit": timeout_hit,
                "elapsed_sec": time.monotonic() - t0,
                "details": {
                    "mode": details.get("mode"),
                    "exit_code": exit_code,
                    "reason": details.get("reason"),
                },
                "error": None,
                "sandbox_id": getattr(sb, "sandbox_id", ""),
            }
    except Exception as e:
        return {
            "resolved": False,
            "applied_cleanly": False,
            "infra_ok": False,
            "timeout_hit": False,
            "elapsed_sec": time.monotonic() - t0,
            "details": {},
            "error": f"{type(e).__name__}: {e}",
        }


async def run_passk_attempt(
    md: dict[str, Any],
    *,
    time_budget: int,
    eval_timeout: int,
    artifact_dir: Path | None,
    skip_health_gate: bool = False,
) -> dict[str, Any]:
    """One live Claude Code → model-diff eval (L2 path)."""
    t0 = time.monotonic()
    image = md["image"]
    instance_id = str(md.get("instance_id") or "unknown")
    workdir = str(md.get("workdir") or "/testbed").strip() or "/testbed"
    problem = str(md.get("problem_statement") or md.get("problem") or "").strip()
    if not problem:
        return {
            "resolved": False,
            "applied_cleanly": False,
            "infra_ok": False,
            "timeout_hit": False,
            "elapsed_sec": 0.0,
            "diff_chars": 0,
            "claude_exit": None,
            "artifact_dir": None,
            "details": {},
            "error": "missing problem_statement",
        }

    try:
        adapter_url = ags_smoke._adapter_public_url()
    except Exception as e:
        return {
            "resolved": False,
            "applied_cleanly": False,
            "infra_ok": False,
            "timeout_hit": False,
            "elapsed_sec": time.monotonic() - t0,
            "diff_chars": 0,
            "claude_exit": None,
            "artifact_dir": None,
            "details": {},
            "error": str(e),
        }

    session_id = f"passk-{instance_id}-{int(time.time())}"
    prompt = str(md.get("agent_prompt") or "").strip() or (
        "Read PROBLEM_STATEMENT.md and fix the issue. Keep changes minimal."
    )
    diff_text = ""
    claude_exit: int | None = None
    art_path: str | None = None

    try:
        async with make_sandbox(image) as sb:
            if not skip_health_gate:
                import shlex

                health_url = f"{adapter_url}/health"
                cmd = (
                    f"curl -fsS --max-time 30 {shlex.quote(health_url)} "
                    f"|| curl -fsS --max-time 30 {shlex.quote(adapter_url + '/healthz')}"
                )
                ec, out, err = await sb.exec(cmd, user="root", check=False, timeout=60)
                if ec != 0:
                    return {
                        "resolved": False,
                        "applied_cleanly": False,
                        "infra_ok": False,
                        "timeout_hit": False,
                        "elapsed_sec": time.monotonic() - t0,
                        "diff_chars": 0,
                        "claude_exit": None,
                        "artifact_dir": None,
                        "details": {"health_stdout": out, "health_stderr": err},
                        "error": "health_gate_failed",
                        "sandbox_id": getattr(sb, "sandbox_id", ""),
                    }

            await agent_runtime.prepare_workspace(
                sb,
                workdir=workdir,
                problem_statement=problem,
                instance_id=instance_id,
                data_source=str(md.get("data_source") or ""),
                base_commit=str(md.get("base_commit") or ""),
                swe_smith_bug_patch=md.get("swe_smith_bug_patch"),
                pre_commands=md.get("pre_commands") or "",
                install_config=md.get("install_config") or {},
                rollout_side=True,
            )
            await agent_runtime.install_toolchain(sb)
            claude_env = ags_smoke._build_claude_env(adapter_url=adapter_url, session_id=session_id)
            agent_result = await agent_runtime.run_claude(
                sb,
                workdir=workdir,
                prompt=prompt,
                env=claude_env,
                time_budget_sec=time_budget,
            )
            claude_exit = int(agent_result.get("exit_code"))
            diff_text = await agent_runtime.git_diff(sb, workdir=workdir)
            if artifact_dir is not None:
                dest = await ags_smoke.export_rollout_artifacts(
                    sb,
                    workdir=workdir,
                    dest_dir=artifact_dir,
                    diff_text=diff_text,
                    session_id=session_id,
                    instance_id=instance_id,
                )
                art_path = str(dest)

        fields = task_fields_from_metadata(md)
        fields.workdir = workdir or fields.workdir
        async with make_sandbox(image) as sb_eval:
            ok = await initialize_task_workspace(sb_eval, fields, rollout_side=False)
            if not ok:
                return {
                    "resolved": False,
                    "applied_cleanly": False,
                    "infra_ok": False,
                    "timeout_hit": claude_exit == EXIT_TIME_BUDGET_EXCEEDED,
                    "elapsed_sec": time.monotonic() - t0,
                    "diff_chars": len(diff_text),
                    "claude_exit": claude_exit,
                    "artifact_dir": art_path,
                    "details": {},
                    "error": "eval_workspace_init_failed",
                }
            result = await swe_eval_dispatch.evaluate(
                sb_eval,
                metadata=md,
                diff_text=diff_text,
                timeout_sec=eval_timeout,
            )
            details = dict(result.details or {})
            return {
                "resolved": bool(result.resolved),
                "applied_cleanly": bool(result.applied_cleanly),
                "infra_ok": True,
                "timeout_hit": claude_exit == EXIT_TIME_BUDGET_EXCEEDED
                or bool(details.get("timeout")),
                "elapsed_sec": time.monotonic() - t0,
                "diff_chars": len(diff_text),
                "claude_exit": claude_exit,
                "artifact_dir": art_path,
                "details": {
                    "mode": details.get("mode"),
                    "exit_code": details.get("exit_code"),
                    "reason": details.get("reason"),
                },
                "error": None,
                "eval_sandbox_id": getattr(sb_eval, "sandbox_id", ""),
            }
    except Exception as e:
        return {
            "resolved": False,
            "applied_cleanly": False,
            "infra_ok": False,
            "timeout_hit": claude_exit == EXIT_TIME_BUDGET_EXCEEDED,
            "elapsed_sec": time.monotonic() - t0,
            "diff_chars": len(diff_text),
            "claude_exit": claude_exit,
            "artifact_dir": art_path,
            "details": {},
            "error": f"{type(e).__name__}: {e}",
        }
