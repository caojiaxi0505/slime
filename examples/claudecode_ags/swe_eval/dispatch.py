"""Dispatch SWE eval: resolve plan → apply patches → run → grade."""

from __future__ import annotations

import logging
from typing import Any

from examples.claudecode_ags.swe_eval import rebench as rebench_mod
from examples.claudecode_ags.swe_eval import scaleswe as scaleswe_mod
from examples.claudecode_ags.swe_eval import simple_cmd
from examples.claudecode_ags.swe_eval import swebench as swebench_mod
from examples.claudecode_ags.swe_eval.base import EvalResult
from examples.claudecode_ags.swe_eval.cmd_resolve import EvalMode, EvalPlan, resolve_eval_plan

logger = logging.getLogger(__name__)

_MODEL_PATCH = "/tmp/slime_model_patch.diff"
_TEST_PATCH = "/tmp/slime_test_patch.diff"
_F2P_PATCH = "/tmp/slime_f2p_patch.diff"
_F2P_SCRIPT = "test_fail_to_pass.py"


async def _apply_patch(sb, workdir: str, path: str, diff_text: str) -> tuple[bool, str]:
    if not diff_text.strip():
        return True, ""
    # Parquet/json rows sometimes drop the trailing newline; patch(1) then fails mid-line.
    text = diff_text if diff_text.endswith("\n") else diff_text + "\n"
    await sb.write_file(path, text, user="agent")
    last_err = ""
    for cmd in (
        f"cd {workdir} && git apply --3way --ignore-space-change --ignore-whitespace --whitespace=nowarn {path}",
        f"cd {workdir} && git apply --ignore-space-change --ignore-whitespace --whitespace=nowarn {path}",
        f"cd {workdir} && git apply --whitespace=nowarn {path}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {path}",
    ):
        ec, out, err = await sb.exec(cmd, user="agent", check=False, timeout=120)
        last_err = (err or out or "").strip()
        if ec == 0:
            return True, ""
    return False, last_err[:2000]


async def evaluate(
    sb,
    *,
    metadata: dict[str, Any],
    diff_text: str = "",
    timeout_sec: int = 600,
    plan: EvalPlan | None = None,
) -> EvalResult:
    plan = plan or resolve_eval_plan(metadata)
    if plan.mode == EvalMode.NONE or not plan.eval_cmd.strip():
        return EvalResult(
            resolved=False,
            applied_cleanly=True,
            details={"reason": "missing_eval_plan", "mode": plan.mode.value},
        )

    workdir = plan.workdir

    # SWE-bench order: test/f2p fixtures first, then model (gold) patch.
    if plan.test_patch:
        ok, err = await _apply_patch(sb, workdir, _TEST_PATCH, plan.test_patch)
        if not ok:
            return EvalResult(
                False,
                True,
                {"reason": "test_patch_failed", "mode": plan.mode.value, "apply_stderr": err},
            )
    if plan.f2p_patch:
        ok, err = await _apply_patch(sb, workdir, _F2P_PATCH, plan.f2p_patch)
        if not ok:
            return EvalResult(
                False,
                True,
                {"reason": "f2p_patch_failed", "mode": plan.mode.value, "apply_stderr": err},
            )
    if plan.f2p_script:
        await sb.write_file(f"{workdir}/{_F2P_SCRIPT}", plan.f2p_script, user="agent")

    applied_cleanly, apply_err = await _apply_patch(sb, workdir, _MODEL_PATCH, diff_text)
    if not applied_cleanly:
        return EvalResult(
            False,
            False,
            {"reason": "model_patch_failed", "mode": plan.mode.value, "apply_stderr": apply_err},
        )

    if plan.mode == EvalMode.SIMPLE_CMD:
        return await simple_cmd.evaluate(
            sb,
            workdir=workdir,
            eval_cmd=plan.eval_cmd,
            diff_text="",  # already applied
            timeout_sec=timeout_sec,
        )

    script = "/tmp/slime_eval_run.sh"
    await sb.write_file(script, "set +e\n" + plan.eval_cmd, user="agent")
    ec, stdout, stderr = await sb.exec(
        f"chmod 755 {script} && bash {script}",
        user="agent",
        check=False,
        timeout=timeout_sec,
    )

    if plan.mode == EvalMode.SCALESWE:
        grade = scaleswe_mod.grade_logs(
            fail_to_pass=plan.fail_to_pass,
            pass_to_pass=plan.pass_to_pass,
            stdout=stdout,
            stderr=stderr,
        )
    elif plan.mode == EvalMode.REBENCH:
        grade = rebench_mod.grade_logs(
            fail_to_pass=plan.fail_to_pass,
            pass_to_pass=plan.pass_to_pass,
            stdout=stdout,
            stderr=stderr,
            log_parser=plan.log_parser,
        )
    else:
        grade = swebench_mod.grade_logs(
            repo=plan.repo,
            fail_to_pass=plan.fail_to_pass,
            pass_to_pass=plan.pass_to_pass,
            stdout=stdout,
            stderr=stderr,
        )

    details = {
        "mode": plan.mode.value,
        "exit_code": ec,
        "stdout": (stdout or "")[-8000:],
        "stderr": (stderr or "")[-4000:],
        **grade,
    }
    return EvalResult(
        resolved=bool(grade.get("resolved")),
        applied_cleanly=True,
        details=details,
    )
