"""Dispatch SWE eval: resolve plan → apply patches → run → grade."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from examples.claudecode_ags.swe_eval import rebench as rebench_mod
from examples.claudecode_ags.swe_eval import scaleswe as scaleswe_mod
from examples.claudecode_ags.swe_eval import simple_cmd
from examples.claudecode_ags.swe_eval import swebench as swebench_mod
from examples.claudecode_ags.swe_eval import swegym as swegym_mod
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
    supervisor = "/tmp/slime_eval_run_supervisor.sh"
    output_path = "/tmp/slime_eval_output.log"
    supervisor_output_path = "/tmp/slime_eval_supervisor.log"
    timeout_marker = "/tmp/slime_eval_timed_out"
    status_path = "/tmp/slime_eval_status"
    test_pid_path = "/tmp/slime_eval_test.pid"
    supervisor_pid_path = "/tmp/slime_eval_supervisor.pid"
    launcher = "/tmp/slime_eval_launch.sh"
    supervisor_body = f"""#!/bin/bash
set +e
rm -f {output_path} {timeout_marker} {status_path} {test_pid_path}
setsid bash {script} > {output_path} 2>&1 &
test_pid=$!
printf '%s\n' "$test_pid" > {test_pid_path}
(
  sleep {timeout_sec}
  if kill -0 "$test_pid" 2>/dev/null; then
    : > {timeout_marker}
    kill -TERM -- "-$test_pid" 2>/dev/null || true
    sleep 10
    kill -KILL -- "-$test_pid" 2>/dev/null || true
  fi
) &
watchdog_pid=$!
wait "$test_pid"
test_rc=$?
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
if [ -f {timeout_marker} ]; then
  test_rc=124
fi
status_tmp={status_path}.tmp.$$
printf '%s\n' "$test_rc" > "$status_tmp"
mv -f "$status_tmp" {status_path}
exit 0
"""
    await sb.write_file(supervisor, supervisor_body, user="agent")
    launcher_body = f"""#!/bin/bash
set -e
rm -f {status_path} {supervisor_pid_path} {supervisor_output_path}
chmod 755 {script} {supervisor}
nohup setsid bash {supervisor} > {supervisor_output_path} 2>&1 < /dev/null &
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" > {supervisor_pid_path}
"""
    await sb.write_file(launcher, launcher_body, user="agent")

    # Do not keep one AGS/SWE-ReX HTTP /execute request open for the whole test.
    # Some server builds fail to return that request after long test processes
    # exit. Start a fully redirected background supervisor, then use short HTTP
    # requests to poll an atomically-written status file and fetch the log.
    launch_ec, _, launch_stderr = await sb.exec(
        f"chmod 755 {launcher} && bash {launcher}",
        user="agent",
        check=False,
        timeout=60,
        idempotent=False,
    )
    if launch_ec != 0:
        raise RuntimeError(f"failed to launch background evaluator: {launch_stderr[:1000]}")

    poll_interval_sec = 15
    completion_grace_sec = 60
    deadline = asyncio.get_running_loop().time() + timeout_sec + completion_grace_sec
    ec: int | None = None
    last_poll_error = ""
    client_poll_timed_out = False
    while ec is None:
        try:
            poll_ec, poll_out, poll_err = await sb.exec(
                f"test -f {status_path} && cat {status_path}",
                user="agent",
                check=False,
                timeout=30,
            )
            if poll_ec == 0 and poll_out.strip():
                try:
                    ec = int(poll_out.strip().splitlines()[-1])
                except ValueError as exc:
                    raise RuntimeError(f"invalid evaluator status: {poll_out!r}") from exc
            elif poll_err:
                last_poll_error = poll_err[-1000:]
        except Exception as exc:
            last_poll_error = f"{type(exc).__name__}: {exc}"

        if ec is not None:
            break
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            client_poll_timed_out = True
            ec = 124
            try:
                await sb.exec(
                    f"test -s {test_pid_path} && "
                    f"kill -KILL -- -$(cat {test_pid_path}) 2>/dev/null || true; "
                    f"test -s {supervisor_pid_path} && "
                    f"kill -KILL -- -$(cat {supervisor_pid_path}) 2>/dev/null || true",
                    user="agent",
                    check=False,
                    timeout=30,
                )
            except Exception as exc:
                last_poll_error = f"cleanup {type(exc).__name__}: {exc}"
            break
        await asyncio.sleep(min(poll_interval_sec, remaining))

    log_ec, stdout, log_stderr = await sb.exec(
        f"cat {output_path}", user="agent", check=False, timeout=180
    )
    if log_ec != 0:
        raise RuntimeError(f"failed to fetch evaluator output: {log_stderr[:1000]}")
    _, supervisor_stderr, _ = await sb.exec(
        f"test -f {supervisor_output_path} && cat {supervisor_output_path}",
        user="agent",
        check=False,
        timeout=60,
    )
    stderr = supervisor_stderr or last_poll_error
    command_timed_out = ec in {124, 137}

    if command_timed_out:
        # A forcibly terminated official test log is allowed to lack the
        # harness boundary markers. It cannot be resolved, so do not turn a
        # known test timeout into a parser/infrastructure error.
        grade = {
            "resolved": False,
            "resolution_status": "TIMEOUT",
            "parser_skipped_reason": "command_timeout",
        }
    elif plan.mode == EvalMode.SCALESWE:
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
    elif plan.mode == EvalMode.SWEGYM:
        grade = swegym_mod.grade_logs(
            repo=plan.repo,
            fail_to_pass=plan.fail_to_pass,
            pass_to_pass=plan.pass_to_pass,
            stdout=stdout,
            stderr=stderr,
        )
    else:
        grade = swebench_mod.grade_logs(
            repo=plan.repo,
            fail_to_pass=plan.fail_to_pass,
            pass_to_pass=plan.pass_to_pass,
            stdout=stdout,
            stderr=stderr,
            test_spec=plan.official_test_spec,
        )

    details = {
        "mode": plan.mode.value,
        "exit_code": ec,
        "command_timed_out": command_timed_out,
        "client_poll_timed_out": client_poll_timed_out,
        "test_timeout_sec": timeout_sec,
        "eval_execution_protocol": "background_poll_v1",
        "eval_poll_interval_sec": poll_interval_sec,
        "eval_completion_grace_sec": completion_grace_sec,
        "stdout": (stdout or "")[-8000:],
        "stderr": (stderr or "")[-4000:],
        **grade,
    }
    return EvalResult(
        resolved=bool(grade.get("resolved")),
        applied_cleanly=True,
        details=details,
    )
