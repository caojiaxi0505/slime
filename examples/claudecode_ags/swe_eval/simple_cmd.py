"""Grade a patch by running a shell eval_cmd; exit 0 means resolved."""

from __future__ import annotations

from examples.claudecode_ags.swe_eval.base import EvalResult

_PATCH = "/tmp/slime_eval_patch.diff"


async def evaluate(
    sb,
    *,
    workdir: str,
    eval_cmd: str,
    diff_text: str = "",
    timeout_sec: int = 600,
) -> EvalResult:
    applied_cleanly = True
    if diff_text.strip():
        await sb.write_file(_PATCH, diff_text, user="agent")
        ec, _, _ = await sb.exec(
            f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
            user="agent",
            check=False,
            timeout=120,
        )
        applied_cleanly = ec == 0
        if not applied_cleanly:
            return EvalResult(False, False, {"exit_code": ec})

    ec, stdout, stderr = await sb.exec(
        f"cd {workdir} && {eval_cmd}",
        user="agent",
        check=False,
        timeout=timeout_sec,
        # Do not re-submit a long-running test command after a severed stream.
        # AGS routes non-idempotent execs through SWE-ReX runtime.execute.
        idempotent=False,
    )
    return EvalResult(
        resolved=ec == 0,
        applied_cleanly=applied_cleanly,
        details={"exit_code": ec, "stdout": stdout, "stderr": stderr},
    )
