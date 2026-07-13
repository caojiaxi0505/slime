"""dispatch.evaluate with FakeSandbox."""

from __future__ import annotations

import asyncio

from examples.claudecode_ags.swe_eval.dispatch import evaluate
from tests.claudecode_ags.fake_sandbox import FakeSandbox


def test_dispatch_missing_plan():
    sb = FakeSandbox()

    async def run():
        return await evaluate(sb, metadata={}, diff_text="")

    result = asyncio.run(run())
    assert result.resolved is False
    assert result.details.get("reason") == "missing_eval_plan"


def test_dispatch_simple_cmd():
    sb = FakeSandbox()

    async def run():
        return await evaluate(
            sb,
            metadata={"eval_cmd": "true", "workdir": "/testbed"},
            diff_text="",
        )

    result = asyncio.run(run())
    assert result.resolved is True
    assert any("true" in c for c in sb.cmds)


def test_dispatch_scaleswe_writes_f2p_script_and_runs():
    sb = FakeSandbox()
    f2p = "test_fail_to_pass.py::test_x"
    pass_log = f"PASSED {f2p}\n"

    orig_exec = sb.exec

    async def exec_with_log(cmd: str, **kwargs):
        ec, out, err = await orig_exec(cmd, **kwargs)
        if "slime_eval_run.sh" in cmd:
            return 0, pass_log, ""
        return ec, out, err

    sb.exec = exec_with_log  # type: ignore[method-assign]

    async def run():
        return await evaluate(
            sb,
            metadata={
                "pre_commands": "echo prep",
                "f2p_script": "def test_x(): assert True\n",
                "FAIL_TO_PASS": [f2p],
                "PASS_TO_PASS": [],
                "workdir": "/testbed",
            },
            diff_text="",
        )

    result = asyncio.run(run())
    assert "/testbed/test_fail_to_pass.py" in sb.files
    assert result.details.get("mode") == "scaleswe"
    assert result.resolved is True
    assert any("slime_eval_run.sh" in c for c in sb.cmds)


def test_dispatch_model_patch_failure():
    sb = FakeSandbox()

    async def fail_apply(cmd: str, **kwargs):
        sb.cmds.append(cmd)
        if "git apply" in cmd:
            return 1, "", "reject"
        return 0, "", ""

    sb.exec = fail_apply  # type: ignore[method-assign]

    async def run():
        return await evaluate(
            sb,
            metadata={"eval_cmd": "true", "workdir": "/testbed"},
            diff_text="diff --git a/x b/x\n",
        )

    result = asyncio.run(run())
    assert result.applied_cleanly is False
    assert result.resolved is False
