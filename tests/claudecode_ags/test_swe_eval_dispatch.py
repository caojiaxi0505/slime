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
    assert sb.exec_idempotent[-1] is False


def test_dispatch_scaleswe_writes_f2p_script_and_runs():
    sb = FakeSandbox()
    f2p = "test_fail_to_pass.py::test_x"
    pass_log = f"PASSED {f2p}\n"

    orig_exec = sb.exec

    async def exec_with_log(cmd: str, **kwargs):
        ec, out, err = await orig_exec(cmd, **kwargs)
        if "slime_eval_launch.sh" in cmd:
            sb.files["/tmp/slime_eval_status"] = "0\n"
            sb.files["/tmp/slime_eval_output.log"] = pass_log
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
    assert "nohup setsid bash /tmp/slime_eval_run_supervisor.sh" in sb.files[
        "/tmp/slime_eval_launch.sh"
    ]
    launch_call = next(i for i, cmd in enumerate(sb.cmds) if "slime_eval_launch.sh" in cmd)
    assert sb.exec_idempotent[launch_call] is False
    assert any("test -f /tmp/slime_eval_status" in c for c in sb.cmds)


def test_dispatch_background_timeout_is_unresolved():
    sb = FakeSandbox()
    f2p = "test_fail_to_pass.py::test_x"
    orig_exec = sb.exec

    async def exec_with_timeout(cmd: str, **kwargs):
        ec, out, err = await orig_exec(cmd, **kwargs)
        if "slime_eval_launch.sh" in cmd:
            sb.files["/tmp/slime_eval_status"] = "124\n"
            sb.files["/tmp/slime_eval_output.log"] = f"PASSED {f2p}\n"
        return ec, out, err

    sb.exec = exec_with_timeout  # type: ignore[method-assign]

    async def run():
        return await evaluate(
            sb,
            metadata={
                "f2p_script": "def test_x(): assert True\n",
                "FAIL_TO_PASS": [f2p],
                "PASS_TO_PASS": [],
                "workdir": "/testbed",
            },
            diff_text="",
            timeout_sec=600,
        )

    result = asyncio.run(run())
    supervisor = sb.files["/tmp/slime_eval_run_supervisor.sh"]
    assert "setsid bash /tmp/slime_eval_run.sh" in supervisor
    assert 'kill -TERM -- "-$test_pid"' in supervisor
    assert 'kill -KILL -- "-$test_pid"' in supervisor
    assert "sleep 600" in supervisor
    assert "nohup setsid bash /tmp/slime_eval_run_supervisor.sh" in sb.files[
        "/tmp/slime_eval_launch.sh"
    ]
    assert any("test -f /tmp/slime_eval_status" in c for c in sb.cmds)
    assert result.resolved is False
    assert result.details["command_timed_out"] is True
    assert result.details["client_poll_timed_out"] is False
    assert result.details["eval_execution_protocol"] == "background_poll_v1"
    assert result.details["exit_code"] == 124


def test_dispatch_model_patch_failure():
    sb = FakeSandbox()

    async def fail_apply(cmd: str, **kwargs):
        sb.cmds.append(cmd)
        if "git apply" in cmd or "patch -p1" in cmd:
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
