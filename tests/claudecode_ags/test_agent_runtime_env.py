import asyncio
from unittest.mock import patch

from slime.agent.harness import common as harness_common

from examples.claudecode_ags.agent_runtime import prepare_workspace, run_claude
from tests.claudecode_ags.fake_sandbox import FakeSandbox


async def _fast_sleep(_sec: float) -> None:
    return None


def test_prepare_workspace_writes_problem_statement():
    sb = FakeSandbox()
    asyncio.run(prepare_workspace(sb, workdir="/workspace/repo", problem_statement="fix the bug"))
    assert sb.files["/workspace/repo/PROBLEM_STATEMENT.md"] == "fix the bug"


def test_run_claude_passes_only_provided_env():
    sb = FakeSandbox()
    env = {
        "ANTHROPIC_BASE_URL": "http://adapter:8080",
        "ANTHROPIC_AUTH_TOKEN": "sess-1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192",
    }

    async def run_case() -> None:
        result = await run_claude(
            sb,
            workdir="/workspace/repo",
            prompt="solve it",
            env=env,
            time_budget_sec=30,
        )
        assert result["exit_code"] == 0
        launch_envs = [
            call_env
            for cmd, call_env in zip(sb.cmds, sb.exec_envs, strict=True)
            if "setsid" in cmd and call_env is not None
        ]
        assert launch_envs == [env]
        assert not any(k.startswith("SWE_") for call_env in sb.exec_envs if call_env for k in call_env)
        assert not any(k.startswith("VERL_") for call_env in sb.exec_envs if call_env for k in call_env)

    with patch.object(harness_common.asyncio, "sleep", new=_fast_sleep):
        asyncio.run(run_case())
