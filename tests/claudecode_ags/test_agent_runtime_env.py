import asyncio
import json
from unittest.mock import patch

from slime.agent.harness import common as harness_common

from examples.claudecode_ags.agent_runtime import (
    git_diff,
    prepare_workspace,
    run_claude,
    run_claude_native_resume,
)
from tests.claudecode_ags.fake_sandbox import FakeSandbox


async def _fast_sleep(_sec: float) -> None:
    return None


def test_prepare_workspace_writes_problem_statement():
    sb = FakeSandbox()
    asyncio.run(prepare_workspace(sb, workdir="/workspace/repo", problem_statement="fix the bug"))
    assert sb.files["/workspace/repo/PROBLEM_STATEMENT.md"] == "fix the bug"


def test_git_diff_captures_staged_unstaged_and_binary_changes():
    sb = FakeSandbox()
    asyncio.run(git_diff(sb, workdir="/workspace/repo"))

    assert sb.cmds == [
        "cd /workspace/repo && git add -N . && "
        "git diff HEAD --binary -- . "
        "':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/**'"
    ]


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
        initial_input = sb.files["/tmp/slime_cc_initial_prompt.jsonl"]
        assert json.loads(initial_input) == {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "solve it"}],
            },
        }
        launcher = sb.files["/tmp/.run.sh"]
        assert launcher.index("conda activate testbed") < launcher.index("/usr/local/bin/claude")
        assert "/nix/swerex/venv/bin/python" in launcher
        assert "--input-format stream-json" in launcher
        assert "< /tmp/slime_cc_initial_prompt.jsonl" in launcher
        assert "solve it" not in launcher

    with patch.object(harness_common.asyncio, "sleep", new=_fast_sleep):
        asyncio.run(run_case())


def test_run_claude_supports_explicit_positional_reference_context(monkeypatch):
    sb = FakeSandbox()
    prompt = (
        "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
        "When finished, print a one-line summary and exit."
    )
    extra_args = [
        "--settings",
        '{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true}',
        "--disable-slash-commands",
        "--disallowedTools",
        "WebFetch",
        "WebSearch",
    ]
    monkeypatch.setenv("SLIME_CC_INITIAL_INPUT_MODE", "positional")
    monkeypatch.setenv("SLIME_CC_EXTRA_ARGS_JSON", json.dumps(extra_args))

    async def run_case() -> None:
        result = await run_claude(
            sb,
            workdir="/workspace/repo",
            prompt=prompt,
            env={"ANTHROPIC_AUTH_TOKEN": "sess-1"},
            time_budget_sec=30,
        )
        assert result["exit_code"] == 0
        launcher = sb.files["/tmp/.run.sh"]
        assert launcher.index("conda activate testbed") < launcher.index("/usr/local/bin/claude")
        assert "--input-format stream-json" not in launcher
        assert "/tmp/slime_cc_initial_prompt.jsonl" not in launcher
        assert prompt in launcher
        assert "--disable-slash-commands" in launcher
        assert "--disallowedTools WebFetch WebSearch" in launcher

    with patch.object(harness_common.asyncio, "sleep", new=_fast_sleep):
        asyncio.run(run_case())


def test_native_resume_activates_testbed_environment():
    sb = FakeSandbox()

    async def run_case() -> None:
        result = await run_claude_native_resume(
            sb,
            workdir="/workspace/repo",
            env={"ANTHROPIC_AUTH_TOKEN": "sess-1"},
            time_budget_sec=30,
            session_jsonl='{"type":"assistant","uuid":"terminal"}\n',
        )
        assert result["exit_code"] == 0
        launcher = sb.files["/tmp/.run.sh"]
        assert launcher.index("conda activate testbed") < launcher.index("/usr/local/bin/claude")
        assert "--resume /tmp/slime_cc_branch.session.jsonl" in launcher
        assert "--fork-session" in launcher

    with patch.object(harness_common.asyncio, "sleep", new=_fast_sleep):
        asyncio.run(run_case())
