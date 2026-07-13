"""Tests for workspace_init phase 1 (swebench / swesmith / scrub)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from examples.claudecode_ags.agent_runtime import prepare_workspace
from examples.claudecode_ags.workspace_init import (
    TaskFields,
    WorkspaceMode,
    detect_workspace_mode,
    initialize_task_workspace,
    reverse_patch,
    task_fields_from_metadata,
)
from tests.claudecode_ags.fake_sandbox import FakeSandbox


def test_detect_swesmith():
    f = TaskFields(data_source="swe_smith_foo", instance_id="repo__issue")
    assert detect_workspace_mode(f) == WorkspaceMode.SWESMITH


def test_detect_swebench_classic_by_base_commit():
    f = TaskFields(base_commit="abc123")
    assert detect_workspace_mode(f) == WorkspaceMode.SWEBENCH_CLASSIC


def test_detect_swebench_from_metadata_swebench_dict():
    fields = task_fields_from_metadata({"swebench": {"base_commit": "deadbeef"}, "workdir": "/testbed"})
    assert fields.base_commit == "deadbeef"
    assert detect_workspace_mode(fields) == WorkspaceMode.SWEBENCH_CLASSIC


def test_detect_generic():
    assert detect_workspace_mode(TaskFields()) == WorkspaceMode.GENERIC


def test_detect_scaleswe_by_pre_commands():
    f = TaskFields(pre_commands="git checkout main")
    assert detect_workspace_mode(f) == WorkspaceMode.SCALESWE


def test_detect_rebench_by_test_cmd():
    f = TaskFields(install_config={"test_cmd": ["pytest -q"]})
    assert detect_workspace_mode(f) == WorkspaceMode.REBENCH


def test_detect_priority_pre_commands_before_base_commit():
    f = TaskFields(pre_commands="echo hi", base_commit="abc")
    assert detect_workspace_mode(f) == WorkspaceMode.SCALESWE


def test_swesmith_wins_over_base_commit():
    f = TaskFields(data_source="swe_smith_x", base_commit="abc")
    assert detect_workspace_mode(f) == WorkspaceMode.SWESMITH


def test_reverse_patch_swaps_plus_minus():
    patch = "--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    rev = reverse_patch(patch)
    assert "-new\n" in rev
    assert "+old\n" in rev


def test_swebench_reset_and_scrub():
    sb = FakeSandbox()
    written: list[str] = []
    orig_write = sb.write_file

    async def tracking_write(path: str, content, *, user: str = "root") -> None:
        written.append(content if isinstance(content, str) else content.decode())
        await orig_write(path, content, user=user)

    sb.write_file = tracking_write  # type: ignore[method-assign]

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(instance_id="django__1", base_commit="abc123", workdir="/testbed"),
            )

    assert asyncio.run(run()) is True
    bodies = "\n".join(written)
    assert "git reset --hard" in bodies
    assert "abc123" in bodies
    assert "slime_git_scrub" in bodies
    assert sum(1 for c in sb.cmds if "bash /tmp/slime_ws_init.sh" in c) == 2


def test_swebench_reset_failure():
    sb = FakeSandbox()

    async def failing_exec(cmd, **kwargs):
        sb.cmds.append(cmd)
        if "bash /tmp/slime_ws_init.sh" in cmd:
            return 1, "", "reset failed"
        return 0, "", ""

    sb.exec = failing_exec  # type: ignore[method-assign]

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(base_commit="bad", workdir="/testbed"),
            )

    assert asyncio.run(run()) is False


def test_swesmith_branch_script():
    sb = FakeSandbox()
    written: list[str] = []
    orig_write = sb.write_file

    async def tracking_write(path: str, content, *, user: str = "root") -> None:
        written.append(content if isinstance(content, str) else content.decode())
        await orig_write(path, content, user=user)

    sb.write_file = tracking_write  # type: ignore[method-assign]

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(
                    data_source="swe_smith",
                    instance_id="repo__123",
                    workdir="/testbed",
                ),
            )

    assert asyncio.run(run()) is True
    bodies = "\n".join(written)
    assert "git checkout" in bodies
    assert "repo__123" in bodies
    assert "slime_git_scrub" in bodies


def test_swesmith_synthetic_needs_patch():
    sb = FakeSandbox()

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(
                    data_source="swe_smith",
                    instance_id="_synthetic_row_1",
                    workdir="/testbed",
                ),
            )

    assert asyncio.run(run()) is False


def test_swesmith_patch_path():
    sb = FakeSandbox()
    patch_text = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"

    async def run() -> bool:
        with (
            patch(
                "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
                new=AsyncMock(),
            ),
            patch(
                "examples.claudecode_ags.workspace_init._diff_applies_cleanly",
                new=AsyncMock(side_effect=[True]),
            ),
            patch(
                "examples.claudecode_ags.workspace_init._apply_diff",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "examples.claudecode_ags.workspace_init._commit_bug_baseline",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "examples.claudecode_ags.workspace_init.apply_swesmith_branch",
                new=AsyncMock(return_value=False),
            ),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(
                    data_source="swe_smith",
                    instance_id="repo__1",
                    workdir="/testbed",
                    swe_smith_bug_patch=patch_text,
                ),
            )

    assert asyncio.run(run()) is True


def test_generic_no_scrub_required():
    sb = FakeSandbox()

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(sb, TaskFields(workdir="/testbed"))

    assert asyncio.run(run()) is True
    assert not any("slime_git_scrub" in v for v in sb.files.values())


def test_prepare_workspace_writes_problem_after_init():
    sb = FakeSandbox()

    async def run() -> None:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            await prepare_workspace(
                sb,
                workdir="/workspace/repo",
                problem_statement="fix the bug",
                base_commit="abc",
                instance_id="x",
            )

    asyncio.run(run())
    assert sb.files["/workspace/repo/PROBLEM_STATEMENT.md"] == "fix the bug"


def test_prepare_workspace_raises_on_init_failure():
    sb = FakeSandbox()

    async def failing_exec(cmd, **kwargs):
        sb.cmds.append(cmd)
        if "bash /tmp/slime_ws_init.sh" in cmd:
            return 1, "", "fail"
        return 0, "", ""

    sb.exec = failing_exec  # type: ignore[method-assign]

    async def run() -> None:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            await prepare_workspace(
                sb,
                workdir="/testbed",
                problem_statement="x",
                base_commit="bad",
            )

    with pytest.raises(RuntimeError, match="workspace init failed"):
        asyncio.run(run())


def test_scaleswe_runs_pre_commands_and_scrub():
    sb = FakeSandbox()
    written: list[str] = []
    orig_write = sb.write_file

    async def tracking_write(path: str, content, *, user: str = "root") -> None:
        written.append(content if isinstance(content, str) else content.decode())
        await orig_write(path, content, user=user)

    sb.write_file = tracking_write  # type: ignore[method-assign]

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(pre_commands="echo prepare", workdir="/testbed", instance_id="s1"),
                rollout_side=True,
            )

    assert asyncio.run(run()) is True
    bodies = "\n".join(written)
    assert "echo prepare" in bodies
    assert "slime_git_scrub" in bodies
    assert "orphan" in bodies or "__slime_buggy" in bodies


def test_eval_scrub_is_light():
    sb = FakeSandbox()
    written: list[str] = []
    orig_write = sb.write_file

    async def tracking_write(path: str, content, *, user: str = "root") -> None:
        written.append(content if isinstance(content, str) else content.decode())
        await orig_write(path, content, user=user)

    sb.write_file = tracking_write  # type: ignore[method-assign]

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(base_commit="abc", workdir="/testbed"),
                rollout_side=False,
            )

    assert asyncio.run(run()) is True
    bodies = "\n".join(written)
    assert "slime_git_scrub_eval" in bodies
    assert "__slime_buggy" not in bodies


def test_rebench_no_scrub():
    sb = FakeSandbox()
    written: list[str] = []
    orig_write = sb.write_file

    async def tracking_write(path: str, content, *, user: str = "root") -> None:
        written.append(content if isinstance(content, str) else content.decode())
        await orig_write(path, content, user=user)

    sb.write_file = tracking_write  # type: ignore[method-assign]

    async def run() -> bool:
        with patch(
            "examples.claudecode_ags.workspace_init.agent_sandbox.ensure_agent_user",
            new=AsyncMock(),
        ):
            return await initialize_task_workspace(
                sb,
                TaskFields(install_config={"test_cmd": "pytest"}, workdir="/testbed"),
            )

    assert asyncio.run(run()) is True
    assert not any("slime_git_scrub" in w for w in written)
