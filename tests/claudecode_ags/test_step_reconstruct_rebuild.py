import asyncio
import json
import os
import stat
import subprocess
import sys
from unittest.mock import AsyncMock, call, patch

from examples.claudecode_ags.claude_stream_input import initial_prompt_jsonl
from examples.claudecode_ags.step_reconstruct.session_capture import (
    SessionBundle,
    _WORKSPACE_METADATA_SCRIPT,
    steps_from_diff_files,
)
from examples.claudecode_ags.step_reconstruct.transcript import (
    build_transcript_prefix,
    pre_tool_snapshot_index,
    validate_transcript,
)
from examples.claudecode_ags.step_reconstruct.workspace_rebuild import (
    apply_diff,
    restore_cumulative_diff,
    verify_rebuild,
)


class _FakeSB:
    def __init__(self):
        self.files = {}
        self.applied = False
        self._diff_out = ""
        self.cmds = []

    async def write_file(self, path, content, user="agent"):
        self.files[path] = content

    async def exec(self, cmd, user="agent", timeout=60, check=False):
        self.cmds.append(cmd)
        if "git apply" in cmd:
            self.applied = True
            return 0, "", ""
        if "git diff" in cmd:
            return 0, self._diff_out, ""
        return 0, "", ""


def test_workspace_metadata_script_restores_exact_mtime_and_mode(tmp_path):
    workdir = tmp_path / "repo"
    workdir.mkdir()
    target = workdir / "source.py"
    target.write_text("original\n")
    expected_mtime = 1_700_000_000_123_456_789
    os.chmod(target, 0o640)
    os.utime(target, ns=(expected_mtime, expected_mtime))

    script = tmp_path / "metadata.py"
    script.write_text(_WORKSPACE_METADATA_SCRIPT)
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"tool_input": {"file_path": str(target)}}))
    tracked = tmp_path / "tracked.json"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, str(script), "capture", str(workdir), str(manifest), str(payload), str(tracked)],
        check=True,
    )

    target.write_text("wrong content\n")
    failed = subprocess.run(
        [sys.executable, str(script), "restore", str(workdir), str(manifest)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "content:source.py" in failed.stderr

    # Simulate the real order: git diff restores content first, metadata is
    # applied afterwards.
    target.write_text("original\n")
    os.chmod(target, 0o600)
    os.utime(target, ns=(expected_mtime + 99, expected_mtime + 99))
    subprocess.run(
        [sys.executable, str(script), "restore", str(workdir), str(manifest)],
        check=True,
    )
    value = target.stat()
    assert stat.S_IMODE(value.st_mode) == 0o640
    assert value.st_mtime_ns == expected_mtime


def test_workspace_metadata_ignores_internal_harness_paths(tmp_path):
    workdir = tmp_path / "repo"
    harness = workdir / ".harness"
    harness.mkdir(parents=True)
    (workdir / "PROBLEM_STATEMENT.md").write_text("bug\n")
    (harness / "trajectory.jsonl").write_text("{}\n")
    script = tmp_path / "metadata.py"
    script.write_text(_WORKSPACE_METADATA_SCRIPT)
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"tool_input": {"file_path": str(harness / "trajectory.jsonl")}})
    )
    manifest = tmp_path / "manifest.json"
    tracked = tmp_path / "tracked.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "capture",
            str(workdir),
            str(manifest),
            str(payload),
            str(tracked),
        ],
        check=True,
    )
    data = json.loads(manifest.read_text())
    assert data["unsupported_paths"] == []
    assert all(not row["path"].startswith(".harness") for row in data["records"])


def test_workspace_metadata_allows_external_read_paths_as_audit_only(tmp_path):
    workdir = tmp_path / "repo"
    workdir.mkdir()
    (workdir / "PROBLEM_STATEMENT.md").write_text("bug\n")
    external = tmp_path / "outside.txt"
    external.write_text("outside\n")
    script = tmp_path / "metadata.py"
    script.write_text(_WORKSPACE_METADATA_SCRIPT)
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"tool_name": "Read", "tool_input": {"file_path": str(external)}})
    )
    manifest = tmp_path / "manifest.json"
    tracked = tmp_path / "tracked.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "capture",
            str(workdir),
            str(manifest),
            str(payload),
            str(tracked),
        ],
        check=True,
    )

    data = json.loads(manifest.read_text())
    assert data["unsupported_paths"] == []
    assert data["external_read_paths"] == [str(external)]


def test_workspace_metadata_blocks_external_write_paths(tmp_path):
    workdir = tmp_path / "repo"
    workdir.mkdir()
    (workdir / "PROBLEM_STATEMENT.md").write_text("bug\n")
    external = tmp_path / "outside.txt"
    script = tmp_path / "metadata.py"
    script.write_text(_WORKSPACE_METADATA_SCRIPT)
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(external)}})
    )
    manifest = tmp_path / "manifest.json"
    tracked = tmp_path / "tracked.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "capture",
            str(workdir),
            str(manifest),
            str(payload),
            str(tracked),
        ],
        check=True,
    )

    data = json.loads(manifest.read_text())
    assert data["unsupported_paths"] == [str(external)]
    assert data["external_read_paths"] == []


def test_apply_diff_empty_ok():
    sb = _FakeSB()
    assert asyncio.run(apply_diff(sb, "/testbed", "")) is True


def test_apply_diff_writes_and_runs():
    sb = _FakeSB()
    assert asyncio.run(apply_diff(sb, "/testbed", "diff --git a/a b/a\n+x\n")) is True
    assert sb.applied
    assert "/tmp/_step_reconstruct_apply.diff" in sb.files


def test_apply_diff_reverse_is_explicit():
    sb = _FakeSB()
    assert asyncio.run(
        apply_diff(sb, "/testbed", "diff --git a/a b/a\n+x\n", reverse=True)
    ) is True
    assert "--reverse" in sb.cmds[-1]


def test_restore_initial_state_uses_initial_diff_as_fingerprint_only():
    sb = _FakeSB()
    initial = "diff --git a/base b/base\n+x\n"
    with patch(
        "examples.claudecode_ags.step_reconstruct.workspace_rebuild._common.workspace_diff",
        new=AsyncMock(return_value=initial),
    ), patch(
        "examples.claudecode_ags.step_reconstruct.workspace_rebuild.apply_diff",
        new=AsyncMock(),
    ) as apply:
        ok = asyncio.run(
            restore_cumulative_diff(
                sb,
                workdir="/testbed",
                initial_diff=initial,
                target_diff=initial,
            )
        )
    assert ok is True
    apply.assert_not_awaited()


def test_restore_nonempty_baseline_reverses_then_applies_cumulative_target():
    sb = _FakeSB()
    initial = "diff --git a/base b/base\n+x\n"
    target = "diff --git a/base b/base\n+x\ndiff --git a/edit b/edit\n+y\n"
    with patch(
        "examples.claudecode_ags.step_reconstruct.workspace_rebuild._common.workspace_diff",
        new=AsyncMock(side_effect=[initial, "", target]),
    ), patch(
        "examples.claudecode_ags.step_reconstruct.workspace_rebuild.apply_diff",
        new=AsyncMock(return_value=True),
    ) as apply:
        ok = asyncio.run(
            restore_cumulative_diff(
                sb,
                workdir="/testbed",
                initial_diff=initial,
                target_diff=target,
            )
        )
    assert ok is True
    assert apply.await_args_list == [
        call(sb, "/testbed", initial, reverse=True),
        call(sb, "/testbed", target),
    ]


def test_verify_rebuild_path_match(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    diff = "diff --git a/foo.py b/foo.py\n+++ b/foo.py\n+x\n"
    steps = steps_from_diff_files(str(d), [diff])
    b = SessionBundle(
        instance_id="i",
        session_id="s",
        cc_session_id="",
        task_metadata={"workdir": "/testbed"},
        steps=steps,
        dir=str(d),
    )
    b.save(str(d))
    bundle = SessionBundle.load(str(d))
    sb = _FakeSB()
    sb._diff_out = diff
    ok, details = asyncio.run(verify_rebuild(sb, bundle, 0))
    assert ok
    assert details["paths_match"]


def _tool_events(*tool_ids):
    events = []
    for tool_id in tool_ids:
        events.extend(
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_id}],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_id}],
                    },
                },
            ]
        )
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_build_transcript_prefix_by_tool_id():
    text = _tool_events("toolu_1", "toolu_2")
    out = build_transcript_prefix(
        text,
        ["toolu_1", "toolu_2"],
        0,
        initial_prompt="fix",
    )
    assert "toolu_1" in out
    assert "toolu_2" not in out
    assert '"text":"fix"' in out


def test_logical_minus_one_contains_only_initial_user_prompt():
    text = _tool_events("toolu_edit")
    out = build_transcript_prefix(text, ["toolu_edit"], -1, initial_prompt="fix")
    assert out == initial_prompt_jsonl("fix")
    events = [json.loads(line) for line in out.splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "user"
    assert events[0]["message"]["content"][0]["text"] == "fix"


def test_parallel_tools_branch_before_whole_assistant_turn():
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_a"},
                    {"type": "tool_use", "id": "toolu_b"},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a"},
                    {"type": "tool_result", "tool_use_id": "toolu_b"},
                ]
            },
        },
    ]
    text = "\n".join(json.dumps(event) for event in events) + "\n"
    assert validate_transcript(text, ["toolu_a", "toolu_b"]).valid
    assert pre_tool_snapshot_index(text, ["toolu_a", "toolu_b"], 1) == -1


def test_split_parallel_stream_rows_are_one_logical_turn():
    events = [
        {
            "type": "assistant",
            "message": {
                "id": "msg_parallel",
                "content": [{"type": "thinking", "thinking": "inspect"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg_parallel",
                "content": [{"type": "tool_use", "id": "toolu_a"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg_parallel",
                "content": [{"type": "tool_use", "id": "toolu_b"}],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "toolu_b"}],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "toolu_a"}],
            },
        },
    ]
    text = "\n".join(json.dumps(event) for event in events) + "\n"
    status = validate_transcript(text, ["toolu_b", "toolu_a"])
    assert status.valid
    assert status.conversation_event_count == 2
    assert pre_tool_snapshot_index(text, ["toolu_b", "toolu_a"], 1) == -1


def test_invalid_transcript_reports_missing_result():
    text = json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "toolu_a"}]},
        }
    )
    status = validate_transcript(text, ["toolu_a"])
    assert status.valid is False
    assert "missing_tool_result" in status.error


def test_transcript_tool_without_snapshot_fails_closed():
    status = validate_transcript(_tool_events("toolu_a", "toolu_b"), ["toolu_a"])
    assert status.valid is False
    assert "snapshot_missing_for_transcript_tool" in status.error


def test_nonadjacent_tool_result_fails_closed():
    events = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "toolu_a"}]},
        },
        {"type": "user", "message": {"content": [{"type": "text", "text": "interleaved"}]}},
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_a"}]},
        },
    ]
    text = "\n".join(json.dumps(event) for event in events) + "\n"
    status = validate_transcript(text, ["toolu_a"])
    assert status.valid is False
    assert "tool_result_not_adjacent" in status.error


def test_prefix_preserves_later_user_text_but_not_initial_echo():
    events = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "fix"}]}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "toolu_a"}]},
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_a"}]},
        },
        {"type": "user", "message": {"content": [{"type": "text", "text": "continue"}]}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "toolu_b"}]},
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_b"}]},
        },
    ]
    text = "\n".join(json.dumps(event) for event in events) + "\n"
    prefix = build_transcript_prefix(
        text,
        ["toolu_a", "toolu_b"],
        1,
        initial_prompt="fix",
    )
    assert prefix.count('"text":"fix"') == 1
    assert '"text":"continue"' in prefix
