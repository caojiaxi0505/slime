"""Mocked unit tests for live runners + capture helpers (no AGS)."""

from __future__ import annotations

import asyncio
import base64
import gzip
import io
import json
import os
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.claudecode_ags.step_reconstruct import live_runners
from examples.claudecode_ags.step_reconstruct.session_capture import (
    SessionBundle,
    build_hook_steps_from_snap_dir,
    capture_snapshots_to_bundle,
    install_snapshot_hook,
    pull_remote_dir,
    steps_from_diff_files,
)
from slime.agent.adapters.anthropic_segmented import canonical_sha256, prompt_ids_sha256
from slime.utils.types import Sample


class _FakeSB:
    def __init__(self, *, files=None, exec_map=None):
        self.files = dict(files or {})
        self.exec_map = dict(exec_map or {})
        self.cmds = []

    async def write_file(self, path, content, user="agent"):
        self.files[path] = content

    async def read_file(self, path, user="agent"):
        return self.files.get(path, "")

    async def exec(self, cmd, user="agent", timeout=60, check=False):
        self.cmds.append(cmd)
        for key, val in self.exec_map.items():
            if key in cmd:
                return val
        return 0, "", ""


def test_build_hook_steps_from_index(tmp_path):
    snap = tmp_path / ".cagent_snapshots"
    snap.mkdir()
    (snap / "step_0001.diff").write_text("diff --git a/a b/a\n+1\n")
    (snap / "step_0002.diff").write_text("diff --git a/a b/a\n+2\n")
    (snap / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"seq": 1, "ts": 1, "source": "hook", "tool_name": "Edit"}),
                json.dumps({"seq": 2, "ts": 2, "source": "hook", "tool_name": "Bash"}),
                json.dumps({"seq": 99, "ts": 3, "source": "manual", "tool_name": "x"}),
            ]
        )
        + "\n"
    )
    steps = build_hook_steps_from_snap_dir(str(snap), str(tmp_path))
    assert len(steps) == 2
    assert steps[0].tool_name == "Edit"
    assert steps[1].seq == 2


def test_build_hook_steps_fallback_sorted(tmp_path):
    snap = tmp_path / ".cagent_snapshots"
    snap.mkdir()
    (snap / "step_0002.diff").write_text("b")
    (snap / "step_0001.diff").write_text("a")
    steps = build_hook_steps_from_snap_dir(str(snap), str(tmp_path))
    assert [s.seq for s in steps] == [1, 2]
    assert steps[0].diff_file.endswith("step_0001.diff")


def test_pull_remote_dir_extracts(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"diff\n"
        info = tarfile.TarInfo(name=".cagent_snapshots/step_0001.diff")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    b64 = base64.b64encode(buf.getvalue()).decode()

    sb = _FakeSB(exec_map={"tar czf": (0, b64, "")})
    out = asyncio.run(pull_remote_dir(sb, "/home/agent/.cagent_snapshots", str(tmp_path)))
    assert out is not None
    assert (tmp_path / ".cagent_snapshots" / "step_0001.diff").is_file()


def test_capture_bundle_persists_real_transcript_and_initial_state(tmp_path):
    snap = tmp_path / ".cagent_snapshots"
    snap.mkdir()
    (snap / "step_0001.diff").write_text("diff --git a/a b/a\n+x\n")
    (snap / "step_0001.payload.json").write_text(json.dumps({"tool_use_id": "toolu_a"}))
    (snap / "index.jsonl").write_text(
        json.dumps({"seq": 1, "source": "hook", "tool_name": "Edit"}) + "\n"
    )
    transcript = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "toolu_a"}]},
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_a"}]
                    },
                }
            ),
        ]
    ) + "\n"
    sb = _FakeSB(files={"/testbed/.harness/trajectory.jsonl": transcript})

    async def fake_pull(*args, **kwargs):
        return str(snap)

    with patch(
        "examples.claudecode_ags.step_reconstruct.session_capture.pull_remote_dir",
        new=fake_pull,
    ):
        bundle = asyncio.run(
            capture_snapshots_to_bundle(
                sb,
                out_dir=str(tmp_path),
                workdir="/testbed",
                instance_id="inst",
                session_id="sid",
                task_metadata={"workdir": "/testbed"},
                initial_diff="",
                final_diff="diff --git a/a b/a\n+x\n",
                claude_exit_code=0,
            )
        )

    assert bundle.transcript_valid is True
    assert bundle.transcript_bytes == len(transcript.encode())
    assert bundle.transcript_tool_uses == 1
    assert bundle.transcript_tool_results == 1
    assert bundle.initial_state_captured is True
    assert bundle.initial_diff() == ""
    assert bundle.transcript() == transcript


def test_capture_bundle_persists_checkpoint_native_session_and_metadata(tmp_path):
    out_dir = tmp_path / "bundle"
    snap = tmp_path / "snap-source"
    snap.mkdir()
    (snap / "step_0001.diff").write_text("diff --git a/a b/a\n+x\n")
    (snap / "step_0001.payload.json").write_text(json.dumps({"tool_use_id": "toolu_a"}))
    (snap / "step_0001.metadata.json").write_text('{"version":1,"records":[]}')
    (snap / "initial.metadata.json").write_text('{"version":1,"records":[]}')
    (snap / "index.jsonl").write_text(
        json.dumps({"seq": 1, "source": "hook", "tool_name": "Edit"}) + "\n"
    )
    projects = tmp_path / "projects"
    native_project = projects / "-testbed"
    native_project.mkdir(parents=True)
    native_text = json.dumps({"type": "user", "uuid": "u", "sessionId": "cc-native"}) + "\n"
    (native_project / "cc-native.jsonl").write_text(native_text)
    transcript = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "toolu_a"}]},
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_a"}]},
                }
            ),
        ]
    ) + "\n"
    checkpoint = {
        "checkpoint_id": "main-0-checkpoint",
        "prompt_ids": [1, 2, 3],
        "prompt_sha256": prompt_ids_sha256([1, 2, 3]),
        "chat_messages": [{"role": "user", "content": "fix"}],
        "tools_schema": None,
        "tools_sha256": canonical_sha256(None),
        "generation_config": {},
        "tokenizer_fingerprint": {},
        "chain_kind": "main",
        "request_kind": "new",
        "request_index": 0,
        "source_tool_use_ids": [],
        "source_tool_result_ids": [],
        "generated_tool_use_ids": ["toolu_a"],
        "generated_tool_use_names": {"toolu_a": "Read"},
    }

    async def fake_pull(_sb, remote_dir, _local_parent):
        return str(projects if remote_dir.endswith("/projects") else snap)

    with patch(
        "examples.claudecode_ags.step_reconstruct.session_capture.pull_remote_dir",
        new=fake_pull,
    ):
        bundle = asyncio.run(
            capture_snapshots_to_bundle(
                _FakeSB(),
                out_dir=str(out_dir),
                workdir="/testbed",
                instance_id="inst",
                session_id="adapter-session",
                task_metadata={"workdir": "/testbed"},
                initial_diff="",
                final_diff="diff --git a/a b/a\n+x\n",
                transcript_text=transcript,
                claude_exit_code=0,
                cc_session_id="cc-native",
                prompt_checkpoints=[checkpoint],
            )
        )

    assert bundle.prompt_checkpoints_valid is True
    assert bundle.native_session_valid is True
    assert bundle.workspace_metadata_valid is True
    assert bundle.checkpoint_for_tool_use_id("toolu_a")["checkpoint_id"] == "main-0-checkpoint"
    assert bundle.native_session() == native_text


def test_snapshot_hook_captures_success_and_failure_boundaries():
    sb = _FakeSB()
    asyncio.run(install_snapshot_hook(sb, "/testbed"))
    settings = json.loads(sb.files["/home/agent/.claude/settings.json"])
    assert "PostToolUse" in settings["hooks"]
    assert "PostToolUseFailure" in settings["hooks"]


def test_run_claude_with_prefix_cmd():
    from examples.claudecode_ags import agent_runtime

    seen = {}

    async def fake_run_agent(sb, *, workdir, start_cmd, env, time_budget_sec):
        seen["cmd"] = start_cmd
        return 0

    sb = _FakeSB(files={"/testbed/.harness/trajectory.jsonl": '{"type":"result"}\n'})

    with patch.object(agent_runtime, "run_agent", new=fake_run_agent):
        ec = asyncio.run(
            agent_runtime.run_claude_with_prefix(
                sb,
                workdir="/testbed",
                env={"A": "1"},
                time_budget_sec=10,
                prefix_path="/tmp/cc_prefix.jsonl",
            )
        )
    assert ec["exit_code"] == 0
    assert "--input-format stream-json" in seen["cmd"]
    assert "/tmp/cc_prefix.jsonl" in seen["cmd"]
    assert "fix me" not in seen["cmd"]
    assert ec["trajectory_jsonl"] == '{"type":"result"}\n'


def test_run_claude_native_resume_uses_documented_resume_and_fork_flags():
    from examples.claudecode_ags import agent_runtime

    seen = {}

    async def fake_run_agent(sb, *, workdir, start_cmd, env, time_budget_sec):
        seen["cmd"] = start_cmd
        return 0

    sb = _FakeSB(files={"/testbed/.harness/trajectory.jsonl": '{"type":"result"}\n'})
    native = json.dumps(
        {
            "type": "assistant",
            "uuid": "terminal",
            "sessionId": "cc-session",
            "message": {"role": "assistant", "content": "done"},
        }
    ) + "\n"
    with patch.object(agent_runtime, "run_agent", new=fake_run_agent):
        result = asyncio.run(
            agent_runtime.run_claude_native_resume(
                sb,
                workdir="/testbed",
                env={"A": "1"},
                time_budget_sec=10,
                session_jsonl=native,
            )
        )
    assert "--resume /tmp/slime_cc_branch.session.jsonl" in seen["cmd"]
    assert "--fork-session" in seen["cmd"]
    assert "--resume-session-at" not in seen["cmd"]
    assert "__SLIME_TOKEN_EXACT_RESUME_HANDSHAKE__" in sb.files[
        "/tmp/slime_cc_resume_handshake.jsonl"
    ]
    assert result["trajectory_jsonl"] == '{"type":"result"}\n'


def test_collect_aligned_turn_logprobs_joins_by_tool_use_id():
    text = SimpleNamespace(output_log_probs=[-0.5])
    multi = SimpleNamespace(output_log_probs=[-1.0, -2.0])
    single = SimpleNamespace(output_log_probs=[-3.0])
    dropped = SimpleNamespace(output_log_probs=[-9.0])
    session = SimpleNamespace(
        turn_log=[
            (text, []),
            (multi, ["toolu_a", "toolu_b"]),
            (single, ["toolu_c"]),
            (dropped, ["toolu_x"]),
        ]
    )
    adapter = SimpleNamespace(store={"sid": session})
    assert live_runners._collect_aligned_turn_logprobs(
        adapter, "sid", step_tool_use_ids=["toolu_a", "toolu_b", "toolu_c"]
    ) == [
        [-1.0, -2.0],
        [-1.0, -2.0],
        [-3.0],
    ]
    assert live_runners._collect_aligned_turn_logprobs(adapter, "missing", step_tool_use_ids=[]) == []


def test_collect_aligned_turn_logprobs_missing_id_raises():
    turn = SimpleNamespace(output_log_probs=[-1.0])
    session = SimpleNamespace(turn_log=[(turn, ["toolu_a"])])
    adapter = SimpleNamespace(store={"sid": session})
    with pytest.raises(live_runners.StepTurnAlignmentError, match="missing_turn_for_steps"):
        live_runners._collect_aligned_turn_logprobs(
            adapter, "sid", step_tool_use_ids=["toolu_a", "toolu_missing"]
        )


def test_hybrid_defaults_to_live_runners():
    called = {"v": 0, "b": 0}

    async def fake_v(**kwargs):
        called["v"] += 1
        d = tempfile.mkdtemp()
        steps = steps_from_diff_files(d, ["", "diff --git a/a b/a\n+1\n"])
        metadata_payload = '{"version":1,"records":[],"unsupported_paths":[]}'
        with open(os.path.join(d, ".cagent_snapshots", "initial.metadata.json"), "w") as f:
            f.write(metadata_payload)
        for i, step in enumerate(steps):
            step.tool_use_id = f"toolu_{i}"
            step.metadata_file = os.path.join(
                ".cagent_snapshots", f"step_{i:04d}.metadata.json"
            )
            with open(os.path.join(d, step.metadata_file), "w") as f:
                f.write(metadata_payload)
        b = SessionBundle(
            instance_id="x",
            session_id="s",
            cc_session_id="",
            task_metadata={"image": "img", "workdir": "/testbed"},
            steps=steps,
            prompt_checkpoints_valid=True,
            native_session_valid=True,
            workspace_metadata_valid=True,
            dir=d,
        )
        events = []
        for i in range(len(steps)):
            tool_id = f"toolu_{i}"
            events.extend(
                [
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "tool_use", "id": tool_id}],
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [{"type": "tool_result", "tool_use_id": tool_id}],
                        },
                    },
                ]
            )
        with open(os.path.join(d, "transcript.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(event) for event in events) + "\n")
        with gzip.open(os.path.join(d, "prompt_checkpoints.json.gz"), "wt", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "generated_tool_use_ids": ["toolu_1"],
                        "generated_tool_use_names": {"toolu_1": "Edit"},
                        "chain_kind": "main",
                        "request_kind": "new",
                    }
                ],
                f,
            )
        b.save(d)
        s = Sample(prompt="p", index=0, group_index=0, reward=0.0, metadata={})
        return b, [s], False, [[], [-5.0]]

    async def fake_b(**kwargs):
        called["b"] += 1
        s = Sample(prompt="p", index=1, group_index=0, reward=0.0, metadata={"sample_kind": "branch"})
        s.loss_mask = [1]
        return [s]

    with patch.dict(os.environ, {"STEP_GRPO_HYBRID_K": "1"}):
        with patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.live_vanilla_runner",
            new=fake_v,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.live_branch_runner",
            new=fake_b,
        ):
            from examples.claudecode_ags.step_reconstruct.hybrid_generate import hybrid_generate

            out = asyncio.run(
                hybrid_generate(SimpleNamespace(), Sample(prompt="p", index=0, group_index=0), {})
            )
    assert called["v"] == 1
    assert called["b"] == 1
    assert len(out) == 2
    assert {s.rollout_id for s in out} == {0}


def test_trial_and_branch_samples_share_parent_rollout_id():
    parent = Sample(prompt="p", index=7, group_index=3, rollout_id=7)
    trial = live_runners._trial_sample(parent, trial_idx=2, base_index=7, group_index=3)
    assert trial.index == 7 * 4096 + 2
    assert trial.rollout_id == 7

    parent_no_rid = Sample(prompt="p", index=5, group_index=1)
    assert live_runners._shared_rollout_id(parent_no_rid) == 5
    trial2 = live_runners._trial_sample(parent_no_rid, trial_idx=0, base_index=5, group_index=1)
    assert trial2.rollout_id == 5


def test_hybrid_stamps_shared_rollout_id_on_mixed_siblings():
    from examples.claudecode_ags.step_reconstruct.hybrid_generate import _stamp_shared_rollout_id

    parent = Sample(prompt="p", index=3, group_index=0)
    siblings = [
        Sample(prompt="p", index=0, rollout_id=0),
        Sample(prompt="p", index=1384, rollout_id=1384),
    ]
    out = _stamp_shared_rollout_id(siblings, parent)
    assert [s.rollout_id for s in out] == [3, 3]


def test_turn_alignment_failure_disables_only_stage2(tmp_path):
    steps = steps_from_diff_files(str(tmp_path), [""])
    steps[0].tool_use_id = "toolu_missing"
    bundle = SessionBundle(
        instance_id="inst",
        session_id="sid",
        cc_session_id="",
        task_metadata={"workdir": "/testbed"},
        steps=steps,
        transcript_valid=True,
        dir=str(tmp_path),
    )
    bundle.save(str(tmp_path))

    aligned = live_runners._collect_stage2_alignment_or_disable(
        SimpleNamespace(store={}),
        "sid",
        bundle,
    )

    assert aligned == [[]]
    assert bundle.transcript_valid is False
    assert "missing_turn_for_steps" in bundle.transcript_error
    persisted = SessionBundle.load(str(tmp_path))
    assert persisted.transcript_valid is False


def test_live_vanilla_runner_happy_path(tmp_path):
    sample = Sample(
        prompt="p",
        index=0,
        group_index=0,
        metadata={
            "instance_id": "inst",
            "image": "img:latest",
            "workdir": "/testbed",
            "problem_statement": "bug",
            "eval_cmd": "true",
            "agent_prompt": "fix",
        },
    )
    md = {
        "instance_id": "inst",
        "image": "img:latest",
        "workdir": "/testbed",
        "problem_statement": "bug",
        "eval_cmd": "true",
        "agent_prompt": "fix",
        "data_source": "swegym",
        "base_commit": "abc123",
        "swe_smith_bug_patch": None,
        "pre_commands": "",
        "install_config": {},
    }

    adapter = MagicMock()
    adapter.open_session = MagicMock()
    adapter.export_prompt_checkpoints_async = AsyncMock(return_value=[{"checkpoint_id": "cp"}])
    turn = SimpleNamespace(output_log_probs=[-0.5])
    adapter.store = {
        "ccags-inst-0-0": SimpleNamespace(
            segments=[],
            active_sub=None,
            main=SimpleNamespace(turns=[turn]),
            turn_log=[(turn, ["toolu_happy"])],
        )
    }
    adapter.finish_session = AsyncMock(return_value=[{"tokens": [1, 2], "response_length": 1}])
    adapter.shutdown_session = AsyncMock()

    state = SimpleNamespace(
        adapter=adapter,
        adapter_url="http://127.0.0.1:9",
        max_context_len=4096,
        tokenizer=object(),
    )

    sb = _FakeSB()
    sandbox_cm = MagicMock()
    sandbox_cm.__aenter__ = AsyncMock(return_value=sb)
    sandbox_cm.__aexit__ = AsyncMock(return_value=False)

    steps = steps_from_diff_files(str(tmp_path), ["diff --git a/a b/a\n+1\n"])
    steps[0].tool_use_id = "toolu_happy"
    bundle = SessionBundle(
        instance_id="inst",
        session_id="sid",
        cc_session_id="",
        task_metadata=md,
        steps=steps,
        dir=str(tmp_path),
    )
    bundle.save(str(tmp_path))

    eval_result = SimpleNamespace(resolved=False, applied_cleanly=True, details={})
    prepare_workspace = AsyncMock()
    capture_bundle = AsyncMock(return_value=bundle)
    timeout_deadlines = []
    real_timeout_at = asyncio.timeout_at

    def record_timeout_at(deadline):
        timeout_deadlines.append(deadline)
        return real_timeout_at(deadline)

    async def _run():
        with patch.dict(os.environ, {"STEP_GRPO_BUNDLE_DIR": str(tmp_path)}), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._parse_metadata",
            return_value=md,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._timeouts",
            return_value=(30, 30, 60),
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._AdapterService",
            return_value=state,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._session_id",
            return_value="ccags-inst-0-0",
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._build_claude_env",
            return_value={"ANTHROPIC_BASE_URL": "x"},
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.make_sandbox",
            return_value=sandbox_cm,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.prepare_workspace",
            prepare_workspace,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.install_toolchain",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.install_snapshot_hook",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.run_claude",
            new_callable=AsyncMock,
            return_value={"exit_code": 0, "trajectory_jsonl": '{"type":"result"}\n'},
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners._common.workspace_diff",
            new_callable=AsyncMock,
            return_value="diff --git a/a b/a\n+1\n",
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.capture_snapshots_to_bundle",
            capture_bundle,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._evaluate_diff",
            new_callable=AsyncMock,
            return_value=eval_result,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.load_function",
            return_value=lambda **kw: (0.0, {}),
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.fan_out_sample_segments",
            return_value=[Sample(prompt="p", index=0, group_index=0, reward=0.0, metadata={})],
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.asyncio.timeout_at",
            side_effect=record_timeout_at,
        ):
            return await live_runners.live_vanilla_runner(
                args=SimpleNamespace(hf_checkpoint="x"),
                sample=sample,
                sampling_params={},
                trial_idx=0,
                group_index=0,
                base_index=0,
            )

    b, samples, solved, lps = asyncio.run(_run())
    assert b is bundle
    assert len(samples) == 1
    assert solved is False
    assert lps == [[-0.5]]
    adapter.finish_session.assert_awaited()
    prepare_workspace.assert_awaited_once()
    kw = prepare_workspace.await_args.kwargs
    assert kw["workdir"] == "/testbed"
    assert kw["problem_statement"] == "bug"
    assert kw["instance_id"] == "inst"
    assert kw["data_source"] == "swegym"
    assert kw["base_commit"] == "abc123"
    assert kw["swe_smith_bug_patch"] is None
    assert kw["pre_commands"] == ""
    assert kw["install_config"] == {}
    assert kw["rollout_side"] is True
    capture_kw = capture_bundle.await_args.kwargs
    assert capture_kw["initial_diff"] == "diff --git a/a b/a\n+1\n"
    assert capture_kw["transcript_text"] == '{"type":"result"}\n'
    assert capture_kw["prompt_checkpoints"] == [{"checkpoint_id": "cp"}]
    assert capture_kw["cc_session_id"]
    adapter.export_prompt_checkpoints_async.assert_awaited_once_with("ccags-inst-0-0", clear=True)
    assert len(timeout_deadlines) == 2
    assert timeout_deadlines[0] == timeout_deadlines[1]


def test_live_branch_runner_sets_step_group_key(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    steps = steps_from_diff_files(str(d), ["diff --git a/a b/a\n+1\n"])
    steps[0].tool_use_id = "toolu_edit"
    md = {
        "instance_id": "inst",
        "image": "img:latest",
        "workdir": "/testbed",
        "problem_statement": "bug",
        "eval_cmd": "true",
        "agent_prompt": "fix",
    }
    bundle = SessionBundle(
        instance_id="inst",
        session_id="s",
        cc_session_id="cc-session",
        task_metadata=md,
        steps=steps,
        cc_session_files=["cc_projects/project/cc-session.jsonl"],
        prompt_checkpoints_valid=True,
        native_session_valid=True,
        workspace_metadata_valid=True,
        dir=str(d),
    )
    (d / ".cagent_snapshots" / "initial.metadata.json").write_text(
        '{"version":1,"records":[],"unsupported_paths":[]}'
    )
    bundle.save(str(d))
    (d / "transcript.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "tool_use", "id": "toolu_edit"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [{"type": "tool_result", "tool_use_id": "toolu_edit"}],
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    with gzip.open(d / "prompt_checkpoints.json.gz", "wt", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "checkpoint_id": "main-0-checkpoint",
                    "generated_tool_use_ids": ["toolu_edit"],
                    "generated_tool_use_names": {"toolu_edit": "Edit"},
                }
            ],
            f,
        )
    native_dir = d / "cc_projects" / "project"
    native_dir.mkdir(parents=True)
    (native_dir / "cc-session.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "native-user",
                        "parentUuid": None,
                        "isSidechain": False,
                        "sessionId": "cc-session",
                        "cwd": "/testbed",
                        "version": "2.1.104",
                        "message": {"role": "user", "content": "fix"},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "native-target",
                        "parentUuid": "native-user",
                        "isSidechain": False,
                        "sessionId": "cc-session",
                        "cwd": "/testbed",
                        "version": "2.1.104",
                        "message": {
                            "id": "native-message",
                            "content": [{"type": "tool_use", "id": "toolu_edit"}],
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    adapter = MagicMock()
    adapter.open_session = MagicMock()
    adapter.open_session_async = AsyncMock()
    adapter.resume_status = MagicMock(
        return_value={
            "mode": "token_exact",
            "checkpoint_id": "main-0-checkpoint",
            "prompt_sha256": "abc",
            "first_prompt_sha256": "abc",
            "first_prompt_exact": True,
            "handshake_validated": True,
            "exact_request_count": 1,
            "last_stop_reason": "end_turn",
            "tool_use_echo_mismatch_count": 2,
            "tool_use_echo_missing_count": 1,
            "tool_use_echo_payload_mismatch_count": 1,
            "runtime_tool_schema_mismatch_count": 3,
            "generated_runtime_tool_unavailable_count": 2,
            "generated_runtime_tool_input_invalid_count": 6,
            "max_tokens_continuation_count": 4,
            "post_end_turn_ack_count": 1,
            "request_replay_count": 5,
            "error": "",
        }
    )
    adapter.finish_session = AsyncMock(return_value=[{"tokens": [1], "response_length": 1}])
    adapter.shutdown_session = AsyncMock()
    state = SimpleNamespace(
        adapter=adapter,
        adapter_url="http://127.0.0.1:9",
        max_context_len=4096,
        tokenizer=object(),
    )

    sb = _FakeSB()
    rebuild_cm = MagicMock()
    rebuild_cm.__aenter__ = AsyncMock(return_value=(sb, True))
    rebuild_cm.__aexit__ = AsyncMock(return_value=False)

    sample = Sample(prompt="p", index=0, group_index=3, metadata={})
    eval_result = SimpleNamespace(resolved=True, applied_cleanly=True, details={})
    timeout_deadlines = []
    real_timeout_at = asyncio.timeout_at

    def record_timeout_at(deadline):
        timeout_deadlines.append(deadline)
        return real_timeout_at(deadline)

    def fake_fan_out(sample, segments, reward=0.0, tokenizer=None, metadata=None, rollout_id=None):
        s = Sample(prompt="p", index=1, group_index=3, reward=float(reward), response_length=2, metadata={})
        s.metadata = dict(metadata or {})
        s.rollout_id = sample.rollout_id if rollout_id is None else rollout_id
        s.loss_mask = None
        return [s]

    async def _run():
        with patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._timeouts",
            return_value=(30, 30, 60),
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._AdapterService",
            return_value=state,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._session_id",
            return_value="branch-sid",
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._build_claude_env",
            return_value={},
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.rebuilt_workspace",
            return_value=rebuild_cm,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.install_toolchain",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.ensure_claude_home_writable",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.install_snapshot_hook",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.run_claude_native_resume",
            new_callable=AsyncMock,
            return_value={"exit_code": 0, "trajectory_jsonl": '{"type":"result"}\n'},
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners._common.workspace_diff",
            new_callable=AsyncMock,
            return_value="diff --git a/a b/a\n+1\n",
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.gen._evaluate_diff",
            new_callable=AsyncMock,
            return_value=eval_result,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.load_function",
            return_value=lambda **kw: (1.0, {}),
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.fan_out_sample_segments",
            side_effect=fake_fan_out,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.asyncio.timeout_at",
            side_effect=record_timeout_at,
        ):
            return await live_runners.live_branch_runner(
                args=SimpleNamespace(hf_checkpoint="x"),
                sample=sample,
                sampling_params={},
                bundle=bundle,
                source_trial_idx=1,
                edit_step_i=0,
                branch_step_t=-1,
                branch_idx=2,
                group_index=3,
                edit_ppl=4.5,
            )

    samples = asyncio.run(_run())
    assert len(samples) == 1
    assert samples[0].metadata["sample_kind"] == "branch"
    assert samples[0].metadata["step_group_key"] == "3:1:edit:0"
    assert samples[0].metadata["edit_step_i"] == 0
    assert samples[0].metadata["branch_step_t"] == -1
    assert samples[0].metadata["edit_ppl"] == 4.5
    assert samples[0].metadata["agent_exit_code"] == 0
    assert samples[0].metadata["prefix_reseed_mode"] == "native-session-checkpoint"
    assert samples[0].metadata["tool_use_echo_mismatch_count"] == 2
    assert samples[0].metadata["tool_use_echo_missing_count"] == 1
    assert samples[0].metadata["tool_use_echo_payload_mismatch_count"] == 1
    assert samples[0].metadata["runtime_tool_schema_mismatch_count"] == 3
    assert samples[0].metadata["generated_runtime_tool_unavailable_count"] == 2
    assert samples[0].metadata["generated_runtime_tool_input_invalid_count"] == 6
    assert samples[0].metadata["max_tokens_continuation_count"] == 4
    assert samples[0].metadata["post_end_turn_ack_count"] == 1
    assert samples[0].metadata["resume_request_replay_count"] == 5
    assert samples[0].metadata["resume_last_stop_reason"] == "end_turn"
    assert samples[0].metadata["stage2_loss_scope"] == "full_continuation"
    assert "agent_queue_wait_sec" in samples[0].metadata
    assert samples[0].metadata["eval_queue_wait_sec"] == 0.0
    assert len(timeout_deadlines) == 2
    assert timeout_deadlines[0] == timeout_deadlines[1]
    # Do not force all-1s loss_mask (breaks TIS); fan_out owns the mask.
    assert samples[0].loss_mask is None
    assert samples[0].rollout_id == 0


def test_stage2_first_turn_scope_masks_later_outputs(monkeypatch):
    from slime.agent.segment_trajectory import TokenSegment

    monkeypatch.setenv("STEP_GRPO_STAGE2_LOSS_SCOPE", "first_turn")
    segments = [
        TokenSegment(
            prompt_ids=[1],
            response_ids=[2, 3, 4, 5],
            loss_mask=[1, 0, 1, 1],
            rollout_log_probs=[-0.1, 0.0, -0.2, -0.3],
            metadata={"assistant_output_spans": [[0, 1], [2, 4]], "assistant_turn_count": 2},
        )
    ]

    scoped, audit = live_runners._scope_stage2_segments(segments)
    assert scoped[0].loss_mask == [1, 0, 0, 0]
    assert scoped[0].rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    assert audit == {
        "scope": "first_turn",
        "assistant_turns": 2,
        "total_trainable_tokens": 3,
        "kept_trainable_tokens": 1,
        "masked_trainable_tokens": 2,
        "first_turn_segment_idx": 0,
    }


def test_stage2_loss_scope_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_STAGE2_LOSS_SCOPE", "typo")
    with pytest.raises(ValueError, match="full_continuation or first_turn"):
        live_runners._stage2_loss_scope()
