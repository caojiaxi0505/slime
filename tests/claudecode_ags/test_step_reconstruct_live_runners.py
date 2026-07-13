"""Mocked unit tests for live runners + capture helpers (no AGS)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from examples.claudecode_ags.step_reconstruct import live_runners
from examples.claudecode_ags.step_reconstruct.session_capture import (
    SessionBundle,
    build_hook_steps_from_snap_dir,
    pull_remote_dir,
    steps_from_diff_files,
)
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


def test_run_claude_with_prefix_cmd():
    from examples.claudecode_ags import agent_runtime

    seen = {}

    async def fake_run_agent(sb, *, workdir, start_cmd, env, time_budget_sec):
        seen["cmd"] = start_cmd
        return 0

    with patch.object(agent_runtime, "run_agent", new=fake_run_agent):
        ec = asyncio.run(
            agent_runtime.run_claude_with_prefix(
                object(),
                workdir="/testbed",
                prompt="fix me",
                env={"A": "1"},
                time_budget_sec=10,
                prefix_path="/tmp/cc_prefix.jsonl",
            )
        )
    assert ec["exit_code"] == 0
    assert "--input-format stream-json" in seen["cmd"]
    assert "/tmp/cc_prefix.jsonl" in seen["cmd"]


def test_collect_turn_logprobs():
    turn = SimpleNamespace(output_log_probs=[-1.0, -2.0])
    main = SimpleNamespace(turns=[turn])
    session = SimpleNamespace(segments=[], active_sub=None, main=main)
    adapter = SimpleNamespace(store={"sid": session})
    assert live_runners._collect_turn_logprobs(adapter, "sid") == [[-1.0, -2.0]]
    assert live_runners._collect_turn_logprobs(adapter, "missing") == []


def test_hybrid_defaults_to_live_runners():
    called = {"v": 0, "b": 0}

    async def fake_v(**kwargs):
        called["v"] += 1
        d = tempfile.mkdtemp()
        steps = steps_from_diff_files(d, ["", "diff --git a/a b/a\n+1\n"])
        b = SessionBundle(
            instance_id="x",
            session_id="s",
            cc_session_id="",
            task_metadata={"image": "img", "workdir": "/testbed"},
            steps=steps,
            dir=d,
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
    }

    adapter = MagicMock()
    adapter.open_session = MagicMock()
    turn = SimpleNamespace(output_log_probs=[-0.5])
    adapter.store = {
        "ccags-inst-0-0": SimpleNamespace(
            segments=[],
            active_sub=None,
            main=SimpleNamespace(turns=[turn]),
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

    bundle = SessionBundle(
        instance_id="inst",
        session_id="sid",
        cc_session_id="",
        task_metadata=md,
        steps=[],
        dir=str(tmp_path),
    )
    bundle.save(str(tmp_path))

    eval_result = SimpleNamespace(resolved=False, applied_cleanly=True, details={})

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
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.install_toolchain",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.install_snapshot_hook",
            new_callable=AsyncMock,
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.agent_runtime.run_claude",
            new_callable=AsyncMock,
            return_value={"exit_code": 0},
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners._common.workspace_diff",
            new_callable=AsyncMock,
            return_value="diff --git a/a b/a\n+1\n",
        ), patch(
            "examples.claudecode_ags.step_reconstruct.live_runners.capture_snapshots_to_bundle",
            new_callable=AsyncMock,
            return_value=bundle,
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


def test_live_branch_runner_sets_step_group_key(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    steps = steps_from_diff_files(str(d), ["diff --git a/a b/a\n+1\n"])
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
        cc_session_id="",
        task_metadata=md,
        steps=steps,
        dir=str(d),
    )
    bundle.save(str(d))
    (d / "transcript.jsonl").write_text('{"type":"tool_result","id":"1"}\n')

    adapter = MagicMock()
    adapter.open_session = MagicMock()
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

    def fake_fan_out(sample, segments, reward=0.0, tokenizer=None, metadata=None):
        s = Sample(prompt="p", index=1, group_index=3, reward=float(reward), response_length=2, metadata={})
        s.metadata = dict(metadata or {})
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
            "examples.claudecode_ags.step_reconstruct.live_runners.resume_and_run",
            new_callable=AsyncMock,
            return_value={"exit_code": 0},
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
        ):
            return await live_runners.live_branch_runner(
                args=SimpleNamespace(hf_checkpoint="x"),
                sample=sample,
                sampling_params={},
                bundle=bundle,
                source_trial_idx=1,
                step_t=0,
                branch_idx=2,
                group_index=3,
                edit_ppl=4.5,
            )

    samples = asyncio.run(_run())
    assert len(samples) == 1
    assert samples[0].metadata["sample_kind"] == "branch"
    assert samples[0].metadata["step_group_key"] == "3:1:0"
    assert samples[0].metadata["edit_ppl"] == 4.5
    assert samples[0].loss_mask == [1, 1]
