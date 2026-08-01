"""Mocked wiring test: pluggable binary reward + fan_out in generate()."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from examples.claudecode_ags.swe_eval.base import EvalResult
from slime.agent.segment_trajectory import TokenSegment
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample
from tests.claudecode_ags.fake_sandbox import FakeSandbox


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def _seg(kind: str, n_resp: int = 2) -> TokenSegment:
    return TokenSegment(
        prompt_ids=[1],
        response_ids=list(range(10, 10 + n_resp)),
        loss_mask=[1] * n_resp,
        rollout_log_probs=[0.0] * n_resp,
        metadata={"segment_kind": kind},
    )


@pytest.fixture(autouse=True)
def _clear_adapter_singleton():
    SingletonMeta._instances = {}
    yield
    SingletonMeta._instances = {}


def test_filter_cc_env_skips_swe_aliases():
    from examples.claudecode_ags.generate import _filter_cc_env

    filtered = _filter_cc_env(
        {
            "ANTHROPIC_BASE_URL": "http://x",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192",
            "BASH_MAX_TIMEOUT_MS": "900000",
            "API_TIMEOUT_MS": "1200000",
            "SWE_AGENT_TIME_BUDGET_SEC": "1800",
            "PATH": "/usr/bin",
        }
    )
    assert "ANTHROPIC_BASE_URL" in filtered
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in filtered
    assert "BASH_MAX_TIMEOUT_MS" in filtered
    assert "API_TIMEOUT_MS" in filtered
    assert "SWE_AGENT_TIME_BUDGET_SEC" not in filtered
    assert "PATH" not in filtered


def test_save_eval_artifacts_preserves_trajectory_patch_and_grade(monkeypatch, tmp_path):
    import examples.claudecode_ags.generate as gen

    monkeypatch.setenv("SLIME_CC_EVAL_ARTIFACT_DIR", str(tmp_path))
    sample = Sample(index=7, group_index=1, prompt="fix it")
    result = EvalResult(
        resolved=True,
        applied_cleanly=True,
        details={"exit_code": 0, "stdout": "ok"},
    )

    asyncio.run(
        gen._save_eval_artifacts(
            sample=sample,
            instance_id="repo/name__1",
            session_id="session-1",
            agent_result={
                "exit_code": 0,
                "trajectory_path": "/testbed/.harness/trajectory.jsonl",
                "trajectory_jsonl": '{"type":"assistant"}\n',
            },
            diff_text="diff --git a/x b/x\n",
            agent_prompt="fix it",
            agent_time_budget_sec=1800,
            eval_result=result,
        )
    )

    trial_dirs = list(tmp_path.glob("repo_name__1/*"))
    assert len(trial_dirs) == 1
    trial_dir = trial_dirs[0]
    assert (trial_dir / "trajectory.jsonl").read_text() == '{"type":"assistant"}\n'
    assert (trial_dir / "model.patch").read_text() == "diff --git a/x b/x\n"
    manifest = json.loads((trial_dir / "manifest.json").read_text())
    assert manifest["evaluation_complete"] is True
    assert manifest["agent_runtime"]["time_budget_sec"] == 1800
    assert manifest["agent_runtime"]["agent_prompt"] == "fix it"
    assert json.loads((trial_dir / "eval_result.json").read_text())["resolved"] is True


def test_eval_infra_timeout_retries_in_fresh_sandbox(monkeypatch):
    import examples.claudecode_ags.generate as gen

    expected = EvalResult(resolved=True, applied_cleanly=True, details={"exit_code": 0})
    evaluate = AsyncMock(side_effect=[TimeoutError(), expected])
    monkeypatch.setattr(gen, "_evaluate_diff_once", evaluate)
    monkeypatch.setenv("SLIME_CC_EVAL_INFRA_RETRIES", "1")
    monkeypatch.setenv("SLIME_CC_EVAL_GUARD_SEC", "1")
    monkeypatch.setenv("SLIME_CC_EVAL_CONCURRENCY", "1")
    gen._EVAL_SEM = None
    gen._EVAL_SEM_LIMIT = None

    result = asyncio.run(
        gen._evaluate_diff_with_infra_retry(
            image="image",
            workdir="/testbed",
            eval_cmd="pytest",
            diff_text="",
            timeout_sec=1,
            metadata={},
        )
    )

    assert result is expected
    assert evaluate.await_count == 2
    assert result.details["eval_queue_wait_sec"] >= 0.0


def test_eval_concurrency_caps_whole_attempt(monkeypatch):
    import examples.claudecode_ags.generate as gen

    monkeypatch.setenv("SLIME_CC_EVAL_CONCURRENCY", "1")
    gen._EVAL_SEM = None
    gen._EVAL_SEM_LIMIT = None
    active = 0
    peak = 0

    async def evaluate_once(**_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return EvalResult(resolved=True, applied_cleanly=True)

    monkeypatch.setattr(gen, "_evaluate_diff_once", evaluate_once)

    async def run():
        kwargs = {
            "image": "image",
            "workdir": "/testbed",
            "eval_cmd": "pytest",
            "diff_text": "",
            "timeout_sec": 1,
            "metadata": {},
        }
        return await asyncio.gather(gen._evaluate_diff(**kwargs), gen._evaluate_diff(**kwargs))

    results = asyncio.run(run())

    assert peak == 1
    assert all(result.details["eval_queue_wait_sec"] >= 0.0 for result in results)


def test_default_pipeline_guard_is_2700_seconds(monkeypatch):
    import examples.claudecode_ags.generate as gen

    monkeypatch.delenv("SLIME_CC_GENERATE_GUARD_SEC", raising=False)
    monkeypatch.setenv("SLIME_CC_TIME_BUDGET_SEC", "1800")
    monkeypatch.setenv("SLIME_CC_EVAL_TIMEOUT_SEC", "600")

    assert gen._timeouts() == (1800, 600, 2700)


def test_pipeline_guard_still_covers_eval_after_agent_slot_is_released(monkeypatch):
    import examples.claudecode_ags.generate as gen

    fake_adapter = MagicMock()
    fake_adapter.open_session = MagicMock()
    fake_adapter.shutdown_session = AsyncMock()
    fake_state = SimpleNamespace(
        tokenizer=_Tok(),
        max_context_len=0,
        adapter=fake_adapter,
        adapter_url="http://adapter:18001",
    )
    monkeypatch.setattr(gen, "_AdapterService", lambda _args: fake_state)
    monkeypatch.setattr(gen, "_timeouts", lambda: (10, 10, 0.01))
    monkeypatch.setenv("SLIME_CC_AGENT_CONCURRENCY", "1")
    gen._AGENT_SEM = None
    gen._AGENT_SEM_LIMIT = None
    monkeypatch.setattr(gen, "make_sandbox", lambda _image: FakeSandbox())
    monkeypatch.setattr(gen.agent_runtime, "prepare_workspace", AsyncMock())
    monkeypatch.setattr(gen.agent_runtime, "install_toolchain", AsyncMock())
    monkeypatch.setattr(
        gen.agent_runtime,
        "run_claude",
        AsyncMock(return_value={"exit_code": 0}),
    )
    monkeypatch.setattr(
        gen.agent_runtime,
        "git_diff",
        AsyncMock(return_value="diff --git a/x b/x\n"),
    )

    async def slow_eval(**_kwargs):
        await asyncio.sleep(0.1)
        return EvalResult(resolved=True, applied_cleanly=True)

    monkeypatch.setattr(gen, "_evaluate_diff", slow_eval)
    sample = Sample(
        index=7,
        group_index=1,
        prompt="fix it",
        metadata={
            "image": "img:tag",
            "workdir": "/testbed",
            "problem_statement": "bug",
            "eval_cmd": "pytest -q",
            "instance_id": "repo__pipeline_timeout",
        },
    )
    args = SimpleNamespace(
        custom_cc_reward_function_path="examples.claudecode_ags.rewards.default.compose",
        hf_checkpoint="unused",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )

    result = asyncio.run(gen.generate(args, sample, {"temperature": 0.0}, evaluation=False))

    assert result[0].status == Sample.Status.ABORTED
    assert result[0].metadata["abort_reason"] == "wall_clock_timeout"
    fake_adapter.shutdown_session.assert_awaited()


def test_custom_cc_reward_arg_default():
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "slime" / "utils" / "arguments.py"
    text = src.read_text()
    assert "--custom-cc-reward-function-path" in text
    assert 'default="examples.claudecode_ags.rewards.default.compose"' in text
    # Placed near the custom generate hook.
    gen_idx = text.index("--custom-generate-function-path")
    reward_idx = text.index("--custom-cc-reward-function-path")
    assert reward_idx > gen_idx
    assert reward_idx - gen_idx < 800


def test_only_eval_sampling_params_override_claude_request():
    import examples.claudecode_ags.generate as gen

    params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 8192,
        "sampling_seed": 7,
    }
    assert gen._session_sampling_overrides(params, evaluation=False) == {}
    assert gen._session_sampling_overrides(params, evaluation=True) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }


def test_generate_wires_binary_reward_and_fan_out(monkeypatch):
    import examples.claudecode_ags.generate as gen

    SingletonMeta._instances = {}

    fake_adapter = MagicMock()
    fake_adapter.open_session = MagicMock()
    fake_adapter.finish_session = AsyncMock(return_value=[_seg("wipe"), _seg("final", n_resp=1)])
    fake_adapter.shutdown_session = AsyncMock()

    fake_state = SimpleNamespace(
        tokenizer=_Tok(),
        max_context_len=0,
        adapter=fake_adapter,
        adapter_url="http://adapter:18001",
    )
    monkeypatch.setattr(gen, "_AdapterService", lambda _args: fake_state)

    sandboxes: list[FakeSandbox] = []

    def _fake_make_sandbox(_image: str, **_kwargs):
        sb = FakeSandbox()
        sandboxes.append(sb)
        return sb

    monkeypatch.setattr(gen, "make_sandbox", _fake_make_sandbox)
    monkeypatch.setattr(gen.agent_runtime, "prepare_workspace", AsyncMock())
    monkeypatch.setattr(gen.agent_runtime, "install_toolchain", AsyncMock())
    monkeypatch.setattr(
        gen.agent_runtime,
        "run_claude",
        AsyncMock(return_value={"exit_code": 0}),
    )
    monkeypatch.setattr(gen.agent_runtime, "git_diff", AsyncMock(return_value="diff --git a/x b/x\n"))
    monkeypatch.setattr(
        gen.swe_eval_dispatch,
        "evaluate",
        AsyncMock(return_value=EvalResult(resolved=True, applied_cleanly=True, details={"exit_code": 0})),
    )

    sample = Sample(
        index=7,
        group_index=1,
        prompt="fix it",
        metadata={
            "image": "img:tag",
            "workdir": "/testbed",
            "problem_statement": "bug",
            "eval_cmd": "pytest -q",
            "instance_id": "repo__1",
        },
    )
    args = SimpleNamespace(
        custom_cc_reward_function_path="examples.claudecode_ags.rewards.default.compose",
        hf_checkpoint="unused",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )

    out = asyncio.run(gen.generate(args, sample, {"temperature": 0.0}, evaluation=False))

    assert len(out) == 2
    assert out[0].reward == 0.5
    assert out[1].reward == 0.5
    assert out[0].rollout_id == 7
    assert out[1].rollout_id == 7
    assert out[0].metadata["reward_details"]["mode"] == "binary"
    assert out[0].metadata["grading_solved"] is True
    assert out[0].metadata["base_eval"]["resolved"] is True

    gen.agent_runtime.run_claude.assert_awaited()
    call_kwargs = gen.agent_runtime.run_claude.await_args.kwargs
    assert call_kwargs["env"]["ANTHROPIC_BASE_URL"] == "http://adapter:18001"
    assert call_kwargs["env"]["ANTHROPIC_AUTH_TOKEN"] == sample.session_id
    assert not any(k.startswith("SWE_") for k in call_kwargs["env"])

    fake_adapter.open_session.assert_called_once()
    assert fake_adapter.open_session.call_args.kwargs["sampling_overrides"] == {}
    fake_adapter.finish_session.assert_awaited()
    assert len(sandboxes) == 2  # agent sandbox + fresh eval sandbox
