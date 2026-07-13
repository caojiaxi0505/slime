"""Offline tests for SWE-Gym filter logic + run store."""

from __future__ import annotations

import json
from pathlib import Path

from examples.claudecode_ags.eval.filter_logic import (
    decide_gold_task,
    decide_passk_task,
    summarize_passk_runs,
)
from examples.claudecode_ags.eval.run_store import RunStore


def test_gold_keep_only_if_all_resolved():
    runs = [{"resolved": True, "infra_ok": True} for _ in range(4)]
    d = decide_gold_task(runs, expected_repeats=4)
    assert d["keep"] is True
    assert d["exclude_reason"] is None


def test_gold_exclude_on_any_unresolved():
    runs = [
        {"resolved": True, "infra_ok": True},
        {"resolved": False, "infra_ok": True},
    ]
    d = decide_gold_task(runs, expected_repeats=4)
    assert d["keep"] is False
    assert d["exclude_reason"] == "gold_unresolved"


def test_gold_exclude_on_infra():
    runs = [{"resolved": False, "infra_ok": False}]
    d = decide_gold_task(runs, expected_repeats=4)
    assert d["exclude_reason"] == "infra"


def test_gold_incomplete():
    runs = [{"resolved": True, "infra_ok": True} for _ in range(2)]
    d = decide_gold_task(runs, expected_repeats=4)
    assert d["keep"] is False
    assert d["exclude_reason"] == "incomplete"


def test_passk_drop_only_always_resolved():
    runs = [{"resolved": True, "infra_ok": True, "diff_chars": 10} for _ in range(8)]
    s = summarize_passk_runs(runs)
    d = decide_passk_task(s, expected_repeats=8)
    assert d["keep"] is False
    assert d["exclude_reason"] == "always_resolved"
    assert s["n_resolved"] == 8
    assert s["pass_at_k"] is True


def test_passk_keep_zero_and_partial():
    zero = summarize_passk_runs([{"resolved": False, "infra_ok": True, "diff_chars": 0}] * 8)
    assert decide_passk_task(zero, expected_repeats=8)["keep"] is True
    assert zero["pass_at_k"] is False
    partial = summarize_passk_runs(
        [{"resolved": True, "infra_ok": True, "diff_chars": 1}] * 3
        + [{"resolved": False, "infra_ok": True, "diff_chars": 0}] * 5
    )
    assert decide_passk_task(partial, expected_repeats=8)["keep"] is True
    assert partial["n_resolved"] == 3
    assert partial["pass_at_k"] is True


def test_run_store_resume_keys(tmp_path: Path):
    store = RunStore(tmp_path / "out")
    store.append_run({"phase": "gold", "instance_id": "a__1", "repeat_idx": 0, "resolved": True})
    store.append_run({"phase": "gold", "instance_id": "a__1", "repeat_idx": 1, "resolved": False})
    store.append_run({"phase": "passk", "instance_id": "a__1", "repeat_idx": 0, "resolved": True})
    assert store.completed_keys("gold") == {("a__1", 0), ("a__1", 1)}
    assert store.completed_keys("passk") == {("a__1", 0)}
    store.write_named_jsonl("kept.jsonl", [{"instance_id": "a__1"}])
    rows = json.loads((tmp_path / "out" / "kept.jsonl").read_text(encoding="utf-8").strip())
    assert rows["instance_id"] == "a__1"
