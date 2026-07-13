"""Offline tests for AGS smoke CLI (no live AGS)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.claudecode_ags.smoke import ags_smoke


def test_build_parser_levels():
    p = ags_smoke.build_parser()
    args = p.parse_args(["--level", "0", "--row-json", "x.json"])
    assert args.level == 0
    args = p.parse_args(["--level", "1", "--dataset-type", "swebench_verified", "--data-path", "a.parquet"])
    assert args.level == 1
    assert args.dataset_type == "swebench_verified"
    args = p.parse_args(["--level", "2", "--row-json", "x.json", "--allow-unresolved"])
    assert args.level == 2
    assert args.allow_unresolved is True


def test_adapter_public_url_requires_env(monkeypatch):
    monkeypatch.delenv("SLIME_ADAPTER_PUBLIC_URL", raising=False)
    with pytest.raises(RuntimeError, match="SLIME_ADAPTER_PUBLIC_URL"):
        ags_smoke._adapter_public_url()


def test_adapter_public_url_rejects_localhost(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_PUBLIC_URL", "http://127.0.0.1:18001")
    with pytest.raises(RuntimeError, match="not reachable"):
        ags_smoke._adapter_public_url()


def test_adapter_public_url_ok(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_PUBLIC_URL", "http://k8s-example.elb.amazonaws.com/")
    assert ags_smoke._adapter_public_url() == "http://k8s-example.elb.amazonaws.com"


def test_load_row_json_object(tmp_path: Path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps({"instance_id": "a__b-1", "patch": "diff", "repo": "a/b"}), encoding="utf-8")
    row = ags_smoke.load_row_json(path)
    assert row["instance_id"] == "a__b-1"


def test_load_row_jsonl_select(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "x__1", "patch": "p1"}),
                json.dumps({"instance_id": "y__2", "patch": "p2"}),
            ]
        ),
        encoding="utf-8",
    )
    row = ags_smoke.load_official_row(data_path=str(path), row_json=None, instance_id="y__2")
    assert row["patch"] == "p2"


def test_prepare_metadata_sets_image():
    row = {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "base_commit": "abc",
        "patch": "diff --git a/x b/x\n",
        "FAIL_TO_PASS": ["t::a"],
        "PASS_TO_PASS": [],
    }
    md = ags_smoke.prepare_metadata(row, "swebench_verified", None)
    assert "sweb.eval.x86_64.astropy_1776_astropy-12907" in md["image"]
    assert md["patch"].startswith("diff")


def test_prepare_metadata_image_override():
    row = {"instance_id": "a__b-1", "patch": "x"}
    md = ags_smoke.prepare_metadata(row, "swebench", "custom.reg/img:tag")
    assert md["image"] == "custom.reg/img:tag"


def test_summarize_claude_stream_counts_tools_and_result():
    lines = [
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit"}]}}),
        json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done"}),
        "not-json",
    ]
    summary = ags_smoke.summarize_claude_stream("\n".join(lines))
    assert summary["lines_total"] == 4
    assert summary["lines_json"] == 3
    assert summary["tools"] == ["Read", "Edit"]
    assert summary["type_counts"]["assistant"] == 2
    assert summary["type_counts"]["result:success"] == 1


def test_export_rollout_artifacts(tmp_path: Path):
    class FakeSB:
        async def read_file(self, path, *, user="root"):
            assert path.endswith("/.harness/trajectory.jsonl")
            return json.dumps({"type": "result", "subtype": "success"}) + "\n"

        async def exec(self, cmd, user="root", timeout=30, check=False):
            if "ls -la" in cmd:
                return 0, "-rw-r--r-- 1 agent agent 12 trajectory.jsonl\n", ""
            if "git status" in cmd:
                return 0, " M foo.py\n", ""
            return 0, "", ""

    async def run():
        return await ags_smoke.export_rollout_artifacts(
            FakeSB(),
            workdir="/testbed",
            dest_dir=tmp_path / "arts",
            diff_text="diff --git a/foo.py\n",
            session_id="l2-test",
            instance_id="astropy__astropy-12907",
        )

    import asyncio

    dest = asyncio.run(run())
    assert dest == tmp_path / "arts"
    assert (dest / "trajectory.jsonl").is_file()
    assert (dest / "model.diff").read_text(encoding="utf-8").startswith("diff")
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    assert meta["diff_chars"] > 0
    assert meta["trajectory_summary"]["lines_json"] == 1


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as ei:
        ags_smoke.build_parser().parse_args(["--help"])
    assert ei.value.code == 0
