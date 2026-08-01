from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from examples.claudecode_ags.swe_eval import audit


def _dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "extra_info": {
                    "instance_id": "x__x-1",
                    "repo": "x/x",
                    "version": "1",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_preflight_writes_passed_report(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    report = tmp_path / "startup.json"
    _dataset(dataset)
    spec = SimpleNamespace(
        instance_id="x__x-1",
        repo="x/x",
        version="1",
        eval_script=">>>>> Start Test Output\n>>>>> End Test Output\n",
    )
    monkeypatch.setattr(audit, "require_swebench_version", lambda version: version)
    monkeypatch.setattr(audit, "make_official_test_spec", lambda metadata: spec)
    monkeypatch.setattr(
        audit,
        "require_official_parser",
        lambda test_spec: SimpleNamespace(__name__="parse_log_x"),
    )

    assert audit.run_preflight(
        dataset=dataset,
        report=report,
        required_version="4.1.0",
        expected_tasks=1,
    ) == 0
    payload = json.loads(report.read_text())
    assert payload["status"] == "passed"
    assert payload["task_count"] == 1
    assert payload["parsers"] == {"x/x": "parse_log_x"}


def test_postflight_rejects_fallback_parser(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    artifacts = tmp_path / "artifacts"
    report = tmp_path / "artifact_check.json"
    _dataset(dataset)
    trial = artifacts / "x__x-1" / "trial"
    trial.mkdir(parents=True)
    (trial / "manifest.json").write_text(
        json.dumps({"instance_id": "x__x-1", "evaluation_complete": True})
    )
    (trial / "trajectory.jsonl").write_text("{}\n")
    (trial / "model.patch").write_text("")
    (trial / "eval_result.json").write_text(
        json.dumps(
            {
                "resolved": False,
                "details": {
                    "mode": "swebench",
                    "parser": "pytest_robust",
                    "swebench_version": "4.1.0",
                },
            }
        )
    )
    monkeypatch.setattr(audit, "require_swebench_version", lambda version: version)

    assert audit.run_postflight(
        dataset=dataset,
        artifact_dir=artifacts,
        report=report,
        required_version="4.1.0",
        expected_tasks=1,
        samples_per_task=1,
    ) == 1
    payload = json.loads(report.read_text())
    assert payload["error_kinds"] == {"fallback_parser": 1}


def test_postflight_accepts_complete_official_artifacts(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    artifacts = tmp_path / "artifacts"
    report = tmp_path / "artifact_check.json"
    _dataset(dataset)
    trial = artifacts / "x__x-1" / "trial"
    trial.mkdir(parents=True)
    (trial / "manifest.json").write_text(
        json.dumps({"instance_id": "x__x-1", "evaluation_complete": True})
    )
    (trial / "trajectory.jsonl").write_text("{}\n")
    (trial / "model.patch").write_text("")
    (trial / "eval_result.json").write_text(
        json.dumps(
            {
                "resolved": True,
                "details": {
                    "mode": "swebench",
                    "parser": "parse_log_x",
                    "swebench_version": "4.1.0",
                },
            }
        )
    )
    monkeypatch.setattr(audit, "require_swebench_version", lambda version: version)

    assert audit.run_postflight(
        dataset=dataset,
        artifact_dir=artifacts,
        report=report,
        required_version="4.1.0",
        expected_tasks=1,
        samples_per_task=1,
    ) == 0
    payload = json.loads(report.read_text())
    assert payload["status"] == "passed"
    assert payload["resolved_count"] == 1
    assert payload["sample_resolved_rate"] == 1.0
    assert payload["pass_at_1"] == 1.0


def test_repair_dataset_contains_only_missing_or_incomplete_trials(tmp_path):
    dataset = tmp_path / "data.jsonl"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "repair.jsonl"
    report = tmp_path / "repair.json"
    records = [
        {"prompt": "one", "extra_info": {"instance_id": "x__x-1", "repo": "x/x", "version": "1"}},
        {"prompt": "two", "extra_info": {"instance_id": "x__x-2", "repo": "x/x", "version": "1"}},
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in records))
    trial = artifacts / "x__x-1" / "trial"
    trial.mkdir(parents=True)
    (trial / "manifest.json").write_text(
        json.dumps({"instance_id": "x__x-1", "evaluation_complete": True})
    )
    (trial / "eval_result.json").write_text(
        json.dumps(
            {
                "resolved": False,
                "details": {
                    "mode": "swebench",
                    "parser": "parse_log_x",
                    "swebench_version": "4.1.0",
                },
            }
        )
    )

    assert audit.run_repair_dataset(
        dataset=dataset,
        artifact_dir=artifacts,
        output_dataset=output,
        report=report,
        required_version="4.1.0",
    ) == 0
    repair_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert repair_rows == [records[1]]
    payload = json.loads(report.read_text())
    assert payload["complete_count"] == 1
    assert payload["missing_instance_ids"] == ["x__x-2"]
