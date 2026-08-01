"""Startup and artifact audits for strict SWE-bench evaluation jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.claudecode_ags.swe_eval.official import (
    make_official_test_spec,
    require_official_parser,
    require_swebench_version,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _load_records(dataset: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with dataset.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset}:{line_number}: invalid JSON: {exc}") from exc
            metadata = row.get("extra_info") or row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{dataset}:{line_number}: missing metadata object")
            records.append(row)
    return records


def _load_rows(dataset: Path) -> list[dict[str, Any]]:
    return [
        record.get("extra_info") or record.get("metadata")
        for record in _load_records(dataset)
    ]


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def run_preflight(
    *,
    dataset: Path,
    report: Path,
    required_version: str,
    expected_tasks: int,
) -> int:
    payload: dict[str, Any] = {
        "check": "swebench_startup",
        "dataset": str(dataset),
        "expected_tasks": expected_tasks,
        "required_swebench_version": required_version,
        "agent_runtime": {
            "time_budget_sec": os.environ.get("SLIME_CC_TIME_BUDGET_SEC"),
            "eval_timeout_sec": os.environ.get("SLIME_CC_EVAL_TIMEOUT_SEC"),
            "initial_input_mode": os.environ.get("SLIME_CC_INITIAL_INPUT_MODE"),
            "max_output_tokens": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
            "auto_compact_window": os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"),
            "auto_compact_pct": os.environ.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"),
            "agent_prompt_sha256": hashlib.sha256(
                (os.environ.get("SLIME_CC_AGENT_PROMPT") or "").encode("utf-8")
            ).hexdigest(),
            "extra_args_sha256": hashlib.sha256(
                (os.environ.get("SLIME_CC_EXTRA_ARGS_JSON") or "").encode("utf-8")
            ).hexdigest(),
        },
        "status": "failed",
    }
    try:
        installed = require_swebench_version(required_version)
        rows = _load_rows(dataset)
        if len(rows) != expected_tasks:
            raise ValueError(f"dataset rows={len(rows)}, expected={expected_tasks}")

        instance_ids: list[str] = []
        repo_versions: Counter[tuple[str, str]] = Counter()
        parser_names: dict[str, str] = {}
        command_markers = Counter()
        for metadata in rows:
            spec = make_official_test_spec(metadata)
            parser = require_official_parser(spec)
            instance_ids.append(spec.instance_id)
            repo_versions[(spec.repo, spec.version)] += 1
            parser_names[spec.repo] = parser.__name__
            command_markers["official_start"] += int(">>>>> Start Test Output" in spec.eval_script)
            command_markers["official_end"] += int(">>>>> End Test Output" in spec.eval_script)

        duplicates = sorted(
            instance_id
            for instance_id, count in Counter(instance_ids).items()
            if count != 1
        )
        if duplicates:
            raise ValueError(f"duplicate instance_id values: {duplicates[:20]}")
        if command_markers["official_start"] != expected_tasks:
            raise ValueError("one or more official eval scripts lack the start marker")
        if command_markers["official_end"] != expected_tasks:
            raise ValueError("one or more official eval scripts lack the end marker")

        payload.update(
            {
                "status": "passed",
                "installed_swebench_version": installed,
                "task_count": len(rows),
                "unique_instance_count": len(set(instance_ids)),
                "repo_count": len(parser_names),
                "parsers": dict(sorted(parser_names.items())),
                "repo_versions": {
                    f"{repo}@{version}": count
                    for (repo, version), count in sorted(repo_versions.items())
                },
                "official_eval_script_markers": dict(command_markers),
            }
        )
        rc = 0
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        rc = 1
    _atomic_json(report, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return rc


def run_postflight(
    *,
    dataset: Path,
    artifact_dir: Path,
    report: Path,
    required_version: str,
    expected_tasks: int,
    samples_per_task: int,
) -> int:
    expected_trials = expected_tasks * samples_per_task
    payload: dict[str, Any] = {
        "check": "swebench_artifacts",
        "dataset": str(dataset),
        "artifact_dir": str(artifact_dir),
        "expected_tasks": expected_tasks,
        "samples_per_task": samples_per_task,
        "expected_trials": expected_trials,
        "required_swebench_version": required_version,
        "status": "failed",
    }
    errors: list[str] = []
    error_kinds: Counter[str] = Counter()

    def fail(kind: str, message: str) -> None:
        error_kinds[kind] += 1
        if len(errors) < 200:
            errors.append(f"{kind}: {message}")

    try:
        installed = require_swebench_version(required_version)
        rows = _load_rows(dataset)
        if len(rows) != expected_tasks:
            fail("dataset_count", f"rows={len(rows)}, expected={expected_tasks}")
        expected_ids = {
            str(metadata.get("instance_id") or "").strip()
            for metadata in rows
        }
        expected_ids.discard("")

        manifests = sorted(artifact_dir.glob("*/*/manifest.json"))
        trials_by_instance: defaultdict[str, int] = defaultdict(int)
        official_parsers: Counter[str] = Counter()
        resolved_count = 0
        for manifest_path in manifests:
            trial_dir = manifest_path.parent
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                fail("invalid_manifest", f"{manifest_path}: {exc}")
                continue

            instance_id = str(manifest.get("instance_id") or "").strip()
            trials_by_instance[instance_id] += 1
            if instance_id not in expected_ids:
                fail("unknown_instance", f"{manifest_path}: {instance_id!r}")
            if manifest.get("evaluation_complete") is not True:
                fail("incomplete_evaluation", str(manifest_path))

            for filename in ("trajectory.jsonl", "model.patch", "eval_result.json"):
                if not (trial_dir / filename).is_file():
                    fail("missing_file", f"{trial_dir / filename}")

            eval_path = trial_dir / "eval_result.json"
            if not eval_path.is_file():
                continue
            try:
                result = json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception as exc:
                fail("invalid_eval_result", f"{eval_path}: {exc}")
                continue

            details = result.get("details")
            if not isinstance(details, dict):
                fail("missing_eval_details", str(eval_path))
                continue
            if details.get("mode") != "swebench":
                fail("wrong_eval_mode", f"{eval_path}: {details.get('mode')!r}")
            parser_name = str(details.get("parser") or "").strip()
            if not parser_name:
                fail("missing_parser", str(eval_path))
            elif parser_name == "pytest_robust" or "+pytest_robust" in parser_name:
                fail("fallback_parser", f"{eval_path}: {parser_name}")
            else:
                official_parsers[parser_name] += 1
            if details.get("swebench_version") != required_version:
                fail(
                    "wrong_swebench_version",
                    f"{eval_path}: {details.get('swebench_version')!r}",
                )
            resolved_count += int(bool(result.get("resolved")))

        if len(manifests) != expected_trials:
            fail("trial_count", f"manifests={len(manifests)}, expected={expected_trials}")
        for instance_id in sorted(expected_ids):
            actual = trials_by_instance.get(instance_id, 0)
            if actual != samples_per_task:
                fail(
                    "instance_trial_count",
                    f"{instance_id}: trials={actual}, expected={samples_per_task}",
                )

        payload.update(
            {
                "installed_swebench_version": installed,
                "manifest_count": len(manifests),
                "instances_with_artifacts": len(
                    {key for key in trials_by_instance if key}
                ),
                "resolved_count": resolved_count,
                "sample_resolved_rate": (
                    resolved_count / expected_trials if expected_trials else 0.0
                ),
                "pass_at_1": (
                    resolved_count / expected_tasks
                    if samples_per_task == 1 and expected_tasks
                    else None
                ),
                "official_parsers": dict(sorted(official_parsers.items())),
                "error_count": sum(error_kinds.values()),
                "error_kinds": dict(sorted(error_kinds.items())),
                "errors": errors,
            }
        )
        if not error_kinds:
            payload["status"] = "passed"
            rc = 0
        else:
            rc = 1
    except Exception as exc:
        fail("postflight_exception", f"{type(exc).__name__}: {exc}")
        payload.update(
            {
                "error_count": sum(error_kinds.values()),
                "error_kinds": dict(sorted(error_kinds.items())),
                "errors": errors,
            }
        )
        rc = 1
    _atomic_json(report, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return rc


def run_repair_dataset(
    *,
    dataset: Path,
    artifact_dir: Path,
    output_dataset: Path,
    report: Path,
    required_version: str,
) -> int:
    """Write the pass@1 rows that do not yet have one complete official result."""
    records = _load_records(dataset)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        metadata = record.get("extra_info") or record.get("metadata") or {}
        instance_id = str(metadata.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError("dataset contains an empty instance_id")
        if instance_id in rows_by_id:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        rows_by_id[instance_id] = record

    complete_ids: set[str] = set()
    for manifest_path in sorted(artifact_dir.glob("*/*/manifest.json")):
        trial_dir = manifest_path.parent
        eval_path = trial_dir / "eval_result.json"
        if not eval_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        details = result.get("details")
        instance_id = str(manifest.get("instance_id") or "").strip()
        if (
            manifest.get("evaluation_complete") is True
            and instance_id in rows_by_id
            and isinstance(details, dict)
            and details.get("mode") == "swebench"
            and bool(str(details.get("parser") or "").strip())
            and details.get("swebench_version") == required_version
        ):
            complete_ids.add(instance_id)

    missing_ids = sorted(set(rows_by_id) - complete_ids)
    _atomic_jsonl(output_dataset, [rows_by_id[instance_id] for instance_id in missing_ids])
    payload = {
        "check": "swebench_repair_dataset",
        "dataset": str(dataset),
        "artifact_dir": str(artifact_dir),
        "output_dataset": str(output_dataset),
        "required_swebench_version": required_version,
        "task_count": len(rows_by_id),
        "complete_count": len(complete_ids),
        "missing_count": len(missing_ids),
        "missing_instance_ids": missing_ids,
        "status": "passed",
    }
    _atomic_json(report, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "postflight"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--report", type=Path, required=True)
        sub.add_argument("--required-version", required=True)
        sub.add_argument("--expected-tasks", type=int, required=True)
        if command == "postflight":
            sub.add_argument("--artifact-dir", type=Path, required=True)
            sub.add_argument("--samples-per-task", type=int, required=True)
    repair = subparsers.add_parser("repair-dataset")
    repair.add_argument("--dataset", type=Path, required=True)
    repair.add_argument("--artifact-dir", type=Path, required=True)
    repair.add_argument("--output-dataset", type=Path, required=True)
    repair.add_argument("--report", type=Path, required=True)
    repair.add_argument("--required-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        return run_preflight(
            dataset=args.dataset,
            report=args.report,
            required_version=args.required_version,
            expected_tasks=args.expected_tasks,
        )
    if args.command == "repair-dataset":
        return run_repair_dataset(
            dataset=args.dataset,
            artifact_dir=args.artifact_dir,
            output_dataset=args.output_dataset,
            report=args.report,
            required_version=args.required_version,
        )
    return run_postflight(
        dataset=args.dataset,
        artifact_dir=args.artifact_dir,
        report=args.report,
        required_version=args.required_version,
        expected_tasks=args.expected_tasks,
        samples_per_task=args.samples_per_task,
    )


if __name__ == "__main__":
    sys.exit(main())
