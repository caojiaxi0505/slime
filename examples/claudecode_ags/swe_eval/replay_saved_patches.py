#!/usr/bin/env python3
"""Replay saved SWE-bench patches without rerunning the coding agent.

This utility is intentionally small and explicit:

* load the normalized SWE-bench dataset metadata;
* load saved ``model.patch`` files from an eval artifact directory;
* rebuild a fresh test sandbox; and
* run the current SWE evaluator on those saved patches.

It is useful for repairing eval-only infrastructure failures such as missing
``eval_result.json`` files or evaluator command timeouts.  It does not resample
the model and therefore does not change agent behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

from examples.claudecode_ags.agent_runtime import _testbed_conda_activation_script
from examples.claudecode_ags.swe_eval import dispatch as swe_eval_dispatch
from examples.claudecode_ags.swe_eval.base import EvalResult
from examples.claudecode_ags.swe_eval.cmd_resolve import EvalMode, resolve_eval_plan
from examples.claudecode_ags.swe_eval.official import require_swebench_version
from examples.claudecode_ags.workspace_init import initialize_task_workspace, task_fields_from_metadata
from slime.agent.sandbox import make_sandbox
from slime.agent.sandbox_ags import _is_transient_request_error

logger = logging.getLogger(__name__)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = {**dict(row.get("metadata") or {}), **dict(row.get("extra_info") or {})}
            instance_id = str(metadata.get("instance_id") or row.get("label") or "")
            if not instance_id:
                raise ValueError(f"{path}:{line_no}: missing instance_id")
            if instance_id in rows:
                raise ValueError(f"{path}:{line_no}: duplicate instance_id {instance_id}")
            rows[instance_id] = metadata
    return rows


def _read_instances(args: argparse.Namespace) -> set[str] | None:
    instances = {str(x).strip() for x in (args.instance_id or []) if str(x).strip()}
    if args.instances_file:
        for line in args.instances_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                instances.add(line)
    return instances or None


def _discover_trials(artifact_root: Path, selected: set[str] | None) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for manifest_path in sorted(artifact_root.glob("*/*/manifest.json")):
        trial_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        instance_id = str(manifest.get("instance_id") or "")
        if not instance_id:
            raise ValueError(f"{manifest_path}: missing instance_id")
        if selected is None and (trial_dir / "eval_result.json").exists():
            continue
        if selected is not None and instance_id not in selected:
            continue
        patch_path = trial_dir / "model.patch"
        if not patch_path.is_file():
            raise FileNotFoundError(f"{instance_id}: missing saved patch {patch_path}")
        trials.append(
            {
                "instance_id": instance_id,
                "trial_id": trial_dir.name,
                "trial_dir": str(trial_dir),
                "patch_path": str(patch_path),
                "manifest": manifest,
                "had_eval_result": (trial_dir / "eval_result.json").exists(),
            }
        )
    if selected is not None:
        found = {str(t["instance_id"]) for t in trials}
        missing = sorted(selected - found)
        if missing:
            raise RuntimeError(f"selected instances not found in artifacts: {missing}")
    return trials


async def _evaluate_once_with_testbed_env(
    *,
    metadata: dict[str, Any],
    diff_text: str,
    timeout_sec: int,
) -> EvalResult:
    fields = task_fields_from_metadata(metadata)
    async with make_sandbox(str(metadata["image"])) as sandbox:
        initialized = await initialize_task_workspace(
            sandbox,
            fields,
            rollout_side=False,
        )
        if not initialized:
            return EvalResult(
                resolved=False,
                applied_cleanly=False,
                details={"reason": "eval_workspace_init_failed"},
            )
        plan = resolve_eval_plan(metadata)
        plan.eval_cmd = _testbed_conda_activation_script() + plan.eval_cmd
        return await swe_eval_dispatch.evaluate(
            sandbox,
            metadata=metadata,
            diff_text=diff_text,
            timeout_sec=timeout_sec,
            plan=plan,
        )


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    installed_version = require_swebench_version(args.swebench_version)
    dataset = _load_dataset(args.dataset)
    selected = _read_instances(args)
    trials = _discover_trials(args.artifact_root, selected)
    if args.expected is not None and len(trials) != args.expected:
        raise RuntimeError(f"expected {args.expected} trials, discovered {len(trials)}")

    for trial in trials:
        instance_id = str(trial["instance_id"])
        metadata = dataset.get(instance_id)
        if metadata is None:
            raise KeyError(f"{instance_id}: missing from dataset")
        plan = resolve_eval_plan(metadata)
        if plan.mode != EvalMode.SWEBENCH or plan.official_test_spec is None:
            raise RuntimeError(f"{instance_id}: expected strict SWE-bench plan, got {plan.mode.value}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    concurrency = max(1, int(os.environ.get("SLIME_CC_EVAL_CONCURRENCY") or str(args.concurrency)))
    retries = max(0, int(os.environ.get("SLIME_CC_EVAL_INFRA_RETRIES") or str(args.infra_retries)))
    guard_sec = max(args.timeout_sec, int(os.environ.get("SLIME_CC_EVAL_GUARD_SEC") or str(args.timeout_sec + 180)))
    semaphore = asyncio.Semaphore(concurrency)
    status_lock = asyncio.Lock()
    states: dict[str, dict[str, Any]] = {}

    def output_name(trial: dict[str, Any]) -> str:
        instance_id = str(trial["instance_id"])
        same_instance = sum(1 for x in trials if str(x["instance_id"]) == instance_id)
        if same_instance <= 1:
            return f"{instance_id}.json"
        return f"{instance_id}__{trial['trial_id']}.json"

    async def write_summary() -> None:
        completed = [x for x in states.values() if x.get("status") == "completed"]
        timeouts = [x for x in states.values() if x.get("status") == "timeout"]
        errors = [x for x in states.values() if x.get("status") == "error"]
        resolved = [x for x in completed if x.get("resolved") is True]
        _atomic_json(
            args.output_dir / "summary.json",
            {
                "updated_at_unix": time.time(),
                "swebench_version": installed_version,
                "timeout_sec": args.timeout_sec,
                "eval_concurrency": concurrency,
                "infra_retries": retries,
                "guard_sec": guard_sec,
                "total": len(trials),
                "pending": sum(1 for x in states.values() if x.get("status") == "pending"),
                "running": sum(1 for x in states.values() if x.get("status") == "running"),
                "completed": len(completed),
                "timeouts": len(timeouts),
                "errors": len(errors),
                "resolved": len(resolved),
                "unresolved": len(completed) - len(resolved),
                "tasks": {key: states[key] for key in sorted(states)},
            },
        )

    async def replay_eval(*, metadata: dict[str, Any], diff_text: str) -> EvalResult:
        queued_at = time.time()
        async with semaphore:
            queue_wait = time.time() - queued_at
            for attempt in range(retries + 1):
                try:
                    async with asyncio.timeout(guard_sec):
                        result = await _evaluate_once_with_testbed_env(
                            metadata=metadata,
                            diff_text=diff_text,
                            timeout_sec=args.timeout_sec,
                        )
                    result.details["eval_queue_wait_sec"] = queue_wait
                    result.details["testbed_env_explicitly_activated"] = True
                    return result
                except Exception as exc:
                    retryable = isinstance(exc, asyncio.TimeoutError) or _is_transient_request_error(exc)
                    if not retryable or attempt >= retries:
                        raise
                    logger.warning(
                        "replay infra failure; retry %d/%d in fresh sandbox: %s",
                        attempt + 1,
                        retries,
                        exc,
                    )
            raise AssertionError("unreachable")

    for trial in trials:
        output_path = results_dir / output_name(trial)
        instance_id = str(trial["instance_id"])
        if output_path.exists() and not args.rerun:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            states[instance_id] = {
                "status": existing.get("status"),
                "resolved": existing.get("resolved"),
                "applied_cleanly": existing.get("applied_cleanly"),
                "elapsed_sec": existing.get("elapsed_sec"),
                "skipped_existing": True,
            }
        else:
            states[instance_id] = {"status": "pending"}
    await write_summary()

    async def run_one(trial: dict[str, Any]) -> None:
        instance_id = str(trial["instance_id"])
        output_path = results_dir / output_name(trial)
        if states[instance_id].get("status") in {"completed", "timeout"} and not args.rerun:
            logger.info("skip existing terminal replay: %s", instance_id)
            return
        metadata = dataset[instance_id]
        patch_text = Path(str(trial["patch_path"])).read_text(encoding="utf-8")
        started = time.time()
        async with status_lock:
            states[instance_id] = {
                "status": "running",
                "started_at_unix": started,
                "had_eval_result": trial["had_eval_result"],
            }
            await write_summary()
        logger.info("replay start: %s patch_chars=%d", instance_id, len(patch_text))
        try:
            result = await replay_eval(metadata=metadata, diff_text=patch_text)
            elapsed = time.time() - started
            payload = {
                "status": "completed",
                "instance_id": instance_id,
                "eval_mode": "swebench",
                "resolved": bool(result.resolved),
                "applied_cleanly": bool(result.applied_cleanly),
                "elapsed_sec": elapsed,
                "timeout_sec": args.timeout_sec,
                "patch_chars": len(patch_text),
                "patch_sha256": hashlib.sha256(patch_text.encode("utf-8")).hexdigest(),
                "source_trial_dir": trial["trial_dir"],
                "source_had_eval_result": trial["had_eval_result"],
                "details": result.details,
            }
            _atomic_json(output_path, payload)
            state = {
                "status": "completed",
                "resolved": payload["resolved"],
                "applied_cleanly": payload["applied_cleanly"],
                "elapsed_sec": elapsed,
                "had_eval_result": trial["had_eval_result"],
            }
            logger.info(
                "replay done: %s resolved=%s applied=%s elapsed=%.1fs",
                instance_id,
                payload["resolved"],
                payload["applied_cleanly"],
                elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - started
            payload = {
                "status": "error",
                "instance_id": instance_id,
                "eval_mode": "swebench",
                "elapsed_sec": elapsed,
                "timeout_sec": args.timeout_sec,
                "patch_chars": len(patch_text),
                "patch_sha256": hashlib.sha256(patch_text.encode("utf-8")).hexdigest(),
                "source_trial_dir": trial["trial_dir"],
                "source_had_eval_result": trial["had_eval_result"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(output_path, payload)
            state = {
                "status": "error",
                "elapsed_sec": elapsed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "had_eval_result": trial["had_eval_result"],
            }
            logger.exception("replay failed: %s", instance_id)
        async with status_lock:
            states[instance_id] = state
            await write_summary()

    await asyncio.gather(*(run_one(trial) for trial in trials))
    await write_summary()
    return 1 if any(x.get("status") == "error" for x in states.values()) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--instances-file", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--timeout-sec", type=int, default=2700)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--infra-retries", type=int, default=0)
    parser.add_argument("--swebench-version", default="4.1.0")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
