#!/usr/bin/env python3
"""SWE-Gym gold×N filter then Claude Code×K (drop always-resolved only).

Examples::

  python -m examples.claudecode_ags.eval.swegym_filter --phase gold \\
    --data-path /mnt/sn-007/jiaxicao/datasets/SWE-Gym/data/train-00000-of-00001.parquet \\
    --out-dir /tmp/swegym_gold --limit 20 --gold-repeats 4

  python -m examples.claudecode_ags.eval.swegym_filter --phase passk \\
    --kept-jsonl /tmp/swegym_gold/kept.jsonl \\
    --out-dir /tmp/swegym_passk --passk-repeats 8 --time-budget 1800
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.claudecode_ags.eval.attempts import run_gold_attempt, run_passk_attempt
from examples.claudecode_ags.eval.filter_logic import (
    decide_gold_task,
    decide_passk_task,
    summarize_passk_runs,
)
from examples.claudecode_ags.eval.run_store import RunStore, read_jsonl
from examples.claudecode_ags.smoke import ags_smoke

# Avoid killing the whole batch when stdout/tee pipe breaks under heavy AGS logs.
try:
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except (AttributeError, ValueError):
    pass


def _log(msg: str) -> None:
    line = f"[swegym-filter] {msg}"
    try:
        print(line, flush=True)
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
    except OSError:
        pass
    # Always mirror to out-dir progress log when configured.
    log_path = os.environ.get("SLIME_SWEGYM_FILTER_PROGRESS_LOG", "").strip()
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _normalize_gather_result(res: Any, *, md: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(res, dict):
        return res
    iid = str((md or {}).get("instance_id") or "")
    if isinstance(res, BaseException):
        return {
            "instance_id": iid,
            "metadata": md,
            "keep": False,
            "exclude_reason": "infra",
            "n_runs": 0,
            "error": f"{type(res).__name__}: {res}",
        }
    return {
        "instance_id": iid,
        "metadata": md,
        "keep": False,
        "exclude_reason": "infra",
        "n_runs": 0,
        "error": f"unexpected_gather_result:{type(res)!r}",
    }


def _load_rows(
    *,
    data_path: str | None,
    kept_jsonl: str | None,
    dataset_type: str,
    instance_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    id_set = {x.strip() for x in instance_ids if x.strip()}
    rows: list[dict[str, Any]] = []

    if kept_jsonl:
        for row in read_jsonl(Path(kept_jsonl)):
            iid = str(row.get("instance_id") or "").strip()
            if id_set and iid not in id_set:
                continue
            # Prefer embedded metadata if present
            md = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
            rows.append(dict(md))
    elif data_path:
        path = Path(data_path)
        if path.suffix == ".parquet":
            # Load all then filter (SWE-Gym train is ~2.4k — fine)
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            cols = table.column_names
            for i in range(table.num_rows):
                raw = {name: table.column(name)[i].as_py() for name in cols}
                iid = str(raw.get("instance_id") or "").strip()
                if id_set and iid not in id_set:
                    continue
                rows.append(raw)
        else:
            # jsonl of raw rows
            for raw in read_jsonl(path):
                iid = str(raw.get("instance_id") or "").strip()
                if id_set and iid not in id_set:
                    continue
                rows.append(raw)
    else:
        raise RuntimeError("provide --data-path (gold) or --kept-jsonl (passk)")

    if limit > 0:
        rows = rows[:limit]

    out: list[dict[str, Any]] = []
    for raw in rows:
        md = ags_smoke.prepare_metadata(raw, dataset_type, None)
        out.append(md)
    return out


async def _phase_gold(
    tasks: list[dict[str, Any]],
    *,
    store: RunStore,
    gold_repeats: int,
    eval_timeout: int,
    concurrency: int,
) -> dict[str, Any]:
    done_keys = store.completed_keys("gold")
    sem = asyncio.Semaphore(max(1, concurrency))
    # Group existing runs by instance
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in store.load_runs(phase="gold"):
        by_id[str(r["instance_id"])].append(r)

    async def run_one_task(md: dict[str, Any]) -> dict[str, Any]:
        iid = str(md.get("instance_id") or "")
        try:
            async with sem:
                existing = list(by_id.get(iid, []))
                # Early stop if already failed
                decision_so_far = decide_gold_task(existing, expected_repeats=gold_repeats) if existing else None
                if decision_so_far and not decision_so_far["keep"] and decision_so_far["exclude_reason"] != "incomplete":
                    return {"instance_id": iid, "metadata": md, **decision_so_far, "n_runs": len(existing)}

                for idx in range(gold_repeats):
                    if (iid, idx) in done_keys:
                        continue
                    # Stop early if prior failure already recorded
                    if existing and decide_gold_task(existing, expected_repeats=gold_repeats)["exclude_reason"] in (
                        "infra",
                        "gold_unresolved",
                    ):
                        break
                    _log(f"gold {iid} r{idx}/{gold_repeats - 1}")
                    result = await run_gold_attempt(md, eval_timeout=eval_timeout)
                    row = {
                        "phase": "gold",
                        "instance_id": iid,
                        "repeat_idx": idx,
                        **result,
                    }
                    store.append_run(row)
                    existing.append(row)
                    done_keys.add((iid, idx))
                    by_id[iid] = existing
                    if not result.get("infra_ok") or not result.get("resolved"):
                        break

                decision = decide_gold_task(existing, expected_repeats=gold_repeats)
                return {"instance_id": iid, "metadata": md, **decision, "n_runs": len(existing)}
        except Exception as e:
            _log(f"gold task ERROR {iid}: {type(e).__name__}: {e}")
            return {
                "instance_id": iid,
                "metadata": md,
                "keep": False,
                "exclude_reason": "infra",
                "n_runs": len(by_id.get(iid, [])),
                "error": f"{type(e).__name__}: {e}",
            }

    raw_results = await asyncio.gather(*[run_one_task(md) for md in tasks], return_exceptions=True)
    results = [_normalize_gather_result(r, md=tasks[i] if i < len(tasks) else None) for i, r in enumerate(raw_results)]
    kept = []
    excluded = []
    task_rows = []
    for r in results:
        task_rows.append(r)
        entry = {
            "instance_id": r["instance_id"],
            "metadata": r.get("metadata"),
            "n_runs": r.get("n_runs"),
            "exclude_reason": r.get("exclude_reason"),
            "error": r.get("error"),
        }
        if r.get("keep"):
            kept.append({"instance_id": r["instance_id"], "metadata": r.get("metadata")})
        else:
            excluded.append(entry)

    store.write_named_jsonl("tasks.jsonl", task_rows)
    store.write_named_jsonl("kept.jsonl", kept)
    store.write_named_jsonl("excluded.jsonl", excluded)
    summary = {
        "phase": "gold",
        "n_tasks": len(tasks),
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "gold_repeats": gold_repeats,
        "eval_timeout": eval_timeout,
    }
    store.write_summary(summary)
    _log(f"gold done kept={len(kept)} excluded={len(excluded)}")
    return summary


async def _phase_passk(
    tasks: list[dict[str, Any]],
    *,
    store: RunStore,
    passk_repeats: int,
    time_budget: int,
    eval_timeout: int,
    concurrency: int,
    skip_health_gate: bool,
    save_artifacts: bool,
) -> dict[str, Any]:
    done_keys = store.completed_keys("passk")
    sem = asyncio.Semaphore(max(1, concurrency))
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in store.load_runs(phase="passk"):
        by_id[str(r["instance_id"])].append(r)

    async def run_one_task(md: dict[str, Any]) -> dict[str, Any]:
        iid = str(md.get("instance_id") or "")
        try:
            async with sem:
                existing = list(by_id.get(iid, []))
                for idx in range(passk_repeats):
                    if (iid, idx) in done_keys:
                        continue
                    _log(f"passk {iid} r{idx}/{passk_repeats - 1}")
                    art = store.artifact_dir(iid, idx) if save_artifacts else None
                    result = await run_passk_attempt(
                        md,
                        time_budget=time_budget,
                        eval_timeout=eval_timeout,
                        artifact_dir=art,
                        skip_health_gate=skip_health_gate,
                    )
                    row = {
                        "phase": "passk",
                        "instance_id": iid,
                        "repeat_idx": idx,
                        **result,
                    }
                    store.append_run(row)
                    existing.append(row)
                    done_keys.add((iid, idx))
                    by_id[iid] = existing

                summary = summarize_passk_runs(existing)
                decision = decide_passk_task(summary, expected_repeats=passk_repeats)
                return {
                    "instance_id": iid,
                    "metadata": md,
                    **summary,
                    **decision,
                }
        except Exception as e:
            _log(f"passk task ERROR {iid}: {type(e).__name__}: {e}")
            existing = list(by_id.get(iid, []))
            summary = summarize_passk_runs(existing)
            return {
                "instance_id": iid,
                "metadata": md,
                **summary,
                "keep": False,
                "exclude_reason": "infra",
                "error": f"{type(e).__name__}: {e}",
            }

    raw_results = await asyncio.gather(*[run_one_task(md) for md in tasks], return_exceptions=True)
    results = [_normalize_gather_result(r, md=tasks[i] if i < len(tasks) else None) for i, r in enumerate(raw_results)]
    candidates = []
    excluded = []
    task_rows = []
    n_resolved_hist: dict[str, int] = defaultdict(int)
    for r in results:
        task_rows.append(r)
        n_resolved_hist[str(r.get("n_resolved"))] += 1
        entry = {
            "instance_id": r["instance_id"],
            "metadata": r.get("metadata"),
            "n_resolved": r.get("n_resolved"),
            "n_runs": r.get("n_runs"),
            "pass_at_k": r.get("pass_at_k"),
            "pass_rate": r.get("pass_rate"),
            "nonempty_diff_rate": r.get("nonempty_diff_rate"),
            "exclude_reason": r.get("exclude_reason"),
            "error": r.get("error"),
        }
        if r.get("keep"):
            candidates.append(
                {
                    "instance_id": r["instance_id"],
                    "metadata": r.get("metadata"),
                    "n_resolved": r.get("n_resolved"),
                    "n_runs": r.get("n_runs"),
                    "pass_rate": r.get("pass_rate"),
                }
            )
        else:
            excluded.append(entry)

    store.write_named_jsonl("tasks.jsonl", task_rows)
    store.write_named_jsonl("train_candidates.jsonl", candidates)
    store.write_named_jsonl("excluded_passk.jsonl", excluded)
    summary = {
        "phase": "passk",
        "n_tasks": len(tasks),
        "n_train_candidates": len(candidates),
        "n_excluded_always_resolved": sum(
            1 for e in excluded if e.get("exclude_reason") == "always_resolved"
        ),
        "n_excluded_other": sum(1 for e in excluded if e.get("exclude_reason") != "always_resolved"),
        "n_resolved_histogram": dict(n_resolved_hist),
        "passk_repeats": passk_repeats,
        "time_budget": time_budget,
        "eval_timeout": eval_timeout,
    }
    store.write_summary(summary)
    _log(
        f"passk done candidates={len(candidates)} "
        f"always_resolved={summary['n_excluded_always_resolved']}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SWE-Gym gold filter + passk (drop always-resolved)")
    p.add_argument("--phase", choices=("gold", "passk"), required=True)
    p.add_argument("--data-path", default="", help="parquet/jsonl for gold phase")
    p.add_argument("--kept-jsonl", default="", help="Phase G kept.jsonl for passk phase")
    p.add_argument("--dataset-type", default="swegym")
    p.add_argument("--gold-repeats", type=int, default=4)
    p.add_argument("--passk-repeats", type=int, default=8)
    p.add_argument(
        "--time-budget",
        type=int,
        default=int(os.environ.get("SLIME_CC_TIME_BUDGET_SEC") or 1800),
        help="Claude Code wall clock seconds (Phase P)",
    )
    p.add_argument(
        "--eval-timeout",
        type=int,
        default=int(os.environ.get("SLIME_CC_EVAL_TIMEOUT_SEC") or 600),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="max tasks (0=all)")
    p.add_argument("--instance-id", action="append", default=[], help="repeatable id filter")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--env-file", default="", help="optional KEY=VALUE env file")
    p.add_argument("--skip-health-gate", action="store_true")
    p.add_argument("--no-save-artifacts", action="store_true", help="Phase P: skip trajectory export")
    return p


def _redirect_stdio_to_runner_log(out_dir: Path) -> None:
    """Avoid BrokenPipe from `python | tee` under concurrent AGS/rich logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = out_dir / "runner.log"
    # Line-buffered append; keep FD alive for process lifetime.
    f = open(runner, "a", encoding="utf-8", buffering=1)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    sys.stdout = f
    sys.stderr = f


async def async_main(args: argparse.Namespace) -> int:
    if args.env_file:
        ags_smoke.load_env_file(args.env_file, override=False)
    os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "ags")

    store = RunStore(args.out_dir)
    _redirect_stdio_to_runner_log(store.out_dir)
    progress_log = store.out_dir / "progress.log"
    os.environ["SLIME_SWEGYM_FILTER_PROGRESS_LOG"] = str(progress_log)
    meta = {
        "phase": args.phase,
        "dataset_type": args.dataset_type,
        "data_path": args.data_path or None,
        "kept_jsonl": args.kept_jsonl or None,
        "gold_repeats": args.gold_repeats,
        "passk_repeats": args.passk_repeats,
        "time_budget": args.time_budget,
        "eval_timeout": args.eval_timeout,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "instance_ids": args.instance_id,
        "ags_timeout": os.environ.get("SLIME_AGENT_AGS_TIMEOUT"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_timeout_ms": os.environ.get("API_TIMEOUT_MS"),
        "bash_default_timeout_ms": os.environ.get("BASH_DEFAULT_TIMEOUT_MS"),
        "bash_max_timeout_ms": os.environ.get("BASH_MAX_TIMEOUT_MS"),
    }
    store.write_meta(meta)

    tasks = _load_rows(
        data_path=args.data_path or None,
        kept_jsonl=args.kept_jsonl or None,
        dataset_type=args.dataset_type,
        instance_ids=list(args.instance_id or []),
        limit=int(args.limit or 0),
    )
    _log(f"phase={args.phase} n_tasks={len(tasks)} out={store.out_dir}")
    if not tasks:
        _log("no tasks loaded")
        return 2

    if args.phase == "gold":
        if not args.data_path and not args.kept_jsonl:
            # gold normally needs data-path; allow kept only if re-running? require data-path
            pass
        if not args.data_path:
            raise RuntimeError("gold phase requires --data-path")
        await _phase_gold(
            tasks,
            store=store,
            gold_repeats=args.gold_repeats,
            eval_timeout=args.eval_timeout,
            concurrency=args.concurrency,
        )
    else:
        if not args.kept_jsonl:
            raise RuntimeError("passk phase requires --kept-jsonl")
        await _phase_passk(
            tasks,
            store=store,
            passk_repeats=args.passk_repeats,
            time_budget=args.time_budget,
            eval_timeout=args.eval_timeout,
            concurrency=args.concurrency,
            skip_health_gate=args.skip_health_gate,
            save_artifacts=not args.no_save_artifacts,
        )

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store.write_meta(meta)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except Exception as e:
        _log(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
