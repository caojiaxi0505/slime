# SWE-Gym Gold Filter + Pass@k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a batch runner that (1) gold-evals SWE-Gym tasks ×4 and keeps only all-pass, then (2) runs Claude Code ×8 and drops only always-resolved tasks into `train_candidates.jsonl`.

**Architecture:** Pure filter/summary helpers + JSONL resume store + thin attempt runners that call the same primitives as L1/L2 (`initialize_task_workspace`, `swe_eval.dispatch`, `agent_runtime`, artifact export). CLI `python -m examples.claudecode_ags.eval.swegym_filter`.

**Tech Stack:** Python asyncio, existing `examples/claudecode_ags` + AGS sandbox, pyarrow for parquet.

**Spec:** `docs/superpowers/specs/2026-07-11-swegym-gold-filter-passk-design.md`

---

## File map

| Path | Role |
|------|------|
| `examples/claudecode_ags/eval/__init__.py` | Package marker |
| `examples/claudecode_ags/eval/filter_logic.py` | Gold keep / passk drop-always-resolved / task summary |
| `examples/claudecode_ags/eval/run_store.py` | Out-dir layout, append JSONL, resume keys, rewrite aggregates |
| `examples/claudecode_ags/eval/attempts.py` | `run_gold_attempt`, `run_passk_attempt` |
| `examples/claudecode_ags/eval/swegym_filter.py` | CLI + concurrency scheduler |
| `tests/claudecode_ags/test_swegym_filter_logic.py` | Offline unit tests |
| `docs/superpowers/runbooks/2026-07-11-swegym-gold-filter-passk-ops.md` | Operator commands |

Defaults: `--gold-repeats 4`, `--passk-repeats 8`, `--time-budget 1800`, `--eval-timeout 600`, `--concurrency 1`.

---

### Task 1: Filter logic (TDD)

**Files:**
- Create: `examples/claudecode_ags/eval/filter_logic.py`
- Test: `tests/claudecode_ags/test_swegym_filter_logic.py`

- [ ] **Step 1: Failing tests for gold + passk decisions**

```python
from examples.claudecode_ags.eval.filter_logic import (
    decide_gold_task,
    decide_passk_task,
    summarize_passk_runs,
)

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
    partial = summarize_passk_runs(
        [{"resolved": True, "infra_ok": True, "diff_chars": 1}] * 3
        + [{"resolved": False, "infra_ok": True, "diff_chars": 0}] * 5
    )
    assert decide_passk_task(partial, expected_repeats=8)["keep"] is True
    assert partial["n_resolved"] == 3
```

- [ ] **Step 2: Implement `filter_logic.py`**

```python
def decide_gold_task(runs: list[dict], *, expected_repeats: int) -> dict:
    if any(not r.get("infra_ok", True) for r in runs):
        return {"keep": False, "exclude_reason": "infra"}
    if any(not r.get("resolved") for r in runs):
        return {"keep": False, "exclude_reason": "gold_unresolved"}
    if len(runs) < expected_repeats:
        return {"keep": False, "exclude_reason": "incomplete"}
    return {"keep": True, "exclude_reason": None}

def summarize_passk_runs(runs: list[dict]) -> dict:
    n = len(runs)
    n_resolved = sum(1 for r in runs if r.get("resolved"))
    n_infra_fail = sum(1 for r in runs if not r.get("infra_ok", True))
    n_nonempty = sum(1 for r in runs if int(r.get("diff_chars") or 0) > 0)
    return {
        "n_runs": n,
        "n_resolved": n_resolved,
        "pass_rate": (n_resolved / n) if n else 0.0,
        "pass_at_k": n_resolved >= 1,
        "nonempty_diff_rate": (n_nonempty / n) if n else 0.0,
        "infra_fail_count": n_infra_fail,
    }

def decide_passk_task(summary: dict, *, expected_repeats: int) -> dict:
    if int(summary.get("n_runs") or 0) < expected_repeats:
        return {"keep": False, "exclude_reason": "incomplete"}
    if int(summary["n_resolved"]) == expected_repeats:
        return {"keep": False, "exclude_reason": "always_resolved"}
    return {"keep": True, "exclude_reason": None}
```

- [ ] **Step 3: `pytest tests/claudecode_ags/test_swegym_filter_logic.py -q` → pass**

---

### Task 2: Run store + resume

**Files:**
- Create: `examples/claudecode_ags/eval/run_store.py`
- Test: extend `test_swegym_filter_logic.py` or `test_swegym_run_store.py`

- [ ] **Step 1: Implement store**

API:
- `RunStore(out_dir)` creates dirs
- `load_completed_keys(phase) -> set[(instance_id, repeat_idx)]` from `runs.jsonl`
- `append_run(row)` append-only
- `write_jsonl(name, rows)` rewrite aggregates (`kept.jsonl`, `excluded.jsonl`, `tasks.jsonl`, `train_candidates.jsonl`, `excluded_passk.jsonl`)
- `write_meta(dict)` / `write_summary(dict)`

- [ ] **Step 2: Test resume key parsing with temp dir**

---

### Task 3: Attempt runners

**Files:**
- Create: `examples/claudecode_ags/eval/attempts.py`
- Reuse: smoke helpers (`prepare_metadata`, `_build_claude_env`, `export_rollout_artifacts`, `_adapter_public_url`) via import from `ags_smoke` or small shared moves

- [ ] **Step 1: `run_gold_attempt(md, *, eval_timeout) -> dict`**

Returns keys: `resolved`, `applied_cleanly`, `infra_ok`, `timeout_hit`, `elapsed_sec`, `details`, `error`.

On exception / init fail: `infra_ok=False`, `resolved=False`.

- [ ] **Step 2: `run_passk_attempt(md, *, time_budget, eval_timeout, artifact_dir, skip_health_gate) -> dict`**

Same shape + `diff_chars`, `claude_exit`, `artifact_dir`.  
`timeout_hit=True` if `claude_exit == EXIT_TIME_BUDGET_EXCEEDED` (-1).

Import Claude env builder from `ags_smoke` to avoid drift.

---

### Task 4: CLI scheduler

**Files:**
- Create: `examples/claudecode_ags/eval/swegym_filter.py`
- Create: `examples/claudecode_ags/eval/__main__.py` → `swegym_filter.main`

- [ ] **Step 1: argparse**

`--phase {gold,passk}`, `--data-path`, `--kept-jsonl`, `--dataset-type` (default `swegym`), `--gold-repeats`, `--passk-repeats`, `--time-budget`, `--eval-timeout`, `--out-dir`, `--limit`, `--instance-id` (repeatable), `--concurrency`, `--env-file`, `--skip-health-gate`, `--save-artifacts` (default on for passk).

- [ ] **Step 2: Phase gold loop**

Load parquet rows → normalize → for each task up to `gold-repeats` with resume; early-stop task on first fail; then `decide_gold_task` → kept/excluded.

- [ ] **Step 3: Phase passk loop**

Load `kept.jsonl` → 8 attempts with resume → `summarize` + `decide_passk_task` → `train_candidates` / `excluded_passk` → `summary.json`.

- [ ] **Step 4: Concurrency via `asyncio.Semaphore`**

Default 1. Schedule attempt-level or task-level; prefer **task-level** semaphore (simpler, one task’s repeats sequential for early-stop).

---

### Task 5: Docs + checklist

**Files:**
- Create: `docs/superpowers/runbooks/2026-07-11-swegym-gold-filter-passk-ops.md`
- Modify: `examples/claudecode_ags/CHECKLIST.md` (partial row)
- Mirror runbook to `/mnt/sn-007/jiaxicao/code/slime/docs/superpowers/runbooks/`

- [ ] **Step 1: Ops commands for limit=20 gold then passk**
- [ ] **Step 2: Note timeouts and AGS lifetime**

---

### Task 6: Verify offline

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m pytest tests/claudecode_ags/test_swegym_filter_logic.py tests/claudecode_ags/test_swegym_run_store.py -q
.venv/bin/python -m examples.claudecode_ags.eval.swegym_filter --help
```

Live AGS gold/passk is operator-owned (not required to finish this plan).

---

## Spec coverage check

| Spec item | Task |
|-----------|------|
| Gold ×4 all-pass keep | 1, 4 |
| Infra fail excludes gold | 1, 3 |
| Passk ×8 drop only 8/8 | 1, 4 |
| pass@k report only | 1, 4 |
| Timeouts 1800/600 + env | 3, 4, 5 |
| Resume / limit / concurrency | 2, 4 |
| Artifacts | 3, 4 |
| Offline tests | 1, 2, 6 |
