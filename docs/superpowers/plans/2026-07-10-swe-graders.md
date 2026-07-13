# SWE Graders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From official SWE metadata (no pre-baked `eval_cmd`), resolve an eval plan, run tests in the sandbox, parse logs into F2P/P2P pass ratios, and emit `resolved` for the binary reward.

**Architecture:** Three layers under `examples/claudecode_ags/swe_eval/`: `resolve_eval_plan` → shared apply/run in `dispatch` → mode-specific `grade_logs`. `generate._evaluate_diff` calls workspace init (light scrub) then `dispatch.evaluate`. Thresholds only via `SLIME_CC_REWARD_*`.

**Tech Stack:** Python asyncio sandbox (`FakeSandbox` in unit tests), pytest log parsing, optional `swebench` / `SLIME_REBENCH_ROOT` parsers.

**Spec:** [2026-07-10-swe-graders-design.md](../specs/2026-07-10-swe-graders-design.md)

---

## File map

| File | Responsibility |
|------|----------------|
| `swe_eval/grade_common.py` | `parse_list`, thresholds, pass ratios, `resolved_from_*` |
| `swe_eval/log_parse.py` | ANSI strip + pytest status map |
| `swe_eval/cmd_resolve.py` | `EvalMode` / `EvalPlan` / `detect_eval_mode` / `resolve_eval_plan` |
| `swe_eval/swebench.py` | SWE-bench-family `grade_logs` |
| `swe_eval/scaleswe.py` | Scale-SWE `grade_logs` |
| `swe_eval/rebench.py` | rebench `grade_logs` (+ optional external parser) |
| `swe_eval/dispatch.py` | apply patches → run → grade → `EvalResult` |
| `swe_eval/simple_cmd.py` | exit-code fallback (unchanged semantics) |
| `generate.py` | `_evaluate_diff` → init + dispatch; preserve grading metadata |
| `tests/claudecode_ags/test_swe_eval_*.py` | unit coverage |

---

### Task 1: grade_common + log_parse

**Files:**
- Create: `examples/claudecode_ags/swe_eval/grade_common.py`
- Create: `examples/claudecode_ags/swe_eval/log_parse.py`
- Test: `tests/claudecode_ags/test_swe_eval_grade_common.py`

- [x] **Step 1:** Implement `parse_list`, `f2p_threshold` / `p2p_threshold`, `bucket_report`, `resolved_from_report` (defaults 1.0 / 0.99).
- [x] **Step 2:** Implement `normalize_log` + `parse_pytest_log`.
- [x] **Step 3:** Tests for threshold truth table and pytest line shapes.

### Task 2: cmd_resolve

**Files:**
- Create: `examples/claudecode_ags/swe_eval/cmd_resolve.py`
- Test: `tests/claudecode_ags/test_swe_eval_cmd_resolve.py`

- [x] **Step 1:** Mode priority: scaleswe → rebench → swebench → simple_cmd → none.
- [x] **Step 2:** Build `eval_cmd` (env activation + pytest / `test_cmd`).
- [x] **Step 3:** Tests: each mode’s metadata → expected mode + command fragments.

### Task 3: graders + dispatch

**Files:**
- Create: `swe_eval/{swebench,scaleswe,rebench,dispatch}.py`
- Test: `tests/claudecode_ags/test_swe_eval_graders.py`, `test_swe_eval_dispatch.py`

- [x] **Step 1:** `grade_logs` per mode → `resolved` + report.
- [x] **Step 2:** `dispatch.evaluate`: model/test/f2p patches, script run, grade.
- [x] **Step 3:** Fixture stdout → resolved; FakeSandbox dispatch smoke.

### Task 4: wire generate + regression

**Files:**
- Modify: `examples/claudecode_ags/generate.py`
- Modify: `examples/claudecode_ags/swe_eval/__init__.py`
- Modify: `examples/claudecode_ags/CHECKLIST.md`

- [x] **Step 1:** `_evaluate_diff` → workspace init + `dispatch.evaluate`.
- [x] **Step 2:** Pass `{**sample.metadata, **md}` so official F2P/P2P/repo fields survive `_parse_metadata`.
- [x] **Step 3:** `pytest tests/claudecode_ags/ -v` all green; update CHECKLIST.

---

## Success criteria

1. Three modes produce `EvalPlan` + `resolved` from official fields.
2. Binary reward still maps `resolved` ↔ `{0,1}`; thresholds `SLIME_*` only.
3. `simple_cmd` still works when only `eval_cmd` is present.
4. No live AGS required for unit tests.
