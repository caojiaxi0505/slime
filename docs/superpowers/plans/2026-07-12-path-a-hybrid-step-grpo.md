# Path A Hybrid Step-GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a self-contained Path A `step_reconstruct` library + unit tests for hybrid step-GRPO (no slime-tencent dependency, no HyperPod launcher this round).

**Architecture:** Port/adapt tencent `step_reconstruct` into `examples/claudecode_ags/step_reconstruct/`, replace hybrid selection with global top-B patch-turn edit-PPL (B=min(K,available)), fork K continuations per point, train full continuation, no PPL_SKIP. Advantage tags vanilla vs branch. Tests use mocks (no live AGS).

**Tech Stack:** Python 3, pytest, asyncio, Path A `claudecode_ags` + `slime.utils.types.Sample`.

**Spec:** [2026-07-12-path-a-hybrid-step-grpo-design.md](../specs/2026-07-12-path-a-hybrid-step-grpo-design.md)

---

## File map

| Path | Responsibility |
|------|----------------|
| `examples/claudecode_ags/step_reconstruct/__init__.py` | Public exports |
| `examples/claudecode_ags/step_reconstruct/_common.py` | Diff normalize / changed_paths / SNAP paths |
| `examples/claudecode_ags/step_reconstruct/edit_ppl.py` | Per-patch-turn edit-PPL |
| `examples/claudecode_ags/step_reconstruct/selection.py` | Global top-B patch-turn selection |
| `examples/claudecode_ags/step_reconstruct/step_grpo_advantage.py` | `post_process_rewards` + `filter` |
| `examples/claudecode_ags/step_reconstruct/session_capture.py` | `SessionBundle` + Stage-1 capture hooks |
| `examples/claudecode_ags/step_reconstruct/workspace_rebuild.py` | rebuild + prefix re-seed + resume_and_run |
| `examples/claudecode_ags/step_reconstruct/hybrid_generate.py` | `hybrid_generate` orchestration |
| `tests/claudecode_ags/test_step_reconstruct_*.py` | Unit tests |

Do **not** modify `examples/claudecode_ags/generate.py` default path.

---

### Task 1: `_common` + `edit_ppl` + `selection` (pure)

**Files:**
- Create: `examples/claudecode_ags/step_reconstruct/__init__.py`
- Create: `examples/claudecode_ags/step_reconstruct/_common.py`
- Create: `examples/claudecode_ags/step_reconstruct/edit_ppl.py`
- Create: `examples/claudecode_ags/step_reconstruct/selection.py`
- Create: `tests/claudecode_ags/test_step_reconstruct_edit_ppl.py`
- Create: `tests/claudecode_ags/test_step_reconstruct_selection.py`

- [ ] **Step 1:** Implement `_common.normalize_diff` / `changed_paths` (port from tencent; no coding_agent_rl imports).

- [ ] **Step 2:** Implement `edit_ppl.iter_patch_turn_ppls(steps_diffs, turn_logprobs, clip) -> list[PatchTurnPPL]` where a patch turn is an index `i` whose normalized cumulative diff changed vs `i-1`; PPL = mean clipped `-log p` over that turn’s tokens. Fallback: if turn/step length mismatch, return empty list (caller may skip).

- [ ] **Step 3:** Implement `selection.select_top_patch_turns(candidates, k) -> list[SelectedTurn]` pooling candidates `(source_trial_idx, step_t, edit_ppl)`, sort by PPL desc, take `B=min(k, len)`.

- [ ] **Step 4:** Tests for top-K across trials, undersubscribe, empty pool, edit-PPL clip.

- [ ] **Step 5:** Run `pytest tests/claudecode_ags/test_step_reconstruct_edit_ppl.py tests/claudecode_ags/test_step_reconstruct_selection.py -v` → PASS.

---

### Task 2: `step_grpo_advantage`

**Files:**
- Create: `examples/claudecode_ags/step_reconstruct/step_grpo_advantage.py`
- Create: `tests/claudecode_ags/test_step_reconstruct_advantage.py`

- [ ] **Step 1:** Port tencent advantage with these Path A changes:
  - package docstring paths → `examples.claudecode_ags.step_reconstruct...`
  - `filter` only runs when `STEP_GRPO_FILTER` is truthy (default `1`); if `0`, no-op
  - `_step_group_key` prefers `metadata["step_group_key"]` (format `{group}:{source_trial}:{step_t}`)

- [ ] **Step 2:** Tests: vanilla group GRPO (sum segments per trial then normalize); branch per `step_group_key`; filter drops std=0; `STEP_GRPO_FILTER=0` keeps samples.

- [ ] **Step 3:** `pytest tests/claudecode_ags/test_step_reconstruct_advantage.py -v` → PASS.

---

### Task 3: `SessionBundle` + rebuild (mockable)

**Files:**
- Create: `examples/claudecode_ags/step_reconstruct/session_capture.py` (dataclass + load/save + hook install helpers; Stage-1 runner adapted to Path A `agent_runtime` / `make_sandbox` where possible)
- Create: `examples/claudecode_ags/step_reconstruct/workspace_rebuild.py`
- Create: `tests/claudecode_ags/test_step_reconstruct_rebuild.py`

- [ ] **Step 1:** Port `SessionBundle` / `StepRecord` persistence (bundle.json, step diffs). No slime-tencent imports.

- [ ] **Step 2:** `rebuilt_workspace(bundle, step_t)` context manager: fresh sandbox → Path A workspace prep → `git apply` step diff; expose `verify_rebuild` using `normalize_diff` / `changed_paths`.

- [ ] **Step 3:** `prefix_reseed_prompt(transcript_path, step_t)` / `resume_and_run` stubs that accept injectable runners for tests.

- [ ] **Step 4:** Mock-sandbox tests for apply + verify path equality; failing apply returns False without raising.

- [ ] **Step 5:** `pytest tests/claudecode_ags/test_step_reconstruct_rebuild.py -v` → PASS.

---

### Task 4: `hybrid_generate` orchestration

**Files:**
- Create: `examples/claudecode_ags/step_reconstruct/hybrid_generate.py`
- Create: `tests/claudecode_ags/test_step_reconstruct_hybrid_orchestration.py`

- [ ] **Step 1:** Implement `hybrid_generate`:
  - `evaluation=True` → delegate `examples.claudecode_ags.generate.generate`
  - else: internal `K=STEP_GRPO_HYBRID_K` vanilla trials (capture when available)
  - selection via Task 1
  - each selected turn → K branches via rebuild+reseed
  - metadata per spec; train scope = full continuation loss mask
  - no PPL_SKIP
  - degradations per spec §7

- [ ] **Step 2:** Unit test orchestration with **fully mocked** trial/branch runners (no AGS): all-solved → no branch; one fail → top turns selected; undersubscribe; eval delegates.

- [ ] **Step 3:** `pytest tests/claudecode_ags/test_step_reconstruct_hybrid_orchestration.py -v` → PASS.

---

### Task 5: Docs checklist + full pytest

**Files:**
- Modify: `examples/claudecode_ags/CHECKLIST.md` (add hybrid library row)
- Mirror plan/spec under `/mnt/sn-007/jiaxicao/code/slime/docs/superpowers/` if missing

- [ ] **Step 1:** Update CHECKLIST.
- [ ] **Step 2:** Run all `test_step_reconstruct_*.py` → PASS.
- [ ] **Step 3:** Confirm `rg slime-tencent examples/claudecode_ags/step_reconstruct` → empty.

---

## Spec coverage check

| Spec item | Task |
|-----------|------|
| Global top-B patch turns | T1 |
| edit-PPL | T1 |
| Advantage + filter | T2 |
| rebuild + prefix re-seed | T3 |
| hybrid_generate knobs | T4 |
| No launcher / no tencent dep | T4–T5 |
| Unit tests | T1–T4 |

## Notes for implementers

- Prefer adapting tencent source text over rewriting from scratch; strip `examples.coding_agent_rl` imports.
- Path A workspace init is `agent_runtime.prepare_workspace` (not tencent `initialize_swe_workspace`); rebuild must use Path A helpers.
- Do not commit unless the user asks.
