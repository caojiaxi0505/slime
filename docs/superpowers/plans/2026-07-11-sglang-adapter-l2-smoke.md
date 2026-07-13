# SGLang Adapter Deploy + L2 Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Ship Phase 1 HyperPod deploy artifacts (SGLang + segmented adapter + dedicated ALB Ingress) and Phase 2 `ags_smoke --level 2` for live Claude Code → adapter → eval.

**Architecture:** Independent Deployment on `sn5-system-intern` runs SGLang + `serve_adapter.py` from FSx worktree (no image rebuild). Service+Ingress expose adapter `/health`. L2 reuses `agent_runtime` + dual-sandbox eval without `train.py`.

**Tech Stack:** kubectl, AWS ALB Ingress, SGLang, SegmentedAnthropicAdapter, AGS smoke CLI.

**Spec:** [2026-07-11-sglang-adapter-l2-smoke-design.md](../specs/2026-07-11-sglang-adapter-l2-smoke-design.md)

---

## File map

| Path | Responsibility |
|------|----------------|
| `examples/claudecode_ags/launch/sglang_adapter/entrypoint.sh` | Start SGLang then adapter |
| `examples/claudecode_ags/launch/sglang_adapter/serve_adapter.py` | HTTP adapter process |
| `examples/claudecode_ags/launch/sglang_adapter/deployment.yaml.template` | GPU Deployment + volumes |
| `examples/claudecode_ags/launch/sglang_adapter/service-ingress.yaml.template` | Service + ALB Ingress |
| `examples/claudecode_ags/launch/sglang_adapter/submit_deploy.sh` | Render + apply + print URL hints |
| `examples/claudecode_ags/launch/sglang_adapter/README.md` | Operator runbook |
| `examples/claudecode_ags/smoke/ags_smoke.py` | Add `--level 2` |
| `examples/claudecode_ags/smoke/README.md` | L2 docs |
| `tests/claudecode_ags/test_ags_smoke_script.py` | Level 2 parser / offline tests |
| `tests/claudecode_ags/test_sglang_adapter_launch.py` | Template / submit --help |
| `docs/superpowers/runbooks/2026-07-11-sglang-adapter-l2-ops.md` | Combined ops |

---

### Task 1: Phase 1 launch artifacts

**Files:** create all under `examples/claudecode_ags/launch/sglang_adapter/`

- [x] **Step 1:** `serve_adapter.py` — load tokenizer from `HF_CHECKPOINT`, bind `0.0.0.0:ADAPTER_PORT`, `sglang_url=http://127.0.0.1:SGLANG_PORT`.
- [x] **Step 2:** `entrypoint.sh` — launch `sglang.launch_server`, wait for `/health` or port, then `serve_adapter.py`.
- [x] **Step 3:** Deployment + Service/Ingress templates (ns `sn5-system-intern`, PVC `youtu-sn2-007`, ALB annotations matching existing ingresses).
- [x] **Step 4:** `submit_deploy.sh` + README (manual URL into `slime_ags.env`).

### Task 2: Phase 2 L2 in ags_smoke

**Files:**
- Modify: `examples/claudecode_ags/smoke/ags_smoke.py`
- Modify: `examples/claudecode_ags/smoke/README.md`
- Test: `tests/claudecode_ags/test_ags_smoke_script.py`

- [x] **Step 1:** `run_l2`: require `SLIME_ADAPTER_PUBLIC_URL`; optional health curl in sandbox; sandbox A CC path; sandbox B eval on model diff (not gold).
- [x] **Step 2:** CLI `--level` choices `(0,1,2)`; docs.
- [x] **Step 3:** Offline tests for level 2 help/parser; mock unit for env gate if cheap.

### Task 3: Ops runbook + checklist

**Files:**
- Create: `docs/superpowers/runbooks/2026-07-11-sglang-adapter-l2-ops.md`
- Modify: `examples/claudecode_ags/CHECKLIST.md`

- [x] **Step 1:** Ops doc (deploy → fill URL → L2 command).
- [x] **Step 2:** CHECKLIST row; mirror docs to worktree.
- [x] **Step 3:** `pytest tests/claudecode_ags/test_ags_smoke_script.py tests/claudecode_ags/test_sglang_adapter_launch.py -v`

---

## Success criteria

1. `submit_deploy.sh --help` / `--dry-run` works; templates have correct ns/ALB annotations.
2. `ags_smoke --level 2` documented; offline tests green.
3. Live deploy/L2 remain operator-owned (need `HF_CHECKPOINT` + cluster GPUs).
