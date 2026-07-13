# Kaniko ECR Slime Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship HyperPod/K8s Kaniko Job artifacts that build the full slime training image from the local worktree and push to ECR `ap-southeast-3` / `sn5/jiaxicao/slime` — without any local `docker build`.

**Architecture:** `docker/Dockerfile.kaniko` mirrors upstream `docker/Dockerfile` but `COPY`s the build context into `/root/slime` (Path A branch) and optionally installs EFA from `docker/efa/`. `submit_build.sh` renders `job.yaml.template` and `kubectl apply -n sn5-system-intern`.

**Tech Stack:** Kaniko executor, kubectl, AWS CLI (ECR describe), hostPath/FSx context.

**Spec:** [2026-07-10-kaniko-ecr-slime-image-design.md](../specs/2026-07-10-kaniko-ecr-slime-image-design.md)

---

## File map

| Path | Responsibility |
|------|----------------|
| `docker/Dockerfile.kaniko` | Full training image; local slime COPY + EFA hook |
| `docker/efa/README.md` | Where to place `install-efa-in-container.sh` / `fix-efa-conflict.sh` |
| `.dockerignore` | Keep Kaniko context small (exclude `.venv`, caches) |
| `examples/claudecode_ags/launch/kaniko/job.yaml.template` | Job template |
| `examples/claudecode_ags/launch/kaniko/submit_build.sh` | Render + apply + print follow-up cmds |
| `examples/claudecode_ags/launch/kaniko/README.md` | Operator runbook |
| `tests/claudecode_ags/test_kaniko_launch.py` | Static / `--help` offline tests |

---

### Task 1: Dockerfile.kaniko + EFA placeholder + .dockerignore

**Files:**
- Create: `docker/Dockerfile.kaniko`
- Create: `docker/efa/README.md`
- Create: `.dockerignore` (if missing)

- [x] **Step 1:** Copy upstream `docker/Dockerfile` body; replace `git clone THUDM/slime` with `COPY . /root/slime` + `pip install -e . --no-deps`; keep int4_qat install.
- [x] **Step 2:** Add `ENABLE_EFA` ARG; when `1`, require and run `docker/efa/install-efa-in-container.sh` (+ fix script).
- [x] **Step 3:** Document EFA script source path in `docker/efa/README.md` (HyperPod guide §5).
- [x] **Step 4:** Add `.dockerignore` excluding `.venv`, `__pycache__`, `.git`, large run artifacts.

### Task 2: Kaniko Job template + submit script + README

**Files:**
- Create: `examples/claudecode_ags/launch/kaniko/job.yaml.template`
- Create: `examples/claudecode_ags/launch/kaniko/submit_build.sh`
- Create: `examples/claudecode_ags/launch/kaniko/README.md`

- [x] **Step 1:** Template with namespace `sn5-system-intern`, hostPath context, destination ECR URI placeholders, resource knobs, `ttlSecondsAfterFinished`, `restartPolicy: Never`.
- [x] **Step 2:** `submit_build.sh`: resolve account/tag/context; optional `--create-repo`; check EFA scripts when `ENABLE_EFA=1`; `envsubst` → apply; print logs/describe-images commands. **Never call docker.**
- [x] **Step 3:** README: prerequisites, create-repository, submit, follow logs, acceptance.

### Task 3: Offline tests + checklist

**Files:**
- Create: `tests/claudecode_ags/test_kaniko_launch.py`
- Modify: `examples/claudecode_ags/CHECKLIST.md`

- [x] **Step 1:** Assert template contains namespace / dockerfile / destination placeholders; `submit_build.sh --help` exits 0.
- [x] **Step 2:** Assert `Dockerfile.kaniko` has `COPY . /root/slime` and no `git clone https://github.com/THUDM/slime`.
- [x] **Step 3:** Update CHECKLIST; mirror plan/spec into worktree `docs/superpowers/` if needed.

---

## Success criteria

1. Artifacts exist under `launch/kaniko/` + `docker/Dockerfile.kaniko`.
2. Docs forbid local docker build; require `-n sn5-system-intern`.
3. ECR target defaults: `ap-southeast-3` / `sn5/jiaxicao/slime`.
4. Image installs worktree slime (COPY), not upstream main clone.
5. Offline tests pass; live Job submit is operator-owned.
