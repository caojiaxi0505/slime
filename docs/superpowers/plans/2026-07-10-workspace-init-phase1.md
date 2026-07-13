# 题目仓库初始化（第一阶段）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CC 启动前，按 SWE-bench classic / SWE-smith 把沙箱仓库调到有 bug 基线，并做 rollout 侧 git scrub，再写 `PROBLEM_STATEMENT.md`。

**Architecture:** 新建 `examples/claudecode_ags/workspace_init.py` 负责模式判定与 reset/分支/补丁/scrub；`agent_runtime.prepare_workspace` 编排调用；`generate._parse_metadata` 传入最少字段。不改 `slime/` 核心。

**Tech Stack:** Python asyncio、FakeSandbox 单测、pytest（worktree `.venv`）

**Spec:** `docs/superpowers/specs/2026-07-10-workspace-init-phase1-design.md`  
**Worktree:** `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`

---

## 文件结构

| 文件 | 职责 |
|------|------|
| Create: `examples/claudecode_ags/workspace_init.py` | `WorkspaceMode`、`TaskFields`、detect、reset、swesmith、scrub、`initialize_task_workspace` |
| Modify: `examples/claudecode_ags/agent_runtime.py` | `prepare_workspace` 接新参数并调用 init |
| Modify: `examples/claudecode_ags/generate.py` | `_parse_metadata` + 调用点传字段；init 失败走 abort |
| Create: `tests/claudecode_ags/test_workspace_init.py` | 模式判定与命令路径 |
| Modify: `tests/claudecode_ags/test_agent_runtime_env.py` | `prepare_workspace` 新签名 |
| Modify: `examples/claudecode_ags/CHECKLIST.md` | 标记 phase1 workspace init |

---

### Task 1: 模式判定 + TaskFields

**Files:**
- Create: `examples/claudecode_ags/workspace_init.py`
- Test: `tests/claudecode_ags/test_workspace_init.py`

- [ ] **Step 1: 写失败测试（模式判定）**

```python
from examples.claudecode_ags.workspace_init import (
    TaskFields,
    WorkspaceMode,
    detect_workspace_mode,
)

def test_detect_swesmith():
    f = TaskFields(data_source="swe_smith_foo", instance_id="repo__issue")
    assert detect_workspace_mode(f) == WorkspaceMode.SWESMITH

def test_detect_swebench_classic_by_base_commit():
    f = TaskFields(base_commit="abc123")
    assert detect_workspace_mode(f) == WorkspaceMode.SWEBENCH_CLASSIC

def test_detect_generic():
    assert detect_workspace_mode(TaskFields()) == WorkspaceMode.GENERIC
```

- [ ] **Step 2: 实现最小 `WorkspaceMode` / `TaskFields` / `detect_workspace_mode` / `task_fields_from_metadata`**

判定顺序：`swe_smith*` → 有 `base_commit` → `generic`。  
`task_fields_from_metadata` 只读：`workdir`、`instance_id`、`data_source`、`base_commit` 或 `swebench.base_commit`、`swe_smith_bug_patch`、`problem_statement`（problem 可不进 TaskFields，由 prepare 单独传）。

- [ ] **Step 3: pytest 通过**

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m pytest tests/claudecode_ags/test_workspace_init.py -k detect -v
```

---

### Task 2: SWE-bench reset + git scrub

**Files:**
- Modify: `examples/claudecode_ags/workspace_init.py`
- Test: `tests/claudecode_ags/test_workspace_init.py`

- [ ] **Step 1: 测试 reset 与 scrub 命令出现在 FakeSandbox.cmds**

```python
def test_swebench_reset_and_scrub(monkeypatch):
    # initialize_task_workspace with base_commit → cmds contain
    # "git reset --hard abc" and scrub marker / orphan commit
```

- [ ] **Step 2: 实现**

- `build_git_scrub_command(workdir)`：对齐现网 rollout scrub（去 remote、清 refs、orphan commit、gc），marker 用 `slime_git_scrub`（不要 `verl_` 前缀）
- `apply_swebench_reset`：`git reset --hard <commit>`，`fail_fast=True`
- `_exec_script`：写临时脚本再 `bash`（路径如 `/tmp/slime_ws_init.sh`）
- `initialize_task_workspace`：classic 分支 = reset（失败 return False）→ scrub（best-effort）→ return True

- [ ] **Step 3: pytest 通过**

---

### Task 3: SWE-smith 分支 + bug patch

**Files:**
- Modify: `examples/claudecode_ags/workspace_init.py`
- Test: `tests/claudecode_ags/test_workspace_init.py`

- [ ] **Step 1: 测试**

- 非 synthetic：cmds 含 `git checkout <instance_id>`
- 有 `swe_smith_bug_patch`：会 `write_file` patch 并尝试 `git apply`
- synthetic id：不跑 checkout 分支名

- [ ] **Step 2: 实现（克制版，语义对齐现网）**

- `apply_swesmith_branch`
- `_reverse_patch` / `_diff_applies_cleanly` / `_apply_diff` / `_commit_bug_baseline`（同模块短函数）
- `apply_swesmith_bug_baseline`：正向可打则打；否则反向可打则视为已 mutated；否则失败；最后 commit baseline
- swesmith 流程：branch → 若需则 patch baseline → scrub → True；必需 patch 失败 → False

- [ ] **Step 3: pytest 通过**

---

### Task 4: 接入 prepare_workspace + generate

**Files:**
- Modify: `examples/claudecode_ags/agent_runtime.py`
- Modify: `examples/claudecode_ags/generate.py`
- Modify: `tests/claudecode_ags/test_agent_runtime_env.py`
- Modify: `tests/claudecode_ags/test_generate_reward_wiring.py`（若 mock 签名变了）

- [ ] **Step 1: 扩展 `prepare_workspace`**

```python
async def prepare_workspace(
    sb,
    *,
    workdir: str,
    problem_statement: str,
    instance_id: str = "",
    data_source: str = "",
    base_commit: str = "",
    swe_smith_bug_patch: str | None = None,
) -> None:
    ok = await initialize_task_workspace(...)
    if not ok:
        raise RuntimeError(...)
    await sb.write_file(f"{workdir}/PROBLEM_STATEMENT.md", problem_statement, user="agent")
```

注意：`ensure_agent_user` 放在 `initialize_task_workspace` 内，避免重复。

- [ ] **Step 2: `generate._parse_metadata` 增加字段并传入；init `RuntimeError` → `_abort`**

- [ ] **Step 3: 全量 `tests/claudecode_ags/` 通过；更新 CHECKLIST**

```bash
.venv/bin/python -m pytest tests/claudecode_ags/ -v
```

---

## Spec 覆盖自检

| Spec 项 | Task |
|---------|------|
| swebench reset | 2 |
| swesmith branch/patch | 3 |
| git scrub | 2–3 |
| generic 只写 PROBLEM | 1+4 |
| 最少 metadata | 1+4 |
| 失败 abort | 4 |
| 单测 | 1–4 |
| 不改 slime 核心 | 全部 |

不提交 commit，除非用户明确要求。
