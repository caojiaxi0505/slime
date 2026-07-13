# 题目仓库初始化（第二阶段）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** ScaleSWE `pre_commands`、Rebench 模式钩子、评测沙箱共用 init（轻 scrub）、parquet 回填 `swe_smith_bug_patch`。不做真判分。

**Architecture:** 扩展 `workspace_init.py`（模式 + scaleswe + 双档 scrub + `rollout_side`）；新建 `bug_patch_source.py`；`generate._evaluate_diff` 先 init；agent 路径 `rollout_side=True`。

**Tech Stack:** Python asyncio、pytest、pandas/pyarrow（parquet fixture）

**Spec:** `docs/superpowers/specs/2026-07-10-workspace-init-phase2-design.md`  
**Worktree:** `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`

---

## 文件

| 文件 | 动作 |
|------|------|
| `examples/claudecode_ags/workspace_init.py` | 改：SCALESWE/REBENCH、pre_commands、轻 scrub、`rollout_side` |
| `examples/claudecode_ags/bug_patch_source.py` | 新：parquet 回填 |
| `examples/claudecode_ags/agent_runtime.py` | 改：prepare 传新字段 + `rollout_side=True` |
| `examples/claudecode_ags/generate.py` | 改：metadata、回填、`_evaluate_diff` 共用 init |
| `tests/claudecode_ags/test_workspace_init.py` | 改/增 |
| `tests/claudecode_ags/test_bug_patch_source.py` | 新 |
| `examples/claudecode_ags/CHECKLIST.md` | 更新 |

---

### Task 1: 模式判定 + TaskFields 扩展

- [ ] 增加 `WorkspaceMode.SCALESWE` / `REBENCH`
- [ ] `TaskFields` 增加 `pre_commands`、`install_config`
- [ ] `detect_workspace_mode` 按 design 顺序
- [ ] `coerce_install_config`、`normalize_pre_commands`
- [ ] 单测优先级表

### Task 2: ScaleSWE + 双档 scrub

- [ ] `build_eval_git_scrub_command`（轻量，marker `slime_git_scrub_eval`）
- [ ] `apply_git_scrub(sb, workdir, *, rollout_side)`
- [ ] `apply_scaleswe_pre_commands` + `scaleswe_root_filesystem_prep`
- [ ] `initialize_task_workspace(..., rollout_side=True)` 接 scaleswe；classic/smith 按档 scrub
- [ ] 单测

### Task 3: bug_patch_source

- [ ] `resolve_swe_smith_bug_patch(metadata) -> str`
- [ ] 临时 parquet fixture 单测（无 pandas 则 skip）

### Task 4: 接入 generate / prepare / eval

- [ ] `prepare_workspace` 传 `pre_commands` / `install_config` / `rollout_side=True`
- [ ] `_parse_metadata` + 回填
- [ ] `_evaluate_diff`：init(`rollout_side=False`) 失败 → unresolved；成功再 `simple_cmd.evaluate`
- [ ] 全量 `tests/claudecode_ags/` 通过；更新 CHECKLIST

不自动 commit。
