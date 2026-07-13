# Path A Hybrid Step-GRPO 设计

日期：2026-07-12  
分支 / worktree：`feature/cc-ags-swe` → `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`  
上级：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
参考（只读，不依赖）：`slime-tencent/examples/coding_agent_rl/step_reconstruct/`、`/mnt/sn-007/jiaxicao/code/算法设计版本7.md`

## 1. 目标

在 Path A（`claudecode_ags` + AGS + SWE）内落地 **hybrid step-GRPO 库代码 + 单测**：

1. 自包含实现，**不 import / 不依赖 `slime-tencent`**。
2. 算法按本设计确认的 knobs（与现网 hybrid **不完全相同**）。
3. 本轮交付可单测的模块；**HyperPod launcher / Job 下一轮**。

## 2. 非目标（本轮）

- HyperPod launcher / PyTorchJob / ALB 接线
- 纯 step-GRPO 模式（只做 hybrid）
- OPSD
- 依赖或 import `slime-tencent`
- 改动现有普通 GRPO 默认路径 `claudecode_ags/generate.py`
- 真集群 AGS 冒烟（单测用 mock）

## 3. 已拍板决策

| 项 | 选择 |
|----|------|
| 范围 | 只迁 **hybrid** |
| 代码放置 | 拷贝/改编进 Path A：`examples/claudecode_ags/step_reconstruct/` |
| 本轮交付 | **库 + 单测**（无 launcher） |
| Stage-1 | 函数**内部**跑 `K` 条 vanilla（朴素 GRPO group） |
| 外层采样 | `--n-samples-per-prompt=1`；`K = STEP_GRPO_HYBRID_K`（默认 8） |
| 触发分支 | group 内**只要有失败 trial** 就触发 |
| 选型 | 所有失败 trial 的 **patch turn** 进全局池；按 **edit-PPL** 取 top-B |
| patch turn | **会改仓库的 tool 步**（相对 baseline workspace 变化） |
| B | `B = min(K, 可用 patch turns)` |
| 每点分叉 | 每个入选 turn 分叉 **K** 条 continuation |
| PPL | 对齐现网 **edit-PPL** 并迁到 Path A |
| PPL_SKIP | **取消**（不再因低 PPL 跳过分支） |
| Advantage | vanilla → prompt group GRPO；branch → `step_group_key` GRPO |
| Filter | 迁过来，**默认启用**（可 env 关） |
| Train scope | branch **整条 continuation** 进 loss（非 `first_action`） |
| Stage-2 续跑 | `rebuild(diff_t)` + **prefix re-seed** |
| Eval | `evaluation=True` → 只跑 **1 条 vanilla**，不分支 |

## 4. 目录与模块

```text
examples/claudecode_ags/step_reconstruct/
  __init__.py
  _common.py              # diff / path / normalize 共享
  session_capture.py      # Stage-1：PostToolUse 快照 + SessionBundle
  workspace_rebuild.py    # rebuild(s_t) + prefix re-seed + resume_and_run
  hybrid_generate.py      # hybrid_generate 入口
  edit_ppl.py             # patch-turn edit-PPL
  step_grpo_advantage.py  # post_process_rewards + filter

tests/claudecode_ags/
  test_step_reconstruct_selection.py
  test_step_reconstruct_advantage.py
  test_step_reconstruct_edit_ppl.py
  test_step_reconstruct_rebuild.py   # mock sandbox
```

约定：

- 普通 GRPO 仍走 `examples.claudecode_ags.generate.generate`（本轮不改默认）。
- Hybrid 入口：`examples.claudecode_ags.step_reconstruct.hybrid_generate.hybrid_generate`。
- Reward：复用 Path A `examples.claudecode_ags.rewards.default.compose`。
- Advantage 入口：`examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards`（及 `filter`）。

## 5. 数据流（一次 `hybrid_generate`）

外层：`--n-samples-per-prompt=1`。内部：`K = STEP_GRPO_HYBRID_K`（默认 8）。

```text
evaluation=True?
  └─ Yes → 跑 1 条 Path A vanilla，emit vanilla sample，结束
  └─ No  ↓

Stage-1（朴素 GRPO group）
  for trial in 0..K-1:
    AGS sandbox + CC rollout + PostToolUse 快照
    → SessionBundle_trial + segments + reward(compose)
    → Path A 分段 fan-out（与普通 GRPO 一致），打上 sample_kind=vanilla / trial_idx
  返回的 vanilla 行数 = 各 trial 的 segment 数之和（不是「刚好 K 行」）

若失败 trial 数为 0 → 只返回 vanilla，结束

Stage-1.5 选型
  收集所有失败 trial 的 patch turns
  对每个 patch turn 算 edit-PPL
  全局池按 PPL 降序取 top-B，B = min(K, 可用数)
  （无 PPL_SKIP）

Stage-2 分叉
  for each 入选 (source_trial_idx, step_t):
    rebuild workspace 到 s_t
    prefix re-seed + 跑 K 条 continuation 至结束
    eval + compose reward
    emit K 条 sample_kind=branch
      step_group_key = "{group_index}:{source_trial_idx}:{step_t}"
      loss mask 覆盖整条 continuation

返回：全部 vanilla segment samples + 最多 B×K 条 branch samples
```

随后 `step_grpo_advantage.post_process_rewards`：

- vanilla → 同一 trial 的 segment reward **先求和**得到 episode reward，再按 prompt group 做 GRPO；advantage **广播**回该 trial 各 segment（对齐现网 / Path A `fanout_grpo` 语义）
- branch → 按 `step_group_key` 做 GRPO 归一化（若 branch 也有多 segment，同样先求和再广播）
- 默认 `filter` 丢掉 std=0 / 全 mask-zero 组

与现网 hybrid 的关键差异：

| | 现网 tencent hybrid | 本设计 |
|--|---------------------|--------|
| 选谁分叉 | 常选 **一条** 最高 edit-PPL 失败 trial，再在其上选若干 `s_t` | **全局池** top-B patch turns（可跨 trial） |
| 每点条数 | `STEP_GRPO_BRANCHES_PER_STEP`（常 8） | 固定为 **K** |
| Train scope | 默认 `first_action` | **整条 continuation** |
| PPL_SKIP | 有 | **无** |
| `step_group_key` | `{group_index}:{step_t}` | `{group_index}:{source_trial_idx}:{step_t}` |

## 6. 接口、metadata、环境变量

### 6.1 对外符号

| 符号 | 作用 |
|------|------|
| `hybrid_generate(args, sample, sampling_params, evaluation=False)` | 自定义 generate |
| `post_process_rewards(...)` | tag-aware GRPO |
| `filter(args, data)` | 退化组过滤（默认开） |

### 6.2 Sample metadata

| 字段 | 含义 |
|------|------|
| `sample_kind` | `"vanilla"` \| `"branch"` |
| `trial_idx` | Stage-1 trial 下标（0..K-1）；vanilla 必有 |
| `step_t` | 分叉前缀步（branch 必有） |
| `source_trial_idx` | 该 `s_t` 来自哪条失败 trial（branch） |
| `step_group_key` | `"{group_index}:{source_trial_idx}:{step_t}"` |
| `branch_idx` | 同一 `s_t` 上第几条 continuation |
| `edit_ppl` | 入选 turn 的 PPL（日志 / debug） |

### 6.3 环境变量

| Env | 默认 | 含义 |
|-----|------|------|
| `STEP_GRPO_HYBRID_K` | `8` | Stage-1 group size；每点分叉条数；top-B 的名义 K |
| `STEP_GRPO_BRANCH_CONCURRENCY` | `8` | Stage-2 并发上限 |
| `STEP_GRPO_PPL_CLIP` | 与迁入实现一致 | edit-PPL clip |
| `STEP_GRPO_FILTER` | `1` | `1` 启用 filter |

**不提供** `STEP_GRPO_PPL_SKIP`（已取消该门槛）。

训练接线约定（下轮 launcher）：

- `--n-samples-per-prompt 1`
- `--custom-generate-function-path examples.claudecode_ags.step_reconstruct.hybrid_generate.hybrid_generate`
- `--custom-reward-post-process-path examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards`
- `--rollout-sample-filter-path examples.claudecode_ags.step_reconstruct.step_grpo_advantage.filter`（当 `STEP_GRPO_FILTER=1`）

## 7. 错误与降级

| 情况 | 行为 |
|------|------|
| 某 vanilla trial 失败 / 超时 | 记日志，跳过该 trial；其余继续 |
| 全部 vanilla 失败 | 不分支；按实现返回空或错误占位（需在实现中与 Path A 错误样本约定对齐） |
| 无失败 trial | 只返回 vanilla |
| 无 patch turn | 只返回 vanilla |
| rebuild apply 失败 | 丢该 branch，不拖垮整 group |
| 单条 continuation 超时 | 丢该 branch |
| Stage-2 整体超时 | 返回已有 vanilla（+ 已完成的 branch） |

## 8. 测试计划（本轮）

| 测试 | 覆盖 |
|------|------|
| 选型 | 全局 top-K；跨 trial；`B=min(K,可用)`；无失败 / 无 patch |
| advantage | vanilla group vs `step_group_key`；filter std=0 / mask-zero |
| edit-PPL | mock / 固定分数；clip 行为 |
| rebuild / re-seed | mock sandbox；不要求真 AGS |

不在本轮：真 AGS capture、真 CC、HyperPod 训一 step。

## 9. 成功标准

1. `examples/claudecode_ags/step_reconstruct/` 在 Path A 自包含可 import。
2. 行为符合 §3 已拍板 knobs。
3. `post_process_rewards` + 默认 `filter` 单测通过。
4. 选型 / edit-PPL / undersubscribe 单测通过。
5. rebuild + prefix re-seed 有可测接口（mock）。
6. 文档写明外层 `n-samples-per-prompt=1` 与 `STEP_GRPO_HYBRID_K` 语义。

## 10. 后续（明确不在库轮；launcher 已落地）

1. ~~`run_hybrid_1node_*.sh` + PyTorchJob 模板~~ → `launch/run_hybrid_1node_debug.sh` + `hybrid_1node_job/` + `hybrid_adapter_alb/`
2. 数据：step-GRPO 倾向 resolved 0–7 池（见 `notes/2026-07-12-swegym-filter-passk-results.md`）。
3. 真 AGS 冒烟与 1-node debug 训通。
4. OPSD（若需要）。

## 11. 审阅结论（已拍板）

- 方案：整模块迁入 Path A（方案 1）
- 本轮：库 + 单测
- Hybrid knobs：§3 表
- 与现网差异：§5 对比表
