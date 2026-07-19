# Hybrid 已运行实验的逻辑变更记录

日期：2026-07-19  
分支：`feature/cc-ags-swe`  
用途：记录 **run 已经跑完或正在跑之后，Hybrid 训练逻辑又发生的变化**。旧 run 的结果不会因
代码更新而自动改变；以后比较曲线或 checkpoint 时，应先查本文。

## 1. 当前变更摘要

| 逻辑编号 | 旧逻辑 | 新逻辑 | 影响范围 |
|---|---|---|---|
| `H-S1-ABORT-1` | Stage-1 trial 抛异常后整条删除 | 保留 `reward=0`、`loss_mask=0` 的独立占位 | Stage-1 episode 数、advantage、loss 权重、W&B 分母 |
| `H-S1-STD0-1` | Stage-1 同分组会被 Hybrid filter 删除 | 与朴素 GRPO 一样保留；只记录同分，不删除 | Stage-1 的 KL 项；PG advantage 仍为 0 |
| `H-S2-SCOPE-1` | Stage-2 完整续跑的全部 assistant turn 都参与 loss | 可选只训练续跑后的第一个 assistant turn | Stage-2 有效 token 与梯度范围；评测和 reward 不变 |

Stage-2 的分叉点选择、token-exact resume、branch advantage、branch 权重和退化 branch 组过滤
均未在这次修改中改变。

## 2. 哪些旧 run 使用了旧逻辑

以下目录属于同一条 Hybrid v9 训练线，均在本次修改前生成：

| 目录 / EXP_TAG | 说明 | 本次新逻辑是否生效 |
|---|---|---|
| `qwen35_9b_cc_ags_2node_hybrid_t30_unlimited_tokenexact_v9_reward` | 初始训练段 | 否 |
| `qwen35_9b_cc_ags_2node_hybrid_t30_unlimited_tokenexact_v9_reward_resume_iter9` | 第一次续训目录 | 否 |
| `qwen35_9b_cc_ags_2node_hybrid_t30_unlimited_tokenexact_v9_reward_resume2_iter9_ro` | 使用只读 AGS 模板后的续训目录 | 否 |

因此，这条 v9 训练线的 W&B 曲线、checkpoint 和 iter14/iter24 评测都表示**旧训练逻辑的
结果**。评测数值本身仍有效，但不能写成“使用新异常占位逻辑训练得到的结果”。

首次使用上述新逻辑的 run：

| 项目 | 值 |
|---|---|
| EXP_TAG | `qwen35_9b_cc_ags_hybrid_i14_i24_firstturn_s1fix_v1` |
| 起始 checkpoint | Hybrid v9 `iter_0000014` |
| 数据状态 | `global_dataset_state_dict_14.pt` |
| 生效范围 | rollout step 15–24 |
| Stage-2 loss scope | `first_turn` |
| W&B 接续方式 | 将 `hy9ro2` 的内部历史 0–59 行原样写入新 run，再用 `resume=must` 追加 |
| W&B 新 run id | [`w17gn551`](https://wandb.ai/models-tencent7723/coding-rl/runs/w17gn551) |

该 run 同时启用 `H-S1-ABORT-1`、`H-S1-STD0-1` 和 `H-S2-SCOPE-1`。因此它是从旧 iter14
开始采用新逻辑的续训线，不是只改变 Stage-2 mask 的单变量消融。

W&B 账号未开通 `fork_from` 预览功能，因此没有直接调用服务端 fork。新 run 已核验包含连续的
内部 step 0–59，覆盖 `rollout/step=0–14`；训练从内部 step 60 开始追加，不会改写旧 run。

## 3. 旧 run 实际少了什么

设定为每步 16 道题，每题 8 次 Stage-1 尝试，因此每步应有 128 个 Stage-1 episode。

对最终对比口径中的 step 10–32 审计如下：

| 项目 | 数量 |
|---|---:|
| 理论 Stage-1 episode | 23 × 128 = 2,944 |
| rollout dump 中实际存在 | 2,665 |
| 被异常路径直接删除 | 279 |

这 279 个单位是 **episode**，不是 compact segment。运行日志中的 279 条
`vanilla trial skipped` 与 dump 中缺少的 `(group_index, trial_idx)` 一一对应。

| 日志中的异常类型 | episode 数 |
|---|---:|
| AGS 连接超时 | 124 |
| 请求体写入失败 | 57 |
| Broken pipe | 36 |
| 响应不完整或被截断 | 24 |
| 服务端断开连接 | 18 |
| 异常文本为空，旧日志未保存具体类型 | 20 |
| **合计** | **279** |

前 259 条可以由日志直接确认为传输或连接异常。剩余 20 条只能确认“runner 抛异常并被删除”，
不能再从旧日志可靠细分类型。

## 4. 删除 episode 为什么会改变训练

朴素 GRPO 遇到同类 runner 异常时不会缩小 8 条一组的结构。它会留下一个占位：reward 为
0，loss mask 为 0。这个占位本身不产生梯度，但 reward=0 仍参与同题 8 条结果的均值和标准差。

旧 Hybrid 使用 `continue` 删除异常 trial，之后又只按剩余有效 trial 计算和加权。假设一题
计划 8 条、其中 1 条异常：

| 项目 | 朴素 GRPO / 新 Hybrid | 旧 Hybrid |
|---|---:|---:|
| 参与 reward 归一化的 episode | 8 | 7 |
| 异常 episode 的 reward | 0 | 不存在 |
| 每个 Stage-1 episode 的名义权重 | 1/8 | 1/7 |
| 异常 episode 的直接梯度 | 0 | 不存在 |
| W&B `resolved_rate` 分母 | 8 | 7 |

所以旧逻辑不只是少了一行统计。它还改变了剩余 episode 的 advantage，并把每条有效 episode
的权重从 1/8 放大到 1/7。不同题缺失数量不同，实际训练目标也会随基础设施异常率漂移。

## 5. 新逻辑如何处理

### 5.1 异常 trial 占位

普通 Stage-1 runner 异常现在生成一个独立 `Sample`：

```text
status = ABORTED
reward = 0
loss_mask = [0]
remove_sample = true
trial_idx = 原 trial_idx
branch_uid = v:{group_index}:t{trial_idx}
stage1_group_size = 8
```

它保留原 trial 的 episode 身份，因此每题始终有 8 个 Stage-1 slot。若 8 条全部异常，则返回
8 个占位，而不是把整题压成 1 个占位。`StepTurnAlignmentError` 仍然直接报错，不会被伪装成
基础设施占位。

### 5.2 Advantage

Stage-1 计算均值和标准差时包含占位的 reward=0，与朴素
`fanout_grpo.post_process_rewards` 一致。占位也会得到一个 advantage 数值，但它的 loss mask
为 0，所以不会直接反向传播。

Stage-2 没有 bundle 的占位不能作为分叉来源；其他成功生成 bundle 的失败 trial 仍可正常
进入候选选择。

### 5.3 Loss 权重

Stage-1 权重按**计划 slot 数**分配，而不是按剩余有效 episode 数分配。当前 `K=8`，所以每条
名义权重固定为 1/8；占位虽然也记录 1/8，但零 loss mask 使其实际贡献为 0。

新样本同时写入 `stage1_group_size=8`。训练前若实际只找到 7 个 episode 身份，会直接报错，
不再静默改成 1/7。

### 5.4 Stage-1 同分组

朴素 GRPO 不使用 Hybrid 的 rollout filter 删除同分组。新 Hybrid 因此只统计 Stage-1
`std_zero` / `all_mask_zero`，不再据此删除整组。Stage-2 的同分组过滤保持不变。

### 5.5 Stage-2 只训练第一个 assistant turn

开关 `STEP_GRPO_STAGE2_LOSS_SCOPE` 默认是 `full_continuation`，旧功能不变。本次续训设置为
`first_turn`：Stage-2 仍完整运行 Claude Code、执行工具、生成最终 patch、评测并计算 episode
reward；训练时只保留恢复后第一次模型响应的 loss mask，后续模型响应全部置零。

代码在合并 trajectory 时保存每次模型响应的精确 token 区间
`assistant_output_spans`，再按区间修改 mask，不通过连续 1 的位置猜测 turn 边界。被 mask 的 token
对应 rollout logprob 也同步置零。若区间缺失或非法，branch 直接报错，不会静默退回完整续跑 loss。

## 6. 新 run 的验收指标

新代码会在 W&B 增加：

| 指标 | 正常含义 |
|---|---|
| `perf/step_grpo/n_stage1_planned_trials` | 每步应为 `16 × 8 = 128` |
| `perf/step_grpo/n_stage1_aborted_placeholders` | 本步由异常转换成占位的 episode 数 |
| `perf/step_grpo/stage1_aborted_placeholder_rate` | 上一项除以 128 |
| `outcome/n_episodes` | 新逻辑下应为 128，占位按 unresolved 计入分母 |
| `perf/step_grpo/n_stage2_first_turn_scoped` | 本步使用 `first_turn` 的 Stage-2 episode 数 |
| `perf/step_grpo/stage2_pre_scope_trainable_tokens` | 修改 mask 前的 Stage-2 assistant token 数 |
| `perf/step_grpo/stage2_kept_trainable_tokens` | 只保留第一个 turn 后的 token 数 |
| `perf/step_grpo/stage2_masked_later_trainable_tokens` | 被屏蔽的后续 turn token 数 |
| `perf/step_grpo/stage2_kept_token_rate` | 保留 token 数除以修改前 token 数 |

rollout dump 还应满足：每个 `(group_index)` 恰有 8 个不同的 vanilla `branch_uid`；异常项为
`ABORTED + reward=0 + loss_mask=[0]`；同一题的每个 Stage-1 `loss_weight` 均为 0.125。

只要 W&B 中没有前三个新指标，就不能仅凭相似 EXP_TAG 认定该 run 已使用本次修复。

## 7. 以后如何追加记录

每次已运行实验之后再修改训练含义，都在本文追加一行和一个小节，至少写清：

| 必填项 | 内容 |
|---|---|
| 逻辑编号 | 唯一、稳定的短名称 |
| 旧 run | EXP_TAG、W&B run id、受影响 step |
| 旧逻辑 / 新逻辑 | 具体到分母、reward、advantage、mask 或权重 |
| 是否影响历史 checkpoint | 通常为“是，不能追溯修复” |
| 首个新 run | EXP_TAG、起始 checkpoint、首个生效 step |
| 验收证据 | W&B 指标、dump 字段和回归测试 |

如果从旧 checkpoint 续训，新逻辑只从续训边界之后生效；这可以作为带变更点的继续训练，但
不能当作从 base checkpoint 开始的纯新逻辑对照。

## 8. 代码与测试位置

- 占位生成：`examples/claudecode_ags/step_reconstruct/hybrid_generate.py`
- Stage-1 advantage、权重和 filter：
  `examples/claudecode_ags/step_reconstruct/step_grpo_advantage.py`
- W&B 新指标：`examples/claudecode_ags/wandb_metrics.py`
- Stage-2 turn 边界与 mask：`slime/agent/segment_trajectory.py`、
  `examples/claudecode_ags/step_reconstruct/live_runners.py`
- 回归测试：
  `tests/claudecode_ags/test_step_reconstruct_hybrid_orchestration.py`、
  `tests/claudecode_ags/test_step_reconstruct_advantage.py`、
  `tests/claudecode_ags/test_wandb_metrics.py`
