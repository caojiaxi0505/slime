# Hybrid run `qwen35_9b_cc_ags_2node_hybrid_c64_t45_bmr3zlex-RANK_0` 问题报告

审计时间：2026-07-15 05:22 UTC
代码分支：`feature/cc-ags-swe`
审计时 HEAD：`b81a8963`
W&B 本地 run id：`ctpm02pd`
本地目录：`/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_2node_hybrid_c64_t45`

> **结论：这次 run 不能作为“正确的 Hybrid step-GRPO 与朴素 GRPO”的算法胜负证据。**
>
> 在已经完成的 step 0–4 上，这个 Hybrid 实现的 Stage-1 解题率确实低于朴素 GRPO，`grad_norm` 也更小；但 run 内至少有三个会改变训练含义的问题：Stage-2 没有继承对话历史、分叉发生在目标 edit 之后、多个独立 episode 被合并成一个 prompt 级损失单元。因此，当前曲线只能说明“这份有缺陷的实现没有打过朴素 GRPO”，不能说明原本想验证的 step-GRPO 算法无效。

本文只做诊断和结果解释，不包含停任务、改代码或重提任务。

## 1. 先用白话说明 Hybrid 本来要做什么

对于一道修 bug 的题，预期流程是：

1. **Stage-1：**让 Claude Code 独立尝试 8 次，得到 8 条完整解题过程。
2. **选择关键 edit：**从失败过程里找出模型最不确定、最值得重试的一次代码修改。
3. **回到 edit 之前：**同时恢复当时的代码状态和对话历史。
4. **Stage-2：**从同一个起点再分叉 8 次，让模型重新决定怎么改。
5. **训练：**分别比较 Stage-1 的完整尝试，以及每组 Stage-2 分支中谁最终解题，再按明确权重合并两部分损失。

这次 run 的实际行为更接近：

> 先运行 8 次完整尝试；选出高 edit-PPL 的修改；保留修改后的代码，但丢掉此前对话，让一个“失忆的新 agent”继续；最后把同一道题下数百次独立尝试按 token 混成一个损失单元。

这与预期算法不是同一件事。

## 2. 当前曲线说明了什么

### 2.1 Stage-1 解题率与 `grad_norm`

Hybrid 和朴素 GRPO 在每个对齐 step 上使用了**完全相同的 16 道题**。以下数据来自双方 `rollout_0.pt` 至 `rollout_4.pt` 和训练日志：

| step | Hybrid Stage-1 | 朴素 GRPO | 差值 | Hybrid `grad_norm` | GRPO `grad_norm` |
|---:|---:|---:|---:|---:|---:|
| 0 | 75/128 = 58.59% | 73/128 = 57.03% | +1.56 pp | 0.346 | 0.590 |
| 1 | 79/128 = 61.72% | 86/128 = 67.19% | -5.47 pp | 0.285 | 4.104 |
| 2 | 81/127 = 63.78% | 85/128 = 66.41% | -2.63 pp | 0.262 | 1.061 |
| 3 | 72/126 = 57.14% | 101/128 = 78.91% | -21.76 pp | 0.286 | 0.498 |
| 4 | 63/127 = 49.61% | 77/128 = 60.16% | -10.55 pp | 0.343 | 2.077 |
| **五步均值** | **58.17%** | **65.94%** | **-7.77 pp** | — | — |

在 80 个“题目 × step”配对点上，Hybrid 更好 24 次、相同 14 次、GRPO 更好 42 次；忽略平局后的双侧 sign test 为 `p≈0.0356`。因此，当前实现的早期落后不是单个 step 的偶然尖峰。

但有两个限制：

- step 0 尚未受到本轮参数更新影响，双方结果接近且 Hybrid 略高，说明数据、初始模型和 reward 管线没有整体错位。
- 从 step 1 开始，模型已经被本 run 的错误目标更新。此后的差距与实现问题相符，但仅凭曲线不能把每一分差距精确归因到某一个 bug。

`grad_norm` 只是“本步合成梯度有多大”，不是解题率，也不是 Adam 实际参数更新量。它变小可能来自更多梯度互相抵消，不能单独解释为“模型没有学”或“step-GRPO 天生更弱”。

## 3. 会改变训练含义的 bug

### 3.1 P0：Stage-2 的对话历史实际为空

**预期：**恢复关键 edit 附近的代码状态，同时把此前的分析、工具调用和工具结果重新喂给 Claude Code。这样，分支才是从同一个中间状态继续。

**实际证据：**审计截点共有 889 个 `bundle.json`、883 个 `transcript.jsonl`；883 个 transcript **全部为 0 字节**，非空文件为 0。run 仍在运行，因此文件总数可能继续增长，但“已生成 transcript 全为空”的结论不受影响。

原因链条如下：

1. 通用 harness 已把 Claude Code 的 stream-json 输出写到沙箱内的 `.harness/trajectory.jsonl`，见 [`slime/agent/harness/common.py`](../../../slime/agent/harness/common.py) 的 `run_agent`。
2. [`session_capture.py`](../../../examples/claudecode_ags/step_reconstruct/session_capture.py) 的 `capture_snapshots_to_bundle` 只拉取 workspace snapshots，没有把该 trajectory 拉回 bundle。
3. [`live_runners.py`](../../../examples/claudecode_ags/step_reconstruct/live_runners.py) 的 `live_vanilla_runner` 在文件不存在时主动创建一个空的 `transcript.jsonl`。
4. `live_branch_runner` 读到空文件后，不报错，而是走 fallback：在重建后的 workspace 上重新执行原始 prompt。

所以 Stage-2 不是“同一个 agent 从中途继续”，而是“新 agent 看到一份已被修改的仓库后，从头读题”。它不知道此前为什么改、查过哪些文件、测试结果是什么，也无法在相同推理前缀下比较不同动作。

**训练影响：严重。**Stage-2 产生的 reward 和 advantage 仍会进入训练，但它们不再对应设计中的 step-level credit assignment。

**为什么测试没有发现：**现有 branch 单测手工写入了一行 transcript，并 mock 了 resume 调用；它没有覆盖“真实 Stage-1 运行 → 捕获 transcript → Stage-2 读取”的端到端链路。

### 3.2 P0：分叉点落在高 edit-PPL 修改之后

PostToolUse snapshot `i` 表示第 `i` 次工具调用**完成之后**的累计 workspace diff。当前实现：

1. [`edit_ppl.py`](../../../examples/claudecode_ags/step_reconstruct/edit_ppl.py) 在发现 `diff[i] != diff[i-1]` 时返回 `step_t=i`；
2. [`workspace_rebuild.py`](../../../examples/claudecode_ags/step_reconstruct/workspace_rebuild.py) 随后应用 `step_diff(i)`。

也就是说，被认为最不确定的代码修改已经写进 workspace，8 条 branch 都继承这份修改。它们只能继续或事后修补，不能重新决定这次关键 edit 应该怎么做。

若目标是“重试高 edit-PPL 的代码动作”，正确起点应是它之前的状态，即对变化发生在 `i` 的 edit 使用 `i-1`。迁移来源 `/mnt/sn-007/jiaxicao/code/slime-tencent/examples/coding_agent_rl/step_reconstruct/stage2_generate.py` 的 `_edit_branch_points` 也明确返回 `i-1`，并说明这是为了让 K 条分支重新决定 edit。

**训练影响：严重。**Stage-2 学到的是“带着同一份可疑 patch 如何继续”，不是“这一处 patch 应该选择什么动作”。它与空 transcript 叠加后，实际变成“失忆 agent 接手一份可能已经改错的代码”。

第一处工具调用就产生 edit 时，目前实现还会从 edit 后分叉；若没有更早 snapshot，应明确跳过或另设 baseline state，不能把 post-edit state 当成 pre-edit state。

### 3.3 P0：数百个独立 episode 共用 16 个 `rollout_id`

这里最容易混淆三个概念：

- **prompt：**一道题；每步有 16 道题。
- **episode：**一次独立的 Claude Code 尝试。Stage-1 的 8 次尝试彼此独立，每条 Stage-2 branch 也彼此独立。
- **segment：**同一次尝试因为上下文压缩而拆出的若干片段；只有这些片段应该共享一个 episode 身份。

当前 [`hybrid_generate.py`](../../../examples/claudecode_ags/step_reconstruct/hybrid_generate.py) 的 `_stamp_shared_rollout_id` 把同一道题下所有 Stage-1 和 Stage-2 sample 都改成同一个 `rollout_id`。优势计算虽然用 `branch_uid` 把独立 episode 分开，但训练损失又按 `rollout_id` 汇总 token 数：

- [`slime/ray/rollout.py`](../../../slime/ray/rollout.py) 按 `rollout_id` 计算整个组的 `rollout_mask_sums`；
- [`slime/backends/megatron_utils/cp_utils.py`](../../../slime/backends/megatron_utils/cp_utils.py) 用这个总数作为损失分母；
- 当前 GBS 为 16，最终仍按 16 个 prompt 单元求平均。

这等价于：每道题只算一票，但这票内部按 token 数混合所有完整尝试和所有 branch。它不是“每个独立尝试先等权，再明确合并 Stage-1/Stage-2”。

run 产物中的实际规模如下：

| step | Stage-1 独立 episode | Stage-2 独立 episode | 训练看到的 `rollout_id` | 活跃 token 中 Stage-2 占比 |
|---:|---:|---:|---:|---:|
| 0 | 128 | 178 | 16 | 24.45% |
| 1 | 128 | 406 | 16 | 66.87% |
| 2 | 127 | 459 | 16 | 80.79% |
| 3 | 126 | 360 | 16 | 63.99% |
| 4 | 127 | 207 | 16 | 64.11% |

直接后果有三项：

1. **Stage-1 权重被稀释。**例如 step 2 中，80.79% 的活跃 token 来自 Stage-2。Stage-1 不再保留一份与朴素 GRPO 等价的基础损失。
2. **两阶段比例随数据漂移。**代码没有一个固定的 `λ` 控制 Stage-2 权重；实际权重由选中多少分叉、每条分支多长、哪些组被过滤共同决定。
3. **更容易发生梯度抵消。**同一道题下数百次独立尝试的正负 advantage 在一个 prompt 级 token 均值里相互抵消，可以自然地产生较小的 `grad_norm`。

这也解释了一个异常旁证：step 2–4 的 Hybrid `train/pg_loss` 分别为 `-0.070459`、`-0.067752`、`-0.057254`，而朴素 GRPO 已接近 0（`-1.39e-4`、`-3.21e-5`、`-1.12e-6`）。GRPO advantage 在 episode 组内已中心化，但 Hybrid 又按 prompt token 长度加权，因此中心化会被重新打破。该指标本身不是独立证明，但与上述代码路径和离线重算一致。

**训练影响：严重。**实际优化目标由 branch token 体量隐式决定，不能解释为预期的 `Stage-1 loss + λ × Stage-2 loss`。

## 4. 只影响观察口径的 bug

### 4.1 P1：W&B 的 Stage-2 episode 去重发生碰撞

[`wandb_metrics.py`](../../../examples/claudecode_ags/wandb_metrics.py) 的 `_episode_key` 使用：

```text
(instance_id, group_index, index, rollout_id, sample_kind)
```

但它没有使用真正区分 branch 的 `branch_uid` 或 `step_t`。同时，`live_branch_runner` 生成 branch `index` 时包含 `source_trial_idx` 和 `branch_idx`，却不包含 `step_t`。同一失败 trial 的同一个 `branch_idx` 如果从多个 edit 点分叉，就会在 W&B 统计中被合并。

用 `branch_uid` 对 rollout dump 重新计数后：

| step | W&B 展示的 Stage-2 | 真实 Stage-2 episode |
|---:|---:|---:|
| 0 | 79，resolved 41.77% | 178，resolved 36.52% |
| 1 | 115，resolved 57.39% | 406，resolved 59.85% |
| 2 | 192，resolved 40.63% | 459，resolved 42.48% |
| 3 | 118，resolved 53.39% | 360，resolved 60.83% |
| 4 | 72，resolved 45.83% | 207，resolved 53.62% |

因此，W&B 的 `outcome/stage-2/*` 在这次 run 中不可信。顶层 `outcome/resolved_rate` 只统计 Stage-1，且 Stage-1 trial 的 index 唯一，所以第 2 节中的 Stage-1 数字不受该问题影响。

### 4.2 P1：退出码与排队时间不足，超时率无法可靠解释

这次 running job 的 Stage-2 样本缺少可靠的 `agent_exit_code` 和独立 `agent_queue_wait_sec`；旧口径还会把等待并发槽的时间算进 `agent_elapsed_sec`。所以“有多少 branch 真正把 Claude Code 的 45 分钟用满”不能从现有字段得到。

这主要污染**统计解释**：并发排队不会直接缩短 Claude Code 自己的 `time_budget_sec`。如果外层 guard 在排队时提前耗尽，个别 branch 也可能被丢弃，但目前没有证据表明它是 resolved rate 和 `grad_norm` 差距的主要来源。本文不采用此前“约 21% Stage-2 顶满 45 分钟”的说法。

当前 worktree 已有退出码、排队字段和“先拿并发槽再计时”的修正，但 running job 只有重提后才能使用新口径。

### 4.3 P2：bundle 目录被重复追加了一层

launcher 已把 `STEP_GRPO_BUNDLE_DIR` 设置为：

```text
.../step_reconstruct_bundles
```

而 `_bundle_root` 又追加一次 `step_reconstruct_bundles`，实际产物落在：

```text
.../step_reconstruct_bundles/step_reconstruct_bundles/...
```

这不改变训练结果，但会让运维脚本、磁盘清理和人工排查更容易找错目录。

### 4.4 P1：`run.log` 中记录了明文 W&B 凭据

启动命令被完整写进 `run.log`，其中包含 W&B credential。本文不引用该值。应轮换这份凭据，并让 launcher 和日志只输出脱敏形式。该问题不影响算法结果，但属于需要单独处理的安全问题。

## 5. 已排除或不应误判为根因的事项

### 5.1 不是题目顺序造成的

Hybrid 与朴素 GRPO 在 step 0、1、2、3、4 的 16 道题集合逐步完全相同。训练集本身允许跨 epoch 重复抽到旧题；这不等于数据损坏，也不能解释同一 step 上双方的差距。

### 5.2 当前 run 的 Stage-1 workspace 已与朴素 GRPO 对齐

本文接受并保留这一前提，不把 workspace 初始化差异列为本次 run 的根因。需要注意的是，两边仍走不同 runner 代码路径，但目前没有发现题目 metadata 或 grading plan 错位。

### 5.3 早期已修的 `loss_mask` 全 1 问题不是本次主因

本 run 的 `train/mis_rs_catastrophic_seq_fraction` 在 step 0–4 为 0%–0.28%，`pg_clipfrac=0`、`ppo_kl=0`，没有出现旧 run 中约一半序列被 RS 否掉的现象。因此，本次较小的 `grad_norm` 不能继续归因于旧的 branch `loss_mask` bug。

### 5.4 没有发现优化器停摆或权重未同步

日志中没有 NaN、整步 clip 或权重同步失败的证据。较小 `grad_norm` 更符合“独立轨迹被 prompt 级混合后发生方向抵消”，而不是“没有任何训练信号”。

## 6. 应如何解释这次实验

可以下的结论：

- 在相同 step 0–4 上，**当前这份 Hybrid 实现**的 Stage-1 resolved rate 低于朴素 GRPO，早期信号为负。
- 当前 `grad_norm` 确实更小，但它不是算法优劣指标；本 run 的损失聚合方式足以解释这种现象的一大部分。
- Stage-1 顶层 resolved rate 可以读取；Stage-2 W&B resolved rate 和旧超时率不能直接使用。

不能下的结论：

- 不能据此说“正确实现的 step-GRPO 打不过朴素 GRPO”。
- 不能把当前 Stage-2 的额外样本数当成有效 step-level 训练量。
- 不能用当前 checkpoint 做干净的 Hybrid 算法终点评测，再与朴素 GRPO 直接归因比较。

更准确的命名是：

> **workspace-only、post-edit、prompt-token-averaged branching RL**

即“只恢复代码、不恢复对话；从 edit 后继续；按题目 token 总量混合所有分支”的训练。它不是原计划中的 Hybrid step-GRPO。

由于 step 0 之后的 checkpoint 已经接受了上述错误目标的更新，修代码后从当前 checkpoint 续训仍不能形成干净对照。正式重跑应从共同 base checkpoint 开始，并使用新的 EXP_TAG。

## 7. 修复顺序与验收标准

### P0：先恢复正确训练语义

1. **真实捕获 transcript，并对空文件 fail closed。**
   - 从沙箱拉取 `.harness/trajectory.jsonl`，写入 bundle。
   - 只要该 trial 被用于 Stage-2，transcript 缺失、为空或不能与 tool snapshot 对齐，就应丢弃该 branch point，而不是启动失忆 agent。
   - 验收：所有产生 branch 的 bundle 都有非空、可解析的 stream-json；tool use/result 与 snapshot 能逐步对齐。

2. **从目标 edit 之前分叉。**
   - 对 `diff[i] != diff[i-1]` 的 edit 使用 pre-edit state `i-1`。
   - 验收：重建 workspace 中尚未包含被选中的 edit；conversation prefix 也结束在该 edit 之前。

3. **分开“调度身份”和“训练 episode 身份”。**
   - 16 个外层 prompt 可以继续作为调度/GBS 单元。
   - 每个 Stage-1 trial、每条 Stage-2 branch 必须有独立的 loss episode id；只有同一次尝试的 compact segments 共享该 id。
   - 不能只粗暴修改现有 `rollout_id` 而不检查 compact validator、动态 GBS 和 loss reducer；应显式增加 episode/loss key，或提供 Hybrid 专用 reducer。

4. **显式定义两阶段权重。**
   - 推荐先分别求均值，再组合，例如：`L = L_vanilla + λ × L_branch`。
   - 分别记录两部分的 token 数、loss 和 grad contribution。
   - 验收：改变 branch 数量或平均长度时，只要 `λ` 不变，两阶段的名义权重就不应被动漂移。

### P1：修正可观测性与测试

5. **W&B 用 `branch_uid` 去重。**branch episode key 至少包含 `step_group_key/step_t` 和 `branch_idx`；branch `index` 也应避免跨 step 冲突。

6. **补端到端不变量测试。**至少覆盖一条真实或最小可运行链路：
   - Stage-1 运行后 transcript 非空；
   - 选中 edit 后，branch workspace 与 transcript 都处于 pre-edit 状态；
   - K 条 branch 有 K 个独立 episode 身份；
   - W&B episode 数等于 `branch_uid` 数；
   - 空 transcript 必须报错，不能 fallback 成新 agent。

7. **保留新的退出码与排队字段。**用 `exit_code=-1`、不含排队的 `agent_elapsed_sec` 和独立 `agent_queue_wait_sec` 重报超时率。

### 最小重跑 Gate

在再次占用 16 卡长跑前，建议只做三个小检查：

1. **Stage-1-only 消融：**使用相同 Hybrid Stage-1 runner，但设 `λ=0`。在相同题目和初始权重下，其训练 loss 口径应与朴素 GRPO 对齐。
2. **单题 branch 不变量：**证明 transcript 非空、分叉发生在 edit 前、8 条 branch 身份互异。
3. **1–2 step 小跑：**同时记录 `L_vanilla`、`L_branch`、显式 `λ`、两部分 active token 和各自梯度统计，再决定是否恢复完整 16 卡实验。

## 8. 证据索引与审计边界

主要证据：

- Hybrid rollout dump：`.../qwen35_9b_cc_ags_2node_hybrid_c64_t45/rollout_dumps/rollout_{0..4}.pt`
- 朴素 GRPO dump：`.../qwen35_9b_cc_ags_2node_grpo_c64_t45/rollout_dumps/rollout_{0..4}.pt`
- Hybrid bundle：`.../step_reconstruct_bundles/step_reconstruct_bundles/`
- Hybrid 训练日志：`.../qwen35_9b_cc_ags_2node_hybrid_c64_t45/run.log`
- 核心实现：`hybrid_generate.py`、`live_runners.py`、`edit_ppl.py`、`workspace_rebuild.py`、`wandb_metrics.py`
- loss 聚合：`slime/ray/rollout.py`、`slime/backends/megatron_utils/cp_utils.py`

审计边界：

- 统计截点只有 step 0–4；后续 step 不能由本文外推。
- running job 启动后，worktree 又加入了退出码、排队时间和 workspace 参数修正。本文对训练语义的结论均同时由 rollout/bundle 实物验证，不只依赖审计时 HEAD。
- 本文没有把迁移来源当作运行代码；`slime-tencent` 只用于核对原始的 pre-edit 与 transcript-reseed 语义。
