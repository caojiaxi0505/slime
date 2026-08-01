# Readfix 后 Hybrid / Stage-2-only、readfix 前 Hybrid 与朴素 GRPO 对比

日期：2026-07-26  
分支：`feature/cc-ags-swe`

## 1. 对比对象

本文记录四条训练线在 SWE-Gym 训练集上的 Stage-1 `outcome/resolved_rate` 对比。所有数值均使用
`128` 分母，即每个 rollout step 的 `16` 道题 × `8` 次尝试。

| 名称 | 含义 | 数据来源 |
|---|---|---|
| readfix 后 fullcont | 应用 readfix 后的正常 Hybrid full-continuation 训练。Stage-1 loss 与 GRPO 对齐，Stage-2 continuation 作为附加训练信号 | `qwen35_9b_cc_ags_hybrid_s0_fullcont_s23_swe_8gpu_readfix_v1` |
| readfix 后 Stage-2-only | 应用 readfix 后的 Stage-2-only 消融。Stage-1 仍执行，但 `stage1_loss_weight=0`，只训练 Stage-2 continuation | step 0–7：`qwen35_9b_cc_ags_hybrid_s0_stage2only_s23_swe_8gpu_readfix_v1`；step 8–13：从 `iter_7` 续跑的 `qwen35_9b_cc_ags_hybrid_s0_stage2only_s23_swe_8gpu_readfix_resume7_v1` |
| readfix 前 Hybrid | readfix 前的 full-continuation Hybrid | step 0：`qwen35_9b_cc_ags_hybrid_s0_fullcont_s43_swe_8gpu_1step_v1`；step 1–13：`qwen35_9b_cc_ags_hybrid_s0_fullcont_s22_swe_8gpu_resume0_v1` |
| 朴素 GRPO | 2 节点朴素 GRPO，对照线 | step 0–4：`qwen35_9b_cc_ags_2node_grpo_t30_unlimited_loop3_reward0_save5_v3`；step 5–13：`qwen35_9b_cc_ags_2node_grpo_t30_unlimited_loop3_reward0_save5_v3_resume_iter4` |

这里的 readfix 指 Stage-2 分叉过滤逻辑的修正：不再因为 Stage-1 读过 testbed 外路径就直接拦截
分叉，重点拦截会改变 testbed 外状态、或无法可靠恢复 workspace 的情况。

Stage-2-only 不是正常 Hybrid。它用于观察“只训练分叉续跑”对完整 episode 行为的影响，不能直接
当作完整 Hybrid 与 GRPO 的公平对照。

## 2. 逐 step resolved_rate

| step | readfix 后 fullcont | readfix 后 stage2-only | readfix 前 Hybrid | 朴素 GRPO |
|---:|---:|---:|---:|---:|
| 0 | 81/128 = 63.28% | 83/128 = 64.84% | 87/128 = 67.97% | 76/128 = 59.38% |
| 1 | 91/128 = 71.09% | 87/128 = 67.97% | 96/128 = 75.00% | 82/128 = 64.06% |
| 2 | 87/128 = 67.97% | 87/128 = 67.97% | 80/128 = 62.50% | 76/128 = 59.38% |
| 3 | 103/128 = 80.47% | 101/128 = 78.91% | 100/128 = 78.12% | 101/128 = 78.91% |
| 4 | 86/128 = 67.19% | 88/128 = 68.75% | 82/128 = 64.06% | 83/128 = 64.84% |
| 5 | 81/128 = 63.28% | 71/128 = 55.47% | 85/128 = 66.41% | 74/128 = 57.81% |
| 6 | 91/128 = 71.09% | 92/128 = 71.88% | 77/128 = 60.16% | 72/128 = 56.25% |
| 7 | 80/128 = 62.50% | 64/128 = 50.00% | 66/128 = 51.56% | 78/128 = 60.94% |
| 8 | - | 77/128 = 60.16% | 73/128 = 57.03% | 78/128 = 60.94% |

注意：

- `readfix 后 fullcont` 当前只有 `rollout_0.pt` 到 `rollout_7.pt`，因此 step 8 暂无数据，用 `-` 占位。
- `readfix 后 stage2-only` 的 step 8 使用从 `iter_7` 续跑后的连续口径，即
  `qwen35_9b_cc_ags_hybrid_s0_stage2only_s23_swe_8gpu_readfix_resume7_v1/rollout_8.pt`。
- 旧 stage2-only 原始 run 曾跑到 `rollout_9.pt`，但在 rollout 9 的 actor train 阶段 CUDA OOM，
  因此主表不把这条失败分支作为连续训练口径继续展开。

## 3. 汇总

共同可比区间是 step 0–7，因为 readfix 后 fullcont 尚无 step 8。

| 区间 | readfix 后 fullcont | readfix 后 stage2-only | readfix 前 Hybrid | 朴素 GRPO |
|---|---:|---:|---:|---:|
| step 0–7 | 700/1024 = 68.36% | 673/1024 = 65.72% | 673/1024 = 65.72% | 642/1024 = 62.70% |

step 8 单独看，readfix 后 stage2-only 为 `77/128 = 60.16%`，readfix 前 Hybrid 为
`73/128 = 57.03%`，朴素 GRPO 为 `78/128 = 60.94%`。readfix 后 fullcont 暂无 step 8 数据。

## 4. 结论

截至 step 0–7，readfix 后 fullcont 在训练集 rollout resolved_rate 上是当前最强的一组：
`700/1024 = 68.36%`，高于朴素 GRPO 的 `642/1024 = 62.70%`，多 `58` 条 resolved，约
`+5.66 pct`。

逐 step 比较 readfix 后 fullcont 与朴素 GRPO，step 0–7 是 `8 胜 0 负`。这说明 readfix 后的
正常 full-continuation Hybrid 目前不是偶然单步偏高，而是在共同可比区间内稳定高于朴素 GRPO。

Stage-2-only 仍只是消融。它在 step 0–7 的平均值也高于朴素 GRPO，但低于 readfix 后 fullcont；
因此当前更合理的主线仍是正常 Hybrid：Stage-1 loss 与 GRPO 对齐，Stage-2 continuation 作为附加
训练信号。

## 5. 后续查看时的注意事项

- 本文只比较训练集 rollout 的 Stage-1 `resolved_rate`，不是 SWE-bench Verified 评测结果。
- `-` 表示该 run 当前还没有对应 step 的 rollout 数据，而不是 resolved 为 0。
- 如果继续追加 step 8 之后的数据，应使用同一口径：只看 Stage-1/root episode 的
  `outcome/resolved_rate × 128`。
- 对 Hybrid rollout dump 直接统计时，需要只取 `sample_kind=vanilla`，并按 `branch_uid` 去重；
  Stage-2 branch 不计入这张表。
