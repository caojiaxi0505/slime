# Hybrid vanilla episode key bug → 无意的 Stage-2-only 消融（2026-07-14）

## 结论（先看这）

正在跑的 **`jiaxicao-hybrid-1node-debug`** / LOG **`qwen35_9b_cc_ags_1node_hybrid_ltpa_v2`**  
在修复合并进该 job 之前，应解读为：

> **无意的 Stage-2-only 消融**：Stage-1（vanilla）GRPO 对比信号几乎全废，真正更新主要来自 Stage-2（branch）。

**请勿删、勿停该实验**——当作消融基线保留完整曲线。代码修复只影响之后新提交的 run。

---

## 现象

| 指标 | Hybrid（ltpa_v2） | 朴素 GRPO |
|------|-------------------|-----------|
| `train/grad_norm` | ~0.03–0.04 | ~0.5–2 |
| filter 日志 | **每步** `vanilla_groups=8 (std0=8)` | 无此 filter |
| Stage-1 resolved | 可波动上涨 | 有对比信号 |

题量只差约一倍（RBS 8 vs 16），**不能**解释「Hybrid 永远全组 std0」。  
若 Stage-1 reward 真是独立 0/1、resolved≈50–80%，「8 个 prompt 组全部同分」的概率接近 0，却每步都出现 → **不是真同分**。

---

## 根因

1. §1 要求：同一次 `hybrid_generate` 的兄弟样本 **必须共享** `rollout_id`（slime compact 校验）。
2. `step_grpo_advantage._branch_key` 在缺少 `branch_uid` 时回退到 **`rollout_id`**。
3. Branch 有独立 `branch_uid` → 正常。  
   Vanilla 只有 `trial_idx`、**没有** `branch_uid` → K 次尝试被合成 **1 个 episode** → `len(rewards)==1` → `std=0` → filter 整组扔掉；advantage 也没有组内对比。

相关代码：

- `live_runners._shared_rollout_id` / fan-out 强制同 `rollout_id`
- `step_grpo_advantage._branch_key`（修前：`branch_uid` → `rollout_id`）
- filter：`STEP_GRPO_FILTER=1` 时 `std_zero` → `remove_sample`

---

## 对当前 run 的解读

- **不是「完全没训练」**：Stage-2 branch 仍有方差组在训（日志里 `branch_groups` 去掉 `std0` 后仍有剩余），`pg_loss` / 很小的 `grad_norm` 来自这里。
- **是「Stage-1 白跑」**：顶层 `outcome/*` 仍反映 Stage-1 rollout 质量，但 **不驱动** 本 run 的参数更新。
- 与朴素 GRPO 比学习曲线时：本 run ≠ 完整 hybrid；完整 hybrid 需修后重跑。

对照 job（勿混）：

| 角色 | Job | LOG / wandb group |
|------|------|-------------------|
| Stage-2-only 消融（本 bug） | `jiaxicao-hybrid-1node-debug` | `qwen35_9b_cc_ags_1node_hybrid_ltpa_v2` |
| **完整 step-GRPO（修后）** | `jiaxicao-hybrid-1node-full` | `qwen35_9b_cc_ags_1node_hybrid_full` |
| 朴素 GRPO | `jiaxicao-grpo-1node-debug` | `qwen35_9b_cc_ags_1node_grpo_debug` |

**实验总手册：** `notes/2026-07-14-step-grpo-experiment-handbook.md`

---

## 修复（已合入本地代码；**当前 Running job 未热更**，修后新 run 才生效）

- `_branch_key`：vanilla 优先 `branch_uid` / `trial_idx`，**不再**用共享 `rollout_id` 当 episode id。
- `live_runners` vanilla metadata 打 `branch_uid=v:{group}:t{trial}`；`hybrid_generate` 兜底 `setdefault`。
- 回归：`test_shared_rollout_id_does_not_collapse_vanilla_trials`（4 passed）。

共享 `rollout_id` 本身 **保留**（§1 仍需要）。未 commit / 未重启 `jiaxicao-hybrid-1node-debug`。
