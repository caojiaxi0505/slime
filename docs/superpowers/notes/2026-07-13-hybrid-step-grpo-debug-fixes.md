# Hybrid step-GRPO 调试修复（2026-07-13）

白话汇总：跑 1-node hybrid 时踩到的坑与改法。每条都能点到代码。

---

## 1. 同一次 hybrid 的片段必须共用 `rollout_id`

**现象：** compact / 训练切分把同 prompt 的片段当成多次独立 rollout。  
**改法：** `hybrid_generate` / fan-out 强制兄弟样本共享父 `rollout_id`。  
**位置：** `examples/claudecode_ags/step_reconstruct/hybrid_generate.py`（`_stamp_shared_rollout_id`）、`live_runners.py`（`fan_out_sample_segments(..., rollout_id=...)`）

---

## 2. Step↔turn 对齐：按 `tool_use_id`，不能只数个数

**现象：** PostToolUse steps 与 adapter turns 数量常对不上 → 以前静默跳过 Stage-2；改成按序对齐后又因「多出来的 tool」整 trial 报错。  
**改法：** turn 上记下 minted `tool_use_id`，step 上取 payload 的 id，按 id join；缺 hook 的 emit 可丢（`dropped_no_hook`），缺 id 才严格失败。  
**位置：**  
- `step_reconstruct/edit_ppl.py` → `align_logprobs_to_steps`  
- `…/anthropic_segmented.py`（turn_log 记 id）  
- `…/session_capture.py`（`StepRecord.tool_use_id`）  
- `live_runners.py`（对齐入口）

---

## 3. Reward 全 0：评测没带上题目元数据（真 bug）

**现象：** `resolved_rate=0`，`base_eval.reason=missing_eval_plan`；patch 能 apply，但**根本没跑测**。  
**原因：** 朴素 GRPO 的 `generate.py` 评测传 `metadata={**sample.metadata, **md}`（含 `FAIL_TO_PASS`/`repo`）；hybrid `live_runners` 漏传 → 评测模式 `NONE`。  
**改法：** vanilla / branch 的 `_evaluate_diff` 都补传 metadata；bundle 的 `task_metadata` 也一并带上。  
**位置：** `step_reconstruct/live_runners.py`（两处 `_evaluate_diff`）；对照 `generate.py` 同名调用。

---

## 4. Wandb：Stage-1 与 GRPO 同口径，Stage-2 / 时间另记

**规则：**  
- 顶层 `outcome/*`、`rollout/*`、`traj/*`、`perf/agent_time`… = **仅 Stage-1**（可和朴素 GRPO 直接比）  
- `*/stage-2/*` = Stage-2  
- `perf/step_grpo/*` = hybrid 整轮墙钟、stage 墙钟、样本数  
- 训练墙钟仍用框架 `perf/actor_train_time`（`train/step`，两边同 key）  

**位置：** `examples/claudecode_ags/wandb_metrics.py`；耗时字段由 `live_runners.py` / `hybrid_generate.py` 写入 metadata。

---

## 5. 对比实验先用同一训练集

**说明：** 原先 hybrid 默认 `train_step_grpo_resolved_0_7`（含从未解出的题），GRPO 用 `train_grpo_resolved_1_7`。首轮 resolved=0 **不全是模型差**，题池更难 + 曾有 §3 bug。  
**改法：** hybrid 默认改与 GRPO 相同：`train_grpo_resolved_1_7.slime.jsonl`。  
**位置：** `launch/run_hybrid_1node_debug.sh`、`launch/hybrid_1node_job/submit_job.sh`（`PROMPT_DATA`）

---

## 快速自检

| 看什么 | 正常 | 仍有 §3 bug |
|--------|------|-------------|
| vanilla log | `reward=` 有 0/1 | 全 0 且 dump 里 `missing_eval_plan` |
| wandb | `outcome/resolved_rate`≈Stage-1；有 `outcome/stage-2/*` | 只有假 0、无 f2p |
| PROMPT_DATA | `…/train_grpo_resolved_1_7.slime.jsonl` | `…resolved_0_7…` |
