# Hybrid branch `loss_mask` 全 1 覆盖破坏 TIS/RS（2026-07-14）

## 白话

训练时有两个计分员：采样时（SGLang）和训练时（Megatron）。两边分数差太多，TIS/RS 会把样本砍权或丢掉。

朴素 GRPO 两边几乎一致；hybrid 却有一半左右序列被 RS「整条否掉」，`grad_norm` 比 GRPO 低一个数量级。

根因不是 TIS/RS 坏了，而是 **Stage-2 branch 把「环境塞回来的内容」也标成了要训练**，而这些位置上只有假的采样分数占位符（0.0），两边一对就炸。

---

## 现象（日志 / wandb）

| 指标（step0） | 朴素 GRPO | full hybrid | 消融（几乎只训 branch） |
|---------------|-----------|-------------|-------------------------|
| `train/mis_kl` | ~0.001 | ~2.32 | ~2.7–3.7 |
| `train/mis_rollout_log_ppl` | ~0.33 | **~0.067（≈0）** | ~0.04–0.05 |
| `train/mis_training_ppl` | ~1.4 | **~695** | ~100+ |
| `train/mis_rs_catastrophic_seq_fraction` | **0** | **~53%** | **~60–83%** |
| `train/grad_norm` | ~0.8–1.4 | ~0.064 | ~0.036 |

`rollout_log_ppl ≈ 0` 强烈暗示：参与 IS 的 token 上，rollout logprob 大量是 **0.0 占位符**（不是真实模型 token 分数）。

---

## 根因

### 正确合同（`merge_turns`）

| token 类型 | `loss_mask` | `rollout_log_probs` |
|------------|-------------|---------------------|
| 模型生成（assistant） | 1 | SGLang 真实 logprob |
| 工具/环境回灌（context_tail） | 0 | `0.0` 占位（不参与 loss/IS） |

### 错误实现

`live_branch_runner` 在 fan-out 后强制：

```python
s.loss_mask = [1] * len(s.loss_mask)  # 把 context 也改成要训
# 且不更新 rollout_log_probs
```

`hybrid_generate` 在 `loss_mask is None` 时也会用全 1 兜底（同类风险）。

设计里的「整条 continuation」本意是：**相对 first_action，续跑里所有模型生成步都训**；不是把 tool 结果也当 policy token。

### 因果链

```
branch 强制 loss_mask=全1
  → tool/context 的 rollout_lp=0.0 进入 TIS/RS
  → Megatron 重算这些位置 → logprob 很负
  → w=exp(train-rollout)≈0 → RS catastrophic veto
  → 有效样本骤减 → grad_norm 远低于 GRPO
```

### 已排除

- 两边 TIS/RS 配置相同；GRPO 同配置正常 → 非阈值问题  
- 日志无 `turn logprob length mismatch; zeroing` → 非整 turn logprob 被清零  
- edit-PPL 的 `turn_logprobs` 不写 `sample.rollout_log_probs`  
- Stage-1 vanilla 未做 mask 覆盖（与 GRPO 同路径）

---

## 修复

**改动：**

1. `examples/claudecode_ags/step_reconstruct/live_runners.py`  
   - 删除 branch fan-out 后的 `loss_mask` 全 1 覆盖  
   - 保留 `merge_turns` / `fan_out_sample_segments` 给出的 mask

2. `examples/claudecode_ags/step_reconstruct/hybrid_generate.py`  
   - 删除 `loss_mask is None` 时填全 1 的兜底

3. `tests/claudecode_ags/test_step_reconstruct_live_runners.py`  
   - 断言 branch 保留 `[1,0,1]` 这类「模型/context/模型」mask，且 `rollout_log_probs` 不被改写

**「整条 continuation」如何仍被满足：**  
`merge_turns` 已对 continuation 中每一段 assistant `output_ids` 置 `loss_mask=1`；只需别再把 context_tail 强行打开。

---

## 验收（需重提 hybrid job 后看）

修后期望（相对 GRPO 同量级）：

- `train/mis_kl` → ~0.001 量级（不再 >2）  
- `train/mis_rs_catastrophic_seq_fraction` → ~0  
- `train/mis_rollout_ppl` 与 `mis_training_ppl` 接近（不再差两个数量级）  
- `train/grad_norm` 明显高于修前的 ~0.06，并更接近 GRPO  

**重提记录（2026-07-14）：**

- 已停：`jiaxicao-hybrid-1node-full`、`jiaxicao-hybrid-1node-debug`（消融）  
- 新提：**`jiaxicao-hybrid-2node-full`**（16 GPU = 2×8）  
  - RBS/GBS **16**（对齐朴素 GRPO 的 RBS=16；原先 hybrid RBS=8）  
  - 含本 loss_mask 修复 + vanilla episode key 修复  
  - LOG / wandb：`qwen35_9b_cc_ags_2node_hybrid_full`  
  - `NUM_ROLLOUT=44`，`SAVE_INTERVAL=5`，独立 ALB `…-2node-full-adapter`  
- 朴素 GRPO `jiaxicao-grpo-1node-debug` 未动

---

## 相关

- 实验手册：`2026-07-14-step-grpo-experiment-handbook.md`（踩坑表已追加本条）  
- 前期修复索引：`2026-07-13-hybrid-step-grpo-debug-fixes.md`  
- TIS/RS 实现：`examples/train_infer_mismatch_helper/mis.py` + `mis.yaml`
