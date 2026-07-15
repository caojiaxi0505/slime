# Step-GRPO 实验手册（设计 · 优势 · 指标 · 踩坑）

日期：2026-07-14  
范围：Path A `claudecode_ags` hybrid step-GRPO vs 朴素 GRPO  
详设：`specs/2026-07-12-path-a-hybrid-step-grpo-design.md` · 算法终局：`/mnt/sn-007/jiaxicao/code/算法设计版本7.md`

**向上汇报版（少技术细节）：**  
- 负责人版：`2026-07-14-step-grpo-report-manager.md`  
- Tech lead / 组内版：`2026-07-14-step-grpo-report-tech-lead.md`

---

## 0. 当前在等什么

| 优先级 | 动作 |
|--------|------|
| P0 | 等 **`jiaxicao-hybrid-2node-full`**（`qwen35_9b_cc_ags_2node_hybrid_full`，16卡 RBS=16）跑出若干 step，验收 loss_mask 修复后的 `mis_*` / `grad_norm` |
| P1 | 朴素 GRPO `grpo-debug` 继续跑作基线 |
| P2 | 本笔记补「结论」栏；**暂不加 OPSD** |

**full 验收 gate（log）：** 不应再每步 `vanilla_groups=8 (std0=8)`；`train/grad_norm` 应明显高于消融的 ~0.04。

---

## 1. 三条对照实验

| 线 | Job | wandb / LOG | 代表什么 |
|----|-----|-------------|----------|
| 朴素 GRPO | `jiaxicao-grpo-1node-debug` | `…_grpo_debug` | 基线：整轨迹 reward，prompt 级 K=8 对比 |
| Stage-2-only 消融 | `jiaxicao-hybrid-1node-debug` | `…_hybrid_ltpa_v2` | **意外消融**：vanilla episode key bug → Stage-1 不训，只训 branch |
| 完整 step-GRPO | `jiaxicao-hybrid-1node-full` | `…_hybrid_full` | 修 bug 后正牌 hybrid：Stage-1 + Stage-2 都训 |

共性：`PROMPT_DATA=train_grpo_resolved_1_7` · `SAVE_INTERVAL=5`（full）· hybrid `RBS=8, K=8`。

**对比注意：** hybrid 一步墙钟 ~2.7h，GRPO ~1.1h → 按 **等 wall-clock** 或 **等 train step** 对齐，不能只看 step 编号。

---

## 2. Step-GRPO（hybrid）设计 — 一句话 + 数据流

**一句话：** Stage-1 用朴素 GRPO 探索整题；Stage-2 在**高 edit-PPL 的 patch 步**分叉 K 条续跑，在**分叉点组内**再做 GRPO，把信用分配到「哪一步改法更好」。

```
每道题（外层 n_samples=1，内层 K=8）
│
├─ Stage-1：K 条 vanilla（AGS + CC + 快照 + 评测）
│     → segment fan-out，sample_kind=vanilla，按 prompt group 做 GRPO
│
├─ 若全对 → 结束（无 Stage-2）
│
├─ Stage-1.5：失败 trial 的 patch turn 进池 → edit-PPL 取 top-B（B≤K）
│
└─ Stage-2：每个入选 (trial, step_t) → rebuild + prefix re-seed → K 条 branch
      → step_group_key = {group}:{trial}:{step_t} 内 GRPO
      → 训练整条 continuation（非 first_action only）
```

**对比组语义**

| 类型 | 分组键 | 组内有几条 | 比的是什么 |
|------|--------|------------|------------|
| vanilla | `group_index`（8 次 trial） | K=8 episode | 整题谁解出 |
| branch | `step_group_key` | K=8 branch | 同一 patch 步哪种续跑更好 |

**filter（`STEP_GRPO_FILTER=1`）：** 组内 reward 方差=0 → 整组 `remove_sample`（无对比不学）。  
**reward：** 仍用 Path A `rewards.default.compose`（F2P 为主，P2P 辅助）。

**本轮未做：** v7 全量「每个 $s_t$ 都分叉」、OPSD、hindsight 自蒸馏。

---

## 3. 相对朴素 GRPO 的理论优势

| 维度 | 朴素 GRPO | Hybrid step-GRPO |
|------|-----------|------------------|
| 信用分配 | 整条轨迹一个 $R$；advantage 广播到所有 token | Stage-2 在**关键 patch 步**分叉，组内对比更贴「改这一步」 |
| 探索 | 只有「重跑整题」 | 额外有「从失败中间状态续跑」 |
| 样本效率 | 失败轨迹 reward=0，信号弱 | 失败 trial 的 patch 步可进 Stage-2，失败里挖局部对比 |
| 方差 | 长轨迹全对/全错 → 组内 std=0 常见 | Stage-2 组更小、更局部，有机会在 sub-trajectory 上产生方差 |

**要证的是：** 在相近 wall-clock / 相近 train step 下，**主指标更好或相当，且成本可接受**；full vs 消融能分出 Stage-1 贡献。

**不能单靠理论推出：** step-GRPO 一定涨分（分叉噪声、filter 丢组、墙钟贵都会抵消优势）。

---

## 4. Wandb / log：什么指标能说明什么

### 4.1 主结论（与 GRPO 直接比）

| 指标 | 含义 | 理论优势若成立，期望 |
|------|------|----------------------|
| **`outcome/resolved_rate`** | Stage-1 解题率（**与 GRPO 同口径**） | full hybrid **≥ GRPO**，且随 step 更稳/更快爬升 |
| `outcome/test_f2p_macro_pass_rate` | F2P 宏平均 | 与 resolved 同向，辅助 |
| `rollout/episode_reward/std` | Stage-1 组内 reward 离散度 | 不必更大；有对比即可（非全 0/1 塌缩） |

Stage-2 质量（step-GRPO 特有，**不能**与 GRPO 顶层直接比）：

| 指标 | 含义 |
|------|------|
| `outcome/stage-2/resolved_rate` | 分叉续跑解出率 |
| `rollout/stage-2/episode_reward/std` | Stage-2 组内方差（有对比才训） |

### 4.2 证明「Stage-2 在工作」（机制指标）

| 指标 / log | 证明什么 |
|------------|----------|
| `perf/step_grpo/n_stage2_episodes` | Stage-2 有产出 |
| `perf/step_grpo/n_stage2_samples` | branch 片段量（≠ 有效训练量） |
| log `branch_groups=N (std0=M)` | 对比组数；`N-M` = 有信号的组 |
| log `vanilla_groups=8 (std0=?)` | full：**std0 应 < 8**；消融：恒 8/8 |
| `train/grad_norm` | full 应 **≫ 消融**（~0.04），与 GRPO 同量级才有 Stage-1 信号 |
| `train/pg_loss` | 非零且随 step 变化 → 优化器在吃信号 |

### 4.3 证明「Stage-1 有贡献」（full vs 消融）

| 对比 | 若 Stage-1 有用 |
|------|-----------------|
| full `outcome/resolved_rate` vs 消融 | full **更高或涨得更快** |
| full `train/grad_norm` vs 消融 | full **明显更大** |
| full vanilla `std0` vs 消融 | full **不是 8/8** |

消融单独：**不能**用 `outcome/resolved_rate` 当训练效果（Stage-1 只 rollout 不反传），只能当环境采样质量。

### 4.4 成本（优势是否值得）

| 指标 | 含义 |
|------|------|
| `perf/rollout_time` | 整步 rollout 墙钟（GRPO 可比） |
| `perf/step_grpo/rollout_time` | hybrid 整轮 |
| `perf/step_grpo/stage1_wall/*` · `stage2_wall/*` | Stage-1 / 2 耗时拆解 |
| `perf/step_grpo/prompt_wall/*` | 单题 hybrid 总墙钟 |
| `perf/agent_time/*` vs `perf/stage-2/agent_time/*` | Agent 时间分布 |

**性价比：** 同等 wall-clock 下 resolved 曲线谁更高；或同等 resolved 谁 step 更少。

### 4.5 训练侧（框架通用，两边都有）

| 指标 | 用途 |
|------|------|
| `train/actor_train_time` | 反传墙钟 |
| `train/kl_loss` | ref KL |
| `train/mis_*` | off-policy / MIS 健康度 |

### 4.6 易误读

| 误读 | 正解 |
|------|------|
| `n_stage2_samples` 大 = 训得多 | 含 filter 前片段；看 `branch_groups - std0` |
| `outcome/resolved_rate` 高 = hybrid 训得好 | 消融里 Stage-1 可不驱动更新 |
| `grad_norm` 小 = 没训 | 消融仍靠 Stage-2 在训，只是信号弱 |
| P2P 高 = 题解了 | **主看 resolved / F2P** |

---

## 5. 踩坑清单（按严重度）

| # | 现象 | 根因 | 修 / 避 |
|---|------|------|---------|
| 1 | `missing_eval_plan`，reward 全 0 | hybrid `_evaluate_diff` 未传 `FAIL_TO_PASS` 等 metadata | `live_runners` 对齐 `generate.py` 传 metadata |
| 2 | Stage-1 每步 `std0=8/8`，grad_norm ~0.03 | 共享 `rollout_id` 被 `_branch_key` 当 episode id，K trial 合成 1 条 | vanilla 用 `branch_uid=v:{g}:t{trial}`；见 `2026-07-14-hybrid-vanilla-episode-key-bug.md` |
| 3 | compact / 切分异常 | 兄弟 segment 缺统一 `rollout_id` | `_stamp_shared_rollout_id`（与 #2 并存，各管一层） |
| 4 | Stage-2 静默跳过 / 对齐炸 | step↔turn 按个数对齐 | 按 `tool_use_id` join |
| 5 | 与 GRPO 曲线不可比 | 题池不同（`resolved_0_7` vs `1_7`） | 统一 `train_grpo_resolved_1_7` |
| 6 | wandb 口径混 | Stage-2 混入顶层 `outcome/*` | 顶层=Stage-1；`outcome/stage-2/*`、`perf/step_grpo/*` |
| 7 | branch 组变少 | 跑挂（超时/沙箱/apply 失败） | log `[hybrid] dropped branch`；≠ std0 filter |
| 8 | 并行第二 job 抢 adapter | 共用 workload label | 每 job 独立 `WORKLOAD_LABEL` + ALB |
| 9 | hybrid `mis_kl`≫GRPO，`grad_norm` 极低，RS catastrophic ~50%+ | branch 强制 `loss_mask` 全 1，把 tool/context 的 `rollout_log_probs=0.0` 占位符送进 TIS/RS | **勿**覆盖 mask；保留 `merge_turns` 合同。见 `2026-07-14-hybrid-branch-loss-mask-tis-rs-bug.md` |

---

## 6. 日志自检（30 秒）

```text
# full hybrid 正常
[step_grpo_adv] filter: vanilla_groups=8 (std0=< 8) branch_groups=... (std0=...)
[step_grpo_adv] rows=... vanilla=... branch=...
train/grad_norm: ~0.1–2（与 GRPO 同量级，非 ~0.03）

# 仍像消融（bug 未修或旧 job）
vanilla_groups=8 (std0=8) 每步出现
train/grad_norm ~0.03–0.04
```

---

## 7. 结论栏（待 full 数据后填）

| 问题 | GRPO | 消融 | full | 备注 |
|------|------|------|------|------|
| step N 的 `outcome/resolved_rate` | | | | |
| 等 wall-clock 下 resolved 趋势 | | | | |
| `train/grad_norm` @ step 5 | | | | |
| vanilla `std0` @ step 5 | 8/8 | | | |
| Stage-1 是否必要（full vs 消融） | — | — | | |
| 是否值得 scale | | | | |

---

## 8. 路线图（简）

```
GRPO 基线 → 完整 step-GRPO（当前）→ 读表 §7 → 若有效则 scale
                                    → 若 full≈消融则查 Stage-1 / filter
                                    → 之后再考虑 OPSD（v7 §5–8，非 slime --use-opd）
```

相关笔记：`2026-07-13-hybrid-step-grpo-debug-fixes.md`（修 bug 索引）· `2026-07-14-hybrid-vanilla-episode-key-bug.md`（消融解读）· `2026-07-14-hybrid-branch-loss-mask-tis-rs-bug.md`（TIS/RS 异常）
