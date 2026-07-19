# Hybrid step-GRPO 踩坑手册（小白版）

日期：2026-07-15  
分支：`feature/cc-ags-swe`  
一句话：**这份文档讲「我们踩过哪些坑、怎么认、怎么修、修没修」**。不要求你先懂强化学习细节。

更细的单坑笔记、某次 run 审计，见文末「延伸阅读」。

---

## 0. 先花 30 秒搞清 Hybrid 在干什么

想象一道修 bug 的题：

1. **Stage-1（vanilla）**：让模型独立试 **K 次**（常见 K=8），像 8 个学生各自交作业。
2. **挑分叉点**：在失败作业里，找出「模型最不确定的那一次改代码」。
3. **Stage-2（branch）**：回到那个时刻，再分叉 **K 次**，比较不同续写谁最后修好。
4. **训练**：Stage-1 之间互相比较；同一分叉点上的 Stage-2 之间互相比较。

朴素 GRPO 只有第 1 步那种「整条作业比好坏」。Hybrid 多了「中途重来」这一段。

**重要：**代码写错时，曲线仍会动、resolved 仍会变——**不等于算法按设计在学**。下面每个坑都是「看起来在跑，实际学偏了」的例子。

---

## 1. 一张总表（先扫一眼）

| # | 坑（人话） | 你会看到什么 | 状态 |
|---|------------|--------------|------|
| A | 评测忘带题面元数据 | reward 全 0，`missing_eval_plan` | ✅ 已修 |
| B | 共享 `rollout_id` 把 8 次尝试合成 1 次 | 每步 `vanilla_groups=… (std0=…)`，`grad_norm` 极小 | ✅ 已修 |
| C | branch 把工具回包也标成「要训练」 | `mis_kl` 爆炸，一半序列被 RS 干掉 | ✅ 已修 |
| D | 排队时间算进 45 分钟预算 | Stage-2「顶满超时」虚高；exit_code 缺失 | ✅ 已修（须**重提** Job 才生效） |
| E | Stage-1 workspace 参数不全 | hybrid 沙箱准备和朴素 GRPO 不一致 | ✅ 已修 |
| F | SWE484 字段还 nested | 评测走错成 SIMPLE_CMD | ✅ 已改数据 |
| G | Stage-2 **对话历史是空的** | transcript 0 字节；branch 等于「失忆重开」 | ❌ 未修（P0） |
| H | 分叉点落在 **改完代码之后** | branch 继承可疑 patch，无法重选这次 edit | ❌ 未修（P0） |
| I | 一道题下成百 episode 共用一个 `rollout_id` | Stage-1 权重被 Stage-2 token 淹没 | ❌ 未修（P0） |
| J | W&B Stage-2 计数撞车 | `outcome/stage-2/*` 偏少/不准 | ❌ 未修（P1） |

图例：✅ 代码/数据已改 · ❌ 仍影响「算法是否公平」的解读

---

## 2. 已修好的坑（仍要会认，避免旧实验误读）

### A. Reward 全 0：评测根本没跑

**比喻：**交了作业，但老师没有试卷——全员 0 分。

**现象：**`resolved_rate≈0`，dump 里 `base_eval.reason=missing_eval_plan`；git patch 其实能 apply。

**原因：**朴素 GRPO 评测会带上 `FAIL_TO_PASS` / `repo` 等 metadata；hybrid 的 `live_runners` 曾漏传 → 评测模式变成 `NONE`。

**改法：**vanilla / branch 的 `_evaluate_diff` 都传完整 metadata（与 `generate.py` 对齐）。

**自检：**log 里应有 `reward=0/1` 混杂，且 f2p 不全是 `None`。

---

### B. Vanilla 假 std0：8 次尝试被当成 1 次

**比喻：**8 个学生各自交卷，系统却写「这是同一份卷子」→ 组内没人可比 → 整组扔进垃圾桶。

**现象：**

- filter 日志几乎每步：`vanilla_groups=8 (std0=8)`
- `train/grad_norm` 掉到 ~0.03–0.04（朴素 GRPO 常在 ~0.5–2）
- Stage-1 的 resolved 曲线还在动，但**几乎不驱动参数更新**

**原因（两步踩雷）：**

1. 同一次 hybrid 的片段必须共享 `rollout_id`（框架 compact 要求）——这没错。
2. 算 GRPO 组内对比时，又用 `rollout_id` 当「一次尝试」的钥匙 → K 条 vanilla 合成 1 个 episode → `std=0` → filter 丢掉。

**改法：**给每次 vanilla 打独立 `branch_uid`（如 `v:{group}:t{trial}`）；`_branch_key` 优先用 `branch_uid` / `trial_idx`，**不要**回退到共享的 `rollout_id`。

**自检：**修后不应再每步「vanilla 全 std0」。  
**旧 run 解读：**修前 hybrid ≈ **无意的 Stage-2-only 消融**，不能当完整 hybrid 和 GRPO 比。详见 `2026-07-14-hybrid-vanilla-episode-key-bug.md`。

---

### C. Branch `loss_mask` 全 1：把「环境回包」也拿去训

**比喻：**考试只该给「学生写的答案」打分，结果把「题目印刷页」也当成答案批改——计分器全乱。

训练里有两个计分员：采样时（SGLang）和训练时（Megatron）。差太多时，TIS/RS 会砍权或整条丢掉。

**现象（相对朴素 GRPO）：**

| 指标 | 正常大致样子 | 踩坑时 |
|------|----------------|--------|
| `train/mis_kl` | ~0.001 | ~2+ |
| `train/mis_rs_catastrophic_seq_fraction` | ~0 | ~50%+ |
| `train/grad_norm` | ~0.5–2 | 低一个数量级 |

**原因：**branch 路径强制 `loss_mask` 全 1，把 tool/context 占位 `rollout_log_probs=0.0` 也送进 IS。

**改法：****不要**覆盖 mask；保留 `merge_turns` 合同——模型 token=`1`，工具回包=`0`。

**自检：**`mis_rs_catastrophic_seq_fraction` 应接近 0；`mis_kl` 与 GRPO 同量级。详见 `2026-07-14-hybrid-branch-loss-mask-tis-rs-bug.md`。

---

### D. 排队烧超时：45 分钟「假顶满」

**比喻：**电影院限时 45 分钟看电影，但计时从「门口排队」就开始——队越长，真正能看的时间越短。

**现象：**Stage-2 扇出大、并发 64 时，大量 `agent_elapsed≈45min`；`agent_exit_code` 常为 `None`。

**原因：**

1. 先开外层 `asyncio.timeout`，再抢并发槽 → **排队计入超时**。
2. Claude 自己的 `time_budget_sec=2700` **不含排队**，两套口径打架。
3. 曾丢弃 `run_claude` 返回值 → 没有可靠 exit_code。

**改法：**先 `agent_concurrency_cm()`，拿到槽再开 guard / 再记 `agent_elapsed`；记录 `agent_exit_code`、`agent_queue_wait_sec`。

**自检：**新 Job 的 dump/metadata 应有 `agent_queue_wait_sec`；exit_code 不应大面积 `None`。  
**注意：**正在跑的旧 Job **不会热更新**，必须重提。

---

### E. Workspace 准备漏参数

**比喻：**朴素 GRPO 进考场会发完整试卷+草稿纸；hybrid 有时只发了题干。

**原因：**`live_vanilla_runner` / rebuild 曾没把 `base_commit`、`data_source` 等传给 `prepare_workspace`。

**改法：**与 `generate.py` 同参调用。

---

### F. SWE484 评测数据 nested

**比喻：**标准答案写在信封夹层里，阅卷只看封面 → 判成「没有标准答案」。

**原因：**`FAIL_TO_PASS` 等埋在 `extra_info.swebench.*`，顶层还有抢路径的 `eval_cmd` → 走 `SIMPLE_CMD` 而不是 `SWEBENCH`。

**改法：**flatten 到顶层；备份原文件为 `*.bak_nested`（可删，评测不依赖）。

---

## 3. 还没修、但已经会「假装在学」的坑（P0）

> 来源：对 2-node hybrid run `…_c64_t45` 的审计（`2026-07-15-hybrid-run-bmr3zlex-bug-report.md`）。  
> **结论：在这些修掉之前，不要用那条曲线宣称「step-GRPO 输给 / 赢过朴素 GRPO」。**

### G. Stage-2 对话历史是空的（失忆重开）

**比喻：**本来要「倒带重讲」；结果只留下改过的作业本，把之前的草稿和对话全扔了，叫一个没听过课的新同学接着写。

**证据：**大量 `transcript.jsonl` 为 **0 字节**；branch 读到空文件后 fallback 成「用原题 prompt 从头跑」。

**影响：**Stage-2 **不是** step-level 续写，advantage 进了训练，但语义错了。

**修的方向（尚未落地）：**Stage-1 把真实 trajectory 拉进 bundle；Stage-2 必须能 resume 非空前缀；空 transcript 应失败而不是静默 fallback。

---

### H. 分叉点在「改完之后」而不是「改之前」

**比喻：**想重考「这一题怎么答」；系统却把你已经写坏的答案先复印好，再让你在复印件上继续涂——没法重新选那一笔。

**原因：**PostToolUse snapshot `i` = 第 i 次工具**完成后**的仓库；实现却用 `step_t=i` 去 rebuild。应对「发生在 i 的 edit」回到 **`i-1`**（tencent 参考实现也是如此）。

**影响：**branch 继承同一份可疑 patch，学的是「带着错改怎么续」，不是「这次 edit 该选什么」。

---

### I. 一道题几百条 episode，损失却按 16 个 `rollout_id` 平均

**比喻：**班里 16 道题，每道题下面塞了几百份独立答卷；算分时却说「每道题只算一票」，票内按字数把所有答卷搅在一起——Stage-2 字多就几乎淹没 Stage-1。

**原因：**`_stamp_shared_rollout_id` 把同 prompt 下 Stage-1+Stage-2 全打成同一个 `rollout_id`；训练按 `rollout_id` 汇总 token 当分母。

**影响：**不是固定的「Stage-1 + λ×Stage-2」；λ 随 branch 数量/长度漂移；`grad_norm` 更容易互相抵消变小。

---

### J.（观察口径）W&B Stage-2 去重撞车

**现象：**W&B 上的 Stage-2 episode 数明显少于 dump 用 `branch_uid` 重数的真实值。

**原因：**`_episode_key` 没用 `branch_uid` / `step_t`。

**影响：**主要坑报表；顶层 Stage-1 `outcome/resolved_rate` 仍可用。

---

## 4. 怎么读实验（防踩「曲线陷阱」）

1. **先问：这次 Job 的代码包含哪些 fix？** 旧 Job 不会自动吃新 commit。
2. **先看训练健康，再看解题率：**
   - `vanilla` 是否整组 `std0`？
   - `mis_rs_catastrophic_seq_fraction` 是否接近 0？
   - Stage-2 transcript 是否非空？（现网审计仍可能空）
3. **顶层 `outcome/*` = Stage-1**；Stage-2 看 `outcome/stage-2/*`（且知可能被 J 坑）。
4. **resolved 上涨 ≠ 算法对。** A/B/C/G/H/I 都会让曲线「有数字、无意义」。
5. **公平对比：**同一训练集、同一 RBS/并发/超时、同一评测协议；Hybrid 修前 run 只能当「有缺陷实现」或消融，不当最终裁决。

---

## 5. 五分钟自检清单

```text
□ log 有 reward=0/1，不是全 0 + missing_eval_plan
□ 不是每步 vanilla_groups 全 std0
□ mis_rs_catastrophic_seq_fraction ≈ 0，mis_kl 与 GRPO 同量级
□ 新 Job：metadata 有 agent_queue_wait_sec、agent_exit_code
□ bundle 里 transcript.jsonl 非空（否则 Stage-2 语义仍错）
□ 分叉用的是 edit 前状态（i-1），不是 edit 后（i）
□ 训练损失是否按「独立 episode」计权，而不是整题一个 rollout_id 搅在一起
□ PROMPT_DATA 与对比的 GRPO 一致
□ 看的是 feature worktree / 重提后的 Job，不是旧进程
```

---

## 6. 代码地图（想改代码时从这跳）

| 区域 | 路径 |
|------|------|
| Hybrid 入口 | `examples/claudecode_ags/step_reconstruct/hybrid_generate.py` |
| Stage-1/2 runner | `…/live_runners.py` |
| Advantage / filter | `…/step_grpo_advantage.py` |
| 选型 / edit-PPL | `…/edit_ppl.py`、`selection.py` |
| Rebuild / resume | `…/workspace_rebuild.py`、`session_capture.py` |
| 朴素 GRPO | `examples/claudecode_ags/generate.py` |
| 指标 | `examples/claudecode_ags/wandb_metrics.py` |
| 2-node 提交 | `launch/hybrid_2node_job/`、`launch/grpo_2node_job/` |

---

## 7. 延伸阅读

| 文档 | 内容 |
|------|------|
| `2026-07-13-hybrid-step-grpo-debug-fixes.md` | 早期坑索引（对齐、评测、wandb 口径） |
| `2026-07-14-hybrid-vanilla-episode-key-bug.md` | 坑 B 专文 + 消融解读 |
| `2026-07-14-hybrid-branch-loss-mask-tis-rs-bug.md` | 坑 C 专文 |
| `2026-07-14-step-grpo-experiment-handbook.md` | 实验手册 / 当前在等什么 |
| `2026-07-15-hybrid-run-bmr3zlex-bug-report.md` | 某次 2-node run 的完整审计（坑 G/H/I） |
| `2026-07-15-adapter-event-loop-blocking.md` | 8 vs 16 卡 agent 变慢：adapter 单 loop 上同步 tokenize/hash（未修） |
| `specs/2026-07-12-path-a-hybrid-step-grpo-design.md` | 设计预期（对照「实际跑成了啥」） |

---

## 8. 给后来者的一句话

> Hybrid 的坑，多半不是「模型笨」，而是 **身份搞混了**（谁算一次尝试）、**mask 搞混了**（什么能训）、**状态搞混了**（从哪一刻续写）、**权重搞混了**（Stage-1/2 各算几票）。  
> 先把这四件事钉死，再谈算法赢不赢。
