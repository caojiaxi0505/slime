# Turn-level Teacher SFT 设计

日期：2026-08-14
入口：`examples.claudecode_ags.step_reconstruct.hybrid_sft_generate.hybrid_sft_generate`

本文记录方法本身：为什么做、数据怎么来、训什么。接线与环境变量见文末；实现细节散落在 `examples/claudecode_ags/step_reconstruct/`。

## 1. 问题

SWE + Claude Code 的一条 episode 往往几十轮 Read / Grep / Edit / Bash，朴素 GRPO 只有终局是否 resolved，再折成整条轨迹共用的单一 advantage。失败样本上，所有可训练 token 都被同一个负 advantage 惩罚，信用分配过粗。

Hybrid Turn-Level GRPO 在失败轨迹的关键 edit 上分叉，用 continuation 的终局奖励做更细的 GRPO。分叉点仍是**学生自己**往后探索：点选错、续写仍然不会修时，信号还是稀疏。

Turn-level Teacher SFT 换监督来源：学生先自己走，暴露它在哪些状态下不会做；再把强模型放到**同一个状态**上，让它真正调工具、看到真实结果，把这几步当作学生该模仿的 suffix。这接近 DAgger——在学生自己的状态分布上用专家动作做监督，而不是在专家自己的轨迹上做 SFT。

## 2. 方法

一次 generate 分两段。

### Stage-1：学生采样，只为留下可 resume 的状态

内部采 `K` 条学生轨迹（复用 Hybrid 的 vanilla runner），落 SessionBundle：PromptCheckpoint、native session、workspace snapshot。这些行默认**不进训练**。`sft_only`（默认）下 Stage-1 的 GRPO 不开；需要同时训学生策略时再开 `hybrid`。

### 选点：失败轨迹上的工具 turn

只对 `is_solved=False` 的 trial 重标注。每个可 resume 的工具 turn 是一个候选（默认含 Read / Grep，不只 patch）。一个候选 = 一次沙箱 rebuild。

默认每个合格 turn 都标。候选池与截断是两层：

- `STEP_GRPO_TEACHER_TURN_SELECT`：池子（`all` / `patch` / `patch_ppl`）
- `STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL`：池里每条 trial 最多标多少；`0` 表示不截断

纯文本收尾 turn 没有 snapshot 边界，无法 token-exact resume，不在池里。

### Stage-2：教师在沙箱里跑 N 步（默认 2）

对每个选中的 turn：

1. rebuild 到该 turn**之前**的 workspace（学生已经执行过的工具结果在盘上）
2. 截断 native Claude Code session，接到 checkpoint（token-exact）
3. Claude Code 改打远程教师
4. 教师跑 2 步：第 1 步给出工具调用，沙箱真执行；第 2 步是教师看到**真实 tool result** 之后的反应
5. adapter 在第 3 个**成功**请求前返回停跑 429，Claude Code 退出。第 2 步的工具已经执行完，非零 exit 属于预期

只采 1 步、不跑沙箱，SFT 目标只是一个未经执行的首动作。2 步是成本和监督质量的默认折中：有动作，也有对动作结果的解读。`MAX_STEPS=0` 则教师走到底并评测，只保留 resolved 续写，成本回到一次完整 rollout。

### 编码：学生前缀 + 教师 suffix

```text
prefix = 学生 checkpoint prompt_ids          loss_mask = 0
         教师步之间的 tool result            loss_mask = 0
suffix = 教师 N 步的 assistant token         loss_mask = 1
```

用学生 tokenizer 重新编码，和训练侧 chat template 对齐。一条样本可训练 token 占比低（长前缀、短后缀）。同一 task 的多条 teacher 样本先等权平均，避免工具 turn 多的 task 主导梯度。

### 更新边界与 loss 归一化

一次更新固定处理 16 个原始 task，每个 task 内部采 8 条学生轨迹。某个 task 的 8 条轨迹一结束，就立即启动该 task 的 teacher relabel；其他 task 的 Stage-1 可同时继续。只有 16 个 task 都完成后才训练一次，不按“先凑够若干 teacher 样本”提前训练。

teacher 样本数可以随更新变化。若 16 个 task 中有 $A$ 个产生可训练目标，则先对每个 task 内的 relabel 求平均，再对这 $A$ 个 task 求平均。没有 teacher 目标的 task 只保留一个零 loss 调度占位，不贡献梯度；若 $A=0$，直接拒绝该次优化，避免 AdamW 在零监督下仍执行权重衰减。这样每个有效 task 等权，loss 尺度不会随 teacher 样本数漂移。

```text
Stage-1 学生 K 条
        │
        ├─ 全 solved → 无 Stage-2
        │
        └─ 失败 trial 的工具 turn
                 │
                 ▼
        每个 turn 一个沙箱：rebuild + native resume
                 │
                 ▼
        教师 2 步（工具真执行）
                 │
                 ▼
        sft_only → 只留 teacher_sft 做 SFT
        hybrid   → vanilla GRPO + teacher SFT
```

## 3. 和 Hybrid GRPO 的差别

| | Hybrid Turn-Level GRPO | Turn-level Teacher SFT |
|---|---|---|
| Stage-1 | 学生 K 条，进 GRPO | 同样采 K 条，默认不训，只留状态 |
| Stage-2 谁在跑 | 学生从分叉点继续 | 教师从同一状态跑 N 步 |
| 学习信号 | continuation 终局奖励 → advantage | 教师 assistant token 的 NLL |
| 默认 Stage-2 长度 | 跑到结束并评测 | 2 步，不评测 |
| 沙箱 | 每个分叉一条 | 每个被标的 turn 一条 |

本轮不同时开「学生 Stage-2 branch GRPO」和教师 SFT。

## 4. 停跑、重试、槽位

Claude Code 没有「只走 N 步」的开关，上限做在 teacher adapter 上，按**成功** turn 计数。

两件不能混的事：

- **停跑 429**：`rate_limit_error`，第 N+1 步之前返回，不打教师。用来结束这轮 relabel。
- **教师限流 429**：远程接口返回，adapter 按 `15,30,60` 秒重试，耗尽后转成 **502**。不占步数，SFT 只吃 JSONL 里真正写成功的 turn。

并发也是两层，都不是「Python 开 64 条线程」：

- `SLIME_CC_AGENT_CONCURRENCY=64`：rollout 事件循环里的 asyncio 槽，限制同时活着的远程沙箱
- `SLIME_REMOTE_OPENAI_MAX_INFLIGHT=64`：adapter 那条 daemon 线程里，同时打教师 HTTP 的上限

沙箱槽覆盖 create → resume → 教师 N 步 → 关闭。默认 2 步不评测，eval 槽用不上。单个 relabel 失败（rebuild / resume / 0 条教师 turn）只丢掉该 turn。

## 5. 默认配置

| 项 | 默认 | 含义 |
|---|---|---|
| `ROLLOUT_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | `16 / 16` | 16 个原始 task 完成后训练一次 |
| `STEP_GRPO_HYBRID_K` | 8 | 每个 task 的学生轨迹数 |
| `STEP_GRPO_TEACHER_SFT_MODE` | `sft_only` | 只训教师 suffix |
| `STEP_GRPO_TEACHER_MAX_STEPS` | 2 | 每个 relabel 教师走几步；`0` 走到底并评测 |
| `STEP_GRPO_TEACHER_TURN_SELECT` | `all` | 失败 trial 全部可 resume 工具 turn |
| `STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL` | 0 | 不截断 |
| `STEP_GRPO_TEACHER_SFT_RESOLVED_ONLY` | 1 | 仅 `MAX_STEPS=0` 时只留 resolved |
| `SLIME_CC_AGENT_CONCURRENCY` | 64 | 同时活着的沙箱 |
| `SLIME_REMOTE_OPENAI_MAX_INFLIGHT` | 64 | 同时打教师的 HTTP |
| `SLIME_REMOTE_OPENAI_THINKING_TYPE` | `enabled` | DeepSeek-v4-pro 请求体 `thinking.type` |
| `SLIME_REMOTE_OPENAI_REASONING_EFFORT` | `max` | 思考深度；`max` 适合复杂 Agent |

必需：`SLIME_REMOTE_OPENAI_BASE_URL` / `SLIME_REMOTE_OPENAI_API_KEY` / `SLIME_REMOTE_OPENAI_MODEL`。

接线示例：`examples/claudecode_ags/launch/print_hybrid_sft_train_flags.sh`。

- `sft_only`：`--loss-type sft_loss --disable-compute-advantages-and-returns` + `sft_only_filter`
- `hybrid`：`--loss-type custom_loss` + `hybrid_teacher_sft_loss`，vanilla 走 GRPO、teacher 走 SFT

## 6. 取舍

- 2 步不评测：保证工具真执行、结果真可见，不保证这几步能修好题。
- 成功轨迹不标，避免把学生已经会的行为再灌一遍。
- 每个 turn 单独 rebuild，暂不做同一沙箱连续标多个 turn。
- 教师和学生使用不同 tokenizer，不要求两侧 token id 相同；但教师可见的逻辑消息历史必须与学生 checkpoint 完全一致。Teacher adapter 维护这条权威历史，Claude Code resume 重放的旧 HTTP 历史只用于恢复 harness，不参与模型输入重建。
