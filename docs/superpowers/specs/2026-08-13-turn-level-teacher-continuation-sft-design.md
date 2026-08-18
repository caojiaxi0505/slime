# Turn-level teacher SFT（实现规格）

日期：2026-08-13
设计说明：[`../notes/2026-08-14-turn-level-teacher-sft-design.md`](../notes/2026-08-14-turn-level-teacher-sft-design.md)
入口：`examples.claudecode_ags.step_reconstruct.hybrid_sft_generate.hybrid_sft_generate`

本文只列模块、开关默认值和接线。方法动机与数据流见 notes。

## 模块

```text
hybrid_sft_generate.py      # 入口；sft_only 丢掉 vanilla
hybrid_generate.py          # Stage-1；stage2_fn 替换选型+branch
teacher_turn_sft.py         # 选 turn，fan-out teacher_branch_runner
teacher_branch_runner.py    # rebuild + native resume + 教师 N 步
teacher_segments.py         # checkpoint 前缀 + 教师 suffix
hybrid_teacher_sft_loss.py  # hybrid：vanilla GRPO + teacher SFT
sft_remote_openai_adapter.py
```

`hybrid_generate(stage2_fn=teacher_turn_samples)` 跳过 edit-PPL 分叉。单个 relabel 失败只丢该 turn。

## 默认开关

| 变量 | 默认 |
|---|---|
| `STEP_GRPO_TEACHER_SFT_MODE` | `sft_only` |
| `STEP_GRPO_TEACHER_MAX_STEPS` | 2（`0` = 走到底并评测） |
| `STEP_GRPO_TEACHER_TURN_SELECT` | `all` |
| `STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL` | 0（不截断） |
| `SLIME_CC_AGENT_CONCURRENCY` | 64 |
| `SLIME_REMOTE_OPENAI_MAX_INFLIGHT` | 64 |
| `SLIME_REMOTE_OPENAI_RETRY_DELAYS_SEC` | `15,30,60` |

教师限流 429 转 502，不占 `MAX_STEPS`；停跑 429 按成功 turn 计数。

## 接线

`examples/claudecode_ags/launch/print_hybrid_sft_train_flags.sh`

- `sft_only`：`--loss-type sft_loss --disable-compute-advantages-and-returns` + `hybrid_sft_generate.sft_only_filter`
- `hybrid`：`--loss-type custom_loss` + `hybrid_teacher_sft_loss` + 现有 `post_process_rewards` / `filter`

`sft_only` 固定 `RBS=GBS=16`、`K=8`。每个 task 的 8 条学生轨迹完成后即可并行启动 teacher，但必须等 16 个 task 全部结束后才训练。有效 task 内先平均 relabel，再平均有效 task；无目标 task 使用零 loss 调度占位。
