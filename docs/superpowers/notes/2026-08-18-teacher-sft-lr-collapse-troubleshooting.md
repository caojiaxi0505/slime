# Turn-Level Teacher Imitation 早期 resolved 崩塌排障

日期：2026-08-18
范围：`sft_only` Turn-Level Teacher Imitation 的早期训练退化；重点比较学习率 `3e-6` 与 `1e-6`。上下文和 JSONL 完整性修复见 `2026-08-17-teacher-sft-context-integrity-fix.md`。

## 1. 问题现象

修复 Teacher 上下文、JSONL 隔离和 token 边界后，从 Base 重新启动的 `3e-6` 训练仍在很早阶段出现能力下降：

- step 0–2 的 Stage-1 resolved 分别为 `86/128`、`97/128`、`87/128`；
- step 3 突降到 `43/128`；
- 同期 imitation loss 从 `0.668` 持续下降到 `0.397`。

这不是“训练没有收敛”。相反，模型正在更快拟合 Teacher continuation，但整题 Agent 能力同时下降。需要解释的是：**为什么局部模仿目标变好，Stage-1 的整题 resolved 反而变差。**

两次对照运行如下：

| 项目 | 旧运行 | 新运行 |
|---|---|---|
| 学习率 | `3e-6` | `1e-6` |
| W&B run | [`pkmg3ewr`](https://wandb.ai/models-tencent7723/imitation-sft/runs/pkmg3ewr) | [`49lo2e9m`](https://wandb.ai/models-tencent7723/imitation-sft/runs/49lo2e9m) |
| W&B group | `fail_imitation_learning_1uegt8q6` | `fail_imitation_learning_lr1e6` |
| Job | `jiaxicao-fail-imitation-learning` | `jiaxicao-fail-imitation-learning-lr1e6` |
| 起点 | Qwen3.5-9B Base | Qwen3.5-9B Base |
| 其他训练设置 | 相同 | 相同 |

新运行使用独立输出目录，没有读取旧训练 checkpoint。两次运行的采样具有随机性，但数据顺序、每步 16 道题、每题 8 条 Stage-1 trial 和 Teacher SFT 逻辑保持一致。

## 2. 当前结论

当前证据最支持的根因是：**`3e-6` 对当前高覆盖 Teacher SFT 目标过大，连续几次更新后产生了明显的策略漂移。模型快速降低局部 Teacher CE，却破坏了 Base 已有的长程 Agent 行为。**

这里的“策略漂移”指模型输出分布在少量更新后变化过大，不是 loss、梯度或参数出现 NaN。该判断有三项直接依据：

1. `3e-6` 的 loss 下降更快，但 resolved 明显更差；
2. step 3 两次运行使用相同的 16 道题，差异不能归因于题目突然变难；
3. 学习率降到 `1e-6` 后，step 3 resolved 从旧运行的 `43/128` 恢复到 `107/128`。

截至当前快照，`1e-6` 已避免旧运行的早期崩塌，但只验证到前 5 轮 rollout，不能据此断言长程训练已经完全稳定。

## 3. 对照数据

### 3.1 Stage-1 resolved

| step | `3e-6` | `1e-6` | 差值（`1e-6 - 3e-6`） |
|---:|---:|---:|---:|
| 0 | 86/128 = 67.19% | 85/128 = 66.41% | -0.78 pp |
| 1 | 97/128 = 75.78% | 100/128 = 78.13% | +2.34 pp |
| 2 | 87/128 = 67.97% | 92/128 = 71.88% | +3.91 pp |
| 3 | 43/128 = 33.59% | 107/128 = 83.59% | +50.00 pp |
| 4 | 74/128 = 57.81% | 100/128 = 78.13% | +20.31 pp |

step 0–2 的差异仍可能包含采样噪声；step 3 的 50 个百分点差异已经远大于正常波动。对 rollout dump 中的 task id 做审计后，step 2 和 step 3 两侧均为相同的 16 道题，集合摘要也完全一致。因此，step 3 的断崖不能用题目组成解释。

### 3.2 Loss 与梯度

| step | `3e-6` loss | `1e-6` loss | `3e-6` grad norm | `1e-6` grad norm |
|---:|---:|---:|---:|---:|
| 0 | 0.668 | 0.679 | 7.86 | 7.66 |
| 1 | 0.605 | 0.648 | 5.67 | 8.00 |
| 2 | 0.496 | 0.569 | 3.14 | 9.79 |
| 3 | 0.397 | 0.595 | 3.68 | 5.79 |

关键观察：旧运行的 loss 更低，却不是更好的 Agent。Teacher CE 只衡量模型对局部 Teacher continuation 的拟合程度，不等价于整题 resolved。若更新过强，loss 可以快速改善，同时损害模型原有的探索、工具使用和长程收尾能力。

两侧 `clip_grad=1.0`，表中记录的是裁剪前 grad norm。所有更新都触发裁剪，因此没有梯度数值爆炸；但 Adam 的最终更新仍由学习率缩放。其他条件相同时，`3e-6` 的更新尺度约为 `1e-6` 的 3 倍。梯度裁剪限制异常大的单步梯度，不能替代合适的学习率。

### 3.3 不是 Teacher 样本数直接放大了 loss

| step | `3e-6` Teacher samples | `1e-6` Teacher samples | 两侧 loss weight sum |
|---:|---:|---:|---:|
| 0 | 1,249 | 1,314 | 16 |
| 1 | 1,264 | 1,036 | 16 |
| 2 | 1,176 | 1,601 | 16 |
| 3 | 517 | 666 | 16 |

每步 Teacher 样本数随失败轨迹长度变化，但三层归一化会把每步有效题目的总 loss weight 固定到 16。旧运行不是因为某一步生成了更多 continuation，导致 loss 被简单求和放大。

大量 continuation 仍会改变梯度内容：当前方法覆盖失败 trial 的全部可恢复工具 turn，又不同时训练成功 Stage-1 轨迹。因此它对“如何模仿 Teacher 的局部动作”提供很强的监督，却没有直接约束模型保留原有整题策略。学习率过大时，这种目标不一致更容易表现为遗忘或输出分布漂移。

## 4. Chat template 假设

### 4.1 实际数据链路

Teacher 和学生不应使用同一个 tokenizer，但它们处理的是同一条逻辑消息历史：

| 阶段 | 实际处理 |
|---|---|
| Teacher 推理 | Adapter 把权威 checkpoint 转成 OpenAI `messages/tools`，发送给 DeepSeek 服务；DeepSeek 服务内部使用自己的模板 |
| Teacher 落盘 | 保存结构化 `assistant content + tool_calls`，不保存 Qwen token |
| 学生训练 | 重建 checkpoint、Teacher assistant 和真实 tool result，再使用 Qwen3.5 chat template 与 tokenizer 编码 |
| Loss | checkpoint 和 tool result mask 为 0；只训练 Teacher assistant token |

因此，Teacher 侧使用 DeepSeek 的输入协议，学生侧必须使用 Qwen3.5 的输出协议。不能直接把 DeepSeek token id 当成 Qwen3.5 的训练标签。

### 4.2 为什么会看到 `<|im_end|>` 和 `<function=Read>`

这两类字符串来自 Qwen3.5 对结构化消息的序列化：

- `<|im_end|>` 是 Qwen 消息结束标记；
- `<function=Read> ...` 是 Qwen 工具调用的文本表示。

它们出现在最终解码后的学生训练序列中是正常现象，不表示 DeepSeek 收到了 Qwen chat template。

对两次运行的原始 Teacher JSONL 做审计：

| 运行 | Teacher response rows | 原始 content 含 `<|im_end|>` | 原始 content 含 `<function=` |
|---|---:|---:|---:|
| `3e-6` | 10,955 | 0 | 0 |
| `1e-6` 快照 | 10,829 | 0 | 0 |

原始记录中的工具调用以结构化 `tool_calls` 保存。由此可以排除“DeepSeek 原始输出混入 Qwen 特殊 token，学生照抄异常文本”作为本次早期崩塌的原因。

## 5. 其他排除项

| 假设 | 结论 | 依据 |
|---|---|---|
| Teacher / 学生 chat template 混用 | 已排除 | 原始 Teacher content 中两类 Qwen 标记均为 0；Qwen 标记只在学生侧序列化后出现 |
| Teacher 样本数越多，loss 被直接放大 | 已排除 | 每步 `teacher_sft_loss_weight_sum=16` |
| 梯度数值爆炸 | 已排除 | loss、grad norm 均有限；`clip_grad=1.0`；无 NaN / OOM |
| step 3 题目更难 | 已排除 | 两次 step 3 的 16 个 task id 完全相同 |
| 旧 JSONL 静默混入 | 当前链路已防复发 | 每次 attempt 独立文件，严格校验 attempt 和 turn index |
| 上下文不一致仍强制拼接 | 当前链路已防复发 | correctness gate 会明确丢弃，不再 fallback |
| DeepSeek / AGS 短暂断连 | 不是本次主因 | 日志中会重试或明确丢弃；未出现与 step 3 同步的 Job 级失败 |

严格 gate 仍可能拒绝个别缺失 tool result 的 continuation。这会减少可训练样本数，但不会把不一致上下文送入训练，和旧版静默拼接不是同一问题。

## 6. 根因链条

当前最符合数据的过程如下：

1. 学生在失败轨迹的许多工具 turn 上产生 Teacher continuation；
2. `sft_only` 只优化 Teacher assistant CE，不训练学生成功轨迹，也没有整题 reward 约束；
3. `3e-6` 在连续裁剪梯度下仍产生过大的参数更新；
4. imitation loss 快速下降，说明模型迅速向局部 Teacher 分布移动；
5. 原有长程 Agent 行为被扰动，step 3 Stage-1 resolved 降到 `43/128`；
6. resolved 下降后会产生更多失败状态和 Teacher 数据，可能继续加强这种偏移。

其中第 1–5 项有当前 A/B 数据支持；第 6 项是由方法的数据生成机制推导出的潜在反馈，不作为已经单独验证的结论。

## 7. 修复

学习率已经改为可配置项：

- `run_hybrid_1node_debug.sh` 使用 `--lr "${LEARNING_RATE:-3e-6}"`；
- Teacher SFT Job 模板显式传递 `LEARNING_RATE`；
- 不设置时仍保持旧默认值 `3e-6`，避免改变其他实验；
- 本次新任务显式设置 `LEARNING_RATE=1e-6`。

旧 Job 已停止，新任务从 Base 干净启动：

```text
Job:      jiaxicao-fail-imitation-learning-lr1e6
W&B:      fail_imitation_learning_lr1e6_ewteueui-RANK_0
Run ID:   49lo2e9m
Output:   /mnt/sn-007/jiaxicao/checkpoints/cc-ags/fail_imitation_learning_lr1e6
```

## 8. 当前验证状态

截至 2026-08-18 03:20 UTC：

- step 0–4 rollout 已完成；
- step 3 resolved 为 `107/128`，没有复现旧运行的 `43/128`；
- step 4 resolved 为 `100/128`；
- step 0–3 loss 保持在 `0.569–0.679`，没有 NaN；
- Pod 无重启，未出现 CUDA OOM、NCCL 错误或致命 traceback；
- 存在少量远端 `Broken pipe`、`Server disconnected` 和 AGS 网络重试，当前均由重试或严格丢弃处理。

当前结论是：**降低到 `1e-6` 已修复前 4 次更新后的早期 resolved 崩塌。** 长程是否稳定，仍需观察后续 step；只有 continued Stage-1 resolved 和独立 SWE 评测都正常，才能认定方法整体有效。

## 9. 复发时的检查顺序

1. 确认比较的是相同 `rollout/step`，并核对两侧 task id；
2. 先看 resolved 的断点，再检查它之前已经执行过哪些参数更新；
3. 同时比较 train loss、裁剪前 grad norm、`clip_grad` 和实际 LR；
4. 检查 `teacher_sft_loss_weight_sum`，区分样本数变化与 loss 尺度变化；
5. 审计 Teacher 原始 `response.message`，不要用 Qwen 解码后的训练文本判断 Teacher 模板；
6. 检查 `n_dropped_context_integrity`、缺失 tool result 和 JSONL attempt 连续性；
7. 将可恢复网络重试、样本级严格丢弃和 Job 级失败分开统计。
