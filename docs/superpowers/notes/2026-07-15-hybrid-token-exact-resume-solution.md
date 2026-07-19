# Hybrid step-GRPO：token-exact resume 修复方案

日期：2026-07-15

适用分支：`feature/cc-ags-swe`

状态：生产代码已实施；`v4`～`v6` 真实扇出发现的重试边界已补齐；`v7` 两节点运行中

详细调查记录：[`2026-07-15-hybrid-token-exact-resume.md`](./2026-07-15-hybrid-token-exact-resume.md)

> 结论：token-exact resume 需要同时恢复模型输入、Claude Code session 和 workspace。不能再用 transcript 重放来恢复模型上下文。

## 1. 当前存在的问题

### 1.1 当前 Stage-2 不是 token-exact resume

当前流程是：

```text
Stage-1 trajectory
→ 截成 stream-json prefix
→ 启动新的 Claude Code
→ Claude Code 重新组织历史
→ adapter 重新计算 prompt_ids
```

这个流程可以让任务继续执行，但不能保证 Stage-2 第一次模型调用与 Stage-1 目标位置完全相同。

原因是 `trajectory.jsonl` 是 Claude Code 的输出记录，不是内部 session 的无损导出格式。把它重新送入 `--input-format stream-json` 后，Claude Code 可能：

- 调整 reminder 和原始任务的位置；
- 合并或重排 assistant/tool result 消息；
- 丢失或替换 tool result；
- 添加新的继续执行提示；
- 根据新沙箱重新生成 system prompt。

因此，transcript 完整、workspace diff 一致、Claude Code 能继续执行，都不能证明 token 一致。

### 1.2 原生 resume 也不能直接当作模型状态恢复

Claude Code 的原生 session 能恢复工具相关状态，但原生 fork 发给 adapter 的历史仍可能被合并或重排。

实测中，正常连续运行保持：

```text
assistant Read
user Read result
assistant Edit
user Edit result
```

原生 fork 可能改成：

```text
assistant [Read, Edit]
user [Read result, Edit result]
```

两者内容接近，但经过 Qwen3.5 chat template 后得到的 token 不同。因此原生 session 只能用于恢复 Claude Code 的执行状态，不能作为模型历史的唯一依据。

### 1.3 新沙箱会改变 system prompt

Claude Code 会把运行环境写入 system prompt。实测中，Stage-1 和 Stage-2 AGS 实例使用了不同内核版本：

```text
Stage-1: Linux 5.4...
Stage-2: Linux 6.6...
```

即使任务、镜像、commit 和文件内容一致，重新生成的 system prompt 仍可能不同。Stage-2 不能使用新沙箱生成的 system prompt 重新计算起始 token。

### 1.4 只恢复文件内容不足以继续 Edit

Claude Code 会记录文件被 Read 时的状态。新沙箱即使恢复了相同内容，文件 `mtime` 不同也可能导致：

```text
File has been modified since read
```

实测结果：

- 只恢复内容：分叉后的 `Edit` 失败；
- 同时恢复 Stage-1 的精确 `mtime`：`Edit` 成功。

当前 snapshot 主要保存 Git diff，没有保存完整文件元数据，因此还不能可靠恢复 Claude Code 的工具状态。

### 1.5 当前没有真正保存 Claude Code session

`SessionBundle` 已有 `cc_session_id` 和 `cc_session_files` 字段，但当前捕获结果仍是空值。

另外，AGS 写入 `/home/agent/.claude/settings.json` 时可能由 root 创建父目录。Claude Code 以 `agent` 用户运行，父目录不可写时不会生成 session JSONL。

这个问题过去不会阻止 transcript 路径运行，但会阻止新的原生 session 恢复方案。

### 1.6 只修第一轮仍然不够

即使 Stage-2 第一轮直接使用正确的 `prompt_ids`，下一轮如果重新接受 Claude Code 发来的整段 history，模型上下文仍会再次漂移。

所以修复必须覆盖两部分：

1. 第一次模型调用直接恢复 Stage-1 checkpoint；
2. 后续模型历史由 adapter 自己维护。

### 1.7 对训练的影响

错误的历史 token 本身通常是 `loss_mask=0`，不会直接计算 loss。但 Stage-2 的新动作是在错误上下文下采样的：

```text
错误的起始 prompt
→ 不同的动作分布
→ 不同的 patch 和 reward
→ 新输出进入训练
```

这会改变 step-GRPO 的分叉语义：branch 不再从选中的 Stage-1 状态重新采样。

## 2. 修复目标

设 Stage-1 在目标动作生成前真正发送给模型的 token 数组为 `P_t`。

Stage-2 第一次模型调用必须满足：

```text
stage2_prompt_ids == P_t
```

验收必须比较完整整数数组和 SHA256，不能只比较 token 数量或可读文本。

Hybrid 只要求 branch 从相同起点开始。第一次生成新动作后，不同 branch 的后续内容可以不同。

## 3. 需要恢复的三类状态

| 状态 | 保存内容 | 作用 |
|---|---|---|
| 模型状态 | `prompt_ids`、规范消息、system、tools、tokenizer 指纹 | 保证模型输入一致 |
| Claude Code 状态 | 原生 session JSONL | 恢复 Read/Edit 等工具内部状态 |
| Workspace 状态 | 文件内容、Git 状态和文件元数据 | 保证工具看到相同环境 |

三类状态缺一不可。

## 4. 总体方案

```text
Stage-1 adapter 保存 PromptCheckpoint
Stage-1 保存 Claude Code 原生 session
Stage-1 保存 workspace snapshot 和文件元数据
                    ↓
选择目标 assistant/tool turn
                    ↓
Stage-2 恢复 pre-turn workspace
Stage-2 恢复截断后的 Claude Code session
Stage-2 adapter 直接加载 PromptCheckpoint
                    ↓
第一次模型调用严格使用 checkpoint.prompt_ids
                    ↓
adapter 维护后续 assistant/tool result 历史
```

不需要修改或定制 Claude Code 二进制。当前使用的 `cc-prefix-2.1.104` 与官方 `@anthropic-ai/claude-code@2.1.104` 内容一致。

## 5. Stage-1：保存可恢复状态

### 5.1 保存 PromptCheckpoint

`SegmentedAnthropicAdapter` 每次调用 SGLang 前已经生成了 `prompt_ids`。应在这个位置保存 checkpoint：

```python
@dataclass
class PromptCheckpoint:
    checkpoint_id: str
    prompt_ids: list[int]
    prompt_sha256: str
    chat_messages: list[dict]
    system: list[dict] | str | None
    tools_schema: list[dict] | None
    tools_sha256: str
    generation_config: dict
    tokenizer_fingerprint: str
    chain_kind: str
```

其中：

- `prompt_ids` 是第一次恢复时的唯一权威输入；
- `chat_messages` 用于追加 Stage-2 新动作和工具结果；
- system 和 tools 用于审计及一致性校验；
- tokenizer 指纹用于阻止模型、chat template 或 tokenizer 变化后的错误恢复。

原始 Anthropic request body 可以额外保存用于排查，但不能替代 `prompt_ids`。

### 5.2 建立 checkpoint 与工具动作的映射

模型生成完成后，adapter 已经知道本轮产生的全部 `tool_use_id`。需要保存：

```text
checkpoint_id
→ generated assistant turn
→ [tool_use_id_1, tool_use_id_2, ...]
```

选择某个 `Edit` 时，就能找到生成该 `Edit` 之前的 checkpoint。

如果一个 assistant turn 同时生成多个工具，这些工具共享同一个 checkpoint。分叉时必须重新采样整个 assistant turn，不能把同一轮拆成多个独立起点。

### 5.3 保存原生 Claude Code session

Stage-1 启动时显式指定 UUID：

```text
claude -p --session-id <uuid> ...
```

启动前必须确保：

```text
/home/agent/.claude
```

及其父目录归 `agent` 用户所有且可写。运行结束后保存该 session 对应的 JSONL 文件，并把路径和 session ID 写入 `SessionBundle`。

缺少 session 文件时必须输出 warning 并禁止该 trial 进入 Stage-2。

### 5.4 保存 workspace 元数据

初始状态 `-1` 和每个工具边界都需要保存：

- 当前累计 diff；
- 文件类型；
- 文件权限；
- symlink 目标；
- 纳秒级 `mtime`；
- 必要的 Git HEAD/index 信息。

元数据至少覆盖目标动作前已经 Read、Edit、Write 或由 Bash 访问过的文件。无法确认 repo 外部状态是否完整时，初版应跳过该分叉点。

## 6. 选择和截断分叉点

对目标 `tool_use_id`：

1. 找到生成它的 assistant turn；
2. 取得该 turn 生成前保存的 PromptCheckpoint；
3. workspace 恢复到该 assistant turn 之前最后一个完整工具边界；
4. 第一轮工具动作使用初始 snapshot `-1`；
5. 原生 session 删除目标 assistant turn及其后续内容。

截断后的 session 需要保留一个合法的结束记录，使 Claude Code 可以通过 `--resume` 打开。该记录只用于启动客户端，不进入模型上下文。

## 7. Stage-2：恢复并重新采样

### 7.1 恢复 workspace

执行顺序必须是：

1. 创建新沙箱并执行与 Stage-1 相同的 workspace 初始化；
2. 恢复目标累计 diff；
3. 恢复文件权限、symlink 和 `mtime`；
4. 校验内容与元数据；
5. 校验通过后再启动 Claude Code。

如果先恢复 `mtime` 再应用 diff，diff 操作会再次改变时间戳，因此顺序不能颠倒。

### 7.2 恢复 Claude Code session

把截断后的 session 写入新沙箱，再使用：

```text
claude -p \
  --resume /tmp/branch.session.jsonl \
  --fork-session \
  --input-format stream-json \
  ...
```

向 stdin 发送一个固定且唯一的握手消息，用于触发第一次 Messages API 请求。

实测不需要隐藏参数 `--resume-session-at`。但本地 session JSONL 格式可能随 Claude Code 版本变化，因此生产环境应固定 Claude Code 版本和包哈希，并保留版本升级 smoke test。

### 7.3 第一次模型调用

Stage-2 创建 adapter session 时传入 checkpoint：

```python
adapter.open_session(
    session_id,
    resume_checkpoint=checkpoint,
)
```

Claude Code 发来的第一次请求只用于建立连接和声明工具。Adapter 必须：

1. 记录 Claude Code 当前声明的工具集合；
2. 忽略新请求中的 system 和 messages；
3. 直接把 `checkpoint.prompt_ids` 发送给 SGLang；
4. 再次计算并校验 prompt hash；
5. hash 不一致时立即中止 branch。

第一次模型输入仍使用 checkpoint 保存的工具 schema。Claude Code 当前声明的
schema 只用于运行时能力审计：描述、顺序或可选字段变化会记录 warning；如果模型
实际生成了当前 Claude Code 不提供的工具，branch 才会中止。

握手消息、新内核版本和新 reminder 都不能进入模型输入。

### 7.4 后续模型调用

第一次生成后，adapter 保存自己刚返回给 Claude Code 的 assistant 内容和全部 `tool_use_id`：

```text
pending_tool_ids = [toolu_a, toolu_b, ...]
```

Claude Code 执行工具并发来下一次请求时，adapter 不使用请求中的整段历史，只执行以下步骤：

1. 找到所有与 `pending_tool_ids` 匹配的 tool result；
2. 检查每个 ID 恰好出现一次；
3. 按生成工具时的顺序排列结果；
4. 在 checkpoint 的规范历史后追加本轮 assistant 消息；
5. 再追加真实 tool result；
6. 使用同一个 tokenizer 和 chat template 渲染下一轮 `prompt_ids`。

这样可以避免 Claude Code 对旧历史的合并、重排和 reminder 注入影响模型输入。

Claude Code 有时只发送 tool result，不回放 assistant 的 `tool_use`；也可能回放但
规范化 name/input。该回放不是权威历史，因此允许缺失或内容变化，只记录计数。
Adapter 自己保存的 assistant 输出和新 `tool_use_id` 不会被替换。真实 tool result
仍必须按 ID 唯一、完整地出现。

如果上一轮没有工具调用，则按上一轮真实停止原因处理：

- `max_tokens`：从 adapter 保存的规范历史继续采样；
- `end_turn`：返回空的结束确认，让 Claude Code 正常退出，不再调用模型；
- 原因未知：中止 branch，不能猜测或静默重启。

### 7.5 并行工具和失败工具

一个 assistant turn 产生多个工具时，adapter 必须等待全部结果：

```text
pending = [toolu_a, toolu_b]
```

只有同时收到 `toolu_a` 和 `toolu_b` 的结果后才能继续。结果在请求中的到达顺序不作为最终顺序，最终顺序使用模型生成时的 ID 顺序。

失败结果按真实内容追加，例如：

```json
{
  "tool_use_id": "toolu_b",
  "is_error": true,
  "content": "Exit code 7"
}
```

不得生成假的 missing result。

### 7.6 无工具的结束响应

如果模型返回普通文本并结束，不再等待 tool result，直接结束该 branch。该响应仍作为 Stage-2 新输出参与训练。

## 8. 训练数据处理

恢复的 Stage-1 历史只作为条件：

```text
checkpoint 历史：loss_mask = 0
Stage-2 新输出：loss_mask = 1
```

不要把 Stage-1 旧 turn 复制到 Stage-2 的训练 turn 列表。只记录 Stage-2 新生成 token 的 logprob、loss mask 和最终 reward。

现有 Stage-1/Stage-2 分组、层级加权和 branch loss weight 逻辑保持不变。

## 9. Fail-closed 规则

以下情况禁止静默退化为 fresh agent 或 transcript replay：

| 条件 | 处理 |
|---|---|
| 找不到目标 checkpoint | 跳过该分叉点并 warning |
| 原生 Claude Code session 缺失 | 禁止该 trial 的 Stage-2 |
| prompt 数组或 hash 不一致 | 中止 branch |
| tokenizer/chat template 指纹不同 | 中止 branch |
| Claude Code 运行时 tools schema 变化 | warning 并计数；checkpoint schema 仍为模型输入依据 |
| 模型调用当前 Claude Code 不提供的工具 | 中止 branch |
| workspace 内容或元数据不一致 | 中止 branch |
| pending result 缺失、重复或出现未知 ID | 中止 branch |
| 目标位于尚未支持的 subagent/compaction 状态 | 跳过并计数 |
| 依赖未捕获的 workdir 外部状态 | 跳过并计数 |

正式训练要求：

```text
resume/prompt_exact_rate == 1.0
```

不是接近 1，而是所有实际进入训练的 branch 都必须通过。

## 10. 已实施的代码改动

### `slime/agent/adapters/anthropic_segmented.py`

- 新增 `PromptCheckpoint` 和序列化接口；
- Hybrid Stage-1 在每次 SGLang 调用前捕获 checkpoint；朴素 GRPO 默认不捕获，避免额外内存开销；
- 保存 checkpoint 与生成 tool IDs 的映射；
- 支持 `open_session(..., resume_checkpoint=...)`；
- 第一次调用直接使用保存的 `prompt_ids`；
- 后续只追加匹配的 pending tool result；
- 增加 prompt/tools/tokenizer 硬校验。

### `examples/claudecode_ags/step_reconstruct/session_capture.py`

- 修复 `/home/agent/.claude` 所有权；
- 保存真实 `cc_session_id` 和 session 文件；
- 扩展 `SessionBundle`，保存 checkpoint 索引和 workspace 元数据；
- transcript 继续保留用于审计，但退出 correctness path。

### `examples/claudecode_ags/step_reconstruct/native_session.py`

- 按 checkpoint 的完整 tool ID 组定位原生 assistant turn；
- 并行工具共享一个起点，截断时删除完整 assistant turn；
- 生成 Claude Code 2.1.104 可打开的合法终止记录；
- 使用公开的 `--resume <file> --fork-session`，不依赖隐藏参数。

### `examples/claudecode_ags/step_reconstruct/workspace_rebuild.py`

- 在应用 diff 后恢复文件元数据；
- 增加内容、Git 和元数据校验；
- 不再把 transcript prefix 当作模型历史。

### `examples/claudecode_ags/step_reconstruct/live_runners.py`

- Stage-1 在 `finish_session` 前导出 checkpoint；
- 将 edit tool ID 对齐到完整 assistant turn；
- Stage-2 加载 checkpoint 和截断 session；
- 记录 exact gate、workspace gate 和失败原因指标。

### `examples/claudecode_ags/agent_runtime.py`

- Stage-1 支持显式 Claude session ID；
- Stage-2 增加 native resume/fork 启动入口；
- 保持现有普通 Stage-1 和朴素 GRPO 路径不变。

### `examples/claudecode_ags/wandb_metrics.py`

- 记录 `resume/prompt_exact_rate`；
- 记录同一 edit group 的 checkpoint hash 一致率；
- 只有通过 exact gate 的 branch 才进入上述分母和训练样本。

### 当前限制

- 只支持主 Claude Code chain；目标 checkpoint 若位于 subagent 或 compaction/wipe，直接跳过；
- Stage-2 新动作若再次启动 `Task`/`Agent`，该 branch 直接失败，不静默降级；
- workspace 元数据覆盖显式 Read/Edit/Write/Notebook 路径、累计 Git 变更和 `PROBLEM_STATEMENT.md`；显式指向 workdir 外的路径会禁止该分叉；
- Bash 命令可以访问任意外部状态，无法只靠命令字符串完整证明其依赖。真实 AGS smoke 和短跑仍需检查这类 branch 的失败率；
- Stage-2 不接受 Claude Code 对旧历史的再次重排或压缩，adapter 始终维护 checkpoint 之后的权威历史。达到上下文上限时 branch 会失败，不会切回 transcript replay。

## 11. 已完成的隔离验证

所有测试使用官方 Claude Code 2.1.104 和训练所用 Qwen3.5-9B tokenizer。

| 测试 | 结果 |
|---|---|
| Stage-2 第一次调用直接加载 checkpoint | 23,617 tokens，完整数组一致 |
| 相同动作后的第二次调用 | 23,732 tokens，完整数组一致 |
| 再执行一个工具后的第三次调用 | 23,787 tokens，完整数组一致 |
| 一个 assistant turn 产生并行 Read+Bash | 23,652 tokens，完整数组一致 |
| 并行 Bash 返回 `is_error=true` | 23,654 tokens，完整数组一致 |
| 恢复内容但不恢复 `mtime` 后执行 Edit | 失败 |
| 同时恢复精确 `mtime` 后执行 Edit | 成功 |
| 不使用隐藏 `--resume-session-at` | resume、工具执行和 exact gate 均通过 |

这些数字只对应测试 prompt。生产验收看完整数组和 hash，不要求 token 数固定为表中的值。

## 12. 验收标准

### Gate A：单元测试（已通过）

- checkpoint 保存、加载后 `prompt_ids` 和 hash 不变；
- 修改 bootstrap system/messages 不影响第一次模型输入；
- pending tool result 缺失、重复和未知 ID 均 fail closed；
- 并行和失败工具按预期追加；
- Stage-1 历史不进入 Stage-2 loss。

截至 2026-07-16，resume、native session、workspace、Hybrid 编排、adapter
事件循环和 W&B 指标相关的 96 项回归全部通过；另用真实 Qwen3.5-9B tokenizer
验证了首轮和下一轮完整 token 数组一致。

### Gate B：真实 AGS smoke（待执行）

- Stage-1 和 Stage-2 使用不同 AGS 实例；
- 真实恢复 native session 和 workspace 元数据；
- 执行 Read → 分叉 Edit → 后续 Read；
- 比较每一轮实际送入 SGLang 的完整 token 数组；
- 验证不同 branch 可以产生不同动作并继续执行。

### Gate C：训练短跑（待执行）

- 同一 checkpoint 的 8 个 branch 首次 prompt hash 全部相同；
- `resume/prompt_exact_rate == 1.0`；
- 无 session 串线；
- Stage-2 输出 token、logprob、loss mask 和 reward 正常；
- 现有层级加权测试继续通过；
- 通过后再提交 16 卡正式 Hybrid 训练。

## 13. 实施进度

1. [已完成] 修复 `.claude` 所有权并保存原生 session；
2. [已完成] 在 Hybrid Stage-1 adapter 捕获 checkpoint；
3. [已完成] 保存并恢复 workspace 内容 hash、权限、symlink 和 `mtime`；
4. [已完成] 实现 Stage-2 第一次 prompt exact gate；
5. [已完成] 实现 adapter 后续权威历史；
6. [已完成] 覆盖多回合、并行、失败工具、缺失结果和无工具结束；
7. [已完成] 真实 AGS resume 协议 Gate；
8. [执行中] 两节点 v7 正式 Hybrid run；首个 rollout 尚未完成。

## 14. 最终方案

生产路径应从：

```text
transcript
→ Claude Code 重建历史
→ adapter 接受重建结果
```

改为：

```text
PromptCheckpoint 恢复模型输入
+ native session 恢复 Claude Code 工具状态
+ workspace snapshot 恢复 workspace 状态
→ adapter 维护 Stage-2 后续历史
```

这套方案已经完成生产代码接线，并在普通多回合、并行工具、失败工具和 Edit 状态恢复上通过隔离验证。真实 SGLang/AGS 流量也已触发并通过 resume 协议 Gate；完整 rollout、训练更新和 checkpoint 仍需等待 v7 继续运行后验收。

## 15. 真实两节点 smoke：残留 transcript 门槛

### 15.1 首轮结果

2026-07-15 提交了首轮两节点 smoke：

- 实验目录：`qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_adapterfix`；
- 配置：2×8 H200、RBS=16、Stage-1 K=8、并发 64、agent 45 分钟；
- 在停止前保存了 53 个真实 Stage-1 bundle。

53 个 bundle 的前置状态为：

| 状态 | 有效数 |
|---|---:|
| Prompt checkpoint | 53/53 |
| Claude Code 原生 session | 53/53 |
| Workspace 元数据 | 45/53 |
| 旧 transcript 校验 | 26/53 |

Prompt checkpoint 和原生 session 已稳定保存，但 `collect_patch_candidates` 与 branch runner 仍把 transcript 校验当成硬门槛。这与本文方案“transcript 只用于审计”不一致，会让合法 Stage-2 候选被大量删除，因此首轮 run 已停止，未作为正式训练继续使用。

### 15.2 根因

Claude Code 2.1.104 会把一个 Messages API assistant turn 写成多条 stream-json：

```text
assistant thinking        message_id=X
assistant tool_use A      message_id=X
assistant tool_use B      message_id=X
user tool_result B
user tool_result A
```

旧校验器逐行判断“assistant 后面必须紧跟匹配的 user result”，没有先按 `message_id` 合并逻辑 turn，因此把正常并行工具误判为 transcript 损坏。Subagent 的嵌套完成顺序也可能与全局 snapshot 完成顺序不同。

### 15.3 本轮修复

1. transcript 校验先合并同一 `message_id` 的连续 assistant 行，并合并连续并行 tool result；
2. transcript 继续保存并在无效时输出 warning，但不再进入 token-exact correctness gate；
3. Stage-2 的 pre-turn snapshot 直接由 checkpoint 的完整 `generated_tool_use_ids` 组确定；并行工具完成顺序可以不同，但同一组 snapshot 必须连续且完整；
4. checkpoint 同时保存工具 ID→名称；包含 `Task`/`Agent` 的目标 turn 按当前限制跳过，不能静默降级；
5. `.git`/`.harness` 是内部状态，metadata 路径扫描不再把它们误报为 workdir 外依赖；真实外部路径仍 fail closed。

对首轮 53 个 bundle 使用新 transcript 校验器离线复核：

```text
旧有效：26/53
新有效：41/53
修正误判：15 条
剩余 warning：8 条缺 hook snapshot，4 条 sidechain/完成顺序不一致
```

剩余 12 条不会被当成 transcript replay 输入，也不会触发 fresh-agent fallback；Stage-2 仍必须通过 checkpoint、原生 session 和目标 workspace 三项硬校验。

### 15.4 验证与重提

修复后共 64 个相关测试通过，覆盖：

- token-exact 首轮及后续轮；
- 原生 session 截断；
- 并行工具拆行与反向完成顺序；
- checkpoint 工具组到 pre-turn snapshot 的映射；
- transcript audit 无效时不降级、不误删权威 checkpoint；
- workspace 内容、权限、纳秒级 `mtime` 与内部 `.harness` 路径；
- Stage-1/Stage-2 分组加权和 adapter event-loop 并发。

新的两节点任务使用独立目录：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v2
```

W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/0w2dqn9k`

最终 Gate 仍以真实 Stage-2 样本为准：`resume/prompt_exact_rate` 必须等于 `1.0`，且进入训练的 branch 必须同时具有正常输出、logprob、loss mask 和 reward。

## 16. v2 真实 Stage-2：Claude Code tool-use 回显差异

### 16.1 现象

v2 首步进入了真实 Stage-2：

- 1 条 branch 完整跑完 Claude Code 和 SWE 评测；
- 首条完成记录为 `exit=0`、agent 运行 32.3 秒、排队 301.2 秒；
- 排队时间已单独统计，没有占用 agent 的 45 分钟预算；
- 之后 39 条 branch 被 `token_exact_resume_not_verified` 删除。

这 39 条的共同状态是：

```text
first_prompt_exact = true
handshake_validated = true
exact_request_count >= 1
error = tool_use echo changed name/input
```

因此，checkpoint 恢复和第一次模型调用已经 token-exact。失败发生在模型生成工具调用后的下一轮。

### 16.2 根因

Adapter 第一次生成工具调用后，已经保存了权威的 assistant 内容和新生成的 `tool_use_id`。Claude Code 执行工具，再发下一次 Messages API 请求时，会回放同一个 ID，但可能规范化工具的 name/input。

旧实现同时要求：

1. ID 完全相同；
2. 每个 ID 恰好有一个 result；
3. Claude Code 回放的 name/input 与 adapter 原始输出逐字段相同。

第 3 项不是构造后续模型输入所必需的条件。后续模型输入使用的是 adapter 自己保存的 assistant 内容，不使用 Claude Code 回放的 assistant payload。把回放差异作为硬错误，会删除本来可以正确继续的 branch。

### 16.3 修复

修复后继续严格检查：

- 新 `tool_use_id` 必须是 adapter 刚生成的 ID；
- 每个 ID 必须恰好出现一次真实 tool result；
- 未知、重复或缺失的 ID/result 仍立即失败；
- 后续 prompt 继续由 checkpoint、adapter 权威 assistant 内容和真实 tool result 生成。

Claude Code 回放的 name/input 如果不同：

- 不再删除 branch；
- 输出 warning；
- 写入 branch 元数据 `tool_use_echo_mismatch_count`；
- W&B 记录 `resume/tool_use_echo_mismatch_count` 和
  `resume/tool_use_echo_mismatch_branch_rate`。

这不是 transcript fallback，也没有用 Claude Code 回放内容替换模型上下文。v2 已停止，避免用大量缺失 Stage-2 branch 的数据继续训练。

### 16.4 回归验证

修复后 99 个相关测试通过，覆盖 token-exact 多轮续跑、并行/失败工具、非权威回显差异、native session、workspace 恢复、Hybrid 编排、分组加权、adapter 并发和 W&B 指标。

下一份独立 v3 run 需要同时满足：

```text
resume/prompt_exact_rate == 1.0
token_exact_resume_not_verified == 0
真实多轮 branch 可以完成
回显差异只产生 warning/计数，不再删除 branch
```

### 16.5 v3 真实运行结果

v3 使用独立目录：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v3
```

W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/iekrfhau`

2026-07-15 18:30 UTC 的早期运行快照：

| 指标 | 结果 |
|---|---:|
| 已启动 resume 校验 | 115 |
| 已完成 Stage-2 branch | 45 |
| `exit != 0` | 0 |
| 正 reward branch | 27/45 |
| token-exact 硬失败 | 0 |
| 运行后 dropped branch | 0 |
| Claude Code 回显 warning | 243 |
| agent 实际运行时间 | 40.0–365.8 秒，均值 233.4 秒 |
| 排队时间 | 0–225.7 秒，均值 48.0 秒 |

真实多轮 branch 已多次出现 `Edit` 回显 input hash 不同。v2 会在第一次差异处终止；v3 保留 warning 和计数，继续执行后续模型请求，并正常完成 Claude Code、diff 收集和 SWE 评测。进入完成日志前，每条 branch 都必须通过：

```text
mode == token_exact
handshake_validated == true
first_prompt_exact == true
exact_request_count > 0
error == ""
```

另有 9 个候选点在 branch 启动前 fail closed：4 个 compact/wipe checkpoint、3 个 pre-turn workspace 外部依赖、2 个并行工具 snapshot 不完整。这些候选没有进入 Stage-2，不计作 resume 失败。

这只是早期快照，不能代表完整 run。后续完整审计见下一节。

## 17. v3 完整审计与三类后续轮修复

### 17.1 完整结果

v3 的 `rollout_0`～`rollout_2` 共计划运行 1,664 条 Stage-2 branch：

| 结果 | 数量 |
|---|---:|
| 成功保存并进入后续处理 | 670 |
| 被删除 | 994（59.7%） |
| 缺少 assistant `tool_use` 回放 | 827 |
| 无 pending tool 时又收到请求 | 130 |
| Claude Code 运行时 tools schema 变化 | 29 |
| workspace rebuild 失败 | 8 |

所有已保存和协议层失败的 branch 首次 prompt 都满足 `first_prompt_exact=true`。
问题不在第一次 token-exact 起点，而在第一次生成之后的协议兼容处理。旧 W&B
`resume/prompt_exact_rate` 只统计幸存 branch，因此单独看它无法发现 59.7% 的删除。

### 17.2 修复一：assistant tool-use 回放缺失

Adapter 已经保存了自己返回的 assistant 工具调用。Claude Code 下一轮只发送真实
tool result 时，不再要求它重复发送同一个 assistant `tool_use`。仍严格要求：

- result ID 必须是刚生成的 ID；
- 每个 pending ID 恰好有一个 result；
- 未知、缺失或重复 result 立即失败。

缺少回放或回放 payload 变化只产生 warning 和独立计数，不会修改 adapter 保存的
assistant 历史。

### 17.3 修复二：无 pending tool 的后续请求

旧代码只知道“当前没有 pending tool”，不知道上一轮为何停止，因此一律报错。现在
保存每一轮的 `stop_reason`：

- `max_tokens` 表示输出被长度上限截断，从 adapter 权威历史继续生成；
- `end_turn` 表示模型已经结束，只向 Claude Code 返回结束确认，不重复采样；
- 未知状态继续 fail closed。

这样既不会丢掉被截断的有效续写，也不会在已经结束后额外生成一轮训练 token。

### 17.4 修复三：运行时 tools schema 漂移

Checkpoint 中的 tools schema 是 Stage-1 模型输入的一部分，Stage-2 仍用它渲染
模型 prompt。Claude Code 运行时 schema 的描述、顺序或可选字段变化只做审计，
不再删除 branch。为防止模型生成 Claude Code 无法执行的动作，每次生成后仍检查
工具名称；若该名称不在当前运行时工具集合中，立即失败。

### 17.5 可观测性与 v4

失败 branch 的 transcript 现在会在 resume 校验前落盘。W&B 新增：

- 计划、完成、删除 branch 数和完成率；
- 三类 resume 删除原因及 timeout/workspace/other 分类；
- tool-use 回放缺失、payload 变化、schema 漂移、`max_tokens` 续写和
  `end_turn` 确认的事件计数。

96 项相关回归全部通过。2026-07-16 02:24 UTC 已用独立目录重新提交两节点 v4：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v4
```

W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/9e869iz0`

v4 不读取 v3 checkpoint，也不会把 v3 已训练/已丢弃的数据当作续训输入。v4 的
真实验证结果和后续修复见下一节。

## 18. v4 真实扇出：HTTP 请求重试必须幂等

### 18.1 现象

v4 的首批真实 Stage-2 中，36 条 branch 完整通过 resume gate 并完成，36/36
`exit=0`；agent 用时 32.7～553.7 秒。多轮 Edit/Bash/Write 的 payload 规范化只产生
warning，没有误删。

更大扇出后出现两种错误：

```text
expected one result for tool_use ..., got 0
received resumed request without pending tool calls after stop_reason=tool_use
```

前一种至少出现了 10 个不同的新 tool ID。失败 transcript 显示，此前多轮
assistant/tool result 都完整，最后在一个正常 tool result 之后收到 409。它们不能
统一解释为“Claude Code 没有执行工具”。

### 18.2 根因

Messages HTTP 响应可能在客户端收到前断开或超时，Claude Code 会重发同一个请求。
旧 adapter 在响应真正送达前已经修改了 session：

- 生成新工具后，先记录了新的 pending tool ID；同一旧请求重试时就被误判为
  “缺少这个新 ID 的 result”；
- 消费 tool result 后，先清空了 pending；生成或传输中断后，同一 result 请求重试
  时就被误判为“无 pending”。

因此，这不是新的模型语义，而是请求重试与 session 提交之间缺少幂等和事务边界。
这也是 §17.3 “无 pending”类别中仅按 `stop_reason` 分流仍未覆盖的子情况。

### 18.3 修复

每个 resume session 现在保存最近一次成功请求的规范 SHA256 和完整 Anthropic 响应：

1. 同一请求再次到达时，不调用 SGLang，不再次修改历史；
2. 直接返回相同 content、tool ID、usage 和 message ID；
3. 非流式和 SSE 流式响应使用同一规则；
4. 记录 `resume/request_replay_count` 和 branch 命中率。

在消费 tool result、修改规范历史之前还会保存轻量事务快照。如果 SGLang 请求、
decode 或 parse 中断，则恢复 pending IDs、completed IDs、chat history、stop reason 和
计数；下一次请求可以从原状态重新执行。Checkpoint 的大 token 数组不会被重复拷贝。

真正缺失/重复/未知的 result 仍 fail closed；Task/Agent 子代理仍按当前限制拒绝。
这两类被单独计为 `resume_missing_result` 和 `resume_subagent`，不再混入笼统的
`resume_other`。

### 18.4 验证与 v5

新增测试覆盖：

- 握手请求重复；
- tool-result 请求重复；
- 流式请求重复且 message ID 不变；
- 上游生成失败后完整回滚，再次请求可继续；
- replay 不增加模型调用数和训练 turn 数。

截至 2026-07-16，相关回归共 116 项：115 通过，1 项按环境跳过。v4 在完成
rollout 0、进入训练前停止，目录只保留作故障证据。随后用全新目录提交 v5：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v5
```

W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/hab0y0fv`

v5 的最终 Gate 除首次 prompt exact 和三类协议计数外，还要求：重复请求只增加
`request_replay_count`，不能再产生 `missing_result` 或 `no_pending` branch drop。

## 19. v5 真实扇出：整包 SHA 仍然过严

### 19.1 结果与新现象

v5 在 rollout 0 完成前停止，没有生成 rollout dump 或 checkpoint，也没有进入训练。
停止前的快照如下：

| 指标 | 数量 |
|---|---:|
| 成功完成 Stage-2 branch | 82 |
| tool payload 回显差异 | 328 |
| `end_turn` 结束确认 | 2 |
| HTTP 503 | 53 |
| 不存在的 `Delete` 工具被拒绝 | 11 |
| `missing_result` 日志 | 47 |
| `missing_result` 的不同 pending ID | 6 |
| `no_pending` | 0 |

前 82 条成功 branch 说明首次 token-exact、payload 回显处理、`end_turn` 分流和事务回滚
都在真实流量中工作。`Delete` 是模型生成了运行时不存在的工具，按设计 fail closed。

但仍有 6 个不同 pending ID 被报为缺少 result。失败 branch 的落盘 transcript 中找不到
这些 ID，说明对应响应没有进入 Claude Code trajectory；这不是 Claude Code 收到工具后
拒绝回传 result，而是响应未送达后发生了请求重试。

### 19.2 根因

v4 修复用完整 HTTP JSON 的 SHA 判断“是否为同一请求”。Claude Code 重试时可能改变
不影响模型输入的 wire 字段，例如 `model`、`stream`、`metadata`、cache-control 标记或
工具列表顺序。此时逻辑请求相同，但整包 SHA 不同，缓存未命中；adapter 已持有新
pending ID，于是把旧请求误判为缺少 result。

另外，真实日志出现了“工具名称存在、但参数不符合运行时 schema”的生成结果。只检查
工具名不足以保证 Claude Code 能执行该调用。

### 19.3 补充修复

请求缓存现在使用“逻辑请求 SHA”，只包含会标识模型请求的字段：

- system、messages、tools 和 tool choice；
- thinking、output/context 配置；
- max tokens、stop、temperature、top-p、top-k。

计算前去掉 cache-control，工具按名称规范排序。`model`、`stream`、`metadata` 等 wire
字段不参与逻辑 SHA；完整 wire SHA 只用于日志审计。逻辑 SHA 相同就返回完全相同的
缓存响应，不再次采样或修改 session。

生成后的每个工具调用还会在 CPU worker 中按当前 Claude Code runtime JSON Schema
校验名称和 input。未知工具、缺少必填参数、额外参数或类型错误都会在写入 pending
之前 fail closed，不再等一个不会到达的 tool result。

新增测试覆盖 wire-only 字段变化、工具顺序变化仍命中同一缓存，以及畸形工具 input
在 pending 前被拒绝。相关 targeted suite 共 117 项：116 通过，1 项按环境跳过。

### 19.4 v6

2026-07-16 03:59 UTC 已用全新目录提交两节点 v6：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v6
```

v6 继续使用 K=8、RBS=16、16 GPU、agent 45 分钟和真实并发 64。v5 目录只保留
故障证据，不作为 v6 的 checkpoint 或数据输入。

## 20. v6 真实扇出：幂等判断必须以 pending 状态为准

### 20.1 结果

v6 W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/ltomfnvr`

v6 在 rollout 0 完成前停止，没有 rollout dump、checkpoint 或训练更新。停止前：

| 指标 | 数量 |
|---|---:|
| 成功完成 Stage-2 branch | 36 |
| payload 回显差异 | 197 |
| `end_turn` 结束确认 | 1 |
| runtime input schema 拒绝的不同 tool ID | 10 |
| `missing_result` 的不同 pending ID | 10 |
| 缓存 replay | 0 |

runtime input schema 校验在真实流量中正确拦截了多种不可执行调用，例如 Edit 多出
`lowerbound_line`、Grep 缺少 `pattern`、Read 多出 `block/timeout`。这些调用在写入
pending 前失败，没有再伪装成正常 tool result。

但逻辑 SHA 仍未命中真实重试。说明 Claude Code 重试时连 system/messages/tools 中也可能
发生重写；继续扩大“忽略字段列表”不能形成可靠协议。

### 20.2 最终状态规则

是否重试不再只由请求内容决定，而由 adapter 的 pending 状态决定：

1. adapter 已返回一个或多个 pending tool call；
2. 新请求对这些 pending ID 一个 result 都没有；
3. 请求也没有未知 result ID；
4. 则该请求不能推进 session，只能返回上一次完整缓存响应。

这次重放保持相同 content、tool ID、usage 和 message ID，不调用 SGLang，也不修改历史。
请求 SHA 只保留作审计。若只返回部分并行 result、重复 result 或未知 result，仍立即
fail closed，避免重复执行已有副作用的工具。

新增测试使用完全不同的 system/messages 模拟被重写的重试，验证 pending 状态会返回
相同缓存；随后真实 result 仍可正常推进。同时保留“并行工具只回一部分必须失败”的
测试。相关 targeted suite 共 119 项：118 通过，1 项按环境跳过。

### 20.3 v7

2026-07-16 04:44 UTC 已用全新目录提交两节点 v7：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v7
```

v7 不读取 v6 checkpoint 或 rollout 数据。最终 Gate 要求真实日志出现 pending-state replay
时不增加模型调用，并且不再由零 pending-result 请求产生 `missing_result/no_pending`。

W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/6qfoon5g`

2026-07-16 05:39 UTC 的早期快照：

| 指标 | 数量 |
|---|---:|
| 已完成 Stage-1 episode | 102 |
| 已完成 Stage-2 branch | 35 |
| pending-state replay 事件 | 117 |
| replay 涉及的不同 pending ID | 62 |
| replay 时逻辑请求 SHA 变化 | 117/117 |
| replay 时完整 wire SHA 变化 | 117/117 |
| `missing_result` | 0 |
| `no_pending` | 0 |
| payload 回显差异 | 304 |
| agent 超时（`exit=-1`） | 0 |

这 117 次不是重新采样：adapter 返回缓存中的完整 content、tool ID、usage 和 message ID，
也不推进 session。其后 Stage-2 branch 仍持续完成，说明真实 tool result 可以接在缓存响应
之后正常继续。62 个不同 pending ID 表明该结果不是单个请求的偶然现象。

日志中另有模型生成的不可执行工具调用被明确拒绝，包括 4 个不同 tool ID 的参数 schema
错误、不存在的 `Install`，以及当前不支持的 `Task/Agent` 子代理。这些属于生成结果不符合
Claude Code 运行时契约，不是 resume 丢失 pending 状态；拒绝发生在把新调用写成 pending
之前，因此不会再转化为假的 `missing_result`。

截至该快照，两个 Hybrid pod 都是 `Running`、0 restart；最长 agent 实际执行约 292 秒，
没有顶满 45 分钟。最长排队约 1492 秒，仍与 agent 时间分开记录。首个 rollout 尚未生成
dump、训练更新或 checkpoint，因此当前结论仅是“三类 resume 与请求重试协议已通过真实流量
早期验收”，不能写成完整训练已通过。
