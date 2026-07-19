# Hybrid step-GRPO：token-exact resume 问题说明与推荐方案

日期：2026-07-15

适用分支：`feature/cc-ags-swe`

状态：历史调查记录；修复代码已经实施，当前状态见 [`2026-07-15-hybrid-token-exact-resume-solution.md`](./2026-07-15-hybrid-token-exact-resume-solution.md)

关联实现：`examples/claudecode_ags/step_reconstruct/`、`slime/agent/adapters/anthropic_segmented.py`

> **结论：当前 `trajectory → resume_prefix → claude --input-format stream-json` 路径能够让 Claude Code 继续执行任务，但不能保证 Stage-2 从与 Stage-1 完全相同的模型状态开始。**
>
> 推荐方案是不再让 Claude Code 根据 transcript 重建模型上下文，而是在 Stage-1 adapter 内保存真正送给 SGLang 的 `prompt_ids` 和规范化对话状态；Stage-2 由 adapter 直接恢复该状态。Claude Code 继续使用原版，只负责执行新生成的工具调用。

本文只说明问题和推荐设计，不修改代码、不重启训练任务。

## 1. 本文讨论的不是旧的“空 transcript”问题

旧 Hybrid run 曾经没有保存真实 trajectory，导致 `transcript.jsonl` 为空，Stage-2 退化成失忆的新 agent。该问题已经在当前 feature worktree 中修正：

- Stage-1 保存真实 `.harness/trajectory.jsonl`；
- transcript 缺失、为空、JSON 损坏或无法与 tool snapshot 对齐时会输出 warning，并禁止 Stage-2；
- Stage-2 workspace 与 transcript 都恢复到目标 edit 之前；
- 当前 bundle 可以保存完整且通过校验的 transcript。

本次发现的是更深一层的问题：

> 即使 transcript 完整、workspace 重建通过，Claude Code 重新读取 stream-json 后，实际发给模型的 prompt 仍可能改变。

因此必须区分：

| 状态 | 含义 | 当前情况 |
|---|---|---|
| transcript 完整 | 保存了 assistant、tool use 和 tool result | 已做到 |
| workspace 内容一致 | 目标时刻的代码 diff 一致 | 已做到 |
| 能继续完成任务 | Stage-2 会继续调用工具并产出 patch | 已做到 |
| token-exact resume | Stage-2 首次模型调用的完整 `prompt_ids` 与 Stage-1 目标调用逐项相同 | **未做到** |

前面三项不能推出第四项。

## 2. 什么叫 token-exact resume

设 Stage-1 在某个目标 edit 之前真正送给模型的 token 数组为：

```text
P_t = tokenizer(chat_template(system_t, messages_<=t, tools_t))
```

Stage-2 从该点分叉时，严格条件是：

```text
P_stage2_first == P_t
```

这里的“相同”是完整整数数组逐项相同，不是：

- token 数量相同；
- 文本肉眼看起来相同；
- tool result 大致相同；
- Claude Code 最后完成了同一个任务；
- 两边 SHA 恰好只差一点。

生产 Gate 应直接检查：

```python
assert stage2_prompt_ids == checkpoint.prompt_ids
assert sha256(stage2_prompt_ids) == checkpoint.prompt_sha256
```

### 2.1 只要求分叉第一步与 Stage-1 相同

Hybrid 的目的本来就是从同一个状态重新采样不同动作。因此：

- 所有 Stage-2 branch 的第一次模型调用必须从同一个 `P_t` 开始；
- 第一次生成不同动作以后，各 branch 的后续上下文自然不同；
- 不要求 Stage-2 的整条 trajectory 与原始 Stage-1 相同。

准确的要求是“同起点”，不是“同结果”。

### 2.2 Workspace exact 与 prompt exact 是两件事

一个可分叉的 agent 状态至少包含两部分：

```text
agent state = workspace state + model prompt state
```

- workspace state 决定工具实际看见和修改什么；
- model prompt state 决定模型根据什么上下文选择下一步动作。

只恢复 workspace，会变成“同一份代码交给一个上下文不同的 agent”；只恢复 prompt，不恢复 workspace，工具结果也会错。两者都需要验证。

## 3. 当前实现是如何 resume 的

2026-07-15 已将新任务入口也统一为 `--input-format stream-json`：朴素 GRPO、Hybrid Stage-1、评测和 Stage-2 现在共用同一个 Claude Code command runner。下面 Stage-2 的 transcript 截断与重放流程不变。

当前 Stage-2 路径如下：

```text
Stage-1 .harness/trajectory.jsonl
        ↓
过滤 system、stream_event、hook 等诊断事件
        ↓
保留完整 user / assistant 事件
        ↓
截到与 workspace snapshot 对齐的 tool_result
        ↓
补回最初通过新任务 stream-json 输入、但没有在输出中回显的 user prompt
        ↓
resume_prefix.jsonl
        ↓ stdin
claude -p --input-format stream-json
        ↓
Claude Code 重新构造 Anthropic Messages 请求
        ↓
adapter 再次执行 chat template，得到 Stage-2 prompt_ids
```

相关代码：

- [`transcript.py`](../../../examples/claudecode_ags/step_reconstruct/transcript.py)：校验并构造 `resume_prefix.jsonl`；
- [`claude_stream_input.py`](../../../examples/claudecode_ags/claude_stream_input.py)：统一新任务与 resume 的 user event/JSONL 序列化；
- [`workspace_rebuild.py`](../../../examples/claudecode_ags/step_reconstruct/workspace_rebuild.py)：恢复 workspace 并启动 prefix resume；
- [`agent_runtime.py`](../../../examples/claudecode_ags/agent_runtime.py)：执行 `claude -p --input-format stream-json`；
- [`live_runners.py`](../../../examples/claudecode_ags/step_reconstruct/live_runners.py)：Stage-2 编排；
- [`anthropic_segmented.py`](../../../slime/agent/adapters/anthropic_segmented.py)：把 Claude Code 请求翻译成模型 `prompt_ids`。

### 3.1 为什么不能把原始 trajectory 按行直接喂回去

`.harness/trajectory.jsonl` 是 Claude Code/SDK 的输出流，不是 Claude Code 的内部 session 存档，也不是纯对话输入。开启 partial messages 后，它同时包含：

- `system:init`；
- API 的 `message_start`、`content_block_delta` 等流式增量；
- 聚合完成的 assistant thinking/tool-use 消息；
- hook 生命周期事件；
- user tool result；
- 最终 result。

若把原始前 N 行原样喂回去，同一段 assistant 内容可能同时出现“增量版”和“聚合版”，还会混入非对话事件；最初的新任务 prompt 通常也不在输出 trajectory 中。

所以当前代码进行“语义截断”是必要的。问题不在于它没有按字节截断，而在于：

> Claude Code 对这份语义前缀的重新读取不是一个无损的模型状态反序列化过程。

## 4. 严格实测

### 4.1 测试方法

2026-07-15 做了一次新的两阶段真实 AGS 测试：

1. 使用普通 `claude -p "prompt"` 启动全新的 Stage-1；
2. 在 AGS sandbox 内放置透明 HTTP 转发器，原样记录 Claude Code 发给 Anthropic adapter 的 request body；
3. Stage-1 第一个完整 `Read` 工具边界后，保存真实 tool snapshot 和 transcript；
4. 使用当前生产代码的 `build_transcript_prefix` 和 `rebuilt_workspace(bundle, 0)` 构造 Stage-2；
5. 记录 Stage-2 第一次真实 Messages API 请求；
6. 使用训练所用的 `/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B` tokenizer，按照 `SegmentedAnthropicAdapter` 的相同路径计算完整 `prompt_ids`；
7. 比较完整数组、长度、SHA256 和首个差异位置。

第 1 步描述的是本节历史测试当时的实现。此后代码已把新任务入口统一为 stream-json；本节的 23,803/23,693 等历史数字不会被反向改写，必须在真实 AGS smoke 可用后重新抓取。

测试控制项均通过：

- Stage-1 exit code 为 0；
- Stage-2 exit code 为 0；
- transcript 校验通过；
- workspace rebuild 校验通过；
- Stage-1 和 Stage-2 最终都写出了相同的 probe 文件；
- 两边 tool schema 相同；
- 本地 tokenizer 算出的长度与 adapter 返回的 `input_tokens` 两边都完全一致。

因此，下面的差异不是日志行号比较错误，也不是用了不同 tokenizer。

### 4.2 结果

比较点是：

```text
Stage-1：第一个 Read 工具结果之后的原始下一次模型请求
Stage-2：从同一个 snapshot 重建后的第一次模型请求
```

| 指标 | Stage-1 原始请求 | Stage-2 resume 请求 |
|---|---:|---:|
| `prompt_ids` 长度 | 23,803 | 23,693 |
| SHA256 | `dacc182576701b8c609ff0c237f944801588790eafda237f79ae28e4b9e724eb` | `e338efd8b64062ebe8cc7d96540f23ce57d5f7cdb3e493035e6807ad68d04c2d` |
| 完整数值数组相同 | 否 | 否 |
| 首个不同位置（0-based） | 23,118 | 23,118 |
| 公共前缀 | 23,118 tokens | 23,118 tokens |

测试脚本的非零退出码表示 exact-match 断言失败，不是 AGS 或 Claude Code 运行失败。

### 4.3 110 token 差异的拆解

通过只替换一个变量的离线重算，可以把长度差异拆开：

| 变体 | token 数 | 相对 Stage-2 增加 |
|---|---:|---:|
| Stage-2 原始请求 | 23,693 | 0 |
| 只恢复真实 tool result | 23,721 | +28 |
| 只恢复 Stage-1 消息顺序，仍保留 missing tool result | 23,774 | +81 |
| 同时恢复消息顺序和真实 tool result | 23,802 | +109 |
| Stage-1 原始请求 | 23,803 | +110 |

剩余 1 token 长度差来自两次 workspace 初始化产生的不同 Git commit SHA。即使长度只差 1，完整 token 数组仍从 commit hash 处开始不同。

这是修复前测试的历史结果。2026-07-15 已为 rollout 合成 commit 固定 author/committer 时间；后续相同 tree 的新 workspace 应得到相同 HEAD。该修复不会反向改变本节的历史测试产物，也不会解决其余 109 token 的 resume 差异。详见 [`2026-07-15-sandbox-commit-hash-drift.md`](./2026-07-15-sandbox-commit-hash-drift.md)。

当前机器上的临时证据目录：

```text
/tmp/claude_exact_prompt_resume_20260715_083556/
```

关键文件：

- `report.json`：主结果；
- `component_analysis.json`：110 token 差异拆解；
- `stage1.target_request.json`：Stage-1 原始目标请求；
- `stage2.first_request.json`：Stage-2 第一次请求；
- `resume_prefix.jsonl`：实际喂给 Claude Code 的前缀；
- `stage1.prompt_ids.json` / `stage2.prompt_ids.json`：完整 token 数组。

本文已写入关键数字，因此不依赖 `/tmp` 目录长期保留。

## 5. 三个直接原因

### 5.1 Tool result 在 prefix 中完整，但被 Claude Code 替换

`resume_prefix.jsonl` 的最后一行包含了真实 Read observation：

```text
1\tThis is an exact-resume diagnostic fixture. ...
```

但 Stage-2 发给 adapter 的实际请求中，同一个 `tool_use_id` 的结果变成：

```text
[Tool result missing due to internal error]
```

所以问题不是：

- bundle 没有 transcript；
- transcript 截断过早；
- prefix 漏写 tool result。

正确的描述是：

> tool result 已经进入 Claude Code stdin，但没有被 Claude Code 的 stream-json 重放桥原样恢复到下一次 Messages API 请求。

这一项造成 28 token 长度差，更重要的是模型失去了真实环境 observation。

### 5.2 Claude Code 调整了 user message 的位置

Stage-1 原始请求中的主要顺序是：

```text
skills reminder
current-date reminder
原始任务 prompt
assistant reasoning + Read
真实 Read result
```

Stage-2 请求变成：

```text
current-date reminder
assistant reasoning + Read
missing Read result
skills reminder
原始任务 prompt
```

语义内容大部分仍在，但位置发生了变化。对 Qwen3.5 chat template 而言，assistant 消息是否位于“最后一个 user query”之后会影响历史 reasoning 的渲染。实测中：

- Stage-1 的 295 字符 reasoning 出现在最终 prompt；
- Stage-2 的同一段 reasoning 不再出现在最终 prompt；
- 恢复消息顺序增加 81 tokens。

这说明“内容都在”仍不等于“模型 prompt 相同”。

### 5.3 修复前重建 workspace 的 Git commit SHA 不同

两次 workspace 的代码内容和 diff 一致，但 `slime initial bug state` 是重新创建的 orphan commit，commit 时间等元数据不同，因此 SHA 不同。

Claude Code 会把类似下面的信息放入 system context：

```text
Recent commits:
80748ea slime initial bug state
```

Stage-2 中该 SHA 变成另一个值。把七位 commit hash 归一化后，两边 system text 完全一致；首个 token 差异也正好位于该 hash。

当前 workspace 校验只比较文件 diff，不比较 Git object identity，所以不会发现这一差异。

该初始 commit 漂移现已单独修复并增加回归测试；这里保留的是发现问题时的原始证据。若 agent 在 episode 中自行创建新 commit，目标分叉点的 Git object identity 仍需要另行保存或校验。

### 5.4 修复前 Stage-1 与 Stage-2 的启动入口不同

历史测试中的 Stage-1 使用位置参数：

```text
claude -p "prompt"
```

Stage-2 使用 stdin 事件流：

```text
claude -p --input-format stream-json < resume_prefix.jsonl
```

这会让 Claude Code 经由不同的初始化状态机组装 reminder、原始任务和历史消息，是消息顺序比较中的结构性混杂变量。

代码现已把朴素 GRPO、Hybrid Stage-1、评测和 Stage-2 统一到 stream-json command runner，并让新任务与 `branch_step_t=-1` 共用同一 JSONL helper。本地相关回归通过；真实 Hybrid AGS smoke 因 `sdt-ltpatoxb` 超过 600 秒仍未 sandbox ready 而未完成。因此该变量已在代码层消除，但新的 wire request/token 数字仍待实测，不能沿用历史数字宣称行为级 exact。

详细记录见 [`2026-07-15-claude-launch-mode-token-drift.md`](./2026-07-15-claude-launch-mode-token-drift.md)。

## 6. 为什么这个问题会影响训练

Stage-2 的 prompt token 使用 `loss_mask=0`，因此错误前缀本身不会被直接计算 policy loss。这一点没有问题。

但 Stage-2 新输出是在错误前缀条件下采样的：

```text
错误的 prompt state
        ↓
不同的动作分布
        ↓
不同的工具调用、patch 和 reward
        ↓
这些新输出 token 仍然进入训练
```

因此影响是间接但实质的：

- branch 不再从选中的 Stage-1 状态重新决策；
- step-level reward/advantage 被归因到一个近似状态；
- K 条 branch 虽然可能共享同一个重建方法，但它们没有从 Stage-1 的真实 `P_t` 开始；
- “任务最后完成”只能证明语义续跑可用，不能证明训练状态正确。

对于普通聊天，“大致记得前文”可能足够；对于 step-GRPO，分叉起点本身就是算法定义的一部分。

## 7. 方案比较

| 方案 | 能否保证 token-exact | 主要问题 | 结论 |
|---|---|---|---|
| 原始 output trajectory 按行截断后直接喂入 | 不能 | 混有 partial stream、hook 和诊断事件；缺少最初 `-p` prompt | 不采用 |
| 当前 canonical `resume_prefix.jsonl` | 实测不能 | Claude Code 会重排消息并丢失 tool result | 只可视为 semantic resume |
| Claude Code 原生 `--resume` / `--fork-session` | 不能直接假定 | 官方保证恢复会话语义，不承诺某个内部 tool step 的 `prompt_ids` 完全相同；session 不包含完整 filesystem | 可单独实验，但必须通过 exact gate |
| Adapter 保存并恢复权威 prompt checkpoint | 可以由代码直接保证 | 需要扩展 adapter 状态机和严格校验 | **推荐** |
| 完全自研 agent loop，不使用 Claude Code loop | 可以 | 需要重新实现工具协议、subagent、hook 等大量能力 | 暂不采用 |

### 7.1 为什么不把原生 `--resume` 直接当作最终答案

Claude Code 官方 session 会保存 prompt、工具调用、工具结果和响应，并支持按 session ID resume/fork；它解决的是“继续一段会话”。官方也明确说明 session 保存 conversation，不保存 filesystem。

但是 Hybrid 需要的是：

- 从一条完整 episode 中间的任意目标 tool turn 分叉；
- 不附加新的 user instruction；
- 第一次模型调用与原始目标 turn 的 `prompt_ids` 完全一致；
- 同时恢复对应的 workspace；
- 一次生成 K 个独立 branch。

原生 resume 是否能满足这些额外条件，必须用相同的请求抓取和 prompt hash 测试证明，不能根据“官方叫 resume”推断为 token-exact。

## 8. 推荐方案：由 adapter 保存和恢复权威状态

### 8.1 核心原则

当前路径把 Claude Code 当作“对话状态的权威来源”：

```text
transcript → Claude Code 重建 → adapter 接受其新请求
```

推荐改成：

```text
Stage-1 adapter 保存真实模型状态
                 ↓
Stage-2 adapter 直接加载该状态
                 ↓
Claude Code 只执行新产生的工具调用
```

简单说：

- Claude Code 继续负责 Read、Edit、Write、Bash 等工具执行；
- adapter 负责决定模型到底看见什么历史；
- Stage-2 不再让 Claude Code 重建旧模型历史。

这不需要定制 Claude Code 二进制，只需要扩展现有 `SegmentedAnthropicAdapter`。

### 8.2 Stage-1 保存 `PromptCheckpoint`

当前 adapter 在 [`anthropic_segmented.py`](../../../slime/agent/adapters/anthropic_segmented.py) 中先执行：

```python
prompt_ids = self._render_prompt_ids(target)
```

然后调用：

```python
turn = await call_sglang_generate(prompt_ids, ...)
```

应在两者之间捕获不可变的 prompt state；本轮生成结束后，再单独建立该 checkpoint 与输出 turn/tool IDs 的映射：

```python
@dataclass
class PromptCheckpoint:
    checkpoint_id: str
    prompt_ids: list[int]
    prompt_sha256: str
    chat_messages: list[dict]
    tools_schema: list[dict] | None
    tools_sha256: str
    request_generation_config: dict
    chain_kind: str
    source_request_index: int
    tokenizer_fingerprint: str

@dataclass
class PromptTurnLink:
    checkpoint_id: str
    generated_turn_index: int
    generated_tool_use_ids: list[str]
```

各字段作用：

- `prompt_ids`：Stage-2 第一次调用的直接输入，唯一的 exact source of truth；
- `chat_messages`：Stage-2 生成新动作后，用于继续追加 assistant 和 tool result；
- `tools_schema`：保证工具模板一致；
- `prompt_sha256` / `tools_sha256`：硬校验和审计；
- `request_generation_config`：记录 `max_tokens`、stop、采样配置等影响生成的参数；Stage-2 可使用相同超参数但独立随机采样；
- `PromptTurnLink.generated_tool_use_ids`：在本轮响应生成后，把该 prompt 与随后产生的 Edit/Write/Bash turn 对齐；
- `tokenizer_fingerprint`：防止模型路径、chat template 或 tokenizer 版本变化。

原始 Anthropic request body 可以作为审计字段保存，但不能替代 `prompt_ids`。request body 仍可能因 adapter 预处理、chat template 版本或 tokenizer 变化而重新渲染成不同 token。

### 8.3 将 checkpoint 与目标 edit 对齐

Stage-1 adapter 当前已经记录每个模型 turn 产生的 `tool_use_ids`；workspace snapshot 也记录相同 ID。

对选中的 `edit_step_i`：

1. 找到 `bundle.steps[edit_step_i].tool_use_id`；
2. 找到生成该 tool ID 的 adapter turn；
3. 取该 turn **生成之前**保存的 `PromptCheckpoint`；
4. workspace 使用该 assistant turn 之前最后一个完整 tool snapshot，通常为 `i-1`；
5. 第一处工具动作使用逻辑初始状态 `-1`。

关系如下：

```text
pre-tool workspace snapshot
        +
prompt checkpoint before assistant turn
        ↓
同一个可分叉状态
        ↓
重新采样新的 assistant/tool action
```

若一个 assistant turn 并行产生多个 tool call，这些 tool call 共享同一个 prompt checkpoint。初版应按 assistant turn 分叉并去重，不能把同一模型动作伪装成多个独立起点。

### 8.4 Stage-2 第一次请求只作为握手

Stage-2 可以启动一个新的原版 Claude Code。它的第一个请求用于：

- 建立 HTTP/SSE 连接；
- 声明当前可执行的工具；
- 接收 adapter 返回的新 assistant/tool-use。

但 adapter 不再用该请求中的 system/messages 渲染 prompt。流程是：

```text
open_session(session_id, resume_checkpoint=checkpoint)
        ↓
Claude Code 发来 bootstrap 请求
        ↓
校验当前工具集合与 checkpoint 兼容
        ↓
忽略 bootstrap system/messages
        ↓
直接使用 checkpoint.prompt_ids 调用 SGLang
        ↓
断言 hash 与 Stage-1 完全相同
        ↓
把新生成的 assistant/tool-use 返回 Claude Code
```

bootstrap prompt 不会进入模型上下文。它可以是一个固定短字符串，也可以保留现有调用方式；关键是 adapter 不把它当作历史来源。

因此，当前 `resume_prefix.jsonl` 不再位于 correctness path。最简单的实现可以完全删除 Stage-2 prefix-reseed；若为调试保留，也只能作为 Claude Code 客户端初始化材料，不能参与模型 prompt 构造。

### 8.5 Stage-2 后续请求由 adapter 维护规范历史

第一轮生成后，模型历史由 adapter 自己推进：

```text
checkpoint.chat_messages
+ Stage-2 新生成的 assistant message
+ Claude Code 执行得到的匹配 tool_result
```

具体规则：

1. adapter 保存刚返回给 Claude Code 的 wire `tool_use_id`；
2. Claude Code 执行工具并发来下一次 request；
3. adapter 不接收整段 request history；
4. 只扫描并提取与 pending tool IDs 匹配的新 tool result；
5. 校验 assistant echo 与 adapter 上一轮输出语义一致；
6. 按稳定顺序把 assistant 和 tool result 追加到 canonical `chat_messages`；
7. 重新渲染下一轮 `prompt_ids`。

这样 Claude Code 即使再次注入 skills、current date、bootstrap prompt 或其他启动提醒，也不会污染模型历史。

若一次 assistant turn 产生多个并行工具，必须等所有预期结果到齐，或明确记录部分失败；不能静默补写 missing result。

### 8.6 Stage-2 只训练新生成的 token

恢复 checkpoint 时：

- `chat_messages` 作为 prompt 历史；
- 旧 Stage-1 turns 不复制进 Stage-2 的 `turns` 训练列表；
- checkpoint prompt 的 `loss_mask` 为 0；
- 只记录 Stage-2 新生成的 assistant token、logprob 和 reward。

这保持当前“历史只作条件、branch 输出才训练”的目标。

## 9. Workspace 的推荐补强

Adapter checkpoint 可以保证 Stage-2 第一次模型输入 exact，但工具执行仍依赖新 sandbox。为了让后续工具 observation 可信，workspace 也需要稳定。

### 9.1 最低要求

- 目标文件内容、模式、symlink 和二进制 diff 一致；
- 目标 edit 尚未发生；
- `PROBLEM_STATEMENT.md` 等初始文件一致；
- 不混入上一个 branch 的 `.harness` 或临时文件；
- 重建后用规范 diff/fingerprint 做硬校验。

### 9.2 Git identity

修复前测试中，重新创建 orphan commit 导致 Git SHA 不同。可采用以下方案：

1. workspace scrub 使用固定 author、committer 和固定时间，使相同 tree 产生确定性 commit SHA；或
2. 在 Stage-1 保存并恢复目标 `.git` HEAD/index/object；或
3. 保存包含 `.git` 的完整 workspace snapshot。

方案 1 已于 2026-07-15 实施：rollout scrub 现在使用固定 author/committer 时间创建合成初始 commit，并通过相同 tree 跨时间得到相同 HEAD 的真实 Git 测试。它适合没有 agent 自建 commit 的常见任务。

若 agent 在 episode 内执行过 `git commit`、修改 `.git` 或操作 repo 外文件，则仍需要方案 2/3 一类更完整的 snapshot，或者明确禁止该状态进入 Stage-2。

即使 adapter 忽略新 Claude Code 的 system prompt，Git identity 仍可能被后续 `git log`、`git status` 等工具观察到，因此不能永久忽略。

## 10. 预计代码改动

### 10.1 `slime/agent/adapters/anthropic_segmented.py`

新增：

- `PromptCheckpoint` 数据结构；
- 每次 `call_sglang_generate` 前保存 prompt checkpoint；
- turn → checkpoint → tool-use ID 映射；
- `open_session(..., resume_checkpoint=...)`；
- authoritative-resume 状态机；
- 第一轮直接使用 checkpoint `prompt_ids`；
- 后续只接受 pending tool result；
- prompt/tools/tokenizer hash 硬校验；
- checkpoint 导出接口。

当前 `_parse_and_blocks` 丢弃了 `_build_reply_parts` 返回的 canonical manager message。推荐保留该 message，供 authoritative chain 在返回响应后立即追加 assistant 状态。

### 10.2 `examples/claudecode_ags/step_reconstruct/session_capture.py`

扩展 `SessionBundle`：

- checkpoint 文件索引；
- checkpoint ID 与 tool-use ID 映射；
- prompt hash、token 数、tools hash；
- tokenizer/model fingerprint；
- checkpoint 捕获/序列化错误原因。

`prompt_ids` 建议使用明确字节序的紧凑二进制格式保存，例如 little-endian int32，并单独保存 JSON metadata。不要依赖 Python `repr` 或未定义字节序的 pickle 计算 hash。

### 10.3 `examples/claudecode_ags/step_reconstruct/live_runners.py`

Stage-1：

- 在 finish session 前导出 checkpoint；
- 将 checkpoint 与 `turn_log`、snapshot tool ID 对齐；
- 选择 edit 后定位正确 checkpoint；
- 对同一 assistant turn 的并行 tools 去重。

Stage-2：

- `state.adapter.open_session(..., resume_checkpoint=...)`；
- 不再把 `resume_prefix` 作为模型历史；
- 启动 Claude Code bootstrap；
- 第一次 SGLang 调用前做 exact gate；
- hash 不一致时 branch fail closed。

### 10.4 `workspace_rebuild.py` / `agent_runtime.py`

- 保留当前 pre-edit workspace rebuild；
- 将 prefix-reseed 从 correctness path 移除；
- 增加一个普通 Stage-2 bootstrap runner，或者复用 `run_claude`；
- 保留 Stage-2 原始输出 trajectory 供审计，但不再用它恢复旧 prompt。

### 10.5 测试

新增 adapter checkpoint、authoritative continuation 和真实 AGS exactness 测试。测试标准见第 12 节。

## 11. Fail-closed 规则与指标

以下情况不得静默退化为 fresh prompt 或 semantic resume：

| 条件 | 行为 |
|---|---|
| 找不到生成目标 edit 的 checkpoint | 跳过该 branch point，warning |
| checkpoint prompt hash 校验失败 | 中止 branch，error |
| Stage-2 第一次 prompt 数组不相同 | 中止 branch，error |
| tokenizer/chat-template fingerprint 不同 | 中止 branch，error |
| tools schema 不兼容 | 中止 branch，error |
| pending tool result 缺失、重复或 ID 不匹配 | 中止 branch，warning/error |
| workspace fingerprint 不一致 | 中止 branch，warning/error |
| source turn 属于尚未支持的 subagent/compaction 状态 | 跳过并计数，不猜测恢复 |

建议增加指标：

```text
resume/checkpoint_found_rate
resume/prompt_exact_rate
resume/prompt_hash_mismatch_count
resume/stage1_prompt_tokens
resume/stage2_first_prompt_tokens
resume/first_diff_index
resume/tools_hash_mismatch_count
resume/pending_tool_result_mismatch_count
resume/workspace_exact_rate
resume/unsupported_subagent_count
resume/unsupported_compaction_count
```

正式训练 Gate 要求 `resume/prompt_exact_rate == 1.0`；该值不是接近 1 即可。

## 12. 验收标准

### Gate A：纯 adapter 单元测试

1. 捕获 checkpoint 后重新渲染，得到相同 `prompt_ids`；
2. 序列化再加载 checkpoint，数组和 hash 不变；
3. Stage-2 bootstrap request 的 system/prompt 任意改变，第一次模型 `prompt_ids` 仍不变；
4. Stage-2 第一轮只记录新输出，不把 Stage-1 历史加入 loss；
5. pending tool result 正确追加；错误 ID、缺失结果和重复结果均 fail closed；
6. 多个并行 tool result 按稳定顺序追加；
7. tools/tokenizer fingerprint 变化会阻止 resume。

### Gate B：最小集成测试

构造：

```text
user prompt
→ assistant Read
→ real tool result
→ target assistant Edit
```

验证：

- Stage-1 target Edit 前的 `prompt_ids` 为 `P_t`；
- Stage-2 K 个 branch 的第一次 `prompt_ids` 都严格等于 `P_t`；
- K 个 branch 使用不同采样随机性，可以产生不同 Edit；
- checkpoint 历史全部 mask=0；
- K 个 branch 的新输出正常记录 logprob 和 reward。

### Gate C：真实 AGS smoke

复用本次严格测试方法：

1. 透明记录 Stage-1 目标请求；
2. 从真实 tool snapshot 分叉；
3. 透明记录 Stage-2 第一次请求和 adapter 最终送入 SGLang 的 IDs；
4. 使用训练 tokenizer 比较完整数组；
5. 报告长度、SHA256、首个差异位置。

通过条件：

```text
exact_array_equal = true
stage1_prompt_len == stage2_prompt_len
stage1_prompt_sha256 == stage2_prompt_sha256
first_diff_index = null
```

不能再用以下条件代替：

- Claude Code exit code 为 0；
- 最终 patch 相同；
- token 数量接近；
- transcript validation 为 true；
- workspace diff validation 为 true。

### Gate D：多分叉与训练短跑

- 同一 checkpoint 的 8 个 branch 第一次 prompt hash 全部相同；
- 不同 edit checkpoint 的 hash 按预期不同；
- Stage-2 产生有效不同动作和 reward 方差；
- adapter/session 无串线；
- `loss_group_id` / `loss_weight` 的现有层级加权测试继续通过；
- `λ=0` 时 Stage-1 训练路径仍与朴素 GRPO 对齐。

## 13. 实施顺序建议

### 阶段 1：只捕获，不改变行为

- 在 Stage-1 adapter 保存 checkpoint；
- 把 checkpoint 与 tool snapshots 对齐；
- 在当前 semantic resume 后只做离线对比和指标记录；
- 不影响现有 rollout 输出。

目标：证明每个候选 edit 都能找到唯一、可重渲染的 checkpoint。

### 阶段 2：只保证 Stage-2 第一轮 exact

- Stage-2 adapter 加载 checkpoint；
- 第一次 SGLang 调用直接使用保存的 `prompt_ids`；
- 加 exact hash Gate；
- Claude Code 仍负责执行返回的工具调用。

目标：真实 AGS 测试达到 `exact_array_equal=true`。

### 阶段 3：实现 authoritative continuation

- adapter 保存自己的 canonical assistant state；
- 后续只追加 pending tool results；
- 忽略 Claude Code 重发的旧 system/history；
- 覆盖并行 tools、失败 tools 和无工具终止。

目标：整个 Stage-2 branch 都由同一套规范状态机驱动，而不只是第一轮覆盖 prompt。

### 阶段 4：补强 workspace 与扩展能力

- 确定性 Git baseline 或完整 Git snapshot；
- subagent/compaction 状态支持；
- 性能、checkpoint 大小和磁盘清理；
- 16 卡短跑前的最终 Gate。

## 14. 风险与边界

### 14.1 不应只覆盖第一轮 prompt 后又回到旧逻辑

若只在 Stage-2 第一次调用中替换 `prompt_ids`，下一轮又接受 Claude Code 的整段 distorted history，那么第一步 exact、后续仍会漂移。

因此第一轮覆盖可以作为阶段性 Gate，但最终实现必须由 adapter 维护 authoritative chain。

### 14.2 Tool feedback 不止普通 stdout

PostToolUse hook、失败工具、图片结果、并行工具和 MCP 结果可能使用不同 content shape。提取逻辑必须以 pending tool ID 为主键，并覆盖真实 wire shape；不能只读取一个字符串字段。

### 14.3 Subagent 与 compaction

当前 `SegmentedAnthropicAdapter` 已有 main/sub chain 和 wipe/compact 相关逻辑，但 checkpoint resume 会增加状态组合。初版若不能证明正确，应显式跳过：

- 目标 prompt 位于 active subagent chain；
- source trajectory 在目标点发生未支持的 compaction/wipe；
- 同一 request 无法唯一映射到 main/sub chain。

宁可减少 Stage-2 数量，也不能把未知状态当作普通 main chain。

### 14.4 Checkpoint 体积

本次 prompt 约 24k tokens。若 int32 保存，仅 `prompt_ids` 约 96 KiB；再加 chat messages 和 tools，单 checkpoint 通常仍远小于一个 AGS workspace snapshot。

可以：

- Stage-1 运行时在 adapter 内存中保留所有 checkpoint；
- edit 选择后只持久化被选中的 checkpoint；
- dump 使用压缩、去重的 tools/system 表；
- 训练结束按 run retention policy 清理。

不要为了节省少量磁盘只保存 request body 而删除 `prompt_ids`。

## 15. 对当前实验的含义

- 当前 feature worktree 已修复 transcript 捕获、pre-edit workspace、层级加权等问题；
- 但当前 prefix-reseed Stage-2 尚未满足 token-exact；
- 正在运行或已经完成的旧 Hybrid job 不会自动获得未来 checkpoint-resume 修复；
- 即使当前 run 的 Stage-2 能继续完成任务，也只能解释为 semantic branch；
- token-exact 修复属于训练语义变更，正式公平对照应从共同 base checkpoint 使用新 EXP_TAG 重跑；
- 在 Gate C 通过前，不应提交新的 16 卡正式 Hybrid 长跑。

## 16. 最终建议

推荐把当前设计从：

```text
保存 output trajectory
→ 重建 stream-json prefix
→ 让新 Claude Code 猜回模型历史
```

改为：

```text
Stage-1 adapter 保存 prompt checkpoint
→ Stage-2 恢复 pre-edit workspace
→ Stage-2 adapter 直接加载相同 prompt_ids
→ 原版 Claude Code 执行新工具调用
→ adapter 维护后续 canonical history
```

这条方案满足三个关键条件：

1. **可证明：**完整 token 数组和 hash 可以直接比较；
2. **改动边界清楚：**不需要修改 Claude Code 二进制；
3. **符合 Hybrid 目标：**K 条 branch 真正从同一个 Stage-1 模型状态重新采样。

若未来发现原生 Claude Code session fork 在任意 tool boundary 上也能稳定通过相同 exact Gate，可以再简化实现；在此之前，不应把“resume/continue”这个名称本身当作正确性证明。

## 17. 参考资料

- Claude Code 官方 session 与 resume/fork：<https://code.claude.com/docs/en/agent-sdk/sessions>
- Claude Code CLI session 管理和本地 session JSONL：<https://code.claude.com/docs/en/sessions>
- Claude Code partial stream event 与完整 assistant message 的区别：<https://code.claude.com/docs/en/agent-sdk/streaming-output>
- Claude Code checkpoint/rewind 的范围与限制：<https://code.claude.com/docs/en/checkpointing>
- 本项目旧 Hybrid run 问题报告：[`2026-07-15-hybrid-run-bmr3zlex-bug-report.md`](2026-07-15-hybrid-run-bmr3zlex-bug-report.md)
- 本项目既有修复计划：[`2026-07-15-hybrid-run-bmr3zlex-fix-plan.md`](../plans/2026-07-15-hybrid-run-bmr3zlex-fix-plan.md)
