# Step-GRPO 实现 token-exact resume：问题、方案与经验教训

日期：2026-07-16

适用项目：Claude Code + AGS + SWE Hybrid step-GRPO

适用分支：`feature/cc-ags-swe`

状态：修复已实现并通过单元测试与真实 Stage-2 早期流量验证；两节点 v7 仍在运行，首个 rollout 尚未完成

## 1. 一页结论

Step-GRPO 的 Stage-2 不是“把旧聊天记录截短后重新运行 Claude Code”。严格的
token-exact resume 要同时恢复：

| 层次 | 必须恢复的内容 | 权威来源 |
|---|---|---|
| 模型输入 | 分叉点前的完整 `prompt_ids`、规范消息、tools schema、tokenizer 指纹 | Adapter 保存的 `PromptCheckpoint` |
| Claude Code 工具状态 | 原生 session，以及 Read/Edit 等工具需要的内部状态 | Stage-1 原生 session JSONL |
| Workspace | 文件内容、Git 状态、权限、symlink、纳秒级 `mtime` | 分叉前 workspace snapshot |
| 请求事务 | pending tool ID、已完成 ID、规范历史和最近一次完整响应 | Adapter session state |

前三项决定“从哪里继续”，第四项保证响应丢失或 HTTP 重试时不会把同一轮执行两次。

最终原则是：

```text
模型历史由 adapter 管理；
工具执行由 Claude Code 管理；
文件状态由 workspace snapshot 管理；
transcript 只用于审计，不作为模型状态的权威来源。
```

修复前后的关键差异如下。不同 run 的运行窗口不同，表中数字用于定位协议问题，
不能直接比较模型能力或 resolved rate。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 相同代码树、不同时间创建的合成 Git HEAD | 2 个不同 hash | 2 次得到同一 hash |
| 严格 prompt 比较 | 前 23,118 tokens 相同，第 23,119 个开始不同 | 隔离测试完整 token 数组和 SHA256 相同 |
| 53 份 Stage-1 transcript 通过旧审计 | 26/53 | 41/53；且 transcript 已退出 correctness gate |
| v2 因 tool-use 回显差异硬失败 | 39 条 | v3 早期 243 次差异只记 warning，硬失败 0 条 |
| 旧协议导致的 Stage-2 删除率 | v3 为 994/1,664，59.7% | v7 早期 `missing_result=0`、`no_pending=0`；完整率待 rollout 结束 |
| v6 响应重试后不同 `missing_result` ID | 10 个，缓存重放 0 次 | v7 早期重放 308 次、涉及 161 个 ID，`missing_result=0` |
| v7 早期 `no_pending` | 不适用 | 0 |
| 定向回归 | 不适用 | 119 项：118 通过、1 项环境跳过 |

## 2. token-exact resume 到底要求什么

设 Stage-1 在目标动作生成前真正送入模型的 token 数组为 `P_t`。Stage-2 分叉后的
第一次模型调用必须满足：

```text
stage2_prompt_ids == P_t
```

比较对象是完整整数数组及其 SHA256，不是文本长度，也不是肉眼可见的聊天内容。

| 检查方式 | 能否证明 token-exact | 原因 |
|---|---|---|
| 两边 token 数一样 | 不能 | 相同长度的 token 值仍可能不同 |
| 可读文本看起来一样 | 不能 | system、tools、chat template 和隐藏 reminder 也参与输入 |
| Claude Code 能继续执行工具 | 不能 | 工具状态可用不等于模型输入相同 |
| 两边模型输出碰巧一样 | 不能 | 输出相同不能反推输入相同 |
| 完整 `prompt_ids` 和 SHA256 相同 | 能证明第一轮 exact | 直接比较了实际模型输入 |

Token-exact 只要求分叉的第一轮从 Stage-1 的同一起点重新采样。分叉产生新输出后，
各 branch 本来就允许不同。后续轮的要求是：只在同一权威历史上追加真实的新 assistant
输出和 tool result，不能再次导入 Claude Code 重写过的旧历史。

训练时还必须区分条件与目标：

```text
Stage-1 checkpoint 历史：loss_mask = 0
Stage-2 新生成 token：     loss_mask = 1
```

否则模型会再次训练旧历史，或者把工具输出当成模型输出计算 loss。

## 3. 最终实现流程

```text
Stage-1 每次模型调用前保存 PromptCheckpoint
        + 保存原生 Claude Code session
        + 保存每个工具边界的 workspace snapshot
                         ↓
按目标 edit 的 tool_use_id 找到整个 assistant turn
                         ↓
恢复该 turn 之前的 workspace、session 和 checkpoint
                         ↓
Claude Code 发握手请求；adapter 忽略其重建历史
                         ↓
adapter 直接把 checkpoint.prompt_ids 送入 SGLang
                         ↓
adapter 保存新 assistant 输出和 pending tool IDs
                         ↓
Claude Code 执行工具，只把匹配的真实 result 交回 adapter
                         ↓
adapter 追加新历史并继续；只训练 Stage-2 新 token
```

这套实现不需要修改 Claude Code 二进制。生产环境固定 Claude Code 版本和包哈希，
通过公开的 `--resume <session-file> --fork-session` 恢复工具侧 session。

## 4. 遇到的问题以及如何解决

### 4.1 初始commit hash不同

最初发现相同任务在不同时间运行时，Claude Code system prompt 中的 `Recent commits`
不同。合成 commit 的代码树和 message 相同，但 author/committer timestamp 没有固定。Git hash
包含时间，因此 hash 会变化，并进一步改变 system prompt。下面是一个具体例子。

相同的 system prompt：

<details>
<summary>展开查看两次真实请求中完全相同的 system prompt 尾部</summary>

```text
Current branch: __slime_buggy

Main branch (you will usually use this for PRs): main

Status:
A PROBLEM_STATEMENT.md
?? .harness/

Recent commits:
```

严格 token 对比中，Stage-1 与 Stage-2 的 0-based 下标 `0–23117` 完全相同，共
23,118 个 token。上面展示的是这段共同 system prompt 的真实尾部；完整共同部分还包含
23 个工具定义、工具调用格式、Claude Code 基础规则、auto memory 规则和运行环境信息。

</details>

Commit 差异：

<details>
<summary>展开查看两次真实请求中的 commit hash</summary>

Stage-1：

```text
Recent commits:
80748ea slime initial bug state
```

Stage-2：

```text
Recent commits:
ec3eb88 slime initial bug state
```

对应的最小 diff：

```diff
 Recent commits:
-80748ea slime initial bug state
+ec3eb88 slime initial bug state
```

首个不同 token 的 0-based 下标是 `23118`，即人类计数的第 23,119 个 token。这个位置
正好落在七位 commit hash 内；此时还没有进入原始任务、assistant reasoning 或工具结果。

</details>

修复方式：

- 固定合成 commit 的 author 和 committer 时间（执行该次 `git commit` 时显式设置
  `GIT_AUTHOR_DATE=2000-01-01T00:00:00+0000` 和
  `GIT_COMMITTER_DATE=2000-01-01T00:00:00+0000`，覆盖沙箱继承的当前时间）；
- 固定 Git identity（在该次 Git 命令中通过 `-c user.name=slime` 和
  `-c user.email=slime@local` 固定 author/committer 身份，不读取宿主机的用户配置）；
- 显式关闭 GPG signing（在该次 `git commit` 中通过 `-c commit.gpgSign=false`
  禁止继承环境里的自动签名配置，避免签名内容进入 commit object）；
- 相同 tree 必须产生相同 HEAD，不同 tree 仍必须产生不同 HEAD（测试分别用两个相同 tree、
  一个不同 tree 和三组不同的外部 Git 时间执行真实 scrub，再断言前两者 HEAD 相同、后者不同）。

这些设置只用于创建 rollout 的临时初始 commit，不会修改原项目的 Git 历史。

经验：在排查 resume 之前，必须先证明两次“全新启动”本身可重复，否则看到的差异可能
根本不是 resume 导致的。

### 4.2 Claude 启动方式不同导致的 `system prompt、tool schema、reminder、用户消息和历史消息` 不对齐

修复前，Stage-1 和 Stage-2 通过两种不同入口启动 Claude Code：

| 路径 | 任务和历史从哪里进入 Claude Code | 启动含义 |
|---|---|---|
| Stage-1 | 命令行位置参数 `-p "原始任务"` | 启动新会话，并把命令行中的任务文本作为第一条用户消息 |
| Stage-2 | stdin 中的 `prefix.jsonl` | 创建新会话，再让 Claude Code 重放截断历史 |

对应命令为：

```text
旧 Stage-1：claude -p "原始任务"
旧 Stage-2：claude -p --input-format stream-json < prefix.jsonl
```

两条命令虽然包含相同的任务文本，但进入 Claude Code 状态机的方式不同。真实严格测试中，
被比较的 Stage-1 请求是 `append`，Stage-2 第一次请求是 `new`；tools schema 相同，消息内容
和顺序不同：

| Stage-1 | Stage-2 |
|---|---|
| skills reminder | current-date reminder |
| current-date reminder | 历史 assistant reasoning + Read |
| 原始任务 | missing Read result |
| assistant reasoning + Read | skills reminder |
| 真实 Read result | 原始任务 |

最终渲染后的 prompt 也不同：

| 严格测试 | Stage-1 | Stage-2 |
|---|---:|---:|
| prompt tokens | 23,803 | 23,693 |
| 与另一侧完整数组相同 | 否 | 否 |

这 110 token 的净长度差同时包含启动入口、历史重排、tool result 丢失和 4.1 的 commit hash
漂移，不能全部归因于某一个命令行参数。它能证明的是：两种启动方式不能被默认视为同一个
模型输入入口。

早期排查先统一了新任务的输入入口：将命令行中的任务文本改成规范的 stream-json user
event：

```json
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"原始任务"}]}}
```

朴素 GRPO、Hybrid Stage-1 和评测的新任务都通过 stdin 启动：

```text
claude -p --input-format stream-json < /tmp/slime_cc_initial_prompt.jsonl
```

Hybrid Stage-1 会为每个 trial 指定一个固定的 `--session-id`，让 Claude Code 保存完整会话，
便于之后从分叉点截断。任务文本仍通过同一种 stream-json event 输入，不再直接写在命令行中。

这一步消除了朴素 GRPO、Hybrid Stage-1 和评测的新任务入口差异，便于公平比较和定位问题，
但它本身不能实现 token-exact resume。`-p` 只表示非交互执行；
`--input-format stream-json < ...` 只规定输入格式。无论握手通过 stream-json 还是命令行文本
发送，Claude Code 都可能重新组装 system prompt 和历史。

最终修复不再让 `prefix.jsonl` 重建模型历史。Stage-2 打开从分叉点截断的原生 session，再发送
固定握手 event，触发第一次模型请求：

```text
claude -p \
  --resume /tmp/slime_cc_branch.session.jsonl \
  --fork-session \
  --input-format stream-json \
  < /tmp/slime_cc_resume_handshake.jsonl
```

其中 `--resume` 用于恢复 Claude Code 会话；`--fork-session` 只是让 Stage-2 使用新的 Claude
Code session ID，避免续写原会话。它用于会话隔离，不参与 Adapter 路由，也不保证
token-exact。握手只负责触发请求，不负责恢复模型上下文。

真正保证 token-exact 的是 Adapter：它忽略第一次请求中 Claude Code 重建的 system/messages，
直接加载 Stage-1 在分叉点保存的 `checkpoint.prompt_ids`。因此最终方案分两层恢复：

| 层次 | 最终修复手段 |
|---|---|
| Claude Code 会话状态 | `--resume` 打开截断 session；`--fork-session` 仅隔离会话 |
| 模型可见上下文（核心） | Adapter 按 branch token 加载 Stage-1 checkpoint，并丢弃 Claude Code 重建的上下文 |

因此，统一 stream-json 是前期的入口对齐和诊断手段，`--resume` 负责恢复 Claude Code 的会话
状态；最终保证 token-exact 的核心是 Adapter 将每个 branch 绑定到对应的 Stage-1 checkpoint，
并接管模型可见上下文。



### 4.3 直接截断 trajectory/transcript 不能恢复模型状态

原始想法是把 `trajectory.jsonl` 截到目标 edit 之前，再通过 stream-json 喂给 Claude Code。
实测出现了三种变化：

- 真实 tool result 被替换为 `Tool result missing due to internal error`；
- reminder、原始任务和历史消息被重新排列；
- assistant reasoning 和工具消息被重新合并。

因此 transcript 包含“发生过什么”，但不是模型实际收到的完整 token 状态。修复后：

- Adapter 在 Stage-1 模型调用前直接保存 `prompt_ids`；
- Stage-2 第一次请求只作为握手，不使用其中重建的 system/messages；
- transcript 始终保存且只用于审计，缺失或不完整时记录 warning。token-exact 由 prompt checkpoint、
  原生 session 和 workspace 状态共同验证，必要状态不完整时拒绝该分叉，不改用 transcript
  重放或重新提交原始任务。

先明确这里使用的术语：

| 术语 | 含义 |
|---|---|
| bundle | 一条 Stage-1 trial 保存的整套恢复材料 |
| transcript | Claude Code 运行时写出的 JSONL 事件日志 |
| assistant `message_id` | 标识哪些 JSONL 行属于同一次 assistant 输出 |
| `tool_use_id` | 连接一次工具调用与其返回结果的唯一 ID |
| transcript 审计 | 检查日志中的工具调用是否能找到对应结果 |

bundle 除 transcript 外，还包括 prompt checkpoints、Claude Code 原生 session 和 workspace
snapshots。transcript 审计“通过”不代表 token 完全一致，“未通过”也不直接代表该 bundle
不能分叉。

对 53 个真实 bundle 使用两版 transcript 校验器，结果如下：

| transcript 校验器 | 判定方法 | 通过 | 未通过 |
|---|---|---:|---:|
| 旧版 | 按 JSONL 行顺序逐条匹配工具调用和结果 | 26 | 27 |
| 修正版 | 先按 assistant `message_id` 合并被拆开的输出，再按 `tool_use_id` 匹配并行工具结果 | 41 | 12 |

旧版少通过的 15 条并非真的缺少工具结果。Claude Code 会把同一条 assistant 消息拆成多行，
并行工具也可能按不同于发起顺序的顺序完成；旧版把这两种正常情况误判成 transcript 损坏。

最终选择由 Adapter checkpoint 恢复模型上下文；上述 transcript 校验修复仍保留并落盘，
只用于提供可审计的工程记录，不作为 Stage-2 恢复模型上下文的依据。

是否允许 Stage-2 分叉不使用上表的“通过/未通过”结果，而是检查三项恢复状态：

| 必须通过的检查 | 用途 |
|---|---|
| prompt checkpoint | 确定模型在分叉点实际看到的 token |
| Claude Code 原生 session | 恢复 Claude Code 的会话状态 |
| workspace snapshot | 恢复分叉点对应的代码状态 |

这三项完整时，transcript 审计失败只记录 warning；任一项不完整时，拒绝该分叉。

经验：日志适合审计，checkpoint 才适合恢复。不能因为日志看起来完整，就把它当成模型状态。

### 4.4 分叉点必须按整个 assistant turn 对齐

这里的 assistant turn 是一次模型请求及其完整回答。一次回答可能同时生成多个工具调用：

```text
checkpoint P
  └── assistant 回答：Bash A + Edit B
```

`Bash A` 和 `Edit B` 都由模型基于同一个 prompt `P` 一次生成，模型并不存在“已经生成
Bash A、但还没有生成 Edit B”的中间状态。因此，即使 Stage-2 是因为 `Edit B` 的 PPL 较高
而选择这个分叉点，也必须回到整次回答之前，让模型重新生成完整回答，不能只从 transcript
中的 `Edit B` 那一行开始。

模型上下文、Claude Code session 和 workspace 也必须回到同一个位置：

```text
模型上下文：checkpoint P
Claude Code session：包含这次 assistant 回答之前的历史
workspace：执行 Bash A 和 Edit B 之前的状态
```

如果模型恢复到 `P`，workspace 却已经执行过 `Bash A` 或 `Edit B`，Stage-2 重新生成工具时
就会在错误的代码状态上继续。

修复方式：

- Adapter 记录每个 checkpoint 一次生成的全部 `tool_use_id`；
- 选中其中任一 Edit 时，恢复这组工具生成前的 checkpoint、原生 session 和 workspace；
- 如果这是第一轮工具调用，使用 snapshot `-1`，即 Agent 尚未执行任何工具时的初始 workspace；
- 无法确认整组工具执行前 workspace 的候选直接跳过。

只用 Git diff 可以恢复文件内容，但应用 diff 会改变部分文件属性。Claude Code 的 Edit 工具
会检查文件在 Read 之后是否被外部修改；即使内容相同，只要修改时间 `mtime` 不同，也可能
拒绝 Edit。真实测试中，恢复相同内容但使用新的 `mtime` 时 Edit 失败，恢复分叉点记录的
`mtime` 后才能继续。

因此 workspace snapshot 还要记录并校验：

| 状态 | 不一致时的影响 |
|---|---|
| 文件权限 | 文件可能无法读取、写入或执行 |
| symlink 类型和目标 | 同一路径可能指向不同文件，或被错误恢复成普通文件 |
| Git index | `git status`、`git diff` 和 Claude Code 看到的仓库状态可能改变 |
| `mtime` | Claude Code 可能把已读取的文件判定为后来被修改，从而拒绝 Edit |

经验：模型起点、Claude Code session 起点和 workspace 起点必须指向同一个逻辑 turn。

### 4.5 第一次请求对齐后，还要正确维护后续工具循环

`checkpoint.prompt_ids` 只保证 Stage-2 第一次模型请求准确。模型生成第一条回答后，Adapter
还要持续连接模型和 Claude Code，直到 branch 结束。先明确三个术语：

| 术语 | 含义 |
|---|---|
| `tool_use` | 模型要求 Claude Code 执行的一次工具调用 |
| `tool_result` | Claude Code 执行该工具后返回的结果 |
| pending tool | Adapter 已发送给 Claude Code、但尚未收到 result 的工具调用 |

一次正常循环是：

```text
Adapter 用 checkpoint.prompt_ids 请求模型
→ 模型返回 assistant 消息和 tool_use
→ Adapter 保存这条模型原始输出，并把 tool_use 标记为 pending
→ Claude Code 执行工具并发送 tool_result
→ Adapter 将 result 接到自己保存的历史后，再请求模型
```

v3 运行中计划执行 1,664 条 Stage-2 branch，其中 994 条在这个后续循环中被删除：

| 当时的删除原因 | 数量 | 占全部计划分支 |
|---|---:|---:|
| 请求没有完整回放上一条 assistant `tool_use` | 827 | 49.7% |
| Adapter 没有等待中的工具，但又收到请求 | 130 | 7.8% |
| Claude Code 当前提供的 tools schema 与 checkpoint 不同 | 29 | 1.7% |
| workspace 恢复失败 | 8 | 0.5% |
| 合计 | 994 | 59.7% |

前三项属于 Adapter 与 Claude Code 的续跑协议问题；workspace 恢复失败是独立问题，处理方式
见 4.4。前三项分别按下面的规则修复。

#### 4.5.1 不要求 Claude Code 重新提供上一条模型输出

Claude Code 发送 `tool_result` 时，有时会同时带回上一条 assistant `tool_use`，有时不带，
也可能调整工具 name/input 的表示方式。这个带回来的副本称为“回显”。

旧代码要求回显与上一条模型输出完全一致，但 Adapter 本来就保存了模型的原始输出，因此
不应再用 Claude Code 的副本覆盖它。修复后的规则是：

- assistant 消息和 `tool_use` 以 Adapter 保存的模型原始输出为准；
- Claude Code 的回显可以缺失或改变表示方式，差异记录 warning；
- `tool_result` 必须使用正确的 pending ID，且每个 ID 恰好返回一次；
- 未知 ID、result 缺失、重复或并行结果不完整仍然立即失败。

修复前后的真实运行差异：

| 运行快照 | 完成 branch | 发现回显差异 | 因回显差异失败 |
|---|---:|---:|---:|
| v2 | 1 | 39 | 39 |
| v3 早期 | 45 | 243 | 0 |

#### 4.5.2 没有 pending tool 时，根据上一轮结束原因处理

模型上一轮可能没有生成工具调用，因此 Adapter 此时没有 pending tool。旧代码在随后收到
Claude Code 请求时一律报错，但两种请求是正常的：

| 上一轮 `stop_reason` | 正确处理 |
|---|---|
| `max_tokens` | 上一轮输出因长度限制被截断，从 Adapter 保存的历史继续生成 |
| `end_turn` | 任务已经结束，返回空的结束确认，不再调用模型 |
| 其他或未知原因 | 拒绝请求并记录错误，不猜测状态 |

#### 4.5.3 checkpoint schema 和 runtime schema 用途不同

tools schema 描述模型可以调用哪些工具以及每个工具的参数格式。这里同时存在两份 schema：

| schema | 用途 |
|---|---|
| Stage-1 checkpoint schema | Stage-1 prompt 的组成部分，用它渲染模型上下文才能保持 token-exact |
| Stage-2 runtime schema | Claude Code 当前实际可以执行的工具，用于检查新工具调用能否执行 |

两份 schema 的顺序、描述或可用工具可能不同。修复后不再用 runtime schema 改写 checkpoint
prompt；Adapter 继续用 checkpoint schema 构造模型输入，同时记录两者差异，并按 runtime
schema 校验模型新生成的工具调用。无法执行的调用仍然拒绝，详见 4.7。

### 4.6 HTTP 响应丢失会把正常重试伪装成 resume 错误

Adapter 生成工具调用后，可能已经更新 session，但 HTTP 响应没有成功到达 Claude Code。
客户端会重发旧请求。若 adapter 再次处理，就会出现：

```text
expected one result for tool_use ..., got 0
received resumed request without pending tool calls
```

最初使用完整 HTTP JSON 的 SHA 判断重复请求，但 Claude Code 重试时会改变 `model`、
`stream`、`metadata`、cache-control、tools 顺序，甚至重写 system/messages。完整 SHA 和
“忽略部分字段”的逻辑 SHA 都不能稳定命中。

最终方案以 session 状态而不是请求文本判断：

```text
adapter 当前有 pending tool calls
+ 新请求没有任何当前 pending ID 的 result
+ 新请求没有未知 result ID
→ 只能重放上一次完整缓存响应
```

重放必须保持原 content、tool ID、usage 和 message ID，不调用 SGLang，不新增训练 turn，
也不修改 session。消费 tool result 到生成下一轮响应之间还要有事务快照；生成或传输失败时
恢复 pending IDs、completed IDs、规范历史和 stop reason。

真实运行的演进如下：

| Run | 已完成 Stage-2 branch | 缓存重放 | 不同 `missing_result` pending ID | 结论 |
|---|---:|---:|---:|---|
| v5 | 82 | 未命中 | 6 | 完整请求 SHA 过严 |
| v6 | 36 | 0 | 10 | 逻辑 SHA 仍会被历史重写绕过 |
| v7 早期 | 87 | 308 次，涉及 161 个 pending ID | 0 | pending-state replay 覆盖了真实重试 |

v7 的 308 次重放中，308/308 的逻辑 SHA 和完整 wire SHA 都与缓存不同。如果仍依赖请求
内容判断，这些请求都会漏掉。状态判断成功重放后，branch 继续完成，`no_pending` 也保持 0。

经验：有副作用的多轮协议必须先定义事务提交点和幂等规则，不能把 HTTP 成功送达当成必然。

### 4.7 工具名称正确，不代表工具调用可执行

真实运行中出现过：

- 模型调用不存在的 `Delete` 或 `Install`；
- `Edit` 多出 `lowerbound_line`；
- `Grep` 缺少必填 `pattern`；
- `Read` 多出 `block/timeout`。

若 adapter 把这些调用写成 pending，Claude Code 不会执行，也不会返回 result，最后会被误报成
resume 丢失。因此每次生成后、写入 pending 前，必须按当前 Claude Code runtime JSON Schema
校验工具名称和 input。

校验失败要记录为 `resume_tool_schema`，不能记为 `missing_result`。这能区分：

```text
协议状态丢失
≠ 模型生成了客户端无法执行的工具调用
```

### 4.8 并发性能也会放大协议故障

Adapter 的请求 handler 曾在 asyncio event loop 中同步执行大 JSON 解析、hash、tokenize、
checkpoint 序列化和模型输出 parse。16 卡和高并发下，其他 session 的响应会排队，增加超时
和重试概率。

修复包括：

- 把 CPU 重任务放入有界线程池；
- 复用 SGLang HTTP 连接；
- 去掉重复 hash；
- 分开记录 event-loop delay、CPU queue、tokenize、SGLang 和 session lock 时间。

这不是 token-exact 算法的一部分，但会直接影响协议是否稳定。正确性测试通过后，仍要在目标
并发下验证延迟和重试。

## 5. 修复后的验证数字

### 5.1 隔离 token 数组验证

| 场景 | prompt tokens | 结果 |
|---|---:|---|
| Stage-2 第一次调用直接加载 checkpoint | 23,617 | 完整数值相同 |
| 同一动作后的第二次调用 | 23,732 | 完整数值相同 |
| 再执行一个工具后的第三次调用 | 23,787 | 完整数值相同 |
| 同一 assistant turn 并行生成 Read+Bash | 23,652 | 完整数值相同 |
| 并行 Bash 返回 `is_error=true` | 23,654 | 完整数值相同 |

这些 token 数只属于测试样例。生产 Gate 检查完整数组和 SHA256，不要求长度等于表中数字。

### 5.2 v7 真实早期快照

快照时间：2026-07-16 05:52 UTC

实验：`qwen35_9b_cc_ags_2node_hybrid_c64_t45_tokenexact_v7`

W&B：`https://wandb.ai/models-tencent7723/coding-rl/runs/6qfoon5g`

| 指标 | 数量 |
|---|---:|
| 已完成 Stage-1 episode | 123 |
| 已完成并保存的 Stage-2 branch | 87 |
| Stage-2 `exit=0` | 68 |
| Stage-2 `exit=1` | 19 |
| 正 reward branch | 17 |
| pending-state replay | 308 |
| replay 涉及的不同 pending ID | 161 |
| replay 时逻辑 SHA 变化 | 308/308 |
| `missing_result` | 0 |
| `no_pending` | 0 |
| agent 45 分钟超时（`exit=-1`） | 0 |
| pod restart | 0 |

`exit=1` 表示 Claude Code 进程结果，不等于 resume 校验失败；已保存 branch 在进入该日志前必须
通过 checkpoint、workspace 和 token-exact gate。正 reward 只反映当前已完成样本的 SWE 结果，
不用于证明 resume 正确。

该快照没有 rollout dump、optimizer update 或 checkpoint，不能写成完整训练验收，也不能与 v3
完整 run 的 resolved rate 直接比较。它能证明的是：旧故障在更高强度的真实重试下没有复现。

### 5.3 回归测试

定向 suite 共 119 项：

| 结果 | 数量 |
|---|---:|
| 通过 | 118 |
| 环境条件跳过 | 1 |
| 失败 | 0 |

覆盖范围包括第一次和后续 prompt exact、并行与失败工具、回显缺失/变化、`max_tokens`、
`end_turn`、runtime schema、请求重放、生成失败回滚、native session、workspace 恢复、
Stage-2 编排、分组加权和 W&B 指标。

## 6. 对训练正确性的影响

Token-exact resume 不只是工程优化。若起点不一致，Stage-2 的 logprob 对应另一个上下文，
edit-PPL 选出的分叉点与实际训练起点也不一致，step-level advantage 会失去明确含义。

最终实现遵守以下边界：

- Stage-1 完整 episode 继续按原有 episode group 计算；
- Stage-2 按 edit group、再按组内 branch 计算和加权；
- 恢复历史不重复进入训练 turn；
- 只训练 Stage-2 新生成的模型 token；
- tool result 和上下文尾部保持 `loss_mask=0`；
- resume 修复不改变 Stage-1/Stage-2 的目标函数和组权重。

还必须同时统计计划、完成、删除 branch 及删除原因。旧 v3 中幸存 branch 的
`resume/prompt_exact_rate` 可以是 1.0，但 59.7% 的计划 branch 已在此前被删除。只看幸存者
比例会掩盖协议问题。

## 7. 必须坚持的实现规则

| 规则 | 原因 |
|---|---|
| 第一次输入只信 checkpoint 的完整 token 数组 | 防止 reminder、schema 和历史重排进入模型输入 |
| 后续 assistant 历史只信 adapter 自己返回的内容 | Claude Code 回显可能缺失或规范化 |
| tool result 按 pending ID 完整、唯一校验 | 防止错接、漏接或重复执行 |
| transcript 缺失必须 warning，但不能静默 fallback | 审计信息缺失不能改变训练语义 |
| workspace、session 或 tokenizer 校验失败时 fail closed | fresh agent 不是同一个分叉起点 |
| runtime 工具调用在 pending 前校验 | 不把不可执行生成误报成 resume 故障 |
| session 变更使用事务快照 | 上游失败后允许安全重试 |
| 重复请求返回完全相同的缓存响应 | 不重新采样、不改变 tool ID、不新增训练 token |
| 指标同时统计分母和失败分类 | 防止只看幸存 branch 得出错误结论 |

## 8. 当前限制

- 当前只支持主 Claude Code chain；目标位于 subagent 或 compaction/wipe 时跳过；
- Stage-2 新生成 `Task/Agent` 子代理调用时 fail closed；
- 显式 workdir 外依赖会阻止分叉，Bash 隐式外部依赖仍无法完全静态证明；
- Claude Code 版本、session 文件格式、tokenizer 和 chat template 变化后必须重新做 smoke；
- v7 首个 rollout 尚未完成，训练更新、checkpoint 和最终 SWE 评测仍需后续验收。

这些限制必须单独计数，不能归入 `missing_result`，也不能静默转成普通 Stage-1 episode。

## 9. 最重要的经验教训

1. Token-exact 的证据只能来自实际 `prompt_ids`，不能来自文本、长度或行为相似。
2. Transcript 是事件日志，不是模型 checkpoint；原生 session 也不能替代模型 checkpoint。
3. 模型状态、工具状态和 workspace 状态必须恢复到同一个逻辑 turn。
4. 第一次 prompt exact 只是起点；后续历史必须由 adapter 继续维护。
5. 客户端回显不是权威 assistant 输出，真实 tool result ID 才是协议边界。
6. 网络重试必须按 session 状态幂等，不能依赖请求 JSON 恰好相同。
7. 不可执行工具调用必须在写入 pending 前拒绝，否则会污染 resume 统计。
8. Fail-closed 必须有清晰分类和分母；静默 fallback 会直接污染训练数据。
9. 并发性能、排队和事务超时会放大正确性问题，必须在真实并发下验证。
10. 完整训练验收必须包含 rollout dump、loss mask、logprob、reward、optimizer update 和 checkpoint；
    早期协议 smoke 不能替代它。

## 10. 相关文档

- [完整修复方案](2026-07-15-hybrid-token-exact-resume-solution.md)
- [早期问题与推荐架构](2026-07-15-hybrid-token-exact-resume.md)
- [合成 Git commit hash 漂移](2026-07-15-sandbox-commit-hash-drift.md)
- [Claude Code 启动入口导致的 prompt 漂移](2026-07-15-claude-launch-mode-token-drift.md)
- [Adapter event loop 阻塞](2026-07-15-adapter-event-loop-blocking.md)
