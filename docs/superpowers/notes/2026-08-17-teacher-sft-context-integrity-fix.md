# Teacher SFT 上下文与 JSONL 完整性修复

日期：2026-08-17
范围：Turn-level Teacher SFT 的上下文、落盘与严格校验；不改变选点和训练目标。

## 1. 不在本次修改范围内

以下行为是当前方法的设计，不作为 bug 修改：

- 只在失败 trial 的可恢复工具 turn 上运行 continuation；
- Teacher 默认只运行两轮，不执行终局 resolved 评测；
- `sft_only` 不训练成功轨迹或 Stage-1 GRPO；
- `all` 模式按现有工具事件口径展开，包括 Read / Grep / Bash。

## 2. 当前问题

### 2.1 Teacher 生成上下文与学生训练上下文不同

同一个 relabel 目前存在两份历史：

| 名称 | 来源 | 实际用途 |
|---|---|---|
| 权威 checkpoint | Stage-1 adapter 保存的 `chat_messages`、`tools_schema`、`prompt_ids` | 学生训练前缀 |
| Claude Code resume 历史 | 截断后的 native session 经 Claude Code 重建的 HTTP 请求 | Teacher 实际推理输入 |

Claude Code resume 会加入合成终止消息、握手消息和 reminder，也可能改变消息排列。因此 Teacher 的回答是在第二份历史下生成的，却被接到第一份历史后训练。这不是 token 数量误差，而是监督条件发生了变化。

正确的数据必须满足：

```text
Teacher 第 1 轮可见历史 = checkpoint.chat_messages
Teacher 第 2 轮可见历史 = checkpoint.chat_messages
                         + Teacher 第 1 轮回答
                         + 第 1 轮工具的真实执行结果
学生训练历史             = 同一条消息链
```

Claude Code 的 native session 仍用于恢复 harness 状态和执行工具，但它重放的旧 HTTP 历史不能再决定 Teacher 可见上下文。

### 2.2 上下文不一致后仍继续拼接

旧编码器有两层 fallback：

1. 下一轮 prompt 与已构造历史不一致时，仍按消息数量取后缀；
2. 渲染 token 与 `checkpoint.prompt_ids` 不一致时，仍按 `prefix_len` 强制切接。

隔离旧 JSONL 后的 step2–4 共 3,665 个 Teacher 样本中：

| 检查 | 数量 |
|---|---:|
| 两轮样本 | 3,657 |
| 下一轮 prompt 更短 | 820 |
| prompt 历史不同但按位置拼接 | 2,837 |
| token prefix mismatch | 1,852 / 3,665 |

3,657 个两轮样本全部进入了消息级 fallback。warning 只能记录问题，不能保证训练数据正确。

### 2.3 旧 JSONL 会在进程重启后重新混入

旧实现同时满足以下条件：

- Teacher session id 可重复；
- session id 固定映射到同一个文件名；
- JSONL 使用追加写；
- `turn_index` 只保存在进程内，重启后从 0 开始；
- 编码器读取整个文件，不检查 turn 是否属于本次 attempt。

因此同一 relabel 重跑后，文件可能变成：

```text
旧 turn 0
旧 turn 1
新 turn 0
新 turn 1
```

按 `turn_index` 排序后会交错成两段不同运行的历史。2026-08-16 10:03 的人工隔离使 step2–4 当前文件恢复干净，但没有消除代码复发条件，也没有回滚 step0–1 已执行的优化更新。

### 2.4 Stage-1 与 Teacher 日志目录未隔离

Teacher adapter 启动时修改了全局 `SLIME_AGENT_SFT_LOG_DIR`。随后启动的 Stage-1 adapter 也会把学生请求写入 `teacher_sft_turns`。当前目录里同时出现普通 Stage-1 文件和 `provider=remote_openai` 的 Teacher 文件。

两类 session id 当前不同，未直接造成训练混入，但会干扰审计、清理和防复发判断。

### 2.5 首次修复在 Lustre 上无法写 Teacher JSONL

首次实现为了独占 attempt 文件名，先用 `O_CREAT | O_EXCL` 创建 0 字节 JSONL，再由请求
处理函数用追加模式写入第一条记录。该顺序在本地临时目录可用，但在训练使用的
`/mnt/sn-007` Lustre 挂载上，第二次打开稳定返回 `EINVAL`。

新任务 step 0 的现场数据如下：

| 现场项 | 数量 |
|---|---:|
| 已创建 Teacher attempt JSONL | 25 |
| 0 字节 JSONL | 25 |
| JSONL 写入 `EINVAL` | 32 |
| 随后出现的上下文拒绝 | 16 |
| 已发生参数更新 | 0 |

对同一挂载的独立探测结果：直接创建并写入正常；先创建空文件再追加，无论追加 16 B、
1 MiB 还是 8 MiB，均返回 `EINVAL`。因此问题与 Teacher 输出长度、请求并发或模型接口无关。

写盘失败还暴露了第二个问题：旧代码在落盘前已经推进 `turn_index`，并记录“正在等待某个
tool result”。Claude Code 没有收到这次 HTTP 响应，重试的仍是原握手请求；Adapter 却把它
当成下一轮，因而报“缺少 tool result”。这里不是 Claude Code 丢了工具结果，而是 Adapter
在持久化失败后留下了半提交状态。

### 2.6 完整重编码会在 resume 边界合并 token

第二次重提后，Teacher 请求和 JSONL 已经正常，但严格编码 gate 仍丢弃部分合法样本。对首批
432 个真实 attempt 审计后发现：360 个可以直接通过，72 个只在 checkpoint 的最后一个 token
处不同。

这 72 个样本的 checkpoint 文本都以 `<think>\n` 结束，Teacher 回答从换行开始。对完整文本
重新做 BPE 时，tokenizer 会把边界两侧的两个换行合成一个 token。因此：

- checkpoint messages 重新渲染得到的 token 与保存的 `prompt_ids` 完全一致：432 / 432；
- Teacher 完整渲染文本以 checkpoint 文本开头：432 / 432；
- 但完整文本的 token 数组不一定以 `prompt_ids` 开头：72 / 432。

这不是上下文漂移，而是 BPE 在字符串拼接边界不具备 token-prefix 稳定性。旧 gate 把“文本
前缀一致”错误地等同于“完整重编码后的 token 数组也是前缀”，导致合法样本被误杀。

## 3. 修复方案

### 3.1 Adapter 维护权威 Teacher 历史

每个 branch 启动前，`teacher_branch_runner` 向 Teacher adapter 注册：

- checkpoint id；
- `checkpoint.chat_messages`；
- `checkpoint.tools_schema`；
- `checkpoint.prompt_sha256`。

Teacher adapter 的处理规则：

1. 第一次 Claude Code 请求只作为握手，远端 Teacher 的 `messages/tools` 直接取 checkpoint；
2. 保存 Teacher 的真实回答及新生成的 tool ids；
3. 第二次请求只从 Claude Code 请求中提取这些 tool ids 对应的真实 tool result；
4. 把 tool result 接到 adapter 自己维护的历史，再请求 Teacher；
5. 不读取 Claude Code 重放的旧 system/messages 来重建模型上下文；
6. 未注册 session、缺失 tool result、重复 tool result 或未知状态均明确拒绝，不退化为普通新会话。

这与现有 token-exact actor adapter 的原则一致：harness 负责执行，adapter 负责模型可见历史。

### 3.2 编码端改为严格 correctness gate

编码时必须同时满足：

- `turn_index` 从 0 连续递增且无重复；
- 第一轮 `prompt.messages` 与 checkpoint messages 完全一致；
- 每一轮 `prompt.messages` 与上一轮权威历史完全一致；
- tools schema hash 与 checkpoint 一致；
- checkpoint messages 单独渲染后与保存的 `prompt_ids` 完全一致；
- Teacher 完整渲染文本必须以 checkpoint 渲染文本开头；
- loss mask 只覆盖 Teacher assistant token。

任一条件失败时抛出带原因的异常，该 relabel 被计入丢弃指标，不能进入训练。删除所有“按消息位置继续”和“按 token 位置强制拼接”的 fallback。

### 3.3 每次运行使用独立 JSONL attempt

Teacher session 注册时创建独立 attempt 文件：

```text
<session_hash>.<attempt_id>.sft_turns.jsonl
```

修正后的实现使用独立的非空 `.claim` 文件，通过 `O_CREAT | O_EXCL` 原子占用 attempt 名称；
JSONL 本身不预先创建，由第一条真实记录正常创建并写入。这样同时满足：

- attempt 名称不会与并发或旧运行冲突；
- 不触发 Lustre 的“空文件预创建后再追加”错误；
- 没有 Teacher 响应时不会留下看似存在但实际为空的训练数据文件。

branch runner 只读取注册时返回的 JSONL 路径，不再根据 session id 推导旧文件。即使进程
重启、step 重跑或 session id 重复，也会产生新文件，不会把两次运行追加到同一 JSONL。

文件内仍严格检查 `turn_index = 0..N-1`，形成第二道防线。

### 3.4 日志目录隔离

明确使用两项配置：

| 配置 | 内容 |
|---|---|
| `SLIME_AGENT_SFT_LOG_DIR` | Stage-1 学生请求 |
| `SLIME_TEACHER_SFT_LOG_DIR` | 远端 Teacher 请求 |

Teacher adapter 不再修改全局 `SLIME_AGENT_SFT_LOG_DIR`。

### 3.5 落盘成功后再推进 Adapter 状态

一次 Teacher 请求现在按以下顺序提交：

1. 基于当前已提交状态构造远端请求；
2. 获得 Teacher 响应后，先生成“待提交记录”和下一状态，但不修改当前状态；
3. JSONL 写入成功后，才同时更新历史、`turn_index`、待执行工具和缓存响应；
4. 写入失败则返回 500，保留同一份待提交记录；Claude Code 重试完全相同的请求时，Adapter
   只重试落盘并返回原 Teacher 响应，不再次调用远端模型；
5. 持久化未完成时若收到不同请求，明确拒绝，不能跨过缺失记录继续运行。

因此“日志已落盘”和“会话已前进”成为同一个提交边界，不会再出现 JSONL 为空但 Adapter
已经等待下一轮 tool result 的状态。

Adapter 启动时还会在真实日志目录执行一次 `claim → append → readback → cleanup` 预检。
挂载不支持当前落盘方式时，服务会在接收 Teacher 请求前直接失败。

### 3.6 保留 checkpoint token，只编码 Teacher 后缀

Qwen3.5 编码按以下方式处理 resume 边界：

1. 单独渲染 checkpoint messages，并验证结果与保存的 `prompt_ids` 完全一致；
2. 渲染 checkpoint + Teacher continuation，验证其文本以 checkpoint 文本开头；
3. 直接保留原始 `prompt_ids`，只对 checkpoint 文本之后的后缀单独分词；
4. 根据完整渲染文本中的 assistant 字符区间，为后缀 token 生成 loss mask；
5. 验证“原始 `prompt_ids` + 后缀 token”解码后与完整渲染文本完全一致。

只有以上五项同时成立才进入训练。这不是忽略 token 差异继续拼接；它明确验证文本、模板、
checkpoint token 和最终解码结果，只避免 BPE 跨 resume 边界重新改写已经实际用于推理的
`prompt_ids`。

## 4. 测试要求

必须覆盖：

1. 第一次 Teacher 请求忽略 resume 握手历史，实际 payload 等于 checkpoint；
2. 第二次请求只追加真实 tool result；
3. message drift、缺失 tool result、重复 `turn_index`、tools hash 不一致、token prefix 不一致均拒绝；
4. 相同 session id 注册两次得到两个独立文件，旧文件内容不会被读取；
5. Stage-1 与 Teacher 日志落入不同目录；
6. 正常两轮样本仍能编码，prefix mask 为 0、Teacher assistant mask 为 1；
7. 注入首次 append 失败后，`turn_index` 和待执行工具不前进；完全相同的请求可只重试落盘，
   不重复调用远端 Teacher；
8. 在训练使用的 Lustre 目录执行真实 claim / append / readback 预检。
9. 构造边界换行被 BPE 合并的案例，验证 checkpoint token 不变、Teacher 后缀可训练；并用
   真实受影响 task 的全部 attempt 回放编码。

## 5. 验收标准

- 运行日志中不再出现 message/token fallback warning；
- 上下文异常只能表现为 `perf/step_grpo/n_dropped_context_integrity`，不能进入训练；
- 每个 Teacher attempt 文件只有连续的 `turn_index`；
- Teacher 落盘的 `prompt.messages` 与实际发给远端模型的 payload 完全相同；
- 第一次 Teacher prompt 的 checkpoint digest 与 Stage-1 checkpoint 一致；
- 重跑相同步骤不会读取此前 attempt 的 JSONL；
- Adapter 启动日志出现 `Teacher SFT filesystem preflight passed`；
- 写盘失败时不出现由半提交状态引起的伪 `missing tool_result`。

## 6. 实施与验证结果

上述修复已经写入代码：

- Teacher adapter 按 branch 注册 checkpoint，并维护后续 assistant / tool-result 历史；
- branch runner 只读取注册时返回的唯一 attempt 路径；
- 编码器删除两处 fallback，旧版 `version=1` JSONL 也会被明确拒绝；
- Qwen3.5 使用真正的 `qwen3_5` loss-mask 实现，不再退回 `qwen3`；
- 启动脚本显式配置两个日志目录，运行时再次检查二者不能相同。

验证结果：

| 验证 | 结果 |
|---|---:|
| Teacher / Hybrid 定向单测 | 76 passed |
| 注入首次 append 失败后原请求重试 | 远端 Teacher 只调用 1 次，状态从 turn 0 正常提交 |
| `/mnt/sn-007` 真实 claim / append / readback | PASS |
| 真实 Qwen3.5 tokenizer + 真实 checkpoint | 23,646 / 23,646 个 prefix token 一致 |
| 上述真实 checkpoint 的两轮示例 | 58 个 Teacher token 可训练 |
| 首个受边界合并影响的真实 task 回放 | 49 / 49 通过；其中 11 条走 checkpoint-preserving 后缀编码 |
| 远端接口对 checkpoint 规范工具消息的合成探测 | HTTP 200 |

首次重提发现 2.5 的 Lustre 与事务边界问题；第二次重提确认 JSONL 落盘正常，同时暴露 2.6
的 BPE 边界误判。两项问题修复后，任务再次从 Base / step 0 干净重提。在线初始验收快照如下：

| 在线验收项 | 结果 |
|---|---:|
| 模型起点 | Base，Megatron iteration 0；`load_ckpt_step=null` |
| Adapter 启动文件系统预检 | PASS |
| step 0 进度 | 5 / 16 个 task 返回 |
| 首批真实 Teacher attempt | 62 个完整 attempt |
| 首批真实 Teacher turn 记录 | 124 条；每个 attempt 均为连续的 `[0, 1]` |
| 空 JSONL / 无效 JSONL | 0 / 0 |
| `Invalid argument` | 0 |
| 已观察到的 `teacher_context_integrity` / `missing tool_result` | 0 / 0 |

两个诊断现场分别归档到
`fail_imitation_learning_pre_lustre_fix_20260817_step0` 和
`fail_imitation_learning_pre_boundary_token_fix_20260817_step0`。最终任务使用清空后的
`fail_imitation_learning` 目录，未加载训练 checkpoint。
