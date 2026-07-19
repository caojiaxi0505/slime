# Claude Code 启动路径不一致导致的 prompt 漂移

日期：2026-07-15

适用分支：`feature/cc-ags-swe`

状态：代码已修复并通过本地回归；真实 AGS wire smoke 因 Hybrid tool 启动超时尚未完成；未提交 commit，未重启训练任务

关联实现：`examples/claudecode_ags/agent_runtime.py`

## 1. 结论

修复前，Stage-1/朴素 GRPO 与 Stage-2 使用了两种不同的 Claude Code 输入路径：

```text
Stage-1 / 朴素 GRPO：claude -p "原始任务"
Stage-2：            claude -p --input-format stream-json < resume_prefix.jsonl
```

两条命令虽然都能让 Claude Code 工作，但 Claude Code 对位置参数和 stream-json stdin 的内部组装方式不同。system reminder、原始任务和历史消息会进入不同的位置，最终送给模型的 Messages 请求和 `prompt_ids` 不一致。

因此，在判断 transcript resume 是否精确之前，必须先消除“启动入口不同”这个混杂变量。

## 2. 影响哪些路径

修复前，公共 `agent_runtime.run_claude()` 使用位置参数启动，调用方包括：

- 朴素 GRPO rollout；
- Hybrid Stage-1；
- SWE 评测 attempt；
- AGS smoke。

`agent_runtime.run_claude_with_prefix()` 使用 stream-json stdin，主要供 Hybrid Stage-2 使用。

如果只把 Hybrid Stage-1 改成 stream-json，而保留朴素 GRPO 的位置参数路径，会引入新的训练公平性问题。因此修复落在公共 `run_claude()`，让朴素 GRPO、Hybrid Stage-1、评测和 smoke 一起使用同一输入协议。

## 3. 修复前证据

2026-07-15 的严格请求抓取中：

```text
Stage-1 target request kind: append
Stage-2 first request kind:  new
```

Stage-1 原始消息顺序是：

```text
skills reminder
current-date reminder
原始任务
assistant reasoning + Read
真实 Read result
```

Stage-2 第一次请求变成：

```text
current-date reminder
历史 assistant reasoning + Read
missing Read result
skills reminder
原始任务
```

这不是 commit hash 导致的 token 对齐错觉。两边在 tokenization 之前的 raw Messages JSON 中，消息顺序就已经不同。

离线分量分析中，恢复 Stage-1 的消息顺序和 reasoning 呈现会增加 81 tokens。这个数字同时包含 stream-json 历史重建行为，不能全部归因于命令行 flag；但两种入口不同是当前实验中必须先消除的结构性变量。

## 4. 为什么两种命令不等价

位置参数路径：

```bash
claude -p "原始任务" ...
```

原始任务在 Claude Code 启动时作为 positional prompt 进入。Claude Code 在同一个 live session 内生成 reminder、执行工具，再通过 append 请求继续。

stream-json 路径：

```bash
claude -p --input-format stream-json ... < prefix.jsonl
```

原始任务和历史 conversation event 都来自 stdin。Claude Code 会创建一个新 session，再把 stdin 事件和自己生成的 reminder 重新组合成第一份 Messages 请求。

两条路径的输入语义相近，但不是同一个状态机入口；“文本都出现了”不能推出“消息顺序、chat template 和 token 完全相同”。

## 5. 已实施的修复

现在使用唯一的 Claude Code 输入协议：

```text
所有新任务和所有 Stage-2 resume 都使用 --input-format stream-json
```

新任务的原始 prompt 先规范化为单条 JSONL user event：

```json
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"原始任务"}]}}
```

然后与 Stage-2 共用同一个 runner：

```bash
claude -p --input-format stream-json ... < input.jsonl
```

具体实现：

1. 新增 `examples/claudecode_ags/claude_stream_input.py`，只使用 Python 标准库生成规范 user event 和紧凑 JSONL；
2. `run_claude()` 将原始任务写入 `/tmp/slime_cc_initial_prompt.jsonl`；
3. `run_claude()` 不再把 prompt 放入 shell command，而是复用 `run_claude_with_prefix()`；
4. 新任务和 Stage-2 最终都执行同一命令模板；
5. Stage-2 的 `build_transcript_prefix()` 使用同一个 event/JSONL helper；
6. Stage-1 初始输入与 Stage-2 `branch_step_t=-1` 的 prefix 由测试强制要求完全相同；
7. `run_claude()` 的调用方 API 保持不变，因此朴素 GRPO、Hybrid Stage-1、评测和 smoke 自动一起切换；
8. 环境变量传递规则保持不变，仍只传调用方明确提供的 `env`。

修复后的公共启动命令为：

```bash
/usr/local/bin/claude -p \
  --input-format stream-json \
  --permission-mode bypassPermissions \
  --output-format stream-json \
  --include-partial-messages \
  --include-hook-events \
  --verbose \
  < /tmp/slime_cc_initial_prompt.jsonl
```

Stage-2 使用相同命令，只把 stdin 文件换成截断后的 `cc_prefix.jsonl`。

## 6. 验证结果

### 6.1 修复前回归

先增加测试，要求 `run_claude()` 写出初始 JSONL 并使用 stream-json。修复前测试按预期失败：

```text
KeyError: /tmp/slime_cc_initial_prompt.jsonl
```

这证明旧代码仍走位置参数路径，没有偷偷满足新断言。

### 6.2 修复后代码级验证

测试确认：

- `run_claude()` 写出的 JSONL 是规范单条 user event；
- 启动命令包含 `--input-format stream-json`；
- 原始 prompt 不出现在 shell command；
- 启动命令只接收调用方给出的环境变量；
- Stage-1 初始 JSONL 与 Stage-2 `branch_step_t=-1` prefix 完全相同；
- Stage-2 prefix runner 的既有命令测试继续通过。

完整相关回归：

```text
53 collected
52 passed
1 deselected
```

被 deselect 的 `test_generate_wires_binary_reward_and_fan_out` 在本次修改前就与当前 worktree 不一致：测试引用 `generate.simple_cmd`，但当前 `generate.py` 已没有该属性。首次不排除运行时结果为 `31 passed, 1 failed`，失败发生在测试 setup、尚未调用 `run_claude()`，与本次启动路径修改无关。

另外通过：

- Ruff；
- Python 3.13 `py_compile`；
- `git diff --check`；
- 修改文件尾随空白检查。

### 6.3 真实 AGS smoke

使用 Hybrid 的 AGS tool `sdt-ltpatoxb` 做了两次真实尝试：

1. 完整两阶段 exact-resume 抓包；
2. 单沙箱、单 Read、60 秒 agent 预算的最小 stream-json 新任务。

两次都只完成了 AGS tool resolution，没有得到 sandbox ready。最小测试等待超过配置的 600 秒 startup timeout 后人工终止，退出码为 130；没有产生 Messages request、trajectory 或 `report.json`。单独检查旧 adapter 的 `/health` 返回 200。

因此当前只能得出：

- 代码级启动协议已经统一；
- 本地 runner/序列化回归通过；
- 真实 Claude Code 的单 user-event stream-json 行为尚未完成验收；
- 本次 AGS 失败不能归因成 token mismatch，也不能用于证明行为级修复成功。

## 7. 不在本修复范围内

统一入口只能消除位置参数与 stream-json 的启动差异。它不会自动解决：

- 历史 tool result 在 resume 时变成 `[Tool result missing due to internal error]`；
- Claude Code 对历史 conversation event 的重排；
- Stage-1 append request 与 Stage-2 new request 的内部状态差异；
- 跨日期变化的 `currentDate` reminder；
- 完整 token-exact resume。

因此修复后需要重新抓取 wire request。若工具边界后的 Stage-1/Stage-2 仍不一致，应继续归因到 history replay，而不能再归因到初始启动协议不同。

## 8. 修复记录

2026-07-15：

- 新增统一的 stream-json 输入序列化模块；
- 将公共 `run_claude()` 从 positional prompt 改为 JSONL stdin；
- 新任务与 Stage-2 共用同一个 command runner；
- transcript 的逻辑初始状态 `-1` 改为共用同一序列化 helper；
- 增加 prompt 不进入 shell、env 不扩散、初始 prefix 相同等回归断言；
- 完成 52 项相关本地回归；
- 尝试真实 Hybrid AGS smoke，但因 sandbox startup 未 ready 而未完成。

## 9. 后续真实验收

当 `sdt-ltpatoxb` 能正常创建 sandbox 后，应按顺序重跑：

1. 单 user-event stream-json 新任务，要求 exit code 0、至少一条 Messages request、trajectory 非空；
2. 相同 workspace 和 prompt 的两个新启动，逐字段比较第一次 wire request 和完整 `prompt_ids`；
3. Stage-1 第一个工具结果后的请求与 Stage-2 snapshot 0 请求，重新拆解 commit、启动入口、tool result 和消息重排差异。

只有第 1 项通过后，这个入口改动才适合用于新提交的训练任务；只有第 3 项完整数组相同，才能宣称工具边界上的 token-exact resume 已实现。
