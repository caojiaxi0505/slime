# Adapter 事件循环被同步 CPU 堵住（8 卡 vs 16 卡 agent 变慢）

日期：2026-07-15  
分支：`feature/cc-ags-swe`  
状态：**修复代码和本地回归已完成；尚未重提 Job，集群 A/B 待验证**  
关联代码：

- `slime/agent/adapters/anthropic_segmented.py`（`_handle_messages`）
- `slime/agent/adapters/common.py`（`call_sglang_generate`、`_run_turn`）
- `slime/agent/aiohttp_threaded.py`（adapter 单线程 event loop）
- `examples/claudecode_ags/generate.py`（`_AdapterService` / `run_app_in_thread`）
- `examples/claudecode_ags/step_reconstruct/live_runners.py`（Hybrid checkpoint 导出与恢复）
- `examples/claudecode_ags/wandb_metrics.py`（adapter 分阶段耗时）

对比 run（step 11）：

| | 8GPU | 16GPU |
|---|---|---|
| 实验 | `qwen35_9b_cc_ags_1node_grpo_debug` | `qwen35_9b_cc_ags_2node_grpo_c64_t45` |
| dump | `rollout_dumps/rollout_11.pt` | 同左 |
| 拓扑 | 1 node / 8 engines | 2 node / 16 engines |
| 并发门闩 | `SLIME_CC_AGENT_CONCURRENCY`≈64 | 同左（显式 64） |
| agent 预算 | ~1800s（撞顶 max≈1845） | 45min（max≈3265） |

---

## 1. 结论（先读这段）

**现象：**同样一批 instance、生成 token 数差不多时，16 卡上单条 agent 墙钟明显更长；有效 tok/agent·s 下降；wipe / `exit=1` 随之升高。

**不是：**

- eval / StartSandbox 变慢；
- SGLang `#queue-req` 堆满（两边几乎都是 0）；
- wipe 分类算错（wipe 段 prompt 中位 ~77k–86k，像真 autocompact）；
- 「卡多 → 单条 agent 必然更快」（agent 环是串行的，多 engine 只抬 LLM 并行上限）。

**代码层面的根因：**

所有 Claude 共用 **一个** `SegmentedAnthropicAdapter` 进程内的 **单线程 aiohttp event loop**；热路径上把 `apply_chat_template` / 全量 message hash / translate / decode+parse 做成了 **同步 CPU**。任一 session 在 tokenize，其它 session 的 HTTP/`await` 回调都会被堵住（head-of-line blocking）。上下文越长、同时活着的大上下文 session 越多，全员越慢——16 卡长预算更容易落入这个正反馈。

上述同步工作现已移到有界 CPU 线程池；同一 session 仍保持严格串行，token、segment 和 token-exact resume 语义不变。修复还没有进入新的集群 Job，因此“它能解释多少 8/16 卡墙钟差异”仍需用新 run 做 A/B，不能只凭本地测试下结论。

---

## 2. 现象拆解（step 11）

### 2.1 墙钟账本

| 指标 | 8GPU | 16GPU | 说明 |
|---|---:|---:|---|
| mean `agent_elapsed_sec` | 550 | 970 | +76% |
| mean `eval_elapsed_sec` | 25 | 23 | 不是瓶颈 |
| AGS StartSandbox Step1+2 | ~10s | ~5s | 16 卡更快 |
| mean episode response tokens | 17.8k | 17.5k | 不是「多生成了」 |
| wipe 段 / ep | 1.18 | 2.57 | 后果，见 §3 |
| `exit≠0`（合计） | 19.5% | 47.7% | = `exit=1` + `exit=-1`；W&B `agent_exit_nonzero_rate` 只报这个合计 |
| **`exit=1`** | **15.6%**（20/128） | **35.4%**（45/127） | Claude 进程自己非 0 退出（非 harness 超时） |
| **`exit=-1`** | **3.9%**（5/128） | **12.6%**（16/127） | `EXIT_TIME_BUDGET_EXCEEDED`：预算内未写出 done marker |
| `exit=0` | 80.5% | 52.0% | 正常结束 |

`agent_exit_code` 语义（`slime/agent/sandbox.py` / `run_claude`）：

| 码 | 含义 |
|---|---|
| `0` | Claude CLI 正常 exit |
| `1`（或其它 `>0`） | Claude 进程 `$?` 非 0（API/内部错误等）；step11 dump 里非 0 正码实际只有 `1` |
| `-1` | harness 轮询超时，**不是** Claude 自己返回的码 |

step11 分码墙钟（有助于区分「早死」vs「拖满预算」）：

| | 8GPU mean agent | 16GPU mean agent |
|---|---:|---:|
| `exit=0` | 472s | 824s |
| `exit=1` | 628s | 484s（中位仅 ~309s，偏「中途挂掉」） |
| `exit=-1` | 1841s（贴 1800s 预算顶） | 2940s（贴 45min 预算顶） |

要点：16 卡 `exit≠0` 变多，**主因是 `exit=1`（15.6%→35.4%）**，不是只靠更长预算堆出来的 `-1`（3.9%→12.6%）。`-1` 升高与 45min 预算把长尾留住有关；`exit=1` 升高与更慢的 turn / wipe 风暴更一致。

从 SGLang Prefill/Decode 日志粗估（`new_token/throughput` 等，/128 ep）：

| 子项 | 8GPU | 16GPU | Δ |
|---|---:|---:|---:|
| Prefill | ~97s | ~260s | +163s |
| Decode | ~52s | ~141s | +89s |
| 残差（tool / I/O / adapter 排队等） | ~401s | ~569s | +168s |

LLM 侧变贵与「更长上下文 + 更多 cold prefill（wipe）」一致；残差里应包含 **adapter 侧排队**，但现有 dump **没有** 分项计时，无法把残差再拆成「纯沙箱 tool」vs「等 adapter」。

### 2.2 同任务、成功轨迹：行为量相近，墙钟仍慢

只看 `exit=0`、按 instance 配对：

| instance | 8→16 时间 | resp | wipe |
|---|---|---|---|
| `dask__dask-9528` | 218→686s（×3.2） | 26k→24k | 0.1→0.7 |
| `dask__dask-9212` | 344→554s（×1.6） | 27k→32k | **0→0** |

`9212` 说明：**不靠多 wipe / 多想，墙钟也会慢** → 优先怀疑「每 turn 环境更慢」，而不是策略突然变了。

exit=0 有效 tok/s 中位：54 → 33。

### 2.3 为何「卡多」推不动单条

- 单条轨迹：`tool → 等 → /v1/messages → tool → …` 串行。
- Prefill/Decode `#queue-req≈0`，`running-req` 中位≈1 → 8 卡时 GPU 也经常在等 agent。
- 再加 8 个 engine **缩短不了** tool 等待，也 **消除不了** 单 adapter loop 上的 CPU 串行。

---

## 3. wipe 变多：机制结果，不是分类 bug

wipe 定义（设计如此）：新请求的 `messages` **不是** 旧 chain 前缀续写 → `select_chain` 冻一段 `segment_kind=wipe`（Claude Code autocompact 典型形态）。

证据：

- wipe 段 prompt 中位 77k / 86k，大量 >90k；`<20k` 的可疑早 wipe 极少（8 卡 0，16 卡 5）。
- autocompact 窗口：`CLAUDE_CODE_AUTO_COMPACT_WINDOW=101000`（`env/claude_code.env`）。
- mean wipe 1.18→2.57 中，约 2/3 可由 `exit=1` 变多解释（两边 `exit=1` 均为 100% 有 wipe、mean≈4）；其余与更长成功轨迹叠在一起。

另有一处 **真实但次要** 的误标风险：subagent 请求若省略 `system`，`_request_system_hash` 用 main 的 hash 去比 sub chain，第二轮可能假 wipe。step11 几乎无 subagent 段，**解释不了** 本次 wipe 暴涨。

**解读：** wipe↑ 主要是「跑得更久 / 失败 compact 风暴」的下游，不是 16 卡把 wipe 计数算坏了。

---

## 4. 修复前根因：async handler 里的同步 CPU

### 4.1 拓扑

```text
最多 64 个 Claude（沙箱内）
        │  HTTP /v1/messages
        ▼
RolloutManager 内唯一 SegmentedAnthropicAdapter
        │  run_app_in_thread → 单线程 asyncio loop
        ▼
SGLang router / 多 engine（这里经常不排队）
```

`examples/claudecode_ags/generate.py` 的 `_AdapterService` 对整个 rollout 单例起一个 adapter。

### 4.2 修复前热路径

`anthropic_segmented._handle_messages` 大意：

```python
async with session.lock:
    select_chain(...)           # 含全量 message_hash
    commit_request(...)         # wipe/new 时整段 _translate_messages
    prompt_ids = _render_prompt_ids(...)  # apply_chat_template（重）
    turn = await call_sglang_generate(...)  # 这里才真正让出
    _parse_and_blocks(...)      # decode + parse_model_output（同步）
```

已确认：

- `call_sglang_generate` **是** async（`aiohttp`），**没有**误写成 `requests` 同步 HTTP。
- 同 `sid` 的 `session.lock` 包住整个 generate：按设计串行化同一会话；**跨 sid 不互斥**。
- 跨 sid 互堵来自：**同步 CPU 占满唯一 event loop**，其它 coroutine 的 `await` 回调无法推进。

### 4.3 为何更伤 16 卡

同一缺陷 × 更「胖」的并发：

1. 两边都是 **1 loop × concurrency≈64**。
2. 16 卡 agent 更长、wipe/大 prompt 更多 → 每次 `apply_chat_template` / hash 更贵。
3. 45min 预算把更多大上下文 session 同时留在 64 槽 → loop 上串行 CPU 排队更长。
4. 每 turn 变慢 → 会话更长 → 更容易顶 autocompact → wipe / `exit=1`↑ → mean 墙钟再被拉高（正反馈）。

这与「SGLang 不排队，但 agent 有效 tok/s 掉」相容：瓶颈可以在 **进 GPU 前后的 adapter 侧**。

### 4.4 修复前的次要问题

- `call_sglang_generate` **每次** `aiohttp.ClientSession(...)`：连接抖动，但是 async。
- 无 `asyncio.to_thread` / `run_in_executor` 包裹 tokenize/parse（全 `slime/agent` 下未找到）。

### 4.5 仍待集群证明的部分

旧 run **没有** adapter 分阶段耗时，因此不能把残差 +168s 定量归因于 event-loop 阻塞。新代码已经增加埋点，但只有新 Job 才会产生这些字段。当前可以确认“同步 CPU 会堵住单线程 loop”这一代码问题；不能确认它占旧 run 变慢的百分之多少。

---

## 5. 已实施的修复

### 5.1 有界 CPU 线程池

`SegmentedAnthropicAdapter` 现在使用专用、有界线程池。默认 worker 数是 `min(4, CPU 核数)`，可用 `SLIME_ADAPTER_CPU_WORKERS` 覆盖；小于 1 会直接报错。

以下工作不再占用 aiohttp event loop：

- 请求 JSON 解析和 system folding；
- 完整 message hash；
- Anthropic message 翻译、checkpoint 深拷贝与哈希；
- `apply_chat_template` 和 tokenize；
- model output decode 与 tool/reasoning parse；
- Hybrid checkpoint 序列化；
- session 结束时的 segment merge。

线程只计算“准备好的结果”，不直接提交会话状态。计算完成后，event loop 在持有 `session.lock` 时一次性更新 chain/checkpoint。若 HTTP 请求在 CPU 工作期间被取消，handler 会等该工作结束再释放锁，避免后台线程继续读取已经变化的 session。

同一 session 的 SGLang 请求仍由原来的锁严格串行；不同 session 不共享这个锁。这样没有改变 turn 顺序，也没有改变训练 token。

### 5.2 去掉重复哈希，保护 token-exact

修复前，一轮请求会在 `select_chain` 和 `commit_request` 中重复计算完整 message hash。热路径现在只计算一次，并把结果同时用于路由和提交。

Hybrid 另有两处同步重工作也已处理：

- Stage-2 `open_session` 的 checkpoint 重渲染改为 `open_session_async`，不会堵住 rollout 主循环；
- Stage-1 checkpoint 导出改为异步序列化。

首个 resumed request 仍直接使用 checkpoint 保存的完整 `prompt_ids`。线程池只改变“在哪个线程计算”，不改变输入数组、chat template、tool schema 或 sampling config。

### 5.3 复用 SGLang HTTP 连接

`call_sglang_generate` 不再每轮创建 `aiohttp.ClientSession`。每个 adapter event loop 复用一个 keep-alive client，并在 app cleanup 时关闭；abort request 也复用同一个 client，但使用独立的 5 秒超时。

### 5.4 新增耗时字段

每个成功或失败的 adapter 请求都会记录分阶段耗时。session 结束时写入 Sample metadata 的均值和最大值：

| 字段前缀 | 含义 |
|---|---|
| `adapter_loop_delay_ms` | handler 进入后一次 event-loop yield 的调度延迟；是近似值 |
| `adapter_session_lock_wait_ms` | 等待同一 session 锁的时间 |
| `adapter_json_ms` | JSON 解析与请求预处理 |
| `adapter_hash_ms` | 完整消息历史哈希；resume 首轮可为 0 |
| `adapter_prepare_tokenize_ms` | message 翻译、checkpoint 构造、chat template 和 tokenize 的合计 |
| `adapter_parse_ms` | decode 与 model output parse |
| `adapter_cpu_queue_ms` | 上述 CPU 工作在线程池中等待 worker 的合计 |
| `adapter_cpu_ms` | 上述 CPU 工作实际执行时间的合计 |
| `adapter_sglang_e2e_ms` | `/generate` HTTP 往返时间 |
| `adapter_total_ms` | adapter 整轮总时间 |

metadata 还包含 `adapter_turn_count`、`adapter_failed_request_count` 和 `adapter_cpu_workers`。W&B 路径为：

- Stage-1 / 朴素 GRPO：`perf/adapter/*`
- Stage-2：`perf/stage-2/adapter/*`

compact fan-out 后的多个 segment 共用同一组 session 指标；W&B 会按 episode 去重，不会重复计数。

### 5.5 明确没有改的内容

- 没有改 wipe / autocompact 判定；
- 没有改 Claude Code prompt、SGLang sampling 参数或 token-exact checkpoint；
- 没有改 agent 45 分钟预算和并发 64；
- 没有重提、重启或删除任何集群 Job。

---

## 6. 验证状态与下一步

### 6.1 已完成的本地验证

- 并发测试：一个 tokenizer worker 被故意阻塞时，adapter health 和另一个 session 仍能响应；
- 取消测试：CPU 工作结束前不会释放相关 session 状态；
- hash 测试：一轮请求中的每条 message 只 hash 一次；
- lifecycle 测试：SGLang HTTP client 跨 turn 复用，并在 app cleanup 后关闭；
- token-exact、parallel tool result、失败 tool result、第三轮 resume 等既有测试继续通过；
- 使用训练所用 Qwen3.5-9B tokenizer，对 32,295-token prompt 并发渲染 12 次，完整 token 数组与 SHA-256 全部一致；
- 完整 `tests/claudecode_ags`：194 项中 191 通过、2 失败、1 个收集错误；剩余问题仍是既有的 generate/SWE eval 测试，与本修复无关。

### 6.2 尚未完成的集群验证

1. 用新代码跑小规模 concurrency=64 smoke，先看 `perf/adapter/cpu_queue_ms/*`、`prepare_tokenize_ms/*` 和 `sglang_e2e_ms/*`。
2. 同数据、同 RBS/n_samples/concurrency，1node-8engine 与 2node-16engine 各跑若干 step。
3. 对 exit=0 的同 instance 比较 `agent_elapsed`；重点复查 `dask__dask-9212` 这类 token 相近且 wipe=0 的任务。
4. 观察 wipe、`exit=1`、`exit=-1` 是否随 adapter 排队下降而回落。若没有，再继续拆沙箱/tool、Claude Code retry 和 SGLang prefill。

---

## 7. 相关笔记

- 8/16 卡公平对比与 hybrid 坑总表：`2026-07-15-hybrid-step-grpo-pitfalls-xiaobai.md`
- wipe / compact 与训练 identity：`anthropic_segmented.py` 顶部模块注释；`test_segment_select.py`
- 并发与超时字段：`2026-07-15-hybrid-run-bmr3zlex-bug-report.md`（`agent_queue_wait_sec` 等；那是 **agent 并发门闩**，与本文 **adapter loop** 不是同一层）
