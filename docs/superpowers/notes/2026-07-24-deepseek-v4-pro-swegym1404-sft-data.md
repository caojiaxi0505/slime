# DeepSeek-v4-pro SWE-Gym SFT 数据说明

日期：2026-07-24  
分支：`feature/cc-ags-swe`  
用途：记录 DeepSeek-v4-pro 在 SWE-Gym 1404 题上生产 SFT 数据的过程、落盘结果和最终筛选口径。

## 1. 结论

本次任务覆盖 SWE-Gym gold-filter 后的 1404 个 task。Pod 最终 `Succeeded`，并为 1404 个 task
都写入了结果文件；但最后阶段出现上游 401，导致 70 条 trial 只有结果记录，没有真实模型交互。

按当前 clean SFT 口径：

| 数据口径 | trial 数 | resolved 数 | 说明 |
|---|---:|---:|---|
| 原始落盘 | 1404 | 484 | `results.jsonl` 和 `sft_trials.jsonl` 均为 1404 行 |
| 排除 401 空上下文和 502 | 1333 | 484 | 去掉明确基础设施坏样本 |
| 最终 clean all-trials | 1169 | 481 | 再去掉 patch / diff / 消息结构异常 |
| 最终 clean resolved-only | 481 | 481 | 只保留 clean 且 `resolved=True` 的 trial |

因此，若训练 all-trials SFT，建议使用 **1169 条**。若只训练 solved SFT，建议使用 **481 条**。

## 2. 基本概念

| 概念 | 含义 |
|---|---|
| task | 一道 SWE-Gym 修复题，包含问题描述、原始代码、测试和 gold patch 元数据 |
| trial | 模型对一个 task 的一次完整尝试：读题、调用工具、修改代码、生成 patch、运行评测 |
| SFT | Supervised Fine-Tuning，监督微调；这里指用模型真实交互轨迹训练模型复现这些行为 |
| trajectory | agent 执行过程中的完整事件流，例如用户输入、模型回复、工具调用和工具结果 |
| messages | 转成训练格式后的对话上下文，是 SFT 直接依赖的核心内容 |
| turn | 一次模型请求/回复记录；`turn_count` 表示该 trial 中模型交互轮数 |
| diff / patch | 模型最终对代码的修改；`diff_chars` 表示 diff 文本长度 |
| resolved | SWE 评测是否通过；`resolved=True` 表示该 trial 解决了题目 |
| clean | 本文定义的可训练样本：有真实上下文，基础设施未失败，patch 和消息结构没有明显异常 |

## 3. 数据生产设置

| 项 | 值 |
|---|---|
| 数据源 | SWE-Gym gold-filter 后 1404 个 task |
| 模型 | `maas/deepseek-v4-pro` |
| API 形式 | OpenAI-compatible endpoint，API key 不写入本文档 |
| 上下文长度 | 128k |
| reasoning effort | `max` |
| 采样 | `temperature=1`，`top_p=0.95`，`top_k=20` |
| 并发 | 16 |
| agent 时间预算 | 30min |
| eval 时间预算 | 10min |
| AGS tool | `sdt-db5nvd67` |
| 输出目录 | `/mnt/sn-007/jiaxicao/checkpoints/cc-ags/deepseek_v4_pro_swegym1404_sft_20260723_c16` |

每个 task 跑一次 DeepSeek-v4-pro agent。agent 完成后保存三类信息：

1. `results.jsonl`：轻量结果，包括 `ok`、`resolved`、退出码、patch 路径、耗时等。
2. `sft_trials.jsonl`：trial 级 SFT 数据，包括 `messages`、`turns`、评测摘要和 patch 元数据。
3. `diffs/` 与 `sft_turns/`：patch 文件和 per-turn 请求/响应记录。

注意：当前目录中的 `slime_sft_all.jsonl` 只有 25 行，是早期旧逻辑留下的临时文件，不代表最终
all-trials 数据。最终 all-trials 数据应从 `sft_trials.jsonl` 重新筛选生成。

## 4. 原始落盘结果

| 文件 | 行数 | 说明 |
|---|---:|---|
| `results.jsonl` | 1404 | 每个 task 一条结果 |
| `sft_trials.jsonl` | 1404 | 每个 task 一条 trial 级 SFT 记录 |
| `slime_sft_resolved.jsonl` | 484 | 已生成的 resolved-only 文件；尚未应用本文最终 clean 过滤 |
| `slime_sft_all.jsonl` | 25 | 旧文件，不应作为最终 all-trials 数据使用 |

原始结果：

| 指标 | 数量 |
|---|---:|
| 总 trial | 1404 |
| `ok=True` | 1403 |
| `ok=False` | 1 |
| `resolved=True` | 484 |
| `resolved=False` | 920 |

## 5. 基础设施问题

### 5.1 上游 401

401 表示上游模型服务返回鉴权错误，本次日志中的错误码为 `401002`，含义是 API Key 不存在或
签名校验失败。这里不判断根因，只记录现象和落盘影响。

| 项 | 数量 / 时间 |
|---|---:|
| 401 warning 次数 | 917 |
| 首次出现 | 2026-07-24 07:04:40 UTC |
| 最后出现 | 2026-07-24 07:37:57 UTC |
| 对应空上下文 trial | 70 |

这 70 条 trial 在结果层被写成 `ok=True`，但 SFT 内容为空：

| 字段 | 值 |
|---|---|
| `turn_count` | 0 |
| `messages` | `[]` |
| `turns` | `[]` |
| `diff_chars` | 0 |
| `resolved` | 全部 `False` |

因此这些记录不能用于 SFT。它们主要分布在最后一批任务：

| repo | 空上下文 trial 数 |
|---|---:|
| `Project-MONAI` | 18 |
| `modin-project` | 52 |

### 5.2 AGS 502

502 表示 AGS sandbox `/execute` 请求返回网关错误。本次只有 1 条：

| index | instance | 问题 |
|---:|---|---|
| 591 | `iterative__dvc-1700` | AGS `/execute` 返回 502，`ok=False` |

该条没有有效 diff 和 trajectory 路径，应过滤。

排除 401 空上下文和 502 后，剩余：

| 口径 | trial 数 | resolved 数 |
|---|---:|---:|
| 基础设施 clean | 1333 | 484 |

## 6. SFT 数据筛选管线

最终筛选从 `sft_trials.jsonl` 开始，按以下顺序过滤。下表的数量是**按优先级归因后**的数量；
独立命中数有少量重叠，实际过滤结果按并集删除。

| 规则 | 数量 | resolved 数 | 解释 |
|---|---:|---:|---|
| 401 空上下文 | 70 | 0 | 没有真实 `messages` / `turns`，不能训练 |
| 502 / `ok=False` | 1 | 0 | 基础设施失败，结果不完整 |
| `agent_exit_code=137` | 1 | 0 | 137 = 128 + SIGKILL(9)，表示进程被外部强杀，常见于 OOM 或资源硬限制 |
| `applied_cleanly=False` | 94 | 0 | 模型生成了 patch，但 patch 不能干净应用到原始代码树 |
| `diff_chars=0` 且有轨迹 | 65 | 0 | agent 有交互，但最终没有产生代码修改 |
| `diff_chars > 200k` | 2 | 2 | diff 过大，容易包含大段无关文件或异常输出 |
| assistant 空消息异常 | 2 | 1 | assistant 消息既没有文本，也没有 tool call |
| **合计过滤** | **235** | **3** | 各规则有少量重叠，合计为并集 |

过滤后：

| 口径 | trial 数 | resolved 数 |
|---|---:|---:|
| 最终 clean all-trials | 1169 | 481 |
| 最终 clean resolved-only | 481 | 481 |

### 6.1 为什么过滤 `agent_exit_code=137`

`agent_exit_code=137` 通常表示进程被 `SIGKILL` 强制杀掉：

```text
137 = 128 + 9
9 = SIGKILL
```

这不是模型自然完成，也不是正常超时退出。它可能来自 OOM、sandbox 资源限制或外部强制终止。
本次只有 1 条，且未 resolved，建议过滤。

### 6.2 为什么过滤 `applied_cleanly=False`

`applied_cleanly=False` 表示模型 patch 无法干净应用到原始 repo。常见原因包括：

- diff 上下文和原文件不匹配；
- patch 反向或重复应用；
- 修改了不存在的路径；
- diff 格式损坏；
- patch hunk 冲突。

这类样本不是“代码改错后测试失败”，而是 patch 应用阶段已经失败。本次 94 条全是
`resolved=False`，建议过滤。

### 6.3 为什么过滤 `diff_chars=0`

`diff_chars=0` 表示最终没有代码修改。若同时有真实轨迹，说明 agent 确实交互过，但没有产出
patch。本次这类 65 条全部未 resolved。对 SFT 来说，它们更像失败行为样本，不适合放入 clean
训练集。

### 6.4 为什么过滤超大 diff

`diff_chars > 200k` 用于拦截异常大的 patch。过大的 diff 容易包含无关文件、大规模格式化、
日志或生成内容污染。本次按优先级归因命中 2 条，均 resolved：

| index | instance | diff_chars |
|---:|---|---:|
| 818 | `dask__dask-8501` | 226,611 |
| 885 | `pandas-dev__pandas-49284` | 22,519,740 |

其中 `885` 的 22MB diff 明显异常。`818` 可人工复核；若确认 diff 合理，可以从过滤名单中放回。

### 6.5 为什么过滤 assistant 空消息异常

正常 assistant 消息可以是文本回复，也可以是 tool call。本文只过滤同时满足以下条件的消息：

```text
无文本 && 无 tool call
```

本次按优先级归因命中 2 条；另有 1 条与 `applied_cleanly=False` 重叠：

| index | instance | resolved | 说明 |
|---:|---|---:|---|
| 140 | `getmoto__moto-5420` | True | 最后一条 assistant 为空，但前面轨迹完整、patch clean、resolved=True |
| 531 | `conan-io__conan-11348` | False | assistant 空消息异常 |
| 299 | `python__mypy-14981` | False | 同时 `applied_cleanly=False` |

`140` 更像末尾 logger / 模型空回复瑕疵，可人工复核后放回。

## 7. 不默认过滤的情况

以下情况会记录，但不默认过滤：

| 情况 | 数量 | resolved 数 | 原因 |
|---|---:|---:|---|
| `agent_exit_code=-1` timeout | 331 | 108 | 有真实轨迹，其中不少 solved；可用于学习长程修复 |
| 非空 `agent_exit_code=1` | 46 | 7 | 有真实轨迹；对 all-trials 可保留，resolved-only 需看具体质量 |
| `turn_count > 100` | 45 | 10 | 表示交互很长，不一定是坏数据 |

`agent_exit_code=-1` 表示 agent 达到时间预算后被停止。本次有 108 条 timeout 仍然 solved，因此
不能简单按退出码删除。

`agent_exit_code=1` 表示 agent 进程非正常退出，但不是 SIGKILL。若该 trial 有完整消息、patch
和评测结果，本文不把它视为基础设施坏样本。

## 8. 推荐训练文件

建议从 `sft_trials.jsonl` 重新生成两个最终文件：

| 文件 | 内容 | 预期行数 |
|---|---|---:|
| `slime_sft_all.clean.jsonl` | 应用本文全部 clean 过滤后的 all-trials SFT | 1169 |
| `slime_sft_resolved.clean.jsonl` | 在 clean 样本中只保留 `resolved=True` | 481 |

若人工复核后决定放回 `140 getmoto__moto-5420` 或 `818 dask__dask-8501`，需要在文件名或伴随
说明中记录放回规则，避免后续复现实验时口径混乱。

## 9. 当前状态

| 项 | 状态 |
|---|---|
| 数据生产 pod | `Succeeded` |
| 原始 trial | 已落盘 1404 条 |
| resolved-only 旧文件 | 已有 484 条，但尚未应用本文最终 clean 过滤 |
| all-trials 旧文件 | 只有 25 条，不可作为最终数据 |
| 推荐下一步 | 基于 `sft_trials.jsonl` 生成 clean all / clean resolved 两个新文件 |
