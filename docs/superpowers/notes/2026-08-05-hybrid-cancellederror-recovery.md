# Hybrid 续训 `CancelledError` 中断修复记录

日期：2026-08-05  
分支：`feature/cc-ags-swe`  
关联 run：`qwen35_9b_cc_ags_hybrid_s0_fullcont_s23_swe_8gpu_readfix_v1`

## 1. 现象

Hybrid 续训不是在 actor train、logp 计算或 checkpoint 保存阶段中断，而是在后续 rollout 的
Stage-1 clean evaluator 阶段中断。

已确认：

| 项 | 结论 |
|---|---|
| step 15 | 已完成训练，日志中有 `train/step: 15` |
| 中断位置 | 后续 rollout 的 Stage-1 eval pipeline |
| 直接异常 | `TypeError: cannot unpack non-iterable CancelledError object` |
| 外部触发 | 多条 AGS `/execute` transient failure，多个 Stage-1 eval pipeline 接近 3600s guard |

日志中同类事件分布如下：

| 时间 UTC | 现象 | 结果 |
|---|---:|---|
| 2026-08-04 14:03 | `stage1:eval_pipeline` timeout，2 条 placeholder | 正常兜底，训练继续 |
| 2026-08-04 19:28 | 1 次 `asyncio.exceptions.CancelledError`，随后包装成 `HybridPipelineTimeoutError` | 正常兜底，训练继续 |
| 2026-08-05 05:07 | 20 次 `asyncio.exceptions.CancelledError`，2 次 `cannot unpack non-iterable CancelledError` | 整个 Ray job 失败 |

## 2. 根因

代码中使用：

```python
raw = await asyncio.gather(*trial_tasks, return_exceptions=True)
```

这表示单个 trial 抛出的异常会作为 `raw` 列表中的一个元素返回。后续代码只判断了普通
`Exception`：

```python
if isinstance(res, Exception):
    ...
bundle, samples, is_solved, turn_lps = res
```

但 `asyncio.CancelledError` 不属于普通 `Exception`，它继承自 `BaseException`。因此当
`CancelledError` 作为单个 trial 的返回值进入 `raw` 后，没有被异常分支接住，而是被当作正常
返回值解包，最终触发：

```text
TypeError: cannot unpack non-iterable CancelledError object
```

Stage-2 branch 也有同类隐患：

```python
branch_raw = await asyncio.gather(*branch_tasks, return_exceptions=True)

for res in branch_raw:
    if isinstance(res, Exception):
        ...
    for s in res:
        ...
```

如果某个 branch 返回 `CancelledError`，这里也会把它当作可迭代的 `list[Sample]` 使用。

## 3. 修复原则

修复的边界要清楚：

| 情况 | 应如何处理 |
|---|---|
| 整个 `hybrid_generate` coroutine 被外部取消 | 不吞掉，继续向外传播 |
| `gather(return_exceptions=True)` 返回的单个 Stage-1 trial `CancelledError` | 转成 Stage-1 aborted placeholder |
| `gather(return_exceptions=True)` 返回的单个 Stage-2 branch `CancelledError` | 丢弃该 branch，并记录 drop bucket |
| `StepTurnAlignmentError` | 仍然直接报错，不伪装成基础设施异常 |
| `KeyboardInterrupt` / `SystemExit` 等非 rollout 失败 | 不吞掉，继续向外传播 |

也就是说，不能简单地全局捕获所有 `BaseException`。只处理已经被 `gather(return_exceptions=True)`
收集到的单个 trial / branch 失败。

## 4. 具体修复方案

### 4.1 Stage-1：`CancelledError` 转成占位 episode

把 Stage-1 结果处理逻辑改成显式识别 `asyncio.CancelledError`：

```python
for i, res in enumerate(raw):
    if isinstance(res, StepTurnAlignmentError):
        raise res

    if isinstance(res, asyncio.CancelledError):
        reason = "vanilla_trial_exception:CancelledError"
        logger.warning("[hybrid] vanilla trial=%d cancelled; replaced by aborted placeholder", i)
        vanilla_samples.append(_aborted_vanilla_trial(..., reason=reason))
        hybrid_stats["hybrid_num_stage1_aborted_placeholders"] += 1
        hybrid_stats["hybrid_num_stage1_cancelled_placeholders"] += 1
        continue

    if isinstance(res, Exception):
        reason = _vanilla_exception_reason(res)
        logger.warning("[hybrid] vanilla trial=%d replaced by aborted placeholder: %s", i, res)
        vanilla_samples.append(_aborted_vanilla_trial(..., reason=reason))
        hybrid_stats["hybrid_num_stage1_aborted_placeholders"] += 1
        continue

    if isinstance(res, BaseException):
        raise res

    bundle, samples, is_solved, turn_lps = res
```

Stage-1 cancelled trial 的语义应与其他基础设施失败一致：

| 字段 | 值 |
|---|---|
| `reward` | `0` |
| `loss_mask` | 全 0 |
| `remove_sample` | `true` |
| episode 身份 | 保留原 `trial_idx` |
| resolved_rate 分母 | 仍计入 128 |

这样可以保持 Stage-1 与朴素 GRPO 的无效样本处理口径一致：失败 trial 不产生梯度，但不改变
同题 8 条 episode 的结构。

### 4.2 Stage-2：`CancelledError` 记为 branch drop

Stage-2 不需要占位进入训练；分叉失败只应减少可用 branch 数，不能打崩整步。

建议新增 bucket：

```python
def _branch_drop_bucket(error: BaseException) -> str:
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    ...
```

并把 Stage-2 loop 改成：

```python
for res in branch_raw:
    if isinstance(res, asyncio.CancelledError):
        bucket = _branch_drop_bucket(res)
        ...
        hybrid_stats["hybrid_num_dropped_cancelled"] += 1
        continue

    if isinstance(res, Exception):
        bucket = _branch_drop_bucket(res)
        ...
        continue

    if isinstance(res, BaseException):
        raise res

    for s in res:
        ...
```

Stage-2 branch 本来就是附加训练信号；单个 branch 因基础设施取消失败时，正确处理是丢弃该
branch，并通过 W&B 指标暴露数量。

### 4.3 指标补齐

建议补充低基数字段：

| 指标 | 含义 |
|---|---|
| `perf/step_grpo/n_stage1_cancelled_placeholders` | Stage-1 中由 `CancelledError` 转成占位的 episode 数 |
| `perf/step_grpo/n_dropped_cancelled` | Stage-2 中因 `CancelledError` 丢弃的 branch 数 |
| `perf/step_grpo/stage1_cancelled_placeholder_rate` | 上一项除以 128 |

这些指标用于区分两类问题：

- 模型行为导致 agent / eval 时间变长；
- AGS/SWE-ReX 链路抖动导致单条 trial 被取消。

## 5. 为什么不能只调大 timeout

调大 `guard_sec` 只能减少外层 timeout 的触发概率，不能修掉 `CancelledError` 解包 bug。

只要 `gather(return_exceptions=True)` 返回了单个 `CancelledError`，旧代码仍可能执行到：

```python
bundle, samples, is_solved, turn_lps = res
```

因此必须先修异常分类。之后是否调整 guard、评测轮询间隔、AGS 重试策略，是训练效率和基础设施
稳定性问题，不是这次 crash 的根修复。

## 6. 验收测试

至少补以下单元测试：

| 测试 | 预期 |
|---|---|
| Stage-1 某个 `vanilla_runner` 抛 `asyncio.CancelledError` | 返回 aborted placeholder，不抛 `TypeError` |
| Stage-2 某个 `branch_runner` 抛 `asyncio.CancelledError` | 该 branch 被 drop，整步继续 |
| `StepTurnAlignmentError` | 仍直接抛出 |
| 普通 `Exception` | 保持原逻辑：Stage-1 转 placeholder，Stage-2 drop branch |

线上验收：

1. 丢弃失败 run 中 rollout16 的部分产物，不把它当作有效 rollout。
2. 从最后一个已确认有效的训练状态续跑。
3. 观察下一个 rollout：
   - 不再出现 `cannot unpack non-iterable CancelledError object`；
   - `outcome/n_episodes` 仍为 128；
   - 若 AGS 仍抖动，应体现为 cancelled placeholder / dropped branch 指标，而不是 Ray job crash。

## 7. 实施状态

已实施。

| 文件 | 修改 |
|---|---|
| `examples/claudecode_ags/step_reconstruct/hybrid_generate.py` | Stage-1 显式处理 `asyncio.CancelledError`，转成 aborted placeholder；Stage-2 显式处理 `asyncio.CancelledError`，按 `cancelled` bucket drop branch；其他 `BaseException` 继续向外抛 |
| `examples/claudecode_ags/wandb_metrics.py` | 增加 `n_stage1_cancelled_placeholders`、`stage1_cancelled_placeholder_rate`、`n_dropped_cancelled` |
| `tests/claudecode_ags/test_step_reconstruct_hybrid_orchestration.py` | 增加 Stage-1 cancelled trial 和 Stage-2 cancelled branch 回归测试 |
| `tests/claudecode_ags/test_wandb_metrics.py` | 增加新指标映射测试 |

本地验证：

| 检查 | 结果 |
|---|---|
| `.venv/bin/python -m py_compile ...` | 通过 |
| `.venv/bin/python -m pytest ...` | 当前本地 `.venv` 缺 `torch`，测试无法收集；需要在训练镜像或完整依赖环境中跑 |
| 系统 `python3 -m pytest ...` | 系统 Python 为 3.9，不支持项目代码中的 `match`，不可作为有效测试环境 |
