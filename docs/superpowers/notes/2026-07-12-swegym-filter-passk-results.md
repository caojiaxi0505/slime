# SWE-Gym 训练集筛选结果（Gold ×4 → Pass@k ×8）

日期：2026-07-11 → 2026-07-12  
状态：**已完成**  
设计：[2026-07-11-swegym-gold-filter-passk-design.md](../specs/2026-07-11-swegym-gold-filter-passk-design.md)  
操作：[2026-07-11-swegym-gold-filter-passk-ops.md](../runbooks/2026-07-11-swegym-gold-filter-passk-ops.md)

## 1. 一句话结论

从 SWE-Gym train **2438** 题，经 gold 稳定性筛选 → **1404**，再经 Claude Code ×8 剔「永远能解」→ **1394**（resolved 0–7）。在此基础上按训练算法再拆：

| 算法 | resolved 范围 | 题数 |
|------|---------------|-----:|
| **GRPO** | 1–7 | **338** |
| **step-GRPO** | 0–7 | **1394** |

```text
SWE-Gym train 2438
    │  Phase G：gold patch ×4，必须 4/4 resolved
    ▼
kept 1404
    │  Phase P：Claude Code ×8（Qwen3.5-9B），只剔 8/8 全对
    ▼
train_candidates 1394（0–7）
    ├─ GRPO:       去掉 0/8 → 338（1–7）  train_grpo_resolved_1_7.jsonl
    └─ step-GRPO:  保留 0–7 → 1394        train_step_grpo_resolved_0_7.jsonl
excluded_passk 10（8/8 always_resolved）
```

## 2. 怎么筛的

### 2.1 Phase G — gold ×4（稳定性）

| 项 | 说明 |
|----|------|
| 输入 | SWE-Gym train parquet，2438 题 |
| 动作 | 每题用 **gold patch** 在 AGS 沙箱评测 **4 次** |
| 保留规则 | **4/4 全部 `resolved=True`** 才进 `kept.jsonl` |
| 排除规则 | 任一次失败（含 unresolved / timeout / 缺镜像 / infra）→ excluded |
| 并发 | 16（task-level） |
| eval 超时 | 600s |

**意图：** 去掉 gold 本身就不稳、或环境/镜像有问题的题，留下「标准答案可复现通过」的池子。

### 2.2 Phase P — Claude Code ×8（剔全对）

| 项 | 说明 |
|----|------|
| 输入 | Phase G 的 `kept.jsonl`（1404） |
| 动作 | 每题跑 **8 次** 真实 agent（Claude Code → adapter → SGLang）再评测 |
| 模型 / 推理 | `Qwen3.5-9B`；部署 `NUM_GPUS=8 TP_SIZE=1 --dp-size 8` |
| 保留规则 | resolved 次数为 **0–7** → `train_candidates.jsonl` |
| 排除规则 | **仅** 8/8 全 `resolved=True`（always_resolved）→ `excluded_passk.jsonl` |
| agent 预算 | `--time-budget 1800`（触顶记 `timeout_hit`，`claude_exit=-1`） |
| eval 超时 | 600s |
| 并发 | 先 16，后提到 **128** 续跑 |

**意图：** 去掉当前策略下「几乎必过」的题，留下仍有失败空间、适合 GRPO 的题。  
**注意：** pass@k 分布只做报告；筛选决策不看 pass@k 数值，只看是否 8/8。

## 3. 数量变化（筛选前后）

| 阶段 | 输入 | 输出 | 排除 |
|------|-----:|-----:|-----:|
| 原始 SWE-Gym train | — | 2438 | — |
| Phase G（gold ×4） | 2438 | **1404** kept | 1034 |
| Phase P（CC ×8） | 1404 | **1394**（resolved 0–7） | 10（8/8） |
| → GRPO 子集 | 1394 | **338**（resolved 1–7） | 1056（0/8） |
| → step-GRPO | 1394 | **1394**（resolved 0–7） | 0（相对 1394） |

进训文件：

```text
# GRPO
/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.jsonl

# step-GRPO
/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_step_grpo_resolved_0_7.jsonl
```

## 4. Phase G 细节

**Out dir：** `/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_gold_20260711_042217/`

| 指标 | 数量 |
|------|-----:|
| 总题数 | 2438 |
| Kept（4/4 gold resolved） | 1404 |
| Excluded | 1034 |
| attempts（`runs.jsonl`） | 6661 |

### 4.1 Phase G 排除原因

| 原因 | 数量 | 说明 |
|------|-----:|------|
| `gold_unresolved` | 952 | gold 评测未过 |
| `infra`（含 timeout / 缺镜像 / BrokenPipe 等） | 82 | 其中约 44 timeout、37 缺镜像、1 BrokenPipe（早期笔记细分） |
| AGS 配额类错误 | 0 | 未见 |

更细的 Phase G 笔记见：[2026-07-11-swegym-gold-filter-results.md](./2026-07-11-swegym-gold-filter-results.md)（其中「暂缓 Phase P」已被后续实跑覆盖）。

## 5. Phase P 细节

**Out dir：** `/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/`

| 指标 | 数量 |
|------|-----:|
| 输入题数 | 1404 |
| attempts（1404 × 8） | 11232 |
| 进训候选 | **1394** |
| 剔除 always_resolved（8/8） | **10** |
| `infra_ok=false` 的 attempt | 9（~0.08%，TimeoutError×7 + 502×2） |

### 5.1 resolved 次数分布（1404 题，含已剔除的 10 题）

| resolved / 8 | 题数 | 占 1404 | 去向 |
|-------------:|-----:|--------:|------|
| 0 | 1056 | 75.2% | 进训 |
| 1 | 121 | 8.6% | 进训 |
| 2 | 74 | 5.3% | 进训 |
| 3 | 37 | 2.6% | 进训 |
| 4 | 28 | 2.0% | 进训 |
| 5 | 31 | 2.2% | 进训 |
| 6 | 31 | 2.2% | 进训 |
| 7 | 16 | 1.1% | 进训 |
| **8** | **10** | **0.7%** | **剔除** |

对 **1394** 进训题：约 **75.8%** 为 0/8（当前模型一轮也解不出），其余为 1–7/8 的部分可解。

### 5.2 进训候选的 repo 分布（1394）

| repo（`instance_id` 前缀） | 题数 |
|---------------------------|-----:|
| Project-MONAI | 322 |
| python（多为 mypy） | 253 |
| getmoto | 249 |
| iterative（dvc） | 170 |
| dask | 132 |
| pandas-dev | 103 |
| conan-io | 62 |
| modin-project | 52 |
| facebookresearch | 31 |
| bokeh | 20 |
| **合计** | **1394**（10 个 repo） |

### 5.3 Timeout / agent 触顶

口径：`timeout_hit=True`（与 `claude_exit=-1` 一致），**包含 agent 1800s 时间预算触顶**，不是只算稀少的 `TimeoutError` infra。

| 范围 | 数量 |
|------|-----:|
| attempt 级 timeout | **479 / 11232**（4.26%） |
| 至少 1 次 timeout 的题 | **327 / 1404** |
| 进训 1394 中至少 1 次 timeout | 325 |

单题 timeout 次数（题级）：1×209，2×95，3×14，4×7，5×2。

**全错（0/8）且 8 次全 timeout：** **0 题**。  
0/8 题内 timeout 次数：0×834，1×147，2×61，3×8，4×5，5×1。  
多数全错是「跑完了但没解出来」，不是「次次触顶」。

## 6. 产物路径

### Phase G

```text
/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_gold_20260711_042217/
├── summary.json
├── kept.jsonl          # 1404
├── excluded.jsonl      # 1034
├── runs.jsonl
├── tasks.jsonl
├── progress.log
└── meta.json
```

### Phase P

```text
/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/
├── summary.json
├── train_candidates.jsonl              # 1394 = resolved 0–7（与 step-GRPO 同口径）
├── train_grpo_resolved_1_7.jsonl       # 338  = resolved 1–7 → GRPO
├── train_step_grpo_resolved_0_7.jsonl  # 1394 = resolved 0–7 → step-GRPO
├── train_split_summary.json            # 拆分策略与计数
├── excluded_passk.jsonl                # 10 always_resolved（8/8）
├── runs.jsonl                          # 11232 attempts
├── tasks.jsonl
├── artifacts/
├── progress.log
├── runner.log
└── meta.json
```

## 7. 进训拆分策略（GRPO vs step-GRPO）

按 Phase P 的 **8 次里 resolved 次数** 再拆两套（均不含 8/8）：

| 用途 | resolved 范围 | 题数 | 文件 |
|------|---------------|-----:|------|
| **GRPO** | **1–7** | **338** | `.../train_grpo_resolved_1_7.jsonl` |
| **step-GRPO** | **0–7** | **1394** | `.../train_step_grpo_resolved_0_7.jsonl` |

说明：

- **8/8**（10 题）已在 Phase P 剔除，两套都不含。
- **GRPO** 去掉 0/8（1056 题）：只留「至少成功过一次、但不是全对」的题，更适合标准 outcome GRPO。
- **step-GRPO** 保留 0/8：需要从全错轨迹里学步骤信号时用完整 0–7 池。
- 行内额外字段：`n_resolved`、`n_repeats=8`（便于核对）。

GRPO（1–7）内部再按 resolved 分布：

| resolved / 8 | 题数 |
|-------------:|-----:|
| 1 | 121 |
| 2 | 74 |
| 3 | 37 |
| 4 | 28 |
| 5 | 31 |
| 6 | 31 |
| 7 | 16 |
| **合计** | **338** |

## 8. 下一步建议

- GRPO：`PROMPT_DATA` → `train_grpo_resolved_1_7.jsonl`（338）。
- step-GRPO：`PROMPT_DATA` → `train_step_grpo_resolved_0_7.jsonl`（1394）。
- 推理栈评测部署（adapter）用完可按需 teardown，训练时再启。
