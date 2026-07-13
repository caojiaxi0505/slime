# SWE-Gym：Gold 稳定筛选 + 8 次轨迹剔全对设计

日期：2026-07-11  
前置：L0/L1/L2 链路已通（`ags_smoke`）；adapter 可独立部署  
数据：`/mnt/sn-007/jiaxicao/datasets/SWE-Gym/data/train-00000-of-00001.parquet`（2438 题，均有非空 `patch`）

## 1. 目标

在接入 GRPO 之前，对 **SWE-Gym train** 做两阶段筛选与摸底：

1. **Phase G — Gold 稳定筛选**：对每题用官方 gold patch 评测 **4 次**；**任意一次** `resolved=False`（或基础设施失败）→ **排除**。  
2. **Phase P — 8 次轨迹摸底 + 过易题剔除**：仅对 Phase G 存活题，用当前 checkpoint + Claude Code 每题跑 **8 次**；**仅排除 8 次全部 `resolved=True` 的题**（模型已稳定做对、对 GRPO 信息量低）。  
   - **不是**按 pass@8（≥1 次做对）做保留筛选；0/8、1/8、…、7/8 均保留进训候选。  
   - pass@k / `n_resolved/8` 仍写入汇总，仅作报表，**不**作为本阶段 keep/drop 规则。

**本篇交付文档与约定**；实现计划另开。不在本篇内提交 GRPO Job。

## 2. 非目标

- 不改 `slime/` 训练核心、不改官方 AnthropicAdapter  
- 不做 Step-GRPO / Path B  
- 不在本轮保证高 `resolved` 率；Phase P 保留难题（含 0/8）  
- 不默认把 Verified / SWE-bench lite 混进本流水线（可后续复用同一 runner）  
- 不把全量 2438×(4+8) 一次打满当作必须；实现需支持 **子集 / 断点续跑**

## 3. 锁定决策

| 项 | 选择 |
|----|------|
| 数据集 | SWE-Gym train parquet；`--dataset-type swegym` |
| Phase G 次数 | **每题 4 次** gold eval |
| Phase G 排除规则 | **4 次全部** `resolved=True` 才保留；任一次失败（含 infra exit≠链路成功）→ 排除 |
| Phase G 评测物 | 行内官方 **`patch`（gold）**；走与 L1 相同的 `swe_eval.dispatch.evaluate` |
| Phase P 次数 | **每题 8 次** live Claude Code（对齐 L2：CC → `git_diff` → 新沙箱 eval） |
| Phase P 排除规则 | **仅当 8/8 `resolved=True`** → 排除（`exclude_reason=always_resolved`）；其余（含 0/8）→ **保留** |
| Phase P 报表 | 仍计算 pass@8、`n_resolved/8`、非空 diff 率等，**只报表、不筛题** |
| 顺序 | **必须先 G 后 P**；P 输入 = G 的 `kept`；P 输出 = `train_candidates.jsonl`（去掉全对题） |
| 并发 | 可配；默认保守（避免打爆 AGS / 单 adapter）；G 与 P 可不同并发 |
| **Agent / 评测超时** | 见 §3.1；默认与现网 L2 / `generate` 对齐 |
| 产物 | 机器可读 JSONL + 汇总 JSON/CSV；轨迹按 run 落盘（复用 L2 `--artifact-dir` 语义） |

### 3.1 Timeout 约定（必须显式配置并可覆盖）

与 `examples/claudecode_ags/generate.py` / L2 smoke / `env/*.env` 对齐；batch runner 通过 CLI 或环境变量注入，并写入每次 run 的 `meta` / `runs.jsonl`。

| 超时 | 默认 | 环境变量 / CLI | 作用 |
|------|------|----------------|------|
| **Claude Code 墙钟** | **1800s** | `SLIME_CC_TIME_BUDGET_SEC` / `--time-budget` | Phase P：`run_claude` / `run_agent` 的 `time_budget_sec`；超时则该次 attempt 结束，记 `claude_exit` 为 budget 超时码（与 harness 一致），**通常 `resolved=False`** |
| **Eval（测例）墙钟** | **600s** | `SLIME_CC_EVAL_TIMEOUT_SEC` / `--eval-timeout` | Phase G 与 Phase P：`swe_eval.dispatch.evaluate(..., timeout_sec=…)` |
| **单次 generate 总护栏** | `budget + eval + 180` | `SLIME_CC_GENERATE_GUARD_SEC`（可选） | 若 batch 包一层总超时，与训练 `generate` 一致；默认可不启，由上面两项分别约束 |
| **AGS 沙箱 lifetime** | `30m` | `SLIME_AGENT_AGS_TIMEOUT` | 沙箱实例最大存活；须 **≥** agent budget + eval + 启动/工具链余量，否则会被 AGS 先杀 |
| **AGS boot / runtime** | `600s` / `600s` | `SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC` / `SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC` | 启动与单次 exec 运行时上限 |
| **CC→adapter HTTP** | 见 `claude_code.env` | `API_TIMEOUT_MS=1200000`（20min）、`CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000`、`API_FORCE_IDLE_TIMEOUT=0` | 模型流式空闲 / API；Phase P 必须加载同一套，避免「墙钟未到但 CC 先断」 |
| **Bash 工具** | 见 `claude_code.env` | `BASH_DEFAULT_TIMEOUT_MS=300000`、`BASH_MAX_TIMEOUT_MS=900000` | agent 内 Bash；与 L2 一致 |

**超时如何计入筛选：**

- Phase G：eval 超时或沙箱被杀 → 该次失败 → **整题排除**（与 infra / unresolved 同等严格）。  
- Phase P：agent 或 eval 超时 → 该次 `resolved=False`（或 `infra_ok=false`）；**不**因此整题排除，除非最终仍出现 8/8 `resolved=True`（超时题几乎不可能进「全对」桶）。  
- 超时不得静默当成功；`runs.jsonl` 须带 `timeout_hit` / `claude_exit` / `reason` 之一可区分。

**全量跑建议：** 保持默认 1800/600；若 AGS `TIMEOUT=30m` 偏紧，将 `SLIME_AGENT_AGS_TIMEOUT` 提到 **45m–60m**，或把 `--time-budget` 降到能装进沙箱 lifetime 的值。二者必须在 `meta.json` 里写清实际生效值。

## 4. 流程

```text
SWE-Gym train (2438)
        │
        ▼
┌───────────────────────┐
│ Phase G: gold ×4/题   │  复用 L1 语义（无 Claude / 无 adapter）
│ keep iff all 4 pass   │
└───────────┬───────────┘
            │ kept.jsonl
            ▼
┌───────────────────────┐
│ Phase P: CC ×8/题     │  复用 L2 语义（需 SLIME_ADAPTER_PUBLIC_URL）
│ drop iff 8/8 resolved │  报表仍含 pass@k，但不按 pass@k 筛
└───────────┬───────────┘
            │ train_candidates.jsonl
            ▼
     → GRPO 题集（后续设计；可再人工收紧）
```

### 4.1 Phase G（Gold 稳定筛选）

对每个 `instance_id`：

1. `normalize_official_row(..., dataset_type=swegym)` → 得到 `image`、`workdir`、评测字段。  
2. 重复 **4** 次（独立沙箱，互不共享状态）：  
   - `make_sandbox(image)`  
   - `initialize_task_workspace(..., rollout_side=False)`（与 L1/eval 侧一致）  
   - `evaluate(diff_text=gold_patch)`  
   - 记录：`resolved`、`applied_cleanly`、`exit_code`、`reason`、耗时、sandbox_id  
3. **判定**  
   - 若任一次：infra 异常（沙箱起不来、init 失败、evaluate 抛错）→ **排除**（记 `exclude_reason=infra`）  
   - 若任一次：`resolved=False` → **排除**（记 `exclude_reason=gold_unresolved`）  
   - 仅当 4/4 `resolved=True` → **保留**

**不做** `--allow-unresolved`：Phase G 的目的就是扔掉不稳定 / 坏环境题。

### 4.2 Phase P（8 次轨迹；只剔全对）

输入：Phase G 的 `kept` 列表（可再加 `--limit` / id 白名单做小规模试跑）。

对每个 kept `instance_id`：

1. 重复 **8** 次（独立 rollout 沙箱 + 独立 eval 沙箱，对齐 L2）：  
   - health gate（可每题首次做，或全局预检一次）  
   - workspace init（rollout）+ toolchain + `run_claude`  
   - `git_diff` → 导出 artifact（trajectory / model.diff）  
   - eval sandbox init + `evaluate(model_diff)`  
2. 单次 run 结果字段至少包含：  
   `resolved`、`applied_cleanly`、`diff_chars`、`claude_exit`、`artifact_dir`、错误摘要  
3. **题级汇总（报表）**  
   - `n_resolved` / 8、`pass_rate = n_resolved/8`  
   - `pass@8` = `1` iff `n_resolved >= 1`（仅报表）  
   - `nonempty_diff_rate`、`infra_fail_count`  
4. **题级筛选（唯一规则）**  
   - 若 `n_resolved == 8`（8 次全部做对）→ **排除**，写入 `excluded_passk.jsonl`，`exclude_reason=always_resolved`  
   - 否则（`n_resolved ∈ {0,…,7}`）→ **保留**，写入 `train_candidates.jsonl`  
5. **infra 失败**：该次计 `resolved=False`（且 `infra_ok=false`）。若 8 次中混有 infra 失败，只要不是「8 次全部 resolved=True」，题仍保留；若实现上无法区分「真 resolved」与「未跑成」，以实际写入的 `resolved` 布尔为准。不默认额度外重试（可选 `--retry-infra N`，默认 0）。

**明确禁止**：用「pass@8==0 排除」或「必须至少对一次才保留」之类规则；本阶段只要不是全对就留着。

Phase P **需要** 已部署的 SGLang+adapter 与 `SLIME_ADAPTER_PUBLIC_URL`（与 L2 相同）。

## 5. CLI / 产物约定（实现时对齐）

建议新入口（名称可微调，语义固定）：

```bash
# Phase G
python -m examples.claudecode_ags.eval.swegym_filter \
  --phase gold \
  --data-path /mnt/sn-007/jiaxicao/datasets/SWE-Gym/data/train-00000-of-00001.parquet \
  --dataset-type swegym \
  --gold-repeats 4 \
  --eval-timeout 600 \
  --out-dir /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_gold_YYYYMMDD \
  [--limit N] [--instance-id ...] [--concurrency C]

# Phase P
python -m examples.claudecode_ags.eval.swegym_filter \
  --phase passk \
  --kept-jsonl /path/to/kept.jsonl \
  --passk-repeats 8 \
  --time-budget 1800 \
  --eval-timeout 600 \
  --out-dir /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_YYYYMMDD \
  [--limit N] [--concurrency C]
```

也可拆成两个模块；但 **题集契约** 必须稳定。

### 5.1 目录布局

```text
$OUT/
  meta.json                 # 数据路径、repeats、git sha、开始/结束时间、concurrency
  runs.jsonl                # 每次 attempt 一行（phase=gold|passk）
  tasks.jsonl               # 每题一行汇总
  kept.jsonl                # Phase G 输出：存活题（供 Phase P）
  excluded.jsonl            # Phase G：排除题 + reason
  train_candidates.jsonl    # Phase P 输出：去掉 8/8 全对后的进训候选
  excluded_passk.jsonl      # Phase P：always_resolved 等
  summary.json              # 全局计数、n_resolved 分布、pass@k 报表
  artifacts/<instance_id>/r<k>/   # Phase P：trajectory 等（可选，磁盘大时可用开关）
```

### 5.2 `runs.jsonl` 行（示意）

```json
{
  "phase": "gold",
  "instance_id": "getmoto__moto-7365",
  "repeat_idx": 0,
  "resolved": true,
  "applied_cleanly": true,
  "infra_ok": true,
  "elapsed_sec": 120.5,
  "details": {"mode": "...", "exit_code": 0}
}
```

### 5.3 断点续跑

- 以 `(phase, instance_id, repeat_idx)` 为幂等键；已成功写入的 run **跳过**。  
- Phase G：某题已出现一次失败即可提前标记 excluded，**不必**强行跑满剩余 gold（实现可选优化；文档允许）。  
- Phase P：默认跑满 8 次（即使中途已多次 resolved），以便判定是否 **8/8 全对** 并估计 `n_resolved`；**不要**因「已经对过一次」早停。可选：若已出现一次 `resolved=False`，可证明「非全对」并早停剩余次数（`--drop-always-resolved-early-stop`，默认关，全量摸底时建议关）。

## 6. 与现有代码的关系

| 能力 | 复用 |
|------|------|
| 行归一化 / 镜像 | `dataset_normalize.normalize_official_row` + `swegym` |
| Gold eval | `ags_smoke.run_l1` 同路径：`initialize_task_workspace` + `swe_eval.dispatch.evaluate(gold)` |
| Live CC + eval | `ags_smoke.run_l2` 同路径：`agent_runtime` + dual sandbox + artifact export |
| 环境 | `env/load_env.sh`、`SLIME_ADAPTER_PUBLIC_URL`（仅 Phase P） |

实现时优先 **抽公共函数**（单次 gold attempt / 单次 L2 attempt），再由 batch runner 调度；避免复制粘贴整份 `ags_smoke.py`。

## 7. 规模与成本（量级）

| 阶段 | 最坏 attempt 数 | 说明 |
|------|-----------------|------|
| G | 2438 × 4 ≈ **9752** | 早停失败可显著减少 |
| P | \|kept\| × 8 | 取决于 G 存活率；若 kept≈1000 → 8000 次 CC |

AGS 沙箱 + 跨区 adapter RTT 下，全量可能数天级。实现必须支持：

- `--limit` / id 文件  
- 高并发可配但默认低  
- 续跑  

**建议试跑**：先 `limit=20` 跑通 G→P 再放大。

## 8. 验收标准

1. 设计文档审阅通过；实现计划另开。  
2. Phase G：对试跑子集，4 次全过的题进入 `kept.jsonl`；任一失败进入 `excluded.jsonl` 且 reason 可解释。  
3. Phase P：仅消费 `kept`；每题 8 次；**仅 8/8 resolved 进 `excluded_passk.jsonl`**；其余进 `train_candidates.jsonl`；`summary.json` 含 `n_resolved` 分布与 pass@k 报表。  
4. 离线单测：汇总逻辑（G：4 全过才 keep；P：仅剔 `n_resolved==8`）；不连 AGS。  
5. 全程 `kubectl` 若涉及仍限 `sn5-system-intern`；评测本身走 AGS API，不依赖本机 docker。

## 9. 风险

| 风险 | 缓解 |
|------|------|
| Gold 本身 flaky（非确定性测试） | 正是 Phase G 要滤掉的；4 次任挂即丢 |
| 镜像拉取 / AGS 配额 | 低并发、续跑、分批 |
| Agent 超时与 AGS lifetime 冲突 | §3.1：`time_budget` + `eval` + 余量 ≤ `SLIME_AGENT_AGS_TIMEOUT`；超时写入 runs |
| Phase P 空 diff（如缺 packaging） | 记入 `nonempty_diff_rate`；环境修复可另开任务，不阻塞本流水线定义 |
| 磁盘被 trajectory 打满 | artifact 开关 / 只保留失败或 resolved 的轨迹 |
| 与 GRPO 题集格式不一致 | `kept.jsonl` / passk `tasks.jsonl` 保留原始 metadata 字段，后续转换 |

## 10. 后续（本篇之后）

1. ~~实现计划 + batch runner~~ — done  
2. ~~Phase G 全量~~ — done：`kept=1404`（见 notes）  
3. ~~Phase P~~ — **deferred**：1404 已够用，暂不跑 8× 剔全对  
4. 以 Phase G `kept.jsonl`（1404）为 GRPO `PROMPT_DATA` 起点（可再人工收紧）  
5. 再提交 GRPO 训练  

结果笔记：[../notes/2026-07-11-swegym-gold-filter-results.md](../notes/2026-07-11-swegym-gold-filter-results.md)

## 11. 审阅要点（已确认）

1. Phase G：4 次全过才保留；任一不过（含 infra）排除 — **同意**  
2. Phase P：只排除 8/8 全对；不按 pass@8 筛 — **同意**  
3. pass@k / `n_resolved` 只做报表 — **同意**  
4. 试跑先 limit=20 — **同意**  
5. Timeout 默认 agent 1800s / eval 600s + `claude_code.env` — **同意**

实现计划：[2026-07-11-swegym-gold-filter-passk.md](../plans/2026-07-11-swegym-gold-filter-passk.md)  
操作手册：[2026-07-11-swegym-gold-filter-passk-ops.md](../runbooks/2026-07-11-swegym-gold-filter-passk-ops.md)
