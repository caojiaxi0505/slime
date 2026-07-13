# Path A：1-node GRPO Debug 设计

日期：2026-07-12  
分支 / worktree：`feature/cc-ags-swe` → `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`  
上级：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
数据：[SWE-Gym 筛选结果](../notes/2026-07-12-swegym-filter-passk-results.md)

## 1. 目标

在 **1 节点 / 8 GPU** 上把 Path A **普通 GRPO** 跑通约 **1 个 epoch**，验证：

1. 筛选后的 GRPO 题集能被 `train.py` 正确加载；  
2. Claude Code + AGS + 训练侧 SGLang/adapter（colocate）能完成 rollout → reward → GRPO update；  
3. 训完后用**同一次 run 的最新权重**跑一次 **SWE-bench Verified** eval。

**本篇交付约定与文档**；实现计划另开。不在本篇内保证 resolved 率或刷榜。

## 2. 非目标

- 多机 / 2-node 全量训练  
- Step-GRPO / Path B / hybrid  
- mid-train / 多 seed / 刷榜式 Verified 反复评测（本篇仅 **训后 1 次** Verified）  
- 复用已停掉的 L2 独立 adapter Deployment 做训练（无权重同步）  
- 改 `slime/` 训练核心或官方 `AnthropicAdapter` 行为  

## 3. 已拍板参数

| 项 | 值 |
|----|-----|
| 节点 | **1 node / 8 GPU**，`--colocate` |
| 模型 | `Qwen3.5-9B` HF + `_torch_dist`（路径已存在） |
| 训练题集 | `resolved ∈ [1,7]`，**338** 题 |
| 源文件 | `.../swegym_passk_20260711_090409/train_grpo_resolved_1_7.jsonl` |
| `rollout-batch-size` | **16** |
| `n-samples-per-prompt` | **8**（每步 128 条轨迹） |
| `num-rollout` | **≈21**（`ceil(338/16)`，约过一遍） |
| `global-batch-size` | **128** |
| Agent timeout | **1800s**（`SLIME_CC_TIME_BUDGET_SEC`，30min） |
| Eval timeout | **600s**（`SLIME_CC_EVAL_TIMEOUT_SEC`，10min） |
| AGS sandbox timeout | **45min**（`SLIME_AGENT_AGS_TIMEOUT=45m`） |
| Mid-train eval | **跳过**（`SKIP_EVAL_BEFORE_TRAIN=1`，不设 mid-interval） |
| Post-train eval | **训完后 1 次** SWE-bench Verified；`n-samples-per-eval-prompt=1` |
| `EVAL_DATA` | 默认 `.../swe_agent_ags_swebench_verified/test.parquet`（可用 env 覆盖） |
| Generate / reward | `examples.claudecode_ags.generate.generate` + binary reward + `slime.rollout.fanout_grpo.post_process_rewards` |
| 命名空间 | 仅 `sn5-system-intern`（若走 K8s Job）；禁止碰其他 ns |

## 4. 方案对比（怎么起训）

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **A. 新建 1-node debug launcher（推荐）** | 从 tencent 2-node GRPO 脚本裁成 1×8GPU colocate；Path A 用 `claudecode_ags` 路径 | 接近真训；权重同步正确 | 要写 launch + 数据转换 |
| B. 只扩 `run_grpo_example.sh` | 在 dry-run skeleton 上硬塞参数 | 改动面小 | 缺 Ray/SGLang/并行度，难一次跑通 |
| C. 训完用旧 L2 ALB adapter 做 eval | 训练自起 SGLang；eval 打旧 Deployment | eval 省事 | **测的是旧权重**；且该部署已按本设计关掉 |

**推荐 A。**

## 5. 数据：筛选 JSONL → slime `PROMPT_DATA`

源行当前只有 `instance_id` / `metadata` / passk 统计字段，**没有** slime 训练需要的：

- `prompt`（`--input-key prompt`）  
- `extra_info`（`--metadata-key extra_info`）

`metadata` 已含 `problem_statement`、`image`、`dataset_type=swegym`、grader 字段等。

**转换约定：**

```text
prompt      := [{"role":"user","content": metadata.problem_statement}]
              （Qwen3.5 会加载 HF processor；slime Dataset 在 processor!=None 时需要 conversation list）
extra_info  := metadata（可附带 n_resolved 等调试字段；不得丢 instance_id / image / FAIL_TO_PASS 等）
```

产出建议路径（实现可微调，但需写进 ops）：

```text
.../swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl
```

转换脚本放在 `examples/claudecode_ags/`（或 `eval/`）下，可离线重跑、可校验行数 = 338。

## 6. 运行形态

```text
                    ┌─────────────────────────────────────┐
  HyperPod 1 node   │  Ray + Megatron actor (colocate)     │
  8×GPU             │  SGLang engines + segmented adapter  │
                    │  train.py GRPO ~21 steps             │
                    └──────────────┬──────────────────────┘
                                   │ SLIME_ADAPTER_PUBLIC_URL
                                   │ (训练节点可达公网/节点 IP，非旧 ALB)
                                   ▼
                            AGS 沙箱 × Claude Code
                                   │
                                   ▼
                              SWE eval（30min budget）
```

要点：

1. **训练必须用本次 Job 拉起的 SGLang/adapter**，保证 update 后权重同步到 rollout。  
2. 独立 L2 Deployment `jiaxicao-cc-ags-adapter` **在进入本阶段前关掉**，避免占 GPU / 误用旧 URL。  
3. `SLIME_ADAPTER_PUBLIC_URL` 需改为训练节点对 AGS 可达的地址（实现计划里写清获取方式；可参考现网 2-node 脚本）。  
4. 镜像默认 ECR：`.../sn5/jiaxicao/slime:cc-ags-swe-20260710-171230`（可用 env 覆盖）。  
5. Checkpoint：`HF_CHECKPOINT` + `REF_MODEL_PATH`（`Qwen3.5-9B` / `Qwen3.5-9B_torch_dist`）。

## 7. 训后 SWE-bench Verified eval

- **时机：** `num-rollout` 跑完、checkpoint 已 save 之后。  
- **题集：** **SWE-bench Verified** 全量（默认 parquet：`/mnt/sn-007/youtu-agent/yuleiqin/SWE_code/DataEng/RL_DATA/data_valid/swe_agent_ags_swebench_verified/test.parquet`，约 **484** 题；可用 `EVAL_DATA` / `QWEN35_EVAL_DATA` 覆盖）。  
- **采样：** `n-samples-per-eval-prompt=1`。  
- **权重：** 加载本 run `slime_save`（或同进程紧接 eval），**禁止**打已删除的 L2 ALB。  
- **超时：** agent **30min**，eval **10min**，沙箱 **45min**（与训练侧一致）。  
- **成功：** 跑完并产出 Verified `resolved` 汇总（pass@1 / 分题结果）；不设 resolved 率门槛。

## 8. 交付物

| # | 交付 | 说明 |
|---|------|------|
| 1 | 数据转换脚本 + `.slime.jsonl` | 338 行，`prompt`/`extra_info` |
| 2 | `run_grpo_1node_debug.sh`（或等价） | 1-node colocate GRPO；默认参数见 §3 |
| 3 | 训后 Verified eval 入口 | 独立脚本或同一 launcher 的 post 阶段 |
| 4 | ops runbook | 起停、env、日志、成功标准、与 L2 部署互斥说明 |

## 9. 成功标准

1. 转换后 `PROMPT_DATA` 行数 = 338，且 `train.py` 能读入不报 schema 错。  
2. 约 **20–21** 步 GRPO 跑完，写出 `slime_save` / run log。  
3. 训后 SWE-bench Verified 跑完，有 per-instance `resolved` 与汇总 pass@1。  
4. 过程中无依赖已删除的 `jiaxicao-cc-ags-adapter` Deployment。

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 1×8GPU colocate OOM / 并行度不够 | 先沿用 9B 现网并行度裁剪（TP/CP/rollout GPUs）；必要时降 `SGLANG_SERVER_CONCURRENCY`，不先降 batch/n |
| 每步 128 条 × 30min 墙钟很长 | 接受 debug 成本；concurrency 与 AGS 配额在 ops 里调；断点续跑按现网 `RESUME_DEBUG_ROLLOUT_DATA` 习惯 |
| `SLIME_ADAPTER_PUBLIC_URL` 不可达 | 启动检查 `/health`；文档写清节点公网/内网选择 |
| 筛选 JSONL 缺字段 | 转换后做必填字段 assert（`instance_id`/`image`/`problem_statement`/`FAIL_TO_PASS`） |
| 与旧 adapter 抢 GPU | 本阶段开始前 `--delete` 掉 L2 部署（已执行） |

## 11. 前置清理（操作）

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/sglang_adapter
bash ./submit_deploy.sh --delete
```

删除：`Deployment` / `Service` / `Ingress` `jiaxicao-cc-ags-adapter`（`sn5-system-intern`）。  
`slime_ags.env` 里旧 ALB URL 仅作历史；**训练不得再依赖该 URL**。
