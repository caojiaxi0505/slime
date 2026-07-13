# Path A 1-node GRPO Debug — 操作手册

日期：2026-07-12  
设计：[2026-07-12-grpo-1node-debug-design.md](../specs/2026-07-12-grpo-1node-debug-design.md)  
计划：[2026-07-12-grpo-1node-debug.md](../plans/2026-07-12-grpo-1node-debug.md)  
数据：[2026-07-12-swegym-filter-passk-results.md](../notes/2026-07-12-swegym-filter-passk-results.md)

## 1. 涉及文件

| 路径 | 作用 |
|------|------|
| `examples/claudecode_ags/data/to_slime_prompt_jsonl.py` | 筛选 JSONL → slime `PROMPT_DATA` |
| `examples/claudecode_ags/launch/run_grpo_1node_debug.sh` | 1×8GPU colocate GRPO + 训后 Verified |
| `examples/claudecode_ags/launch/grpo_adapter_alb/` | Service + ALB Ingress（AGS → Master :9002） |
| `examples/claudecode_ags/launch/grpo_1node_job/` | 1×8GPU PyTorchJob（Master only） |
| `examples/claudecode_ags/env/load_env.sh` | 加载 `claude_code.env` + `slime_ags.env` |
| `examples/claudecode_ags/env/slime_ags.env` | AGS 密钥、超时（勿提交 git） |

默认镜像：`085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:cc-ags-swe-20260710-171230`（可用 env 覆盖）

默认 `LOG_DIR`：`/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_1node_grpo_debug`

## 2. 前置条件

1. **L2 adapter 已删除** — `jiaxicao-cc-ags-adapter` Deployment/Service/Ingress 不得存在（与训练抢 GPU / 误用旧 ALB URL）。
2. **GPU 空闲** — HyperPod 目标节点 8×GPU 无其他 Job。
3. **模型路径存在** — `HF_CHECKPOINT`（`Qwen3.5-9B`）与 `REF_MODEL_PATH`（`Qwen3.5-9B_torch_dist`）。
4. **AGS 凭证** — `SLIME_AGENT_AGS_ENV_FILE` 或 `SLIME_AGENT_AGS_SECRET_ID` + `SLIME_AGENT_AGS_SECRET_KEY`。
5. **训练数据** — 已生成 `.slime.jsonl`（338 行，见 §3）。

确认 L2 已删：

```bash
kubectl -n sn5-system-intern get deploy,svc,ingress jiaxicao-cc-ags-adapter 2>&1 | rg -q 'NotFound' && echo OK
# 或重新执行（幂等）：
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/sglang_adapter
bash ./submit_deploy.sh --delete
```

## 3. 数据转换（338 行）

源：`train_grpo_resolved_1_7.jsonl`（resolved 1–7，338 题）。  
产出：`train_grpo_resolved_1_7.slime.jsonl`（`prompt` 为 `[{role,content}]` + `extra_info`）。

> **2026-07-12 fix**：Qwen3.5 会加载 HF `processor`；`prompt` 必须是 conversation list，不能是纯字符串。
>
> **2026-07-12 Plan A（host OOM）**：默认 `OPTIMIZER_CPU_OFFLOAD=0`（关 Adam CPU offload，避免 colocate `sleep` 时 host 只剩 ~14Gi）；`SAVE_INTERVAL=1`；PyTorchJob `restartPolicy: Never`。需要旧行为时设 `OPTIMIZER_CPU_OFFLOAD=1`。

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m examples.claudecode_ags.data.to_slime_prompt_jsonl \
  --src /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.jsonl \
  --dst /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl \
  --expect-rows 338
```

校验：`wc -l` = 338；`train.py` 读入无 schema 错。

## 4. Adapter 公网入口（ALB）

现网做法：internet-facing ALB → Service → **Master Pod :9002**。

本仓库已提供同款清单：

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/grpo_adapter_alb
bash ./submit_alb.sh
# 等待 ADDRESS 后：
export SLIME_ADAPTER_PUBLIC_URL=http://<ADDRESS>   # ALB 听 :80，URL 不要写 :9002
```

要点：

| 项 | 值 |
|----|-----|
| 资源名 | `jiaxicao-grpo-1node-debug-adapter`（Service + Ingress） |
| ns | `sn5-system-intern` |
| 后端端口 | **9002**（`SLIME_ADAPTER_PORT` / `SHIM_PORT`） |
| Master labels | `app=cc-ags-recorder`,`workload=jiaxicao-grpo-1node-debug` |
| Worker | **不要**带上述 labels（避免 ALB 打到无 adapter 的节点） |

- **禁止** `127.0.0.1` / 已删 L2 `jiaxicao-cc-ags-adapter` ALB。
- Adapter 在 `generate` 内晚于 `train.py` 启动；ALB `/health` 在就绪前 **UNHEALTHY** 属正常（与现网一致）。
- 删除：`bash ./submit_alb.sh --delete`

## 4b. 提交 1-node 训练 Job

Master-only PyTorchJob；labels 与 §4 ALB selector 一致。

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/grpo_1node_job
# 可选 dry-run
bash ./submit_job.sh --dry-run
# 提交（默认 PHASE=all，注入已创建的 ALB URL + AGS K8s secret）
bash ./submit_job.sh

kubectl -n sn5-system-intern get pytorchjob jiaxicao-grpo-1node-debug -w
kubectl -n sn5-system-intern get pods -l workload=jiaxicao-grpo-1node-debug -w
```

删除 Job：`bash ./submit_job.sh --delete`（ALB 可保留给下次用）。

## 5. 默认超时

| 变量 | 默认 | 含义 |
|------|------|------|
| `SLIME_CC_TIME_BUDGET_SEC` | **1800** | Agent 墙钟 30min |
| `SLIME_CC_EVAL_TIMEOUT_SEC` | **600** | Eval 10min |
| `SLIME_AGENT_AGS_TIMEOUT` | **45m** | AGS 沙箱 |
| `SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC` | **2700** | AGS HTTP 单请求（≥ agent 预算） |

## 6. 启动（dry-run → 实跑）

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
source examples/claudecode_ags/env/load_env.sh
export SLIME_ADAPTER_PUBLIC_URL=http://<alb-hostname>

# dry-run（打印 train.py 参数，不执行）
bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh

# 实跑：训练 + 训后 1 次 Verified
RUN=1 PHASE=all bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh
```

`PHASE`：

| 值 | 行为 |
|----|------|
| `all`（默认） | ~21 步 GRPO → 训后 SWE-bench Verified |
| `train` | 仅 GRPO，不 eval |
| `eval` | 仅 Verified（slime eval-only：`--num-rollout 0` + `--eval-interval 1`）；需 `LOG_DIR/slime_save` 非空 |

`PHASE=eval` 走 slime 的 eval-only 路径：`train.py` 在 `num_rollout == 0` 且 `eval_interval` 已设时只跑 Verified，不训练。

训后单独重跑 Verified（checkpoint 已存在时）：

```bash
# 指向已完成训练的 run 目录（覆盖默认 LOG_DIR）
LOG_DIR=/mnt/sn-007/jiaxicao/checkpoints/cc-ags/<prior-run> \
  RUN=1 PHASE=eval bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh
```

**注意：** `PHASE=eval` 在 **dry-run（`RUN=0`）时也会检查** `LOG_DIR/slime_save` 非空；无 checkpoint 会直接 exit 1。预览 eval 参数前请先 `LOG_DIR=<prior-run>` 指向已有 `slime_save` 的目录。

## 7. 怎么监控

| 看什么 | 命令 / 路径 |
|--------|-------------|
| 主日志 | `tail -f ${LOG_DIR}/run.log` |
| Rollout dump | `${LOG_DIR}/rollout_dumps/` |
| Checkpoint | `${LOG_DIR}/slime_save/` |
| WandB 本地 | `${LOG_DIR}/wandb/` |
| OOM / CUDA | `run.log`、Ray worker 日志、`nvidia-smi` |
| AGS 失败 | `[generate]` / sandbox 超时、`SLIME_AGENT_AGS_TIMEOUT` 相关栈 |

每步约 128 条轨迹（batch 16 × n=8）× 30min agent budget，墙钟很长属预期。

## 8. 与 L2 adapter 互斥

| 资源 | GRPO 1-node debug | `jiaxicao-cc-ags-adapter` |
|------|-------------------|---------------------------|
| GPU | 本 Job colocate 占满 8 GPU | 独立 Deployment 占 GPU |
| Adapter URL | 本 Job SGLang + segmented adapter | 旧 ALB Ingress（已删） |
| 权重 | 训练 update 后同步到 rollout | 静态快照，无训后同步 |

**进入本阶段前必须 `--delete` L2 部署**；训练全程不得依赖该 Deployment。

## 9. 验收（设计 §9）

| # | 标准 |
|---|------|
| 1 | `.slime.jsonl` **338** 行；`train.py` 加载无 schema 错 |
| 2 | 约 **20–21** 步 GRPO 完成；写出 `slime_save` + `run.log` |
| 3 | 训后 **SWE-bench Verified** 跑完；有 per-instance `resolved` 与 pass@1 汇总 |
| 4 | 全程无依赖已删除的 `jiaxicao-cc-ags-adapter` |

不设 resolved 率门槛；本篇为链路 debug。

## 10. 离线自检（开发机）

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m pytest tests/claudecode_ags/test_to_slime_prompt_jsonl.py -v
bash -n examples/claudecode_ags/launch/run_grpo_1node_debug.sh
SLIME_ADAPTER_PUBLIC_URL=http://10.0.0.1:18001 RUN=0 \
  bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh >/tmp/grpo_1node_dry.txt
rg -n "claudecode_ags.generate|fanout_grpo|rollout-batch-size 16|n-samples-per-prompt 8|num-rollout 21|actor-num-nodes 1|colocate|swebench_verified" /tmp/grpo_1node_dry.txt
```

预期：pytest PASS；`bash -n` 无语法错；dry-run 输出含上述 token。
