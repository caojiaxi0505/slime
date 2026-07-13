# HyperPod 部署新镜像 + L2（真 Claude Code）冒烟设计

日期：2026-07-11  
分支：`feature/cc-ags-swe`  
上级：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
相关：[AGS 冒烟 L0/L1](./2026-07-10-ags-smoke-design.md)、[Kaniko ECR 镜像](./2026-07-10-kaniko-ecr-slime-image-design.md)、[Kaniko 操作手册](../runbooks/2026-07-11-kaniko-ecr-slime-image-ops.md)

## 目标

在 **镜像已就绪** 的前提下，分两阶段打通：

1. **Phase 1 — 部署**：用 ECR 新镜像在 `sn5-system-intern` 起 **SGLang + Anthropic adapter**，并申请 **专属公网入口**；操作者把得到的 URL **手动**写入 `SLIME_ADAPTER_PUBLIC_URL`。  
2. **Phase 2 — L2 冒烟**：扩展 `ags_smoke.py --level 2`，在真 AGS 里跑 Claude Code → adapter → SGLang，再 eval，验证 Path A 端到端最短链路。

本设计回答：

- 新镜像能否被 HyperPod Pod 拉起并提供推理？  
- AGS 沙箱能否经专属入口打到 adapter `/health` 与 Messages API？  
- 真 CC 一轮能否产出 diff，并在评测沙箱得到 `resolved`（或明确的未解决结果）？

## 非目标

- 完整 GRPO / 多机大规模训练 Job（可复用本设计产物，但不在本篇交付）  
- 改 `slime/` 训练核心或官方 `AnthropicAdapter` 行为  
- Path B / Step-GRPO  
- 本机构建镜像（继续禁止；只用已推 ECR 的 tag）  
- 把 L2 接入默认 CI / 无凭证 pytest  
- 自动改 DNS / 自动把 ALB URL 写回 git（URL **手动**配置）

## 已拍板决策

| 项 | 选择 |
|----|------|
| 范围 | **B**：Phase 1 部署 + Phase 2 L2，同一设计、分阶段验收 |
| 资源 | **可配**（`NUM_GPUS` / 节点数等变量）；文档给「最小冒烟」建议默认，不绑死 2 节点 9B |
| 公网入口 | **申请专属** Service + Ingress/ALB；URL **手动**填入 `SLIME_ADAPTER_PUBLIC_URL` |
| L2 入口 | 扩展现有 **`ags_smoke.py --level 2`**（与 L0/L1 同一 CLI） |
| 命名空间 | 仅 **`sn5-system-intern`** |
| 镜像 | 默认 `.../sn5/jiaxicao/slime:cc-ags-swe-20260710-171230`（可用 env 覆盖 tag） |

## 方案对比（Phase 1 怎么起服务）

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A. 直接上完整 `train.py` / Ray GRPO | 训练脚本顺带起 SGLang+adapter | 最接近训练 | 过重；L2 被训练编排绑架 |
| **B. 独立「推理 + adapter」部署（推荐）** | K8s Job/Deployment：SGLang serve + 同进程或同 Pod 起 segmented adapter；前面挂专属 Service/Ingress | 边界清晰；专供 L2；资源可配 | 与最终 GRPO yaml 略有分叉（可文档对齐） |
| C. 只换现网训练 yaml 的 image | 改现网模板镜像 tag | 少写新文件 | 耦合他人实验；难做「最小冒烟」 |

**推荐 B。** L2 只需「模型可达 + adapter 公网可达」，不必先拉起完整 RL。

## 架构与数据流

```text
Phase 1
  操作者
    → launch/sglang_adapter/ 提交清单（镜像、GPU、PVC、Service、Ingress）
    → Pod: SGLang (router/engine) + SegmentedAnthropicAdapter HTTP
    → Service → Ingress/ALB（专属）
    → 人工记录 ALB DNS → 写入 slime_ags.env 的 SLIME_ADAPTER_PUBLIC_URL

Phase 2
  ags_smoke --level 2
    → normalize 官方行
    → 沙箱 A: workspace_init + toolchain + Claude Code
         （ANTHROPIC_BASE_URL = SLIME_ADAPTER_PUBLIC_URL）
         → adapter → SGLang
    → git_diff
    → 沙箱 B: dispatch.evaluate(diff)
    → 打印 resolved / 细节
```

与训练 `generate()` 对齐的要点：

- 双沙箱：rollout 侧跑 CC，eval 侧另开实例打分（与 `generate.py` 一致）。  
- Adapter：复用 `SegmentedAnthropicAdapter` + `ANTHROPIC_BASE_URL` / session token 约定。  
- **不**强制走完整 `generate()` / fan-out GRPO（避免 Ray args）；L2 编排放在 smoke 内，调用 `agent_runtime` + `swe_eval.dispatch`。

### 公网入口约定

- 创建 **本实验专属** K8s `Service` + `Ingress`（名称带可识别前缀，如 `jiaxicao-cc-ags-adapter`）。  
- HyperPod ALB 控制器依赖 **Service 后端 + Pod target**（现网经验：不要把宿主机 ENI IP 直接挂 TargetGroup）。  
- Ingress 就绪后，操作者执行：

```bash
# 示例：从 Ingress 状态读到 ADDRESS 后
export SLIME_ADAPTER_PUBLIC_URL=http://<alb-dns>
# 写入 examples/claudecode_ags/env/slime_ags.env（勿提交密钥）
```

- Adapter 必须监听 `0.0.0.0:<port>`，并提供 **`/health`**（或现网等价健康路径），供 ALB 与人工 `curl` 探测。  
- AGS 在广州等区域访问 `ap-southeast-3` ALB 有跨区 RTT；设计允许，但 L2 超时预算要留足（沿用 `SLIME_CC_TIME_BUDGET_SEC` 等）。

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/launch/sglang_adapter/` | Phase 1：yaml 模板、`submit_deploy.sh`、README（如何申请入口、如何填 URL） |
| `examples/claudecode_ags/smoke/ags_smoke.py` | 扩展 `--level 2` |
| `examples/claudecode_ags/smoke/README.md` | 补充 L2 前置与命令 |
| `examples/claudecode_ags/env/slime_ags.env` | 操作者本地填 `SLIME_ADAPTER_PUBLIC_URL`（不入库） |
| `docs/superpowers/runbooks/…-l2-ops.md`（实现阶段可补） | 部署 + L2 操作手册 |

不改默认 `docker/Dockerfile`；继续用已推 ECR 镜像。

## Phase 1 详细设计

### 输入

| 变量 | 含义 | 建议默认 |
|------|------|----------|
| `IMAGE_URI` | 完整训练镜像 | `085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:cc-ags-swe-20260710-171230` |
| `HF_CHECKPOINT` | 模型权重路径（FSx） | **必填**，由操作者指定 |
| `NUM_GPUS` | 每副本 GPU | 可配（冒烟建议从单节点少卡起） |
| `TP_SIZE` / `DP` 等 | SGLang 并行 | 可配，与 `NUM_GPUS` 一致约束 |
| `ADAPTER_PORT` | 容器内 adapter 端口 | `18001`（与现 `SLIME_ADAPTER_PORT` 对齐） |
| `K8S_NAMESPACE` | | `sn5-system-intern` |
| `FSX_PVC_NAME` | | `youtu-sn2-007` |
| `SERVICE_NAME` / `INGRESS_NAME` | 专属入口名 | `jiaxicao-cc-ags-adapter` 一类 |

### 行为

1. `submit_deploy.sh` 渲染并 `kubectl apply -n sn5-system-intern`：  
   - Pod/Job：拉 `IMAGE_URI`，挂 PVC，起 SGLang（命令/脚本可配），再起 adapter（可用镜像内 Python 调 `SegmentedAnthropicAdapter` 的最小入口，或复用现有 launch 片段）。  
   - `Service` 指向 adapter 端口。  
   - `Ingress` 申请 ALB（注解按集群现网惯例；实现时对照平台已验证的 Ingress 模板）。  
2. 打印：如何 `kubectl get ingress`、如何 `curl http://$URL/health`。  
3. **不**自动改 `slime_ags.env`；README 写明手动粘贴 URL。

### Phase 1 成功标准

1. Pod Running，SGLang 可本地（Pod 内）打通。  
2. `curl -fsS "$SLIME_ADAPTER_PUBLIC_URL/health"` 从登录节点成功。  
3. （推荐）AGS 沙箱内 `curl` 同一 URL `/health` 成功——可做为 Phase 1.5 小检查，或并入 L2 开头。

## Phase 2 详细设计（L2）

### CLI

```bash
python -m examples.claudecode_ags.smoke.ags_smoke --level 2 \
  --dataset-type swebench_verified \
  --data-path /path/to/test.parquet \
  --instance-id <id>
```

前置：已 `source` / load `claude_code.env` + `slime_ags.env`；`SLIME_ADAPTER_PUBLIC_URL` 已指向专属入口；AGS 凭证与 L0/L1 相同；toolchain（COS mount 等）可用。

### L2 步骤（单题）

1. Load + `normalize_official_row`（与 L1 同）。  
2. **Reachability gate（可选但推荐）**：在临时沙箱或 L2 沙箱内 `curl` `$SLIME_ADAPTER_PUBLIC_URL/health`；失败则明确退出，避免空跑 CC。  
3. 沙箱 A：`make_sandbox(image)` → `agent_runtime.prepare_workspace(..., rollout_side=True)` → `install_toolchain` → `run_claude`（env 含 `ANTHROPIC_BASE_URL=SLIME_ADAPTER_PUBLIC_URL` 与 session token）。  
4. `git_diff`；销毁或退出沙箱 A。  
5. 沙箱 B：`dispatch.evaluate`（与 L1 相同 grader 路径；patch = 模型 diff，不是 gold）。  
6. 打印 `resolved`、关键 metrics、diff 长度；exit：成功 resolved → 0；跑通但未解决 → 1；硬错误 → 2（与 L1 风格一致）。

### 刻意不做

- 不启动 Ray / `train.py`。  
- 不 fan-out 多 segment、不算 GRPO。  
- 不把 L2 设为默认 pytest。

### Phase 2 成功标准

1. `--level 2` 在文档指定样例题上跑完主路径无硬错误。  
2. 日志可见：CC 调用 adapter、SGLang 有请求、产出非空或明确空 diff、eval 返回结构化结果。  
3. （理想）至少一题 `resolved=true`；若模型弱导致未解决，只要链路完整且 exit 码语义正确，仍算冒烟基础设施通过——文档需区分「链路通过」与「题目解决」。

**本设计约定：** Phase 2 验收以 **链路通过** 为必须；**题目解决** 为加分项（依赖所选 checkpoint 能力）。

## 测试与验收

| 层 | 内容 |
|----|------|
| 离线 | `ags_smoke --help` 含 level 2；参数校验；不连 AGS/GPU 的单测 |
| Phase 1 人工 | apply → Pod Ready → Ingress ADDRESS → curl `/health` |
| Phase 2 人工 | `--level 2` 单题；看日志与 exit code |
| 回归 | 现有 L0/L1 行为不变 |

## 风险

| 风险 | 缓解 |
|------|------|
| ALB Target 为空 / EndpointSlice vs Endpoints | 对照现网 Ingress+Service 模板；文档写排查步骤 |
| 跨区 RTT（AGS 广州 → 新加披 ALB） | 加大 CC/HTTP 超时；健康检查先过再跑 CC |
| 镜像大、拉起慢 | 提前 ImagePull；PVC 缓存权重 |
| Adapter 与 SGLang 地址配错 | Pod 内用 localhost/router 固定约定；公网只暴露 adapter |
| L2 误用 gold patch | level 2 禁止走 L1 gold 路径；diff 仅来自 `git_diff` |
| 资源变量配错导致 OOM | README 给最小建议组合；失败看事件 |

## 成功标准（整篇）

1. 仓库内有可提交的 Phase 1 部署清单 + 提交脚本 + 说明（专属入口 + 手动 URL）。  
2. `ags_smoke --level 2` 实现并文档化。  
3. 在有权限环境：Phase 1 health 通；Phase 2 链路通过（题目解决加分）。  
4. 全程 `kubectl -n sn5-system-intern`；镜像来自 ECR 已构建 tag。

## 审阅结论（已确认并实现）

1. Phase 1：独立 sglang+adapter 部署（方案 B）  
2. 专属 Ingress/ALB + 手动写 `SLIME_ADAPTER_PUBLIC_URL`  
3. L2 = `ags_smoke --level 2`  
4. GPU/并行可配；验收以链路通过为准  

实现计划：[2026-07-11-sglang-adapter-l2-smoke.md](../plans/2026-07-11-sglang-adapter-l2-smoke.md)  
操作手册：[2026-07-11-sglang-adapter-l2-ops.md](../runbooks/2026-07-11-sglang-adapter-l2-ops.md)
