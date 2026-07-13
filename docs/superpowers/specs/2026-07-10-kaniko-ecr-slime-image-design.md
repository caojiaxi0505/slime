# HyperPod Kaniko 构建完整 slime 镜像并推 ECR 设计

日期：2026-07-10  
分支：`feature/cc-ags-swe`  
上级设计：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
相关：[真 AGS 冒烟](./2026-07-10-ags-smoke-design.md)、现网 [HyperPod 训练指南](../../../../slime-tencent/tutorials/aws_hyperpod_training_guide_zh.md)（路径以本机 checkout 为准）

## 目标

在 **AWS HyperPod / K8s** 上用 **Kaniko Job** 构建 **完整 slime 训练镜像**（含 SGLang + Megatron 等，基于仓库 `docker/Dockerfile`），并推送到 **AWS ECR**，供后续 HyperPod 部署与 L2（真 Claude Code）使用。

## 非目标

- **本机 `docker build` / `docker push`**（硬禁止）  
- 腾讯 TCR（仅 SWE 评测镜像；与本设计无关）  
- 本轮不写完整 GRPO 训练 yaml（可引用现网模板，镜像 tag 换成本次产物）  
- 不改 `slime/` 训练核心逻辑  
- 不在本设计内完成 L2 冒烟（镜像就绪后的下一步）

## 已拍板决策

| 项 | 选择 |
|----|------|
| 构建位置 | HyperPod/K8s Job，**禁止本机构建** |
| 构建器 | **Kaniko** |
| 镜像内容 | **完整 slime 训练镜像（B）** |
| Dockerfile 基线 | 仓库 `docker/Dockerfile`（可加薄包装适配 Kaniko/本地源码） |
| K8s 命名空间 | **`sn5-system-intern`**（`kubectl -n sn5-system-intern`） |
| ECR region | **`ap-southeast-3`** |
| ECR repository | **`sn5/jiaxicao/slime`** |
| 建仓命令（人工一次性） | `aws ecr create-repository --region ap-southeast-3 --repository-name sn5/jiaxicao/slime` |

完整镜像 URI：

```text
${AWS_ACCOUNT_ID}.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:${IMAGE_TAG}
```

`AWS_ACCOUNT_ID` 由 `aws sts get-caller-identity` 得到；设计与脚本用环境变量，不写死账号（现网教程里曾出现 `054486717055`，以调用方实际账号为准）。

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/launch/kaniko/job.yaml.template` | Kaniko Job 模板（namespace、镜像、args、volume） |
| `examples/claudecode_ags/launch/kaniko/submit_build.sh` | 渲染 tag/路径后 `kubectl apply -n sn5-system-intern` |
| `examples/claudecode_ags/launch/kaniko/README.md` | 前置条件、建仓、提交、看日志、验收 |
| `docker/Dockerfile.kaniko`（可选） | 若需相对上游 Dockerfile 改为 `COPY` 本地源码而非 `git clone THUDM/slime` |

不改默认 `docker/Dockerfile` 的上游语义时，优先用 **build-arg + 薄 Dockerfile** 覆盖最后「安装 slime」阶段，避免分叉过大。

## 架构与数据流

```text
操作者（本机只 kubectl / aws cli）
  → submit_build.sh
  → kubectl apply -n sn5-system-intern Job/kaniko-slime-build-...
  → Pod(Kaniko)
       · context = FSx 上的 slime 源码树（含 docker/）
       · dockerfile = docker/Dockerfile 或 docker/Dockerfile.kaniko
       · destination = ECR .../sn5/jiaxicao/slime:${IMAGE_TAG}
  → ECR 仓库出现新 tag
```

本机职责仅限：配置 kubecontext、提交 Job、拉日志、确认 ECR 有 tag。  
**构建与 push 全部在集群内完成。**

## 构建上下文（context）

Kaniko 需要能读到完整 build context（`docker/Dockerfile`、`docker/patch/`、源码）。

**推荐（默认）：**

- 在 FSx（或集群已挂载的共享盘）上放置/同步 `feature/cc-ags-swe` worktree，例如：  
  `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`  
  （若 HyperPod 节点挂载路径不同，以实际 `volumeMount` 为准，脚本用 `BUILD_CONTEXT` 变量。）
- Job 通过 `hostPath` / PVC / FSx CSI 将该目录挂进 Kaniko 容器，例如挂到 `/workspace/slime`。
- Kaniko：`--context=dir:///workspace/slime`，`--dockerfile=docker/Dockerfile.kaniko`（或约定路径）。

**备选：** Job 启动时 `git clone` 你们的 remote 分支（需集群能访问 git、且分支已 push）。本轮默认不依赖公网拉私有分支，优先 FSx 目录。

### 与上游 Dockerfile 的差异（必做适配）

上游 `docker/Dockerfile` 末尾为：

```dockerfile
RUN git clone https://github.com/THUDM/slime.git ... && pip install -e .
```

Path A 需要的是 **当前 worktree（含 cc-ags-swe 改动）**，不能只装官方 `main`。

**约定：**

- 提供 `docker/Dockerfile.kaniko`：在复制上游主体的基础上，将「clone slime」改为：  
  `COPY . /root/slime` + `pip install -e /root/slime --no-deps`（或等价），并保证 `docker/patch/` 仍按原 `COPY docker/patch/...` 可用。  
- 或：多阶段中最后一阶段 `COPY` 覆盖 `/root/slime`。  
- Build-arg 保留：`SGLANG_IMAGE_TAG`、`PATCH_VERSION`、`MEGATRON_COMMIT` 等，默认与上游 Dockerfile 一致，可用环境变量覆盖。

EFA：现网完整训练镜像通常需容器内 EFA 用户态（见 HyperPod 指南 §5）。若上游 Dockerfile 未含 EFA，本轮 **文档标明**：要么在 `Dockerfile.kaniko` 增加现网 `install-efa-in-container` 步骤（context 中附带脚本），要么验收时注明「镜像无 EFA、仅单机/非 EFA 任务可用」。**默认目标是完整可多机训练**，设计要求 Dockerfile.kaniko **包含 EFA 安装步骤**（脚本路径可配置，来自 FSx 上现网拷贝）。

## Kaniko Job 要点

- **namespace：** `sn5-system-intern`  
- **镜像：** 官方 Kaniko executor（pin 具体 digest/tag，避免 latest 漂移），例如 `gcr.io/kaniko-project/executor:v1.23.2`（实现时锁定经确认可用的版本）  
- **关键 args（示意）：**
  - `--context=dir:///workspace/slime`
  - `--dockerfile=docker/Dockerfile.kaniko`
  - `--destination=${ECR_URI}:${IMAGE_TAG}`
  - `--cache=true`（可选；cache repo 可同 ECR 下 `sn5/jiaxicao/slime/cache` 或禁用先求稳）
  - 如需：`--build-arg SGLANG_IMAGE_TAG=...`
- **资源：** 完整 slime 镜像编译（flash-attn 等）极重；Job 需足够 CPU/内存（实现时给可调 requests/limits；文档写清「勿用过小的 build 节点」）。  
- **TTL：** `ttlSecondsAfterFinished` 便于清理。  
- **重启：** `restartPolicy: Never`。

### ECR 认证

Kaniko 推 ECR 常见方式（实现选一，文档写死一种）：

1. **IRSA / Pod Identity**（推荐）：ServiceAccount 绑定可 `ecr:GetAuthorizationToken` + 对该 repo 的 push 权限。  
2. **Kaniko Docker config secret**：预先用 `aws ecr get-login-password` 生成的 config 挂为 secret（有过期问题，不如 IRSA）。

设计默认：**优先 IRSA + 专用 ServiceAccount**（名称如 `jiaxicao-ecr-kaniko`，实现时与平台同学确认是否已有可复用 SA）。若命名空间已有通用 ECR push SA，脚本通过 `SERVICE_ACCOUNT` 变量复用，不新建。

拉 **基础镜像** `slimerl/sglang:...`：集群需能访问 Docker Hub / 镜像加速；若不能，需先把 base 镜像同步到 ECR 并改 `FROM`（作为风险项，见下）。

## 提交脚本行为

`submit_build.sh`：

1. 检查：`kubectl` 可读、`namespace sn5-system-intern` 存在。  
2. 解析：`AWS_ACCOUNT_ID`、`IMAGE_TAG`（默认 `cc-ags-swe-$(date +%Y%m%d-%H%M%S)`）、`BUILD_CONTEXT`、`DOCKERFILE`。  
3. 可选：`aws ecr describe-repositories`；不存在则提示执行 create-repository（脚本默认 **不** 自动建仓，避免误建；可用 `--create-repo` 显式打开）。  
4. 渲染 Job 名（唯一）→ `kubectl apply -n sn5-system-intern -f ...`  
5. `kubectl logs -f` 提示命令打印到 stdout。  
6. 成功标准：Job Complete + `aws ecr describe-images` 能看到该 tag。

本机 **不** 调用 `docker`。

## 环境变量（脚本）

| 变量 | 含义 | 示例/默认 |
|------|------|-----------|
| `AWS_REGION` | ECR region | `ap-southeast-3` |
| `ECR_REPOSITORY` | 仓库名 | `sn5/jiaxicao/slime` |
| `AWS_ACCOUNT_ID` | 账号 | sts 查询 |
| `IMAGE_TAG` | tag | `cc-ags-swe-YYYYMMDD-HHMMSS` |
| `K8S_NAMESPACE` | 命名空间 | `sn5-system-intern` |
| `BUILD_CONTEXT` | 集群内源码路径 | FSx 上 worktree |
| `SERVICE_ACCOUNT` | Kaniko SA | 平台约定 |
| `KANIKO_EXECUTOR_IMAGE` | executor 镜像 | pin 版本 |

## 与后续 HyperPod 部署的衔接（本设计交付边界）

镜像推送成功后，训练/SGLang Job 的 `containers.image` 改为：

```text
${AWS_ACCOUNT_ID}.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:${IMAGE_TAG}
```

并 `kubectl -n sn5-system-intern`（或训练实际使用的 ns，若与 build ns 不同则文档注明）提交。  
**起 SGLang + 配 `SLIME_ADAPTER_PUBLIC_URL` + L2** 作为 **后续设计/计划**，不在本篇实现范围内，但成功标准要求镜像可被该 namespace 的 Pod 拉取（ImagePull 不报错的冒烟可另做一极小 Pod）。

## 测试与验收

1. **静态：** template 含正确 namespace、destination 占位符；`submit_build.sh --help`。  
2. **人工（有集群权限时）：**  
   - create-repository（若尚未建）  
   - submit Job → Complete  
   - ECR 可见 tag  
   - （可选）同 namespace 起一短暂 Pod `image: .../slime:tag` `command: ["true"]` 验证可拉  
3. 默认 `pytest` **不** 提交真实 Kaniko Job。

## 风险

| 风险 | 缓解 |
|------|------|
| 构建时间/资源极大（编译 FA 等） | 大资源 Job；cache；错峰 |
| 拉不到 `slimerl/sglang` | 预先 mirror 到 ECR，改 FROM |
| IRSA/权限不足 | 与平台确认 SA；失败日志明确 |
| FSx 路径在 build 节点不可见 | 文档写清挂载；用实际可调度的 nodeSelector/affinity |
| 误用官方 clone 丢 Path A 改动 | 强制 Dockerfile.kaniko COPY 本地 context |
| 无 EFA 导致多机失败 | Dockerfile.kaniko 纳入 EFA 安装 |

## 成功标准

1. 仓库内有可提交的 Kaniko Job 模板 + `submit_build.sh` + README。  
2. 文档明确：**禁止本机构建**；必须 `-n sn5-system-intern`。  
3. ECR 目标为 `ap-southeast-3` / `sn5/jiaxicao/slime`。  
4. 镜像内容为完整训练栈，且安装的是 **build context 中的 slime（含本分支改动）**。  
5. 在有权限环境跑通一次：Job Complete + ECR 有 tag（实现阶段验收）。

## 审阅结论（已拍板）

- 构建：Kaniko @ `sn5-system-intern`  
- 镜像：完整 slime 训练镜像  
- ECR：`ap-southeast-3` + `sn5/jiaxicao/slime`  
- 本机只负责提交与观察，不 build
