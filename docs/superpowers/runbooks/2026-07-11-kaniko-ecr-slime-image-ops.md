# Kaniko 构建完整 slime 镜像并推 ECR — 操作手册

日期：2026-07-11  
分支 / worktree：`feature/cc-ags-swe` → `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`  
设计：[2026-07-10-kaniko-ecr-slime-image-design.md](../specs/2026-07-10-kaniko-ecr-slime-image-design.md)  
计划：[2026-07-10-kaniko-ecr-slime-image.md](../plans/2026-07-10-kaniko-ecr-slime-image.md)

本文说明：**怎么执行、涉及哪些文件、怎么监控、怎么验收**。不讲设计取舍细节。

---

## 1. 做什么

在 HyperPod 命名空间 **`sn5-system-intern`** 里起一个 **Kaniko Job**，用当前 worktree 源码构建**完整 slime 训练镜像**（SGLang + Megatron + Path A + EFA 用户态），推到：

```text
085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:<IMAGE_TAG>
```

账号以 `aws sts get-caller-identity` 为准；上例为现网实测账号。

### 硬约束

| 规则 | 说明 |
|------|------|
| 禁止本机 `docker build` / `docker push` | 构建只在集群 Kaniko Pod 内完成 |
| 权限范围 | 所有 `kubectl` 仅 `-n sn5-system-intern` |
| 源码可见性 | context 必须在 FSx PVC 挂载路径下（`/mnt/sn-007/...`） |
| EFA 设备 | **构建 Job 不申请** `vpc.amazonaws.com/efa`；只把用户态装进镜像。训练时再申请 EFA |

---

## 2. 涉及哪些文件

路径均相对 worktree 根：  
`/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`

### 2.1 提交与 Job

| 文件 | 作用 |
|------|------|
| `examples/claudecode_ags/launch/kaniko/submit_build.sh` | 主入口：检查环境、刷新 ECR secret、渲染模板、`kubectl apply` |
| `examples/claudecode_ags/launch/kaniko/job.yaml.template` | Kaniko Job 模板（资源、PVC、secret、Kaniko args） |
| `examples/claudecode_ags/launch/kaniko/README.md` | 目录内短说明（指向本手册） |

### 2.2 镜像配方

| 文件 | 作用 |
|------|------|
| `docker/Dockerfile.kaniko` | 完整训练镜像：基于上游 `docker/Dockerfile`，末尾 `COPY . /root/slime`（不用 `git clone THUDM/slime`），并安装 EFA |
| `docker/Dockerfile` | 上游基线（Kaniko 配方与之对齐，但 slime 安装方式不同） |
| `.dockerignore` | 缩小 context（排除 `.venv`、缓存等） |
| `docker/patch/<PATCH_VERSION>/` | Megatron / SGLang 补丁（构建时 COPY） |
| `requirements.txt` | Python 依赖 |

### 2.3 EFA 用户态（`ENABLE_EFA=1` 时必需）

| 文件 | 作用 |
|------|------|
| `docker/efa/install-efa-in-container.sh` | 容器内安装 EFA 用户态（Kaniko 友好：无硬件时不因 `fi_info` 失败） |
| `docker/efa/fix-efa-conflict.sh` | 处理包冲突后重装；使用合法 installer 参数（含 `--disable-ngc`） |
| `docker/efa/README.md` | EFA 脚本说明 |
| `/mnt/sn-007/jiaxicao/code/efa_install/` | 个人备份副本（可选；提交时也可用 `EFA_SCRIPTS_DIR` 指向此处） |

### 2.4 集群侧对象（运行时）

| 对象 | 名称 / 值 |
|------|-----------|
| Namespace | `sn5-system-intern` |
| PVC | `youtu-sn2-007` → 挂载 `/mnt/sn-007` |
| ServiceAccount | 默认 `default` |
| Secret | `jiaxicao-kaniko-ecr`（脚本用 `aws ecr get-login-password` 刷新，约 12h 有效） |
| ECR repo | `ap-southeast-3` / `sn5/jiaxicao/slime` |

### 2.5 文档

| 文件 | 作用 |
|------|------|
| 本手册 | 操作 / 监控 / 验收 |
| `docs/superpowers/specs/2026-07-10-kaniko-ecr-slime-image-design.md` | 设计 |
| `docs/superpowers/plans/2026-07-10-kaniko-ecr-slime-image.md` | 实现计划 |
| `tests/claudecode_ags/test_kaniko_launch.py` | 离线静态测试（不提交真实 Job） |

---

## 3. 怎么执行

### 3.1 一次性准备

```bash
# 1) kubecontext 能操作 sn5-system-intern
kubectl get ns sn5-system-intern
kubectl auth can-i create jobs -n sn5-system-intern

# 2) ECR 仓库（仅首次；已存在会报错可忽略）
aws ecr create-repository --region ap-southeast-3 --repository-name sn5/jiaxicao/slime

# 3) 确认 EFA 脚本在 worktree 内（ENABLE_EFA=1）
ls /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/docker/efa/install-efa-in-container.sh
ls /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/docker/efa/fix-efa-conflict.sh
```

### 3.2 提交构建

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/kaniko

# 可选：只看渲染后的 YAML，不 apply
./submit_build.sh --dry-run

# 正式提交（默认 ENABLE_EFA=1，大资源）
./submit_build.sh
```

脚本会打印：

- Job 名（如 `kaniko-slime-cc-ags-swe-YYYYMMDD-HHMMSS-xxxxx`）
- 完整镜像 URI
- 后续 `logs` / `wait` / `describe-images` 命令

### 3.3 常用环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `K8S_NAMESPACE` | `sn5-system-intern` | 命名空间 |
| `BUILD_CONTEXT` | 当前 worktree 根 | 必须在 `/mnt/sn-007` 下 |
| `DOCKERFILE` | `docker/Dockerfile.kaniko` | 相对 context 的 Dockerfile |
| `IMAGE_TAG` | `cc-ags-swe-YYYYMMDD-HHMMSS` | ECR tag |
| `ENABLE_EFA` | `1` | 是否装 EFA 用户态 |
| `EFA_SCRIPTS_DIR` | 空 | 若设了，提交前拷贝脚本到 `docker/efa/` |
| `FSX_PVC_NAME` | `youtu-sn2-007` | FSx PVC |
| `FSX_MOUNT_PATH` | `/mnt/sn-007` | PVC 挂载点 |
| `SERVICE_ACCOUNT` | `default` | Pod SA |
| `DOCKER_CONFIG_SECRET` | `jiaxicao-kaniko-ecr` | ECR 登录 secret |
| `CPU_REQUEST` / `CPU_LIMIT` | `180` / `192` | CPU（完整编译很重，勿随意调小） |
| `MEMORY_REQUEST` / `MEMORY_LIMIT` | `1024Gi` / `1800Gi` | 内存 |
| `SGLANG_IMAGE_TAG` | `v0.5.13-cu129` | 基础镜像 tag |
| `PATCH_VERSION` | `latest` | `docker/patch/` 子目录 |

示例：指定 tag 提交

```bash
IMAGE_TAG=cc-ags-swe-manual-001 ./submit_build.sh
```

### 3.4 脚本实际做了什么（顺序）

1. 解析账号、tag、Job 名  
2. 检查 `BUILD_CONTEXT` 在 FSx 挂载下，且 Dockerfile /（可选）EFA 脚本存在  
3. 检查 namespace、PVC  
4. 检查 ECR 仓库（可用 `--create-repo`）  
5. 刷新 `jiaxicao-kaniko-ecr` secret  
6. `envsubst` 渲染 `job.yaml.template`  
7. `kubectl apply -n sn5-system-intern`  
8. 打印监控与验收命令  

**不会**调用本机 `docker`。

---

## 4. 怎么监控

### 4.1 跟日志（推荐）

```bash
NS=sn5-system-intern
JOB=kaniko-slime-cc-ags-swe-...   # submit_build.sh 输出的名字

kubectl -n "$NS" get job "$JOB" -o wide
kubectl -n "$NS" get pods -l job-name="$JOB" -o wide
kubectl -n "$NS" logs -f job/"$JOB"
```

完整编译通常 **约 1 小时**（现网成功一次约 59 分钟），前期长时间停在 `Unpacking rootfs` / 编译 `flash-attn` 是正常的。

### 4.2 等完成

```bash
kubectl -n sn5-system-intern wait --for=condition=complete --timeout=6h job/"$JOB"
```

失败时：

```bash
kubectl -n sn5-system-intern get job "$JOB" -o jsonpath='{.status}'
kubectl -n sn5-system-intern logs job/"$JOB" --tail=120
```

### 4.3 看资源是否按预期

```bash
kubectl -n sn5-system-intern get job "$JOB" \
  -o jsonpath='{.spec.template.spec.containers[0].resources}{"\n"}'
```

期望默认接近：requests `180 CPU / 1Ti`，limits `192 CPU / 1800Gi`。

### 4.4 后台轮询（可选）

可自行挂一个轮询脚本：每 1–2 分钟查 `succeeded` / `failed`，成功则 `describe-images`。注意状态行里的 `failed=0` 不要当成失败关键字误报。

提交时也可：

```bash
./submit_build.sh --follow   # apply 后直接 logs -f
```

---

## 5. 怎么验收

### 5.1 Job Complete + ECR 有 tag

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_TAG=cc-ags-swe-20260710-171230   # 换成你的 tag

aws ecr describe-images --region ap-southeast-3 \
  --repository-name sn5/jiaxicao/slime \
  --image-ids imageTag="$IMAGE_TAG" \
  --query 'imageDetails[0].{tags:imageTags,pushed:imagePushedAt,size:imageSizeInBytes}' \
  --output json
```

完整 URI：

```text
${AWS_ACCOUNT_ID}.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:${IMAGE_TAG}
```

### 5.2 可选：同 namespace 拉镜像冒烟

```bash
kubectl -n sn5-system-intern run slime-pull-smoke --rm -it --restart=Never \
  --image="${AWS_ACCOUNT_ID}.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:${IMAGE_TAG}" \
  --command -- true
```

### 5.3 现网已成功一例（参考）

| 项 | 值 |
|----|-----|
| Job | `kaniko-slime-cc-ags-swe-20260710-171230-03550` |
| Tag | `cc-ags-swe-20260710-171230` |
| 耗时 | ~59m |
| 镜像大小 | ~22.8 GB |
| EFA | 用户态已装入（构建节点无 EFA 硬件属正常） |

---

## 6. 失败后怎么处理

| 现象 | 常见原因 | 处理 |
|------|----------|------|
| Job Pending | 资源过大、节点忙 | 看 `describe pod` 的 `FailedScheduling`；必要时略降 requests（勿降到无法编译） |
| 拉不到 `slimerl/sglang` | Docker Hub / 网络 | 先 mirror 到 ECR，改 `SGLANG_IMAGE_TAG` / Dockerfile `FROM` |
| ECR push 401/拒绝 | secret 过期或权限不足 | 重新跑 `submit_build.sh`（会刷新 secret）；确认实例角色能 push 该 repo |
| EFA 步骤 exit 1 | 旧脚本 `-y` bug / 非法 `--force-overwrite` / 无硬件却强验 `fi_info` | 使用当前 `docker/efa/*.sh`（已修）；**不要**给构建 Job 申请 EFA 设备 |
| 误用上游 clone | 用了 `docker/Dockerfile` 而非 `Dockerfile.kaniko` | 确认 `DOCKERFILE=docker/Dockerfile.kaniko` |

删除失败 / 旧 Job：

```bash
kubectl -n sn5-system-intern delete job <JOB_NAME>
```

---

## 7. 构建成功后下一步

把训练 / SGLang Job 的 `containers.image` 换成新 URI，例如：

```text
085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:cc-ags-swe-20260710-171230
```

多机训练时再在 **训练 Pod** 上申请 EFA，并 `source /etc/profile.d/efa.sh`。  
L2（真 Claude Code）与完整 GRPO 启动不在本手册范围。

---

## 8. 离线自检（不碰集群）

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m pytest tests/claudecode_ags/test_kaniko_launch.py -v
```

覆盖：模板占位符、`Dockerfile.kaniko` 使用 `COPY` 本地源码、`submit_build.sh --help` / `--dry-run`。
