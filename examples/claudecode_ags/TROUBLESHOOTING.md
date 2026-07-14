# Path A / AGS 排障记录

日期：2026-07-10（冒烟）、2026-07-13（1-node GRPO 训练）  
相关：`env/load_env.sh`、`env/slime_ags.env.example`、`slime/agent/sandbox_ags.py`、SWE-ReX `swerex/deployment/ags.py`

## `load_env.sh` 用 example 盖掉 Job 配置（总类 bug）

**机制：** 启动脚本 `source slime_ags.env.example` 时，若未把 Job 已注入的键列入保留列表，example 默认值会覆盖 Job。

已踩过的具体表现：

| Job 想要的 | 被盖成 | 症状 |
|------------|--------|------|
| `SLIME_ADAPTER_PORT=9002` | `18001` | ALB `/health` → **502**，样本 `adapter_session_empty` |
| `SLIME_AGENT_AGS_TIMEOUT=45m`（及 runtime 2700） | `30m` / runtime `600` | 长 rollout 中后期沙箱被收，`/execute` **404** |

**修复：** `load_env.sh` 在 source 文件前保存、source 后再恢复下列键（至少）：`SLIME_ADAPTER_PORT`、`SHIM_PORT`、`SLIME_ADAPTER_BIND_HOST`、`SLIME_ADAPTER_PUBLIC_URL`、`SLIME_AGENT_AGS_TIMEOUT`、`SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC`、`SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC`、以及 `MOUNT_*` / `TOOL_ID` 等。改完需**重提 Job** 才进进程。

---

## `FailedOperation.ContainerStart` / `port binding failed`

### 现象

- AGS `StartSandboxInstance` 失败：`ContainerStart` / `port binding failed`
- 训练侧 abort → `response_len≈0`；通常还没进 COS 装 toolchain

### 根因（简）

AGS probe 等沙箱内 **8000**（SWE-ReX）就绪。无 `SLIME_AGENT_AGS_MOUNT_*` 时启动退化为镜像内 `swerex-remote` / 现场 `pipx`，常来不及绑端口。

### 解决方案：配齐 AGS `MOUNT_*`（swerex-runtime）

```bash
SLIME_AGENT_AGS_MOUNT_NAME=rex
SLIME_AGENT_AGS_MOUNT_IMAGE=swebenchdocker.tencentcloudcr.com/swebench/swehub:swerex-runtime
SLIME_AGENT_AGS_MOUNT_IMAGE_REGISTRY_TYPE=enterprise
SLIME_AGENT_AGS_MOUNT_PATH=/nix
SLIME_AGENT_AGS_IMAGE_SUBPATH=/nix
```

有 mount 时优先跑 `/nix/swerex/run-swerex --port 8000`，probe 更容易通过。  
**注意：** 这与 `SLIME_AGENT_COS_MOUNT`（装 node/CC）不是一回事。

冒烟（2026-07-10）也曾遇同类报错；留空 `TOOL_ID` 按当前 env（含 mount）新建 tool，或确认旧 tool 配方匹配，可缓解。

---

## `TypeError: AGSSandbox.exec() ... 'idempotent'`

**现象：** 沙箱已 `InstanceId` / ALIVE，一进 `run_claude` 就 abort。

**原因：** `exec_and_wait` 传 `idempotent=True`，`AGSSandbox.exec` 未接受该参数。

**修复：** `AGSSandbox.exec`（及 `Sandbox` Protocol）增加 `idempotent: bool = True`（AGS 可忽略该提示）。

---

## `adapter_session_empty` + ALB 502

**现象：** 沙箱已起，样本全 `adapter_session_empty`；公网 ALB `/health` → 502。

**根因：** 见上文「load_env 盖掉 Job」——进程听 **18001**，Service/Ingress 转 **9002**。  
pod 内 `curl 127.0.0.1:18001/health` 可能仍是 200（adapter 本身好），但 ALB 打 9002 无人听。

**修复：** 保留 `SLIME_ADAPTER_PORT=9002`；example 默认与集群对齐为 9002。

---

## 沙箱 `/execute` 404（及少量 401）与 timeout

**现象（2026-07-13 rollout 0）：** 约 34 次 `aborted: exception`；其中多数为  
`404 Not Found` on `https://8000-<instance>.ap-guangzhou.tencentags.com/execute`（少量 `401 Unauthorized`）。  
同轮仍有非空轨迹（`response_len/mean≈1529`）。

**白话：** 房间（沙箱）跑到一半被收走了，训练机再问「命令完了没」得到「资源不存在」。

**根因：** Job 配 `SLIME_AGENT_AGS_TIMEOUT=45m`，但被 example 盖成 **`30m`**；第一轮墙钟约 31 分钟，租期到后实例回收 → `/execute` 404。进程里曾见：

| 来源 | `AGS_TIMEOUT` | `RUNTIME_TIMEOUT_SEC` |
|------|---------------|------------------------|
| Job YAML | 45m | 2700 |
| RolloutManager（被盖后） | **30m** | **600** |
| 日志 `StartSandbox` | **Timeout=30m** | startup/runtime **600** |

**修复：** `load_env.sh` 保留 timeout 相关键；example 默认改为与 Job 一致的 `45m` / `2700`。重提后用 `/proc/<RolloutMan>/environ` 与日志 `Timeout=` 核对。

---

### 排查时可看的日志信号

```text
config.mount_name / mount_image   # 空 → port binding 可疑
config.timeout = '30m'            # Job 若是 45m，说明被 example 盖掉
Timeout=30m                       # StartSandbox 实际租期
LISTEN ...:18001                  # 应为 :9002（集群 ALB）
ALB /health → 502                 # 与端口不一致一致
aborted: adapter_session_empty
404 ... /execute                  # 沙箱已回收 / 租期到
perf N: response_len/mean ...     # 0 = 仍空；>0 = 已有真实轨迹
```

---

## Hybrid step-GRPO（2026-07-13）

联调修复（`missing_eval_plan` 全 0 分、tool_use_id 对齐、wandb Stage-1/2 口径、同池数据等）见：

**[`docs/superpowers/notes/2026-07-13-hybrid-step-grpo-debug-fixes.md`](../../docs/superpowers/notes/2026-07-13-hybrid-step-grpo-debug-fixes.md)**
