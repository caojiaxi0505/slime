# COS 中 Claude Code 与 Node 的安装机制

本文记录 Path A 训练 / 评测中，AGS 沙箱如何从 COS 挂载恢复 Node 和 Claude Code。

## 1. 背景

Claude Code 不直接依赖训练镜像中的 Node / npm / claude。每个 AGS sandbox 启动后，会先从 COS 挂载目录解压一份固定工具链，再运行 Claude Code。

这样做的目的：

- 训练镜像不需要频繁重打，只要 COS 中的工具包稳定即可；
- GRPO、Hybrid 和 SWE484 评测使用同一套 Claude Code / Node；
- sandbox 是临时环境，每次启动都能从干净状态安装。

## 2. 当前默认配置

| 变量 | 默认值 | 含义 |
|---|---|---|
| `SLIME_AGENT_TOOLCHAIN_MODE` | `cos` | 使用 COS 挂载里的工具包 |
| `SLIME_AGENT_COS_MOUNT` | `/mnt/code_agent` | AGS sandbox 内的 COS 挂载路径 |
| `SLIME_AGENT_COS_NODE_PACKAGE` | `node-v20.18.1-linux-x64.tar.xz` | Node 安装包 |
| `SLIME_AGENT_COS_CC_PACKAGE` | `cc-prefix-2.1.104-linux-x64.tar.gz` | Claude Code 安装包 |
| `SLIME_AGENT_COS_NODE_DIR` | `/opt/node-cos` | Node 解压目标目录 |
| `SLIME_AGENT_COS_CC_DIR` | `/opt/cc-cos` | Claude Code 解压目标目录 |
| `CLAUDE_BIN` | `/usr/local/bin/claude` | 训练代码调用的 claude 路径 |

这些默认值定义在：

- `examples/claudecode_ags/agent_runtime.py`
- `examples/claudecode_ags/launch/*/pytorchjob.yaml.template`
- `examples/claudecode_ags/launch/run_hybrid_1node_debug.sh`

## 3. 安装流程

每条 agent episode 进入 AGS sandbox 后，都会调用 `install_toolchain()`。

当 `SLIME_AGENT_TOOLCHAIN_MODE=cos` 时，实际执行逻辑是：

1. 检查 COS 挂载中是否存在两个包：
   - `/mnt/code_agent/node-v20.18.1-linux-x64.tar.xz`
   - `/mnt/code_agent/cc-prefix-2.1.104-linux-x64.tar.gz`
2. 删除旧目录：
   - `/opt/node-cos`
   - `/opt/cc-cos`
3. 重新创建目录。
4. 解压 Node：
   - `tar -xJf node-v20.18.1-linux-x64.tar.xz -C /opt/node-cos --strip-components=1`
5. 解压 Claude Code：
   - `tar -xzf cc-prefix-2.1.104-linux-x64.tar.gz -C /opt/cc-cos`
6. 建立软链：
   - `/usr/local/bin/node -> /opt/node-cos/bin/node`
   - `/usr/local/bin/npm -> /opt/node-cos/bin/npm`
   - `/usr/local/bin/npx -> /opt/node-cos/bin/npx`
   - `/usr/local/bin/claude -> /opt/cc-cos/bin/claude`
7. 校验版本：
   - `node --version`
   - `npm --version`
   - `claude --version`

只要任一步失败，当前 sandbox 的 agent 会失败；训练侧会按 episode 失败处理。

## 4. 为什么 Node 和 Claude Code 要一起放 COS

Claude Code 是 npm / Node 生态里的 CLI。即使 `claude` 包已经存在，也需要稳定的 Node runtime 才能执行。

因此 COS 中必须同时保留两类文件：

| 包 | 作用 | 缺失后现象 |
|---|---|---|
| Node tarball | 提供 `node`、`npm`、`npx` | `node` / `npm` 不存在，Claude Code 无法启动 |
| Claude Code prefix | 提供 `bin/claude` 及其依赖 | `/usr/local/bin/claude` 链接失败或 `claude --version` 失败 |

之前训练中出现过 COS 中 `cc` / `node` 文件被删的情况。此时新建 sandbox 无法完成工具链安装，RL rollout 会大量失败；已经启动并完成安装的 sandbox 不一定立刻受影响。

## 5. 训练和评测如何接入

GRPO、Hybrid 和 SWE484 评测模板都显式设置：

```yaml
- name: SLIME_AGENT_TOOLCHAIN_MODE
  value: "cos"
- name: SLIME_AGENT_COS_MOUNT
  value: "/mnt/code_agent"
- name: SLIME_AGENT_COS_NODE_PACKAGE
  value: "node-v20.18.1-linux-x64.tar.xz"
- name: SLIME_AGENT_COS_CC_PACKAGE
  value: "cc-prefix-2.1.104-linux-x64.tar.gz"
```

因此，只要使用当前模板提交任务，默认就是 COS 工具链口径。

## 6. 和 tarball 模式的区别

代码里还保留了 `tarball` 模式：

| 模式 | 来源 | 使用场景 |
|---|---|---|
| `cos` | AGS sandbox 内的 `/mnt/code_agent` | 集群正式训练 / 评测 |
| `tarball` | `SLIME_AGENT_NODE_TARBALL` 和 `SLIME_AGENT_CC_TARBALL` 指向的本地路径 | 本地或特殊调试 |
| `skip` / `none` | 不安装 | 只在环境已经自带工具链时使用 |

`load_env.sh` 有保护逻辑：如果配置成 `tarball` 但本地 tarball 不存在，会 warning 并回退到 `cos`。

## 7. 最小健康检查

进入一个能访问 AGS sandbox 的任务后，工具链安装成功至少应满足：

```bash
test -r /mnt/code_agent/node-v20.18.1-linux-x64.tar.xz
test -r /mnt/code_agent/cc-prefix-2.1.104-linux-x64.tar.gz
node --version
npm --version
claude --version
ls -l /usr/local/bin/claude
```

预期：

- 两个 COS 包都可读；
- `node`、`npm`、`claude` 都能输出版本；
- `/usr/local/bin/claude` 指向 `/opt/cc-cos/bin/claude`。

## 8. 常见问题

| 问题 | 原因 | 处理 |
|---|---|---|
| `test -r /mnt/code_agent/...` 失败 | COS 挂载缺文件，或 AGS tool 没挂载只读目录 | 恢复 COS 文件；确认提交任务时使用正确 AGS tool / mount 模板 |
| `tar -xJf` 失败 | Node 包损坏或不是 `.tar.xz` | 重新上传 Node 包 |
| `tar -xzf` 失败 | Claude Code 包损坏或不是 gzip tar | 重新上传 Claude Code prefix 包 |
| `claude --version` 失败 | `bin/claude` 缺失、Node 不可用、包内依赖不完整 | 重新打包 Claude Code prefix，并在 sandbox 内验证 |
| 新任务大量 agent 失败 | 新 sandbox 无法安装工具链 | 先停任务，恢复 COS 包后再 resume / 重提 |

## 9. 运维原则

- COS 中的 Node 和 Claude Code 包属于训练基础设施，不应随意删除。
- 替换包时必须换文件名或先完成小规模 smoke test，再让主训练使用。
- 不同实验可以共用同一套工具链包，但不要在训练运行中覆盖包内容。
- 记录使用的 Claude Code 版本和 Node 版本，保证评测和训练可复现。

