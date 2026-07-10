# Slime 重构设计：CC + AGS + SWE（方案 2）

日期：2026-07-10  
状态：已确认  
范围：将 `slime-tencent` 的 Claude Code + AGS + SWE 评测能力干净迁入官方 `slime`

---

## 1. 目标

1. **少侵入核心**，便于定期同步上游  
2. **逻辑克制、目录清晰**，避免过度抽象  
3. **训练语义与现网 Path A 一致**  
4. **环境变量好记**：只用两个文件，不留旧名兼容  
5. **奖励可插拔**：传参指定函数；默认 binary  
6. **实现风格克制**：理解现网语义后按模块重写，不搬 tencent 的厚防御 / 多别名兼容壳  

---

## 2. 范围

### 2.1 做

| 项 | 说明 |
|---|---|
| Path A | Claude Code + AGS + SWE 评测 + 普通 GRPO |
| 沙箱 | `sandbox.py` 约定 + 工厂；`sandbox_ags.py` 实现 |
| 分段训练 | subagent / wipe(compact) / final 进训练；逻辑上一次任务 |
| 奖励标量 | 可插拔；默认 binary（通过 1 / 不通过 0） |
| 段间分配 | 保持现状：`R/K` 均分 |
| Advantage | 保持现状：尝试内加总 → 组内 GRPO → 广播到各段 |
| 布局 | 可复用进 `slime/`；业务与启动进 `examples/claudecode_ags/` |
| Env | 两个文件；无旧名别名层 |

### 2.2 不做

| 项 | 说明 |
|---|---|
| Path B | SWE-Agent、rollout_buffer/swe_agent、ModelProxy 等全部清理 |
| Step-GRPO / hybrid | 本次不实现 |
| 旧 env 兼容 | 不保留 `SWE_CLAUDE_*`→`CLAUDE_*`、`VERL_*` 双轨等 |
| 整迁 `reward_compose.py` | 改为 rewards 插件函数 |
| 原样搬迁 tencent 实现 | 禁止厚包装、多布尔兼容、多 env fallback 风格 |

### 2.3 「AWS」澄清

AWS 侧主要是 HyperPod / K8s / ALB / EFA；沙箱是 **腾讯 AGS**，工具链常走 **COS**。不引入 ECS/Fargate 沙箱。

---

## 3. 总体架构

```text
examples/claudecode_ags/                 # 业务入口
  generate.py                            # 编排主流程
  agent_runtime.py                       # 沙箱内：装工具链 / 写题 / 跑 CC
  rewards/*.py                           # 可插拔奖励（default=binary）
  swe_eval/                              # 评测解析
  env/                                   # 两个 env 文件
  launch/                                # 启动脚本
        │
        ▼
slime/agent/sandbox.py                   # 约定 + E2B + 薄工厂
slime/agent/sandbox_ags.py               # AGS 实现
slime/agent/adapters/anthropic.py        # 分段记账 subagent/wipe/final
slime/agent/trajectory.py                # fan_out：均分 + 共享 rollout_id
slime/rollout/fanout_grpo.py             # fan-out 安全 GRPO post_process
        │
        ▼
train.py → RolloutManager → Megatron
```

原则：

- 核心：可复用能力（沙箱后端、分段轨迹、fan-out GRPO）  
- examples：业务怎么跑（编排、runtime、评测、奖励插件、启动）  
- 官方 `examples/coding_agent_rl/`（E2B）尽量不绑架  

---

## 4. 目录结构

```text
slime/
  agent/
    sandbox.py                      # Protocol + E2B + make_sandbox()
    sandbox_ags.py                  # 仅 AGS
    adapters/anthropic.py           # Session：main/sub；段：subagent/wipe/final
    trajectory.py                   # TokenSegment / fan_out_sample_segments
  rollout/
    fanout_grpo.py                  # 按现网语义重写，与 SWE-Agent 脱钩

examples/
  coding_agent_rl/                  # 官方 E2B 示例，保持独立
  claudecode_ags/                   # AGS+CC+SWE 业务入口
    README.md
    generate.py                     # boot→runtime→diff→eval→reward插件→fan_out
    agent_runtime.py                # COS装Node/CC、写PROBLEM、跑claude等
    rewards/
      default.py                    # binary：resolved→1.0 / else→0.0
    swe_eval/                       # swebench / rebench / scaleswe …
    env/
      claude_code.env               # 仅 Claude Code 原生变量
      slime_ags.env                 # 其余全部
      load_env.sh                   # 只 source 上述两个文件（无改名映射）
    launch/                         # HyperPod/ALB/多机脚本
```

密钥：`SLIME_AGENT_AGS_ENV_FILE` 指向本机文件；不进 git，不写进两个 env 的默认机密值。

---

## 5. 运行时调用链（普通 GRPO）

```text
launch/*.sh
  → source env/load_env.sh
  → ray job → train.py
  → RolloutManager
  → --custom-generate-function-path examples.claudecode_ags.generate
       ├─ AnthropicAdapter.open_session + HTTP shim
       ├─ make_sandbox(backend=ags)
       ├─ agent_runtime：装工具链 / 写题 / 跑 claude -p
       │    └─ CC → SLIME_ADAPTER_PUBLIC_URL → adapter → SGLang
       ├─ git_diff
       ├─ 新沙箱 evaluate（swe_eval）→ 得到 resolved 等字段
       ├─ load_function(--custom-cc-reward-function-path)
       │    └─ 默认 rewards.default：binary → 标量 R
       └─ fan_out_sample_segments(R/K，共享 rollout_id)
  → --custom-reward-post-process-path slime.rollout.fanout_grpo.post_process_rewards
       └─ 尝试内加总 → 题内 GRPO → advantage 广播到各段
  → Megatron 更新
```

---

## 6. 模块职责

### 6.1 核心沙箱：`sandbox.py` / `sandbox_ags.py`

- `sandbox.py`：接口约定 + E2B + 薄工厂  
- `sandbox_ags.py`：AGS 实现  
- `SLIME_AGENT_SANDBOX_BACKEND=ags|e2b` **只选后端**，不是整文件替换  

### 6.2 `agent_runtime.py`（examples，不进核心）

沙箱 **已创建之后**：

- 安装 Node / Claude Code（如 COS）  
- 写 PROBLEM、准备 workdir  
- 启动并等待 `claude`  
- 与跑 agent 紧耦合的步骤（如 git diff）  

不负责创建/销毁 AGS（核心 sandbox）。

命名说明：不用 `sandbox_helpers` / `sandbox_launcher`，避免与「启动沙箱」混淆。

### 6.3 Adapter（anthropic）

- 显式分段：`subagent` / `wipe` / `final`  
- **`final`** = 结束时仍留在 **main** 上的回合；compact 前冻结的是 **wipe**  
- 以分段模型满足「逻辑一次任务」，不单靠官方 `TrajectoryManager` fork  

### 6.4 Trajectory / fan_out

- 一次 CC 执行 → ≥1 段 Sample  
- 共享 `rollout_id`  
- 各段 `reward = R / K`  

### 6.5 可插拔 Reward（examples）

不整迁旧 `reward_compose.py`。

```text
--custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
```

接口：

```text
compose(*, base_eval: dict, sample, args=None) -> tuple[float, dict]
# (reward, details)
```

**`rewards/default.py` = binary（默认）：**

```text
resolved == True  →  1.0
resolved == False →  0.0
```

- `resolved` 由 `swe_eval` 按约定字段给出（如 F2P/P2P 达阈值）  
- 默认不做 linear / sqrt / 复杂 shaping  
- 新策略 = 新文件 + 改启动参数一行  
- 不把 CC 奖励公式写进 `slime/` 核心  
- 不从 tencent 拷 `reward_compose` / `reward_weighting` / `outcome_reward` 壳；理解语义后写短实现  

### 6.6 三层分工

| 层 | 职责 | 配置 |
|---|---|---|
| Reward 函数 | 评测 → 标量 `R`（默认 binary） | `--custom-cc-reward-function-path` |
| fan_out | `R/K` + 共享 `rollout_id` | 核心，行为固定 |
| fanout_grpo | 加总 → GRPO → 广播 | `--custom-reward-post-process-path` |

### 6.7 `fanout_grpo.py`

语义保持现网（原 CC 路径对 `swe_agent_grpo_std.post_process_rewards` 的依赖），代码重写：

1. 按 `group_index` 聚题  
2. 按一次尝试（`index`）把各段 reward **加总**  
3. 多次尝试间减均值（可选除标准差）  
4. 同一 advantage 赋给该尝试每一段  

不是全 batch 所有 session 加总再算一个全局 advantage。

---

## 7. 训练语义（必须保持）

| 点 | 约定 |
|---|---|
| subagent / compact | 必须进训练 |
| 与 main | 逻辑上一次任务（可多 Sample） |
| 默认 R | binary：通过 1 / 不通过 0 |
| 段奖励 | 均分 `R/K` |
| Advantage | 尝试内加总 → 题内 GRPO → 广播 |
| Step-GRPO | 本次不做 |
| `final` vs `main` | main=运行中主链；final=结束时主链残留标签 |

---

## 8. 环境变量

### 8.1 原则

1. 只用两个文件  
2. 无旧名兼容 / 无别名翻译层  
3. CC 官方变量用官方原名；其余统一 `SLIME_*`  

### 8.2 文件 1：`env/claude_code.env`

仅 Claude Code 原生，例如：

- `API_TIMEOUT_MS`、流式/watchdog 相关  
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW`、`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`  
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` 及各类 `CLAUDE_CODE_*` 开关  
- `BASH_*`、`TASK_MAX_OUTPUT_LENGTH`、`MAX_MCP_OUTPUT_TOKENS`、`MAX_THINKING_TOKENS`  
- `ANTHROPIC_CUSTOM_HEADERS`  

由 `agent_runtime` 启动 `claude` 时原样注入沙箱。

### 8.3 文件 2：`env/slime_ags.env`

其余全部，建议分组：

- 沙箱：`SLIME_AGENT_SANDBOX_BACKEND`、`SLIME_AGENT_AGS_*`  
- 工具链：`SLIME_AGENT_TOOLCHAIN_MODE`、`SLIME_AGENT_COS_*`、`SLIME_AGENT_NODE_*`、`SLIME_AGENT_CC_*`  
- 网络：`SLIME_ADAPTER_PUBLIC_URL`、`SLIME_ADAPTER_BIND_HOST`、`SLIME_ADAPTER_PORT`  
- 时限：`SLIME_CC_TIME_BUDGET_SEC`、`SLIME_CC_EVAL_TIMEOUT_SEC`、`SLIME_CC_GENERATE_GUARD_SEC`  
- 采样：`SLIME_CC_TRAIN_*`、`SLIME_CC_EVAL_*`  
- 奖励阈值等：`SLIME_CC_REWARD_*`（替代历史 `VERL_*` 命名，不留 VERL 别名）  

### 8.4 加载

```bash
set -a
source examples/claudecode_ags/env/claude_code.env
source examples/claudecode_ags/env/slime_ags.env
set +a
# 本机：export SLIME_AGENT_AGS_ENV_FILE=~/.cos_ags.conf
```

`load_env.sh` 只 source，不做改名映射。启动脚本不散落大段重复 export。

---

## 9. 实现风格（全迁移适用）

语义可对照 tencent / 现网；代码按本设计模块重写。

| 不要 | 要 |
|---|---|
| 多套 env 名、层层 fallback | 只读两文件中的正式名 |
| 布尔兼容堆砌、历史字段三选一 | 模块间约定清晰字段 |
| 厚包装、间接再间接 | 短函数、直路径、易读 |
| 整文件拷贝 tencent | 理解后按模块推进实现 |

防御性只保留真正必要的（如外部 API 失败要报错清理），不为「兼容旧配置」加分支。

---

## 10. 删除 / 不迁清单

- `slime_plugins/rollout_buffer/generator/swe_agent/**`  
- `scripts/run-sweagent-*.sh`  
- Path B 专用 buffer / ModelProxy / sweagent 子进程 / RL monkey-patch  
- 一切旧 env 别名逻辑  
- 旧 `reward_compose.py` 及对 swe_agent reward 的硬依赖  
- Step-GRPO / `step_reconstruct` 本次交付  
- tencent 式厚兼容壳（即使功能相似也不原样搬）  

现网若引用 `swe_agent_grpo_std`，改为重写后的 `slime.rollout.fanout_grpo`。

---

## 11. 与官方示例关系

| 路径 | 角色 |
|---|---|
| `examples/coding_agent_rl/` | 官方 E2B + harness，跟上游 |
| `examples/claudecode_ags/` | AGS+CC+SWE 入口 |

---

## 12. 建议实施顺序

1. 官方 `slime` 开干净分支（勿整包保留无关 core diff）  
2. `sandbox_ags` + 工厂分发（重写，风格克制）  
3. adapter/trajectory 分段 + fan_out（对齐训练语义）  
4. 重写 `fanout_grpo`  
5. `examples/claudecode_ags`：`generate` + `agent_runtime` + `rewards/default`(binary) + 两 env + 一条 GRPO launch  
6. 删除 Path B 与旧别名  
7. README：一条链路 + env 说明 + 如何换 reward 函数  

---

## 13. 成功标准

1. AGS + Claude Code 跑通 SWE 普通 GRPO  
2. subagent/compact 分段进训；逻辑一次任务；均分 + GRPO 广播正确  
3. 默认奖励为 binary；换奖励只改函数路径  
4. 核心无 SWE-Agent 旁路  
5. 环境变量仅两文件，无旧名映射  
6. 代码易读，无 tencent 式厚兼容堆砌  
7. 核心改动可解释：sandbox 分文件、分段 adapter/trajectory、fanout_grpo  

---

## 14. 错误处理与测试（设计约束）

### 错误处理

- 沙箱创建/执行失败：向上抛出明确错误，并确保 AGS 实例尽量清理  
- adapter thinking 解析失败等：按现网语义中止该轨迹，不把脏段写入可训练样本  
- 评测失败：`resolved=False`，binary 得 0；details 记录原因  
- 不引入「静默吞错后假装成功」的兼容分支  

### 测试重点

- `fan_out`：多段均分、共享 `rollout_id`  
- `fanout_grpo`：尝试内加总、组内 baseline、advantage 广播  
- `rewards/default`：binary 真值表  
- sandbox 工厂：`backend=ags|e2b` 分发  
- （集成）一条最小 generate 路径可用 mock/sandbox 冒烟时再补  

---

## 15. 已拍板决策一览

| 决策 | 结论 |
|---|---|
| 路径 | 只 Path A；SWE-Agent 全删 |
| 沙箱 | `sandbox.py` + `sandbox_ags.py` + BACKEND |
| 布局 | 核心可复用；examples 业务 |
| 策略 | 方案 2 |
| 沙箱内业务模块 | `agent_runtime.py` |
| 轨迹 | 逻辑一次任务；可多 Sample |
| 段奖励 / advantage | 均分 + 加总 + GRPO + 广播 |
| Reward | 可插拔；default=binary；不整迁 compose |
| 实现 | 理解后重写；不搬厚防御/多别名风格 |
| Step-GRPO | 先不做 |
| Env | 两文件；无旧名兼容 |

---

## 16. 参考来源（只读对照，非拷贝对象）

- `code/slime-tencent/examples/coding_agent_rl/`（Path A 编排与评测语义）  
- `code/slime-tencent/examples/claudecode-ags/`（启动与 env 实践）  
- `code/slime-tencent/slime/agent/{sandbox,adapters,trajectory}.py`（分段与 AGS 行为语义）  
- 官方 `code/slime/examples/coding_agent_rl/`、`slime/agent/harness/`（上游分层参考）  
