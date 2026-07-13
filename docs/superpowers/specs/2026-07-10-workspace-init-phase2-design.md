# 题目仓库初始化（第二阶段）设计

日期：2026-07-10  
分支：`feature/cc-ags-swe`  
上级设计：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
上一阶段：[题目仓库初始化（第一阶段）](./2026-07-10-workspace-init-phase1-design.md)

## 目标

在第一阶段（SWE-bench reset / SWE-smith 分支与补丁 / agent 侧重 scrub）之上，补齐仓库侧剩余能力：

1. **ScaleSWE**：按 `pre_commands` 把仓库调到本题起点  
2. **Rebench**：模式可识别，仓库侧与 generic 同级（为后续真判分留钩子）  
3. **评测沙箱共用 init**：评测前走同一套初始化；scrub 分 agent / eval 两档  
4. **parquet 回填 `swe_smith_bug_patch`**：metadata 缺补丁时从源 parquet 按题取回  

本阶段仍只做「把考场摆好」，**不做** ScaleSWE / Rebench / SWE-bench 的真判分 grader。

## 非目标

- 真判分（pytest runner、F2P/P2P 解析、rebench log parser 等）  
- 整迁 tencent `normalize_parquet_sample` / `scaleswe_eval` 全文件  
- 新增 `SWE_*` / `VERL_*` 别名层  
- Step-GRPO、Path B  

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/workspace_init.py` | 扩展模式、ScaleSWE pre_commands、双档 scrub、`rollout_side` |
| `examples/claudecode_ags/bug_patch_source.py` | 从 parquet 回填 `swe_smith_bug_patch`（短实现） |
| `examples/claudecode_ags/agent_runtime.py` | `prepare_workspace` 传入 `pre_commands` / `install_config` 等 |
| `examples/claudecode_ags/generate.py` | metadata 解析、parquet 回填、`_evaluate_diff` 先 init 再评测 |

不改核心 `slime/`（除非已有工厂无需改动）。

## 模式判定（更新）

按顺序：

1. `data_source` 小写以 `swe_smith` 开头 → **swesmith**  
2. 存在非空 `pre_commands` → **scaleswe**  
3. 存在非空 `base_commit`（顶层或 `swebench.base_commit`）→ **swebench_classic**  
4. `install_config` 含非空 `test_cmd`（str 或 list）→ **rebench**  
5. 否则 → **generic**  

第一阶段已有的 1 / 3 / 5 保持语义；本阶段插入 2 与 4。

## 元数据字段（本阶段新增/明确）

| 字段 | 用途 |
|------|------|
| `pre_commands` | ScaleSWE 前置命令（str 或 list；需做 `\\n` 等转义还原） |
| `install_config` | dict；若为 JSON 字符串则解析；用于 Rebench 判定 |
| `data_path` | 源 parquet 路径（回填 bug patch） |
| `cc_source.data_path` | 同上，优先于顶层 `data_path` 亦可 |
| `cc_source.instance_index` | 可选行号，加速定位 |
| （沿用）`swe_smith_bug_patch`、`instance_id`、`data_source`、`base_commit`、`workdir` | 同 phase1 |

## 各模式行为（相对 phase1 的增量）

公共前缀仍为：`ensure_agent_user(sb, workdir)`。

### scaleswe（新）

1. 还原并执行 `pre_commands`（fail-fast；失败 → init 失败）  
2. 必要的 root 可写准备（现网：若存在 `/workspace/esp-idf/examples` 则 `chmod -R a+w`，失败可忽略）  
3. scrub（见下节 `rollout_side`）  
4. 由 `prepare_workspace` 写 `PROBLEM_STATEMENT.md`（agent 路径）  

`pre_commands` 执行方式对齐现网意图：在 workdir 下以 heredoc / 脚本块跑，失败则 abort。

### rebench（新）

1. Ensure agent 用户  
2. **不做** reset / pre_commands / scrub（与 generic 相同）  
3. 模式枚举存在，便于日志与后续 grader 分支  

### swebench_classic / swesmith

逻辑同 phase1；scrub 改为显式依赖 `rollout_side`（见下）。

### generic

同 phase1：只 ensure user（+ agent 路径写 PROBLEM）。

## Git scrub 两档

| 场景 | `rollout_side` | 行为 |
|------|----------------|------|
| Agent 沙箱（跑 CC） | `True` | **重 scrub**：去 remote、清多余 refs、orphan commit、gc（phase1 已有） |
| 评测沙箱 | `False` | **轻 scrub**：把各 ref 指到当前 HEAD、清 stash/reflog 等，对齐现网「remove future commits」语义，避免 orphan 重写过狠 |

对 **scaleswe / swebench_classic / swesmith**：init 末尾按 `rollout_side` 选档。  
对 **rebench / generic**：本阶段仍不 scrub。

## 评测沙箱共用 init

`_evaluate_diff`（或等价路径）在 `simple_cmd.evaluate` 之前：

```text
async with make_sandbox(image) as sb:
    ok = await initialize_task_workspace(fields, rollout_side=False)
    if not ok: → 视为未解决 / abort 评测（与 generate 约定一致，不静默当 resolved）
    return await simple_cmd.evaluate(...)  # apply diff + eval_cmd，判分逻辑暂不变
```

评测侧**不写** `PROBLEM_STATEMENT.md`（除非后续需要）；只做仓库基线。

Agent 路径：`prepare_workspace` → `initialize_task_workspace(..., rollout_side=True)` → 写 PROBLEM。

## parquet 回填 bug patch

时机：`generate` 解析 metadata 之后、`prepare_workspace` 之前。

条件：`data_source` 为 swesmith（或已判定将走 swesmith），且 `swe_smith_bug_patch` 为空。

步骤（克制版）：

1. 取路径：`cc_source.data_path` 或 `data_path`；无路径则跳过  
2. 读 parquet（可对路径做小缓存，避免同文件反复全表扫描）  
3. 优先 `instance_index` 行，否则按行匹配 `instance_id`  
4. 从该行 `reward_model`（或现网等价字段）取出 gold / bug patch 字符串  
5. 填入本次使用的 metadata / `TaskFields.swe_smith_bug_patch`  

读失败：打日志并继续（若仍无补丁且分支也失败，则由 swesmith init 失败路径 abort）。  
不在本阶段做完整 parquet→sample 字段归一化。

## API 调整要点

```text
initialize_task_workspace(sb, fields, *, rollout_side: bool = True) -> bool

prepare_workspace(..., pre_commands=..., install_config=..., rollout_side=True)

# generate
patch = resolve_swe_smith_bug_patch(md)  # bug_patch_source
_evaluate_diff(..., fields / metadata, rollout_side=False)
```

`TaskFields` 增加：`pre_commands`、`install_config`。

## 测试

`tests/claudecode_ags/`：

- 模式判定：有 `pre_commands` → scaleswe；有 `test_cmd` → rebench；与 smith/base_commit 优先级  
- ScaleSWE：FakeSandbox 能见到 pre_commands 脚本内容；失败返回 False  
- scrub：`rollout_side=True` 含重 scrub marker；`False` 含轻 scrub 特征、不含 orphan 重 scrub  
- `_evaluate_diff`（或包装函数）在 evaluate 前调用 init  
- parquet：临时 `.parquet` fixture，缺 patch 时能回填；无路径时不炸  

## 成功标准

1. ScaleSWE 样本在跑 CC / 评测前执行 `pre_commands`  
2. Rebench 可被判定，仓库侧不误走 reset/pre_commands  
3. 评测沙箱与 agent 共用 init，scrub 分档正确  
4. swesmith 缺补丁时可从 parquet 回填  
5. 单测不依赖真实 AGS；不引入真判分大模块  

## 与后续工作的边界

真判分（ScaleSWE pytest / Rebench parser / SWE-bench F2P）单开阶段，不在本文交付。
