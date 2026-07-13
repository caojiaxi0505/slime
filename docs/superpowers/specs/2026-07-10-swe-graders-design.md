# 真判分（SWE Graders）设计

日期：2026-07-10  
分支：`feature/cc-ags-swe`  
上级设计：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
相关：[题目仓库初始化 phase2](./2026-07-10-workspace-init-phase2-design.md)

## 目标

在沙箱里对模型产出的 diff **跑测并解析日志**，得到 `resolved`，供默认 binary 奖励使用。

覆盖三种评测路径（一次交付、分文件实现）：

1. **SWE-bench 系**（Verified / SWE-bench / SWE-Gym / 无 `test_cmd` 的 smith 类）  
2. **Scale-SWE**  
3. **SWE-rebench**  

官方数据集已落在 `/mnt/sn-007/jiaxicao/datasets/`（HF 下载，非现网 pass@k 拷贝）。**官方集均无现成 `eval_cmd`**，必须拼命令。

## 非目标

- linear / sqrt 等复杂奖励 shaping（仍用 `rewards/default` binary）  
- 整迁 tencent `eval_cmd_builder` / `sandbox.evaluate_with_details` 厚壳  
- error2task、现网内部 parquet 字段兼容层  
- 构建/拉取 Docker 镜像本身  
- 真 AGS E2E（单测以离线 grade + FakeSandbox 为主）

## Binary 与 `resolved` 约定

**`rewards/default.compose`：**

```text
resolved == True  →  1.0
resolved == False →  0.0
```

**`resolved` 由 grader 给出**（不是看 shell 退出码）：

```text
F2P pass_ratio >= SLIME_CC_REWARD_F2P_THRESHOLD   # 默认 1.0
且
P2P pass_ratio >= SLIME_CC_REWARD_P2P_THRESHOLD   # 默认 0.99
```

两者都达标 → `resolved=True`；否则 `False`。  
（与现网语义对齐；阈值只读 `SLIME_*`，不留 `VERL_*` 别名。）

## 三层结构

```text
metadata
  → 1) resolve_eval_plan(metadata)     # 拼命令 + 需打的 test/f2p 补丁
  → 2) run_eval(sb, plan, model_diff) # 共用 workspace init 后执行
  → 3) grade_logs(stdout/stderr, …)   # 解析 → pass_ratio → resolved
  → EvalResult(resolved, applied_cleanly, details)
```

`simple_cmd` **保留**：仅当 metadata 已带非空 `eval_cmd` 且无法走结构化 F2P/P2P 时作 fallback（退出码 0 ⇒ resolved）；主路径不依赖它。

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/swe_eval/base.py` | `EvalResult`（已有） |
| `examples/claudecode_ags/swe_eval/cmd_resolve.py` | 按模式拼 `EvalPlan` |
| `examples/claudecode_ags/swe_eval/grade_common.py` | F2P/P2P 比率、阈值、`resolved` |
| `examples/claudecode_ags/swe_eval/swebench.py` | SWE-bench 系：跑测 + swebench/pytest 解析 |
| `examples/claudecode_ags/swe_eval/scaleswe.py` | Scale-SWE：f2p_script/patch + pytest |
| `examples/claudecode_ags/swe_eval/rebench.py` | rebench：`test_cmd` + `log_parser` |
| `examples/claudecode_ags/swe_eval/dispatch.py` | 模式选择 + 统一 `evaluate(...)` |
| `examples/claudecode_ags/generate.py` | `_evaluate_diff` 改走 `dispatch` |

不改 `slime/` 核心。

## 模式判定（评测侧）

与 workspace 模式对齐，优先读官方字段：

1. 有非空 `pre_commands` 或 `f2p_script` / `f2p_patch`（Scale-SWE 形态）→ **scaleswe**  
2. 有 `install_config.test_cmd`（或等价）→ **rebench**  
3. 有 `FAIL_TO_PASS`/`PASS_TO_PASS` + `repo`（SWE-bench / Gym / Verified / 多数 smith）→ **swebench**  
4. 已有 `eval_cmd` 且上面都不匹配 → **simple_cmd**  
5. 否则 → 无法评测：`resolved=False`，`details.reason=missing_eval_plan`

（`data_source` 若存在可作为辅助，但不依赖现网内部命名。）

## 各模式：拼命令与跑测

### 公共跑测步骤

1. `initialize_task_workspace(..., rollout_side=False)`（phase2 已有）  
2. 应用 **模型 diff**（失败 → `applied_cleanly=False`，`resolved=False`）  
3. 按 plan 应用 **test_patch / f2p_patch / 写入 f2p_script**（失败记入 details；是否继续以各模式约定为准，默认失败则 unresolved）  
4. 执行 `eval_cmd`（或脚本块），收集 stdout/stderr  
5. `grade_logs` → `EvalResult`

### swebench

**拼命令（克制版）：**

- 需要：`repo`、`version`（可选）、`test_patch`、`FAIL_TO_PASS`/`PASS_TO_PASS`、`workdir`  
- 激活测试环境（conda `testbed` / `.venv` 等，短脚本，少兼容分支）  
- 应用官方 `test_patch`  
- 测试命令：优先从 swebench 仓库规格解析；若未安装 `swebench` 或规格缺失 → 明确失败（不静默 0 分装成功）  
- SWE-smith：无 `test_patch`/`base_commit` 时，以镜像内已有测试 + `FAIL_TO_PASS` 列表为准；命令侧用 pytest 节点或镜像约定（实现时按官方 `image_name` 语义写最短路径）

**解析：**

- 优先 `swebench.harness.log_parsers.MAP_REPO_TO_PARSER`  
- 合并 SWE 风格 pytest 行（`PASSED path::test` / `path::test PASSED`）  
- 算 F2P/P2P pass_ratio → 阈值 → `resolved`

### scaleswe

**拼命令：**

- 将 `f2p_script` 写为仓库下 `test_fail_to_pass.py`（若有）  
- 若有 `f2p_patch` 则先 apply  
- 跑 pytest：针对 `FAIL_TO_PASS` 列表（及脚本）；可用短 runner，避免整迁巨型 scaleswe_eval  

**解析：** pytest 日志 → F2P/P2P 比率 → `resolved`  
（`pre_commands` 已在 workspace init 执行，评测侧不重复除非 init 未跑。）

### rebench

**拼命令：**

- `install_config.test_cmd`（str 或 list 拼接）  
- 环境激活与 swebench 共用短 helper  

**解析：**

- 使用 `install_config.log_parser`（或顶层 `log_parser`）  
- 解析器：优先本地实现的稳健 pytest 解析；多语言 parser 通过可选依赖 `SLIME_REBENCH_ROOT`（SWE-rebench-V2）加载，缺失时仅支持 pytest 类并打日志  
- F2P/P2P 比率 → `resolved`

## `EvalPlan` / `EvalResult`

```text
EvalPlan:
  mode: scaleswe | rebench | swebench | simple_cmd
  eval_cmd: str
  workdir: str
  test_patch: str = ""
  f2p_patch: str = ""
  f2p_script: str = ""
  fail_to_pass: list[str]
  pass_to_pass: list[str]
  repo: str = ""
  log_parser: str = ""
  details: dict

EvalResult:  # 已有
  resolved: bool
  applied_cleanly: bool
  details: dict  # 含 pass_ratio、parser、stdout/stderr 截断等
```

## 与 generate 的衔接

`_evaluate_diff`：

```text
make_sandbox(image)
  → initialize_task_workspace(fields, rollout_side=False)
  → swe_eval.dispatch.evaluate(sb, metadata=..., diff_text=..., timeout=...)
  → EvalResult
```

`base_eval = {"resolved": ..., **details}` → `rewards.default.compose` → 标量 R。

## 官方数据字段对照（实现时只认这些）

| 来源 | 关键字段 |
|------|----------|
| SWE-bench* / Gym | `repo`, `base_commit`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `version`, `problem_statement` |
| SWE-smith | `repo`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `image_name`, `patch`, `problem_statement` |
| SWE-rebench | `install_config`（含 `test_cmd`,`log_parser`）, `FAIL_TO_PASS`, `PASS_TO_PASS`, `docker_image`/`image_name`, `test_patch` |
| Scale-SWE | `workdir`, `image_url`, `pre_commands`, `f2p_script`, `f2p_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `parent_commit` |

训练管线若要把官方行变成 slime `Sample.metadata`，另做薄 adapter（本设计假定 metadata 已具备上表字段；adapter 可同阶段最小实现或紧随其后）。

## 测试

`tests/claudecode_ags/`：

- `grade_common`：阈值边界（F2P=1.0 / P2P=0.99）真值表  
- swebench / scaleswe / rebench：固定 stdout fixture → `resolved`  
- `cmd_resolve`：各模式样例 metadata → 命令含关键片段（pytest / test_cmd / f2p 文件名）  
- dispatch：模式选择表  
- 不依赖真 AGS；`swebench` / rebench 外部包缺失时的失败路径有断言  

## 成功标准

1. 三种模式均可从官方字段得到 `EvalPlan` 并产出 `resolved`  
2. binary：`resolved` ↔ `{0,1}`；阈值默认同现网  
3. 已有 `eval_cmd` 时 simple_cmd 仍可用  
4. 单测全绿；无 `VERL_*` 别名  

## 后续可选项（不在本文必交付）

- 官方 HF → slime jsonl/parquet 的批量转换脚本  
- 真 AGS + 官方镜像冒烟  
- 多语言 rebench parser 打包进依赖，而非 `SLIME_REBENCH_ROOT`
