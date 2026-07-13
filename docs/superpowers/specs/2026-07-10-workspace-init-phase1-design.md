# 题目仓库初始化（第一阶段）设计

日期：2026-07-10  
分支：`feature/cc-ags-swe`  
上级设计：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)

## 目标

在 Claude Code 启动前，把沙箱里的代码仓库调到**本题对应的有 bug 基线**，再写入 `PROBLEM_STATEMENT.md`。

长期五种模式：`swebench_classic`、`swesmith`、`scaleswe`、`rebench`、`generic`。  
**本阶段只做：** `swebench_classic` + `swesmith` + rollout 侧 **git scrub**。

## 非目标（本阶段不做）

- ScaleSWE 的 `pre_commands`、Rebench 专用准备
- Timeline / 诊断落盘
- 从 source parquet 回填 `swe_smith_bug_patch`
- 评测沙箱用同一套 init（评测暂时仍用 `simple_cmd`）
- COS API 下载、tencent 厚 metadata 兼容 / `SWE_*` 别名
- 整份拷贝 `examples/coding_agent_rl/workspace_init.py`

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/workspace_init.py` | 模式判定 + reset / 切分支 / 打补丁 / scrub |
| `examples/claudecode_ags/agent_runtime.py` | `prepare_workspace` 编排 init，再写 PROBLEM |
| `examples/claudecode_ags/generate.py` | 把少量 metadata 字段传入 prepare |

不改核心 `slime/`。

## 模式判定

按任务字段顺序判断：

1. `data_source` 小写后以 `swe_smith` 开头 → **swesmith**
2. 否则存在非空 `base_commit`（顶层或 `swebench.base_commit`）→ **swebench_classic**
3. 否则 → **generic**（只 ensure agent 用户 + 写 PROBLEM；给后续阶段留钩子）

第一阶段不加其它启发式。

## 元数据字段（第一阶段）

只读这些（少做 fallback）：

| 字段 | 用途 |
|------|------|
| `workdir` | 仓库根目录（默认 `/testbed`） |
| `problem_statement` | 写入 `PROBLEM_STATEMENT.md` |
| `instance_id` | SWE-smith 分支名；abort 标签 |
| `data_source` | 模式判定 |
| `base_commit` 或 `swebench.base_commit` | SWE-bench hard reset |
| `swe_smith_bug_patch` | 可选；分支路径不够时打补丁 |

缺少 `image` / `workdir` 时，仍按现有逻辑在 `generate` 里 abort。

## 各模式行为

公共前缀：`ensure_agent_user(sb, workdir)`。

### swebench_classic

1. 在 `workdir` 执行 `git reset --hard <base_commit>`（失败 → abort）
2. Git scrub（rollout 侧）
3. 写 `PROBLEM_STATEMENT.md`

### swesmith

1. 若 `instance_id` 为合成 id（`_synthetic_row_*`）：跳过切分支；有补丁则靠补丁
2. 否则尝试切分支（尽力而为，语义对齐现网）：
   - `git fetch`（失败可忽略）
   - `git checkout <instance_id>`（或新建分支）
   - `git checkout HEAD~1`（失败可忽略）
3. 若分支路径未建立基线且提供了 `swe_smith_bug_patch`：
   - 正向补丁能干净应用则打上；否则若反向补丁能应用，视为镜像已是 bug 态；否则失败
   - `git add -A && git commit` 固化 bug 基线，便于后续 `git diff`
4. 若分支路径成功，仍按需 commit 干净基线，使 diff 相对 bug 态
5. Git scrub（rollout 侧）
6. 写 `PROBLEM_STATEMENT.md`

打补丁的辅助函数写在 `workspace_init` 本模块内（短实现），不依赖 tencent 的 `sandbox.py`。

### generic

1. Ensure agent 用户  
2. 写 PROBLEM  
（第一阶段不做 reset / scrub。）

## Git scrub（rollout 侧）

语义对齐现网意图，短实现重写：

- 删除 remotes
- 清掉非当前 HEAD 的多余本地分支 / 标签（保留干净 tip）
- 按 scrub 脚本既有逻辑，避免留下「未来历史 / 像答案」的 refs

评测侧「只删未来 commit」变体：**本阶段不做**，等评测复用同一套 init 再加。

失败策略：scrub 尽力而为（可打日志、不致命），与现网 rollout 一致；**reset / 必需的补丁应用**失败则致命。

## `prepare_workspace` API

```text
async def prepare_workspace(
    sb,
    *,
    workdir: str,
    problem_statement: str,
    instance_id: str = "",
    data_source: str = "",
    base_commit: str = "",
    swe_smith_bug_patch: str | None = None,
) -> None:
```

- 调用 `workspace_init.initialize_task_workspace(...)`
- 返回 `False` / 明确失败时：抛 `RuntimeError`（带 mode + reason）
- 成功后再以 agent 用户写 `PROBLEM_STATEMENT.md`

`generate` 扩展 `_parse_metadata` 传入上述字段；调用点仍保持一次 `prepare_workspace(...)`。

## 失败如何回到 generate

Init 失败 → 与其它硬错误同一套 abort（sample metadata 里带 `status` / reason），**不**静默继续跑 `claude`。

## 测试

放在 `tests/claudecode_ags/`：

- 模式判定表（smith / base_commit / generic）
- SWE-bench：FakeSandbox 能记录到 `git reset --hard <commit>`
- SWE-smith：会跑分支脚本；提供补丁时走补丁路径
- classic + smith 会执行 scrub；generic 不要求 scrub
- `prepare_workspace` 在 init 成功后写入 PROBLEM
- reset 失败会抛错 / 传到调用方

## 后续阶段（不在本文交付）

| 阶段 | 增加内容 |
|------|----------|
| 2 | ScaleSWE `pre_commands`（及确有必要的 root 准备） |
| 3 | Rebench / 更丰富的 generic |
| 4 | 评测沙箱共用 init；可选从 parquet 加载 bug patch |

## 成功标准

1. 带 `base_commit` 的 SWE-bench 样本在跑 CC 前完成 reset  
2. `data_source=swe_smith*` 的 SWE-smith 样本走分支和/或 bug 补丁基线  
3. 以上两种在写 PROBLEM 前都会跑 scrub  
4. 单测不依赖真实 AGS 即可通过  
5. 不新增 env 别名层；本功能不改核心 `slime/`
