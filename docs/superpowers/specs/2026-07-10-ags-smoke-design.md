# 真 AGS 冒烟设计

日期：2026-07-10  
分支：`feature/cc-ags-swe`  
上级设计：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
相关：[HF metadata normalize](./2026-07-10-hf-metadata-normalize-design.md)、[SWE Graders](./2026-07-10-swe-graders-design.md)

## 目标

在**真实腾讯 AGS**上跑一条最短、可人工触发的冒烟路径，验证 Path A 已落地代码在真环境能通，而不是只靠 FakeSandbox 单测。

冒烟回答的问题：

1. AGS 凭证 / SWE-ReX / `make_sandbox` 能否创建并销毁实例？  
2. normalize 拼出的镜像能否被 AGS 拉起？  
3. `workspace_init` + `swe_eval.dispatch` 在真容器里能否跑完并给出 `resolved`？

**默认不跑完整 Claude Code ↔ adapter ↔ SGLang 训练链路**（那是更重的 L2，见分层）。

## 非目标

- 接入 CI / 默认 `pytest` 必跑（无凭证环境必须跳过）  
- 多题回归、pass@k、训练 GRPO  
- 构建 / 推送 Docker 镜像到 TCR  
- 离线造全量 parquet  
- Path B、Step-GRPO  
- 替代现网 `host_reachability/08_ags_eval_smoke.py` 的全量 CC 冒烟（可后续对齐）

## 分层（本设计交付范围）

| 层 | 内容 | 本轮 |
|----|------|------|
| **L0** | 创建沙箱 → `exec` 简单命令（如 `echo ok` / `pwd`）→ 销毁 | **必做** |
| **L1** | 官方一行 → normalize → init（`rollout_side=False`）→ 应用 **gold `patch`** → `dispatch.evaluate` → 打印 `resolved` | **必做** |
| **L2** | 再开沙箱装 toolchain → 跑 Claude Code（需 `SLIME_ADAPTER_PUBLIC_URL` + 可达 adapter）→ `git_diff` → 新沙箱 eval | **可选 / 后续** |

推荐审阅默认：**只交付 L0 + L1 脚本**；L2 在文档中留接口与前置条件，不作为本轮成功标准。

理由：L1 已覆盖「镜像 + init + grader」这条训练奖励路径的核心；L2 额外依赖公网/ALB adapter 与模型服务，失败面更大，适合单独一轮。

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/smoke/ags_smoke.py` | CLI：`--level 0|1`（可选日后 `2`） |
| `examples/claudecode_ags/smoke/README.md` | 前置条件、环境变量、示例命令、如何读输出 |
| `tests/claudecode_ags/test_ags_smoke_script.py` | **仅**静态/帮助/参数解析或 dry 导入；**不**连真 AGS |

不改 `slime/` 核心（沿用已有 `sandbox_ags` / `make_sandbox`）。

## 前置条件（人工）

1. 已配置 AGS（推荐 `SLIME_AGENT_AGS_ENV_FILE`，或 `slime_ags.env` + 密钥文件）。  
2. `SLIME_AGENT_SANDBOX_BACKEND=ags`。  
3. `SLIME_AGENT_AGS_SWE_REX_ROOT` 指向可用 SWE-ReX（若未 pip 安装）。  
4. 所选实例的镜像在 AGS 侧可拉取（默认 TCR：`SLIME_CC_IMAGE_REGISTRY`，与 normalize 一致）。  
5. L1 需要本机可读的官方数据一行（HF 目录或单行 json）。  
6. L2（若做）另需：COS mount / toolchain、`SLIME_ADAPTER_PUBLIC_URL` 从沙箱可达、adapter+模型已起。

缺凭证时脚本应 **清晰报错退出**（非 0），而不是挂起。

## 默认样例（可 CLI 覆盖）

| 项 | 默认建议 | 说明 |
|----|----------|------|
| 数据集 | `swebench_verified` | 有 `base_commit` + `test_patch` + gold `patch`，L1 可验证 `resolved` |
| 数据路径 | `/mnt/sn-007/jiaxicao/datasets/SWE-bench_Verified/...` 或 `--row-json` | 由 CLI 指定；不写死不可移植绝对路径为唯一入口 |
| 选题 | `--instance-id` 或文件中第一行 | 冒烟一题即可 |
| `dataset_type` | CLI 必填或与默认一致 | 交给 `normalize_official_row` |

也允许 `--dataset-type scaleswe|swesmith|rebench|...` 换族，但文档示例以 Verified 为准。

**Gold 语义（L1）：** 使用官方行的 `patch`（修复补丁）作为 `diff_text` 交给 `dispatch.evaluate`。期望在环境正常时 `resolved=True`（F2P/P2P 阈值默认 1.0 / 0.99）。若某题 gold 在当前镜像上不稳定，允许 `--allow-unresolved` 只检查「跑通且返回 EvalResult」，但默认应断言 `resolved`。

## CLI 草图

```bash
# L0
python -m examples.claudecode_ags.smoke.ags_smoke \
  --level 0 \
  --image "$IMAGE"   # 或 --dataset-type swebench_verified --row-json one.json

# L1
python -m examples.claudecode_ags.smoke.ags_smoke \
  --level 1 \
  --dataset-type swebench_verified \
  --data-path /path/to/SWE-bench_Verified/data/test-*.parquet \
  --instance-id astropy__astropy-12907
```

行为摘要：

1. 加载 env（`SLIME_AGENT_AGS_ENV_FILE` / 可选 `--env-file`）。  
2. 读一行官方数据 → `normalize_official_row(..., dataset_type=...)` → 得到 `image` 等。  
3. **L0：** `async with make_sandbox(image)` → `exec` → 打印 sandbox_id / 输出 → 退出码 0。  
4. **L1：**  
   - 沙箱 A：`initialize_task_workspace(..., rollout_side=False)`  
   - `dispatch.evaluate(sb, metadata=..., diff_text=gold_patch, ...)`  
   - 打印 `resolved` / `applied_cleanly` / `details` 摘要（截断 stdout）  
   - 默认 `resolved` 为假则 exit 1  
5. 任意层用 `try/finally` 或 context manager 保证销毁。

**注意：** L1 按现网习惯可用**同一沙箱**先 init 再 evaluate（`dispatch` 内再 apply gold）；不必强行「双沙箱」，除非后续对齐 generate 的「agent 沙箱 / eval 沙箱」分离。本设计 **L1 单沙箱** 即可（更短）；在 README 注明与 `generate` 双沙箱的差异。

## 与现有代码的关系

```text
ags_smoke.py
  → dataset_normalize.normalize_official_row
  → slime.agent.sandbox.make_sandbox   # backend=ags
  → workspace_init.initialize_task_workspace
  → swe_eval.dispatch.evaluate
  →（不调用 generate() 全链路，避免拖入 adapter singleton）
```

刻意 **不** 调用 `generate()`：冒烟应可在无 SGLang 时跑 L0/L1。

## 输出与成功标准

**stdout 建议前缀：** `[ags-smoke]`，含 `level`、`instance_id`、`image`、`sandbox_id`、`resolved`。

**本轮成功标准：**

1. 在有凭证的环境，L0 对一可拉镜像 exit 0。  
2. 同一环境，L1 对默认 Verified 一题（或文档指定一题）`resolved=True`（或文档记录的已知例外 + `--allow-unresolved`）。  
3. 无凭证时本地 `pytest` 仍全绿（冒烟不进默认 suite）。  
4. README 写清前置与示例命令。

## 风险与说明

| 风险 | 缓解 |
|------|------|
| TCR 镜像拉取失败 / 权限 | 文档写明 registry；允许 `--image` 覆盖 |
| 单题 gold 不稳定 | `--instance-id` 可换；`--allow-unresolved` |
| AGS 配额 / 启动慢 | 打印 boot 耗时；超时用已有 `SLIME_AGENT_AGS_*_TIMEOUT` |
| 与 generate 双沙箱不一致 | L1 单沙箱；L2/后续再对齐 |

## 审阅时请确认的两点

1. **本轮是否只做 L0+L1**（推荐），L2 明确延后？  
2. **L1 默认数据集是否 SWE-bench Verified + gold `patch`**（推荐），还是改 Scale-SWE / smith？

确认后进入实现计划与编码。
