# 官方 HF → slime metadata 加载时 normalize 设计

日期：2026-07-10  
分支：`feature/cc-ags-swe`  
上级设计：[CC + AGS + SWE 重构设计](./2026-07-10-cc-ags-swe-refactor-design.md)  
相关：[真判分（SWE Graders）](./2026-07-10-swe-graders-design.md)、[workspace init phase2](./2026-07-10-workspace-init-phase2-design.md)

## 目标

在 **加载 / generate 入口** 把官方 HF 行（或已接近官方字段的 dict）规范成 Path A 可用的扁平 `Sample.metadata`，使：

1. `generate` 能拿到非空 `image` / `workdir`  
2. `workspace_init` / `swe_eval.dispatch` 能直接读官方评测字段  
3. 镜像拼装与现网 DataEng `_get_ags_image_name` **同规则**；默认腾讯 TCR，**registry 前缀可配置**

## 非目标

- 离线批量造 parquet/jsonl（本轮不做；库函数可被日后 CLI 复用）  
- 兼容腾讯 `extra_info.sandbox_overrides` 壳（不读、不写）  
- 自动猜测 `dataset_type`（必须由调用方显式传入）  
- 改 `slime/` 核心 dataset loader  
- 真 AGS 拉镜像 / 构建镜像  
- Path B、Step-GRPO、复杂 reward

## 已拍板决策

| 项 | 选择 |
|----|------|
| 交付 | 加载时 normalize |
| metadata 形态 | 扁平 slime 字段（`image`、`FAIL_TO_PASS`、…） |
| 族类型 | 调用方显式 `dataset_type` |
| 镜像 | DataEng 同款公式；默认 `swebenchdocker.tencentcloudcr.com`；前缀可配 |
| 放置 | `examples/claudecode_ags/` 独立模块，由 `generate._parse_metadata` 调用 |

## 放置位置

| 路径 | 职责 |
|------|------|
| `examples/claudecode_ags/dataset_normalize.py` | `normalize_official_row`、`resolve_image`、`dataset_type` 校验 |
| `examples/claudecode_ags/generate.py` | `_parse_metadata`：若需 normalize 则调用；合并结果 |
| `tests/claudecode_ags/test_dataset_normalize.py` | 各 `dataset_type` 镜像公式 + 字段透传 |

不改 `slime/`。

## 公共 API

```text
normalize_official_row(
    row: dict[str, Any],
    *,
    dataset_type: str,
    registry: str | None = None,   # None → 读环境变量 / 默认 TCR
) -> dict[str, Any]
```

**行为：**

1. 校验 `dataset_type`（见下表；大小写不敏感；允许少量别名）  
2. 从 `row` 透传 / 规范化评测与 init 字段  
3. 计算 `image`（若 `row` 已有非空 `image` 则 **保留不覆盖**）  
4. 补默认 `workdir`、写入规范化 `data_source`  
5. 返回 **新 dict**（不原地改 `row`）

`resolve_image(row, dataset_type, registry=...) -> str` 可单独导出，便于单测。

### `dataset_type` 取值

| 规范名 | 别名（输入可接受） | 用途 |
|--------|-------------------|------|
| `swebench` | `swe-bench`, `swe_bench` | SWE-bench / 无 image 的 bench 系默认 |
| `swebench_verified` | `verified` | 与 `swebench` **同镜像公式**；仅 `data_source` 标记不同 |
| `swegym` | `swe_gym`, `swe-gym` | Gym：`__` → `_s_` |
| `swesmith` | `swe_smith`, `swe-smith` | 官方 `image_name` |
| `rebench` | `swerebench`, `swe_rebench`, `swerebenchv2` | 官方 `docker_image` / `image_name` |
| `scaleswe` | `scale_swe`, `scale-swe` | 官方 `image_url` |

未知 `dataset_type` → 抛 `ValueError`（fail fast，不静默当 swebench）。

## 镜像公式

**Registry：**

```text
registry = (显式参数)
        or SLIME_CC_IMAGE_REGISTRY
        or "swebenchdocker.tencentcloudcr.com"
```

去掉末尾 `/`。最终 `image` 一律 `.lower()`（与 DataEng 一致）。

| `dataset_type` | 规则 |
|----------------|------|
| `swebench` / `swebench_verified` | `{registry}/swebench/sweb.eval.x86_64.{id}:latest`，`instance_id` 中 `__` → `_1776_` |
| `swegym` | 同上，但 `__` → `_s_` |
| `swesmith` | 取 `image_name`；若含 `/` 则去掉第一段 owner（如 `jyangballin/`）；无 tag 则补 `:latest`；结果 `{registry}/swebench/{path}` |
| `rebench` | 优先 `docker_image`，否则 `image_name`；取最后一段 `repo:tag`（去掉 `docker.io/` 等前缀路径）；结果 `{registry}/swerebenchv2/{repo:tag}` |
| `scaleswe` | 取 `image_url` 的 tag（`:` 后）；若无 `:` 则用 `instance_id`；结果 `{registry}/aweaiteam/scaleswe:{tag}` |

**已有 `image`：** 非空则原样保留（调用方 / 上游已写死镜像时不二次改写）。

**缺关键字段：**  
- smith 无 `image_name`、scaleswe 无 `image_url` 且无可用 tag、rebench 无 `docker_image`/`image_name` → `image=""`，由 `generate` 现有 `missing_image_or_workdir` 中止（normalize 本身可记 `details` 或仅返回空，不抛，避免批量加载中断；单测覆盖空串路径）。

> 说明：现网 DataEng 对 rebench/scaleswe 会把官方名 **改挂到 TCR 路径**；本设计同样改挂到 `{registry}/...`，而不是保留 `docker.io/swerebench/...` 原串。换 registry 时只改前缀。

## 字段映射（输出扁平 metadata）

所有类型公共输出（有则写，无则省略或空串按下列默认）：

| 输出键 | 来源 / 规则 |
|--------|-------------|
| `dataset_type` | 规范化后的规范名 |
| `data_source` | 见下表（供 workspace_init：smith 必须 `swe_smith*` 前缀） |
| `instance_id` | `row.instance_id` |
| `problem_statement` | `row.problem_statement`（可作 prompt） |
| `image` | 上节公式或已有 `image` |
| `workdir` | `row.workdir` 或默认 `/testbed`（Scale-SWE 官方常自带） |
| `repo` | `row.repo` |
| `base_commit` | `row.base_commit` 或 `row.parent_commit`（Scale-SWE） |
| `version` | `row.version` |
| `FAIL_TO_PASS` / `PASS_TO_PASS` | 原样透传（list 或 JSON 字符串均可；grader 侧已有 `parse_list`） |
| `test_patch` | `row.test_patch` |
| `patch` | `row.patch`（保留；smith 另见 bug patch） |

### 按类型附加

| 类型 | 附加 |
|------|------|
| `swesmith` | `swe_smith_bug_patch` ← `row.patch`（若尚未有 `swe_smith_bug_patch`）；`data_source` 默认 `swe_smith` |
| `scaleswe` | `pre_commands`, `f2p_script`, `f2p_patch`, `parent_commit`；`image_url` 可保留原文备查 |
| `rebench` | `install_config`（dict 原样）, `log_parser`（若在顶层则保留）；`docker_image`/`image_name` 可保留原文备查 |
| `swebench*` / `swegym` | `environment_setup_commit`, `hints_text` 可选透传 |

### `data_source` 默认

| `dataset_type` | 默认 `data_source` |
|----------------|-------------------|
| `swesmith` | `swe_smith` |
| `scaleswe` | `scaleswe` |
| `rebench` | `swerebench` |
| `swegym` | `swegym` |
| `swebench` | `swebench` |
| `swebench_verified` | `swebench_verified` |

若 `row` 已有非空 `data_source`，**保留不覆盖**（显式优先）。

## 与 `generate` 的衔接

```text
Sample.metadata (官方行字段 + 可选已填 image)
  + 调用约定：metadata["dataset_type"] 必须存在
        ↓
_parse_metadata(sample)
  → 若缺 image 或显式要求 normalize：
       md = normalize_official_row(sample.metadata, dataset_type=...)
  → 再做现有解析（bug_patch 回填、pre_commands 规范化等）
        ↓
agent / eval 使用扁平 md
```

**触发条件（推荐，写死一种）：**

- 只要 `metadata` 含非空 `dataset_type`，**始终**先走 `normalize_official_row`（幂等：已有 `image` 不改；字段只补不删）。  
- 若无 `dataset_type`：保持现状（假定已是 slime 扁平行，如手写 `eval_cmd` fixture）；不强制 normalize。

这样：官方 HF 行必须带 `dataset_type`（由数据配置 / 加载脚本注入一行字段即可）；单测与 simple_cmd 样例可不带。

**注入 `dataset_type` 的责任：** 本设计不实现通用 HF DataLoader；约定训练配置或薄包装在读入后写入该键（例如 jsonl 转换时加一列，或 launch 里按文件绑定）。normalize 模块本身只消费该键。

## 环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `SLIME_CC_IMAGE_REGISTRY` | 镜像 registry 前缀（无方案名） | `swebenchdocker.tencentcloudcr.com` |

仅 `SLIME_*`；无 `SWE_*` / `VERL_*` 别名。

## 错误与边界

| 情况 | 行为 |
|------|------|
| 未知 `dataset_type` | `ValueError` |
| 公式所需字段缺失 | `image=""`；其它字段照常；generate 侧 abort |
| `instance_id` 缺失（bench/gym 公式） | `image=""` |
| 输入非 dict | `TypeError` |
| 已是扁平且含 `image` | 保留 `image`；仍补齐缺省键 |

## 测试

`tests/claudecode_ags/test_dataset_normalize.py`：

1. **镜像真值表**（registry 默认 TCR）：  
   - swebench：`astropy__astropy-12907` → `.../sweb.eval.x86_64.astropy_1776_astropy-12907:latest`  
   - swegym：`getmoto__moto-7365` → `.../getmoto_s_moto-7365:latest`  
   - swesmith：官方 `jyangballin/swesmith.x86_64....` → 挂到 `{registry}/swebench/swesmith.x86_64....:latest`  
   - rebench：`docker.io/swerebenchv2/behat-gherkin:343-e522894` → `{registry}/swerebenchv2/behat-gherkin:343-e522894`  
   - scaleswe：`aweaiteam/scaleswe:auth0_...` → `{registry}/aweaiteam/scaleswe:auth0_...`
2. **自定义 registry**：`SLIME_CC_IMAGE_REGISTRY=example.registry` 时前缀替换。  
3. **已有 `image` 不覆盖**。  
4. **字段透传**：scaleswe 含 `pre_commands`/`f2p_script`；rebench 含 `install_config`；smith 含 `swe_smith_bug_patch`。  
5. **`data_source`**：smith → `swe_smith`；显式 `data_source` 优先。  
6. **未知 type → ValueError**。  
7. **`generate` 接线（可选轻测）**：metadata 带 `dataset_type`+官方字段、无 `image` 时，`_parse_metadata` 后 `image` 非空。

不依赖真 AGS / 真拉镜像。

## 成功标准

1. 六类 `dataset_type` 均可从官方样例行得到可用扁平 metadata（含正确 `image` 公式）。  
2. 默认 registry 为腾讯 TCR；改 `SLIME_CC_IMAGE_REGISTRY` 只影响前缀。  
3. 无 `dataset_type` 的现有单测 / simple_cmd 路径行为不变。  
4. 不引入腾讯 `sandbox_overrides` 壳；不改 `slime/` 核心。  
5. `tests/claudecode_ags/` 全绿。

## 后续可选项（非本设计必交付）

- 离线 CLI：批量读 HF → 写 jsonl（内部调用同一 `normalize_official_row`）  
- 按文件路径自动填 `dataset_type` 的 launch 辅助  
- 官方镜像名「不改挂、只原样使用」的开关（若某环境无 TCR 镜像副本）
