# Notes: SWE-Gym Phase G gold filter results

Date: 2026-07-11  
Operator decision: **1404 kept tasks are enough** — skip Phase P (8× pass@k / drop-always-resolved) for now; use this set as the GRPO train candidate pool baseline.

## 结果落盘位置（绝对路径）

```text
/tmp/swegym_gold_20260711_042217/
├── summary.json       # {"n_kept":1404,"n_excluded":1034,"n_tasks":2438,...}
├── kept.jsonl         # 1404 行 — GRPO 候选池（每行含 instance_id + metadata）
├── excluded.jsonl     # 1034 行 — 被排除的题
├── runs.jsonl         # 6661 行 — 每次 gold attempt 记录
├── tasks.jsonl        # 每题汇总
├── progress.log       # 运行进度日志
└── meta.json          # 本次 run 配置（concurrency/timeouts/data-path）
```

主文件：
- **候选池**：`/tmp/swegym_gold_20260711_042217/kept.jsonl`
- **汇总**：`/tmp/swegym_gold_20260711_042217/summary.json`

## Run

| Field | Value |
|-------|-------|
| Dataset | SWE-Gym train |
| Path | `/mnt/sn-007/jiaxicao/datasets/SWE-Gym/data/train-00000-of-00001.parquet` |
| `dataset_type` | `swegym` |
| Phase | G only (gold ×4) |
| Concurrency | 16 (task-level) |
| Eval timeout | 600s |
| AGS timeout | 30m |
| Out dir | `/tmp/swegym_gold_20260711_042217` |
| Finished | `2026-07-11T08:10:15Z` (resume run meta) |

First attempt died mid-run (BrokenPipe / stdout tee under heavy AGS logs; no `summary.json`). Resume with hardened runner (`return_exceptions=True`, SIGPIPE ignored, `progress.log`) completed successfully.

**Storage note:** Results were originally under `/tmp/...`, then briefly under `code/eval_runs/`, and on 2026-07-11 were moved to the permanent path above under the **slime** tree (`/mnt/sn-007/jiaxicao/code/slime/eval_runs/...`) to avoid accidental tmp cleanup.

## Results

| Metric | Count |
|--------|------:|
| Total tasks | 2438 |
| **Kept (4/4 gold `resolved=True`)** | **1404** |
| Excluded | 1034 |
| Attempts written (`runs.jsonl`) | 6661 |

### Exclude-reason breakdown

| Reason | Count | Note |
|--------|------:|------|
| `gold_unresolved` | 952 | Genuine gold eval fail — do not re-admit |
| `infra_timeout` | 44 | ~600s; optional retry later |
| `infra_image_missing` | 37 | TCR image missing — need image publish |
| `infra_broken_pipe` | 1 | Runner/stdout pipe; optional retry |

Quota/rate-limit strings (`TooManyRequests`, `ResourceLimit`, etc.): **0**. See also agent analysis of exclusions — no AGS quota false positives among the 952 unresolved.

### Kept repo families (top)

| Family (from `instance_id`) | Count |
|-----------------------------|------:|
| Project-MONAI | 323 |
| getmoto | 253 |
| python (mypy) | 253 |
| iterative (dvc) | 170 |
| dask | 133 |
| pandas-dev | 105 |
| conan-io | 64 |
| modin-project | 52 |
| facebookresearch | 31 |
| bokeh | 20 |

## Decision

- **1404 kept tasks are sufficient** for the next training path.
- **Phase P deferred** (8× live Claude Code / drop-always-resolved). Not required given current pool size.
- Next when ready: use `kept.jsonl` as the starting pool for GRPO `PROMPT_DATA` (may still subset further).

## Key artifact paths

```text
/tmp/swegym_gold_20260711_042217/
  summary.json
  kept.jsonl          # 1404 rows — GRPO candidate pool
  excluded.jsonl      # 1034 rows
  runs.jsonl          # per-attempt records
  tasks.jsonl
  progress.log
  meta.json
```
