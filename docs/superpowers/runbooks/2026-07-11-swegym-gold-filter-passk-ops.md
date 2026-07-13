# SWE-Gym Gold 筛选 + 8 次轨迹剔全对 — 操作手册

日期：2026-07-11（Phase P 完成于 2026-07-12）  
设计：[2026-07-11-swegym-gold-filter-passk-design.md](../specs/2026-07-11-swegym-gold-filter-passk-design.md)  
计划：[2026-07-11-swegym-gold-filter-passk.md](../plans/2026-07-11-swegym-gold-filter-passk.md)  
**完整结果笔记（推荐阅读）：** [2026-07-12-swegym-filter-passk-results.md](../notes/2026-07-12-swegym-filter-passk-results.md)

## 结论速览

```text
SWE-Gym 2438 → Phase G → 1404 → Phase P → 1394 train_candidates（剔 10 道 8/8）
```

进训主文件：

`/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_candidates.jsonl`

## 前置

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
source examples/claudecode_ags/env/load_env.sh
# Phase P 需要 SLIME_ADAPTER_PUBLIC_URL
```

默认超时：`--time-budget 1800`、`--eval-timeout 600`；建议 `SLIME_AGENT_AGS_TIMEOUT` ≥ 45m。  
结果统一落在：`/mnt/sn-007/jiaxicao/code/slime/eval_runs/<run_name>/`（勿用 `/tmp`）。

## Phase G — gold ×4（已完成）

```bash
.venv/bin/python -m examples.claudecode_ags.eval.swegym_filter \
  --phase gold \
  --dataset-type swegym \
  --data-path /mnt/sn-007/jiaxicao/datasets/SWE-Gym/data/train-00000-of-00001.parquet \
  --gold-repeats 4 \
  --eval-timeout 600 \
  --concurrency 16 \
  --out-dir /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_gold_20260711_042217
```

| 项 | 值 |
|----|----|
| Out dir | `.../swegym_gold_20260711_042217` |
| Kept | **1404** |
| Excluded | 1034 |

## Phase P — CC ×8，只剔 8/8（已完成）

实跑配置：`Qwen3.5-9B`，`NUM_GPUS=8 TP_SIZE=1 --dp-size 8`，`concurrency=128`。

```bash
OUT=/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409
nohup .venv/bin/python -m examples.claudecode_ags.eval.swegym_filter \
  --phase passk \
  --dataset-type swegym \
  --kept-jsonl /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_gold_20260711_042217/kept.jsonl \
  --passk-repeats 8 \
  --time-budget 1800 \
  --eval-timeout 600 \
  --concurrency 128 \
  --out-dir "$OUT" \
  >>"$OUT/nohup.out" 2>&1 &
```

| 项 | 值 |
|----|----|
| Out dir | `.../swegym_passk_20260711_090409` |
| train_candidates | **1394** |
| excluded always_resolved | **10** |

勿用 `python | tee`（易 BrokenPipe）；进程自写 `runner.log`。

## 验收

- Gold：`n_kept + n_excluded = 2438`（1404 + 1034）。**已满足。**
- Passk：仅 8/8 进 `excluded_passk`；0–7 进 `train_candidates`（1394 + 10 = 1404）。**已满足。**

分布、timeout、repo 明细见结果笔记。
