# Path A 1-node GRPO Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship data conversion + 1×8GPU colocate GRPO launcher (~21 steps on 338 SWE-Gym prompts) and post-train SWE-bench Verified (484) eval using Path A `claudecode_ags`.

**Architecture:** Convert filter JSONL → slime `{prompt,extra_info}` rows. New `run_grpo_1node_debug.sh` mirrors tencent 9B 2-node GRPO knobs scaled to 1 node / 8 GPU `--colocate`, but wires Path A generate/reward/`fanout_grpo`. Mid-train eval skipped; `eval-interval=num-rollout` runs Verified once after the last train step (same process / same weights). Optional `PHASE=eval` reloads `slime_save` for Verified-only reruns.

**Tech Stack:** bash, Ray, Megatron, SGLang, AGS, Claude Code, pytest

**Spec:** [2026-07-12-grpo-1node-debug-design.md](../specs/2026-07-12-grpo-1node-debug-design.md)

**Worktree:** `/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe`

**Commits:** Do **not** auto-commit unless the user explicitly asks (user rule overrides frequent-commit defaults).

---

## File map

| Path | Responsibility |
|------|----------------|
| `examples/claudecode_ags/data/to_slime_prompt_jsonl.py` | Filter JSONL → slime `PROMPT_DATA` |
| `tests/claudecode_ags/test_to_slime_prompt_jsonl.py` | Conversion unit tests |
| `examples/claudecode_ags/launch/run_grpo_1node_debug.sh` | 1-node train (+ optional post Verified) |
| `examples/claudecode_ags/env/slime_ags.env` | Timeouts + clear obsolete L2 ALB |
| `docs/superpowers/runbooks/2026-07-12-grpo-1node-debug-ops.md` | Operator runbook |
| `examples/claudecode_ags/CHECKLIST.md` | Mark GRPO launch progress |
| `eval_runs/.../train_grpo_resolved_1_7.slime.jsonl` | Generated artifact (338 rows; not committed) |

---

### Task 1: Filter JSONL → slime `PROMPT_DATA` converter

**Files:**
- Create: `examples/claudecode_ags/data/to_slime_prompt_jsonl.py`
- Create: `examples/claudecode_ags/data/__init__.py` (empty)
- Create: `tests/claudecode_ags/test_to_slime_prompt_jsonl.py`

- [x] **Step 1: Write the failing test**

```python
# tests/claudecode_ags/test_to_slime_prompt_jsonl.py
import json
from pathlib import Path

from examples.claudecode_ags.data.to_slime_prompt_jsonl import convert_row, convert_file

REQUIRED_EXTRA = ("instance_id", "image", "problem_statement", "FAIL_TO_PASS", "dataset_type")


def test_convert_row_maps_prompt_and_extra_info():
    src = {
        "instance_id": "getmoto__moto-5386",
        "n_resolved": 3,
        "n_repeats": 8,
        "metadata": {
            "instance_id": "getmoto__moto-5386",
            "problem_statement": "Fix hibernation",
            "image": "swebenchdocker.tencentcloudcr.com/swebench/x:latest",
            "FAIL_TO_PASS": ["t1"],
            "PASS_TO_PASS": ["t2"],
            "dataset_type": "swegym",
            "data_source": "swegym",
            "repo": "getmoto/moto",
        },
    }
    out = convert_row(src)
    assert out["prompt"] == "Fix hibernation"
    assert out["extra_info"]["instance_id"] == "getmoto__moto-5386"
    assert out["extra_info"]["n_resolved"] == 3
    for k in REQUIRED_EXTRA:
        assert k in out["extra_info"] and out["extra_info"][k] not in (None, "")


def test_convert_row_rejects_missing_problem_statement():
    import pytest

    with pytest.raises(ValueError, match="problem_statement"):
        convert_row({"instance_id": "x", "metadata": {"instance_id": "x", "image": "i"}})


def test_convert_file_roundtrip(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    row = {
        "instance_id": "a__b-1",
        "n_resolved": 1,
        "metadata": {
            "instance_id": "a__b-1",
            "problem_statement": "p",
            "image": "img",
            "FAIL_TO_PASS": ["t"],
            "dataset_type": "swegym",
        },
    }
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    n = convert_file(src, dst)
    assert n == 1
    got = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
    assert set(got) >= {"prompt", "extra_info"}
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m pytest tests/claudecode_ags/test_to_slime_prompt_jsonl.py -v
```

Expected: FAIL (module not found / import error).

- [x] **Step 3: Implement converter**

```python
# examples/claudecode_ags/data/to_slime_prompt_jsonl.py
"""Convert swegym_filter JSONL rows into slime PROMPT_DATA JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REQUIRED_META = ("instance_id", "image", "problem_statement", "FAIL_TO_PASS")


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    md = dict(row.get("metadata") or {})
    if not md.get("instance_id"):
        md["instance_id"] = row.get("instance_id")
    problem = str(md.get("problem_statement") or "").strip()
    if not problem:
        raise ValueError(f"missing problem_statement for {md.get('instance_id')!r}")
    for key in _REQUIRED_META:
        if key == "problem_statement":
            continue
        if key not in md or md[key] in (None, ""):
            raise ValueError(f"missing {key} for {md.get('instance_id')!r}")
    extra = dict(md)
    for k in ("n_resolved", "n_repeats", "n_runs", "pass_rate"):
        if k in row:
            extra[k] = row[k]
    return {"prompt": problem, "extra_info": extra}


def convert_file(src: Path, dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(convert_row(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--expect-rows", type=int, default=None)
    args = p.parse_args(argv)
    n = convert_file(args.src, args.dst)
    if args.expect_rows is not None and n != args.expect_rows:
        raise SystemExit(f"row count {n} != expect {args.expect_rows}")
    print(f"wrote {n} rows -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Also create empty `examples/claudecode_ags/data/__init__.py`.

- [x] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/claudecode_ags/test_to_slime_prompt_jsonl.py -v
```

Expected: PASS.

- [x] **Step 5: Generate the live 338-row artifact**

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m examples.claudecode_ags.data.to_slime_prompt_jsonl \
  --src /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.jsonl \
  --dst /mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl \
  --expect-rows 338
```

Expected: `wrote 338 rows -> ...slime.jsonl`.

---

### Task 2: Env knobs for this debug run

**Files:**
- Modify: `examples/claudecode_ags/env/slime_ags.env`

- [x] **Step 1: Set timeouts to match spec**

Ensure:

```bash
SLIME_CC_TIME_BUDGET_SEC=1800
SLIME_CC_EVAL_TIMEOUT_SEC=600
SLIME_AGENT_AGS_TIMEOUT=45m
```

- [x] **Step 2: Neutralize obsolete L2 ALB**

Replace the old `SLIME_ADAPTER_PUBLIC_URL=http://k8s-sn5syste-jiaxicao-226b2ab12d-...` line with a commented placeholder so `load_env.sh` does not silently reuse a deleted Ingress:

```bash
# Training sets this to a URL AGS can reach for THIS run's adapter (not the deleted L2 deploy).
# SLIME_ADAPTER_PUBLIC_URL=http://REPLACE_WITH_TRAIN_ADAPTER_PUBLIC_URL
```

Launcher must require a non-empty, non-placeholder URL at runtime.

---

### Task 3: `run_grpo_1node_debug.sh` (train + post Verified)

**Files:**
- Create: `examples/claudecode_ags/launch/run_grpo_1node_debug.sh`
- Keep: `examples/claudecode_ags/launch/run_grpo_example.sh` (untouched skeleton)

Reference when porting knobs (do **not** copy Path B / step-grpo / swe_agent generate paths):

- `/mnt/sn-007/jiaxicao/code/slime-tencent/examples/claudecode-ags/run_qwen35_9b_swe_2nodes_ags_common.sh` (Ray start, SGLANG_ARGS, PERF/ALGO/OPTIMIZER, `--colocate`)
- Path A wiring from `run_grpo_example.sh`

- [x] **Step 1: Create launcher with defaults from spec**

Key defaults (all overridable via env):

| Env / flag | Default |
|------------|---------|
| `PHASE` | `all` (`train` \| `eval` \| `all`) |
| `ACTOR_NUM_NODES` | `1` |
| `ACTOR_NUM_GPUS_PER_NODE` | `8` |
| `ROLLOUT_NUM_GPUS` | `8` |
| `ROLLOUT_TP_SIZE` | `1` |
| `TP_SIZE` | `1` |
| `PP_SIZE` | `1` |
| `CP_SIZE` | `8` |
| `PROMPT_DATA` | `.../train_grpo_resolved_1_7.slime.jsonl` |
| `EVAL_DATA` | `.../swe_agent_ags_swebench_verified/test.parquet` |
| `HF_CHECKPOINT` | `/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B` |
| `REF_MODEL_PATH` | `/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist` |
| `ROLLOUT_BATCH_SIZE` | `16` |
| `N_SAMPLES_PER_PROMPT` | `8` |
| `NUM_ROLLOUT` | `21` |
| `GLOBAL_BATCH_SIZE` | `128` |
| `EVAL_INTERVAL` | same as `NUM_ROLLOUT` when `PHASE=all` |
| `SKIP_EVAL_BEFORE_TRAIN` | `1` |
| `N_SAMPLES_PER_EVAL_PROMPT` | `1` |
| `SAVE_INTERVAL` | `10` |
| `EXP_TAG` | `qwen35_9b_cc_ags_1node_grpo_debug` |
| `LOG_DIR` | `/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}` |

Script structure:

1. `source examples/claudecode_ags/env/load_env.sh`
2. Assert paths exist; assert `SLIME_ADAPTER_PUBLIC_URL` set and not `REPLACE_WITH*` / empty.
3. Export timeouts if not already set: `SLIME_CC_TIME_BUDGET_SEC=1800`, `SLIME_CC_EVAL_TIMEOUT_SEC=600`, `SLIME_AGENT_AGS_TIMEOUT=45m`.
4. Source `scripts/models/qwen3.5-9B.sh` → `MODEL_ARGS`.
5. Build `CKPT_ARGS`, `ROLLOUT_ARGS`, `EVAL_ARGS`, `PERF_ARGS`, `ALGO_ARGS`, `OPTIMIZER_ARGS`, `SGLANG_ARGS`, `MISC_ARGS` (`--colocate`).
6. Path A only:

```bash
--custom-generate-function-path examples.claudecode_ags.generate.generate
--custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
--custom-reward-post-process-path slime.rollout.fanout_grpo.post_process_rewards
--advantage-estimator grpo
```

Do **not** wire `examples.coding_agent_rl.*` or step-grpo.

7. `PHASE` behavior:
   - `train`: set `EVAL_INTERVAL` empty / omit eval args (or very large); `NUM_ROLLOUT=21`.
   - `all` (default): `NUM_ROLLOUT=21`, `--eval-interval 21`, `--skip-eval-before-train`, `--eval-prompt-data swebench_verified "${EVAL_DATA}"`, `--n-samples-per-eval-prompt 1`.
   - `eval`: `NUM_ROLLOUT=0` is invalid in many setups — instead use `NUM_ROLLOUT=1` with a tiny dummy prompt **or** prefer: `NUM_ROLLOUT=0` unsupported → document eval as `PHASE=all` after train resume, OR copy the eval-only pattern from tencent `run_swe484_eval_only_9b_ags.sh` using Path A generate. **Preferred for this plan:** implement `PHASE=eval` as a second `train.py` invocation with `--num-rollout 0` if supported; if not, use `--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1` on a 1-row dummy prompt file plus `--eval-interval 1` so the real work is Verified eval. Check `arguments.py` during implementation; pick the smallest working pattern and document it in the ops runbook.

8. Ray: 1-node head only (`ray start --head --num-gpus 8`); no worker loop.
9. Default `RUN=0` dry-prints the full command; `RUN=1` executes.
10. `chmod +x` the script.

Minimal dry-run check after writing:

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
SLIME_ADAPTER_PUBLIC_URL=http://10.0.0.1:18001 RUN=0 \
  bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh | head -80
```

Expected: prints `train.py` argv including `--actor-num-nodes 1`, `--rollout-batch-size 16`, `--n-samples-per-prompt 8`, `--num-rollout 21`, Path A generate path, `--colocate`.

- [x] **Step 2: Document `SLIME_ADAPTER_PUBLIC_URL` requirement in script header comments**

AGS cannot use `127.0.0.1`. Operator must supply a public/ALB URL that routes to this job’s adapter port (same constraint as tencent 2-node). Deleted L2 Ingress must not be used.

---

### Task 4: Ops runbook + checklist

**Files:**
- Create: `docs/superpowers/runbooks/2026-07-12-grpo-1node-debug-ops.md`
- Modify: `examples/claudecode_ags/CHECKLIST.md`
- Sync copies under `/mnt/sn-007/jiaxicao/code/slime/docs/superpowers/` as done for prior docs

- [x] **Step 1: Write ops runbook** covering:

1. Prerequisites: L2 adapter deleted; GPUs free; HF + torch_dist present; AGS secrets; `.slime.jsonl` generated.
2. Convert command (Task 1 Step 5).
3. Set `SLIME_ADAPTER_PUBLIC_URL` for the train node.
4. Dry-run then `RUN=1 PHASE=all bash .../run_grpo_1node_debug.sh`.
5. What to watch: `LOG_DIR/run.log`, rollout dumps, AGS failures, OOM.
6. Success criteria from spec §9.
7. Optional `PHASE=eval` after train if Verified needs rerun.
8. Mutual exclusion with `jiaxicao-cc-ags-adapter` Deployment.

- [x] **Step 2: Update CHECKLIST**

Replace deferred/partial GRPO rows with:

| Item | Status | Note |
|------|--------|------|
| SWE-Gym passk×8 | **done** | see filter notes; GRPO 338 / step-GRPO 1394 |
| GRPO 1-node debug launch | **partial** | script+ops shipped; live HyperPod run operator-owned |

- [x] **Step 3: Offline verification**

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
.venv/bin/python -m pytest tests/claudecode_ags/test_to_slime_prompt_jsonl.py -v
bash -n examples/claudecode_ags/launch/run_grpo_1node_debug.sh
SLIME_ADAPTER_PUBLIC_URL=http://10.0.0.1:18001 RUN=0 \
  bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh >/tmp/grpo_1node_dry.txt
rg -n "claudecode_ags.generate|fanout_grpo|rollout-batch-size 16|n-samples-per-prompt 8|num-rollout 21|actor-num-nodes 1|colocate|swebench_verified" /tmp/grpo_1node_dry.txt
```

Expected: pytest PASS; `bash -n` clean; dry-run contains all matched tokens.

---

## Success criteria (plan done)

1. Converter tests green; `train_grpo_resolved_1_7.slime.jsonl` has **338** rows with `prompt`/`extra_info`.
2. `run_grpo_1node_debug.sh` dry-run prints 1-node Path A GRPO command with spec defaults + post Verified eval args.
3. Ops runbook exists; env no longer points at deleted L2 ALB by default.
4. Live `RUN=1` on HyperPod remains **operator-owned** (needs GPUs + reachable adapter URL); not required to mark the plan’s code tasks complete.

---

## Spec coverage check

| Spec item | Task |
|-----------|------|
| 338 → slime JSONL | Task 1 |
| 1 node / batch 16 / n=8 / ~21 steps / gbs 128 | Task 3 |
| timeouts 30min / 10min / sandbox 45min | Task 2–3 |
| skip mid-train eval; post Verified 484×1 | Task 3 (`PHASE=all`) |
| Path A generate + fanout_grpo | Task 3 |
| no L2 adapter dependency | Task 2 + ops |
| ops + success criteria | Task 4 |

## Placeholder / consistency scan

- No TBD left for eval set (Verified parquet path fixed).
- `PHASE=eval` has an explicit fallback strategy if `--num-rollout 0` unsupported.
- Commits gated on user request.
