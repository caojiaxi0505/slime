# AGS smoke (L0 / L1 / L2)

Manual real-AGS checks for Path A. **Not** part of default `pytest`.

| Level | What it does |
|-------|----------------|
| **L0** | Create sandbox → `echo ok && pwd` → destroy |
| **L1** | normalize → workspace init → apply **gold `patch`** → `dispatch.evaluate` |
| **L2** | normalize → sandbox A: toolchain + **live Claude Code** via `SLIME_ADAPTER_PUBLIC_URL` → `git_diff` → sandbox B: eval **model** diff |

Ops for Phase 1 deploy + L2:  
[`docs/superpowers/runbooks/2026-07-11-sglang-adapter-l2-ops.md`](../../../docs/superpowers/runbooks/2026-07-11-sglang-adapter-l2-ops.md)

## Prerequisites

1. AGS credentials via `SLIME_AGENT_AGS_ENV_FILE` (or keys in env / `--env-file`).
2. `SLIME_AGENT_SANDBOX_BACKEND=ags` (script sets this if unset).
3. `SLIME_AGENT_AGS_SWE_REX_ROOT` if SWE-ReX is not installed as a package.
4. Image pullable by AGS (default registry from `SLIME_CC_IMAGE_REGISTRY`, TCR).
5. For parquet: `pyarrow` installed. Or pass `--row-json` instead.
6. **L2 only:** Phase 1 SGLang+adapter up; `SLIME_ADAPTER_PUBLIC_URL` = dedicated ALB (not localhost); toolchain (COS mount) available.

```bash
source examples/claudecode_ags/env/load_env.sh
```

## Examples

Verified parquet:

```text
/mnt/sn-007/jiaxicao/datasets/SWE-bench_Verified/data/test-00000-of-00001.parquet
```

```bash
cd /path/to/slime   # worktree root

# L0 / L1 — see prior commands with --level 0|1

# L2 live CC (requires adapter URL)
python -m examples.claudecode_ags.smoke.ags_smoke \
  --level 2 \
  --dataset-type swebench_verified \
  --data-path /mnt/sn-007/jiaxicao/datasets/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --instance-id astropy__astropy-12907 \
  --allow-unresolved
```

`--allow-unresolved` on L2 means **link passed** even if the model did not solve the issue (design: 链路通过必达，题目解决加分).

Optional: `--skip-health-gate`, `--time-budget 1800`, `--eval-timeout 600`.

## Exit codes

- `0` — success (L2: resolved, or `--allow-unresolved` after full link)
- `1` — ran but unresolved (L1/L2 without `--allow-unresolved`)
- `2` — hard error (missing URL, health gate, init failure, …)
