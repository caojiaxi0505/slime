# Claude Code + AGS + SWE (Path A)

End-to-end SWE coding-agent RL with **Claude Code** inside **Tencent AGS** sandboxes, segmented trajectories, and fan-out GRPO. Business logic lives here; reusable sandbox/adapter/GRPO pieces live under `slime/`.

Design spec: [CC + AGS + SWE refactor design](../../docs/superpowers/specs/2026-07-10-cc-ags-swe-refactor-design.md)

## Call chain

```text
launch/run_grpo_example.sh
  → source env/load_env.sh
  → train.py (Ray job on cluster)
  → RolloutManager
  → --custom-generate-function-path examples.claudecode_ags.generate.generate
       ├─ SegmentedAnthropicAdapter (subagent / wipe / final segments)
       ├─ make_sandbox(backend=ags)
       ├─ agent_runtime: toolchain / PROBLEM / claude -p
       │    └─ CC → SLIME_ADAPTER_PUBLIC_URL → adapter → SGLang
       ├─ git_diff → fresh eval sandbox (swe_eval)
       ├─ --custom-cc-reward-function-path → scalar R (default: binary)
       └─ fan_out_sample_segments(R/K, shared rollout_id)
  → --custom-reward-post-process-path slime.rollout.fanout_grpo.post_process_rewards
       └─ sum per attempt → group GRPO → broadcast advantage to segments
  → Megatron update
```

## Two env files (no aliases)

| File | Contents |
|------|----------|
| `env/claude_code.env` | Official Claude Code / Anthropic / `BASH_*` vars only |
| `env/slime_ags.env` | All `SLIME_*` knobs (copy from `slime_ags.env.example`) |

Load both with:

```bash
source examples/claudecode_ags/env/load_env.sh
```

`load_env.sh` sources `claude_code.env`, then `slime_ags.env` if present (otherwise warns and falls back to `slime_ags.env.example`). There is **no** rename or `SWE_*` → `CLAUDE_*` mapping layer.

Put AGS secrets in a local file and point `SLIME_AGENT_AGS_ENV_FILE` at it; do not commit `slime_ags.env`.

## Quick start (skeleton)

```bash
cp examples/claudecode_ags/env/slime_ags.env.example examples/claudecode_ags/env/slime_ags.env
# edit slime_ags.env: AGS creds, adapter URL, tarball paths, datasets

export HF_CHECKPOINT=/path/to/model
export REF_MODEL_PATH=/path/to/model_torch_dist
export PROMPT_DATA=/path/to/train.jsonl
export EVAL_DATA=/path/to/eval.jsonl

# Dry-run: print train.py command
bash examples/claudecode_ags/launch/run_grpo_example.sh

# Execute (needs Ray + GPUs)
RUN=1 bash examples/claudecode_ags/launch/run_grpo_example.sh
```

## Swap the reward function

Default reward is **binary** (`resolved` → 1.0, else 0.0) in `rewards/default.py`.

To use a different shaping policy:

1. Add `examples/claudecode_ags/rewards/your_reward.py` with:

   ```python
   def compose(*, base_eval: dict, sample=None, args=None) -> tuple[float, dict]:
       ...
   ```

2. Point training at it:

   ```bash
   --custom-cc-reward-function-path examples.claudecode_ags.rewards.your_reward.compose
   ```

`fan_out` (even `R/K` split) and `fanout_grpo` (sum-then-GRPO-broadcast) stay unchanged.

## Layout

| Path | Role |
|------|------|
| `generate.py` | Per-sample orchestration |
| `agent_runtime.py` | Sandbox interior: toolchain, workspace, `claude`, `git diff` |
| `rewards/` | Pluggable reward plugins |
| `swe_eval/` | SWE test evaluation |
| `env/` | `claude_code.env`, `slime_ags.env`, `load_env.sh` |
| `launch/` | Cluster launch scripts |
