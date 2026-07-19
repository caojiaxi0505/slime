#!/usr/bin/env bash
# Branch Hybrid v9 at iter14 and train through iter24. Stage-2 still runs the
# complete continuation, but only its first assistant turn contributes loss.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-i14-firstturn}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

export EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_hybrid_i14_i24_firstturn_s1fix_v1}"
export LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
export LOAD_PATH="${LOAD_PATH:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_2node_hybrid_t30_unlimited_tokenexact_v9_reward/slime_save}"
export LOAD_CKPT_STEP="${LOAD_CKPT_STEP:-14}"
export SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
export RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"

export PHASE="train"
export NUM_ROLLOUT="${NUM_ROLLOUT:-25}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
export STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"
export STEP_GRPO_BRANCH_LOSS_WEIGHT="${STEP_GRPO_BRANCH_LOSS_WEIGHT:-1.0}"
export STEP_GRPO_STAGE2_LOSS_SCOPE="first_turn"
export STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-64}"

# Match Hybrid v9's 30-minute, unlimited-agent-concurrency reward setting.
export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export SLIME_AGENT_AGS_TOOL_ID="sdt-db5nvd67"

# ``hy9ro2`` is the connected Hybrid v9 history. Its W&B internal step 59 is
# the final log row for rollout/step=14; the fork therefore starts logging at 15.
export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
export WANDB_RESUME_FROM="${WANDB_RESUME_FROM:-hy9ro2?_step=59}"
export WANDB_RESUME="auto"
unset WANDB_RUN_ID

exec bash "${LAUNCH_DIR}/hybrid_2node_job/submit_job.sh" "$@"
