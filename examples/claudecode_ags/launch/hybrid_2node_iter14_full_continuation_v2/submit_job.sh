#!/usr/bin/env bash
# Resume Hybrid from iter14 through iter24 and train every resumed assistant
# turn. Uses an independent ALB/tool/W&B run from the first-turn experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-i14-full-v2}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

export EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_hybrid_i14_i24_fullcont_s1fix_agsretry_v2}"
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
export STEP_GRPO_STAGE2_LOSS_SCOPE="full_continuation"
export STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-64}"

export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export SLIME_AGENT_AGS_TOOL_ID="sdt-du5s75k5"

# This independent run already contains the old Hybrid history through rollout 14.
export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-fsh4z38b}"
export WANDB_RESUME="${WANDB_RESUME:-must}"
unset WANDB_FORK_FROM WANDB_RESUME_FROM

exec bash "${LAUNCH_DIR}/hybrid_2node_job/submit_job.sh" "$@"
