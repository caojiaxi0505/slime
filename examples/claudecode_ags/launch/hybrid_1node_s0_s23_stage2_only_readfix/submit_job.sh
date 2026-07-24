#!/usr/bin/env bash
# Fresh Hybrid Stage-2-only 8-GPU run from step0 for 23 rollout steps.
# Stage-1 is still executed to produce branch points, but its loss weight is 0.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-s0-stage2only-s23-readfix}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

export EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_hybrid_s0_stage2only_s23_swe_8gpu_readfix_v1}"
export LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
export SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
export RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"

unset LOAD_CKPT_STEP

export PHASE="train"
export NUM_ROLLOUT="${NUM_ROLLOUT:-23}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
export STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"
export STEP_GRPO_STAGE1_LOSS_WEIGHT="0"
export STEP_GRPO_BRANCH_LOSS_WEIGHT="${STEP_GRPO_BRANCH_LOSS_WEIGHT:-1.0}"
export STEP_GRPO_STAGE2_LOSS_SCOPE="full_continuation"
export STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-64}"

export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-g11z9k6n}"

export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
unset WANDB_RUN_ID WANDB_RESUME_FROM WANDB_FORK_FROM
export WANDB_RESUME="${WANDB_RESUME:-auto}"

exec bash "${LAUNCH_DIR}/hybrid_1node_job/submit_job.sh" "$@"
