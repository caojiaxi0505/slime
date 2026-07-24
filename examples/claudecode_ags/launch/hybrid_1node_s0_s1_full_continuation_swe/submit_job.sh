#!/usr/bin/env bash
# Fresh Hybrid full-continuation 8-GPU smoke run from step0 for 1 rollout step.
# Purpose: measure one-step wall time without consuming a full 43-step run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-s0-fullcont-s43-swe}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

export EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_hybrid_s0_fullcont_s43_swe_8gpu_1step_v1}"
export LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
export SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
export RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"

unset LOAD_CKPT_STEP

export PHASE="train"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export ACTOR_NUM_NODES="1"
export ACTOR_NUM_GPUS_PER_NODE="8"
export WORKER_REPLICAS="0"
export ROLLOUT_NUM_GPUS="8"

export STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
export STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"
export STEP_GRPO_BRANCH_LOSS_WEIGHT="${STEP_GRPO_BRANCH_LOSS_WEIGHT:-1.0}"
export STEP_GRPO_STAGE2_LOSS_SCOPE="full_continuation"
export STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-64}"

export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-du5s75k5}"

export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
unset WANDB_RUN_ID WANDB_RESUME_FROM WANDB_FORK_FROM
export WANDB_RESUME="${WANDB_RESUME:-auto}"

exec bash "${LAUNCH_DIR}/hybrid_2node_job/submit_job.sh" "$@"
