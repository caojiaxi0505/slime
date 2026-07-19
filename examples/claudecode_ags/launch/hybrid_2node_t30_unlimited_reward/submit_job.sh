#!/usr/bin/env bash
# Fresh 16-GPU Hybrid: 30-minute agents, no global agent gate, shared reward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Reuse the current Hybrid ALB and service selector after deleting the old Job.
export JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-2node-full}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

export EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_2node_hybrid_t30_unlimited_tokenexact_v9_reward}"
export LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
export SLIME_DIR

export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export NUM_ROLLOUT="${NUM_ROLLOUT:-44}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
export STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"
export STEP_GRPO_BRANCH_LOSS_WEIGHT="${STEP_GRPO_BRANCH_LOSS_WEIGHT:-1.0}"

exec bash "${SLIME_DIR}/examples/claudecode_ags/launch/hybrid_2node_job/submit_job.sh" "$@"
