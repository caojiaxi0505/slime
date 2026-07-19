#!/usr/bin/env bash
# Fresh 16-GPU naive GRPO: 30-minute agents, no global agent gate, shared reward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Reuse the current naive-GRPO ALB and service selector after deleting the old Job.
export JOB_NAME="${JOB_NAME:-jiaxicao-grpo-2node-c64t45}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

export EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_2node_grpo_t30_unlimited_loop3_reward_save5}"
export LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
export LOAD_PATH="${LOAD_PATH:-${LOG_DIR}/slime_save}"
export SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
export RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"
export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
export SLIME_DIR

export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export NUM_ROLLOUT="${NUM_ROLLOUT:-44}"
export SAVE_INTERVAL="5"
export RESUME_DEBUG_ROLLOUT_DATA="0"

exec bash "${SLIME_DIR}/examples/claudecode_ags/launch/grpo_2node_job/submit_job.sh" "$@"
