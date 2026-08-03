#!/usr/bin/env bash
# Continue the 8-GPU readfix full-continuation run from iter_0000009
# through step 100 (101 total training updates).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-readfix-fullcont-s101-resume10-v1}"
export WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
export INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"
export K8S_NODE_GROUP="${K8S_NODE_GROUP:-shennong-5}"

export EXP_TAG="qwen35_9b_cc_ags_hybrid_s0_fullcont_s23_swe_8gpu_readfix_v1"
export LOG_DIR="/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}"
export LOAD_PATH="${LOG_DIR}/slime_save"
export SAVE_PATH="${LOG_DIR}/slime_save"
export RUN_ROOT="${LOG_DIR}"
export LOAD_CKPT_STEP="9"
export ALLOW_RESUME="1"

export PHASE="train"
export NUM_ROLLOUT="101"
# Each checkpoint is about 101 GiB. Save every five updates and always save
# the final step 100 to limit this continuation to about 2 TiB of new ckpts.
export SAVE_INTERVAL="5"
export ROLLOUT_BATCH_SIZE="16"
export GLOBAL_BATCH_SIZE="16"

export ACTOR_NUM_NODES="1"
export ACTOR_NUM_GPUS_PER_NODE="8"
export WORKER_REPLICAS="0"
export ROLLOUT_NUM_GPUS="8"

export STEP_GRPO_HYBRID_K="8"
export STEP_GRPO_FILTER="1"
export STEP_GRPO_STAGE1_LOSS_WEIGHT="1.0"
export STEP_GRPO_BRANCH_LOSS_WEIGHT="1.0"
export STEP_GRPO_STAGE2_LOSS_SCOPE="full_continuation"
export STEP_GRPO_BRANCH_SUBMIT_BATCH="64"
export STEP_GRPO_BUNDLE_DIR="${LOG_DIR}/step_reconstruct_bundles"

export SLIME_CC_TIME_BUDGET_SEC="1800"
export SLIME_CC_AGENT_CONCURRENCY="0"
export SLIME_CC_EVAL_CONCURRENCY="0"
export SLIME_CC_TOOL_LOOP_PENALTY="1"
export SLIME_CC_TIMEOUT_OUTCOME_REWARD="1"
export SLIME_AGENT_AGS_TOOL_ID="sdt-du5s75k5"

export WANDB_GROUP="${EXP_TAG}"
export WANDB_RUN_ID="waamqfbw"
export WANDB_RESUME="must"
unset WANDB_FORK_FROM WANDB_RESUME_FROM

exec bash "${LAUNCH_DIR}/hybrid_1node_job/submit_job.sh" "$@"
