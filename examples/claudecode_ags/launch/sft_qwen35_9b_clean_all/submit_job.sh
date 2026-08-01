#!/usr/bin/env bash
# Render + apply/delete Qwen3.5-9B clean all-trials SFT PyTorchJob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pytorchjob.yaml.template"

usage() {
  cat <<'EOF'
Usage: submit_job.sh [--delete] [--dry-run] [--help]

Creates a 1×8GPU PyTorchJob for Qwen3.5-9B SFT.

Useful env:
  JOB_NAME            default jiaxicao-qwen35-9b-sft-clean-all-e3
  K8S_NAMESPACE       default sn5-system-intern
  IMAGE_URI           slime training image
  SLIME_DIR           default /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
  HF_CHECKPOINT       default Qwen3.5-9B HF path
  REF_MODEL_PATH      default Qwen3.5-9B_torch_dist
  PROMPT_DATA         default slime_sft_all.clean.jsonl
  EXP_TAG / LOG_DIR   default qwen35_9b_sft_deepseekv4_swegym1404_clean_all_e3
  LOAD_PATH           default REF_MODEL_PATH
  SAVE_PATH           default LOG_DIR/slime_save
  NUM_EPOCH           default 3
  ROLLOUT_BATCH_SIZE  default 16
  GLOBAL_BATCH_SIZE   default 16
  SAVE_INTERVAL       default 20
  ALLOW_RESUME        set 1 to allow an existing SAVE_PATH
  WANDB_KEY           optional if ~/.config/jiaxicao/wandb_api_key exists
EOF
}

_load_wandb_key() {
  if [[ -n "${WANDB_KEY:-}" ]]; then
    return 0
  fi
  local f="${WANDB_KEY_FILE:-${HOME}/.config/jiaxicao/wandb_api_key}"
  if [[ -f "${f}" ]]; then
    WANDB_KEY="$(tr -d '[:space:]' < "${f}")"
  fi
}

DELETE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    --delete) DELETE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
JOB_NAME="${JOB_NAME:-jiaxicao-qwen35-9b-sft-clean-all-e3}"
WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/youtu-agent:slime-nightly-dev-20260530a-efa-swe-mooncake}"
SLIME_DIR="${SLIME_DIR:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/deepseek_v4_pro_swegym1404_sft_20260723_c16/slime_sft_all.clean.jsonl}"
EXP_TAG="${EXP_TAG:-qwen35_9b_sft_deepseekv4_swegym1404_clean_all_e3}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
LOAD_PATH="${LOAD_PATH:-${REF_MODEL_PATH}}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
NUM_EPOCH="${NUM_EPOCH:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-131071}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
_load_wandb_key
WANDB_KEY="${WANDB_KEY:-}"

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete pytorchjob "${JOB_NAME}" --ignore-not-found
  echo "Deleted pytorchjob/${JOB_NAME} in ${K8S_NAMESPACE}"
  exit 0
fi

for p in "${SLIME_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: path not found: ${p}" >&2
    exit 1
  fi
done

if [[ -z "${WANDB_KEY}" ]]; then
  echo "WARNING: WANDB_KEY missing; training will run without W&B." >&2
fi

if [[ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "ERROR: ${SAVE_PATH} already has a checkpoint." >&2
  echo "       Use a fresh EXP_TAG/LOG_DIR, or set ALLOW_RESUME=1 to continue." >&2
  exit 4
fi

export JOB_NAME WORKLOAD_LABEL K8S_NAMESPACE IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export PROMPT_DATA EXP_TAG LOG_DIR LOAD_PATH SAVE_PATH NUM_EPOCH ROLLOUT_BATCH_SIZE GLOBAL_BATCH_SIZE
export SAVE_INTERVAL MAX_CONTEXT_LEN ROLLOUT_MAX_PROMPT_LEN MAX_TOKENS_PER_GPU
export WANDB_KEY WANDB_PROJECT WANDB_GROUP WANDB_TEAM

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${WORKLOAD_LABEL} ${K8S_NAMESPACE} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${PROMPT_DATA} ${EXP_TAG} ${LOG_DIR} ${LOAD_PATH} ${SAVE_PATH} ${NUM_EPOCH} ${ROLLOUT_BATCH_SIZE} ${GLOBAL_BATCH_SIZE} ${SAVE_INTERVAL} ${MAX_CONTEXT_LEN} ${ROLLOUT_MAX_PROMPT_LEN} ${MAX_TOKENS_PER_GPU} ${WANDB_KEY} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} in ${K8S_NAMESPACE}"
echo "    image: ${IMAGE_URI}"
echo "    SLIME_DIR=${SLIME_DIR}"
echo "    PROMPT_DATA=${PROMPT_DATA}"
echo "    LOAD_PATH=${LOAD_PATH}"
echo "    SAVE_PATH=${SAVE_PATH}"
echo "    epochs=${NUM_EPOCH} RBS=${ROLLOUT_BATCH_SIZE} GBS=${GLOBAL_BATCH_SIZE} save_interval=${SAVE_INTERVAL}"
echo "    max_context=${MAX_CONTEXT_LEN} max_prompt=${ROLLOUT_MAX_PROMPT_LEN} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
echo "    WANDB project=${WANDB_PROJECT} team=${WANDB_TEAM} group=${WANDB_GROUP} key=$([[ -n \"${WANDB_KEY}\" ]] && echo '***' || echo 'none')"

if [[ "${DRY_RUN}" == "1" ]]; then
  kubectl apply --dry-run=client -f "${RENDERED}"
  rm -f "${RENDERED}"
  exit 0
fi

kubectl apply -f "${RENDERED}"
rm -f "${RENDERED}"

echo
echo "Watch:"
echo "  kubectl -n ${K8S_NAMESPACE} get pytorchjob ${JOB_NAME} -w"
echo "  kubectl -n ${K8S_NAMESPACE} get pods -l workload=${WORKLOAD_LABEL} -w"
echo "  kubectl -n ${K8S_NAMESPACE} logs -f -l training.kubeflow.org/job-name=${JOB_NAME},training.kubeflow.org/replica-type=master"

