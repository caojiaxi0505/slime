#!/usr/bin/env bash
# Render + apply / delete the Path A GRPO 1-node debug PyTorchJob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pytorchjob.yaml.template"

usage() {
  cat <<'EOF'
Usage: submit_job.sh [--delete] [--dry-run] [--help]

Creates a 1×8GPU PyTorchJob (Master only) for Path A GRPO debug.
Master labels match grpo_adapter_alb Service (app=cc-ags-recorder,
workload=jiaxicao-grpo-1node-debug).

Useful env:
  JOB_NAME                 jiaxicao-grpo-1node-debug
  K8S_NAMESPACE            sn5-system-intern
  IMAGE_URI                youtu-agent slime-nightly (default; has Ray/Megatron/EFA)
  SLIME_DIR                cc-ags-swe worktree
  SLIME_ADAPTER_PUBLIC_URL ALB URL (default: jiaxicao GRPO ALB)
  PHASE                    all|train|eval (default all)
  AGS_SECRET_NAME          qwen35-9b-ags-credentials
  WANDB_SECRET_NAME        Kubernetes Secret name (default wandb-credentials)
  WANDB_SECRET_KEY         Secret data key (default WANDB_API_KEY)
EOF
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
JOB_NAME="${JOB_NAME:-jiaxicao-grpo-1node-debug}"
IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/youtu-agent:slime-nightly-dev-20260530a-efa-swe-mooncake}"
SLIME_DIR="${SLIME_DIR:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl}"
EVAL_DATA="${EVAL_DATA:-/mnt/sn-007/youtu-agent/yuleiqin/SWE_code/DataEng/RL_DATA/data_valid/swe_agent_ags_swebench_verified/test.parquet}"
EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_1node_grpo_debug}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
PHASE="${PHASE:-train}"
AGS_SECRET_NAME="${AGS_SECRET_NAME:-qwen35-9b-ags-credentials}"
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL:-http://k8s-sn5syste-jiaxicao-a24fd39ae4-711009045.ap-southeast-3.elb.amazonaws.com}"
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"
SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-exb9o2gb}"
NUM_ROLLOUT="${NUM_ROLLOUT:-22}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
WANDB_SECRET_NAME="${WANDB_SECRET_NAME:-wandb-credentials}"
WANDB_SECRET_KEY="${WANDB_SECRET_KEY:-WANDB_API_KEY}"

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete pytorchjob "${JOB_NAME}" --ignore-not-found
  echo "Deleted pytorchjob/${JOB_NAME} in ${K8S_NAMESPACE}"
  exit 0
fi

for p in "${SLIME_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}" "${EVAL_DATA}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: path not found: ${p}" >&2
    exit 1
  fi
done

if ! kubectl -n "${K8S_NAMESPACE}" get secret "${AGS_SECRET_NAME}" >/dev/null 2>&1; then
  echo "ERROR: missing secret ${AGS_SECRET_NAME} in ${K8S_NAMESPACE}" >&2
  exit 1
fi

if ! kubectl -n "${K8S_NAMESPACE}" get secret "${WANDB_SECRET_NAME}" >/dev/null 2>&1; then
  echo "ERROR: missing W&B secret ${WANDB_SECRET_NAME} in ${K8S_NAMESPACE}" >&2
  exit 1
fi

if ! kubectl -n "${K8S_NAMESPACE}" get ingress jiaxicao-grpo-1node-debug-adapter >/dev/null 2>&1; then
  echo "WARNING: adapter Ingress not found; apply grpo_adapter_alb first." >&2
fi

export JOB_NAME K8S_NAMESPACE IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export PROMPT_DATA EVAL_DATA EXP_TAG LOG_DIR PHASE AGS_SECRET_NAME SLIME_ADAPTER_PUBLIC_URL
export SLIME_AGENT_AGS_TOOL_ID NUM_ROLLOUT SAVE_INTERVAL
export WANDB_PROJECT WANDB_GROUP WANDB_TEAM WANDB_SECRET_NAME WANDB_SECRET_KEY

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${K8S_NAMESPACE} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${PROMPT_DATA} ${EVAL_DATA} ${EXP_TAG} ${LOG_DIR} ${PHASE} ${AGS_SECRET_NAME} ${SLIME_ADAPTER_PUBLIC_URL} ${SLIME_AGENT_AGS_TOOL_ID} ${NUM_ROLLOUT} ${SAVE_INTERVAL} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM} ${WANDB_SECRET_NAME} ${WANDB_SECRET_KEY}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} in ${K8S_NAMESPACE}"
echo "    image: ${IMAGE_URI}"
echo "    SLIME_DIR=${SLIME_DIR}"
echo "    SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "    PHASE=${PHASE} LOG_DIR=${LOG_DIR}"

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
echo "  kubectl -n ${K8S_NAMESPACE} get pods -l workload=jiaxicao-grpo-1node-debug -w"
echo "  kubectl -n ${K8S_NAMESPACE} logs -f job/${JOB_NAME}-master-0 2>/dev/null || \\"
echo "    kubectl -n ${K8S_NAMESPACE} logs -f -l training.kubeflow.org/job-name=${JOB_NAME},training.kubeflow.org/replica-type=master"
echo "ALB /health (after adapter up):"
echo "  curl -fsS ${SLIME_ADAPTER_PUBLIC_URL}/health"
