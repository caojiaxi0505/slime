#!/usr/bin/env bash
# Render + apply / delete Path A SWE484 (Verified) eval-only PyTorchJob.
# Usage: EVAL_ROLE=base|grpo ./submit_job.sh [--delete] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pytorchjob.yaml.template"

usage() {
  cat <<'EOF'
Usage: EVAL_ROLE=base|grpo submit_job.sh [--delete] [--dry-run] [--help]

Creates a 1×8GPU PyTorchJob (Master only) for SWE-bench Verified 484 eval.
Master labels: app=cc-ags-recorder, workload=<JOB_NAME>.

EVAL_ROLE=base  → load REF megatron (Qwen3.5-9B_torch_dist)
EVAL_ROLE=grpo  → load naive GRPO ckpt (iter from slime_save)

Useful env:
  JOB_NAME / WORKLOAD / LOG_DIR / LOAD_PATH / SAVE_PATH / SLIME_ADAPTER_PUBLIC_URL
  K8S_NAMESPACE  sn5-system-intern
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

EVAL_ROLE="${EVAL_ROLE:-}"
case "${EVAL_ROLE}" in
  base|grpo) ;;
  *)
    echo "ERROR: set EVAL_ROLE=base or EVAL_ROLE=grpo" >&2
    usage >&2
    exit 2
    ;;
esac

K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
JOB_NAME="${JOB_NAME:-jiaxicao-swe484-eval-${EVAL_ROLE}}"
WORKLOAD="${WORKLOAD:-${JOB_NAME}}"
IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/youtu-agent:slime-nightly-dev-20260530a-efa-swe-mooncake}"
SLIME_DIR="${SLIME_DIR:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl}"
# Official 484 Verified rows with image/workdir/eval_cmd under extra_info
# (parquet variant nests image under sandbox_overrides and often lacks eval_cmd).
EVAL_DATA="${EVAL_DATA:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/data/swe484_verified.extra_info.slime.jsonl}"
AGS_SECRET_NAME="${AGS_SECRET_NAME:-qwen35-9b-ags-credentials}"
SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-exb9o2gb}"
# SWE Verified pass@k: K independent samples per task (default pass@4).
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"

GRPO_CKPT_DIR="${GRPO_CKPT_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_1node_grpo_debug/slime_save}"

if [[ "${EVAL_ROLE}" == "base" ]]; then
  EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_swe484_eval_base}"
  LOAD_PATH="${LOAD_PATH:-${REF_MODEL_PATH}}"
else
  EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_swe484_eval_grpo}"
  LOAD_PATH="${LOAD_PATH:-${GRPO_CKPT_DIR}}"
fi
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"

WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"

SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL:-}"
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete pytorchjob "${JOB_NAME}" --ignore-not-found
  echo "Deleted pytorchjob/${JOB_NAME} in ${K8S_NAMESPACE}"
  exit 0
fi

for p in "${SLIME_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}" "${EVAL_DATA}" "${LOAD_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: path not found: ${p}" >&2
    exit 1
  fi
done

if [[ -z "${SLIME_ADAPTER_PUBLIC_URL}" ]]; then
  ADDR="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${JOB_NAME}-adapter" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
  if [[ -n "${ADDR}" ]]; then
    SLIME_ADAPTER_PUBLIC_URL="http://${ADDR}"
  else
    echo "ERROR: set SLIME_ADAPTER_PUBLIC_URL or apply submit_alb.sh first for ${JOB_NAME}-adapter" >&2
    exit 1
  fi
fi

if ! kubectl -n "${K8S_NAMESPACE}" get secret "${AGS_SECRET_NAME}" >/dev/null 2>&1; then
  echo "ERROR: missing secret ${AGS_SECRET_NAME} in ${K8S_NAMESPACE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}/slime_save" "${LOG_DIR}/wandb" "${LOG_DIR}/rollout_dumps" "${LOG_DIR}/launcher_logs"

export JOB_NAME K8S_NAMESPACE IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export LOAD_PATH SAVE_PATH PROMPT_DATA EVAL_DATA EXP_TAG LOG_DIR
export AGS_SECRET_NAME SLIME_ADAPTER_PUBLIC_URL SLIME_AGENT_AGS_TOOL_ID
export N_SAMPLES_PER_EVAL_PROMPT
export WANDB_PROJECT WANDB_GROUP WANDB_TEAM
export WORKLOAD EVAL_ROLE

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${K8S_NAMESPACE} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${LOAD_PATH} ${SAVE_PATH} ${PROMPT_DATA} ${EVAL_DATA} ${EXP_TAG} ${LOG_DIR} ${AGS_SECRET_NAME} ${SLIME_ADAPTER_PUBLIC_URL} ${SLIME_AGENT_AGS_TOOL_ID} ${N_SAMPLES_PER_EVAL_PROMPT} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM} ${WORKLOAD} ${EVAL_ROLE}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} (role=${EVAL_ROLE}) in ${K8S_NAMESPACE}"
echo "    image: ${IMAGE_URI}"
echo "    LOAD_PATH=${LOAD_PATH}"
echo "    SAVE_PATH=${SAVE_PATH}"
echo "    LOG_DIR=${LOG_DIR}"
echo "    SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "    EVAL_DATA=${EVAL_DATA}"
echo "    N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT} (pass@k)"

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
echo "  kubectl -n ${K8S_NAMESPACE} logs -f -l training.kubeflow.org/job-name=${JOB_NAME},training.kubeflow.org/replica-type=master"
echo "ALB /health (after adapter up):"
echo "  curl -fsS ${SLIME_ADAPTER_PUBLIC_URL}/health"
