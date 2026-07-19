#!/usr/bin/env bash
# Render + apply / delete the Path A naive GRPO 2-node / 16-GPU PyTorchJob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pytorchjob.yaml.template"

usage() {
  cat <<'EOF'
Usage: submit_job.sh [--delete] [--dry-run] [--help]

Creates a 2×8GPU PyTorchJob (Master + Worker) for Path A naive GRPO.
Master labels match grpo ALB Service (app=cc-ags-recorder, workload=JOB_NAME).

Useful env:
  JOB_NAME                 jiaxicao-grpo-2node-c64t45
  K8S_NAMESPACE            sn5-system-intern
  IMAGE_URI                youtu-agent slime-nightly (default)
  SLIME_DIR                cc-ags-swe worktree
  SLIME_ADAPTER_PUBLIC_URL GRPO ALB URL (auto from Ingress if unset)
  SLIME_AGENT_AGS_TOOL_ID  default sdt-exb9o2gb
  PHASE                    all|train|eval (default train)
  AGS_SECRET_NAME          qwen35-9b-ags-credentials
  EXP_TAG / LOG_DIR        default qwen35_9b_cc_ags_2node_grpo_c64_t45
  LOAD_PATH                checkpoint directory to read (default LOG_DIR/slime_save)
  SAVE_PATH                checkpoint directory to write (default LOG_DIR/slime_save)
  RUN_ROOT                 rollout dump root (default LOG_DIR)
  RESUME_DEBUG_ROLLOUT_DATA 1 to reuse an existing dump in RUN_ROOT (default 1)
  ROLLOUT_BATCH_SIZE       default 16
  N_SAMPLES_PER_PROMPT     default 8 (GBS = RBS * n_samples)
  NUM_ROLLOUT              default 44 (~2 epochs @ RBS=16)
  SLIME_CC_TIME_BUDGET_SEC agent budget in seconds (default 2700)
  SLIME_CC_AGENT_CONCURRENCY global agent gate; <=0 disables it (default 64)
  SLIME_CC_TOOL_LOOP_PENALTY 1 enables the >=3 repeated-signature override
  SLIME_CC_TIMEOUT_OUTCOME_REWARD 1 enables timeout/resolved reward levels
  ACTOR_NUM_NODES          default 2
  WORKER_REPLICAS          default 1
  ALLOW_RESUME             1 to allow submitting into a LOG_DIR that already has ckpts
  WANDB_KEY                optional if ~/.config/jiaxicao/wandb_api_key exists
  WANDB_PROJECT            coding-rl
  WANDB_TEAM               models-tencent7723 (entity)
  WANDB_GROUP              experiment group (default: EXP_TAG)
  WANDB_RESUME_FROM        optional W&B rewind point: RUN_ID?_step=N
  WANDB_RUN_ID             optional existing W&B run ID
  WANDB_RESUME             optional W&B resume policy, e.g. must
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
JOB_NAME="${JOB_NAME:-jiaxicao-grpo-2node-c64t45}"
INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"
WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/youtu-agent:slime-nightly-dev-20260530a-efa-swe-mooncake}"
SLIME_DIR="${SLIME_DIR:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl}"
EVAL_DATA="${EVAL_DATA:-/mnt/sn-007/youtu-agent/yuleiqin/SWE_code/DataEng/RL_DATA/data_valid/swe_agent_ags_swebench_verified/test.parquet}"
EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_2node_grpo_c64_t45}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
LOAD_PATH="${LOAD_PATH:-${LOG_DIR}/slime_save}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"
RESUME_DEBUG_ROLLOUT_DATA="${RESUME_DEBUG_ROLLOUT_DATA:-1}"
PHASE="${PHASE:-train}"
AGS_SECRET_NAME="${AGS_SECRET_NAME:-qwen35-9b-ags-credentials}"
SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-exb9o2gb}"
NUM_ROLLOUT="${NUM_ROLLOUT:-44}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
if [[ -z "${WORKER_REPLICAS:-}" ]]; then
  if [[ "${ACTOR_NUM_NODES}" -gt 1 ]]; then
    WORKER_REPLICAS=$((ACTOR_NUM_NODES - 1))
  else
    WORKER_REPLICAS=0
  fi
fi
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))}"
SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-2700}"
SLIME_CC_AGENT_CONCURRENCY="${SLIME_CC_AGENT_CONCURRENCY:-64}"
SLIME_CC_TOOL_LOOP_PENALTY="${SLIME_CC_TOOL_LOOP_PENALTY:-0}"
SLIME_CC_TIMEOUT_OUTCOME_REWARD="${SLIME_CC_TIMEOUT_OUTCOME_REWARD:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
WANDB_RESUME_FROM="${WANDB_RESUME_FROM:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
_load_wandb_key
WANDB_KEY="${WANDB_KEY:-}"
if [[ -z "${WANDB_GROUP:-}" || "${WANDB_GROUP}" == "qwen35_9b_cc_ags_1node_grpo_debug" ]]; then
  WANDB_GROUP="${EXP_TAG}"
fi

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete pytorchjob "${JOB_NAME}" --ignore-not-found
  echo "Deleted pytorchjob/${JOB_NAME} in ${K8S_NAMESPACE}"
  exit 0
fi

ADDR="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${INGRESS_NAME}" \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
if [[ -n "${ADDR}" ]]; then
  SLIME_ADAPTER_PUBLIC_URL="http://${ADDR}"
  echo "Using Ingress ${INGRESS_NAME} → ${SLIME_ADAPTER_PUBLIC_URL}"
elif [[ -z "${SLIME_ADAPTER_PUBLIC_URL:-}" || "${SLIME_ADAPTER_PUBLIC_URL}" == *REPLACE_WITH* ]]; then
  echo "ERROR: set SLIME_ADAPTER_PUBLIC_URL or apply ALB first:" >&2
  echo "  NAME=${JOB_NAME}-adapter WORKLOAD_LABEL=${WORKLOAD_LABEL} \\" >&2
  echo "    bash examples/claudecode_ags/launch/hybrid_adapter_alb/submit_alb.sh" >&2
  exit 3
fi
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"

for p in "${SLIME_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}" "${EVAL_DATA}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: path not found: ${p}" >&2
    exit 1
  fi
done

if [[ ! -f "${SLIME_DIR}/examples/claudecode_ags/launch/run_grpo_1node_debug.sh" ]]; then
  echo "ERROR: GRPO launcher missing under SLIME_DIR=${SLIME_DIR}" >&2
  exit 1
fi

if ! kubectl -n "${K8S_NAMESPACE}" get secret "${AGS_SECRET_NAME}" >/dev/null 2>&1; then
  echo "ERROR: missing secret ${AGS_SECRET_NAME} in ${K8S_NAMESPACE}" >&2
  exit 1
fi

if ! kubectl -n "${K8S_NAMESPACE}" get ingress "${INGRESS_NAME}" >/dev/null 2>&1; then
  echo "WARNING: adapter Ingress ${INGRESS_NAME} not found; apply ALB first." >&2
fi

if [[ -z "${WANDB_KEY}" ]]; then
  echo "ERROR: WANDB_KEY missing. Set WANDB_KEY or put the key in ${WANDB_KEY_FILE:-$HOME/.config/jiaxicao/wandb_api_key}" >&2
  exit 1
fi

if [[ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "ERROR: ${SAVE_PATH} already has a checkpoint." >&2
  echo "       Use a fresh EXP_TAG/LOG_DIR, or set ALLOW_RESUME=1 to continue." >&2
  exit 4
fi

_expected_gbs=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [[ "${GLOBAL_BATCH_SIZE}" -ne "${_expected_gbs}" ]]; then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must equal RBS*n_samples=${_expected_gbs}" >&2
  exit 2
fi

export JOB_NAME WORKLOAD_LABEL K8S_NAMESPACE IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export PROMPT_DATA EVAL_DATA EXP_TAG LOG_DIR LOAD_PATH SAVE_PATH RUN_ROOT RESUME_DEBUG_ROLLOUT_DATA
export PHASE AGS_SECRET_NAME SLIME_ADAPTER_PUBLIC_URL
export SLIME_AGENT_AGS_TOOL_ID
export WANDB_KEY WANDB_PROJECT WANDB_GROUP WANDB_TEAM WANDB_RESUME_FROM WANDB_RUN_ID WANDB_RESUME
export NUM_ROLLOUT SAVE_INTERVAL GLOBAL_BATCH_SIZE ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT
export ACTOR_NUM_NODES WORKER_REPLICAS ACTOR_NUM_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export SLIME_CC_TIME_BUDGET_SEC SLIME_CC_AGENT_CONCURRENCY SLIME_CC_TOOL_LOOP_PENALTY
export SLIME_CC_TIMEOUT_OUTCOME_REWARD

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${WORKLOAD_LABEL} ${K8S_NAMESPACE} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${PROMPT_DATA} ${EVAL_DATA} ${EXP_TAG} ${LOG_DIR} ${LOAD_PATH} ${SAVE_PATH} ${RUN_ROOT} ${RESUME_DEBUG_ROLLOUT_DATA} ${PHASE} ${AGS_SECRET_NAME} ${SLIME_ADAPTER_PUBLIC_URL} ${SLIME_AGENT_AGS_TOOL_ID} ${WANDB_KEY} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM} ${WANDB_RESUME_FROM} ${WANDB_RUN_ID} ${WANDB_RESUME} ${NUM_ROLLOUT} ${SAVE_INTERVAL} ${GLOBAL_BATCH_SIZE} ${ROLLOUT_BATCH_SIZE} ${N_SAMPLES_PER_PROMPT} ${ACTOR_NUM_NODES} ${WORKER_REPLICAS} ${ACTOR_NUM_GPUS_PER_NODE} ${ROLLOUT_NUM_GPUS} ${SLIME_CC_TIME_BUDGET_SEC} ${SLIME_CC_AGENT_CONCURRENCY} ${SLIME_CC_TOOL_LOOP_PENALTY} ${SLIME_CC_TIMEOUT_OUTCOME_REWARD}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} in ${K8S_NAMESPACE}"
echo "    image: ${IMAGE_URI}"
echo "    SLIME_DIR=${SLIME_DIR}"
echo "    SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "    SLIME_AGENT_AGS_TOOL_ID=${SLIME_AGENT_AGS_TOOL_ID}"
echo "    PHASE=${PHASE} LOG_DIR=${LOG_DIR}"
echo "    LOAD_PATH=${LOAD_PATH}"
echo "    SAVE_PATH=${SAVE_PATH} RUN_ROOT=${RUN_ROOT}"
echo "    PROMPT_DATA=${PROMPT_DATA}"
echo "    NUM_ROLLOUT=${NUM_ROLLOUT} SAVE_INTERVAL=${SAVE_INTERVAL} RBS=${ROLLOUT_BATCH_SIZE} n_samples=${N_SAMPLES_PER_PROMPT} GBS=${GLOBAL_BATCH_SIZE}"
echo "    nodes=${ACTOR_NUM_NODES} workers=${WORKER_REPLICAS} gpus/node=${ACTOR_NUM_GPUS_PER_NODE} rollout_gpus=${ROLLOUT_NUM_GPUS}"
echo "    agent_budget=${SLIME_CC_TIME_BUDGET_SEC}s agent_concurrency=${SLIME_CC_AGENT_CONCURRENCY} tool_loop_penalty=${SLIME_CC_TOOL_LOOP_PENALTY} timeout_outcome_reward=${SLIME_CC_TIMEOUT_OUTCOME_REWARD}"
echo "    WANDB project=${WANDB_PROJECT} team=${WANDB_TEAM} group=${WANDB_GROUP} key=***"

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
echo "ALB /health (after adapter up):"
echo "  curl -fsS ${SLIME_ADAPTER_PUBLIC_URL}/health"
