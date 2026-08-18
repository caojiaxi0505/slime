#!/usr/bin/env bash
# Render + apply / delete the turn-level teacher SFT 1-node PyTorchJob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pytorchjob.yaml.template"

usage() {
  cat <<'EOF'
Usage: submit_job.sh [--delete] [--dry-run] [--help]

Creates a 1×8GPU PyTorchJob (Master only) for fail imitation learning.
Master labels match teacher_sft_adapter_alb Service (app=cc-ags-recorder,
workload=jiaxicao-fail-imitation-learning).

Each optimizer update consumes exactly 16 raw SWE tasks.  Every task launches
8 student trials; its teacher relabels start as soon as those 8 trials finish,
while training waits for the complete 16-task barrier.

Useful env:
  JOB_NAME                        jiaxicao-fail-imitation-learning
  K8S_NAMESPACE                   sn5-system-intern
  K8S_NODE_GROUP                  shennong-5
  WANDB_PROJECT                   imitation-sft
  WANDB_GROUP                     fail_imitation_learning (default: EXP_TAG)
  LEARNING_RATE                   3e-6
  SLIME_ADAPTER_PUBLIC_URL        student ALB (auto from Ingress if unset)
  SLIME_TEACHER_ADAPTER_PUBLIC_URL teacher ALB (auto from Ingress if unset)
  SLIME_AGENT_SFT_LOG_DIR          Stage-1 request logs (default: RUN_ROOT/student_sft_turns)
  SLIME_TEACHER_SFT_LOG_DIR        Teacher request logs (default: RUN_ROOT/teacher_sft_turns)
  STEP_GRPO_TEACHER_SFT_MODE      sft_only (default) | hybrid
  STEP_GRPO_TEACHER_MAX_STEPS     2
  STEP_GRPO_TEACHER_TURN_SELECT   all
  STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL  0
  SLIME_REMOTE_OPENAI_BASE_URL / MODEL  from slime_ags.env if unset
  SLIME_REMOTE_OPENAI_API_KEY     must be in gitignored slime_ags.env (FSx)
  SLIME_REMOTE_OPENAI_KEEPALIVE_SEC  60 by default; set 0 to disable
  SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT  4 by default
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

_load_teacher_openai_from_env_file() {
  local f="${SLIME_DIR}/examples/claudecode_ags/env/slime_ags.env"
  if [[ ! -f "${f}" ]]; then
    return 0
  fi
  local line key val
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" == \#* || -z "${line}" ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    case "${key}" in
      SLIME_REMOTE_OPENAI_BASE_URL)
        SLIME_REMOTE_OPENAI_BASE_URL="${SLIME_REMOTE_OPENAI_BASE_URL:-${val}}"
        ;;
      SLIME_REMOTE_OPENAI_MODEL)
        SLIME_REMOTE_OPENAI_MODEL="${SLIME_REMOTE_OPENAI_MODEL:-${val}}"
        ;;
      SLIME_REMOTE_OPENAI_API_KEY)
        if [[ -n "${val}" ]]; then
          TEACHER_API_KEY_SET=1
        fi
        ;;
    esac
  done < "${f}"
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
K8S_NODE_GROUP="${K8S_NODE_GROUP:-shennong-5}"
JOB_NAME="${JOB_NAME:-jiaxicao-fail-imitation-learning}"
WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"
TEACHER_INGRESS_NAME="${TEACHER_INGRESS_NAME:-${JOB_NAME}-teacher}"
IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/youtu-agent:slime-nightly-dev-20260530a-efa-swe-mooncake}"
SLIME_DIR="${SLIME_DIR:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl}"
EVAL_DATA="${EVAL_DATA:-/mnt/sn-007/youtu-agent/yuleiqin/SWE_code/DataEng/RL_DATA/data_valid/swe_agent_ags_swebench_verified/test.parquet}"
EXP_TAG="${EXP_TAG:-fail_imitation_learning}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
LOAD_PATH="${LOAD_PATH:-${LOG_DIR}/slime_save}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"
SLIME_AGENT_SFT_LOG_DIR="${SLIME_AGENT_SFT_LOG_DIR:-${RUN_ROOT}/student_sft_turns}"
SLIME_TEACHER_SFT_LOG_DIR="${SLIME_TEACHER_SFT_LOG_DIR:-${RUN_ROOT}/teacher_sft_turns}"
LOAD_CKPT_STEP="${LOAD_CKPT_STEP:-}"
RESUME_DEBUG_ROLLOUT_DATA="${RESUME_DEBUG_ROLLOUT_DATA:-1}"
PHASE="${PHASE:-train}"
AGS_SECRET_NAME="${AGS_SECRET_NAME:-qwen35-9b-ags-credentials}"
SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-ltpatoxb}"
STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-${STEP_GRPO_BRANCH_CONCURRENCY:-64}}"
STEP_GRPO_BRANCH_CONCURRENCY="${STEP_GRPO_BRANCH_SUBMIT_BATCH}"
STEP_GRPO_PPL_CLIP="${STEP_GRPO_PPL_CLIP:-20}"
STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"
STEP_GRPO_STAGE1_LOSS_WEIGHT="${STEP_GRPO_STAGE1_LOSS_WEIGHT:-1.0}"
STEP_GRPO_BRANCH_LOSS_WEIGHT="${STEP_GRPO_BRANCH_LOSS_WEIGHT:-1.0}"
STEP_GRPO_STAGE2_LOSS_SCOPE="${STEP_GRPO_STAGE2_LOSS_SCOPE:-full_continuation}"
STEP_GRPO_TEACHER_SFT_MODE="${STEP_GRPO_TEACHER_SFT_MODE:-sft_only}"
STEP_GRPO_TEACHER_MAX_STEPS="${STEP_GRPO_TEACHER_MAX_STEPS:-2}"
STEP_GRPO_TEACHER_TURN_SELECT="${STEP_GRPO_TEACHER_TURN_SELECT:-all}"
STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL="${STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL:-0}"
SLIME_REMOTE_OPENAI_MAX_INFLIGHT="${SLIME_REMOTE_OPENAI_MAX_INFLIGHT:-64}"
SLIME_REMOTE_OPENAI_CONNECTOR_LIMIT="${SLIME_REMOTE_OPENAI_CONNECTOR_LIMIT:-128}"
SLIME_REMOTE_OPENAI_KEEPALIVE_SEC="${SLIME_REMOTE_OPENAI_KEEPALIVE_SEC:-60}"
SLIME_REMOTE_OPENAI_KEEPALIVE_MAX_TOKENS="${SLIME_REMOTE_OPENAI_KEEPALIVE_MAX_TOKENS:-1}"
SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT="${SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT:-4}"
SLIME_REMOTE_OPENAI_THINKING_TYPE="${SLIME_REMOTE_OPENAI_THINKING_TYPE:-enabled}"
SLIME_REMOTE_OPENAI_REASONING_EFFORT="${SLIME_REMOTE_OPENAI_REASONING_EFFORT:-max}"
SLIME_AGENT_AGS_START_CONCURRENCY="${SLIME_AGENT_AGS_START_CONCURRENCY:-16}"
SLIME_AGENT_AGS_THREADPOOL_WORKERS="${SLIME_AGENT_AGS_THREADPOOL_WORKERS:-16}"
STEP_GRPO_BUNDLE_DIR="${STEP_GRPO_BUNDLE_DIR:-${LOG_DIR}/step_reconstruct_bundles}"
NUM_ROLLOUT="${NUM_ROLLOUT:-88}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-2700}"
SLIME_CC_GENERATE_GUARD_SEC="${SLIME_CC_GENERATE_GUARD_SEC:-2700}"
SLIME_CC_AGENT_CONCURRENCY="${SLIME_CC_AGENT_CONCURRENCY:-64}"
SLIME_CC_EVAL_CONCURRENCY="${SLIME_CC_EVAL_CONCURRENCY:-0}"
SLIME_CC_EVAL_CONTROL_CONCURRENCY="${SLIME_CC_EVAL_CONTROL_CONCURRENCY:-64}"
SLIME_CC_EVAL_POLL_INTERVAL_SEC="${SLIME_CC_EVAL_POLL_INTERVAL_SEC:-60}"
SLIME_CC_EVAL_POLL_JITTER_SEC="${SLIME_CC_EVAL_POLL_JITTER_SEC:-10}"
SLIME_CC_EVAL_COMPLETION_GRACE_SEC="${SLIME_CC_EVAL_COMPLETION_GRACE_SEC:-90}"
SLIME_CC_TOOL_LOOP_PENALTY="${SLIME_CC_TOOL_LOOP_PENALTY:-0}"
SLIME_CC_TIMEOUT_OUTCOME_REWARD="${SLIME_CC_TIMEOUT_OUTCOME_REWARD:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-imitation-sft}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
WANDB_FORK_FROM="${WANDB_FORK_FROM:-}"
WANDB_RESUME_FROM="${WANDB_RESUME_FROM:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-auto}"
_load_wandb_key
WANDB_KEY="${WANDB_KEY:-}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
TEACHER_API_KEY_SET=0
_load_teacher_openai_from_env_file
SLIME_REMOTE_OPENAI_BASE_URL="${SLIME_REMOTE_OPENAI_BASE_URL:-}"
SLIME_REMOTE_OPENAI_MODEL="${SLIME_REMOTE_OPENAI_MODEL:-}"

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
  echo "ERROR: set SLIME_ADAPTER_PUBLIC_URL or apply teacher_sft_adapter_alb first:" >&2
  echo "  bash examples/claudecode_ags/launch/teacher_sft_adapter_alb/submit_alb.sh" >&2
  exit 3
fi
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"

TEACHER_ADDR="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${TEACHER_INGRESS_NAME}" \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
if [[ -n "${TEACHER_ADDR}" ]]; then
  SLIME_TEACHER_ADAPTER_PUBLIC_URL="http://${TEACHER_ADDR}"
  echo "Using Ingress ${TEACHER_INGRESS_NAME} → ${SLIME_TEACHER_ADAPTER_PUBLIC_URL}"
elif [[ -z "${SLIME_TEACHER_ADAPTER_PUBLIC_URL:-}" || "${SLIME_TEACHER_ADAPTER_PUBLIC_URL}" == *REPLACE_WITH* ]]; then
  echo "ERROR: set SLIME_TEACHER_ADAPTER_PUBLIC_URL or apply teacher_sft_adapter_alb first:" >&2
  echo "  bash examples/claudecode_ags/launch/teacher_sft_adapter_alb/submit_alb.sh" >&2
  exit 3
fi
SLIME_TEACHER_ADAPTER_PUBLIC_URL="${SLIME_TEACHER_ADAPTER_PUBLIC_URL%/}"

for p in "${SLIME_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}" "${EVAL_DATA}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: path not found: ${p}" >&2
    exit 1
  fi
done

if [[ ! -f "${SLIME_DIR}/examples/claudecode_ags/launch/run_hybrid_1node_debug.sh" ]]; then
  echo "ERROR: hybrid launcher missing under SLIME_DIR=${SLIME_DIR}" >&2
  exit 1
fi

if ! kubectl -n "${K8S_NAMESPACE}" get secret "${AGS_SECRET_NAME}" >/dev/null 2>&1; then
  echo "ERROR: missing secret ${AGS_SECRET_NAME} in ${K8S_NAMESPACE}" >&2
  exit 1
fi

if [[ -z "${WANDB_KEY}" ]]; then
  echo "ERROR: WANDB_KEY missing. Set WANDB_KEY or put the key in ${WANDB_KEY_FILE:-$HOME/.config/jiaxicao/wandb_api_key}" >&2
  exit 1
fi

if [[ -z "${SLIME_REMOTE_OPENAI_BASE_URL}" || -z "${SLIME_REMOTE_OPENAI_MODEL}" ]]; then
  echo "ERROR: SLIME_REMOTE_OPENAI_BASE_URL and SLIME_REMOTE_OPENAI_MODEL are required" >&2
  exit 1
fi
if [[ "${TEACHER_API_KEY_SET}" != "1" ]]; then
  echo "ERROR: SLIME_REMOTE_OPENAI_API_KEY is empty in ${SLIME_DIR}/examples/claudecode_ags/env/slime_ags.env" >&2
  echo "       Put the key in that gitignored file (Job loads it from FSx; it is not copied into the YAML)." >&2
  exit 1
fi

if [[ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "ERROR: ${SAVE_PATH} already has a checkpoint." >&2
  echo "       Use a fresh EXP_TAG/LOG_DIR, or set ALLOW_RESUME=1 to continue." >&2
  exit 4
fi
if [[ -n "${LOAD_CKPT_STEP}" ]]; then
  if [[ ! "${LOAD_CKPT_STEP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LOAD_CKPT_STEP must be a non-negative integer, got: ${LOAD_CKPT_STEP}" >&2
    exit 2
  fi
  _iter_dir="${LOAD_PATH}/iter_$(printf '%07d' "${LOAD_CKPT_STEP}")"
  _dataset_state="${LOAD_PATH}/rollout/global_dataset_state_dict_${LOAD_CKPT_STEP}.pt"
  if [[ ! -d "${_iter_dir}" || ! -f "${_dataset_state}" ]]; then
    echo "ERROR: exact resume requires ${_iter_dir} and ${_dataset_state}" >&2
    exit 1
  fi
fi

if [[ "${STEP_GRPO_TEACHER_SFT_MODE}" == "sft_only" ]]; then
  if [[ "${ROLLOUT_BATCH_SIZE}" != "16" || "${GLOBAL_BATCH_SIZE}" != "16" || \
        "${STEP_GRPO_HYBRID_K}" != "8" ]]; then
    echo "ERROR: sft_only fixed-task scheduling requires RBS=16, GBS=16, and K=8" >&2
    echo "       got RBS=${ROLLOUT_BATCH_SIZE} GBS=${GLOBAL_BATCH_SIZE} K=${STEP_GRPO_HYBRID_K}" >&2
    exit 2
  fi
fi

export JOB_NAME WORKLOAD_LABEL K8S_NAMESPACE K8S_NODE_GROUP IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export PROMPT_DATA EVAL_DATA EXP_TAG LOG_DIR LOAD_PATH SAVE_PATH RUN_ROOT LOAD_CKPT_STEP
export SLIME_AGENT_SFT_LOG_DIR SLIME_TEACHER_SFT_LOG_DIR
export RESUME_DEBUG_ROLLOUT_DATA PHASE AGS_SECRET_NAME SLIME_ADAPTER_PUBLIC_URL SLIME_TEACHER_ADAPTER_PUBLIC_URL
export SLIME_AGENT_AGS_TOOL_ID SLIME_AGENT_AGS_START_CONCURRENCY SLIME_AGENT_AGS_THREADPOOL_WORKERS
export WANDB_KEY WANDB_PROJECT WANDB_GROUP WANDB_TEAM WANDB_FORK_FROM WANDB_RESUME_FROM WANDB_RUN_ID WANDB_RESUME
export STEP_GRPO_HYBRID_K STEP_GRPO_BRANCH_SUBMIT_BATCH STEP_GRPO_BRANCH_CONCURRENCY STEP_GRPO_PPL_CLIP STEP_GRPO_FILTER STEP_GRPO_STAGE1_LOSS_WEIGHT STEP_GRPO_BRANCH_LOSS_WEIGHT STEP_GRPO_STAGE2_LOSS_SCOPE
export STEP_GRPO_TEACHER_SFT_MODE STEP_GRPO_TEACHER_MAX_STEPS STEP_GRPO_TEACHER_TURN_SELECT STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL
export SLIME_REMOTE_OPENAI_BASE_URL SLIME_REMOTE_OPENAI_MODEL SLIME_REMOTE_OPENAI_MAX_INFLIGHT SLIME_REMOTE_OPENAI_CONNECTOR_LIMIT
export SLIME_REMOTE_OPENAI_KEEPALIVE_SEC SLIME_REMOTE_OPENAI_KEEPALIVE_MAX_TOKENS SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT
export SLIME_REMOTE_OPENAI_THINKING_TYPE SLIME_REMOTE_OPENAI_REASONING_EFFORT
export STEP_GRPO_BUNDLE_DIR NUM_ROLLOUT SAVE_INTERVAL GLOBAL_BATCH_SIZE ROLLOUT_BATCH_SIZE
export LEARNING_RATE
export SLIME_CC_TIME_BUDGET_SEC SLIME_CC_GENERATE_GUARD_SEC
export SLIME_CC_AGENT_CONCURRENCY SLIME_CC_EVAL_CONCURRENCY
export SLIME_CC_EVAL_CONTROL_CONCURRENCY SLIME_CC_EVAL_POLL_INTERVAL_SEC
export SLIME_CC_EVAL_POLL_JITTER_SEC SLIME_CC_EVAL_COMPLETION_GRACE_SEC
export SLIME_CC_TOOL_LOOP_PENALTY SLIME_CC_TIMEOUT_OUTCOME_REWARD

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${WORKLOAD_LABEL} ${K8S_NAMESPACE} ${K8S_NODE_GROUP} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${PROMPT_DATA} ${EVAL_DATA} ${EXP_TAG} ${LOG_DIR} ${LOAD_PATH} ${SAVE_PATH} ${RUN_ROOT} ${SLIME_AGENT_SFT_LOG_DIR} ${SLIME_TEACHER_SFT_LOG_DIR} ${LOAD_CKPT_STEP} ${RESUME_DEBUG_ROLLOUT_DATA} ${PHASE} ${AGS_SECRET_NAME} ${SLIME_ADAPTER_PUBLIC_URL} ${SLIME_TEACHER_ADAPTER_PUBLIC_URL} ${SLIME_AGENT_AGS_TOOL_ID} ${SLIME_AGENT_AGS_START_CONCURRENCY} ${SLIME_AGENT_AGS_THREADPOOL_WORKERS} ${WANDB_KEY} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM} ${WANDB_FORK_FROM} ${WANDB_RESUME_FROM} ${WANDB_RUN_ID} ${WANDB_RESUME} ${STEP_GRPO_HYBRID_K} ${STEP_GRPO_BRANCH_SUBMIT_BATCH} ${STEP_GRPO_BRANCH_CONCURRENCY} ${STEP_GRPO_PPL_CLIP} ${STEP_GRPO_FILTER} ${STEP_GRPO_STAGE1_LOSS_WEIGHT} ${STEP_GRPO_BRANCH_LOSS_WEIGHT} ${STEP_GRPO_STAGE2_LOSS_SCOPE} ${STEP_GRPO_TEACHER_SFT_MODE} ${STEP_GRPO_TEACHER_MAX_STEPS} ${STEP_GRPO_TEACHER_TURN_SELECT} ${STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL} ${SLIME_REMOTE_OPENAI_BASE_URL} ${SLIME_REMOTE_OPENAI_MODEL} ${SLIME_REMOTE_OPENAI_MAX_INFLIGHT} ${SLIME_REMOTE_OPENAI_CONNECTOR_LIMIT} ${SLIME_REMOTE_OPENAI_KEEPALIVE_SEC} ${SLIME_REMOTE_OPENAI_KEEPALIVE_MAX_TOKENS} ${SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT} ${SLIME_REMOTE_OPENAI_THINKING_TYPE} ${SLIME_REMOTE_OPENAI_REASONING_EFFORT} ${STEP_GRPO_BUNDLE_DIR} ${NUM_ROLLOUT} ${SAVE_INTERVAL} ${GLOBAL_BATCH_SIZE} ${ROLLOUT_BATCH_SIZE} ${LEARNING_RATE} ${SLIME_CC_TIME_BUDGET_SEC} ${SLIME_CC_GENERATE_GUARD_SEC} ${SLIME_CC_AGENT_CONCURRENCY} ${SLIME_CC_EVAL_CONCURRENCY} ${SLIME_CC_EVAL_CONTROL_CONCURRENCY} ${SLIME_CC_EVAL_POLL_INTERVAL_SEC} ${SLIME_CC_EVAL_POLL_JITTER_SEC} ${SLIME_CC_EVAL_COMPLETION_GRACE_SEC} ${SLIME_CC_TOOL_LOOP_PENALTY} ${SLIME_CC_TIMEOUT_OUTCOME_REWARD}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} in ${K8S_NAMESPACE}"
echo "    node group: ${K8S_NODE_GROUP}"
echo "    image: ${IMAGE_URI}"
echo "    SLIME_DIR=${SLIME_DIR}"
echo "    SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "    SLIME_TEACHER_ADAPTER_PUBLIC_URL=${SLIME_TEACHER_ADAPTER_PUBLIC_URL}"
echo "    SLIME_AGENT_AGS_TOOL_ID=${SLIME_AGENT_AGS_TOOL_ID}"
echo "    teacher model=${SLIME_REMOTE_OPENAI_MODEL} thinking=${SLIME_REMOTE_OPENAI_THINKING_TYPE} effort=${SLIME_REMOTE_OPENAI_REASONING_EFFORT} key=from slime_ags.env"
echo "    teacher keepalive=${SLIME_REMOTE_OPENAI_KEEPALIVE_SEC}s max_tokens=${SLIME_REMOTE_OPENAI_KEEPALIVE_MAX_TOKENS} inflight=${SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT}"
echo "    teacher mode=${STEP_GRPO_TEACHER_SFT_MODE} max_steps=${STEP_GRPO_TEACHER_MAX_STEPS} select=${STEP_GRPO_TEACHER_TURN_SELECT} max_per_trial=${STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL}"
echo "    PHASE=${PHASE} LOG_DIR=${LOG_DIR}"
echo "    LOAD_PATH=${LOAD_PATH} LOAD_CKPT_STEP=${LOAD_CKPT_STEP:-latest}"
echo "    SAVE_PATH=${SAVE_PATH} RUN_ROOT=${RUN_ROOT}"
echo "    student SFT logs=${SLIME_AGENT_SFT_LOG_DIR}"
echo "    teacher SFT logs=${SLIME_TEACHER_SFT_LOG_DIR}"
echo "    PROMPT_DATA=${PROMPT_DATA}"
echo "    NUM_ROLLOUT=${NUM_ROLLOUT} SAVE_INTERVAL=${SAVE_INTERVAL} RBS=${ROLLOUT_BATCH_SIZE} GBS=${GLOBAL_BATCH_SIZE}"
echo "    LEARNING_RATE=${LEARNING_RATE}"
echo "    hybrid K=${STEP_GRPO_HYBRID_K} agent_concurrency=${SLIME_CC_AGENT_CONCURRENCY}"
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
echo "ALB /health (after adapters up):"
echo "  curl -fsS ${SLIME_ADAPTER_PUBLIC_URL}/health"
echo "  curl -fsS ${SLIME_TEACHER_ADAPTER_PUBLIC_URL}/health"
echo "W&B: https://wandb.ai/${WANDB_TEAM}/${WANDB_PROJECT}/groups/${WANDB_GROUP}"
