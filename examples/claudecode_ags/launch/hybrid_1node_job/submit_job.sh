#!/usr/bin/env bash
# Render + apply / delete the Path A hybrid step-GRPO 1-node debug PyTorchJob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pytorchjob.yaml.template"
INGRESS_NAME="${INGRESS_NAME:-${JOB_NAME}-adapter}"

usage() {
  cat <<'EOF'
Usage: submit_job.sh [--delete] [--dry-run] [--help]

Creates a 1×8GPU PyTorchJob (Master only) for Path A hybrid step-GRPO.
Master labels match hybrid_adapter_alb Service (app=cc-ags-recorder,
workload=jiaxicao-hybrid-1node-debug).

Useful env:
  JOB_NAME                 jiaxicao-hybrid-1node-debug
  K8S_NAMESPACE            sn5-system-intern
  K8S_NODE_GROUP           shennong-5 (set shennong-5-dev to target dev pool)
  IMAGE_URI                youtu-agent slime-nightly (default)
  SLIME_DIR                cc-ags-swe worktree
  SLIME_ADAPTER_PUBLIC_URL hybrid ALB URL (auto from Ingress if unset)
  SLIME_AGENT_AGS_TOOL_ID  default sdt-ltpatoxb (separate from GRPO sdt-exb9o2gb)
  PHASE                    all|train|eval (default train)
  AGS_SECRET_NAME          qwen35-9b-ags-credentials
  EXP_TAG / LOG_DIR        default fresh tag …_hybrid_ltpa (not old empty-run dir)
  LOAD_PATH / SAVE_PATH    checkpoint source and destination (default LOG_DIR/slime_save)
  RUN_ROOT                 rollout dump root (default LOG_DIR)
  LOAD_CKPT_STEP           optional exact checkpoint iteration to load
  NUM_ROLLOUT              default 88 (one pass @ RBS=16; raise if RBS is smaller)
  STEP_GRPO_HYBRID_K       8
  STEP_GRPO_FILTER         1
  STEP_GRPO_STAGE1_LOSS_WEIGHT  1.0 (set 0 for Stage-2-only loss)
  STEP_GRPO_BRANCH_LOSS_WEIGHT  1.0 (lambda)
  STEP_GRPO_STAGE2_LOSS_SCOPE   full_continuation|first_turn
  SLIME_CC_AGENT_CONCURRENCY global agent gate; <=0 disables it (default 64)
  SLIME_CC_EVAL_CONCURRENCY  eval gate; <=0 disables it (default 0 for Hybrid training)
  ALLOW_RESUME             1 to allow submitting into a LOG_DIR that already has ckpts
  WANDB_KEY                optional if ~/.config/jiaxicao/wandb_api_key exists
  WANDB_KEY_FILE           override path to key file (default above)
  WANDB_PROJECT            coding-rl
  WANDB_TEAM               models-tencent7723 (entity)
  WANDB_GROUP              experiment group (default: EXP_TAG)
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
K8S_NODE_GROUP="${K8S_NODE_GROUP:-shennong-5}"
JOB_NAME="${JOB_NAME:-jiaxicao-hybrid-1node-debug}"
WORKLOAD_LABEL="${WORKLOAD_LABEL:-${JOB_NAME}}"
IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/youtu-agent:slime-nightly-dev-20260530a-efa-swe-mooncake}"
SLIME_DIR="${SLIME_DIR:-/mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl}"
EVAL_DATA="${EVAL_DATA:-/mnt/sn-007/youtu-agent/yuleiqin/SWE_code/DataEng/RL_DATA/data_valid/swe_agent_ags_swebench_verified/test.parquet}"
EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_1node_hybrid_ltpa}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
LOAD_PATH="${LOAD_PATH:-${LOG_DIR}/slime_save}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"
LOAD_CKPT_STEP="${LOAD_CKPT_STEP:-}"
RESUME_DEBUG_ROLLOUT_DATA="${RESUME_DEBUG_ROLLOUT_DATA:-1}"
PHASE="${PHASE:-train}"
AGS_SECRET_NAME="${AGS_SECRET_NAME:-qwen35-9b-ags-credentials}"
SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-ltpatoxb}"
STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
# Submit-batch size (waves of StartSandbox), not concurrent-run cap.
STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-${STEP_GRPO_BRANCH_CONCURRENCY:-64}}"
STEP_GRPO_BRANCH_CONCURRENCY="${STEP_GRPO_BRANCH_SUBMIT_BATCH}"
STEP_GRPO_PPL_CLIP="${STEP_GRPO_PPL_CLIP:-20}"
STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"
STEP_GRPO_STAGE1_LOSS_WEIGHT="${STEP_GRPO_STAGE1_LOSS_WEIGHT:-1.0}"
STEP_GRPO_BRANCH_LOSS_WEIGHT="${STEP_GRPO_BRANCH_LOSS_WEIGHT:-1.0}"
STEP_GRPO_STAGE2_LOSS_SCOPE="${STEP_GRPO_STAGE2_LOSS_SCOPE:-full_continuation}"
case "${STEP_GRPO_STAGE2_LOSS_SCOPE}" in
  full_continuation|first_turn) ;;
  *)
    echo "ERROR: STEP_GRPO_STAGE2_LOSS_SCOPE must be full_continuation or first_turn" >&2
    exit 2
    ;;
esac
STEP_GRPO_BUNDLE_DIR="${STEP_GRPO_BUNDLE_DIR:-${LOG_DIR}/step_reconstruct_bundles}"
NUM_ROLLOUT="${NUM_ROLLOUT:-88}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
# Must match slime: GBS == RBS * n_samples_per_prompt (hybrid outer n_samples=1).
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-2700}"
SLIME_CC_AGENT_CONCURRENCY="${SLIME_CC_AGENT_CONCURRENCY:-64}"
SLIME_CC_EVAL_CONCURRENCY="${SLIME_CC_EVAL_CONCURRENCY:-0}"
SLIME_CC_TOOL_LOOP_PENALTY="${SLIME_CC_TOOL_LOOP_PENALTY:-0}"
SLIME_CC_TIMEOUT_OUTCOME_REWARD="${SLIME_CC_TIMEOUT_OUTCOME_REWARD:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
WANDB_FORK_FROM="${WANDB_FORK_FROM:-}"
WANDB_RESUME_FROM="${WANDB_RESUME_FROM:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-auto}"
_load_wandb_key
WANDB_KEY="${WANDB_KEY:-}"
# Prefer hybrid EXP_TAG; do not inherit stale GRPO / old-hybrid group from the shell.
if [[ -z "${WANDB_GROUP:-}" \
   || "${WANDB_GROUP}" == "qwen35_9b_cc_ags_1node_grpo_debug" \
   || "${WANDB_GROUP}" == "qwen35_9b_cc_ags_1node_hybrid" ]]; then
  WANDB_GROUP="${EXP_TAG}"
fi

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete pytorchjob "${JOB_NAME}" --ignore-not-found
  echo "Deleted pytorchjob/${JOB_NAME} in ${K8S_NAMESPACE}"
  exit 0
fi

# Always prefer this job's Ingress ADDRESS so a stale SLIME_ADAPTER_PUBLIC_URL
# from another launcher (e.g. GRPO ALB) cannot silently misroute AGS traffic.
ADDR="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${INGRESS_NAME}" \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
if [[ -n "${ADDR}" ]]; then
  SLIME_ADAPTER_PUBLIC_URL="http://${ADDR}"
  echo "Using Ingress ${INGRESS_NAME} → ${SLIME_ADAPTER_PUBLIC_URL}"
elif [[ -z "${SLIME_ADAPTER_PUBLIC_URL:-}" || "${SLIME_ADAPTER_PUBLIC_URL}" == *REPLACE_WITH* ]]; then
  echo "ERROR: set SLIME_ADAPTER_PUBLIC_URL or apply hybrid_adapter_alb first:" >&2
  echo "  bash examples/claudecode_ags/launch/hybrid_adapter_alb/submit_alb.sh" >&2
  exit 3
fi
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"

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

if ! kubectl -n "${K8S_NAMESPACE}" get ingress "${INGRESS_NAME}" >/dev/null 2>&1; then
  echo "WARNING: adapter Ingress ${INGRESS_NAME} not found; apply hybrid_adapter_alb first." >&2
fi

# Refuse accidental resume into the old empty-run checkpoint dir (or any dir with ckpts).
if [[ -z "${WANDB_KEY}" ]]; then
  echo "ERROR: WANDB_KEY missing. Set WANDB_KEY or put the key in ${WANDB_KEY_FILE:-$HOME/.config/jiaxicao/wandb_api_key}" >&2
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
if [[ "${LOG_DIR}" == */qwen35_9b_cc_ags_1node_hybrid && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "ERROR: refusing default old hybrid LOG_DIR (empty-run history): ${LOG_DIR}" >&2
  echo "       Default is now …/qwen35_9b_cc_ags_1node_hybrid_ltpa, or set ALLOW_RESUME=1." >&2
  exit 4
fi

export JOB_NAME WORKLOAD_LABEL K8S_NAMESPACE K8S_NODE_GROUP IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export PROMPT_DATA EVAL_DATA EXP_TAG LOG_DIR LOAD_PATH SAVE_PATH RUN_ROOT LOAD_CKPT_STEP
export RESUME_DEBUG_ROLLOUT_DATA PHASE AGS_SECRET_NAME SLIME_ADAPTER_PUBLIC_URL
export SLIME_AGENT_AGS_TOOL_ID
export WANDB_KEY WANDB_PROJECT WANDB_GROUP WANDB_TEAM WANDB_FORK_FROM WANDB_RESUME_FROM WANDB_RUN_ID WANDB_RESUME
export STEP_GRPO_HYBRID_K STEP_GRPO_BRANCH_SUBMIT_BATCH STEP_GRPO_BRANCH_CONCURRENCY STEP_GRPO_PPL_CLIP STEP_GRPO_FILTER STEP_GRPO_STAGE1_LOSS_WEIGHT STEP_GRPO_BRANCH_LOSS_WEIGHT STEP_GRPO_STAGE2_LOSS_SCOPE
export STEP_GRPO_BUNDLE_DIR NUM_ROLLOUT SAVE_INTERVAL GLOBAL_BATCH_SIZE ROLLOUT_BATCH_SIZE
export SLIME_CC_TIME_BUDGET_SEC SLIME_CC_AGENT_CONCURRENCY SLIME_CC_EVAL_CONCURRENCY
export SLIME_CC_TOOL_LOOP_PENALTY SLIME_CC_TIMEOUT_OUTCOME_REWARD

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${WORKLOAD_LABEL} ${K8S_NAMESPACE} ${K8S_NODE_GROUP} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${PROMPT_DATA} ${EVAL_DATA} ${EXP_TAG} ${LOG_DIR} ${LOAD_PATH} ${SAVE_PATH} ${RUN_ROOT} ${LOAD_CKPT_STEP} ${RESUME_DEBUG_ROLLOUT_DATA} ${PHASE} ${AGS_SECRET_NAME} ${SLIME_ADAPTER_PUBLIC_URL} ${SLIME_AGENT_AGS_TOOL_ID} ${WANDB_KEY} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM} ${WANDB_FORK_FROM} ${WANDB_RESUME_FROM} ${WANDB_RUN_ID} ${WANDB_RESUME} ${STEP_GRPO_HYBRID_K} ${STEP_GRPO_BRANCH_SUBMIT_BATCH} ${STEP_GRPO_BRANCH_CONCURRENCY} ${STEP_GRPO_PPL_CLIP} ${STEP_GRPO_FILTER} ${STEP_GRPO_STAGE1_LOSS_WEIGHT} ${STEP_GRPO_BRANCH_LOSS_WEIGHT} ${STEP_GRPO_STAGE2_LOSS_SCOPE} ${STEP_GRPO_BUNDLE_DIR} ${NUM_ROLLOUT} ${SAVE_INTERVAL} ${GLOBAL_BATCH_SIZE} ${ROLLOUT_BATCH_SIZE} ${SLIME_CC_TIME_BUDGET_SEC} ${SLIME_CC_AGENT_CONCURRENCY} ${SLIME_CC_EVAL_CONCURRENCY} ${SLIME_CC_TOOL_LOOP_PENALTY} ${SLIME_CC_TIMEOUT_OUTCOME_REWARD}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} in ${K8S_NAMESPACE}"
echo "    node group: ${K8S_NODE_GROUP}"
echo "    image: ${IMAGE_URI}"
echo "    SLIME_DIR=${SLIME_DIR}"
echo "    SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "    SLIME_AGENT_AGS_TOOL_ID=${SLIME_AGENT_AGS_TOOL_ID}"
echo "    PHASE=${PHASE} LOG_DIR=${LOG_DIR}"
echo "    LOAD_PATH=${LOAD_PATH} LOAD_CKPT_STEP=${LOAD_CKPT_STEP:-latest}"
echo "    SAVE_PATH=${SAVE_PATH} RUN_ROOT=${RUN_ROOT}"
echo "    PROMPT_DATA=${PROMPT_DATA}"
echo "    NUM_ROLLOUT=${NUM_ROLLOUT} SAVE_INTERVAL=${SAVE_INTERVAL} RBS=${ROLLOUT_BATCH_SIZE} GBS=${GLOBAL_BATCH_SIZE}"
echo "    hybrid K=${STEP_GRPO_HYBRID_K} filter=${STEP_GRPO_FILTER} stage1_weight=${STEP_GRPO_STAGE1_LOSS_WEIGHT} lambda=${STEP_GRPO_BRANCH_LOSS_WEIGHT} stage2_loss_scope=${STEP_GRPO_STAGE2_LOSS_SCOPE} submit_batch=${STEP_GRPO_BRANCH_SUBMIT_BATCH}"
echo "    agent_budget=${SLIME_CC_TIME_BUDGET_SEC}s agent_concurrency=${SLIME_CC_AGENT_CONCURRENCY} eval_concurrency=${SLIME_CC_EVAL_CONCURRENCY}"
echo "    tool_loop_penalty=${SLIME_CC_TOOL_LOOP_PENALTY} timeout_outcome_reward=${SLIME_CC_TIMEOUT_OUTCOME_REWARD}"
echo "    AGS timeout=75m"
echo "    WANDB project=${WANDB_PROJECT} team=${WANDB_TEAM} group=${WANDB_GROUP} key=***"
echo "    WANDB fork_from=${WANDB_FORK_FROM:-none} resume_from=${WANDB_RESUME_FROM:-none} run_id=${WANDB_RUN_ID:-new}"

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
echo "  kubectl -n ${K8S_NAMESPACE} logs -f job/${JOB_NAME}-master-0 2>/dev/null || \\"
echo "    kubectl -n ${K8S_NAMESPACE} logs -f -l training.kubeflow.org/job-name=${JOB_NAME},training.kubeflow.org/replica-type=master"
echo "ALB /health (after adapter up):"
echo "  curl -fsS ${SLIME_ADAPTER_PUBLIC_URL}/health"
