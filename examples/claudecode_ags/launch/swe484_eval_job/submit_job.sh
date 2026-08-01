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
  JOB_NAME / WORKLOAD / LOG_DIR / LOAD_PATH / LOAD_CKPT_STEP / SAVE_PATH
  SLIME_ADAPTER_PUBLIC_URL
  K8S_NAMESPACE  sn5-system-intern
  K8S_NODE_GROUP shennong-5 (set shennong-5-dev for the dev pool)
  SLIME_AGENT_SFT_LOG_DIR / SLIME_CC_EVAL_ARTIFACT_DIR
  SLIME_SWEBENCH_VERSION  exact official harness version (default 4.1.0)
  SWE_EVAL_EXPECTED_TASKS expected dataset rows (default 484)
  SLIME_CC_TIME_BUDGET_SEC agent runtime (default 1800 = 30 min)
  SLIME_CC_INITIAL_INPUT_MODE / SLIME_CC_AGENT_PROMPT / SLIME_CC_EXTRA_ARGS_JSON
  K8S_PRIORITY_CLASS / K8S_SCHEDULING_GATE optional scheduling controls
  K8S_TARGET_NODE optional exact node pin (via node affinity metadata.name)
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
K8S_NODE_GROUP="${K8S_NODE_GROUP:-shennong-5}"
K8S_PRIORITY_CLASS="${K8S_PRIORITY_CLASS:-}"
K8S_SCHEDULING_GATE="${K8S_SCHEDULING_GATE:-}"
K8S_TARGET_NODE="${K8S_TARGET_NODE:-}"
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
LOAD_CKPT_STEP="${LOAD_CKPT_STEP:-}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
SAVE_PATH="${SAVE_PATH:-${LOG_DIR}/slime_save}"
SLIME_AGENT_SFT_LOG_DIR="${SLIME_AGENT_SFT_LOG_DIR:-${LOG_DIR}/model_turns}"
SLIME_CC_EVAL_ARTIFACT_DIR="${SLIME_CC_EVAL_ARTIFACT_DIR:-${LOG_DIR}/eval_artifacts}"
SLIME_SWEBENCH_VERSION="${SLIME_SWEBENCH_VERSION:-4.1.0}"
SWE_EVAL_EXPECTED_TASKS="${SWE_EVAL_EXPECTED_TASKS:-484}"

# Keep the Base comparison's agent-visible context aligned with the 2026-07-22
# reference run. These are eval-only defaults; training launchers are unchanged.
SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-1800}"
SLIME_CC_EVAL_TIMEOUT_SEC="${SLIME_CC_EVAL_TIMEOUT_SEC:-600}"
SLIME_CC_EVAL_GUARD_SEC="${SLIME_CC_EVAL_GUARD_SEC:-$((SLIME_CC_EVAL_TIMEOUT_SEC + 180))}"
SLIME_CC_EVAL_INFRA_RETRIES="${SLIME_CC_EVAL_INFRA_RETRIES:-1}"
SLIME_CC_GENERATE_GUARD_SEC="${SLIME_CC_GENERATE_GUARD_SEC:-2700}"
SLIME_CC_AGENT_CONCURRENCY="${SLIME_CC_AGENT_CONCURRENCY:-64}"
SLIME_CC_EVAL_CONCURRENCY="${SLIME_CC_EVAL_CONCURRENCY:-64}"
SLIME_CC_INITIAL_INPUT_MODE="${SLIME_CC_INITIAL_INPUT_MODE:-positional}"
SLIME_CC_AGENT_PROMPT="${SLIME_CC_AGENT_PROMPT:-Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. Edit source files only (do NOT touch tests). After editing, run the relevant tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do NOT commit. When finished, print a one-line summary and exit.}"
if [[ -z "${SLIME_CC_EXTRA_ARGS_JSON:-}" ]]; then
  SLIME_CC_EXTRA_ARGS_JSON='["--settings","{\"permissions\":{\"defaultMode\":\"bypassPermissions\"},\"autoCompactEnabled\":true}","--disable-slash-commands","--agents","{\"investigator\":{\"description\":\"Searches the repo for relevant files before any edit\",\"prompt\":\"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.\",\"tools\":[\"Grep\",\"Read\",\"Glob\"]}}","--disallowedTools","WebFetch","WebSearch"]'
fi
CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-101000}"
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE:-95}"
CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"
CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING="${CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING:-1}"
CLAUDE_CODE_SKIP_PROMPT_HISTORY="${CLAUDE_CODE_SKIP_PROMPT_HISTORY:-1}"
CLAUDE_CODE_DISABLE_TERMINAL_TITLE="${CLAUDE_CODE_DISABLE_TERMINAL_TITLE:-1}"
SLIME_AGENT_AGS_TIMEOUT="${SLIME_AGENT_AGS_TIMEOUT:-45m}"
SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC="${SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC:-2700}"
SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC="${SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC:-600}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-${MAX_CONTEXT_LEN}}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-131072}"

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
if [[ -n "${LOAD_CKPT_STEP}" ]]; then
  if [[ ! "${LOAD_CKPT_STEP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LOAD_CKPT_STEP must be a non-negative integer, got: ${LOAD_CKPT_STEP}" >&2
    exit 2
  fi
  ITER_DIR="${LOAD_PATH}/iter_$(printf '%07d' "${LOAD_CKPT_STEP}")"
  if [[ ! -d "${ITER_DIR}" ]]; then
    echo "ERROR: checkpoint iteration not found: ${ITER_DIR}" >&2
    exit 1
  fi
fi

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

mkdir -p \
  "${LOG_DIR}/slime_save" \
  "${LOG_DIR}/wandb" \
  "${LOG_DIR}/rollout_dumps" \
  "${LOG_DIR}/launcher_logs" \
  "${SLIME_AGENT_SFT_LOG_DIR}" \
  "${SLIME_CC_EVAL_ARTIFACT_DIR}"

export JOB_NAME K8S_NAMESPACE K8S_NODE_GROUP IMAGE_URI SLIME_DIR HF_CHECKPOINT REF_MODEL_PATH
export LOAD_PATH LOAD_CKPT_STEP SAVE_PATH PROMPT_DATA EVAL_DATA EXP_TAG LOG_DIR
export AGS_SECRET_NAME SLIME_ADAPTER_PUBLIC_URL SLIME_AGENT_AGS_TOOL_ID
export SLIME_AGENT_SFT_LOG_DIR SLIME_CC_EVAL_ARTIFACT_DIR
export SLIME_SWEBENCH_VERSION SWE_EVAL_EXPECTED_TASKS
export N_SAMPLES_PER_EVAL_PROMPT
export MAX_CONTEXT_LEN MAX_GEN_LEN EVAL_MAX_PROMPT_LEN ROLLOUT_MAX_CONTEXT_LEN
export SLIME_CC_TIME_BUDGET_SEC SLIME_CC_EVAL_TIMEOUT_SEC SLIME_CC_GENERATE_GUARD_SEC
export SLIME_CC_EVAL_GUARD_SEC SLIME_CC_EVAL_INFRA_RETRIES
export SLIME_CC_AGENT_CONCURRENCY SLIME_CC_EVAL_CONCURRENCY
export SLIME_CC_AGENT_PROMPT SLIME_CC_INITIAL_INPUT_MODE
export SLIME_CC_EXTRA_ARGS_JSON
export CLAUDE_CODE_AUTO_COMPACT_WINDOW CLAUDE_AUTOCOMPACT_PCT_OVERRIDE
export CLAUDE_CODE_MAX_OUTPUT_TOKENS CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING
export CLAUDE_CODE_SKIP_PROMPT_HISTORY CLAUDE_CODE_DISABLE_TERMINAL_TITLE
export SLIME_AGENT_AGS_TIMEOUT SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC
export SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC
export WANDB_PROJECT WANDB_GROUP WANDB_TEAM
export WORKLOAD EVAL_ROLE

K8S_SCHEDULING_SPEC=""
K8S_TARGET_NODE_AFFINITY=""
if [[ -n "${K8S_PRIORITY_CLASS}" ]]; then
  if [[ ! "${K8S_PRIORITY_CLASS}" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]]; then
    echo "ERROR: invalid K8S_PRIORITY_CLASS=${K8S_PRIORITY_CLASS}" >&2
    exit 2
  fi
  printf -v K8S_SCHEDULING_SPEC '          priorityClassName: %s\n' "${K8S_PRIORITY_CLASS}"
fi
if [[ -n "${K8S_SCHEDULING_GATE}" ]]; then
  if [[ ! "${K8S_SCHEDULING_GATE}" =~ ^[a-z0-9]([-a-z0-9./]*[a-z0-9])?$ ]]; then
    echo "ERROR: invalid K8S_SCHEDULING_GATE=${K8S_SCHEDULING_GATE}" >&2
    exit 2
  fi
  printf -v K8S_SCHEDULING_SPEC '%s          schedulingGates:\n            - name: %s\n' \
    "${K8S_SCHEDULING_SPEC}" "${K8S_SCHEDULING_GATE}"
fi
if [[ -n "${K8S_TARGET_NODE}" ]]; then
  if [[ ! "${K8S_TARGET_NODE}" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]]; then
    echo "ERROR: invalid K8S_TARGET_NODE=${K8S_TARGET_NODE}" >&2
    exit 2
  fi
  printf -v K8S_TARGET_NODE_AFFINITY \
    '          affinity:\n            nodeAffinity:\n              requiredDuringSchedulingIgnoredDuringExecution:\n                nodeSelectorTerms:\n                  - matchFields:\n                      - key: metadata.name\n                        operator: In\n                        values:\n                          - %s\n' \
    "${K8S_TARGET_NODE}"
fi
export K8S_SCHEDULING_SPEC K8S_TARGET_NODE_AFFINITY

RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${K8S_NAMESPACE} ${K8S_NODE_GROUP} ${K8S_SCHEDULING_SPEC} ${K8S_TARGET_NODE_AFFINITY} ${IMAGE_URI} ${SLIME_DIR} ${HF_CHECKPOINT} ${REF_MODEL_PATH} ${LOAD_PATH} ${LOAD_CKPT_STEP} ${SAVE_PATH} ${PROMPT_DATA} ${EVAL_DATA} ${EXP_TAG} ${LOG_DIR} ${AGS_SECRET_NAME} ${SLIME_ADAPTER_PUBLIC_URL} ${SLIME_AGENT_AGS_TOOL_ID} ${SLIME_AGENT_SFT_LOG_DIR} ${SLIME_CC_EVAL_ARTIFACT_DIR} ${SLIME_SWEBENCH_VERSION} ${SWE_EVAL_EXPECTED_TASKS} ${N_SAMPLES_PER_EVAL_PROMPT} ${MAX_CONTEXT_LEN} ${MAX_GEN_LEN} ${EVAL_MAX_PROMPT_LEN} ${ROLLOUT_MAX_CONTEXT_LEN} ${SLIME_CC_TIME_BUDGET_SEC} ${SLIME_CC_EVAL_TIMEOUT_SEC} ${SLIME_CC_GENERATE_GUARD_SEC} ${SLIME_CC_EVAL_GUARD_SEC} ${SLIME_CC_EVAL_INFRA_RETRIES} ${SLIME_CC_AGENT_CONCURRENCY} ${SLIME_CC_EVAL_CONCURRENCY} ${SLIME_CC_AGENT_PROMPT} ${SLIME_CC_INITIAL_INPUT_MODE} ${SLIME_CC_EXTRA_ARGS_JSON} ${CLAUDE_CODE_AUTO_COMPACT_WINDOW} ${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE} ${CLAUDE_CODE_MAX_OUTPUT_TOKENS} ${CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING} ${CLAUDE_CODE_SKIP_PROMPT_HISTORY} ${CLAUDE_CODE_DISABLE_TERMINAL_TITLE} ${SLIME_AGENT_AGS_TIMEOUT} ${SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC} ${SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC} ${WANDB_PROJECT} ${WANDB_GROUP} ${WANDB_TEAM} ${WORKLOAD} ${EVAL_ROLE}' \
  < "${TEMPLATE}" > "${RENDERED}"

echo "==> job ${JOB_NAME} (role=${EVAL_ROLE}) in ${K8S_NAMESPACE}"
echo "    image: ${IMAGE_URI}"
echo "    LOAD_PATH=${LOAD_PATH} LOAD_CKPT_STEP=${LOAD_CKPT_STEP:-latest}"
echo "    SAVE_PATH=${SAVE_PATH}"
echo "    LOG_DIR=${LOG_DIR}"
echo "    K8S_NODE_GROUP=${K8S_NODE_GROUP}"
echo "    K8S_TARGET_NODE=${K8S_TARGET_NODE:-<none>}"
echo "    K8S_PRIORITY_CLASS=${K8S_PRIORITY_CLASS:-<none>}"
echo "    K8S_SCHEDULING_GATE=${K8S_SCHEDULING_GATE:-<none>}"
echo "    SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "    EVAL_DATA=${EVAL_DATA}"
echo "    SLIME_SWEBENCH_VERSION=${SLIME_SWEBENCH_VERSION}"
echo "    SWE_EVAL_EXPECTED_TASKS=${SWE_EVAL_EXPECTED_TASKS}"
echo "    N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT} (pass@k)"
echo "    agent runtime=${SLIME_CC_TIME_BUDGET_SEC}s, eval timeout=${SLIME_CC_EVAL_TIMEOUT_SEC}s"
echo "    eval guard=${SLIME_CC_EVAL_GUARD_SEC}s, fresh-sandbox retries=${SLIME_CC_EVAL_INFRA_RETRIES}"
echo "    pipeline guard=${SLIME_CC_GENERATE_GUARD_SEC}s, agent/eval concurrency=${SLIME_CC_AGENT_CONCURRENCY}/${SLIME_CC_EVAL_CONCURRENCY}"
echo "    agent input=${SLIME_CC_INITIAL_INPUT_MODE}, max output=${CLAUDE_CODE_MAX_OUTPUT_TOKENS}"
echo "    eval prompt/context=${EVAL_MAX_PROMPT_LEN}/${ROLLOUT_MAX_CONTEXT_LEN}"
echo "    model turns: ${SLIME_AGENT_SFT_LOG_DIR}"
echo "    eval artifacts: ${SLIME_CC_EVAL_ARTIFACT_DIR}"

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
