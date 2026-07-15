#!/usr/bin/env bash
# Path A: 1-node / 8-GPU colocate GRPO debug launcher (train + post Verified).
#
# Defaults match docs/superpowers/specs/2026-07-12-grpo-1node-debug-design.md.
# Mid-train eval is skipped; with PHASE=all, SWE-bench Verified runs once after
# NUM_ROLLOUT steps (--eval-interval == NUM_ROLLOUT + --skip-eval-before-train).
#
# ---------------------------------------------------------------------------
# SLIME_ADAPTER_PUBLIC_URL (required)
# ---------------------------------------------------------------------------
# AGS sandboxes cannot reach 127.0.0.1 / localhost on the train node.
# Export the internet-facing ALB from launch/grpo_adapter_alb/submit_alb.sh:
#   export SLIME_ADAPTER_PUBLIC_URL=http://<ADDRESS>   # ALB :80, no :9002
# The Master pod must expose adapter on SLIME_ADAPTER_PORT=9002 (also set
# SHIM_PORT=9002 for parity with existing HyperPod GRPO jobs) and carry labels
#   app=cc-ags-recorder,workload=jiaxicao-grpo-1node-debug
# Do NOT reuse the deleted L2 Ingress / jiaxicao-cc-ags-adapter Deployment URL.
#
# ---------------------------------------------------------------------------
# PHASE
# ---------------------------------------------------------------------------
#   train — GRPO only (no eval args); NUM_ROLLOUT=21
#   all   — train then one Verified eval (default)
#   eval  — Verified only via slime's eval-only path:
#           --num-rollout 0 + --eval-interval set (see train.py special case).
#           Prefer this over a 1-row dummy prompt; 0 is explicitly supported.
#
# Usage:
#   export SLIME_ADAPTER_PUBLIC_URL=http://<alb-hostname>
#   bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh          # dry-run
#   RUN=1 bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh    # execute
#   PHASE=eval RUN=1 bash examples/claudecode_ags/launch/run_grpo_1node_debug.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${EXAMPLE_DIR}/../.." && pwd)}"
export SLIME_DIR

# shellcheck disable=SC1091
source "${EXAMPLE_DIR}/env/load_env.sh"

RUN="${RUN:-0}"
PHASE="${PHASE:-all}"
case "${PHASE}" in
  train|eval|all) ;;
  *)
    echo "ERROR: PHASE must be train|eval|all, got: ${PHASE}" >&2
    exit 2
    ;;
esac

# ============ topology (colocate; override ACTOR_NUM_NODES for multi-node) ============
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-1}"
export ETP_SIZE="${ETP_SIZE:-1}"

# Default rollout GPUs = all actor GPUs (colocate).
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
SLIME_PREFILL_NUM_SERVERS="${SLIME_PREFILL_NUM_SERVERS:-0}"
SLIME_ROUTER_PORT="${SLIME_ROUTER_PORT:-18000}"
SLIME_SGLANG_ROUTER_POLICY="${SLIME_SGLANG_ROUTER_POLICY:-consistent_hashing}"
SLIME_SGLANG_ROUTER_PREFILL_POLICY="${SLIME_SGLANG_ROUTER_PREFILL_POLICY:-consistent_hashing}"
SLIME_SGLANG_ROUTER_DECODE_POLICY="${SLIME_SGLANG_ROUTER_DECODE_POLICY:-consistent_hashing}"
SGLANG_MAMBA_SCHEDULER_STRATEGY="${SGLANG_MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
export SGLANG_ENABLE_HICACHE="${SGLANG_ENABLE_HICACHE:-1}"
export SGLANG_HICACHE_RATIO="${SGLANG_HICACHE_RATIO:-1.0}"
export SGLANG_HICACHE_WRITE_POLICY="${SGLANG_HICACHE_WRITE_POLICY:-write_through}"
export SGLANG_HICACHE_IO_BACKEND="${SGLANG_HICACHE_IO_BACKEND:-kernel}"
export SGLANG_HICACHE_MEM_LAYOUT="${SGLANG_HICACHE_MEM_LAYOUT:-layer_first}"

# ============ context length ============
SLIME_SGLANG_CONTEXT_LENGTH="${SLIME_SGLANG_CONTEXT_LENGTH:-131072}"
export SLIME_SGLANG_CONTEXT_LENGTH
SLIME_SGLANG_CONTEXT_SAFETY_MARGIN="${SLIME_SGLANG_CONTEXT_SAFETY_MARGIN:-256}"
export SLIME_SGLANG_CONTEXT_SAFETY_MARGIN
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-4160}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_GEN_LEN="${MAX_GEN_LEN:-16384}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-128000}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-${MAX_GEN_LEN}}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"

# ============ paths / batch ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/sn-007/jiaxicao/huggingface/models/Qwen/Qwen3.5-9B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/sn-007/jiaxicao/code/slime/eval_runs/swegym_passk_20260711_090409/train_grpo_resolved_1_7.slime.jsonl}"
EVAL_DATA="${EVAL_DATA:-/mnt/sn-007/youtu-agent/yuleiqin/SWE_code/DataEng/RL_DATA/data_valid/swe_agent_ags_swebench_verified/test.parquet}"

# Debug-friendly batch defaults (override: ROLLOUT_BATCH_SIZE=16 for formal).
# slime requires: GBS == RBS * n_samples_per_prompt // num_steps_per_rollout
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
# Outer fan-out is 1; Stage-1 group size is STEP_GRPO_HYBRID_K (internal).
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
# Formal 1-node: save every 5 steps (override SAVE_INTERVAL=1 for crash-debug).
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

# ============ hybrid step-GRPO knobs (exported into Ray runtime) ============
export STEP_GRPO_HYBRID_K="${STEP_GRPO_HYBRID_K:-8}"
# Stage-2: submit sandboxes in waves of 64 (not an inflight-run cap).
# Prefer STEP_GRPO_BRANCH_SUBMIT_BATCH; BRANCH_CONCURRENCY is a legacy alias.
export STEP_GRPO_BRANCH_SUBMIT_BATCH="${STEP_GRPO_BRANCH_SUBMIT_BATCH:-${STEP_GRPO_BRANCH_CONCURRENCY:-64}}"
export STEP_GRPO_BRANCH_CONCURRENCY="${STEP_GRPO_BRANCH_SUBMIT_BATCH}"
export STEP_GRPO_PPL_CLIP="${STEP_GRPO_PPL_CLIP:-20}"
export STEP_GRPO_FILTER="${STEP_GRPO_FILTER:-1}"

if [[ "${PHASE}" == "eval" ]]; then
  # train.py: if num_rollout == 0 and eval_interval is set → eval-only.
  NUM_ROLLOUT="${NUM_ROLLOUT:-0}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
else
  # One pass over 1394 prompts @ rollout_batch=16 → ceil(1394/16)=88.
  # With RBS=8 default: ceil(1394/8)=175; override NUM_ROLLOUT if you change RBS.
  NUM_ROLLOUT="${NUM_ROLLOUT:-88}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-${NUM_ROLLOUT}}"
fi

# Fail fast if batch knobs violate slime_validate_args (GBS == RBS * n_samples // steps).
_expected_gbs=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / 1 ))
if [[ "${GLOBAL_BATCH_SIZE}" -ne "${_expected_gbs}" ]]; then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must equal ROLLOUT_BATCH_SIZE*N_SAMPLES=${_expected_gbs}" >&2
  exit 2
fi

# Fresh default tag — do NOT reuse qwen35_9b_cc_ags_1node_hybrid (empty-run ckpts).
EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_1node_hybrid_ltpa}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"
export STEP_GRPO_BUNDLE_DIR="${STEP_GRPO_BUNDLE_DIR:-${LOG_DIR}/step_reconstruct_bundles}"
# Avoid inheriting a stale vanilla-GRPO WANDB_GROUP from slime_ags.env.
if [[ -z "${WANDB_GROUP:-}" || "${WANDB_GROUP}" == "qwen35_9b_cc_ags_1node_grpo_debug" || "${WANDB_GROUP}" == "qwen35_9b_cc_ags_1node_hybrid" ]]; then
  WANDB_GROUP="${EXP_TAG}"
fi
export WANDB_GROUP
WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
export WANDB_PROJECT WANDB_TEAM

# ============ AGS + toolchain (cluster defaults; Job env / slime_ags.env win) ============
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-ags}"
# Separate AGS tool from vanilla GRPO (sdt-exb9o2gb).
export SLIME_AGENT_AGS_TOOL_ID="${SLIME_AGENT_AGS_TOOL_ID:-sdt-ltpatoxb}"
# swerex-runtime mount — avoids ContainerStart / port binding failed (see TROUBLESHOOTING.md)
export SLIME_AGENT_AGS_MOUNT_NAME="${SLIME_AGENT_AGS_MOUNT_NAME:-rex}"
export SLIME_AGENT_AGS_MOUNT_IMAGE="${SLIME_AGENT_AGS_MOUNT_IMAGE:-swebenchdocker.tencentcloudcr.com/swebench/swehub:swerex-runtime}"
export SLIME_AGENT_AGS_MOUNT_IMAGE_REGISTRY_TYPE="${SLIME_AGENT_AGS_MOUNT_IMAGE_REGISTRY_TYPE:-enterprise}"
export SLIME_AGENT_AGS_MOUNT_PATH="${SLIME_AGENT_AGS_MOUNT_PATH:-/nix}"
export SLIME_AGENT_AGS_IMAGE_SUBPATH="${SLIME_AGENT_AGS_IMAGE_SUBPATH:-/nix}"
export SLIME_AGENT_AGS_REGION="${SLIME_AGENT_AGS_REGION:-ap-guangzhou}"
export SLIME_AGENT_AGS_DOMAIN="${SLIME_AGENT_AGS_DOMAIN:-ap-guangzhou.tencentags.com}"
export SLIME_AGENT_AGS_ROLE_ARN="${SLIME_AGENT_AGS_ROLE_ARN:-qcs::cam::uin/100034032793:roleName/AGS_cam}"
export SLIME_AGENT_AGS_HTTP_ENDPOINT="${SLIME_AGENT_AGS_HTTP_ENDPOINT:-ags.tencentcloudapi.com}"
export SLIME_AGENT_TOOLCHAIN_MODE="${SLIME_AGENT_TOOLCHAIN_MODE:-cos}"
export SLIME_AGENT_COS_MOUNT="${SLIME_AGENT_COS_MOUNT:-/mnt/code_agent}"
export SLIME_AGENT_COS_NODE_PACKAGE="${SLIME_AGENT_COS_NODE_PACKAGE:-node-v20.18.1-linux-x64.tar.xz}"
export SLIME_AGENT_COS_CC_PACKAGE="${SLIME_AGENT_COS_CC_PACKAGE:-cc-prefix-2.1.104-linux-x64.tar.gz}"

# ============ timeouts / concurrency (export if unset) ============
# Agent budget 45m; AGS lifetime above that so sandbox outlives CC.
export SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-2700}"
export SLIME_CC_EVAL_TIMEOUT_SEC="${SLIME_CC_EVAL_TIMEOUT_SEC:-600}"
export SLIME_AGENT_AGS_TIMEOUT="${SLIME_AGENT_AGS_TIMEOUT:-75m}"
export SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC="${SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC:-4500}"
export SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC="${SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC:-600}"
export STEP_GRPO_BRANCH_BUDGET_SEC="${STEP_GRPO_BRANCH_BUDGET_SEC:-${SLIME_CC_TIME_BUDGET_SEC}}"
# True in-flight agent cap (vanilla + branch); <=0 disables.
export SLIME_CC_AGENT_CONCURRENCY="${SLIME_CC_AGENT_CONCURRENCY:-64}"

# ============ asserts ============
if [[ -z "${SLIME_ADAPTER_PUBLIC_URL:-}" || "${SLIME_ADAPTER_PUBLIC_URL}" == *REPLACE_WITH* ]]; then
  echo "ERROR: set SLIME_ADAPTER_PUBLIC_URL to a public/ALB URL AGS can reach for THIS job's adapter." >&2
  echo "       Do not use 127.0.0.1 or the deleted L2 Ingress URL." >&2
  exit 3
fi
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"
# Reject loopback / bind-any hosts — AGS sandboxes cannot reach them.
_adapter_host="${SLIME_ADAPTER_PUBLIC_URL#*://}"
_adapter_host="${_adapter_host%%/*}"
_adapter_host="${_adapter_host%%:*}"
_adapter_host="$(printf '%s' "${_adapter_host}" | tr '[:upper:]' '[:lower:]')"
case "${_adapter_host}" in
  127.0.0.1|localhost|0.0.0.0)
    echo "ERROR: SLIME_ADAPTER_PUBLIC_URL host '${_adapter_host}' is not reachable from AGS sandboxes." >&2
    echo "       Export a public/ALB (or node) URL that routes to THIS job's adapter port." >&2
    exit 3
    ;;
esac
unset _adapter_host
export SLIME_ADAPTER_PUBLIC_URL
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-${SLIME_ADAPTER_PUBLIC_URL}}"
# Path A listens on SLIME_ADAPTER_PORT; existing HyperPod GRPO ALBs target :9002.
export SLIME_ADAPTER_PORT="${SLIME_ADAPTER_PORT:-9002}"
export SHIM_PORT="${SHIM_PORT:-${SLIME_ADAPTER_PORT}}"
export SHIM_BIND_HOST="${SHIM_BIND_HOST:-0.0.0.0}"
export SLIME_ADAPTER_BIND_HOST="${SLIME_ADAPTER_BIND_HOST:-0.0.0.0}"

if [[ -z "${SLIME_AGENT_SANDBOX_BACKEND:-}" ]]; then
  echo "ERROR: SLIME_AGENT_SANDBOX_BACKEND is unset (expected ags)." >&2
  exit 4
fi
if [[ -n "${SLIME_AGENT_AGS_ENV_FILE:-}" ]]; then
  if [[ ! -f "${SLIME_AGENT_AGS_ENV_FILE}" ]]; then
    echo "ERROR: SLIME_AGENT_AGS_ENV_FILE does not exist: ${SLIME_AGENT_AGS_ENV_FILE}" >&2
    exit 4
  fi
elif [[ -z "${SLIME_AGENT_AGS_SECRET_ID:-}" || -z "${SLIME_AGENT_AGS_SECRET_KEY:-}" ]]; then
  echo "ERROR: set SLIME_AGENT_AGS_ENV_FILE to an existing credentials file," >&2
  echo "       or export both SLIME_AGENT_AGS_SECRET_ID and SLIME_AGENT_AGS_SECRET_KEY." >&2
  exit 4
fi

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "ERROR: HF_CHECKPOINT not found: ${HF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${REF_MODEL_PATH}" ]]; then
  echo "ERROR: REF_MODEL_PATH not found: ${REF_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "ERROR: PROMPT_DATA not found: ${PROMPT_DATA}" >&2
  exit 1
fi
if [[ "${PHASE}" != "train" && ! -f "${EVAL_DATA}" ]]; then
  echo "ERROR: EVAL_DATA not found: ${EVAL_DATA}" >&2
  exit 1
fi
if [[ "${PHASE}" == "eval" ]]; then
  _save_dir="${LOG_DIR}/slime_save"
  if [[ ! -d "${_save_dir}" ]] || [[ -z "$(find "${_save_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: PHASE=eval requires a non-empty checkpoint at ${_save_dir}." >&2
    echo "       Run PHASE=train|all first, or set LOG_DIR to an existing run." >&2
    exit 1
  fi
  unset _save_dir
fi

LOG_FILE="${LOG_DIR}/run.log"

# ============ model args ============
MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"
# shellcheck disable=SC1090
source "${MODEL_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${LOG_DIR}/slime_save"
  --save "${LOG_DIR}/slime_save"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --metadata-key extra_info
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --num-steps-per-rollout 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${MAX_GEN_LEN}"
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --rollout-abort-grace-sec "${ROLLOUT_ABORT_GRACE_SEC:-45}"
  --eval-temperature 0.6
  --eval-top-p 0.95
  --eval-top-k 20
  --rollout-stop-token-ids 248046 248044
  --use-dynamic-global-batch-size
  --micro-batch-size 1
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
  --custom-generate-function-path examples.claudecode_ags.step_reconstruct.hybrid_generate.hybrid_generate
  --custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
  --custom-reward-post-process-path examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards
  --custom-rollout-log-function-path examples.claudecode_ags.wandb_metrics.log_rollout_data
)

# Default-on filter (STEP_GRPO_FILTER=1); set 0 to skip wiring the path.
if [[ "${STEP_GRPO_FILTER}" != "0" ]]; then
  ROLLOUT_ARGS+=(
    --rollout-sample-filter-path examples.claudecode_ags.step_reconstruct.step_grpo_advantage.filter
  )
fi

RESUME_DEBUG_ROLLOUT_DATA="${RESUME_DEBUG_ROLLOUT_DATA:-1}"
if [[ "${RESUME_DEBUG_ROLLOUT_DATA}" = "0" ]]; then
  ROLLOUT_ARGS+=(--no-resume-debug-rollout-data)
fi

EVAL_ARGS=()
if [[ "${PHASE}" == "all" || "${PHASE}" == "eval" ]]; then
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data swebench_verified "${EVAL_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --eval-max-response-len "${MAX_GEN_LEN}"
  )
  if [[ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]]; then
    EVAL_ARGS+=(--skip-eval-before-train)
  fi
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
  --use-dynamic-batch-size
)

ALGO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.0001
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef "${ENTROPY_COEF:-0.0}"
  --eps-clip 1e-4
  --eps-clip-high 2e-4
)
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
USE_TIS="${USE_TIS:-1}"
if [[ "${USE_ROLLOUT_LOGPROBS}" == "1" && "${USE_TIS}" == "1" ]]; then
  echo "ERROR: USE_ROLLOUT_LOGPROBS=1 and USE_TIS=1 are mutually exclusive." >&2
  exit 1
fi
if [[ "${USE_ROLLOUT_LOGPROBS}" == "1" ]]; then
  ALGO_ARGS+=(--use-rollout-logprobs)
fi
if [[ "${USE_TIS}" == "1" ]]; then
  ALGO_ARGS+=(
    --use-tis
    --get-mismatch-metrics
    --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
    --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
    --eps-clip-c "${EPS_CLIP_C:-10.0}"
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 3e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
)
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi

SGLANG_HICACHE_ARGS=()
if [[ "${SGLANG_ENABLE_HICACHE}" == "1" ]]; then
  SGLANG_HICACHE_ARGS+=(--sglang-enable-hierarchical-cache)
  SGLANG_HICACHE_ARGS+=(--sglang-hicache-ratio "${SGLANG_HICACHE_RATIO}")
  if [[ -n "${SGLANG_HICACHE_SIZE:-}" ]]; then
    SGLANG_HICACHE_ARGS+=(--sglang-hicache-size "${SGLANG_HICACHE_SIZE}")
  fi
  SGLANG_HICACHE_ARGS+=(--sglang-hicache-write-policy "${SGLANG_HICACHE_WRITE_POLICY}")
  SGLANG_HICACHE_ARGS+=(--sglang-hicache-io-backend "${SGLANG_HICACHE_IO_BACKEND}")
  SGLANG_HICACHE_ARGS+=(--sglang-hicache-mem-layout "${SGLANG_HICACHE_MEM_LAYOUT}")
  if [[ -n "${SGLANG_HICACHE_STORAGE_BACKEND:-}" ]]; then
    SGLANG_HICACHE_ARGS+=(--sglang-hicache-storage-backend "${SGLANG_HICACHE_STORAGE_BACKEND}")
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-router-port "${SLIME_ROUTER_PORT}"
  --router-policy "${SLIME_SGLANG_ROUTER_POLICY}"
  --router-prefill-policy "${SLIME_SGLANG_ROUTER_PREFILL_POLICY}"
  --router-decode-policy "${SLIME_SGLANG_ROUTER_DECODE_POLICY}"
  --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}"
  "${SGLANG_HICACHE_ARGS[@]}"
  --sglang-context-length "${SLIME_SGLANG_CONTEXT_LENGTH}"
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
  --sglang-mamba-scheduler-strategy "${SGLANG_MAMBA_SCHEDULER_STRATEGY}"
)
if [[ "${SLIME_PREFILL_NUM_SERVERS}" != "0" ]]; then
  SGLANG_ARGS+=(--prefill-num-servers "${SLIME_PREFILL_NUM_SERVERS}")
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --colocate
)

if [[ -z "${WANDB_KEY:-}" ]]; then
  _wandb_key_file="${WANDB_KEY_FILE:-${HOME}/.config/jiaxicao/wandb_api_key}"
  if [[ -f "${_wandb_key_file}" ]]; then
    WANDB_KEY="$(tr -d '[:space:]' < "${_wandb_key_file}")"
  fi
fi

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-coding-rl}"
    --wandb-group "${WANDB_GROUP:-${EXP_TAG}}"
    --wandb-key "${WANDB_KEY}"
    --wandb-dir "${LOG_DIR}/wandb"
  )
  if [[ -n "${WANDB_TEAM:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
fi

# ============ ray network ============
# Kubeflow sets MASTER_ADDR for multi-node; fall back to local IP for 1-node.
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
export MASTER_PORT="${MASTER_PORT:-6379}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export SLIME_HEAD_HOST="${SLIME_HEAD_HOST:-${MASTER_ADDR}}"
export no_proxy="127.0.0.1,${MASTER_ADDR},${SLIME_HEAD_HOST}"
export NO_PROXY="${no_proxy}"

TRAIN_CMD=(
  python3 -u train.py
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${WANDB_ARGS[@]}"
  "${ALGO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
)

echo "======================================================================"
echo "Path A hybrid step-GRPO (PHASE=${PHASE} RUN=${RUN} nodes=${ACTOR_NUM_NODES})"
echo "SLIME_DIR=${SLIME_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
echo "TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} ROLLOUT_GPUS=${ROLLOUT_NUM_GPUS}"
echo "batch: rollout=${ROLLOUT_BATCH_SIZE} n_samples=${N_SAMPLES_PER_PROMPT} global=${GLOBAL_BATCH_SIZE} num_rollout=${NUM_ROLLOUT}"
echo "SAVE_INTERVAL=${SAVE_INTERVAL} OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD}"
echo "PROMPT_DATA=${PROMPT_DATA}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "SLIME_ADAPTER_PUBLIC_URL=${SLIME_ADAPTER_PUBLIC_URL}"
echo "======================================================================"
printf ' %q' "${TRAIN_CMD[@]}"
echo
echo "======================================================================"

if [[ "${RUN}" != "1" ]]; then
  echo "Dry run only. Set RUN=1 to start Ray head and submit train.py."
  exit 0
fi

mkdir -p \
  "${LOG_DIR}/slime_save" \
  "${LOG_DIR}/wandb" \
  "${LOG_DIR}/rollout_dumps" \
  "${LOG_DIR}/launcher_logs"

# ============ bring up ray (multi-node via Kubeflow RANK) ============
# RANK=0 (Master): start Ray head + train. RANK>0 (Worker): join and idle.
NODE_RANK="${RANK:-${PET_NODE_RANK:-0}}"
EXPECTED_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))
USE_EXTERNAL_RAY="${USE_EXTERNAL_RAY:-0}"

if [[ "${USE_EXTERNAL_RAY}" != "1" ]]; then
  pkill -9 sglang 2>/dev/null || true
  sleep 2
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  sleep 2

  if [[ "${NODE_RANK}" != "0" ]]; then
    echo "Ray worker NODE_RANK=${NODE_RANK}; joining ${MASTER_ADDR}:${RAY_GCS_PORT}"
    _worker_ip="$(hostname -I | awk '{print $1}')"
    for _i in $(seq 1 180); do
      if ray start --address="${MASTER_ADDR}:${RAY_GCS_PORT}" \
          --node-ip-address "${_worker_ip}" \
          --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
          --disable-usage-stats; then
        echo "Ray worker joined."
        break
      fi
      echo "Waiting for Ray head... (${_i}/180)"
      sleep 5
    done
    echo "Worker idle (Ray joined); sleeping until Master finishes."
    sleep infinity
  fi

  # Master / single-node head
  # Prefer operator-provided MASTER_ADDR when multi-node; else local IP.
  if [[ "${ACTOR_NUM_NODES}" -le 1 ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
  fi
  _head_ip="$(hostname -I | awk '{print $1}')"
  ray start --head \
    --node-ip-address "${_head_ip}" \
    --port "${RAY_GCS_PORT}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}"

  echo "Waiting for Ray cluster GPUs >= ${EXPECTED_GPUS} (nodes=${ACTOR_NUM_NODES})..."
  for _i in $(seq 1 180); do
    _gpus="$(python3 - <<'PY'
import re, subprocess
try:
    out = subprocess.check_output(["ray", "status"], text=True, stderr=subprocess.STDOUT)
except Exception as e:
    print(0)
    raise SystemExit
# Match lines like "0.0/16.0 GPU" or "16.0/16.0 GPU"
m = re.search(r"([\d.]+)/([\d.]+)\s+GPU", out)
print(int(float(m.group(2))) if m else 0)
PY
)"
    echo "  ray GPUs total=${_gpus} (want ${EXPECTED_GPUS}) try=${_i}"
    if [[ "${_gpus}" -ge "${EXPECTED_GPUS}" ]]; then
      break
    fi
    sleep 5
  done
  ray status
else
  echo "USE_EXTERNAL_RAY=1; reusing existing Ray cluster."
  ray status --address="${MASTER_ADDR}:${RAY_GCS_PORT}" || true
fi

cd "${SLIME_DIR}"

RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

env = {}
exact = {
    "no_proxy", "NO_PROXY",
    "MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
    "SLIME_HEAD_HOST", "SLIME_DIR", "CUDA_DEVICE_MAX_CONNECTIONS",
}
prefixes = ("SLIME_", "ANTHROPIC_", "AGS_", "CLAUDE_", "SGLANG_", "BASH_", "STEP_GRPO_", "WANDB_")
for key, value in os.environ.items():
    if not isinstance(value, str) or value == "":
        continue
    if key in exact or any(key.startswith(prefix) for prefix in prefixes):
        env[key] = value

env["TP_SOCKET_IFNAME"] = os.environ.get("GLOO_SOCKET_IFNAME", "eth0")
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}:{os.environ.get('PYTHONPATH', '')}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env.setdefault("NCCL_NVLS_ENABLE", "0")
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_CMD[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "LOG_FILE=${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
