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
# Export a public/ALB (or node) URL that routes to THIS job's adapter port
# (SLIME_ADAPTER_PORT, default 18001). Do NOT reuse the deleted L2 Ingress /
# jiaxicao-cc-ags-adapter Deployment URL.
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
#   export SLIME_ADAPTER_PUBLIC_URL=http://<reachable-host>:18001
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

# ============ topology (1 node / 8 GPU colocate) ============
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-1}"
export ETP_SIZE="${ETP_SIZE:-1}"

ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
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

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

if [[ "${PHASE}" == "eval" ]]; then
  # train.py: if num_rollout == 0 and eval_interval is set → eval-only.
  NUM_ROLLOUT="${NUM_ROLLOUT:-0}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-21}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-${NUM_ROLLOUT}}"
fi

EXP_TAG="${EXP_TAG:-qwen35_9b_cc_ags_1node_grpo_debug}"
LOG_DIR="${LOG_DIR:-/mnt/sn-007/jiaxicao/checkpoints/cc-ags/${EXP_TAG}}"
RUN_ROOT="${RUN_ROOT:-${LOG_DIR}}"

# ============ timeouts (export if unset) ============
export SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-1800}"
export SLIME_CC_EVAL_TIMEOUT_SEC="${SLIME_CC_EVAL_TIMEOUT_SEC:-600}"
export SLIME_AGENT_AGS_TIMEOUT="${SLIME_AGENT_AGS_TIMEOUT:-45m}"

# ============ asserts ============
if [[ -z "${SLIME_ADAPTER_PUBLIC_URL:-}" || "${SLIME_ADAPTER_PUBLIC_URL}" == *REPLACE_WITH* ]]; then
  echo "ERROR: set SLIME_ADAPTER_PUBLIC_URL to a public/ALB URL AGS can reach for THIS job's adapter." >&2
  echo "       Do not use 127.0.0.1 or the deleted L2 Ingress URL." >&2
  exit 3
fi
SLIME_ADAPTER_PUBLIC_URL="${SLIME_ADAPTER_PUBLIC_URL%/}"
export SLIME_ADAPTER_PUBLIC_URL
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-${SLIME_ADAPTER_PUBLIC_URL}}"

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

mkdir -p \
  "${LOG_DIR}/slime_save" \
  "${LOG_DIR}/wandb" \
  "${LOG_DIR}/rollout_dumps" \
  "${LOG_DIR}/launcher_logs"
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
  --custom-generate-function-path examples.claudecode_ags.generate.generate
  --custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
  --custom-reward-post-process-path slime.rollout.fanout_grpo.post_process_rewards
  --custom-rollout-log-function-path examples.claudecode_ags.wandb_metrics.log_rollout_data
)

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
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

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

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-cc-ags}"
    --wandb-group "${WANDB_GROUP:-${EXP_TAG}}"
    --wandb-key "${WANDB_KEY}"
    --wandb-dir "${LOG_DIR}/wandb"
  )
fi

# ============ ray network (1-node head only) ============
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
echo "Path A GRPO 1-node debug (PHASE=${PHASE} RUN=${RUN})"
echo "SLIME_DIR=${SLIME_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
echo "TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} ROLLOUT_GPUS=${ROLLOUT_NUM_GPUS}"
echo "batch: rollout=${ROLLOUT_BATCH_SIZE} n_samples=${N_SAMPLES_PER_PROMPT} global=${GLOBAL_BATCH_SIZE} num_rollout=${NUM_ROLLOUT}"
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

# ============ bring up ray (1-node head; no worker loop) ============
USE_EXTERNAL_RAY="${USE_EXTERNAL_RAY:-0}"
if [[ "${USE_EXTERNAL_RAY}" != "1" ]]; then
  pkill -9 sglang 2>/dev/null || true
  sleep 2
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  sleep 2

  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --port "${RAY_GCS_PORT}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}"

  echo "Waiting for Ray head to stabilize..."
  sleep 10
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
prefixes = ("SLIME_", "ANTHROPIC_", "AGS_", "CLAUDE_", "SGLANG_", "BASH_")
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
