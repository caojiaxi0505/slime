#!/usr/bin/env bash
# Qwen3.5-9B SFT on DeepSeek-v4-pro SWE-Gym clean all-trials data.
set -euo pipefail

# Clean stale local processes inside a reused training container.
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

export PYTHONUNBUFFERED=1

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

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-131071}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

WANDB_PROJECT="${WANDB_PROJECT:-coding-rl}"
WANDB_TEAM="${WANDB_TEAM:-models-tencent7723}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"

for p in "${SLIME_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: path not found: ${p}" >&2
    exit 1
  fi
done

if [[ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "ERROR: ${SAVE_PATH} already has a checkpoint." >&2
  echo "       Use a fresh EXP_TAG/LOG_DIR, or set ALLOW_RESUME=1 to continue." >&2
  exit 4
fi

mkdir -p "${SAVE_PATH}" "${WANDB_DIR}" "${LOG_DIR}/launcher_logs"

cd "${SLIME_DIR}"
git config --global --add safe.directory "${SLIME_DIR}" || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"
# shellcheck disable=SC1090
source "${MODEL_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${LOAD_PATH}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${PROMPT_DATA}"
  --input-key messages
  --metadata-key metadata
  --rollout-shuffle
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --n-samples-per-prompt 1
  --num-steps-per-rollout 1
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

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
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-5}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR:-1e-6}"
  --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.1}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.95}"
  --use-distributed-optimizer
)

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-key "${WANDB_KEY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-team "${WANDB_TEAM}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
    --disable-wandb-random-suffix
  )
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
