#!/usr/bin/env bash
# Skeleton GRPO launcher for Claude Code + AGS + SWE.
#
# Sources the two env files (claude_code.env + slime_ags.env), checks required
# paths, and prints the train.py invocation. Set RUN=1 to execute (needs Ray +
# GPUs). Default is dry-run only.
#
# Required overrides (export before running or in slime_ags.env):
#   HF_CHECKPOINT      — HuggingFace model directory
#   REF_MODEL_PATH     — Megatron torch_dist checkpoint
#   PROMPT_DATA        — training JSONL/parquet path(s)
#   EVAL_DATA          — eval dataset path
#   SLIME_ADAPTER_PUBLIC_URL — host:port reachable from AGS sandboxes
#   SLIME_AGENT_AGS_*  — AGS credentials (or SLIME_AGENT_AGS_ENV_FILE)
#
# Usage:
#   bash examples/claudecode_ags/launch/run_grpo_example.sh
#   RUN=1 bash examples/claudecode_ags/launch/run_grpo_example.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${EXAMPLE_DIR}/../.." && pwd)}"

# shellcheck disable=SC1091
source "${EXAMPLE_DIR}/env/load_env.sh"

RUN="${RUN:-0}"

# --- paths (override in environment) ---
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/model}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/path/to/model_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/path/to/swe_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-/path/to/swe_eval.jsonl}"
LOG_DIR="${LOG_DIR:-${SLIME_DIR}/runs/cc_ags_grpo_example}"

REQUIRED_VARS=(
  HF_CHECKPOINT
  REF_MODEL_PATH
  PROMPT_DATA
  EVAL_DATA
  SLIME_ADAPTER_PUBLIC_URL
  SLIME_AGENT_SANDBOX_BACKEND
)

missing=()
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("${var}")
  fi
done
if ((${#missing[@]} > 0)); then
  echo "ERROR: missing required env vars: ${missing[*]}" >&2
  echo "Set them in slime_ags.env or export before launching." >&2
  exit 1
fi

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "ERROR: HF_CHECKPOINT not found: ${HF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${REF_MODEL_PATH}" ]]; then
  echo "ERROR: REF_MODEL_PATH not found: ${REF_MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}/rollout_dumps"

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --metadata-key extra_info
  --eval-data "${EVAL_DATA}"
  --num-rollout 1
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --num-steps-per-rollout 1
  --global-batch-size 1
  --custom-generate-function-path examples.claudecode_ags.generate.generate
  --custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
  --custom-reward-post-process-path slime.rollout.fanout_grpo.post_process_rewards
)

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${LOG_DIR}/slime_save"
  --save "${LOG_DIR}/slime_save"
)

TRAIN_CMD=(
  python3 -u train.py
  --actor-num-nodes 1
  --actor-num-gpus-per-node 1
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
)

echo "======================================================================"
echo "Claude Code + AGS GRPO example (RUN=${RUN})"
echo "SLIME_DIR=${SLIME_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "Loaded env from: ${EXAMPLE_DIR}/env/"
echo "======================================================================"
printf ' %q' "${TRAIN_CMD[@]}"
echo
echo "======================================================================"

if [[ "${RUN}" != "1" ]]; then
  echo "Dry run only. Set RUN=1 to execute (requires Ray cluster + GPUs)."
  exit 0
fi

cd "${SLIME_DIR}"
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/run.log"
