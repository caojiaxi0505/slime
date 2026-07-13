#!/usr/bin/env bash
# Start SGLang then the Anthropic segmented adapter in one container.
# Intended to run from the ECR slime image with FSx worktree mounted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer worktree on FSx when mounted; fall back to image copy of this script's dir.
LAUNCH_DIR="${LAUNCH_DIR:-${SCRIPT_DIR}}"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${LAUNCH_DIR}/../../../.." && pwd)}"

HF_CHECKPOINT="${HF_CHECKPOINT:?HF_CHECKPOINT is required}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
ADAPTER_PORT="${ADAPTER_PORT:-${SLIME_ADAPTER_PORT:-18001}}"
TP_SIZE="${TP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-${TP_SIZE}}"
MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

export PYTHONPATH="${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SLIME_ADAPTER_PORT="${ADAPTER_PORT}"
export ADAPTER_PORT
export SGLANG_PORT
export SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:${SGLANG_PORT}}"
export HF_CHECKPOINT

echo "[entrypoint] SLIME_ROOT=${SLIME_ROOT}"
echo "[entrypoint] HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "[entrypoint] SGLANG_PORT=${SGLANG_PORT} TP_SIZE=${TP_SIZE} NUM_GPUS=${NUM_GPUS} ADAPTER_PORT=${ADAPTER_PORT}"
echo "[entrypoint] EXTRA_SGLANG_ARGS=${EXTRA_SGLANG_ARGS}"

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "ERROR: HF_CHECKPOINT not a directory: ${HF_CHECKPOINT}" >&2
  exit 1
fi

# shellcheck disable=SC2086
python -m sglang.launch_server \
  --model-path "${HF_CHECKPOINT}" \
  --host 0.0.0.0 \
  --port "${SGLANG_PORT}" \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  ${EXTRA_SGLANG_ARGS} &
SGLANG_PID=$!

cleanup() {
  echo "[entrypoint] shutting down sglang pid=${SGLANG_PID}"
  kill "${SGLANG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[entrypoint] waiting for SGLang on :${SGLANG_PORT}"
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1 \
    || curl -fsS "http://127.0.0.1:${SGLANG_PORT}/healthz" >/dev/null 2>&1 \
    || curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[entrypoint] SGLang ready after ${i}s"
    break
  fi
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "ERROR: sglang process exited early" >&2
    wait "${SGLANG_PID}" || true
    exit 1
  fi
  if [[ "${i}" -eq 180 ]]; then
    echo "ERROR: timed out waiting for SGLang" >&2
    exit 1
  fi
  sleep 1
done

echo "[entrypoint] starting adapter"
exec python "${LAUNCH_DIR}/serve_adapter.py"
