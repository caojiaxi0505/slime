#!/usr/bin/env bash
# Submit SGLang + adapter Deployment and dedicated ALB Ingress in sn5-system-intern.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DEP_TEMPLATE="${SCRIPT_DIR}/deployment.yaml.template"
ING_TEMPLATE="${SCRIPT_DIR}/service-ingress.yaml.template"

usage() {
  cat <<'EOF'
Usage: submit_deploy.sh [options]

Deploy SGLang + SegmentedAnthropicAdapter with a dedicated ALB Ingress.
Does not auto-write slime_ags.env — copy the printed URL manually.

Options:
  --help       Show help
  --dry-run    Print rendered YAML only
  --delete     Delete Deployment/Service/Ingress for this name prefix

Required env:
  HF_CHECKPOINT   Model dir on FSx (under /mnt/sn-007)

Useful env (defaults):
  IMAGE_URI          ECR slime image (cc-ags-swe-20260710-171230)
  K8S_NAMESPACE      sn5-system-intern
  NAME_PREFIX        jiaxicao-cc-ags-adapter
  NUM_GPUS / TP_SIZE 1
  ADAPTER_PORT       18001
  SGLANG_PORT        30000
  SGLANG_TOOL_CALL_PARSER qwen3_coder
  SGLANG_REASONING_PARSER qwen3
  FSX_PVC_NAME       youtu-sn2-007
  CPU/MEMORY         16/64Gi request, 32/256Gi limit
EOF
}

DRY_RUN=0
DELETE=0
for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --delete) DELETE=1 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
NAME_PREFIX="${NAME_PREFIX:-jiaxicao-cc-ags-adapter}"
DEPLOY_NAME="${DEPLOY_NAME:-${NAME_PREFIX}}"
SERVICE_NAME="${SERVICE_NAME:-${NAME_PREFIX}}"
INGRESS_NAME="${INGRESS_NAME:-${NAME_PREFIX}}"
APP_LABEL="${APP_LABEL:-${NAME_PREFIX}}"

IMAGE_URI="${IMAGE_URI:-085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:cc-ags-swe-20260710-171230}"
FSX_PVC_NAME="${FSX_PVC_NAME:-youtu-sn2-007}"
FSX_MOUNT_PATH="${FSX_MOUNT_PATH:-/mnt/sn-007}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}}"
LAUNCH_DIR="${LAUNCH_DIR:-${SCRIPT_DIR}}"
ENTRYPOINT_PATH="${ENTRYPOINT_PATH:-${LAUNCH_DIR}/entrypoint.sh}"

ADAPTER_PORT="${ADAPTER_PORT:-18001}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-${TP_SIZE}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
CPU_REQUEST="${CPU_REQUEST:-16}"
CPU_LIMIT="${CPU_LIMIT:-32}"
MEMORY_REQUEST="${MEMORY_REQUEST:-64Gi}"
MEMORY_LIMIT="${MEMORY_LIMIT:-256Gi}"
SHM_SIZE="${SHM_SIZE:-64Gi}"

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete deployment "${DEPLOY_NAME}" --ignore-not-found
  kubectl -n "${K8S_NAMESPACE}" delete service "${SERVICE_NAME}" --ignore-not-found
  kubectl -n "${K8S_NAMESPACE}" delete ingress "${INGRESS_NAME}" --ignore-not-found
  echo "Deleted ${DEPLOY_NAME} / ${SERVICE_NAME} / ${INGRESS_NAME} in ${K8S_NAMESPACE}"
  exit 0
fi

if [[ -z "${HF_CHECKPOINT:-}" ]]; then
  echo "ERROR: set HF_CHECKPOINT to the model directory on FSx" >&2
  exit 1
fi
case "${HF_CHECKPOINT}" in
  "${FSX_MOUNT_PATH}"|"${FSX_MOUNT_PATH}"/*) ;;
  *)
    echo "ERROR: HF_CHECKPOINT must be under ${FSX_MOUNT_PATH} (PVC ${FSX_PVC_NAME})" >&2
    exit 1
    ;;
esac
if [[ ! -d "${HF_CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: HF_CHECKPOINT not found: ${HF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -x "${ENTRYPOINT_PATH}" ]]; then
  chmod +x "${ENTRYPOINT_PATH}" || true
fi

export DEPLOY_NAME SERVICE_NAME INGRESS_NAME APP_LABEL K8S_NAMESPACE
export IMAGE_URI HF_CHECKPOINT FSX_PVC_NAME FSX_MOUNT_PATH
export SLIME_ROOT LAUNCH_DIR ENTRYPOINT_PATH
export ADAPTER_PORT SGLANG_PORT TP_SIZE NUM_GPUS SGLANG_MEM_FRACTION_STATIC
export SGLANG_TOOL_CALL_PARSER SGLANG_REASONING_PARSER
export EXTRA_SGLANG_ARGS CPU_REQUEST CPU_LIMIT MEMORY_REQUEST MEMORY_LIMIT SHM_SIZE

# EXTRA_SGLANG_ARGS may be empty — envsubst still needs the var
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS}"

RENDERED="$(mktemp)"
{
  envsubst '${DEPLOY_NAME} ${SERVICE_NAME} ${INGRESS_NAME} ${APP_LABEL} ${K8S_NAMESPACE} ${IMAGE_URI} ${HF_CHECKPOINT} ${FSX_PVC_NAME} ${FSX_MOUNT_PATH} ${SLIME_ROOT} ${LAUNCH_DIR} ${ENTRYPOINT_PATH} ${ADAPTER_PORT} ${SGLANG_PORT} ${TP_SIZE} ${NUM_GPUS} ${SGLANG_MEM_FRACTION_STATIC} ${SGLANG_TOOL_CALL_PARSER} ${SGLANG_REASONING_PARSER} ${EXTRA_SGLANG_ARGS} ${CPU_REQUEST} ${CPU_LIMIT} ${MEMORY_REQUEST} ${MEMORY_LIMIT} ${SHM_SIZE}' \
    < "${DEP_TEMPLATE}"
  echo '---'
  envsubst '${DEPLOY_NAME} ${SERVICE_NAME} ${INGRESS_NAME} ${APP_LABEL} ${K8S_NAMESPACE} ${IMAGE_URI} ${HF_CHECKPOINT} ${FSX_PVC_NAME} ${FSX_MOUNT_PATH} ${SLIME_ROOT} ${LAUNCH_DIR} ${ENTRYPOINT_PATH} ${ADAPTER_PORT} ${SGLANG_PORT} ${TP_SIZE} ${NUM_GPUS} ${SGLANG_MEM_FRACTION_STATIC} ${SGLANG_TOOL_CALL_PARSER} ${SGLANG_REASONING_PARSER} ${EXTRA_SGLANG_ARGS} ${CPU_REQUEST} ${CPU_LIMIT} ${MEMORY_REQUEST} ${MEMORY_LIMIT} ${SHM_SIZE}' \
    < "${ING_TEMPLATE}"
} > "${RENDERED}"

echo "==> deploy ${DEPLOY_NAME} in ${K8S_NAMESPACE}"
echo "    image: ${IMAGE_URI}"
echo "    model: ${HF_CHECKPOINT}"
echo "    gpus:  ${NUM_GPUS}  tp: ${TP_SIZE}  adapter_port: ${ADAPTER_PORT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${RENDERED}"
  rm -f "${RENDERED}"
  exit 0
fi

kubectl apply -n "${K8S_NAMESPACE}" -f "${RENDERED}"
rm -f "${RENDERED}"

cat <<EOF

Applied Deployment/${DEPLOY_NAME}, Service/${SERVICE_NAME}, Ingress/${INGRESS_NAME}

Watch pod:
  kubectl -n ${K8S_NAMESPACE} get pods -l app=${APP_LABEL} -w
  kubectl -n ${K8S_NAMESPACE} logs -f deploy/${DEPLOY_NAME}

Wait for Ingress ADDRESS (ALB DNS):
  kubectl -n ${K8S_NAMESPACE} get ingress ${INGRESS_NAME} -w

Then manually set (do not commit secrets/env with real URL if undesired):
  export SLIME_ADAPTER_PUBLIC_URL=http://<ADDRESS>
  # also put into examples/claudecode_ags/env/slime_ags.env

Health check:
  curl -fsS "\$SLIME_ADAPTER_PUBLIC_URL/health"

Teardown:
  ./submit_deploy.sh --delete
EOF
