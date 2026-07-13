#!/usr/bin/env bash
# Submit a Kaniko Job to build the full slime image and push to ECR.
# Local docker build/push is intentionally unsupported.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/job.yaml.template"

usage() {
  cat <<'EOF'
Usage: submit_build.sh [options]

Build the full slime training image with Kaniko on HyperPod/K8s and push to ECR.
This script never runs local docker build/push.

Options:
  --help                 Show this help
  --create-repo          Create ECR repository if missing (default: do not create)
  --dry-run              Render Job YAML to stdout; do not kubectl apply
  --follow               After apply, kubectl logs -f the Job pod

Environment (defaults):
  AWS_REGION             ap-southeast-3
  ECR_REPOSITORY         sn5/jiaxicao/slime
  AWS_ACCOUNT_ID         from: aws sts get-caller-identity
  IMAGE_TAG              cc-ags-swe-YYYYMMDD-HHMMSS
  K8S_NAMESPACE          sn5-system-intern
  BUILD_CONTEXT          this worktree root (must be visible on cluster nodes)
  DOCKERFILE             docker/Dockerfile.kaniko
  SERVICE_ACCOUNT        default  (override if you have an IRSA SA)
  DOCKER_CONFIG_SECRET   jiaxicao-kaniko-ecr  (ECR auth for Kaniko; auto-refreshed)
  SKIP_DOCKER_CONFIG     0  set 1 to skip creating/refreshing the secret
  FSX_PVC_NAME           youtu-sn2-007  (PVC in sn5-system-intern; mounts at /mnt/sn-007)
  KANIKO_EXECUTOR_IMAGE  gcr.io/kaniko-project/executor:v1.23.2
  ENABLE_EFA             1  (requires docker/efa/*.sh; see docker/efa/README.md)
  EFA_SCRIPTS_DIR        optional dir to copy EFA scripts into docker/efa/
  KANIKO_CACHE           false
  SGLANG_IMAGE_TAG       v0.5.13-cu129
  PATCH_VERSION          latest
  CPU_REQUEST / LIMIT    180 / 192
  MEMORY_REQUEST/LIMIT   1024Gi / 1800Gi
  TTL_SECONDS_AFTER_FINISHED  86400
  NODE_SELECTOR_YAML     optional raw YAML lines (indented) for nodeSelector

Examples:
  ./submit_build.sh
  ENABLE_EFA=0 ./submit_build.sh --dry-run
  EFA_SCRIPTS_DIR=/path/to/efa_scripts ./submit_build.sh --follow
  ./submit_build.sh --create-repo
EOF
}

CREATE_REPO=0
DRY_RUN=0
FOLLOW=0
for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    --create-repo) CREATE_REPO=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --follow) FOLLOW=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

AWS_REGION="${AWS_REGION:-ap-southeast-3}"
ECR_REPOSITORY="${ECR_REPOSITORY:-sn5/jiaxicao/slime}"
K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${REPO_ROOT}}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.kaniko}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-default}"
DOCKER_CONFIG_SECRET="${DOCKER_CONFIG_SECRET:-jiaxicao-kaniko-ecr}"
SKIP_DOCKER_CONFIG="${SKIP_DOCKER_CONFIG:-0}"
FSX_PVC_NAME="${FSX_PVC_NAME:-youtu-sn2-007}"
FSX_MOUNT_PATH="${FSX_MOUNT_PATH:-/mnt/sn-007}"
KANIKO_EXECUTOR_IMAGE="${KANIKO_EXECUTOR_IMAGE:-gcr.io/kaniko-project/executor:v1.23.2}"
ENABLE_EFA="${ENABLE_EFA:-1}"
KANIKO_CACHE="${KANIKO_CACHE:-false}"
SGLANG_IMAGE_TAG="${SGLANG_IMAGE_TAG:-v0.5.13-cu129}"
PATCH_VERSION="${PATCH_VERSION:-latest}"
CPU_REQUEST="${CPU_REQUEST:-180}"
CPU_LIMIT="${CPU_LIMIT:-192}"
MEMORY_REQUEST="${MEMORY_REQUEST:-1024Gi}"
MEMORY_LIMIT="${MEMORY_LIMIT:-1800Gi}"
TTL_SECONDS_AFTER_FINISHED="${TTL_SECONDS_AFTER_FINISHED:-86400}"
NODE_SELECTOR_YAML="${NODE_SELECTOR_YAML:-}"
EFA_SCRIPTS_DIR="${EFA_SCRIPTS_DIR:-}"

IMAGE_TAG="${IMAGE_TAG:-cc-ags-swe-$(date +%Y%m%d-%H%M%S)}"
# K8s label values: replace chars that are invalid in labels
IMAGE_TAG_LABEL="$(echo "${IMAGE_TAG}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g' | cut -c1-63)"

if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
  AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
fi
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
JOB_NAME="kaniko-slime-$(echo "${IMAGE_TAG_LABEL}" | tr '.' '-' | cut -c1-32)-$(date +%s | tail -c 6)"

echo "==> Kaniko slime image build"
echo "    namespace:     ${K8S_NAMESPACE}"
echo "    context:       ${BUILD_CONTEXT}"
echo "    dockerfile:    ${DOCKERFILE}"
echo "    destination:   ${ECR_URI}:${IMAGE_TAG}"
echo "    serviceAccount:${SERVICE_ACCOUNT}"
echo "    dockerSecret:  ${DOCKER_CONFIG_SECRET}"
echo "    fsx PVC:       ${FSX_PVC_NAME} -> ${FSX_MOUNT_PATH}"
echo "    ENABLE_EFA:    ${ENABLE_EFA}"
echo "    job:           ${JOB_NAME}"
echo
echo "NOTE: Local docker build/push is forbidden. Build runs only inside the cluster."

# BUILD_CONTEXT must live under the FSx mount path so the PVC can see it.
case "${BUILD_CONTEXT}" in
  "${FSX_MOUNT_PATH}"|"${FSX_MOUNT_PATH}"/*) ;;
  *)
    echo "ERROR: BUILD_CONTEXT (${BUILD_CONTEXT}) must be under ${FSX_MOUNT_PATH} (PVC ${FSX_PVC_NAME})." >&2
    exit 1
    ;;
esac

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: missing template ${TEMPLATE}" >&2
  exit 1
fi
if [[ ! -d "${BUILD_CONTEXT}" ]]; then
  echo "ERROR: BUILD_CONTEXT does not exist: ${BUILD_CONTEXT}" >&2
  exit 1
fi
if [[ ! -f "${BUILD_CONTEXT}/${DOCKERFILE}" ]]; then
  echo "ERROR: dockerfile not found: ${BUILD_CONTEXT}/${DOCKERFILE}" >&2
  exit 1
fi

# Optional: stage EFA scripts from an explicit path under your own tree only.
# Do not default to other users' directories.
if [[ -n "${EFA_SCRIPTS_DIR}" ]]; then
  mkdir -p "${BUILD_CONTEXT}/docker/efa"
  cp -f "${EFA_SCRIPTS_DIR}/install-efa-in-container.sh" \
        "${EFA_SCRIPTS_DIR}/fix-efa-conflict.sh" \
        "${BUILD_CONTEXT}/docker/efa/"
  chmod +x "${BUILD_CONTEXT}/docker/efa/"*.sh
  echo "==> Copied EFA scripts from ${EFA_SCRIPTS_DIR}"
fi

if [[ "${ENABLE_EFA}" == "1" ]]; then
  if [[ ! -f "${BUILD_CONTEXT}/docker/efa/install-efa-in-container.sh" ]] || \
     [[ ! -f "${BUILD_CONTEXT}/docker/efa/fix-efa-conflict.sh" ]]; then
    echo "ERROR: ENABLE_EFA=1 but EFA scripts missing under ${BUILD_CONTEXT}/docker/efa/" >&2
    echo "Place install-efa-in-container.sh and fix-efa-conflict.sh in your own docker/efa/," >&2
    echo "or set EFA_SCRIPTS_DIR to a path you own, or pass ENABLE_EFA=0 for a non-EFA image." >&2
    exit 1
  fi
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl not found" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if ! kubectl get namespace "${K8S_NAMESPACE}" >/dev/null 2>&1; then
    echo "ERROR: namespace ${K8S_NAMESPACE} not found / not readable" >&2
    exit 1
  fi
  if ! kubectl get pvc "${FSX_PVC_NAME}" -n "${K8S_NAMESPACE}" >/dev/null 2>&1; then
    echo "ERROR: PVC ${FSX_PVC_NAME} not found in ${K8S_NAMESPACE}" >&2
    exit 1
  fi
fi

if command -v aws >/dev/null 2>&1; then
  if ! aws ecr describe-repositories --region "${AWS_REGION}" --repository-names "${ECR_REPOSITORY}" >/dev/null 2>&1; then
    if [[ "${CREATE_REPO}" == "1" ]]; then
      echo "==> Creating ECR repository ${ECR_REPOSITORY} in ${AWS_REGION}"
      aws ecr create-repository --region "${AWS_REGION}" --repository-name "${ECR_REPOSITORY}"
    else
      echo "WARNING: ECR repository ${ECR_REPOSITORY} not found in ${AWS_REGION}."
      echo "  Create once with:"
      echo "    aws ecr create-repository --region ${AWS_REGION} --repository-name ${ECR_REPOSITORY}"
      echo "  Or re-run with --create-repo"
      if [[ "${DRY_RUN}" != "1" ]]; then
        exit 1
      fi
    fi
  fi
else
  echo "WARNING: aws CLI not found; skipping ECR repository check"
fi

# Refresh short-lived ECR docker config for Kaniko (token ~12h).
if [[ "${SKIP_DOCKER_CONFIG}" != "1" && "${DRY_RUN}" != "1" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI required to refresh ${DOCKER_CONFIG_SECRET}" >&2
    exit 1
  fi
  echo "==> Refreshing docker-config secret ${DOCKER_CONFIG_SECRET}"
  TMP_DOCKER_DIR="$(mktemp -d)"
  ECR_PASSWORD="$(aws ecr get-login-password --region "${AWS_REGION}")"
  AUTH_B64="$(printf 'AWS:%s' "${ECR_PASSWORD}" | base64 -w0 2>/dev/null || printf 'AWS:%s' "${ECR_PASSWORD}" | base64)"
  REGISTRY_HOST="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  cat > "${TMP_DOCKER_DIR}/config.json" <<EOF
{
  "auths": {
    "${REGISTRY_HOST}": {
      "auth": "${AUTH_B64}"
    }
  }
}
EOF
  kubectl -n "${K8S_NAMESPACE}" create secret generic "${DOCKER_CONFIG_SECRET}" \
    --from-file=config.json="${TMP_DOCKER_DIR}/config.json" \
    --dry-run=client -o yaml | kubectl apply -f -
  rm -rf "${TMP_DOCKER_DIR}"
fi

export JOB_NAME K8S_NAMESPACE IMAGE_TAG IMAGE_TAG_LABEL TTL_SECONDS_AFTER_FINISHED
export SERVICE_ACCOUNT KANIKO_EXECUTOR_IMAGE DOCKERFILE ECR_URI AWS_REGION
export SGLANG_IMAGE_TAG PATCH_VERSION ENABLE_EFA KANIKO_CACHE
export CPU_REQUEST CPU_LIMIT MEMORY_REQUEST MEMORY_LIMIT BUILD_CONTEXT
export DOCKER_CONFIG_SECRET FSX_PVC_NAME FSX_MOUNT_PATH
# NODE_SELECTOR_YAML may be empty — leave a blank line comment if unset
if [[ -z "${NODE_SELECTOR_YAML}" ]]; then
  NODE_SELECTOR_YAML="# nodeSelector: (none)"
fi
export NODE_SELECTOR_YAML

RENDERED="$(mktemp)"
# shellcheck disable=SC2016
envsubst '${JOB_NAME} ${K8S_NAMESPACE} ${IMAGE_TAG} ${IMAGE_TAG_LABEL} ${TTL_SECONDS_AFTER_FINISHED} ${SERVICE_ACCOUNT} ${KANIKO_EXECUTOR_IMAGE} ${DOCKERFILE} ${ECR_URI} ${AWS_REGION} ${SGLANG_IMAGE_TAG} ${PATCH_VERSION} ${ENABLE_EFA} ${KANIKO_CACHE} ${CPU_REQUEST} ${CPU_LIMIT} ${MEMORY_REQUEST} ${MEMORY_LIMIT} ${BUILD_CONTEXT} ${NODE_SELECTOR_YAML} ${DOCKER_CONFIG_SECRET} ${FSX_PVC_NAME} ${FSX_MOUNT_PATH}' \
  < "${TEMPLATE}" > "${RENDERED}"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${RENDERED}"
  rm -f "${RENDERED}"
  exit 0
fi

echo "==> kubectl apply -n ${K8S_NAMESPACE}"
kubectl apply -n "${K8S_NAMESPACE}" -f "${RENDERED}"
rm -f "${RENDERED}"

cat <<EOF

Submitted Job/${JOB_NAME}

Follow logs:
  kubectl -n ${K8S_NAMESPACE} logs -f job/${JOB_NAME}

Wait for completion:
  kubectl -n ${K8S_NAMESPACE} wait --for=condition=complete --timeout=6h job/${JOB_NAME}

Verify ECR tag:
  aws ecr describe-images --region ${AWS_REGION} --repository-name ${ECR_REPOSITORY} \\
    --image-ids imageTag=${IMAGE_TAG}

Image URI:
  ${ECR_URI}:${IMAGE_TAG}
EOF

if [[ "${FOLLOW}" == "1" ]]; then
  # Wait briefly for pod
  sleep 3
  kubectl -n "${K8S_NAMESPACE}" logs -f "job/${JOB_NAME}" || true
fi
