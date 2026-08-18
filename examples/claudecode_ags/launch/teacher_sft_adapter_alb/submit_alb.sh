#!/usr/bin/env bash
# Apply / delete student + teacher adapter Service + ALB Ingresses.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/service-ingress.yaml.template"
K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
NAME="${NAME:-jiaxicao-fail-imitation-learning-adapter}"
TEACHER_NAME="${TEACHER_NAME:-jiaxicao-fail-imitation-learning-teacher}"
WORKLOAD_LABEL="${WORKLOAD_LABEL:-jiaxicao-fail-imitation-learning}"

usage() {
  cat <<'EOF'
Usage: submit_alb.sh [--delete] [--dry-run] [--help]

Creates Service + two internet-facing ALB Ingresses:
  NAME          Master :9002  (student SGLang adapter)
  TEACHER_NAME  Master :18002 (teacher remote-OpenAI adapter)

Namespace: sn5-system-intern (override with K8S_NAMESPACE).

After apply, wait for ADDRESS then:
  export SLIME_ADAPTER_PUBLIC_URL=http://<STUDENT_ADDRESS>
  export SLIME_TEACHER_ADAPTER_PUBLIC_URL=http://<TEACHER_ADDRESS>
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

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: missing ${TEMPLATE}" >&2
  exit 1
fi

export NAME TEACHER_NAME K8S_NAMESPACE WORKLOAD_LABEL
RENDERED="$(mktemp)"
envsubst '${NAME} ${TEACHER_NAME} ${K8S_NAMESPACE} ${WORKLOAD_LABEL}' < "${TEMPLATE}" > "${RENDERED}"
YAML="${RENDERED}"

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete ingress "${NAME}" "${TEACHER_NAME}" --ignore-not-found
  kubectl -n "${K8S_NAMESPACE}" delete service "${NAME}" --ignore-not-found
  echo "Deleted ${NAME} / ${TEACHER_NAME} Service/Ingress in ${K8S_NAMESPACE}"
  rm -f "${RENDERED}"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" apply --dry-run=client -f "${YAML}"
  rm -f "${RENDERED}"
  exit 0
fi

echo "==> apply ${NAME} + ${TEACHER_NAME} in ${K8S_NAMESPACE}"
kubectl apply -f "${YAML}"
rm -f "${RENDERED}"

_wait_addr() {
  local ingress_name="$1"
  local addr=""
  for _ in $(seq 1 60); do
    addr="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${ingress_name}" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
    if [[ -n "${addr}" ]]; then
      echo "${addr}"
      return 0
    fi
    sleep 5
    printf '.' >&2
  done
  return 1
}

echo "==> waiting for Ingress ADDRESS (ALB provision can take 1–3 min)..."
STUDENT_ADDR="$(_wait_addr "${NAME}" || true)"
TEACHER_ADDR="$(_wait_addr "${TEACHER_NAME}" || true)"
echo
if [[ -n "${STUDENT_ADDR}" && -n "${TEACHER_ADDR}" ]]; then
  echo "ALB ready:"
  echo "  export SLIME_ADAPTER_PUBLIC_URL=http://${STUDENT_ADDR}"
  echo "  export SLIME_TEACHER_ADAPTER_PUBLIC_URL=http://${TEACHER_ADDR}"
  echo
  echo "Notes:"
  echo "  - ALBs listen on :80 (no :9002 / :18002 in the URL)."
  echo "  - Master pod must have labels app=cc-ags-recorder,workload=${WORKLOAD_LABEL}"
  echo "  - Student adapter binds :9002; teacher adapter binds 0.0.0.0:18002"
  echo "  - /health will be UNHEALTHY until train adapters are up — expected"
  exit 0
fi
echo "WARNING: ADDRESS not ready yet. Check:"
echo "  kubectl -n ${K8S_NAMESPACE} get ingress ${NAME} ${TEACHER_NAME} -w"
exit 0
