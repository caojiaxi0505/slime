#!/usr/bin/env bash
# Apply / delete SWE484 eval adapter Service + ALB Ingress for one role.
# Usage: EVAL_ROLE=base|grpo ./submit_alb.sh [--delete] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/service-ingress.yaml.template"

usage() {
  cat <<'EOF'
Usage: EVAL_ROLE=base|grpo submit_alb.sh [--delete] [--dry-run] [--help]

Creates Service + internet-facing ALB Ingress for SWE484 eval Master :9002.
Requires EVAL_ROLE=base or grpo (sets JOB_NAME/WORKLOAD).
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
JOB_NAME="${JOB_NAME:-jiaxicao-swe484-eval-${EVAL_ROLE}}"
WORKLOAD="${WORKLOAD:-${JOB_NAME}}"
NAME="${JOB_NAME}-adapter"

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete ingress "${NAME}" --ignore-not-found
  kubectl -n "${K8S_NAMESPACE}" delete service "${NAME}" --ignore-not-found
  echo "Deleted ${NAME} Service/Ingress in ${K8S_NAMESPACE}"
  exit 0
fi

export JOB_NAME K8S_NAMESPACE WORKLOAD
RENDERED="$(mktemp)"
envsubst '${JOB_NAME} ${K8S_NAMESPACE} ${WORKLOAD}' < "${TEMPLATE}" > "${RENDERED}"

if [[ "${DRY_RUN}" == "1" ]]; then
  kubectl apply --dry-run=client -f "${RENDERED}"
  rm -f "${RENDERED}"
  exit 0
fi

echo "==> apply ${NAME} (workload=${WORKLOAD}) in ${K8S_NAMESPACE}"
kubectl apply -f "${RENDERED}"
rm -f "${RENDERED}"

echo "==> waiting for Ingress ADDRESS (ALB provision can take 1–3 min)..."
for _ in $(seq 1 60); do
  ADDR="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${NAME}" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
  if [[ -n "${ADDR}" ]]; then
    echo
    echo "ALB ready:"
    echo "  export SLIME_ADAPTER_PUBLIC_URL=http://${ADDR}"
    exit 0
  fi
  sleep 5
  printf '.'
done
echo
echo "WARNING: ADDRESS not ready yet. Check:"
echo "  kubectl -n ${K8S_NAMESPACE} get ingress ${NAME} -w"
exit 0
