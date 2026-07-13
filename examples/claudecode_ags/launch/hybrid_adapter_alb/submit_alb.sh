#!/usr/bin/env bash
# Apply / delete the Path A hybrid 1-node debug adapter Service + ALB Ingress.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="${SCRIPT_DIR}/service-ingress.yaml"
K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
NAME="${NAME:-jiaxicao-hybrid-1node-debug-adapter}"

usage() {
  cat <<'EOF'
Usage: submit_alb.sh [--delete] [--dry-run] [--help]

Creates Service + internet-facing ALB Ingress for hybrid Master :9002.
Namespace: sn5-system-intern (override with K8S_NAMESPACE).

After apply, wait for ADDRESS then:
  export SLIME_ADAPTER_PUBLIC_URL=http://<ADDRESS>
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

if [[ ! -f "${YAML}" ]]; then
  echo "ERROR: missing ${YAML}" >&2
  exit 1
fi

if [[ "${DELETE}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" delete ingress "${NAME}" --ignore-not-found
  kubectl -n "${K8S_NAMESPACE}" delete service "${NAME}" --ignore-not-found
  echo "Deleted ${NAME} Service/Ingress in ${K8S_NAMESPACE}"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  kubectl -n "${K8S_NAMESPACE}" apply --dry-run=client -f "${YAML}"
  exit 0
fi

echo "==> apply ${NAME} in ${K8S_NAMESPACE}"
kubectl apply -f "${YAML}"

echo "==> waiting for Ingress ADDRESS (ALB provision can take 1–3 min)..."
for _ in $(seq 1 60); do
  ADDR="$(kubectl -n "${K8S_NAMESPACE}" get ingress "${NAME}" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
  if [[ -n "${ADDR}" ]]; then
    echo
    echo "ALB ready:"
    echo "  export SLIME_ADAPTER_PUBLIC_URL=http://${ADDR}"
    echo
    echo "Notes:"
    echo "  - ALB listens on :80 (no :9002 in the URL)."
    echo "  - Master pod must have labels app=cc-ags-recorder,workload=jiaxicao-hybrid-1node-debug"
    echo "  - Adapter must bind SLIME_ADAPTER_PORT=9002 (Path A) / SHIM_PORT=9002"
    echo "  - /health will be UNHEALTHY until train adapter is up — expected"
    exit 0
  fi
  sleep 5
  printf '.'
done
echo
echo "WARNING: ADDRESS not ready yet. Check:"
echo "  kubectl -n ${K8S_NAMESPACE} get ingress ${NAME} -w"
exit 0
