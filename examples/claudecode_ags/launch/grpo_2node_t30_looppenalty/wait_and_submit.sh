#!/usr/bin/env bash
# Submit the loop-penalty experiment only after the current naive GRPO succeeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-jiaxicao-grpo-2node-c64t45}"
NEW_JOB_NAME="${JOB_NAME:-jiaxicao-grpo-2node-t30-loop3}"
POLL_SEC="${POLL_SEC:-300}"

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

echo "$(timestamp) waiting for pytorchjob/${WAIT_FOR_JOB} to succeed"
while true; do
  if ! kubectl -n "${K8S_NAMESPACE}" get pytorchjob "${WAIT_FOR_JOB}" >/dev/null 2>&1; then
    echo "$(timestamp) waiting job is missing; refusing to submit automatically" >&2
    exit 2
  fi

  succeeded="$(kubectl -n "${K8S_NAMESPACE}" get pytorchjob "${WAIT_FOR_JOB}" \
    -o jsonpath='{range .status.conditions[?(@.type=="Succeeded")]}{.status}{end}')"
  failed="$(kubectl -n "${K8S_NAMESPACE}" get pytorchjob "${WAIT_FOR_JOB}" \
    -o jsonpath='{range .status.conditions[?(@.type=="Failed")]}{.status}{end}')"

  if [[ "${succeeded}" == "True" ]]; then
    echo "$(timestamp) ${WAIT_FOR_JOB} succeeded"
    break
  fi
  if [[ "${failed}" == "True" ]]; then
    echo "$(timestamp) ${WAIT_FOR_JOB} failed; new experiment was not submitted" >&2
    exit 3
  fi

  echo "$(timestamp) ${WAIT_FOR_JOB} still running"
  sleep "${POLL_SEC}"
done

if kubectl -n "${K8S_NAMESPACE}" get pytorchjob "${NEW_JOB_NAME}" >/dev/null 2>&1; then
  echo "$(timestamp) pytorchjob/${NEW_JOB_NAME} already exists; refusing duplicate submission" >&2
  exit 4
fi

echo "$(timestamp) provisioning adapter ingress for ${NEW_JOB_NAME}"
NAME="${NEW_JOB_NAME}-adapter" \
WORKLOAD_LABEL="${NEW_JOB_NAME}" \
K8S_NAMESPACE="${K8S_NAMESPACE}" \
  bash "${SLIME_DIR}/examples/claudecode_ags/launch/hybrid_adapter_alb/submit_alb.sh"

echo "$(timestamp) submitting ${NEW_JOB_NAME}"
JOB_NAME="${NEW_JOB_NAME}" \
K8S_NAMESPACE="${K8S_NAMESPACE}" \
  bash "${SCRIPT_DIR}/submit_job.sh"
echo "$(timestamp) submitted pytorchjob/${NEW_JOB_NAME}"
