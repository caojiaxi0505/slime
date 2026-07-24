#!/usr/bin/env bash
set -uo pipefail

JOB_NAME="jiaxicao-hybrid-s0-fullcont-s43-swe"
NS="sn5-system-intern"
RUN_DIR="/mnt/sn-007/jiaxicao/checkpoints/cc-ags/qwen35_9b_cc_ags_hybrid_s0_fullcont_s43_swe_8gpu_1step_v1"
MONITOR_DIR="/mnt/sn-007/jiaxicao/analysis/monitors/${JOB_NAME}"
LOG_FILE="${MONITOR_DIR}/monitor_progress.log"

mkdir -p "${MONITOR_DIR}"

while true; do
  {
    echo "================================================================================"
    date -u "+%Y-%m-%d %H:%M:%S UTC"
    echo
    echo "[pods]"
    timeout 45s kubectl get pods -n "${NS}" | grep -E "${JOB_NAME}|jiaxicao-hybrid-s0-firstturn-s43" || true
    echo
    echo "[pytorchjob]"
    timeout 45s kubectl -n "${NS}" get pytorchjob "${JOB_NAME}" || true
    echo
    echo "[artifacts]"
    find "${RUN_DIR}" -maxdepth 3 -type f \( -name 'rollout_*.pt' -o -name 'latest_checkpointed_iteration.txt' -o -name 'run.log' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort || true
    echo
    echo "[latest rollout/train markers]"
    if [[ -f "${RUN_DIR}/run.log" ]]; then
      grep -E 'Save debug rollout|rollout [0-9]+:|Timer .* (start|end)|eval 0:|grad_norm|actor train|ref_log_probs|log_probs|Finished|Succeeded|Traceback|ERROR|RuntimeError|rollout failed|agent_exit|stage1|stage2|hybrid-live' "${RUN_DIR}/run.log" | tail -n 80 || true
    else
      echo "run.log not found yet"
    fi
    echo
  } >> "${LOG_FILE}" 2>&1

  sleep 1800
done
