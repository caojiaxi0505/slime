#!/usr/bin/env bash
# Load Claude Code + Slime/AGS env for Path A.
# Job / operator exports win over slime_ags.env(.example) defaults.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Critical: example file must not overwrite cluster-injected knobs
# (adapter port 9002 vs example 18001 → ALB 502; timeout 45m vs 30m → mid-rollout 404).
_PRESERVE_KEYS=(
  SLIME_ADAPTER_PUBLIC_URL
  SLIME_ADAPTER_PORT
  SLIME_ADAPTER_BIND_HOST
  SHIM_PORT
  ANTHROPIC_BASE_URL
  WANDB_API_KEY
  WANDB_PROJECT
  WANDB_GROUP
  WANDB_TEAM
  PROMPT_DATA
  EVAL_DATA
  EXP_TAG
  LOG_DIR
  PHASE
  NUM_ROLLOUT
  SAVE_INTERVAL
  STEP_GRPO_HYBRID_K
  STEP_GRPO_FILTER
  STEP_GRPO_BRANCH_LOSS_WEIGHT
  STEP_GRPO_BUNDLE_DIR
  SLIME_AGENT_AGS_SECRET_ID
  SLIME_AGENT_AGS_SECRET_KEY
  SLIME_AGENT_AGS_TOOL_ID
  SLIME_AGENT_AGS_MOUNT_NAME
  SLIME_AGENT_AGS_MOUNT_IMAGE
  SLIME_AGENT_AGS_MOUNT_IMAGE_REGISTRY_TYPE
  SLIME_AGENT_AGS_MOUNT_PATH
  SLIME_AGENT_AGS_IMAGE_SUBPATH
  SLIME_AGENT_AGS_REGION
  SLIME_AGENT_AGS_DOMAIN
  SLIME_AGENT_AGS_ROLE_ARN
  SLIME_AGENT_AGS_HTTP_ENDPOINT
  SLIME_AGENT_AGS_TIMEOUT
  SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC
  SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC
  SLIME_AGENT_AGS_CPU
  SLIME_AGENT_AGS_MEMORY
  SLIME_AGENT_AGS_PORT
  SLIME_AGENT_TOOLCHAIN_MODE
  SLIME_AGENT_COS_MOUNT
  SLIME_AGENT_COS_NODE_PACKAGE
  SLIME_AGENT_COS_CC_PACKAGE
  SLIME_CC_TIME_BUDGET_SEC
  SLIME_CC_EVAL_TIMEOUT_SEC
  SLIME_CC_GENERATE_GUARD_SEC
  SLIME_CC_AGENT_CONCURRENCY
  SLIME_CC_EVAL_CONCURRENCY
  SLIME_CC_AGENT_PROMPT
  SLIME_CC_INITIAL_INPUT_MODE
  SLIME_CC_EXTRA_ARGS_JSON
  CLAUDE_CODE_AUTO_COMPACT_WINDOW
  CLAUDE_AUTOCOMPACT_PCT_OVERRIDE
  CLAUDE_CODE_MAX_OUTPUT_TOKENS
  CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS
  CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING
  CLAUDE_CODE_SKIP_PROMPT_HISTORY
  CLAUDE_CODE_DISABLE_TERMINAL_TITLE
  BASH_MAX_OUTPUT_LENGTH
  TASK_MAX_OUTPUT_LENGTH
  MAX_MCP_OUTPUT_TOKENS
  MAX_THINKING_TOKENS
)
_PRESERVED=()
for _k in "${_PRESERVE_KEYS[@]}"; do
  if [[ -n "${!_k:-}" ]]; then
    _PRESERVED+=("${_k}=${!_k}")
  fi
done

set -a
# shellcheck disable=SC1091
source "${DIR}/claude_code.env"
if [[ -f "${DIR}/slime_ags.env" ]]; then
  # shellcheck disable=SC1091
  source "${DIR}/slime_ags.env"
elif [[ -f "${DIR}/slime_ags.env.example" ]]; then
  echo "WARNING: using slime_ags.env.example; copy to slime_ags.env for real runs" >&2
  # shellcheck disable=SC1091
  source "${DIR}/slime_ags.env.example"
fi
set +a

for _kv in "${_PRESERVED[@]:-}"; do
  export "${_kv?}"
done

# Drop placeholder / missing local paths so Job-injected secrets can be used.
if [[ -n "${SLIME_AGENT_AGS_ENV_FILE:-}" && ! -f "${SLIME_AGENT_AGS_ENV_FILE}" ]]; then
  echo "WARNING: SLIME_AGENT_AGS_ENV_FILE missing (${SLIME_AGENT_AGS_ENV_FILE}); unsetting" >&2
  unset SLIME_AGENT_AGS_ENV_FILE
fi
if [[ -n "${SLIME_AGENT_AGS_SWE_REX_ROOT:-}" && ! -d "${SLIME_AGENT_AGS_SWE_REX_ROOT}" ]]; then
  echo "WARNING: SLIME_AGENT_AGS_SWE_REX_ROOT missing (${SLIME_AGENT_AGS_SWE_REX_ROOT}); unsetting" >&2
  unset SLIME_AGENT_AGS_SWE_REX_ROOT
fi
if [[ "${SLIME_AGENT_TOOLCHAIN_MODE:-}" == "tarball" ]]; then
  if [[ ! -f "${SLIME_AGENT_NODE_TARBALL:-}" || ! -f "${SLIME_AGENT_CC_TARBALL:-}" ]]; then
    echo "WARNING: tarball toolchain paths missing; falling back to SLIME_AGENT_TOOLCHAIN_MODE=cos" >&2
    export SLIME_AGENT_TOOLCHAIN_MODE=cos
    unset SLIME_AGENT_NODE_TARBALL SLIME_AGENT_CC_TARBALL || true
  fi
fi

unset _k _kv _PRESERVE_KEYS _PRESERVED
