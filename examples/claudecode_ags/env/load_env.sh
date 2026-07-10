#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
