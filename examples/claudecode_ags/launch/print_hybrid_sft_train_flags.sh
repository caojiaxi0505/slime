#!/usr/bin/env bash
# Example wiring for turn-level teacher SFT (debug / documentation).
#
# Which rows are trained (STEP_GRPO_TEACHER_SFT_MODE):
#   sft_only  → --loss-type sft_loss
#   hybrid    → --loss-type custom_loss + hybrid_teacher_sft_loss
#
# Stage-2 relabels selected turns of failed trials: rebuild the workspace at the
# turn, resume the native session, let the teacher take STEP_GRPO_TEACHER_MAX_STEPS
# real turns (tool calls execute), then SFT the student on those turns.
#
# Required remote teacher env:
#   SLIME_REMOTE_OPENAI_BASE_URL
#   SLIME_REMOTE_OPENAI_API_KEY
#   SLIME_REMOTE_OPENAI_MODEL
#   SLIME_REMOTE_OPENAI_THINKING_TYPE=enabled
#   SLIME_REMOTE_OPENAI_REASONING_EFFORT=max
#
# Optional (one relabeled turn is one sandbox; selection is the cost knob):
#   STEP_GRPO_TEACHER_MAX_STEPS=2            # 0 = run to the end and grade
#   STEP_GRPO_TEACHER_TURN_SELECT=all        # all | patch | patch_ppl
#   STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL=0   # 0 = every eligible turn
#   STEP_GRPO_TEACHER_SFT_RESOLVED_ONLY=1    # only with MAX_STEPS=0
#   STEP_GRPO_BRANCH_BUDGET_SEC
#   SLIME_TEACHER_ADAPTER_PORT=18002
#   SLIME_REMOTE_OPENAI_MAX_INFLIGHT=64      # <=0 disables
#   STEP_GRPO_TEACHER_SFT_LOSS_WEIGHT=1
#   STEP_GRPO_STAGE1_LOSS_WEIGHT=1

set -euo pipefail

MODE="${STEP_GRPO_TEACHER_SFT_MODE:-sft_only}"
SELECT="${STEP_GRPO_TEACHER_TURN_SELECT:-all}"
MAX_STEPS="${STEP_GRPO_TEACHER_MAX_STEPS:-2}"
GENERATE_PATH="examples.claudecode_ags.step_reconstruct.hybrid_sft_generate.hybrid_sft_generate"

common=(
  --custom-generate-function-path "${GENERATE_PATH}"
  --n-samples-per-prompt 1
)

if [[ "${MODE}" == "sft_only" ]]; then
  echo "loss wiring: sft_only (select=${SELECT} teacher_steps=${MAX_STEPS}) → sft_loss"
  printf '%s\n' "${common[@]}" \
    --loss-type sft_loss \
    --disable-compute-advantages-and-returns \
    --loss-mask-type qwen3_5
elif [[ "${MODE}" == "hybrid" ]]; then
  echo "loss wiring: hybrid (select=${SELECT} teacher_steps=${MAX_STEPS}) → custom_loss (GRPO vanilla + teacher SFT)"
  printf '%s\n' "${common[@]}" \
    --loss-type custom_loss \
    --custom-loss-function-path examples.claudecode_ags.step_reconstruct.hybrid_teacher_sft_loss.hybrid_teacher_sft_loss \
    --custom-reward-post-process-path examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards \
    --rollout-sample-filter-path examples.claudecode_ags.step_reconstruct.step_grpo_advantage.filter \
    --loss-mask-type qwen3_5
else
  echo "Unknown STEP_GRPO_TEACHER_SFT_MODE=${MODE}" >&2
  exit 1
fi
