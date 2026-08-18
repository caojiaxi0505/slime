"""Path A hybrid step-GRPO Stage-1/2 building blocks (no slime-tencent deps)."""

from __future__ import annotations

__all__ = [
    "hybrid_generate",
    "hybrid_sft_generate",
    "hybrid_teacher_sft_loss",
    "teacher_turn_samples",
    "post_process_rewards",
    "filter",
]


def __getattr__(name: str):
    """Keep lightweight capture/rebuild helpers importable without torch/ray."""
    if name == "hybrid_generate":
        from examples.claudecode_ags.step_reconstruct.hybrid_generate import hybrid_generate

        return hybrid_generate
    if name == "hybrid_sft_generate":
        from examples.claudecode_ags.step_reconstruct.hybrid_sft_generate import hybrid_sft_generate

        return hybrid_sft_generate
    if name == "teacher_turn_samples":
        from examples.claudecode_ags.step_reconstruct.teacher_turn_sft import teacher_turn_samples

        return teacher_turn_samples
    if name == "hybrid_teacher_sft_loss":
        from examples.claudecode_ags.step_reconstruct.hybrid_teacher_sft_loss import hybrid_teacher_sft_loss

        return hybrid_teacher_sft_loss
    if name in {"post_process_rewards", "filter"}:
        from examples.claudecode_ags.step_reconstruct import step_grpo_advantage

        return getattr(step_grpo_advantage, name)
    raise AttributeError(name)
