"""Path A hybrid step-GRPO Stage-1/2 building blocks (no slime-tencent deps)."""

from __future__ import annotations

__all__ = [
    "hybrid_generate",
    "post_process_rewards",
    "filter",
]


def __getattr__(name: str):
    """Keep lightweight capture/rebuild helpers importable without torch/ray."""
    if name == "hybrid_generate":
        from examples.claudecode_ags.step_reconstruct.hybrid_generate import hybrid_generate

        return hybrid_generate
    if name in {"post_process_rewards", "filter"}:
        from examples.claudecode_ags.step_reconstruct import step_grpo_advantage

        return getattr(step_grpo_advantage, name)
    raise AttributeError(name)
