"""Custom loss: Stage-1 GRPO (vanilla) + Stage-2 teacher continuation SFT."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from slime.backends.megatron_utils.loss import get_sum_of_sample_mean, policy_loss_function, sft_loss_function
from slime.utils.types import RolloutBatch


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _stage_weights(
    stage_ids: list[Any],
    base_weights: list[float],
    *,
    keep: set[int],
) -> list[float]:
    return [
        float(weight) if int(stage) in keep else 0.0 for weight, stage in zip(base_weights, stage_ids, strict=True)
    ]


def hybrid_teacher_sft_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mix vanilla policy loss with teacher_sft NLL via loss_stage_ids.

    Stage ids (see ``build_loss_group_fields``):
      0 = vanilla (GRPO / policy)
      2 = teacher_sft (SFT)
    """
    stage_ids = batch.get("loss_stage_ids")
    if stage_ids is None:
        # No explicit hybrid objective markers → pure SFT.
        return sft_loss_function(args, batch, logits, sum_of_sample_mean)

    n = len(stage_ids)
    base_weights = batch.get("loss_weights")
    if base_weights is None:
        base_weights = [1.0] * n

    sample_denoms = batch.get("loss_group_mask_sums")
    if sample_denoms is None:
        sample_denoms = batch["rollout_mask_sums"]

    pg_weights = _stage_weights(stage_ids, base_weights, keep={0})
    sft_weights = _stage_weights(stage_ids, base_weights, keep={2})

    pg_reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        sample_denoms,
        args.calculate_per_token_loss,
        pg_weights,
    )
    sft_reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        sample_denoms,
        args.calculate_per_token_loss,
        sft_weights,
    )

    # Zero advantages on non-vanilla rows so a mis-weighted reducer cannot
    # push a teacher suffix through the PPO path.
    pg_batch = dict(batch)
    pg_batch["advantages"] = [
        (adv * 0 if int(stage) != 0 else adv) for adv, stage in zip(batch["advantages"], stage_ids, strict=True)
    ]

    has_vanilla = any(w > 0 for w in pg_weights)
    has_teacher = any(w > 0 for w in sft_weights)

    zero = logits.sum() * 0
    if has_vanilla:
        pg_loss, pg_metrics = policy_loss_function(args, pg_batch, logits, pg_reducer)
    else:
        pg_loss = zero
        pg_metrics = {"loss": zero.detach()}

    if has_teacher:
        sft_loss, sft_metrics = sft_loss_function(args, batch, logits, sft_reducer)
    else:
        sft_loss = zero
        sft_metrics = {"loss": zero.detach()}

    # STEP_GRPO_STAGE1_LOSS_WEIGHT is already baked into the vanilla loss_weights
    # by step_grpo_advantage._assign_loss_weights; teacher rows are only
    # normalized there (equal split to 1 per prompt), so their coefficient is
    # applied here. Applying either one twice would silently square it.
    sft_coef = _env_float("STEP_GRPO_TEACHER_SFT_LOSS_WEIGHT", 1.0)
    loss = pg_loss + sft_coef * sft_loss

    metrics: dict[str, torch.Tensor] = {
        "loss": loss.clone().detach(),
        "pg_loss_vanilla": pg_loss.detach() if torch.is_tensor(pg_loss) else pg_metrics["loss"],
        "teacher_sft_loss": sft_loss.detach() if torch.is_tensor(sft_loss) else sft_metrics["loss"],
    }
    for key, value in pg_metrics.items():
        if key == "loss":
            continue
        metrics[f"vanilla/{key}"] = value
    return loss, metrics
