"""Pure helpers for separating rollout scheduling from loss aggregation."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any


def build_loss_group_fields(
    samples: Sequence[Any],
    rollout_ids: Sequence[Hashable],
    loss_masks: Sequence[Sequence[int]],
) -> dict[str, list[Any]]:
    """Build whole-step episode denominators and explicit coefficients.

    ``rollout_ids`` remain the scheduler's fixed-GBS units. A sample without
    an explicit ``loss_group_id`` falls back to that rollout id, reproducing
    the legacy compact-rollout objective exactly.
    """
    if not (len(samples) == len(rollout_ids) == len(loss_masks)):
        raise ValueError(
            f"loss grouping length mismatch: samples={len(samples)} "
            f"rollout_ids={len(rollout_ids)} loss_masks={len(loss_masks)}"
        )

    loss_group_ids: list[Hashable] = []
    loss_weights: list[float] = []
    group_rollout_ids: dict[Hashable, Hashable] = {}
    group_weights: dict[Hashable, float] = {}
    group_mask_sums: dict[Hashable, int] = {}
    explicit_objective = any(
        getattr(sample, "loss_group_id", None) is not None
        or getattr(sample, "loss_weight", None) is not None
        for sample in samples
    )
    loss_stage_ids: list[int] = []

    for sample, rollout_id, loss_mask in zip(samples, rollout_ids, loss_masks, strict=True):
        loss_group_id = getattr(sample, "loss_group_id", None)
        if loss_group_id is None:
            loss_group_id = rollout_id
        try:
            hash(loss_group_id)
        except TypeError as exc:
            raise ValueError(f"loss_group_id must be hashable, got {loss_group_id!r}") from exc

        raw_weight = getattr(sample, "loss_weight", None)
        loss_weight = 1.0 if raw_weight is None else float(raw_weight)
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError(f"loss_weight must be finite and non-negative, got {loss_weight!r}")

        previous_rollout_id = group_rollout_ids.setdefault(loss_group_id, rollout_id)
        if previous_rollout_id != rollout_id:
            raise ValueError(
                f"loss_group_id {loss_group_id!r} spans rollout_ids "
                f"{previous_rollout_id!r} and {rollout_id!r}"
            )
        previous_weight = group_weights.setdefault(loss_group_id, loss_weight)
        if not math.isclose(previous_weight, loss_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"loss_group_id {loss_group_id!r} has inconsistent weights "
                f"{previous_weight!r} and {loss_weight!r}"
            )

        loss_group_ids.append(loss_group_id)
        loss_weights.append(loss_weight)
        metadata = getattr(sample, "metadata", None) or {}
        loss_stage_ids.append(1 if metadata.get("sample_kind") == "branch" else 0)
        group_mask_sums[loss_group_id] = group_mask_sums.get(loss_group_id, 0) + sum(int(x) for x in loss_mask)

    output = {
        "loss_group_ids": loss_group_ids,
        "loss_group_mask_sums": [group_mask_sums[group_id] for group_id in loss_group_ids],
        "loss_weights": loss_weights,
    }
    if explicit_objective:
        # Only explicit objectives need stage-aware train metrics. Omitting
        # this field on legacy rollouts keeps their batch/log schema unchanged.
        output["loss_stage_ids"] = loss_stage_ids
    return output
