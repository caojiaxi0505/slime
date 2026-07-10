"""GRPO/GSPO reward post-process that is safe under segment fan-out."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from slime.utils.types import Sample


def _reward_value(s: Sample, args: Any) -> float:
    if not hasattr(args, "reward_key"):
        return float(s.reward)
    return float(s.get_reward_value(args))


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    flat: list[Sample] = []
    for item in samples:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    raw = [float(_reward_value(s, args)) for s in flat]
    if not flat:
        return raw, raw

    by_group: dict[Any, list[tuple[int, Sample]]] = defaultdict(list)
    for i, s in enumerate(flat):
        gkey = s.group_index if s.group_index is not None else s.index
        by_group[gkey].append((i, s))

    advantages = [0.0] * len(flat)
    use_std = bool(getattr(args, "grpo_std_normalization", True)) and getattr(args, "advantage_estimator", "grpo") in (
        "grpo",
        "gspo",
    )

    for entries in by_group.values():
        episode_reward: dict[Any, float] = {}
        positions: dict[Any, list[int]] = defaultdict(list)
        for pos, s in entries:
            rkey = s.index if s.index is not None else id(s)
            positions[rkey].append(pos)
            episode_reward[rkey] = episode_reward.get(rkey, 0.0) + _reward_value(s, args)

        keys = list(episode_reward.keys())
        tensor = torch.tensor([episode_reward[k] for k in keys], dtype=torch.float)
        centered = tensor - tensor.mean()
        if use_std and tensor.numel() > 1:
            centered = centered / (centered.std(unbiased=False) + 1e-6)
        adv_map = {k: float(centered[i].item()) for i, k in enumerate(keys)}
        for rkey, pos_list in positions.items():
            a = adv_map[rkey]
            for pos in pos_list:
                advantages[pos] = a

    return raw, advantages
