"""Step-level GRPO advantage (Path A hybrid) -- tag-aware.

Wire-up::

    --custom-reward-post-process-path \\
        examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards
    --rollout-sample-filter-path \\
        examples.claudecode_ags.step_reconstruct.step_grpo_advantage.filter

* ``sample_kind="vanilla"`` -- normalized per prompt group over K trials
  (episode reward = sum of segment rewards).
* ``sample_kind="branch"`` -- normalized per ``step_group_key``
  (``{group_index}:{source_trial_idx}:{step_t}``).
"""

from __future__ import annotations

import logging
import os
import statistics
from collections import defaultdict
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["post_process_rewards", "filter"]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _iter_leaves(node: Any):
    if isinstance(node, Sample):
        yield node
    elif isinstance(node, (list, tuple)):
        for child in node:
            yield from _iter_leaves(child)


def _flatten(samples: Any) -> list[Sample]:
    return list(_iter_leaves(samples))


def _is_vanilla(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("sample_kind") == "vanilla")


def _vanilla_group_key(sample: Sample) -> Any:
    gi = sample.group_index if sample.group_index is not None else sample.index
    return f"v:{gi}"


def _step_group_key(sample: Sample) -> Any:
    md = sample.metadata or {}
    key = md.get("step_group_key")
    if key is not None:
        return key
    if md.get("step_t") is not None:
        gi = sample.group_index if sample.group_index is not None else sample.index
        src = md.get("source_trial_idx", "x")
        return f"{gi}:{src}:{md.get('step_t')}"
    if sample.group_index is not None:
        return f"g{sample.group_index}"
    return f"i{sample.index}"


def _branch_key(sample: Sample) -> Any:
    md = sample.metadata or {}
    if md.get("branch_uid") is not None:
        return md["branch_uid"]
    if sample.rollout_id is not None:
        return sample.rollout_id
    if sample.index is not None:
        return sample.index
    if sample.session_id:
        return sample.session_id
    return id(sample)


def _is_active(sample: Sample) -> bool:
    return not (getattr(sample, "remove_sample", False) or getattr(sample, "is_filtered_out", False))


def _normalize_positions(
    flat: list[Sample], positions: list[int], args: Any, use_std: bool, outer_key_fn
) -> dict[int, float]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for pos in positions:
        groups[outer_key_fn(flat[pos])].append(pos)

    out: dict[int, float] = {}
    for entries in groups.values():
        ep_reward: dict[Any, float] = {}
        ep_positions: dict[Any, list[int]] = defaultdict(list)
        ep_active: dict[Any, bool] = defaultdict(bool)
        for pos in entries:
            s = flat[pos]
            bk = _branch_key(s)
            ep_positions[bk].append(pos)
            ep_reward[bk] = ep_reward.get(bk, 0.0) + float(s.get_reward_value(args))
            ep_active[bk] = ep_active[bk] or _is_active(s)

        active = [bk for bk in ep_reward if ep_active[bk]]
        rewards = [ep_reward[bk] for bk in active]
        adv_map: dict[Any, float] = {}
        if rewards:
            mean = statistics.fmean(rewards)
            if use_std and len(rewards) > 1:
                std = statistics.pstdev(rewards)
                for bk in active:
                    adv_map[bk] = (ep_reward[bk] - mean) / (std + 1e-6)
            else:
                for bk in active:
                    adv_map[bk] = ep_reward[bk] - mean

        for bk, pos_list in ep_positions.items():
            adv = adv_map.get(bk, 0.0)
            for pos in pos_list:
                out[pos] = adv
    return out


def post_process_rewards(
    args: Any, samples: list[Sample] | list[list[Sample]]
) -> tuple[list[float], list[float]]:
    flat = _flatten(samples)
    raw_rewards = [float(s.get_reward_value(args)) for s in flat]
    if not flat:
        return raw_rewards, raw_rewards

    estimator = getattr(args, "advantage_estimator", "grpo")
    use_std = _env_bool("STEP_GRPO_STD_NORMALIZATION", True) and estimator in ("grpo", "gspo")

    vanilla_idx = [i for i, s in enumerate(flat) if _is_vanilla(s)]
    branch_idx = [i for i in range(len(flat)) if i not in set(vanilla_idx)]

    adv: dict[int, float] = {}
    n_vg = n_bg = 0
    if vanilla_idx:
        adv.update(_normalize_positions(flat, vanilla_idx, args, use_std, _vanilla_group_key))
        n_vg = len({_vanilla_group_key(flat[i]) for i in vanilla_idx})
    if branch_idx:
        adv.update(_normalize_positions(flat, branch_idx, args, use_std, _step_group_key))
        n_bg = len({_step_group_key(flat[i]) for i in branch_idx})

    advantages = [adv.get(i, 0.0) for i in range(len(flat))]
    logger.info(
        "[step_grpo_adv] rows=%d vanilla=%d branch=%d vanilla_groups=%d step_groups=%d "
        "estimator=%s use_std=%s",
        len(flat),
        len(vanilla_idx),
        len(branch_idx),
        n_vg,
        n_bg,
        estimator,
        use_std,
    )
    return raw_rewards, advantages


def filter(args: Any, data: list[Any]) -> None:
    """Mark ``remove_sample`` on degenerate groups when ``STEP_GRPO_FILTER=1``."""
    del args
    if not _env_bool("STEP_GRPO_FILTER", True):
        return

    flat = _flatten(data)
    if not flat:
        return

    vanilla = [s for s in flat if _is_vanilla(s)]
    branch = [s for s in flat if not _is_vanilla(s)]

    def _drop_group(members: list[Sample], key_fn) -> tuple[int, int, int]:
        groups: dict[Any, list[Sample]] = defaultdict(list)
        for s in members:
            groups[key_fn(s)].append(s)
        n_std = 0
        n_mask = 0
        for mem in groups.values():
            total_mask = sum(sum(int(x) for x in (s.loss_mask or [])) for s in mem)
            per_ep: dict[Any, float] = defaultdict(float)
            for s in mem:
                per_ep[_branch_key(s)] += float(s.reward or 0.0)
            rewards = list(per_ep.values())
            std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            reason = ""
            if total_mask == 0:
                reason = "all_mask_zero"
                n_mask += 1
            elif std <= 0.0:
                reason = "std_zero"
                n_std += 1
            if reason:
                for s in mem:
                    s.is_filtered_out = True
                    s.remove_sample = True
                    if s.metadata is None:
                        s.metadata = {}
                    s.metadata.setdefault("rollout_filter_drop_reason", reason)
        return len(groups), n_std, n_mask

    vg, vs, vm = _drop_group(vanilla, _vanilla_group_key)
    bg, bs, bm = _drop_group(branch, _step_group_key)
    logger.info(
        "[step_grpo_adv] filter: vanilla_groups=%d (std0=%d mask0=%d) branch_groups=%d (std0=%d mask0=%d)",
        vg,
        vs,
        vm,
        bg,
        bs,
        bm,
    )
