"""Step-level GRPO advantage (Path A hybrid) -- tag-aware.

Wire-up::

    --custom-reward-post-process-path \\
        examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards
    --rollout-sample-filter-path \\
        examples.claudecode_ags.step_reconstruct.step_grpo_advantage.filter

* ``sample_kind="vanilla"`` -- normalized per prompt group over K trials
  (episode reward = sum of segment rewards).
* ``sample_kind="branch"`` -- normalized per ``step_group_key``
  (``{group_index}:{source_trial_idx}:edit:{edit_step_i}``).
"""

from __future__ import annotations

import logging
import math
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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    result = default if value is None or value == "" else float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative float, got {value!r}")
    return result


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
    """Episode identity for summing segment rewards / std filter.

    Hybrid siblings share one ``rollout_id`` (slime compact-rollout). Do **not**
    use that as the episode key for vanilla trials, or all K attempts collapse
    into one reward and every group looks ``std0``. Prefer ``branch_uid``, then
    vanilla ``trial_idx``, then index / session — never shared rollout_id alone.
    """
    md = sample.metadata or {}
    if md.get("branch_uid") is not None:
        return md["branch_uid"]
    if md.get("sample_kind") == "vanilla" and md.get("trial_idx") is not None:
        gi = sample.group_index if sample.group_index is not None else sample.index
        return f"v:{gi}:t{md['trial_idx']}"
    if sample.index is not None:
        return sample.index
    if sample.session_id:
        return sample.session_id
    if sample.rollout_id is not None:
        return sample.rollout_id
    return id(sample)


def _is_active(sample: Sample) -> bool:
    return not (getattr(sample, "remove_sample", False) or getattr(sample, "is_filtered_out", False))


def _active_mask_tokens(sample: Sample) -> int:
    if not _is_active(sample):
        return 0
    if sample.loss_mask is None:
        # Match rollout conversion, which materializes an all-ones mask when
        # a generator leaves it unspecified.
        return int(sample.response_length)
    return sum(int(x) for x in sample.loss_mask)


def _outer_loss_key(sample: Sample) -> Any:
    if sample.group_index is not None:
        return sample.group_index
    if sample.rollout_id is not None:
        return sample.rollout_id
    return sample.index


def _default_loss_group_id(sample: Sample) -> str:
    kind = "vanilla" if _is_vanilla(sample) else "branch"
    return f"hybrid:{_outer_loss_key(sample)}:{kind}:{_branch_key(sample)}"


def _assign_loss_weights(flat: list[Sample]) -> tuple[int, int, int]:
    """Assign the exact Hybrid objective after filtering.

    For each outer prompt, active Stage-1 episodes sum to weight 1.  Active
    Stage-2 edit groups sum to ``lambda``; groups are equal-weighted first,
    then the active branch episodes within each group are equal-weighted.
    Compact segments inherit the weight of their episode.
    """
    branch_lambda = _env_float("STEP_GRPO_BRANCH_LOSS_WEIGHT", 1.0)
    episode_members: dict[tuple[Any, str, Any], list[Sample]] = defaultdict(list)
    for s in flat:
        if s.loss_group_id is None:
            s.loss_group_id = _default_loss_group_id(s)
        kind = "vanilla" if _is_vanilla(s) else "branch"
        episode_members[(_outer_loss_key(s), kind, _branch_key(s))].append(s)
        s.loss_weight = 0.0
        s.metadata = s.metadata or {}
        s.metadata["loss_weight"] = 0.0

    active_episode: dict[tuple[Any, str, Any], bool] = {
        key: sum(_active_mask_tokens(s) for s in members) > 0 for key, members in episode_members.items()
    }
    loss_group_owners: dict[str | int, tuple[Any, str, Any]] = {}
    for key, members in episode_members.items():
        for s in members:
            owner = loss_group_owners.setdefault(s.loss_group_id, key)
            if owner != key:
                raise ValueError(f"loss_group_id {s.loss_group_id!r} is shared by episodes {owner!r} and {key!r}")

    vanilla_by_outer: dict[Any, set[tuple[Any, str, Any]]] = defaultdict(set)
    branch_by_outer_group: dict[Any, dict[Any, set[tuple[Any, str, Any]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for key, members in episode_members.items():
        if not active_episode[key]:
            continue
        sample = members[0]
        outer = key[0]
        if _is_vanilla(sample):
            vanilla_by_outer[outer].add(key)
        else:
            branch_by_outer_group[outer][_step_group_key(sample)].add(key)

    def _set_episode_weight(key: tuple[Any, str, Any], weight: float) -> None:
        expected_ids = {s.loss_group_id for s in episode_members[key]}
        if len(expected_ids) != 1:
            raise ValueError(f"compact episode has inconsistent loss_group_id values: {expected_ids}")
        for s in episode_members[key]:
            s.loss_weight = float(weight)
            s.metadata = s.metadata or {}
            s.metadata["loss_weight"] = float(weight)

    for keys in vanilla_by_outer.values():
        weight = 1.0 / len(keys)
        for key in keys:
            _set_episode_weight(key, weight)

    active_branch_groups = 0
    active_branch_episodes = 0
    for groups in branch_by_outer_group.values():
        nonempty_groups = [keys for keys in groups.values() if keys]
        if not nonempty_groups:
            continue
        active_branch_groups += len(nonempty_groups)
        for keys in nonempty_groups:
            active_branch_episodes += len(keys)
            weight = branch_lambda / (len(nonempty_groups) * len(keys))
            for key in keys:
                _set_episode_weight(key, weight)

    active_vanilla_episodes = sum(len(keys) for keys in vanilla_by_outer.values())
    return active_vanilla_episodes, active_branch_groups, active_branch_episodes


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
        ep_active_tokens: dict[Any, int] = defaultdict(int)
        for pos in entries:
            s = flat[pos]
            bk = _branch_key(s)
            ep_positions[bk].append(pos)
            ep_reward[bk] = ep_reward.get(bk, 0.0) + float(s.get_reward_value(args))
            ep_active_tokens[bk] += _active_mask_tokens(s)

        # A zero-token episode has no gradient contribution and therefore must
        # not change the mean/std used by episodes that do train. This matches
        # the loss-weight denominator assigned later by ``filter``.
        active = [bk for bk in ep_reward if ep_active_tokens[bk] > 0]
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
    """Filter degenerate groups, then assign the Hybrid loss objective."""
    del args
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
            total_mask = sum(_active_mask_tokens(s) for s in mem)
            per_ep: dict[Any, float] = defaultdict(float)
            per_ep_tokens: dict[Any, int] = defaultdict(int)
            for s in mem:
                episode_key = _branch_key(s)
                per_ep[episode_key] += float(s.reward or 0.0)
                per_ep_tokens[episode_key] += _active_mask_tokens(s)
            rewards = [reward for key, reward in per_ep.items() if per_ep_tokens[key] > 0]
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

    if _env_bool("STEP_GRPO_FILTER", True):
        vg, vs, vm = _drop_group(vanilla, _vanilla_group_key)
        bg, bs, bm = _drop_group(branch, _step_group_key)
    else:
        vg = len({_vanilla_group_key(s) for s in vanilla})
        bg = len({_step_group_key(s) for s in branch})
        vs = vm = bs = bm = 0

    active_v, active_bg, active_b = _assign_loss_weights(flat)
    logger.info(
        "[step_grpo_adv] filter: vanilla_groups=%d (std0=%d mask0=%d) "
        "branch_groups=%d (std0=%d mask0=%d) active_vanilla_episodes=%d "
        "active_branch_groups=%d active_branch_episodes=%d branch_lambda=%.6g",
        vg,
        vs,
        vm,
        bg,
        bs,
        bm,
        active_v,
        active_bg,
        active_b,
        _env_float("STEP_GRPO_BRANCH_LOSS_WEIGHT", 1.0),
    )
