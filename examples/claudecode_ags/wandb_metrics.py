"""Custom rollout wandb logger: F2P/P2P reward sources + per-sample timing.

Wired via::

    --custom-rollout-log-function-path examples.claudecode_ags.wandb_metrics.log_rollout_data

Returns True so slime skips the default logger (we emit defaults + extras once).

Metric口径 (hybrid / step-GRPO)::

* Top-level ``outcome/*``, ``rollout/*``, ``traj/*``, ``perf/agent_time`` …
  = Stage-1 only (same keys as naive GRPO → comparable on wandb).
* ``*/stage-2/*`` = Stage-2 branch extras.
* ``perf/step_grpo/*`` = hybrid-only totals (full ``rollout_time``, stage walls, sample counts).
* Train wall stays framework ``perf/actor_train_time`` on ``train/step`` (same key for both).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _meta(sample) -> dict[str, Any]:
    md = getattr(sample, "metadata", None) or {}
    return md if isinstance(md, dict) else {}


def _sample_kind(sample) -> str:
    """``vanilla`` / ``branch``; missing kind (naive GRPO) counts as stage-1."""
    kind = str(_meta(sample).get("sample_kind") or "").strip().lower()
    if kind == "branch":
        return "branch"
    return "vanilla"


def _episode_key(sample) -> tuple[Any, ...]:
    md = _meta(sample)
    return (
        md.get("instance_id"),
        getattr(sample, "group_index", None),
        getattr(sample, "index", None),
        getattr(sample, "rollout_id", None),
        _sample_kind(sample),
    )


def _episode_samples(samples: list) -> list:
    """One row per CC attempt (fan-out segments share the same F2P/P2P/timing)."""
    seen: set[tuple[Any, ...]] = set()
    out = []
    for s in samples:
        md = _meta(s)
        seg = md.get("segment_idx")
        if seg not in (None, 0):
            continue
        key = _episode_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out or list(samples)


def _split_stage_episodes(samples: list) -> tuple[list, list]:
    """Stage-1 (vanilla / unlabeled) vs Stage-2 (branch) episode reps."""
    eps = _episode_samples(samples)
    stage1 = [s for s in eps if _sample_kind(s) != "branch"]
    stage2 = [s for s in eps if _sample_kind(s) == "branch"]
    return stage1, stage2


def _md_int(sample, *keys: str) -> int:
    md = _meta(sample)
    details = md.get("reward_details") if isinstance(md.get("reward_details"), dict) else {}
    for key in keys:
        for src in (md, details):
            if key in src and src.get(key) is not None:
                try:
                    return int(src.get(key))
                except (TypeError, ValueError):
                    pass
    return 0


def _group_by_episode(samples: list) -> dict[tuple[Any, ...], list]:
    groups: dict[tuple[Any, ...], list] = defaultdict(list)
    for s in samples:
        groups[_episode_key(s)].append(s)
    return groups


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _series(samples: list, *keys: str) -> list[float]:
    vals: list[float] = []
    for s in samples:
        md = _meta(s)
        details = md.get("reward_details") if isinstance(md.get("reward_details"), dict) else {}
        for key in keys:
            if key in md:
                v = _safe_float(md.get(key))
            elif key in details:
                v = _safe_float(details.get(key))
            else:
                continue
            if v is not None:
                vals.append(v)
                break
    return vals


def _status_name(sample) -> str:
    status = getattr(sample, "status", None)
    if status is None:
        return ""
    return str(getattr(status, "name", status)).lower()


def _response_length(sample) -> float:
    for attr in ("effective_response_length", "response_length"):
        v = getattr(sample, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    tokens = getattr(sample, "tokens", None)
    if tokens is not None:
        try:
            return float(len(tokens))
        except TypeError:
            pass
    return 0.0


def _prompt_length(sample) -> float:
    tokens = getattr(sample, "tokens", None)
    resp = getattr(sample, "response_length", None)
    if tokens is not None and resp is not None:
        try:
            return float(max(len(tokens) - int(resp), 0))
        except (TypeError, ValueError):
            pass
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, list):
        return float(len(prompt))
    if prompt is not None:
        try:
            return float(len(prompt))
        except TypeError:
            pass
    return 0.0


def _total_length(sample) -> float:
    tokens = getattr(sample, "tokens", None)
    if tokens is not None:
        try:
            return float(len(tokens))
        except TypeError:
            pass
    return _prompt_length(sample) + _response_length(sample)


def _reward_source_metrics_for_episodes(eps: list, *, prefix: str) -> dict[str, float]:
    """Outcome / F2P-P2P metrics for one stage; ``prefix`` e.g. ``outcome`` or ``outcome/stage-2``."""
    from slime.utils.metric_utils import compute_statistics

    out: dict[str, float] = {}
    if not eps:
        out[f"{prefix}/n_episodes"] = 0.0
        out[f"{prefix}/resolved_rate"] = 0.0
        return out

    n = float(len(eps))
    solved = [1.0 if _meta(s).get("grading_solved") else 0.0 for s in eps]
    out[f"{prefix}/resolved_rate"] = sum(solved) / n
    out[f"{prefix}/n_episodes"] = n

    f2p = _series(eps, "test_f2p_ratio")
    p2p = _series(eps, "test_p2p_ratio")
    if f2p:
        out.update({f"{prefix}/test_f2p_ratio/{k}": v for k, v in compute_statistics(f2p).items()})
    if p2p:
        out.update({f"{prefix}/test_p2p_ratio/{k}": v for k, v in compute_statistics(p2p).items()})

    f2p_p = sum(_md_int(s, "test_f2p_passed") for s in eps)
    f2p_t = sum(_md_int(s, "test_f2p_total") for s in eps)
    p2p_p = sum(_md_int(s, "test_p2p_passed") for s in eps)
    p2p_t = sum(_md_int(s, "test_p2p_total") for s in eps)
    out[f"{prefix}/test_f2p_passed_total"] = float(f2p_p)
    out[f"{prefix}/test_f2p_total_total"] = float(f2p_t)
    out[f"{prefix}/test_p2p_passed_total"] = float(p2p_p)
    out[f"{prefix}/test_p2p_total_total"] = float(p2p_t)
    if f2p_t:
        out[f"{prefix}/test_f2p_macro_pass_rate"] = f2p_p / f2p_t
    if p2p_t:
        out[f"{prefix}/test_p2p_macro_pass_rate"] = p2p_p / p2p_t

    both = 0
    f2p_full = 0
    p2p_full = 0
    for s in eps:
        md = _meta(s)
        details = md.get("reward_details") if isinstance(md.get("reward_details"), dict) else {}
        fr = _safe_float(md.get("test_f2p_ratio"))
        if fr is None:
            fr = _safe_float(details.get("test_f2p_ratio"))
        pr = _safe_float(md.get("test_p2p_ratio"))
        if pr is None:
            pr = _safe_float(details.get("test_p2p_ratio"))
        if fr is None or pr is None:
            continue
        both += 1
        if fr >= 1.0:
            f2p_full += 1
        if pr >= 0.99:
            p2p_full += 1
    if both:
        out[f"{prefix}/reward_source/f2p_full_rate"] = f2p_full / both
        out[f"{prefix}/reward_source/p2p_full_rate"] = p2p_full / both
        out[f"{prefix}/reward_source/n_graded"] = float(both)
    return out


def _reward_source_metrics(samples: list) -> dict[str, float]:
    """``outcome/*`` = Stage-1; ``outcome/stage-2/*`` = Stage-2 branch episodes."""
    stage1, stage2 = _split_stage_episodes(samples)
    out = _reward_source_metrics_for_episodes(stage1, prefix="outcome")
    # Always emit stage-2 keys when any branch samples exist (or hybrid may have none).
    if stage2 or any(_sample_kind(s) == "branch" for s in samples):
        out.update(_reward_source_metrics_for_episodes(stage2, prefix="outcome/stage-2"))
    return out


def _filter_kind(samples: list, kind: str) -> list:
    if kind == "branch":
        return [s for s in samples if _sample_kind(s) == "branch"]
    return [s for s in samples if _sample_kind(s) != "branch"]


def _has_hybrid(samples: list) -> bool:
    return any(_sample_kind(s) == "branch" for s in samples) or any(
        "hybrid_stage1_wall_sec" in _meta(s) for s in samples
    )


def _timing_metrics_for_episodes(eps: list, *, prefix: str) -> dict[str, float]:
    from slime.utils.metric_utils import compute_statistics

    out: dict[str, float] = {}
    for key, leaf in (
        ("agent_elapsed_sec", "agent_time"),
        ("eval_elapsed_sec", "eval_time"),
        ("total_elapsed_sec", "sample_wall_time"),
    ):
        vals = _series(eps, key)
        if vals:
            out.update({f"{prefix}/{leaf}/{k}": v for k, v in compute_statistics(vals).items()})
            out[f"{prefix}/{leaf}/sum"] = float(sum(vals))
    return out


def _timing_metrics(samples: list, rollout_time: float) -> dict[str, float]:
    """Top-level ``perf/*`` times = Stage-1 (GRPO-comparable); stage-2 under ``perf/stage-2/*``."""
    stage1, stage2 = _split_stage_episodes(samples)
    out: dict[str, float] = {"perf/rollout_time": float(rollout_time)}
    out.update(_timing_metrics_for_episodes(stage1, prefix="perf"))
    if stage2 or _has_hybrid(samples):
        out.update(_timing_metrics_for_episodes(stage2, prefix="perf/stage-2"))

    if _has_hybrid(samples):
        # Full hybrid wall equals framework rollout_time; also emit step_grpo aliases.
        out["perf/step_grpo/rollout_time"] = float(rollout_time)
        # Per-prompt hybrid phase walls (dedupe by group_index + instance).
        seen: set[tuple[Any, ...]] = set()
        s1_walls: list[float] = []
        s2_walls: list[float] = []
        tot_walls: list[float] = []
        for s in _episode_samples(samples):
            md = _meta(s)
            key = (md.get("instance_id"), getattr(s, "group_index", None), md.get("hybrid_total_wall_sec"))
            if key in seen:
                continue
            # One stamp per prompt generate(); prefer vanilla episode reps.
            if _sample_kind(s) == "branch":
                continue
            if "hybrid_stage1_wall_sec" not in md and "hybrid_total_wall_sec" not in md:
                continue
            seen.add(key)
            v1 = _safe_float(md.get("hybrid_stage1_wall_sec"))
            v2 = _safe_float(md.get("hybrid_stage2_wall_sec"))
            vt = _safe_float(md.get("hybrid_total_wall_sec"))
            if v1 is not None:
                s1_walls.append(v1)
            if v2 is not None:
                s2_walls.append(v2)
            if vt is not None:
                tot_walls.append(vt)
        from slime.utils.metric_utils import compute_statistics

        if s1_walls:
            out.update(
                {f"perf/step_grpo/stage1_wall/{k}": v for k, v in compute_statistics(s1_walls).items()}
            )
            out["perf/step_grpo/stage1_wall/sum"] = float(sum(s1_walls))
        if s2_walls:
            out.update(
                {f"perf/step_grpo/stage2_wall/{k}": v for k, v in compute_statistics(s2_walls).items()}
            )
            out["perf/step_grpo/stage2_wall/sum"] = float(sum(s2_walls))
        if tot_walls:
            out.update(
                {f"perf/step_grpo/prompt_wall/{k}": v for k, v in compute_statistics(tot_walls).items()}
            )
            out["perf/step_grpo/prompt_wall/sum"] = float(sum(tot_walls))
        out["perf/step_grpo/n_stage1_episodes"] = float(len(stage1))
        out["perf/step_grpo/n_stage2_episodes"] = float(len(stage2))
        out["perf/step_grpo/n_samples"] = float(len(samples))
        out["perf/step_grpo/n_stage1_samples"] = float(len(_filter_kind(samples, "vanilla")))
        out["perf/step_grpo/n_stage2_samples"] = float(len(_filter_kind(samples, "branch")))
    return out


def _length_reward_metrics(samples: list, *, prefix_rollout: str, prefix_traj: str) -> dict[str, float]:
    """Episode length/reward stats for one stage (GRPO-comparable paths when prefix empty-ish)."""
    from slime.utils.metric_utils import compute_statistics
    import numpy as np

    out: dict[str, float] = {}
    groups = _group_by_episode(samples)
    if not groups:
        return out
    n_eps = max(len(groups), 1)

    num_segments: list[float] = []
    episode_response_lens: list[float] = []
    episode_rewards: list[float] = []
    episode_prompt_lens: list[float] = []
    episode_total_lens: list[float] = []
    kind_counter: Counter[str] = Counter()

    for members in groups.values():
        md0 = _meta(members[0])
        ns = _safe_float(md0.get("num_segments"))
        if ns is None:
            ns = float(len(members))
        num_segments.append(float(ns))
        episode_response_lens.append(sum(_response_length(s) for s in members))
        episode_rewards.append(sum(float(getattr(s, "reward", 0.0) or 0.0) for s in members))
        episode_prompt_lens.append(max(_prompt_length(s) for s in members))
        episode_total_lens.append(max(_total_length(s) for s in members))
        for s in members:
            kind = _meta(s).get("segment_kind")
            if kind:
                kind_counter[str(kind)] += 1

    out.update({f"{prefix_traj}/num_segments/{k}": v for k, v in compute_statistics(num_segments).items()})
    out[f"{prefix_traj}/n_samples"] = float(len(samples))
    out[f"{prefix_traj}/samples_per_episode"] = float(len(samples)) / float(n_eps)
    total_kinds = sum(kind_counter.values()) or 1
    for kind in ("wipe", "subagent", "final"):
        out[f"{prefix_traj}/segment_kind/{kind}_rate"] = kind_counter.get(kind, 0) / total_kinds
    out[f"{prefix_traj}/segment_kind/n_labeled"] = float(sum(kind_counter.values()))

    if episode_rewards:
        out.update(
            {f"{prefix_rollout}/episode_reward/{k}": v for k, v in compute_statistics(episode_rewards).items()}
        )
        out[f"{prefix_rollout}/episode_reward/std"] = float(np.std(np.array(episode_rewards)))
    if episode_response_lens:
        out.update(
            {
                f"{prefix_traj}/episode_response_len/{k}": v
                for k, v in compute_statistics(episode_response_lens).items()
            }
        )
    if episode_prompt_lens:
        out.update(
            {f"{prefix_rollout}/prompt_len/{k}": v for k, v in compute_statistics(episode_prompt_lens).items()}
        )
    if episode_total_lens:
        out.update(
            {f"{prefix_rollout}/total_len/{k}": v for k, v in compute_statistics(episode_total_lens).items()}
        )

    sample_prompt = [_prompt_length(s) for s in samples]
    sample_total = [_total_length(s) for s in samples]
    if sample_prompt:
        out.update(
            {f"{prefix_rollout}/sample_prompt_len/{k}": v for k, v in compute_statistics(sample_prompt).items()}
        )
    if sample_total:
        out.update(
            {f"{prefix_rollout}/sample_total_len/{k}": v for k, v in compute_statistics(sample_total).items()}
        )
    return out


def _outcome_infra_metrics(samples: list) -> dict[str, float]:
    """Abort / exit / apply rates: Stage-1 → ``outcome/*``, Stage-2 → ``outcome/stage-2/*``."""
    out: dict[str, float] = {}
    groups = _group_by_episode(samples)
    stage_stats: dict[str, dict[str, Any]] = {
        "vanilla": {
            "n_eps": 0,
            "n_abort": 0,
            "abort_reasons": Counter(),
            "n_exit_nonzero": 0,
            "n_exit_known": 0,
            "n_applied_true": 0,
            "n_applied_known": 0,
        },
        "branch": {
            "n_eps": 0,
            "n_abort": 0,
            "abort_reasons": Counter(),
            "n_exit_nonzero": 0,
            "n_exit_known": 0,
            "n_applied_true": 0,
            "n_applied_known": 0,
        },
    }
    for members in groups.values():
        stage = _sample_kind(members[0])
        st = stage_stats[stage]
        st["n_eps"] += 1
        aborted = any(_status_name(s) == "aborted" for s in members) or any(
            _meta(s).get("abort_reason") for s in members
        )
        if aborted:
            st["n_abort"] += 1
            reason = next(
                (_meta(s).get("abort_reason") for s in members if _meta(s).get("abort_reason")),
                "unknown",
            )
            st["abort_reasons"][str(reason)] += 1
        exit_code = None
        for s in members:
            if "agent_exit_code" in _meta(s):
                exit_code = _meta(s).get("agent_exit_code")
                break
        if exit_code is not None:
            st["n_exit_known"] += 1
            try:
                if int(exit_code) != 0:
                    st["n_exit_nonzero"] += 1
            except (TypeError, ValueError):
                st["n_exit_nonzero"] += 1
        applied = None
        for s in members:
            if "applied_cleanly" in _meta(s):
                applied = _meta(s).get("applied_cleanly")
                break
        if applied is not None:
            st["n_applied_known"] += 1
            if applied:
                st["n_applied_true"] += 1

    def _emit(prefix: str, st: dict[str, Any]) -> None:
        n = float(st["n_eps"] or 0)
        if n <= 0:
            return
        out[f"{prefix}/abort_rate"] = st["n_abort"] / n
        for reason, count in st["abort_reasons"].items():
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in reason)[:64]
            out[f"{prefix}/abort_reason/{safe}"] = float(count) / n
        if st["n_exit_known"]:
            out[f"{prefix}/agent_exit_nonzero_rate"] = st["n_exit_nonzero"] / float(st["n_exit_known"])
            out[f"{prefix}/agent_exit_known"] = float(st["n_exit_known"])
        if st["n_applied_known"]:
            out[f"{prefix}/applied_cleanly_rate"] = st["n_applied_true"] / float(st["n_applied_known"])
            out[f"{prefix}/applied_cleanly_known"] = float(st["n_applied_known"])

    _emit("outcome", stage_stats["vanilla"])
    if stage_stats["branch"]["n_eps"] or _has_hybrid(samples):
        _emit("outcome/stage-2", stage_stats["branch"])
    return out


def _trajectory_metrics(samples: list) -> dict[str, float]:
    """Stage-1 at GRPO paths; Stage-2 under ``*/stage-2/``."""
    stage1 = _filter_kind(samples, "vanilla")
    stage2 = _filter_kind(samples, "branch")
    out = _length_reward_metrics(stage1 or samples, prefix_rollout="rollout", prefix_traj="traj")
    out.update(_outcome_infra_metrics(samples))
    if stage2 or _has_hybrid(samples):
        out.update(
            _length_reward_metrics(stage2, prefix_rollout="rollout/stage-2", prefix_traj="traj/stage-2")
        )
    return out


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

    log_dict = dict(rollout_extra_metrics or {})
    stage1 = _filter_kind(samples, "vanilla")
    stage2 = _filter_kind(samples, "branch")
    # Top-level rollout/perf sample stats = Stage-1 only (comparable to naive GRPO).
    try:
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, stage1 or samples), "rollout/")
    except Exception:
        logger.exception("compute_metrics_from_samples failed")
    try:
        # Token throughput vs full rollout wall (same as GRPO's perf/rollout_time denominator).
        log_dict |= dict_add_prefix(
            compute_perf_metrics_from_samples(args, stage1 or samples, rollout_time), "perf/"
        )
    except Exception:
        logger.exception("compute_perf_metrics_from_samples failed")
    if stage2 or _has_hybrid(samples):
        try:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, stage2), "rollout/stage-2/")
        except Exception:
            logger.exception("stage-2 compute_metrics_from_samples failed")
        try:
            log_dict |= dict_add_prefix(
                compute_perf_metrics_from_samples(args, stage2, rollout_time), "perf/stage-2/"
            )
        except Exception:
            logger.exception("stage-2 compute_perf_metrics_from_samples failed")
        # Full hybrid (stage1+stage2) throughput vs same wall — step-GRPO cost view.
        try:
            full_perf = compute_perf_metrics_from_samples(args, samples, rollout_time)
            log_dict["perf/step_grpo/tokens_per_gpu_per_sec"] = float(
                full_perf.get("tokens_per_gpu_per_sec") or 0.0
            )
            if "effective_tokens_per_gpu_per_sec" in full_perf:
                log_dict["perf/step_grpo/effective_tokens_per_gpu_per_sec"] = float(
                    full_perf["effective_tokens_per_gpu_per_sec"]
                )
        except Exception:
            logger.exception("step_grpo full perf failed")
    try:
        log_dict.update(_reward_source_metrics(samples))
    except Exception:
        logger.exception("reward source metrics failed")
    try:
        log_dict.update(_timing_metrics(samples, rollout_time))
    except Exception:
        logger.exception("timing metrics failed")
    try:
        log_dict.update(_trajectory_metrics(samples))
    except Exception:
        logger.exception("trajectory metrics failed")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logger.info("perf %s: %s", rollout_id, log_dict)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True