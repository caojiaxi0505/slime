"""Custom rollout wandb logger: F2P/P2P reward sources + per-sample timing.

Wired via::

    --custom-rollout-log-function-path examples.claudecode_ags.wandb_metrics.log_rollout_data

Returns True so slime skips the default logger (we emit defaults + extras once).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _meta(sample) -> dict[str, Any]:
    md = getattr(sample, "metadata", None) or {}
    return md if isinstance(md, dict) else {}


def _episode_samples(samples: list) -> list:
    """One row per CC attempt (fan-out segments share the same F2P/P2P/timing)."""
    seen: set[tuple[Any, ...]] = set()
    out = []
    for s in samples:
        md = _meta(s)
        seg = md.get("segment_idx")
        if seg not in (None, 0):
            continue
        key = (
            md.get("instance_id"),
            getattr(s, "group_index", None),
            getattr(s, "index", None),
            getattr(s, "rollout_id", None),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out or list(samples)


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


def _reward_source_metrics(samples: list) -> dict[str, float]:
    from slime.utils.metric_utils import compute_statistics

    eps = _episode_samples(samples)
    out: dict[str, float] = {}
    n = max(len(eps), 1)

    solved = [1.0 if _meta(s).get("grading_solved") else 0.0 for s in eps]
    out["outcome/resolved_rate"] = sum(solved) / n
    out["outcome/n_episodes"] = float(len(eps))

    f2p = _series(eps, "test_f2p_ratio")
    p2p = _series(eps, "test_p2p_ratio")
    if f2p:
        out.update({f"outcome/test_f2p_ratio/{k}": v for k, v in compute_statistics(f2p).items()})
    if p2p:
        out.update({f"outcome/test_p2p_ratio/{k}": v for k, v in compute_statistics(p2p).items()})

    f2p_p = sum(int(_meta(s).get("test_f2p_passed") or 0) for s in eps)
    f2p_t = sum(int(_meta(s).get("test_f2p_total") or 0) for s in eps)
    p2p_p = sum(int(_meta(s).get("test_p2p_passed") or 0) for s in eps)
    p2p_t = sum(int(_meta(s).get("test_p2p_total") or 0) for s in eps)
    out["outcome/test_f2p_passed_total"] = float(f2p_p)
    out["outcome/test_f2p_total_total"] = float(f2p_t)
    out["outcome/test_p2p_passed_total"] = float(p2p_p)
    out["outcome/test_p2p_total_total"] = float(p2p_t)
    if f2p_t:
        out["outcome/test_f2p_macro_pass_rate"] = f2p_p / f2p_t
    if p2p_t:
        out["outcome/test_p2p_macro_pass_rate"] = p2p_p / p2p_t

    # Reward-source mix among episodes that have both ratios.
    both = 0
    f2p_full = 0
    p2p_full = 0
    for s in eps:
        md = _meta(s)
        fr = _safe_float(md.get("test_f2p_ratio"))
        pr = _safe_float(md.get("test_p2p_ratio"))
        if fr is None or pr is None:
            continue
        both += 1
        if fr >= 1.0:
            f2p_full += 1
        if pr >= 0.99:
            p2p_full += 1
    if both:
        out["outcome/reward_source/f2p_full_rate"] = f2p_full / both
        out["outcome/reward_source/p2p_full_rate"] = p2p_full / both
        out["outcome/reward_source/n_graded"] = float(both)
    return out


def _timing_metrics(samples: list, rollout_time: float) -> dict[str, float]:
    from slime.utils.metric_utils import compute_statistics

    eps = _episode_samples(samples)
    out: dict[str, float] = {"perf/rollout_time": float(rollout_time)}

    for key, prefix in (
        ("agent_elapsed_sec", "perf/agent_time"),
        ("eval_elapsed_sec", "perf/eval_time"),
        ("total_elapsed_sec", "perf/sample_wall_time"),
    ):
        vals = _series(eps, key)
        if vals:
            out.update({f"{prefix}/{k}": v for k, v in compute_statistics(vals).items()})
    return out


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

    log_dict = dict(rollout_extra_metrics or {})
    try:
        log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    except Exception:
        logger.exception("compute_metrics_from_samples failed")
    try:
        log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    except Exception:
        logger.exception("compute_perf_metrics_from_samples failed")
    try:
        log_dict.update(_reward_source_metrics(samples))
    except Exception:
        logger.exception("reward source metrics failed")
    try:
        log_dict.update(_timing_metrics(samples, rollout_time))
    except Exception:
        logger.exception("timing metrics failed")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logger.info("perf %s: %s", rollout_id, log_dict)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True
