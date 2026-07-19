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
import os
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
    kind = _sample_kind(sample)
    # Hybrid compact segments carry the true episode identity explicitly.
    # Prefer it over Sample.index: the old branch index omitted edit_step_i,
    # so branches with the same source trial / branch number at different
    # edit points collided in W&B.
    explicit_uid = md.get("branch_uid")
    if explicit_uid is None:
        explicit_uid = getattr(sample, "loss_group_id", None)
    if explicit_uid is not None:
        return (
            md.get("instance_id"),
            getattr(sample, "group_index", None),
            kind,
            "episode_uid",
            explicit_uid,
        )
    if kind == "branch":
        return (
            md.get("instance_id"),
            getattr(sample, "group_index", None),
            kind,
            md.get("step_group_key"),
            md.get("edit_step_i", md.get("step_t")),
            md.get("source_trial_idx"),
            md.get("branch_idx"),
            getattr(sample, "index", None),
            getattr(sample, "rollout_id", None),
        )
    return (
        md.get("instance_id"),
        getattr(sample, "group_index", None),
        getattr(sample, "index", None),
        getattr(sample, "rollout_id", None),
        kind,
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


def _tool_loop_metrics_for_episodes(eps: list, *, prefix: str) -> dict[str, float]:
    """Repeated tool-signature prevalence for one rollout step and stage."""
    from slime.utils.metric_utils import compute_statistics

    checked = [
        s
        for s in eps
        if bool(_meta(s).get("tool_loop_detection_enabled"))
        or bool(
            (_meta(s).get("reward_details") or {}).get("tool_loop_detection_enabled")
            if isinstance(_meta(s).get("reward_details"), dict)
            else False
        )
    ]
    if not checked:
        return {}
    runs = [float(_md_int(s, "consecutive_tool_signature_max")) for s in checked]
    out = {
        f"{prefix}/n_checked_episodes": float(len(runs)),
    }
    out.update(
        {
            f"{prefix}/consecutive_signature_max/{stat}": value
            for stat, value in compute_statistics(runs).items()
        }
    )
    for threshold in (3, 4, 5):
        count = sum(value >= threshold for value in runs)
        out[f"{prefix}/ge_{threshold}_count"] = float(count)
        out[f"{prefix}/ge_{threshold}_rate"] = float(count) / float(len(runs))
    return out


def _tool_loop_metrics(samples: list) -> dict[str, float]:
    """Stage-1 curves at the common path; Hybrid Stage-2 under stage-2."""
    stage1, stage2 = _split_stage_episodes(samples)
    out = _tool_loop_metrics_for_episodes(stage1, prefix="behavior/tool_loop")
    if stage2 or _has_hybrid(samples):
        out.update(
            _tool_loop_metrics_for_episodes(
                stage2,
                prefix="behavior/stage-2/tool_loop",
            )
        )
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
        ("agent_queue_wait_sec", "agent_queue_wait"),
        ("agent_elapsed_sec", "agent_time"),
        ("eval_elapsed_sec", "eval_time"),
        ("total_elapsed_sec", "sample_wall_time"),
    ):
        vals = _series(eps, key)
        if vals:
            out.update({f"{prefix}/{leaf}/{k}": v for k, v in compute_statistics(vals).items()})
            out[f"{prefix}/{leaf}/sum"] = float(sum(vals))

    # The adapter stores one aggregate per Claude attempt.  These are already
    # deduplicated here at episode granularity, so compact segment fan-out does
    # not multiply the measurements.
    for key, leaf in (
        ("adapter_loop_delay_ms_mean", "loop_delay_ms"),
        ("adapter_session_lock_wait_ms_mean", "session_lock_wait_ms"),
        ("adapter_json_ms_mean", "json_ms"),
        ("adapter_hash_ms_mean", "hash_ms"),
        ("adapter_prepare_tokenize_ms_mean", "prepare_tokenize_ms"),
        ("adapter_parse_ms_mean", "parse_ms"),
        ("adapter_cpu_queue_ms_mean", "cpu_queue_ms"),
        ("adapter_cpu_ms_mean", "cpu_ms"),
        ("adapter_sglang_e2e_ms_mean", "sglang_e2e_ms"),
        ("adapter_total_ms_mean", "turn_total_ms"),
    ):
        vals = _series(eps, key)
        if vals:
            out.update(
                {
                    f"{prefix}/adapter/{leaf}/{stat}": value
                    for stat, value in compute_statistics(vals).items()
                }
            )

    turn_counts = _series(eps, "adapter_turn_count")
    failed_counts = _series(eps, "adapter_failed_request_count")
    if turn_counts:
        total_turns = float(sum(turn_counts))
        out[f"{prefix}/adapter/n_turns"] = total_turns
        out[f"{prefix}/adapter/n_episodes_with_timing"] = float(len(turn_counts))
        if failed_counts and total_turns > 0:
            out[f"{prefix}/adapter/request_failure_rate"] = float(sum(failed_counts)) / total_turns
    return out


def _hybrid_objective_metrics(samples: list) -> dict[str, float]:
    """Audit counts for the explicit ``L_v + lambda * L_b`` objective."""
    out: dict[str, float] = {}
    groups = _group_by_episode(samples)
    active_eps = {"vanilla": 0, "branch": 0}
    active_tokens = {"vanilla": 0, "branch": 0}
    nominal_weights = {"vanilla": 0.0, "branch": 0.0}
    active_edit_groups: set[tuple[Any, ...]] = set()
    transcript_known = 0
    transcript_invalid = 0
    resume_known = 0
    resume_verified = 0
    resume_echo_mismatches = 0
    resume_echo_mismatch_branches = 0
    resume_echo_missing = 0
    resume_echo_payload_mismatches = 0
    resume_schema_mismatches = 0
    resume_schema_mismatch_branches = 0
    resume_unavailable_tools = 0
    resume_unavailable_tool_branches = 0
    resume_invalid_tool_inputs = 0
    resume_invalid_tool_input_branches = 0
    resume_max_tokens_continuations = 0
    resume_post_end_turn_acks = 0
    resume_request_replays = 0
    resume_request_replay_branches = 0
    resume_hashes: dict[tuple[Any, ...], set[str]] = defaultdict(set)

    for members in groups.values():
        stage = _sample_kind(members[0])
        if stage == "vanilla":
            validity = next(
                (_meta(s).get("transcript_valid") for s in members if "transcript_valid" in _meta(s)),
                None,
            )
            if validity is not None:
                transcript_known += 1
                transcript_invalid += int(not bool(validity))
        else:
            md = _meta(members[0])
            if "prompt_exact" in md or "prefix_reseed_verified" in md:
                resume_known += 1
                resume_verified += int(
                    bool(md.get("prompt_exact", md.get("prefix_reseed_verified")))
                )
                echo_mismatches = int(md.get("tool_use_echo_mismatch_count") or 0)
                resume_echo_mismatches += echo_mismatches
                resume_echo_mismatch_branches += int(echo_mismatches > 0)
                resume_echo_missing += int(md.get("tool_use_echo_missing_count") or 0)
                resume_echo_payload_mismatches += int(
                    md.get("tool_use_echo_payload_mismatch_count") or 0
                )
                schema_mismatches = int(md.get("runtime_tool_schema_mismatch_count") or 0)
                resume_schema_mismatches += schema_mismatches
                resume_schema_mismatch_branches += int(schema_mismatches > 0)
                unavailable_tools = int(
                    md.get("generated_runtime_tool_unavailable_count") or 0
                )
                resume_unavailable_tools += unavailable_tools
                resume_unavailable_tool_branches += int(unavailable_tools > 0)
                invalid_tool_inputs = int(
                    md.get("generated_runtime_tool_input_invalid_count") or 0
                )
                resume_invalid_tool_inputs += invalid_tool_inputs
                resume_invalid_tool_input_branches += int(invalid_tool_inputs > 0)
                resume_max_tokens_continuations += int(
                    md.get("max_tokens_continuation_count") or 0
                )
                resume_post_end_turn_acks += int(md.get("post_end_turn_ack_count") or 0)
                request_replays = int(md.get("resume_request_replay_count") or 0)
                resume_request_replays += request_replays
                resume_request_replay_branches += int(request_replays > 0)
            prompt_hash = str(md.get("prompt_sha256") or "")
            if prompt_hash:
                resume_hashes[
                    (
                        md.get("instance_id"),
                        getattr(members[0], "group_index", None),
                        md.get("step_group_key"),
                    )
                ].add(prompt_hash)

        tokens = 0
        for s in members:
            if getattr(s, "remove_sample", False) or getattr(s, "is_filtered_out", False):
                continue
            mask = getattr(s, "loss_mask", None)
            tokens += int(getattr(s, "response_length", 0) or 0) if mask is None else sum(int(x) for x in mask)
        if tokens <= 0:
            continue

        active_eps[stage] += 1
        active_tokens[stage] += tokens
        weight = next(
            (
                _safe_float(getattr(s, "loss_weight", None))
                for s in members
                if _safe_float(getattr(s, "loss_weight", None)) is not None
            ),
            1.0,
        )
        nominal_weights[stage] += float(weight)
        if stage == "branch":
            md = _meta(members[0])
            active_edit_groups.add(
                (
                    md.get("instance_id"),
                    getattr(members[0], "group_index", None),
                    md.get("step_group_key"),
                )
            )

    out["perf/step_grpo/n_stage1_active_episodes"] = float(active_eps["vanilla"])
    out["perf/step_grpo/n_stage2_active_episodes"] = float(active_eps["branch"])
    out["perf/step_grpo/n_active_edit_groups"] = float(len(active_edit_groups))
    out["perf/step_grpo/stage1_active_tokens"] = float(active_tokens["vanilla"])
    out["perf/step_grpo/stage2_active_tokens"] = float(active_tokens["branch"])
    out["perf/step_grpo/stage1_nominal_loss_weight"] = nominal_weights["vanilla"]
    out["perf/step_grpo/stage2_nominal_loss_weight"] = nominal_weights["branch"]
    out["perf/step_grpo/branch_loss_weight"] = float(os.environ.get("STEP_GRPO_BRANCH_LOSS_WEIGHT", "1"))
    out["perf/step_grpo/transcript_status_known"] = float(transcript_known)
    out["perf/step_grpo/transcript_invalid_trials"] = float(transcript_invalid)
    if resume_known:
        out["resume/prompt_exact_rate"] = float(resume_verified) / float(resume_known)
        out["resume/n_verified_branches"] = float(resume_verified)
        out["resume/n_checked_branches"] = float(resume_known)
        out["resume/tool_use_echo_mismatch_count"] = float(resume_echo_mismatches)
        out["resume/tool_use_echo_mismatch_branch_rate"] = float(
            resume_echo_mismatch_branches
        ) / float(resume_known)
        out["resume/tool_use_echo_missing_count"] = float(resume_echo_missing)
        out["resume/tool_use_echo_payload_mismatch_count"] = float(
            resume_echo_payload_mismatches
        )
        out["resume/runtime_tool_schema_mismatch_count"] = float(resume_schema_mismatches)
        out["resume/runtime_tool_schema_mismatch_branch_rate"] = float(
            resume_schema_mismatch_branches
        ) / float(resume_known)
        out["resume/generated_runtime_tool_unavailable_count"] = float(
            resume_unavailable_tools
        )
        out["resume/generated_runtime_tool_unavailable_branch_rate"] = float(
            resume_unavailable_tool_branches
        ) / float(resume_known)
        out["resume/generated_runtime_tool_input_invalid_count"] = float(
            resume_invalid_tool_inputs
        )
        out["resume/generated_runtime_tool_input_invalid_branch_rate"] = float(
            resume_invalid_tool_input_branches
        ) / float(resume_known)
        out["resume/max_tokens_continuation_count"] = float(
            resume_max_tokens_continuations
        )
        out["resume/post_end_turn_ack_count"] = float(resume_post_end_turn_acks)
        out["resume/request_replay_count"] = float(resume_request_replays)
        out["resume/request_replay_branch_rate"] = float(
            resume_request_replay_branches
        ) / float(resume_known)
    if resume_hashes:
        consistent = sum(len(hashes) == 1 for hashes in resume_hashes.values())
        out["resume/checkpoint_hash_consistency_rate"] = float(consistent) / float(
            len(resume_hashes)
        )
        out["resume/n_checked_edit_groups"] = float(len(resume_hashes))

    # These values are stamped once per prompt but repeated on every compact
    # segment. Deduplicate before summing them across the rollout.
    prompt_reps: dict[tuple[Any, Any], Any] = {}
    for s in samples:
        md = _meta(s)
        key = (md.get("instance_id"), getattr(s, "group_index", None))
        prompt_reps.setdefault(key, s)
    for metadata_key, metric_leaf in (
        ("hybrid_num_patch_candidates", "n_patch_candidates"),
        ("hybrid_num_selected_edits", "n_selected_edits"),
        ("hybrid_num_branch_tasks", "n_branch_tasks"),
        ("hybrid_num_dropped_branches", "n_dropped_branches"),
        ("hybrid_num_dropped_timeout", "n_dropped_timeout"),
        ("hybrid_num_dropped_resume_tool_echo", "n_dropped_resume_tool_echo"),
        ("hybrid_num_dropped_resume_missing_result", "n_dropped_resume_missing_result"),
        ("hybrid_num_dropped_resume_no_pending", "n_dropped_resume_no_pending"),
        ("hybrid_num_dropped_resume_tool_schema", "n_dropped_resume_tool_schema"),
        ("hybrid_num_dropped_resume_subagent", "n_dropped_resume_subagent"),
        ("hybrid_num_dropped_resume_other", "n_dropped_resume_other"),
        ("hybrid_num_dropped_workspace_rebuild", "n_dropped_workspace_rebuild"),
        ("hybrid_num_dropped_other", "n_dropped_other"),
    ):
        values = [_safe_float(_meta(s).get(metadata_key)) for s in prompt_reps.values()]
        known_values = [v for v in values if v is not None]
        if known_values:
            out[f"perf/step_grpo/{metric_leaf}"] = float(sum(known_values))
    planned = out.get("perf/step_grpo/n_branch_tasks")
    dropped = out.get("perf/step_grpo/n_dropped_branches")
    if planned is not None and planned > 0 and dropped is not None:
        completed = max(planned - dropped, 0.0)
        out["perf/step_grpo/n_completed_branches"] = completed
        out["perf/step_grpo/branch_completion_rate"] = completed / planned
        out["perf/step_grpo/branch_drop_rate"] = dropped / planned
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
        out.update(_hybrid_objective_metrics(samples))
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
    out.update(_tool_loop_metrics(samples))
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
