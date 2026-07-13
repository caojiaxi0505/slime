"""Load swe_smith_bug_patch from a source parquet when metadata omits it."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from examples.claudecode_ags.workspace_init import is_swesmith_data_source

logger = logging.getLogger(__name__)


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return value


def _row_get(row: Any, key: str) -> Any:
    try:
        return row[key]
    except Exception:
        return None


def gold_from_reward_model(reward_model: Any) -> str:
    rm = _json_loads_maybe(reward_model)
    if not isinstance(rm, dict):
        return ""
    gt = _json_loads_maybe(rm.get("ground_truth"))
    if isinstance(gt, dict) and gt.get("gold_patch"):
        return str(gt["gold_patch"])
    return str(rm.get("gold_patch") or "")


def _row_instance_id(row: Any) -> str:
    for key in ("extra_info", "metadata", "reward_model"):
        value = _json_loads_maybe(_row_get(row, key))
        if isinstance(value, dict) and value.get("instance_id"):
            return str(value["instance_id"])
    return ""


@lru_cache(maxsize=8)
def _read_source_parquet(path: str):
    import pandas as pd

    return pd.read_parquet(path)


def resolve_swe_smith_bug_patch(metadata: dict[str, Any]) -> str:
    """Return existing patch, or load from parquet when swesmith and patch missing."""
    existing = metadata.get("swe_smith_bug_patch")
    if isinstance(existing, str) and existing.strip():
        return existing
    if existing not in (None, ""):
        return str(existing)

    data_source = str(metadata.get("data_source") or "")
    if not is_swesmith_data_source(data_source):
        return ""

    instance_id = str(metadata.get("instance_id") or "")
    cc_source = metadata.get("cc_source") if isinstance(metadata.get("cc_source"), dict) else {}
    data_path = str(cc_source.get("data_path") or metadata.get("data_path") or "")
    if not data_path:
        return ""

    try:
        df = _read_source_parquet(data_path)
    except Exception as exc:
        logger.warning("[bug_patch_source] could not read parquet %s: %s", data_path, exc)
        return ""

    instance_index = cc_source.get("instance_index", metadata.get("instance_index"))
    candidates: list[Any] = []
    if instance_index is not None:
        try:
            candidates.append(df.iloc[int(instance_index)])
        except Exception:
            pass
    if instance_id:
        for _, row in df.iterrows():
            if _row_instance_id(row) == instance_id:
                candidates.append(row)
                break

    for row in candidates:
        rid = _row_instance_id(row)
        if instance_id and rid and rid != instance_id:
            continue
        row_ds = _row_get(row, "data_source")
        if row_ds and not is_swesmith_data_source(row_ds):
            continue
        patch = gold_from_reward_model(_row_get(row, "reward_model"))
        if patch:
            return patch
    return ""
