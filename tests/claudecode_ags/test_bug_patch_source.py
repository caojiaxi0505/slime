"""Tests for parquet swe_smith_bug_patch backfill."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from examples.claudecode_ags.bug_patch_source import resolve_swe_smith_bug_patch


@pytest.fixture
def smith_parquet(tmp_path):
    path = tmp_path / "src.parquet"
    df = pd.DataFrame(
        [
            {
                "data_source": "swe_smith",
                "reward_model": json.dumps({"gold_patch": "diff --git a/x b/x\n"}),
                "extra_info": json.dumps({"instance_id": "repo__42"}),
            }
        ]
    )
    df.to_parquet(path)
    # Clear lru_cache between tests that rewrite files at same path is N/A (unique tmp).
    from examples.claudecode_ags import bug_patch_source as bps

    bps._read_source_parquet.cache_clear()
    return path


def test_returns_existing_patch_without_parquet():
    assert (
        resolve_swe_smith_bug_patch(
            {"data_source": "swe_smith", "swe_smith_bug_patch": "PATCH"}
        )
        == "PATCH"
    )


def test_non_smith_returns_empty():
    assert resolve_swe_smith_bug_patch({"data_source": "other", "data_path": "/nope"}) == ""


def test_loads_from_parquet_by_instance_id(smith_parquet):
    patch = resolve_swe_smith_bug_patch(
        {
            "data_source": "swe_smith",
            "instance_id": "repo__42",
            "data_path": str(smith_parquet),
        }
    )
    assert "diff --git" in patch


def test_loads_from_parquet_by_index(smith_parquet):
    patch = resolve_swe_smith_bug_patch(
        {
            "data_source": "swe_smith",
            "instance_id": "repo__42",
            "cc_source": {"data_path": str(smith_parquet), "instance_index": 0},
        }
    )
    assert "diff --git" in patch


def test_missing_path_returns_empty():
    assert resolve_swe_smith_bug_patch({"data_source": "swe_smith", "instance_id": "x"}) == ""
