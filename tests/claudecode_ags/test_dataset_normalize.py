"""Tests for official HF → flat metadata normalize."""

from __future__ import annotations

import pytest

from examples.claudecode_ags.dataset_normalize import (
    canonical_dataset_type,
    normalize_official_row,
    resolve_image,
)
from slime.utils.types import Sample

_TCR = "swebenchdocker.tencentcloudcr.com"


def test_canonical_aliases():
    assert canonical_dataset_type("SWE_SMITH") == "swesmith"
    assert canonical_dataset_type("verified") == "swebench_verified"
    assert canonical_dataset_type("swerebenchv2") == "rebench"


def test_unknown_dataset_type():
    with pytest.raises(ValueError, match="unknown dataset_type"):
        canonical_dataset_type("not-a-real-type")


def test_swebench_image():
    img = resolve_image({"instance_id": "astropy__astropy-12907"}, "swebench")
    assert img == f"{_TCR}/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


def test_swegym_image():
    img = resolve_image({"instance_id": "getmoto__moto-7365"}, "swegym")
    assert img == f"{_TCR}/swebench/sweb.eval.x86_64.getmoto_s_moto-7365:latest"


def test_swesmith_image_strips_owner():
    img = resolve_image(
        {"image_name": "jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536"},
        "swesmith",
    )
    assert img == f"{_TCR}/swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536:latest"


def test_rebench_image_last_segment():
    img = resolve_image(
        {"docker_image": "docker.io/swerebenchv2/behat-gherkin:343-e522894"},
        "rebench",
    )
    assert img == f"{_TCR}/swerebenchv2/behat-gherkin:343-e522894"


def test_scaleswe_image_tag():
    img = resolve_image(
        {"image_url": "aweaiteam/scaleswe:auth0_auth0-python_pr671"},
        "scaleswe",
    )
    assert img == f"{_TCR}/aweaiteam/scaleswe:auth0_auth0-python_pr671"


def test_custom_registry(monkeypatch):
    monkeypatch.setenv("SLIME_CC_IMAGE_REGISTRY", "example.registry/")
    img = resolve_image({"instance_id": "a__b-1"}, "swebench")
    assert img.startswith("example.registry/swebench/")


def test_preserve_existing_image():
    img = resolve_image(
        {"image": "my.reg/custom:tag", "instance_id": "a__b-1"},
        "swebench",
    )
    assert img == "my.reg/custom:tag"


def test_normalize_swesmith_fields():
    out = normalize_official_row(
        {
            "instance_id": "oauthlib__oauthlib.1fd52536",
            "image_name": "jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536",
            "patch": "diff --git a/x b/x\n",
            "FAIL_TO_PASS": ["t::a"],
            "problem_statement": "bug",
        },
        dataset_type="swesmith",
    )
    assert out["data_source"] == "swe_smith"
    assert out["swe_smith_bug_patch"].startswith("diff")
    assert out["FAIL_TO_PASS"] == ["t::a"]
    assert out["image"].startswith(f"{_TCR}/swebench/swesmith")


def test_normalize_scaleswe_and_rebench_passthrough():
    scaleswe = normalize_official_row(
        {
            "instance_id": "x",
            "image_url": "aweaiteam/scaleswe:tag1",
            "pre_commands": "git checkout a",
            "f2p_script": "def test_x(): pass",
            "workdir": "/repo",
        },
        dataset_type="scaleswe",
    )
    assert scaleswe["pre_commands"] == "git checkout a"
    assert scaleswe["f2p_script"].startswith("def")
    assert scaleswe["workdir"] == "/repo"

    rebench = normalize_official_row(
        {
            "instance_id": "y",
            "docker_image": "swerebenchv2/pkg:1",
            "install_config": {"test_cmd": "pytest -q"},
        },
        dataset_type="rebench",
    )
    assert rebench["install_config"]["test_cmd"] == "pytest -q"
    assert rebench["data_source"] == "swerebench"


def test_explicit_data_source_preserved():
    out = normalize_official_row(
        {"instance_id": "a__b-1", "data_source": "custom_src"},
        dataset_type="swebench",
    )
    assert out["data_source"] == "custom_src"


def test_missing_image_fields_yield_empty():
    assert resolve_image({}, "swesmith") == ""
    assert resolve_image({}, "rebench") == ""


def test_parse_metadata_normalizes_when_dataset_type_set():
    from examples.claudecode_ags.generate import _parse_metadata

    sample = Sample(
        prompt="fix",
        metadata={
            "dataset_type": "swebench",
            "instance_id": "astropy__astropy-12907",
            "problem_statement": "bug",
            "base_commit": "abc",
            "FAIL_TO_PASS": ["t::a"],
        },
    )
    md = _parse_metadata(sample)
    assert md["image"] == f"{_TCR}/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    assert md["data_source"] == "swebench"
    assert md["workdir"] == "/testbed"


def test_parse_metadata_skips_without_dataset_type():
    from examples.claudecode_ags.generate import _parse_metadata

    sample = Sample(
        prompt="fix",
        metadata={"image": "img:tag", "workdir": "/testbed", "eval_cmd": "true"},
    )
    md = _parse_metadata(sample)
    assert md["image"] == "img:tag"
    assert md["eval_cmd"] == "true"
