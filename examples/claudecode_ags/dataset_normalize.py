"""Normalize official HF SWE rows into flat Path A metadata."""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_REGISTRY = "swebenchdocker.tencentcloudcr.com"

_TYPE_ALIASES: dict[str, str] = {
    "swebench": "swebench",
    "swe-bench": "swebench",
    "swe_bench": "swebench",
    "swebench_verified": "swebench_verified",
    "verified": "swebench_verified",
    "swegym": "swegym",
    "swe_gym": "swegym",
    "swe-gym": "swegym",
    "swesmith": "swesmith",
    "swe_smith": "swesmith",
    "swe-smith": "swesmith",
    "rebench": "rebench",
    "swerebench": "rebench",
    "swe_rebench": "rebench",
    "swe-rebench": "rebench",
    "swerebenchv2": "rebench",
    "scaleswe": "scaleswe",
    "scale_swe": "scaleswe",
    "scale-swe": "scaleswe",
}

_DEFAULT_DATA_SOURCE: dict[str, str] = {
    "swebench": "swebench",
    "swebench_verified": "swebench_verified",
    "swegym": "swegym",
    "swesmith": "swe_smith",
    "rebench": "swerebench",
    "scaleswe": "scaleswe",
}

_PASSTHROUGH_KEYS = (
    "instance_id",
    "problem_statement",
    "repo",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "test_patch",
    "patch",
    "environment_setup_commit",
    "hints_text",
    "pre_commands",
    "f2p_script",
    "f2p_patch",
    "parent_commit",
    "image_url",
    "image_name",
    "docker_image",
    "install_config",
    "log_parser",
    "eval_cmd",
    "agent_prompt",
    "cc_source",
    "data_path",
)


def canonical_dataset_type(dataset_type: str) -> str:
    key = str(dataset_type or "").strip().lower()
    if not key:
        raise ValueError("dataset_type is required")
    canon = _TYPE_ALIASES.get(key)
    if canon is None:
        raise ValueError(f"unknown dataset_type: {dataset_type!r}")
    return canon


def image_registry(registry: str | None = None) -> str:
    if registry is not None and str(registry).strip():
        return str(registry).strip().rstrip("/")
    env = (os.environ.get("SLIME_CC_IMAGE_REGISTRY") or "").strip().rstrip("/")
    return env or _DEFAULT_REGISTRY


def _str(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key]).strip()
    return ""


def resolve_image(
    row: dict[str, Any],
    dataset_type: str,
    *,
    registry: str | None = None,
) -> str:
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")
    existing = _str(row, "image")
    if existing:
        return existing

    canon = canonical_dataset_type(dataset_type)
    reg = image_registry(registry)
    instance_id = _str(row, "instance_id")

    if canon in {"swebench", "swebench_verified"}:
        if not instance_id:
            return ""
        docker_id = instance_id.replace("__", "_1776_")
        return f"{reg}/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()

    if canon == "swegym":
        if not instance_id:
            return ""
        docker_id = instance_id.replace("__", "_s_")
        return f"{reg}/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()

    if canon == "swesmith":
        image_name = _str(row, "image_name")
        if not image_name:
            return ""
        path = image_name.split("/", 1)[1] if "/" in image_name else image_name
        if ":" not in path:
            path = f"{path}:latest"
        return f"{reg}/swebench/{path}".lower()

    if canon == "rebench":
        raw = _str(row, "docker_image", "image_name")
        if not raw:
            return ""
        # Drop registry/path prefixes; keep final repo:tag (or bare name).
        repo_and_tag = raw.split("/")[-1]
        if not repo_and_tag:
            return ""
        return f"{reg}/swerebenchv2/{repo_and_tag}".lower()

    if canon == "scaleswe":
        image_url = _str(row, "image_url")
        if ":" in image_url:
            tag = image_url.rsplit(":", 1)[-1]
        else:
            tag = instance_id or image_url
        if not tag:
            return ""
        return f"{reg}/aweaiteam/scaleswe:{tag}".lower()

    return ""


def normalize_official_row(
    row: dict[str, Any],
    *,
    dataset_type: str,
    registry: str | None = None,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")
    canon = canonical_dataset_type(dataset_type)
    out: dict[str, Any] = dict(row)

    out["dataset_type"] = canon
    if not _str(out, "data_source"):
        out["data_source"] = _DEFAULT_DATA_SOURCE[canon]

    workdir = _str(out, "workdir") or "/testbed"
    out["workdir"] = workdir

    base_commit = _str(out, "base_commit")
    if not base_commit and canon == "scaleswe":
        base_commit = _str(out, "parent_commit")
    if base_commit:
        out["base_commit"] = base_commit

    out["image"] = resolve_image(out, canon, registry=registry)

    if canon == "swesmith" and not _str(out, "swe_smith_bug_patch"):
        patch = out.get("patch")
        if patch not in (None, ""):
            out["swe_smith_bug_patch"] = patch

    # Ensure passthrough keys exist only if present in input (already via dict copy).
    for key in _PASSTHROUGH_KEYS:
        if key in row and key not in out:
            out[key] = row[key]

    return out
