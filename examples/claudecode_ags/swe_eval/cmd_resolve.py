"""Resolve EvalPlan: mode detection + eval command assembly from official fields."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from examples.claudecode_ags.swe_eval.grade_common import parse_list
from examples.claudecode_ags.workspace_init import coerce_install_config, normalize_pre_commands


class EvalMode(str, Enum):
    SCALESWE = "scaleswe"
    REBENCH = "rebench"
    SWEBENCH = "swebench"
    SIMPLE_CMD = "simple_cmd"
    NONE = "none"


@dataclass
class EvalPlan:
    mode: EvalMode
    eval_cmd: str = ""
    workdir: str = "/testbed"
    test_patch: str = ""
    f2p_patch: str = ""
    f2p_script: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    repo: str = ""
    log_parser: str = ""
    official_test_spec: Any | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _md_get(md: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in md and md[key] not in (None, ""):
            return md[key]
    return None


def _has_test_cmd(install_config: dict[str, Any]) -> bool:
    test_cmd = install_config.get("test_cmd")
    if isinstance(test_cmd, list):
        return any(str(c).strip() for c in test_cmd)
    return bool(isinstance(test_cmd, str) and test_cmd.strip())


def _test_cmd_str(install_config: dict[str, Any]) -> str:
    test_cmd = install_config.get("test_cmd")
    if isinstance(test_cmd, list):
        return " && ".join(str(c).strip() for c in test_cmd if str(c).strip())
    return str(test_cmd or "").strip()


def env_activation(workdir: str) -> str:
    wd = shlex.quote(workdir)
    return (
        "set +u\n"
        f"export PYTHONPATH={wd}:${{PYTHONPATH:-}}\n"
        "_ACT=0\n"
        "if [ -f /opt/miniconda3/bin/activate ] && [ -d /opt/miniconda3/envs/testbed ]; then\n"
        "  if source /opt/miniconda3/bin/activate 2>/dev/null && conda activate testbed 2>/dev/null; then _ACT=1; fi\n"
        "fi\n"
        'if [ "$_ACT" != "1" ] && [ -f /opt/conda/bin/activate ] && [ -d /opt/conda/envs/testbed ]; then\n'
        "  if source /opt/conda/bin/activate 2>/dev/null && conda activate testbed 2>/dev/null; then _ACT=1; fi\n"
        "fi\n"
        f'if [ "$_ACT" != "1" ] && [ -x {wd}/.venv/bin/python ]; then export PATH={wd}/.venv/bin:$PATH; fi\n'
    )


def detect_eval_mode(md: dict[str, Any]) -> EvalMode:
    pre = normalize_pre_commands(md.get("pre_commands"))
    f2p_script = str(_md_get(md, "f2p_script") or "").strip()
    f2p_patch = str(_md_get(md, "f2p_patch") or "").strip()
    if pre or f2p_script or f2p_patch:
        return EvalMode.SCALESWE

    install_config = coerce_install_config(md.get("install_config"))
    if _has_test_cmd(install_config):
        return EvalMode.REBENCH

    f2p = parse_list(_md_get(md, "FAIL_TO_PASS", "fail_to_pass"))
    p2p = parse_list(_md_get(md, "PASS_TO_PASS", "pass_to_pass"))
    repo = str(_md_get(md, "repo") or "").strip()
    if repo and (f2p or p2p):
        return EvalMode.SWEBENCH

    if str(_md_get(md, "eval_cmd") or "").strip():
        return EvalMode.SIMPLE_CMD
    return EvalMode.NONE


def _pytest_cmd(workdir: str, nodeids: list[str], extra_files: list[str] | None = None) -> str:
    cmd = "python -m pytest -rA --tb=short"
    files = [f for f in (extra_files or []) if f]
    ids = [n for n in nodeids if n]
    targets = files + ids
    if targets:
        cmd += " " + " ".join(shlex.quote(t) for t in targets)
    return f"{env_activation(workdir)}\ncd {shlex.quote(workdir)}\n{cmd}\n"


def resolve_eval_plan(md: dict[str, Any]) -> EvalPlan:
    workdir = str(_md_get(md, "workdir") or "/testbed").strip() or "/testbed"
    mode = detect_eval_mode(md)
    f2p = parse_list(_md_get(md, "FAIL_TO_PASS", "fail_to_pass"))
    p2p = parse_list(_md_get(md, "PASS_TO_PASS", "pass_to_pass"))
    repo = str(_md_get(md, "repo") or "").strip()
    test_patch = str(_md_get(md, "test_patch") or "").strip()
    f2p_patch = str(_md_get(md, "f2p_patch") or "").strip()
    f2p_script = str(_md_get(md, "f2p_script") or "").strip()
    install_config = coerce_install_config(md.get("install_config"))
    log_parser = str(
        install_config.get("log_parser") or _md_get(md, "log_parser") or ""
    ).strip()

    if mode == EvalMode.SCALESWE:
        extras = ["test_fail_to_pass.py"] if f2p_script else []
        return EvalPlan(
            mode=mode,
            eval_cmd=_pytest_cmd(workdir, f2p + p2p, extra_files=extras),
            workdir=workdir,
            f2p_patch=f2p_patch,
            f2p_script=f2p_script,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            repo=repo,
            details={"pre_commands": normalize_pre_commands(md.get("pre_commands"))},
        )

    if mode == EvalMode.REBENCH:
        return EvalPlan(
            mode=mode,
            eval_cmd=f"{env_activation(workdir)}\ncd {shlex.quote(workdir)}\n{_test_cmd_str(install_config)}\n",
            workdir=workdir,
            test_patch=test_patch,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            repo=repo,
            log_parser=log_parser or "parse_log_pytest",
            details={"install_config_keys": sorted(install_config.keys())},
        )

    if mode == EvalMode.SWEBENCH:
        # The official harness owns the repo-specific command, environment
        # activation, test patch, output markers, and parser TestSpec.
        from examples.claudecode_ags.swe_eval.official import make_official_test_spec

        test_spec = make_official_test_spec(md)
        return EvalPlan(
            mode=mode,
            eval_cmd=test_spec.eval_script,
            workdir=workdir,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            repo=repo,
            official_test_spec=test_spec,
            details={
                "version": str(_md_get(md, "version") or ""),
                "runner": "official_swebench",
                "test_patch_owner": "official_eval_script",
            },
        )

    if mode == EvalMode.SIMPLE_CMD:
        return EvalPlan(
            mode=mode,
            eval_cmd=str(_md_get(md, "eval_cmd") or "").strip(),
            workdir=workdir,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            repo=repo,
        )

    return EvalPlan(mode=EvalMode.NONE, workdir=workdir, details={"reason": "missing_eval_plan"})
