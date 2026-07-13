"""Sandbox-side CC runtime: workspace prep, toolchain install, claude launch, diff."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from slime.agent.harness.common import install_npm_cli, run_agent

from examples.claudecode_ags.workspace_init import TaskFields, initialize_task_workspace

CLAUDE_BIN = "/usr/local/bin/claude"
CLAUDE_FLAGS = (
    "--permission-mode bypassPermissions "
    "--output-format stream-json --include-partial-messages "
    "--include-hook-events --verbose"
)

_DEFAULT_COS_MOUNT = "/mnt/code_agent"
_DEFAULT_COS_NODE_PACKAGE = "node-v20.18.1-linux-x64.tar.xz"
_DEFAULT_COS_CC_PACKAGE = "cc-prefix-2.1.104-linux-x64.tar.gz"
_DEFAULT_COS_NODE_DIR = "/opt/node-cos"
_DEFAULT_COS_CC_DIR = "/opt/cc-cos"


async def prepare_workspace(
    sb,
    *,
    workdir: str,
    problem_statement: str,
    instance_id: str = "",
    data_source: str = "",
    base_commit: str = "",
    swe_smith_bug_patch: str | None = None,
    pre_commands: str | list[str] | None = None,
    install_config: dict | str | None = None,
    rollout_side: bool = True,
) -> None:
    """Init buggy baseline then write PROBLEM_STATEMENT.md."""
    from examples.claudecode_ags.workspace_init import coerce_install_config, normalize_pre_commands

    fields = TaskFields(
        instance_id=instance_id,
        data_source=data_source,
        workdir=workdir,
        base_commit=base_commit,
        swe_smith_bug_patch=swe_smith_bug_patch,
        pre_commands=normalize_pre_commands(pre_commands),
        install_config=coerce_install_config(install_config),
    )
    ok = await initialize_task_workspace(sb, fields, rollout_side=rollout_side)
    if not ok:
        raise RuntimeError(
            f"workspace init failed mode={fields.data_source or fields.base_commit or fields.pre_commands[:40]!r} "
            f"instance_id={instance_id!r}"
        )
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        problem_statement,
        user="agent",
    )


async def _install_cos_toolchain(sb) -> None:
    """Untar Node + Claude Code from an AGS-mounted COS directory."""
    mount = (os.environ.get("SLIME_AGENT_COS_MOUNT") or _DEFAULT_COS_MOUNT).strip()
    node_pkg = (os.environ.get("SLIME_AGENT_COS_NODE_PACKAGE") or _DEFAULT_COS_NODE_PACKAGE).strip()
    cc_pkg = (os.environ.get("SLIME_AGENT_COS_CC_PACKAGE") or _DEFAULT_COS_CC_PACKAGE).strip()
    node_dir = (os.environ.get("SLIME_AGENT_COS_NODE_DIR") or _DEFAULT_COS_NODE_DIR).strip()
    cc_dir = (os.environ.get("SLIME_AGENT_COS_CC_DIR") or _DEFAULT_COS_CC_DIR).strip()
    node_pkg_path = shlex.quote(f"{mount.rstrip('/')}/{node_pkg}")
    cc_pkg_path = shlex.quote(f"{mount.rstrip('/')}/{cc_pkg}")
    node_dir_q = shlex.quote(node_dir)
    cc_dir_q = shlex.quote(cc_dir)
    await sb.exec(
        "set -euxo pipefail && "
        f"test -r {node_pkg_path} && test -r {cc_pkg_path} && "
        f"rm -rf {node_dir_q} {cc_dir_q} && mkdir -p {node_dir_q} {cc_dir_q} && "
        f"tar -xJf {node_pkg_path} -C {node_dir_q} --strip-components=1 && "
        f"tar -xzf {cc_pkg_path} -C {cc_dir_q} && "
        f"ln -sf {shlex.quote(node_dir)}/bin/node /usr/local/bin/node && "
        f"ln -sf {shlex.quote(node_dir)}/bin/npm /usr/local/bin/npm && "
        f"ln -sf {shlex.quote(node_dir)}/bin/npx /usr/local/bin/npx && "
        f"ln -sf {shlex.quote(cc_dir)}/bin/claude {shlex.quote(CLAUDE_BIN)} && "
        "hash -r 2>/dev/null || true && node --version && npm --version && claude --version",
        user="root",
        timeout=180,
        check=True,
    )


async def install_toolchain(sb) -> None:
    """Install Node + Claude Code into the sandbox.

    Modes (``SLIME_AGENT_TOOLCHAIN_MODE``):
      - ``skip`` / empty: no-op
      - ``cos`` (default for cluster runs): untar from mounted COS
      - ``tarball``: host paths via ``SLIME_AGENT_NODE_TARBALL`` / ``SLIME_AGENT_CC_TARBALL``
    """
    mode = (os.environ.get("SLIME_AGENT_TOOLCHAIN_MODE") or "cos").strip().lower()
    if mode in ("", "skip", "none"):
        return
    if mode == "cos":
        await _install_cos_toolchain(sb)
        return
    if mode == "tarball":
        node_tar = os.environ.get("SLIME_AGENT_NODE_TARBALL", "").strip()
        cc_tar = os.environ.get("SLIME_AGENT_CC_TARBALL", "").strip()
        if not node_tar or not cc_tar:
            raise RuntimeError(
                "SLIME_AGENT_TOOLCHAIN_MODE=tarball requires "
                "SLIME_AGENT_NODE_TARBALL and SLIME_AGENT_CC_TARBALL"
            )
        await install_npm_cli(
            sb,
            node_runtime=Path(node_tar),
            npm_package=Path(cc_tar),
            check_cmd=f"ls -la {CLAUDE_BIN} && {CLAUDE_BIN} --version",
        )
        return
    raise ValueError(f"Unknown SLIME_AGENT_TOOLCHAIN_MODE: {mode!r}")


async def run_claude(
    sb,
    *,
    workdir: str,
    prompt: str,
    env: dict[str, str],
    time_budget_sec: int,
) -> dict:
    """Run ``claude -p`` with exactly the caller-provided ``env`` dict."""
    cmd = f"{CLAUDE_BIN} -p {shlex.quote(prompt)} {CLAUDE_FLAGS}"
    exit_code = await run_agent(
        sb,
        workdir=workdir,
        start_cmd=cmd,
        env=env,
        time_budget_sec=time_budget_sec,
    )
    return {"exit_code": exit_code}


async def run_claude_with_prefix(
    sb,
    *,
    workdir: str,
    prompt: str,
    env: dict[str, str],
    time_budget_sec: int,
    prefix_path: str,
) -> dict:
    """Prefix re-seed: feed stream-json transcript then continue with ``-p``.

    Uses ``--input-format stream-json`` so Claude Code warms history from
    ``prefix_path`` before taking the next turn (Stage-2 bridge).
    """
    cmd = (
        f"{CLAUDE_BIN} -p {shlex.quote(prompt)} "
        f"--input-format stream-json {CLAUDE_FLAGS} "
        f"< {shlex.quote(prefix_path)}"
    )
    exit_code = await run_agent(
        sb,
        workdir=workdir,
        start_cmd=cmd,
        env=env,
        time_budget_sec=time_budget_sec,
    )
    return {"exit_code": exit_code}


async def git_diff(sb, *, workdir: str) -> str:
    cmd = (
        f"cd {workdir} && git add -N . && "
        "git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out
