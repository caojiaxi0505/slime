"""Sandbox-side CC runtime: workspace prep, toolchain install, claude launch, diff."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from slime.agent import sandbox as agent_sandbox
from slime.agent.harness.common import install_npm_cli, run_agent

CLAUDE_BIN = "/usr/local/bin/claude"
CLAUDE_FLAGS = (
    "--permission-mode bypassPermissions "
    "--output-format stream-json --include-partial-messages "
    "--include-hook-events --verbose"
)


async def prepare_workspace(sb, *, workdir: str, problem_statement: str) -> None:
    await agent_sandbox.ensure_agent_user(sb, workdir)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        problem_statement,
        user="agent",
    )


async def install_toolchain(sb) -> None:
    """Install Node + Claude Code when ``SLIME_AGENT_TOOLCHAIN_MODE=tarball``.

  Required env (see ``env/slime_ags.env``):
    - ``SLIME_AGENT_TOOLCHAIN_MODE``: ``skip`` (default) or ``tarball``
    - ``SLIME_AGENT_NODE_TARBALL``: host path to Node 22 tarball (tarball mode)
    - ``SLIME_AGENT_CC_TARBALL``: host path to Claude Code npm tarball (tarball mode)

  COS-based install (``SLIME_AGENT_COS_*``) is not implemented here yet.
    """
    mode = (os.environ.get("SLIME_AGENT_TOOLCHAIN_MODE") or "skip").strip().lower()
    if mode in ("", "skip", "none"):
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


async def git_diff(sb, *, workdir: str) -> str:
    cmd = (
        f"cd {workdir} && git add -N . && "
        "git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out
