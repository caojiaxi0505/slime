"""Sandbox-side CC runtime: workspace prep, toolchain install, claude launch, diff."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.harness.common import install_npm_cli, run_agent

from examples.claudecode_ags.claude_stream_input import initial_prompt_jsonl
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
_INITIAL_PROMPT_PATH = "/tmp/slime_cc_initial_prompt.jsonl"
_RESUME_HANDSHAKE_PATH = "/tmp/slime_cc_resume_handshake.jsonl"
_RESUME_HANDSHAKE = "__SLIME_TOKEN_EXACT_RESUME_HANDSHAKE__"


def _testbed_conda_activation_script() -> str:
    """Activate the task's prebuilt SWE-bench environment for Agent tools.

    SWE-ReX prepends its own Python environment to ``PATH``. Without an
    explicit activation, Bash calls made by Claude Code resolve ``python`` to
    ``/nix/swerex/venv/bin/python`` instead of the task image's ``testbed``
    environment.
    """
    return (
        "# --- Activate task test environment ---\n"
        "set +u\n"
        "_SLIME_TESTBED_ACTIVE=0\n"
        "if [ -f /opt/miniconda3/bin/activate ] "
        "&& [ -d /opt/miniconda3/envs/testbed ]; then\n"
        "    if source /opt/miniconda3/bin/activate 2>/dev/null "
        "&& conda activate testbed 2>/dev/null; then\n"
        "        _SLIME_TESTBED_ACTIVE=1\n"
        "    fi\n"
        "fi\n"
        'if [ "$_SLIME_TESTBED_ACTIVE" != "1" ] '
        "&& [ -f /opt/conda/bin/activate ] "
        "&& [ -d /opt/conda/envs/testbed ]; then\n"
        "    if source /opt/conda/bin/activate 2>/dev/null "
        "&& conda activate testbed 2>/dev/null; then\n"
        "        _SLIME_TESTBED_ACTIVE=1\n"
        "    fi\n"
        "fi\n"
        'if [ "$_SLIME_TESTBED_ACTIVE" != "1" ] '
        "&& [ -d /opt/miniconda3/bin ]; then\n"
        "    export PATH=/opt/miniconda3/bin:$PATH\n"
        'elif [ "$_SLIME_TESTBED_ACTIVE" != "1" ] '
        "&& [ -d /opt/conda/bin ]; then\n"
        "    export PATH=/opt/conda/bin:$PATH\n"
        "fi\n"
        'if [ "$_SLIME_TESTBED_ACTIVE" != "1" ] '
        '&& [ "$(command -v python 2>/dev/null || true)" = '
        "/nix/swerex/venv/bin/python ] "
        "&& [ -x /usr/local/bin/python ]; then\n"
        "    export PATH=/usr/local/bin:$PATH\n"
        "fi\n"
        "unset _SLIME_TESTBED_ACTIVE\n"
        "# --- End task test environment activation ---\n"
    )


def _initial_input_mode() -> str:
    mode = (os.environ.get("SLIME_CC_INITIAL_INPUT_MODE") or "stream-json").strip().lower()
    if mode not in {"stream-json", "positional"}:
        raise ValueError(
            "SLIME_CC_INITIAL_INPUT_MODE must be 'stream-json' or 'positional', "
            f"got {mode!r}"
        )
    return mode


def _extra_claude_args() -> str:
    raw = (os.environ.get("SLIME_CC_EXTRA_ARGS_JSON") or "").strip()
    if not raw:
        return ""
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("SLIME_CC_EXTRA_ARGS_JSON must be a JSON string array") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("SLIME_CC_EXTRA_ARGS_JSON must be a JSON string array")
    return " ".join(shlex.quote(value) for value in values)


async def _run_claude_command(
    sb,
    *,
    workdir: str,
    env: dict[str, str],
    time_budget_sec: int,
    cmd: str,
) -> dict:
    cmd = _testbed_conda_activation_script() + cmd
    exit_code = await run_agent(
        sb,
        workdir=workdir,
        start_cmd=cmd,
        env=env,
        time_budget_sec=time_budget_sec,
    )
    trajectory_path = f"{workdir}/.harness/trajectory.jsonl"
    trajectory = await sb.read_file(trajectory_path, user="agent")
    return {
        "exit_code": exit_code,
        "trajectory_path": trajectory_path,
        "trajectory_jsonl": str(trajectory or ""),
    }


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


async def ensure_claude_home_writable(sb) -> None:
    """Create Claude Code state directories with ownership matching its user."""
    await sb.exec(
        "mkdir -p /home/agent/.claude/projects && chown -R agent:agent /home/agent/.claude",
        user="root",
        timeout=60,
        check=True,
    )


async def run_claude(
    sb,
    *,
    workdir: str,
    prompt: str,
    env: dict[str, str],
    time_budget_sec: int,
    claude_session_id: str | None = None,
) -> dict:
    """Start a new task using the configured, auditable Claude input mode."""
    if _initial_input_mode() == "positional":
        session_flag = (
            f"--session-id {shlex.quote(claude_session_id)} "
            if claude_session_id
            else ""
        )
        extra_args = _extra_claude_args()
        cmd = (
            f"{CLAUDE_BIN} -p {shlex.quote(prompt)} {session_flag}{CLAUDE_FLAGS}"
            f"{f' {extra_args}' if extra_args else ''}"
        )
        return await _run_claude_command(
            sb,
            workdir=workdir,
            env=env,
            time_budget_sec=time_budget_sec,
            cmd=cmd,
        )

    await sb.write_file(
        _INITIAL_PROMPT_PATH,
        initial_prompt_jsonl(prompt),
        user="agent",
    )
    return await run_claude_with_prefix(
        sb,
        workdir=workdir,
        env=env,
        time_budget_sec=time_budget_sec,
        prefix_path=_INITIAL_PROMPT_PATH,
        claude_session_id=claude_session_id,
    )


async def run_claude_with_prefix(
    sb,
    *,
    workdir: str,
    env: dict[str, str],
    time_budget_sec: int,
    prefix_path: str,
    claude_session_id: str | None = None,
) -> dict:
    """Run Claude Code from canonical stream-json input events.

    A one-event file starts a new task; a longer prefix warms history before
    taking the next turn for the Stage-2 bridge.
    """
    session_flag = f"--session-id {shlex.quote(claude_session_id)} " if claude_session_id else ""
    extra_args = _extra_claude_args()
    cmd = (
        f"{CLAUDE_BIN} -p {session_flag}--input-format stream-json {CLAUDE_FLAGS} "
        f"{f'{extra_args} ' if extra_args else ''}< {shlex.quote(prefix_path)}"
    )
    return await _run_claude_command(
        sb,
        workdir=workdir,
        env=env,
        time_budget_sec=time_budget_sec,
        cmd=cmd,
    )


async def run_claude_native_resume(
    sb,
    *,
    workdir: str,
    env: dict[str, str],
    time_budget_sec: int,
    session_jsonl: str,
    session_path: str = "/tmp/slime_cc_branch.session.jsonl",
) -> dict:
    """Fork a truncated native Claude Code session and trigger one handshake.

    The handshake is intentionally not model context: the token-exact adapter
    ignores bootstrap system/messages and substitutes the saved checkpoint.
    """
    if not (session_jsonl or "").strip():
        raise RuntimeError("native Claude Code session prefix is empty")
    await sb.write_file(session_path, session_jsonl, user="agent")
    await sb.write_file(
        _RESUME_HANDSHAKE_PATH,
        initial_prompt_jsonl(_RESUME_HANDSHAKE),
        user="agent",
    )
    cmd = (
        f"{CLAUDE_BIN} -p --resume {shlex.quote(session_path)} --fork-session "
        f"--input-format stream-json {CLAUDE_FLAGS} < {shlex.quote(_RESUME_HANDSHAKE_PATH)}"
    )
    result = await _run_claude_command(
        sb,
        workdir=workdir,
        env=env,
        time_budget_sec=time_budget_sec,
        cmd=cmd,
    )
    result["native_resume_path"] = session_path
    return result


async def git_diff(sb, *, workdir: str) -> str:
    cmd = (
        f"cd {workdir} && git add -N . && "
        "git diff HEAD --binary -- . "
        "':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/**'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out
