#!/usr/bin/env python3
"""Real AGS smoke: L0 create/exec, L1 gold-patch evaluate, L2 live Claude Code.

L0/L1 do not call ``generate()`` (no adapter / SGLang required).
L2 requires ``SLIME_ADAPTER_PUBLIC_URL`` reachable from the AGS sandbox.

Examples::

  python -m examples.claudecode_ags.smoke.ags_smoke --level 0 \\
    --dataset-type swebench_verified \\
    --data-path /path/to/test.parquet --instance-id astropy__astropy-12907

  python -m examples.claudecode_ags.smoke.ags_smoke --level 2 \\
    --dataset-type swebench_verified \\
    --data-path /path/to/test.parquet --instance-id astropy__astropy-12907
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from examples.claudecode_ags import agent_runtime
from examples.claudecode_ags.dataset_normalize import normalize_official_row
from examples.claudecode_ags.swe_eval import dispatch as swe_eval_dispatch
from examples.claudecode_ags.workspace_init import initialize_task_workspace, task_fields_from_metadata
from slime.agent.sandbox import make_sandbox


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"env file not found: {p}")
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _row_instance_id(row: dict[str, Any]) -> str:
    return str(row.get("instance_id") or "").strip()


def load_row_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"empty json: {path}")
    # Single object or first jsonl line
    if path.suffix == ".jsonl" or "\n" in text:
        for line in text.splitlines():
            if line.strip():
                return json.loads(line)
        raise RuntimeError(f"no rows in {path}")
    data = json.loads(text)
    if isinstance(data, list):
        if not data:
            raise RuntimeError(f"empty list in {path}")
        return dict(data[0])
    if isinstance(data, dict):
        # Optionally wrapped as {"metadata": {...}}
        if "metadata" in data and isinstance(data["metadata"], dict) and "instance_id" not in data:
            return dict(data["metadata"])
        return data
    raise RuntimeError(f"unsupported json root in {path}")


def load_row_parquet(path: Path, instance_id: str | None) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("reading parquet requires pyarrow; pip install pyarrow or use --row-json") from e

    table = pq.read_table(path)
    cols = table.column_names
    n = table.num_rows
    if n == 0:
        raise RuntimeError(f"empty parquet: {path}")

    def row_at(i: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in cols:
            val = table.column(name)[i].as_py()
            out[name] = val
        return out

    if instance_id:
        want = instance_id.strip()
        for i in range(n):
            row = row_at(i)
            if _row_instance_id(row) == want:
                return row
        raise RuntimeError(f"instance_id not found in {path}: {want}")
    return row_at(0)


def load_official_row(
    *,
    data_path: str | None,
    row_json: str | None,
    instance_id: str | None,
) -> dict[str, Any]:
    if row_json:
        return load_row_json(Path(row_json))
    if not data_path:
        raise RuntimeError("provide --data-path or --row-json")
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"data path not found: {path}")
    if path.suffix in {".json", ".jsonl"}:
        row = load_row_json(path)
        if instance_id and _row_instance_id(row) != instance_id.strip():
            # jsonl: scan
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        cand = json.loads(line)
                        if _row_instance_id(cand) == instance_id.strip():
                            return cand
                raise RuntimeError(f"instance_id not found in {path}: {instance_id}")
            raise RuntimeError(f"instance_id mismatch in {path}")
        return row
    if path.suffix == ".parquet":
        return load_row_parquet(path, instance_id)
    raise RuntimeError(f"unsupported data file type: {path.suffix}")


def prepare_metadata(row: dict[str, Any], dataset_type: str, image_override: str | None) -> dict[str, Any]:
    md = normalize_official_row(dict(row), dataset_type=dataset_type)
    if image_override and image_override.strip():
        md["image"] = image_override.strip()
    if not (md.get("image") or "").strip():
        raise RuntimeError("normalized metadata has empty image; set --image or check dataset_type/fields")
    return md


def _log(msg: str) -> None:
    print(f"[ags-smoke] {msg}", flush=True)


async def run_l0(md: dict[str, Any]) -> int:
    image = md["image"]
    instance_id = md.get("instance_id") or "unknown"
    _log(f"level=0 instance_id={instance_id}")
    _log(f"image={image}")
    t0 = time.monotonic()
    async with make_sandbox(image) as sb:
        _log(f"sandbox_id={getattr(sb, 'sandbox_id', '')} boot_sec={time.monotonic() - t0:.1f}")
        ec, out, err = await sb.exec("echo ok && pwd", user="root", check=False, timeout=120)
        _log(f"exec_exit={ec} stdout={out!r} stderr={err!r}")
        if ec != 0:
            return ec or 1
    _log("L0 OK")
    return 0


async def run_l1(md: dict[str, Any], *, allow_unresolved: bool, eval_timeout: int) -> int:
    image = md["image"]
    instance_id = md.get("instance_id") or "unknown"
    gold = str(md.get("patch") or "").strip()
    if not gold:
        raise RuntimeError("L1 requires official gold patch field 'patch' on the row")

    _log(f"level=1 instance_id={instance_id}")
    _log(f"image={image}")
    _log(f"gold_patch_chars={len(gold)}")

    fields = task_fields_from_metadata(md)
    t0 = time.monotonic()
    async with make_sandbox(image) as sb:
        _log(f"sandbox_id={getattr(sb, 'sandbox_id', '')} boot_sec={time.monotonic() - t0:.1f}")
        ok = await initialize_task_workspace(sb, fields, rollout_side=False)
        if not ok:
            _log("workspace_init FAILED")
            return 2
        _log("workspace_init OK")
        result = await swe_eval_dispatch.evaluate(
            sb,
            metadata=md,
            diff_text=gold,
            timeout_sec=eval_timeout,
        )
        details = result.details or {}
        _log(f"resolved={result.resolved} applied_cleanly={result.applied_cleanly}")
        _log(f"mode={details.get('mode')} exit_code={details.get('exit_code')}")
        reason = details.get("reason")
        if reason:
            _log(f"reason={reason}")
        apply_err = details.get("apply_stderr")
        if apply_err:
            _log(f"apply_stderr={apply_err!r}")
        stdout = str(details.get("stdout") or "")
        if stdout:
            _log(f"stdout_tail={stdout[-2000:]!r}")

    if result.resolved:
        _log("L1 OK (resolved=True)")
        return 0
    if allow_unresolved:
        _log("L1 OK (--allow-unresolved; resolved=False)")
        return 0
    _log("L1 FAILED (resolved=False)")
    return 1


def _cc_env_prefixes() -> tuple[str, ...]:
    return ("ANTHROPIC_", "CLAUDE_", "BASH_")


_CC_ENV_EXACT = frozenset(
    {
        "API_TIMEOUT_MS",
        "API_FORCE_IDLE_TIMEOUT",
        "TASK_MAX_OUTPUT_LENGTH",
        "MAX_MCP_OUTPUT_TOKENS",
        "MAX_THINKING_TOKENS",
        "IS_SANDBOX",
    }
)


def _build_claude_env(*, adapter_url: str, session_id: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(_cc_env_prefixes()) or key in _CC_ENV_EXACT:
            env[key] = value
    env.update(
        {
            "ANTHROPIC_BASE_URL": adapter_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": session_id,
            "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL") or "claude-sonnet",
            "IS_SANDBOX": env.get("IS_SANDBOX") or "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") or "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": env.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") or "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": env.get("CLAUDE_CODE_ATTRIBUTION_HEADER") or "0",
        }
    )
    return env


def _adapter_public_url() -> str:
    url = (os.environ.get("SLIME_ADAPTER_PUBLIC_URL") or "").strip().rstrip("/")
    if not url:
        raise RuntimeError(
            "L2 requires SLIME_ADAPTER_PUBLIC_URL (dedicated ALB URL reachable from AGS). "
            "Deploy Phase 1 then set it in slime_ags.env."
        )
    if "127.0.0.1" in url or "localhost" in url:
        raise RuntimeError(
            f"SLIME_ADAPTER_PUBLIC_URL={url!r} is not reachable from AGS sandboxes; use the ALB DNS"
        )
    return url


def _default_artifact_dir(instance_id: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id) or "unknown"
    return Path(os.environ.get("SLIME_L2_ARTIFACT_DIR") or f"/tmp/ags_smoke_l2_{safe}_{stamp}")


def summarize_claude_stream(text: str, *, max_types: int = 20) -> dict[str, Any]:
    """Summarize Claude Code stream-json / NDJSON lines for smoke logs."""
    type_counts: dict[str, int] = {}
    tool_names: list[str] = []
    errors: list[str] = []
    lines_total = 0
    lines_json = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lines_total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        lines_json += 1
        if not isinstance(obj, dict):
            continue
        t = str(obj.get("type") or obj.get("event") or "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        # Common Claude Code stream shapes
        name = obj.get("name")
        tool_use = obj.get("tool_use")
        if not name and isinstance(tool_use, dict):
            name = tool_use.get("name")
        if not name and isinstance(obj.get("message"), dict):
            for block in obj["message"].get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
                    tool_names.append(str(block["name"]))
        if name:
            tool_names.append(str(name))
        if t == "result" and isinstance(obj, dict):
            subtype = obj.get("subtype")
            if subtype:
                type_counts[f"result:{subtype}"] = type_counts.get(f"result:{subtype}", 0) + 1
            if obj.get("is_error"):
                errors.append(str(obj.get("result") or obj.get("error") or "is_error")[:200])
        if t == "error":
            err = obj.get("error") or obj.get("message")
            if isinstance(err, str):
                errors.append(err[:200])
    # Dedupe tools preserving order
    seen: set[str] = set()
    tools_unique: list[str] = []
    for n in tool_names:
        if n not in seen:
            seen.add(n)
            tools_unique.append(n)
    top_types = sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_types]
    return {
        "lines_total": lines_total,
        "lines_json": lines_json,
        "type_counts": dict(top_types),
        "tools": tools_unique[:50],
        "errors": errors[:10],
    }


async def export_rollout_artifacts(
    sb,
    *,
    workdir: str,
    dest_dir: Path,
    diff_text: str,
    session_id: str,
    instance_id: str,
) -> Path:
    """Copy Claude harness artifacts out of the rollout sandbox before teardown."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "instance_id": instance_id,
        "session_id": session_id,
        "workdir": workdir,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diff_chars": len(diff_text),
    }

    (dest / "model.diff").write_text(diff_text, encoding="utf-8")

    harness_dir = f"{workdir.rstrip('/')}/.harness"
    _, listing, _ = await sb.exec(f"ls -la {shlex.quote(harness_dir)} 2>/dev/null || true", user="root", timeout=30, check=False)
    (dest / "harness_ls.txt").write_text(listing or "", encoding="utf-8")

    traj_path = f"{harness_dir}/trajectory.jsonl"
    traj_text = ""
    try:
        traj_text = await sb.read_file(traj_path, user="root")
    except Exception as e:
        meta["trajectory_error"] = str(e)
        # Fallback: cat via exec
        ec, out, err = await sb.exec(f"cat {shlex.quote(traj_path)} 2>/dev/null", user="root", timeout=120, check=False)
        if ec == 0:
            traj_text = out or ""
        else:
            meta["trajectory_cat_stderr"] = (err or "")[:500]

    (dest / "trajectory.jsonl").write_text(traj_text, encoding="utf-8")
    summary = summarize_claude_stream(traj_text)
    meta["trajectory_summary"] = summary
    (dest / "trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # git status for quick "did anything change?" signal
    _, status_out, _ = await sb.exec(
        f"cd {shlex.quote(workdir)} && git status --short 2>/dev/null || true",
        user="agent",
        timeout=60,
        check=False,
    )
    (dest / "git_status.txt").write_text(status_out or "", encoding="utf-8")
    meta["git_status_lines"] = len([ln for ln in (status_out or "").splitlines() if ln.strip()])

    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"artifacts_dir={dest}")
    _log(
        f"trajectory_lines={summary['lines_json']}/{summary['lines_total']} "
        f"tools={summary['tools'][:8]} types={list(summary['type_counts'].keys())[:8]}"
    )
    if summary["errors"]:
        _log(f"trajectory_errors={summary['errors'][:3]}")
    return dest


async def run_l2(
    md: dict[str, Any],
    *,
    allow_unresolved: bool,
    eval_timeout: int,
    time_budget: int,
    skip_health_gate: bool,
    artifact_dir: Path | None = None,
) -> int:
    """Live Claude Code → public adapter → dual-sandbox eval (model diff, not gold)."""
    image = md["image"]
    instance_id = md.get("instance_id") or "unknown"
    workdir = str(md.get("workdir") or "/testbed").strip() or "/testbed"
    problem = str(md.get("problem_statement") or md.get("problem") or "").strip()
    if not problem:
        raise RuntimeError("L2 requires problem_statement (or problem) on the normalized row")

    adapter_url = _adapter_public_url()
    session_id = f"l2-{instance_id}-{int(time.time())}"
    prompt = str(md.get("agent_prompt") or "").strip() or (
        "Read PROBLEM_STATEMENT.md and fix the issue. Keep changes minimal."
    )

    _log(f"level=2 instance_id={instance_id}")
    _log(f"image={image}")
    _log(f"adapter={adapter_url}")
    _log(f"time_budget_sec={time_budget} eval_timeout={eval_timeout}")

    # --- Sandbox A: toolchain + Claude Code ---
    t0 = time.monotonic()
    async with make_sandbox(image) as sb:
        _log(f"rollout_sandbox_id={getattr(sb, 'sandbox_id', '')} boot_sec={time.monotonic() - t0:.1f}")
        if not skip_health_gate:
            health_url = f"{adapter_url}/health"
            cmd = (
                f"curl -fsS --max-time 30 {shlex.quote(health_url)} "
                f"|| curl -fsS --max-time 30 {shlex.quote(adapter_url + '/healthz')}"
            )
            ec, out, err = await sb.exec(cmd, user="root", check=False, timeout=60)
            _log(f"health_gate exit={ec} stdout={out!r} stderr={err!r}")
            if ec != 0:
                _log("L2 FAILED: adapter health unreachable from sandbox")
                return 2

        try:
            await agent_runtime.prepare_workspace(
                sb,
                workdir=workdir,
                problem_statement=problem,
                instance_id=str(md.get("instance_id") or ""),
                data_source=str(md.get("data_source") or ""),
                base_commit=str(md.get("base_commit") or ""),
                swe_smith_bug_patch=md.get("swe_smith_bug_patch"),
                pre_commands=md.get("pre_commands") or "",
                install_config=md.get("install_config") or {},
                rollout_side=True,
            )
        except RuntimeError as e:
            _log(f"workspace_init FAILED: {e}")
            return 2
        _log("workspace_init OK")

        await agent_runtime.install_toolchain(sb)
        _log("toolchain OK")

        claude_env = _build_claude_env(adapter_url=adapter_url, session_id=session_id)
        agent_result = await agent_runtime.run_claude(
            sb,
            workdir=workdir,
            prompt=prompt,
            env=claude_env,
            time_budget_sec=time_budget,
        )
        _log(f"claude_exit={agent_result.get('exit_code')}")
        diff_text = await agent_runtime.git_diff(sb, workdir=workdir)
        _log(f"diff_chars={len(diff_text)}")

        # Export before sandbox A teardown (trajectory lives only inside the sandbox).
        out_dir = artifact_dir or _default_artifact_dir(str(instance_id))
        await export_rollout_artifacts(
            sb,
            workdir=workdir,
            dest_dir=out_dir,
            diff_text=diff_text,
            session_id=session_id,
            instance_id=str(instance_id),
        )

    # --- Sandbox B: eval workspace init (light) + evaluate model diff (never gold) ---
    t1 = time.monotonic()
    fields = task_fields_from_metadata(md)
    fields.workdir = workdir or fields.workdir
    async with make_sandbox(image) as sb_eval:
        _log(f"eval_sandbox_id={getattr(sb_eval, 'sandbox_id', '')} boot_sec={time.monotonic() - t1:.1f}")
        ok = await initialize_task_workspace(sb_eval, fields, rollout_side=False)
        if not ok:
            _log("eval_workspace_init FAILED")
            return 2
        _log("eval_workspace_init OK")
        result = await swe_eval_dispatch.evaluate(
            sb_eval,
            metadata=md,
            diff_text=diff_text,
            timeout_sec=eval_timeout,
        )
        details = result.details or {}
        _log(f"resolved={result.resolved} applied_cleanly={result.applied_cleanly}")
        _log(f"mode={details.get('mode')} exit_code={details.get('exit_code')}")
        reason = details.get("reason")
        if reason:
            _log(f"reason={reason}")

    if result.resolved:
        _log("L2 OK (link+resolved=True)")
        return 0
    if allow_unresolved:
        _log("L2 OK (--allow-unresolved; link passed, resolved=False)")
        return 0
    _log("L2 link OK but unresolved (exit 1); use --allow-unresolved to treat as pass")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real AGS smoke (L0/L1/L2)")
    p.add_argument("--level", type=int, choices=(0, 1, 2), required=True, help="0=boot, 1=gold eval, 2=live CC")
    p.add_argument("--dataset-type", default="swebench_verified", help="normalize dataset_type")
    p.add_argument("--data-path", default="", help="official parquet/json/jsonl path")
    p.add_argument("--row-json", default="", help="single-row json/jsonl instead of --data-path")
    p.add_argument("--instance-id", default="", help="select row by instance_id")
    p.add_argument("--image", default="", help="override normalized image")
    p.add_argument("--env-file", default="", help="KEY=VALUE file loaded before run")
    p.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="L1/L2: exit 0 even if resolved=False (L2: still requires link success)",
    )
    p.add_argument("--eval-timeout", type=int, default=int(os.environ.get("SLIME_CC_EVAL_TIMEOUT_SEC") or 600))
    p.add_argument(
        "--time-budget",
        type=int,
        default=int(os.environ.get("SLIME_CC_TIME_BUDGET_SEC") or 1800),
        help="L2 Claude Code wall time budget (seconds)",
    )
    p.add_argument(
        "--skip-health-gate",
        action="store_true",
        help="L2: skip curl of SLIME_ADAPTER_PUBLIC_URL/health inside sandbox",
    )
    p.add_argument(
        "--artifact-dir",
        default="",
        help="L2: host dir for trajectory.jsonl / model.diff before sandbox teardown "
        "(default: /tmp/ags_smoke_l2_<instance>_<ts> or SLIME_L2_ARTIFACT_DIR)",
    )
    return p


async def async_main(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file, override=False)
    # Prefer AGS for this smoke unless already set.
    os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "ags")

    row = load_official_row(
        data_path=args.data_path or None,
        row_json=args.row_json or None,
        instance_id=args.instance_id or None,
    )
    md = prepare_metadata(row, args.dataset_type, args.image or None)
    if args.level == 0:
        return await run_l0(md)
    if args.level == 1:
        return await run_l1(md, allow_unresolved=args.allow_unresolved, eval_timeout=args.eval_timeout)
    art = Path(args.artifact_dir) if str(args.artifact_dir or "").strip() else None
    return await run_l2(
        md,
        allow_unresolved=args.allow_unresolved,
        eval_timeout=args.eval_timeout,
        time_budget=args.time_budget,
        skip_health_gate=args.skip_health_gate,
        artifact_dir=art,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except Exception as e:
        _log(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
