"""Dataset-aware SWE workspace init (phase 1–2)."""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from slime.agent import sandbox as agent_sandbox

logger = logging.getLogger(__name__)

_SCRIPT = "/tmp/slime_ws_init.sh"
_PATCH = "/tmp/slime_ws_patch.diff"
_SCRUB_MARKER = "slime_git_scrub"
_SCRUB_EVAL_MARKER = "slime_git_scrub_eval"
_SWEBENCH_GIT_SETUP_MARKER = "slime_swebench_git_setup"
_SCRUB_COMMIT_DATE = "2000-01-01T00:00:00+0000"
_ESP_IDF_EXAMPLES = "/workspace/esp-idf/examples"
_HUNK_RE = re.compile(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)")


class WorkspaceMode(str, Enum):
    SWESMITH = "swesmith"
    SCALESWE = "scaleswe"
    SWEBENCH_CLASSIC = "swebench_classic"
    REBENCH = "rebench"
    GENERIC = "generic"


@dataclass
class TaskFields:
    instance_id: str = ""
    data_source: str = ""
    workdir: str = "/testbed"
    base_commit: str = ""
    swe_smith_bug_patch: str | None = None
    pre_commands: str = ""
    install_config: dict[str, Any] = field(default_factory=dict)


def is_swesmith_data_source(data_source: Any) -> bool:
    return str(data_source or "").strip().lower().startswith("swe_smith")


def is_synthetic_instance_id(instance_id: str) -> bool:
    return str(instance_id or "").strip().startswith("_synthetic_row_")


def coerce_install_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_pre_commands(pre: list[str] | str | None) -> str:
    if pre is None:
        return ""
    if isinstance(pre, list):
        body = "\n".join(str(c) for c in pre if c)
    else:
        body = str(pre)
    return body.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").rstrip()


def _has_test_cmd(install_config: dict[str, Any]) -> bool:
    test_cmd = install_config.get("test_cmd")
    if isinstance(test_cmd, list):
        return any(str(c).strip() for c in test_cmd)
    return bool(isinstance(test_cmd, str) and test_cmd.strip())


def detect_workspace_mode(fields: TaskFields) -> WorkspaceMode:
    if is_swesmith_data_source(fields.data_source):
        return WorkspaceMode.SWESMITH
    if fields.pre_commands.strip():
        return WorkspaceMode.SCALESWE
    if fields.base_commit:
        return WorkspaceMode.SWEBENCH_CLASSIC
    if _has_test_cmd(fields.install_config):
        return WorkspaceMode.REBENCH
    return WorkspaceMode.GENERIC


def task_fields_from_metadata(md: dict[str, Any]) -> TaskFields:
    swebench = md.get("swebench") if isinstance(md.get("swebench"), dict) else {}
    base_commit = str(md.get("base_commit") or swebench.get("base_commit") or "").strip()
    install_config = coerce_install_config(md.get("install_config"))
    return TaskFields(
        instance_id=str(md.get("instance_id") or ""),
        data_source=str(md.get("data_source") or ""),
        workdir=str(md.get("workdir") or "/testbed"),
        base_commit=base_commit,
        swe_smith_bug_patch=md.get("swe_smith_bug_patch"),
        pre_commands=normalize_pre_commands(md.get("pre_commands")),
        install_config=install_config,
    )


def build_git_scrub_command(workdir: str) -> str:
    """Synthetic-dataset scrub: hide all history behind an orphan commit.

    SWE-bench Classic uses :func:`build_swebench_git_setup_command` instead so
    the agent sees the real base commit and its ancestors, as in the official
    SWE-bench harness.
    """
    wd = shlex.quote(workdir)
    return f"""\
# ---- {_SCRUB_MARKER} --------------------------------------------
if [ -d {wd}/.git ]; then
    (
        cd {wd} || exit 0
        for _remote in $(git remote 2>/dev/null); do
            git remote remove "$_remote" 2>/dev/null || true
        done
        git for-each-ref --format='%(refname)' \\
            refs/heads/ refs/remotes/ refs/tags/ 2>/dev/null \\
            | while read -r _ref; do
                git update-ref -d "$_ref" 2>/dev/null || true
            done
        git checkout --orphan __slime_buggy 2>/dev/null || true
        git -c user.email=slime@local -c user.name=slime add -A 2>/dev/null || true
        # Claude Code exposes this synthetic commit in its system prompt. Fix
        # both Git timestamps so identical trees produce identical prompt text.
        GIT_AUTHOR_DATE={shlex.quote(_SCRUB_COMMIT_DATE)} \\
        GIT_COMMITTER_DATE={shlex.quote(_SCRUB_COMMIT_DATE)} \\
        git -c user.email=slime@local -c user.name=slime -c commit.gpgSign=false \\
            commit --allow-empty -q -m 'slime initial bug state' 2>/dev/null || true
        git reflog expire --expire=now --all 2>/dev/null || true
        git gc --prune=now --aggressive >/dev/null 2>&1 || true
    )
fi
# ---- {_SCRUB_MARKER} end -----------------------------------------
"""


def build_swebench_git_setup_command(workdir: str, base_commit: str) -> str:
    """Prepare SWE-bench Git state while retaining base history.

    This mirrors the official harness's observable repository state: HEAD is a
    normal ``SWE-bench`` child of the real base commit, the base's ancestors
    remain visible, and remotes/future refs are removed.  The setup commit uses
    the base commit's timestamp so repeated rollouts also get the same HEAD.
    """
    wd = shlex.quote(workdir)
    base = shlex.quote(base_commit)
    return f"""\
# ---- {_SWEBENCH_GIT_SETUP_MARKER} --------------------------------
set -o pipefail
if [ ! -d {wd}/.git ]; then
    echo "[workspace_init][swebench] missing Git repository: {wd}" >&2
    exit 1
fi
cd {wd}
_target_ref={base}
_target_commit=$(git rev-parse --verify "${{_target_ref}}^{{commit}}")
_current_head=$(git rev-parse --verify HEAD)
if [ "$_current_head" != "$_target_commit" ]; then
    echo "[workspace_init][swebench] HEAD is not base_commit" >&2
    exit 1
fi
_target_timestamp=$(git show -s --format=%ct "$_target_commit")
_setup_date=$(git show -s --format=%cI "$_target_commit")
_current_branch=$(git symbolic-ref -q --short HEAD 2>/dev/null || true)

# The official harness starts from a single-branch clone and removes origin.
# Runtime images may contain additional local/remote refs, so remove those
# explicitly while retaining the checked-out branch and non-future tags.
for _remote in $(git remote 2>/dev/null); do
    git remote remove "$_remote"
done
git for-each-ref --format='%(refname)' refs/remotes/ 2>/dev/null \\
    | while read -r _ref; do
        git update-ref -d "$_ref"
    done
git for-each-ref --format='%(refname)' refs/heads/ 2>/dev/null \\
    | while read -r _ref; do
        if [ -z "$_current_branch" ] || [ "$_ref" != "refs/heads/$_current_branch" ]; then
            git update-ref -d "$_ref"
        fi
    done
git for-each-ref --format='%(refname)' 2>/dev/null \\
    | while read -r _ref; do
        case "$_ref" in
            refs/heads/*|refs/remotes/*|refs/tags/*) ;;
            *) git update-ref -d "$_ref" ;;
        esac
    done
git for-each-ref --format='%(refname)' refs/tags/ 2>/dev/null \\
    | while read -r _ref; do
        _tag_commit=$(git rev-parse --verify "${{_ref}}^{{commit}}" 2>/dev/null || true)
        if [ -n "$_tag_commit" ]; then
            _tag_timestamp=$(git show -s --format=%ct "$_tag_commit")
            if [ "$_tag_timestamp" -gt "$_target_timestamp" ]; then
                git update-ref -d "$_ref"
            fi
        fi
    done
git reflog expire --expire=now --all
git gc --prune=now --aggressive >/dev/null 2>&1

_future_commit=$(
    git rev-list --all --timestamp \\
        | awk -v cutoff="$_target_timestamp" '$1 > cutoff {{ print $2; exit }}'
)
if [ -n "$_future_commit" ]; then
    echo "[workspace_init][swebench] future commit remains reachable: $_future_commit" >&2
    exit 1
fi

git config user.email 'setup@swebench.config'
git config user.name 'SWE-bench'
git config commit.gpgSign false
GIT_AUTHOR_NAME='SWE-bench' \\
GIT_AUTHOR_EMAIL='setup@swebench.config' \\
GIT_COMMITTER_NAME='SWE-bench' \\
GIT_COMMITTER_EMAIL='setup@swebench.config' \\
GIT_AUTHOR_DATE="$_setup_date" \\
GIT_COMMITTER_DATE="$_setup_date" \\
git -c commit.gpgSign=false commit --allow-empty --no-verify --no-gpg-sign \\
    -q -am 'SWE-bench'

if [ "$(git rev-parse --verify HEAD^)" != "$_target_commit" ]; then
    echo "[workspace_init][swebench] setup commit parent is not base_commit" >&2
    exit 1
fi
# ---- {_SWEBENCH_GIT_SETUP_MARKER} end ----------------------------
"""


def build_eval_git_scrub_command(workdir: str) -> str:
    """Eval setup: keep the image's original Git graph and version tags intact.

    The evaluator runs in a separate sandbox that is never exposed to the
    coding agent. Rewriting its refs therefore provides no isolation benefit,
    while moving release tags changes versions derived by ``setuptools_scm``
    and can make the official tests exercise a different package version.
    """
    wd = shlex.quote(workdir)
    return f"""\
# ---- {_SCRUB_EVAL_MARKER} ----------------------------------------
cd {wd} || exit 0
git config user.email 'slime@local' 2>/dev/null || true
git config user.name 'slime' 2>/dev/null || true
# ---- {_SCRUB_EVAL_MARKER} end ------------------------------------
"""


async def _exec_script(sb, workdir: str, body: str, *, fail_fast: bool = False, timeout: int = 600) -> int:
    prefix = "set -e\n" if fail_fast else "set +e\n"
    await sb.write_file(_SCRIPT, prefix + body, user="agent")
    ec, _, _ = await sb.exec(
        f"chmod 755 {_SCRIPT} && cd {shlex.quote(workdir)} && bash {_SCRIPT}",
        user="agent",
        check=False,
        timeout=timeout,
    )
    return ec


async def apply_swebench_reset(sb, workdir: str, base_commit: str) -> bool:
    if not base_commit:
        return True
    script = f"cd {shlex.quote(workdir)} && git reset --hard {shlex.quote(base_commit)}"
    return await _exec_script(sb, workdir, script, fail_fast=True) == 0


async def apply_swebench_git_setup(sb, workdir: str, base_commit: str) -> bool:
    body = build_swebench_git_setup_command(workdir, base_commit)
    return await _exec_script(sb, workdir, body, fail_fast=True) == 0


async def apply_git_scrub(sb, workdir: str, *, rollout_side: bool = True) -> None:
    body = build_git_scrub_command(workdir) if rollout_side else build_eval_git_scrub_command(workdir)
    await _exec_script(sb, workdir, body, fail_fast=False)


async def apply_swesmith_branch(sb, workdir: str, instance_id: str) -> bool:
    if is_synthetic_instance_id(instance_id):
        return False
    branch = shlex.quote(instance_id)
    wd = shlex.quote(workdir)
    script = (
        f"cd {wd}\n"
        "git fetch origin 2>/dev/null || git fetch 2>/dev/null || true\n"
        f"git checkout {branch} 2>/dev/null || git checkout -b {branch} 2>/dev/null || true\n"
        "git checkout HEAD~1 2>/dev/null || true\n"
    )
    return await _exec_script(sb, workdir, script, fail_fast=False) == 0


def build_scaleswe_pre_commands_block(pre_commands: str) -> str:
    if not pre_commands.strip():
        return ""
    delim = "EOF_SCALESWE_PRE_CMDS_114329324912"
    return "\n".join(
        [
            "# Scale-SWE: restore canonical git state before agent/eval",
            f"if ! /bin/bash <<'{delim}'",
            "set -eo pipefail",
            pre_commands,
            delim,
            "then",
            '    echo "[workspace_init][scaleswe] pre_commands failed" >&2',
            "    exit 1",
            "fi",
        ]
    )


async def apply_scaleswe_pre_commands(sb, workdir: str, pre_commands: str) -> bool:
    body = normalize_pre_commands(pre_commands)
    if not body:
        return True
    wd = shlex.quote(workdir)
    block = build_scaleswe_pre_commands_block(body)
    script = "\n".join([f"cd {wd}", f"git config --global --add safe.directory {wd}", block])
    return await _exec_script(sb, workdir, script, fail_fast=True) == 0


async def scaleswe_root_filesystem_prep(sb) -> None:
    await sb.exec(
        f"[ -d {_ESP_IDF_EXAMPLES} ] && chmod -R a+w {_ESP_IDF_EXAMPLES} 2>/dev/null || true",
        user="root",
        check=False,
        timeout=120,
    )


def _split_patch_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def reverse_patch(patch: str) -> str:
    """Reverse a unified diff (SWE-Smith clean→mutated patches)."""
    if not patch:
        return ""
    out: list[str] = []
    for line in patch.splitlines(keepends=True):
        body, nl = _split_patch_newline(line)
        if body.startswith("diff --git "):
            out.append(line)
        elif body.startswith("new file mode "):
            out.append("deleted file mode " + body[len("new file mode ") :] + nl)
        elif body.startswith("deleted file mode "):
            out.append("new file mode " + body[len("deleted file mode ") :] + nl)
        elif body.startswith("rename from "):
            out.append("rename to " + body[len("rename from ") :] + nl)
        elif body.startswith("rename to "):
            out.append("rename from " + body[len("rename to ") :] + nl)
        elif body.startswith("--- a/") or body.startswith("--- b/"):
            path = body.split("/", 1)[1] if "/" in body else body[6:]
            out.append(f"--- a/{path}{nl}")
        elif body.startswith("+++ b/") or body.startswith("+++ a/"):
            path = body.split("/", 1)[1] if "/" in body else body[6:]
            out.append(f"+++ b/{path}{nl}")
        elif body.startswith("@@"):
            match = _HUNK_RE.match(body)
            if match:
                old_range, new_range, rest = match.group(1), match.group(2), match.group(3)
                out.append(f"@@ -{new_range} +{old_range} @@{rest}{nl}")
            else:
                out.append(line)
        elif body.startswith("+"):
            out.append("-" + body[1:] + nl)
        elif body.startswith("-"):
            out.append("+" + body[1:] + nl)
        else:
            out.append(line)
    return "".join(out)


async def _diff_applies_cleanly(sb, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await sb.write_file(_PATCH, diff_text, user="agent")
    for cmd in (
        f"cd {workdir} && git apply --check --3way --ignore-space-change --ignore-whitespace --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --check --ignore-space-change --ignore-whitespace --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --dry-run --no-backup-if-mismatch < {_PATCH}",
    ):
        ec, _, _ = await sb.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _apply_diff(sb, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await sb.write_file(_PATCH, diff_text, user="agent")
    for cmd in (
        f"cd {workdir} && git apply --3way --ignore-space-change --ignore-whitespace --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --ignore-space-change --ignore-whitespace --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ):
        ec, _, _ = await sb.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _commit_bug_baseline(sb, workdir: str) -> bool:
    wd = shlex.quote(workdir)
    ec, out, err = await sb.exec(f"cd {wd} && git status --porcelain", user="agent", check=False, timeout=120)
    if ec != 0:
        logger.warning("[workspace_init] git status failed: %s", (err or out or "").strip())
        return False
    if not (out or "").strip():
        return True
    commit_cmd = (
        f"cd {wd} && git add -A && "
        "git -c user.email=slime@example.invalid -c user.name=slime "
        "commit --no-verify -m 'SWE-Smith bug baseline'"
    )
    ec, out, err = await sb.exec(commit_cmd, user="agent", check=False, timeout=120)
    if ec != 0:
        logger.warning("[workspace_init] bug baseline commit failed: %s", (err or out or "").strip())
        return False
    return True


async def apply_swesmith_bug_baseline(sb, workdir: str, bug_patch: str) -> bool:
    if not bug_patch:
        return True
    if await _diff_applies_cleanly(sb, workdir, bug_patch):
        if not await _apply_diff(sb, workdir, bug_patch):
            logger.warning("[workspace_init] swe_smith_bug_patch did not apply")
            return False
    elif await _diff_applies_cleanly(sb, workdir, reverse_patch(bug_patch)):
        logger.info(
            "[workspace_init] swe_smith_bug_patch does not apply; "
            "reverse applies — assuming image already mutated"
        )
    else:
        logger.warning("[workspace_init] swe_smith_bug_patch did not apply")
        return False
    return await _commit_bug_baseline(sb, workdir)


def _needs_scrub(mode: WorkspaceMode) -> bool:
    return mode in {
        WorkspaceMode.SWESMITH,
        WorkspaceMode.SCALESWE,
        WorkspaceMode.SWEBENCH_CLASSIC,
    }


async def initialize_task_workspace(sb, fields: TaskFields, *, rollout_side: bool = True) -> bool:
    """Initialize repo to the expected buggy baseline. Returns False on hard failure."""
    workdir = fields.workdir
    mode = detect_workspace_mode(fields)
    await agent_sandbox.ensure_agent_user(sb, workdir)

    if mode == WorkspaceMode.SWEBENCH_CLASSIC:
        if not await apply_swebench_reset(sb, workdir, fields.base_commit):
            logger.warning(
                "[workspace_init] base_commit reset failed for %s",
                fields.instance_id,
            )
            return False
    elif mode == WorkspaceMode.SWESMITH:
        branch_ok = False
        if not is_synthetic_instance_id(fields.instance_id):
            branch_ok = await apply_swesmith_branch(sb, workdir, fields.instance_id)
        patch = fields.swe_smith_bug_patch or ""
        if not branch_ok and patch:
            if not await apply_swesmith_bug_baseline(sb, workdir, patch):
                return False
        elif branch_ok:
            if not await _commit_bug_baseline(sb, workdir):
                return False
        elif not branch_ok and not patch:
            logger.warning(
                "[workspace_init] swesmith: branch failed and no swe_smith_bug_patch for %s",
                fields.instance_id,
            )
            return False
    elif mode == WorkspaceMode.SCALESWE:
        if not await apply_scaleswe_pre_commands(sb, workdir, fields.pre_commands):
            logger.warning(
                "[workspace_init] scaleswe pre_commands failed for %s",
                fields.instance_id,
            )
            return False
        await scaleswe_root_filesystem_prep(sb)
    # rebench / generic: ensure user only

    if mode == WorkspaceMode.SWEBENCH_CLASSIC and rollout_side:
        if not await apply_swebench_git_setup(sb, workdir, fields.base_commit):
            logger.warning(
                "[workspace_init] official-style Git setup failed for %s",
                fields.instance_id,
            )
            return False
    elif _needs_scrub(mode):
        await apply_git_scrub(sb, workdir, rollout_side=rollout_side)
    return True
