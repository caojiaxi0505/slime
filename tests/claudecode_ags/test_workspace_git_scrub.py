"""Focused tests for deterministic rollout-side Git history scrubbing."""

from __future__ import annotations

import os
import subprocess

from examples.claudecode_ags.workspace_init import build_git_scrub_command


def _git(repo, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _create_and_scrub_repo(repo, content: str, inherited_date: str) -> str:
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.txt").write_text(content, encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": inherited_date,
        "GIT_COMMITTER_DATE": inherited_date,
    }
    subprocess.run(
        ["bash", "-c", build_git_scrub_command(str(repo))],
        check=True,
        env=env,
    )
    return _git(repo, "rev-parse", "HEAD")


def test_rollout_git_scrub_commit_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    different = tmp_path / "different"

    first_head = _create_and_scrub_repo(
        first,
        "same tree\n",
        "2001-02-03T04:05:06+0000",
    )
    second_head = _create_and_scrub_repo(
        second,
        "same tree\n",
        "2031-12-13T14:15:16+0000",
    )
    different_head = _create_and_scrub_repo(
        different,
        "different tree\n",
        "2041-01-02T03:04:05+0000",
    )

    assert first_head == second_head
    assert first_head != different_head
    for repo in (first, second, different):
        assert _git(repo, "branch", "--show-current") == "__slime_buggy"
        assert _git(repo, "show", "-s", "--format=%at:%ct", "HEAD") == (
            "946684800:946684800"
        )
        assert len(_git(repo, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 1
