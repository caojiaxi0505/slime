"""Focused tests for rollout/eval Git repository setup."""

from __future__ import annotations

import os
import subprocess

from examples.claudecode_ags.workspace_init import (
    build_eval_git_scrub_command,
    build_git_scrub_command,
    build_swebench_git_setup_command,
)


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


def _commit(repo, content: str, message: str, date: str) -> str:
    (repo / "tracked.txt").write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-q",
            "-m",
            message,
        ],
        check=True,
        env=env,
    )
    return _git(repo, "rev-parse", "HEAD")


def _create_and_setup_swebench_repo(repo, inherited_date: str) -> dict[str, str]:
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    ancestor = _commit(
        repo,
        "ancestor\n",
        "ancestor",
        "2020-01-01T00:00:00+0000",
    )
    subprocess.run(["git", "-C", str(repo), "tag", "v1.0"], check=True)
    base = _commit(
        repo,
        "buggy base\n",
        "base",
        "2020-02-01T00:00:00+0000",
    )
    subprocess.run(["git", "-C", str(repo), "tag", "v1.1"], check=True)
    future = _commit(
        repo,
        "future fix\n",
        "future",
        "2020-03-01T00:00:00+0000",
    )
    subprocess.run(["git", "-C", str(repo), "tag", "v2.0"], check=True)
    subprocess.run(["git", "-C", str(repo), "branch", "future-line"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "https://example.invalid/repo.git"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", future],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "reset", "--hard", "-q", base], check=True)

    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": inherited_date,
        "GIT_COMMITTER_DATE": inherited_date,
    }
    subprocess.run(
        ["bash", "-c", build_swebench_git_setup_command(str(repo), base)],
        check=True,
        env=env,
    )
    return {
        "ancestor": ancestor,
        "base": base,
        "branch": _git(repo, "branch", "--show-current"),
        "future": future,
        "setup": _git(repo, "rev-parse", "HEAD"),
    }


def test_synthetic_git_scrub_commit_is_deterministic(tmp_path):
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


def test_swebench_git_setup_matches_official_history_shape(tmp_path):
    first = tmp_path / "first-swebench"
    second = tmp_path / "second-swebench"
    first_commits = _create_and_setup_swebench_repo(
        first,
        "2001-02-03T04:05:06+0000",
    )
    second_commits = _create_and_setup_swebench_repo(
        second,
        "2031-12-13T14:15:16+0000",
    )

    assert first_commits == second_commits
    base = first_commits["base"]
    setup = first_commits["setup"]
    assert _git(first, "rev-parse", "HEAD^") == base
    assert _git(first, "rev-parse", "HEAD") == setup
    assert _git(first, "rev-parse", "HEAD^{tree}") == _git(first, "rev-parse", f"{base}^{{tree}}")
    assert len(_git(first, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 2
    subprocess.run(
        ["git", "-C", str(first), "merge-base", "--is-ancestor", first_commits["ancestor"], "HEAD"],
        check=True,
    )

    assert _git(first, "remote") == ""
    assert _git(first, "for-each-ref", "--format=%(refname)", "refs/remotes/") == ""
    assert _git(first, "for-each-ref", "--format=%(refname)", "refs/heads/") == (
        f"refs/heads/{first_commits['branch']}"
    )
    assert _git(first, "tag", "--list") == "v1.0\nv1.1"
    assert first_commits["future"] not in _git(first, "rev-list", "--all").splitlines()
    assert _git(
        first,
        "show",
        "-s",
        "--format=%an|%ae|%cn|%ce|%at|%ct|%s",
        "HEAD",
    ) == (
        "SWE-bench|setup@swebench.config|SWE-bench|setup@swebench.config|"
        "1580515200|1580515200|SWE-bench"
    )
    assert _git(first, "config", "--get", "commit.gpgSign") == "false"


def test_eval_git_setup_preserves_tags_branches_and_history(tmp_path):
    repo = tmp_path / "eval"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.txt").write_text("release\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "add",
            "tracked.txt",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-q",
            "-m",
            "release",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "tag", "v1.2.3"], check=True)
    subprocess.run(["git", "-C", str(repo), "branch", "release-line"], check=True)
    refs_before = _git(repo, "show-ref")
    head_before = _git(repo, "rev-parse", "HEAD")

    subprocess.run(
        ["bash", "-c", build_eval_git_scrub_command(str(repo))],
        check=True,
    )

    assert _git(repo, "show-ref") == refs_before
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "rev-parse", "v1.2.3") == head_before
