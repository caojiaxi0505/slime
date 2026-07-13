"""Shared pytest-style log parsing for SWE graders."""

from __future__ import annotations

import re

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PYTEST_PROGRESS_SUFFIX_RE = re.compile(r"\s+\[\s*\d+%\]\s*$", re.MULTILINE)
_PYTEST_STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")
_PYTEST_STATUS_FIRST_RE = re.compile(
    r"^(?P<status>" + "|".join(_PYTEST_STATUSES) + r")\s+(?P<nodeid>\S.*?)\s*$"
)
_PYTEST_NODEID_FIRST_RE = re.compile(
    r"^(?P<nodeid>\S.*?::.+?)\s+(?P<status>" + "|".join(_PYTEST_STATUSES) + r")(?:\s.*)?$"
)
_STATUS_NORMALIZE = {"XPASS": "PASSED"}


def normalize_log(log: str) -> str:
    log = _ANSI_ESCAPE_RE.sub("", log or "")
    return _PYTEST_PROGRESS_SUFFIX_RE.sub("", log)


def _looks_like_pytest_nodeid(nodeid: str) -> bool:
    return "::" in nodeid or nodeid.endswith(".py") or "/" in nodeid


def _strip_pytest_nodeid_suffix(nodeid: str) -> str:
    nodeid = nodeid.strip()
    if "[" not in nodeid:
        return re.split(r"\s+-\s+", nodeid, maxsplit=1)[0].strip()
    last_rb = nodeid.rfind("]")
    if last_rb == -1:
        return re.split(r"\s+-\s+", nodeid, maxsplit=1)[0].strip()
    tail = nodeid[last_rb + 1 :]
    if re.match(r"\s+-\s+", tail):
        return nodeid[: last_rb + 1].strip()
    return nodeid


def parse_pytest_log(log: str) -> dict[str, str]:
    status_map: dict[str, str] = {}
    for raw in normalize_log(log).split("\n"):
        line = raw.rstrip()
        if not line:
            continue
        m = _PYTEST_STATUS_FIRST_RE.match(line)
        if m is None:
            m = _PYTEST_NODEID_FIRST_RE.match(line)
        if m is None:
            continue
        nodeid = _strip_pytest_nodeid_suffix(m.group("nodeid"))
        if not _looks_like_pytest_nodeid(nodeid):
            continue
        status = _STATUS_NORMALIZE.get(m.group("status"), m.group("status"))
        prev = status_map.get(nodeid)
        if prev in ("FAILED", "ERROR") and status == "PASSED":
            continue
        status_map[nodeid] = status
    return status_map
