"""In-memory sandbox for claudecode_ags unit tests."""

from __future__ import annotations

import re

_POLL_RE = re.compile(r"test -f (\S+) && cat \1")
_SETSID_RE = re.compile(r"setsid bash (\S+)")


class FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.cmds: list[str] = []
        self.exec_envs: list[dict[str, str] | None] = []
        self.sandbox_id = "fake"

    async def __aenter__(self) -> FakeSandbox:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> tuple[int, str, str]:
        del user, timeout, check, idempotent
        self.cmds.append(cmd)
        self.exec_envs.append(env)

        m = _SETSID_RE.search(cmd)
        if m:
            launcher = m.group(1)
            body = self.files.get(launcher, "")
            if "echo $? >" in body:
                done = body.rsplit("echo $? >", 1)[-1].strip()
                self.files[done] = "0"
            return 0, "", ""

        poll = _POLL_RE.search(cmd)
        if poll:
            path = poll.group(1)
            if path in self.files:
                return 0, self.files[path], ""
            return 1, "", ""

        return 0, "", ""

    async def write_file(self, path: str, content, *, user: str = "root") -> None:
        del user
        self.files[path] = content if isinstance(content, str) else content.decode()

    async def read_file(self, path: str, *, user: str = "root") -> str:
        del user
        return self.files.get(path, "")
