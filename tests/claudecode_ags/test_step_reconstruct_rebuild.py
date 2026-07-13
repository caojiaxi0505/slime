import asyncio

from examples.claudecode_ags.step_reconstruct.session_capture import SessionBundle, steps_from_diff_files
from examples.claudecode_ags.step_reconstruct.workspace_rebuild import (
    apply_diff,
    truncate_transcript_prefix,
    verify_rebuild,
)


class _FakeSB:
    def __init__(self):
        self.files = {}
        self.applied = False
        self._diff_out = ""

    async def write_file(self, path, content, user="agent"):
        self.files[path] = content

    async def exec(self, cmd, user="agent", timeout=60, check=False):
        if "git apply" in cmd:
            self.applied = True
            return 0, "", ""
        if "git diff" in cmd:
            return 0, self._diff_out, ""
        return 0, "", ""


def test_apply_diff_empty_ok():
    sb = _FakeSB()
    assert asyncio.run(apply_diff(sb, "/testbed", "")) is True


def test_apply_diff_writes_and_runs():
    sb = _FakeSB()
    assert asyncio.run(apply_diff(sb, "/testbed", "diff --git a/a b/a\n+x\n")) is True
    assert sb.applied
    assert "/tmp/_step_reconstruct_apply.diff" in sb.files


def test_verify_rebuild_path_match(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    diff = "diff --git a/foo.py b/foo.py\n+++ b/foo.py\n+x\n"
    steps = steps_from_diff_files(str(d), [diff])
    b = SessionBundle(
        instance_id="i",
        session_id="s",
        cc_session_id="",
        task_metadata={"workdir": "/testbed"},
        steps=steps,
        dir=str(d),
    )
    b.save(str(d))
    bundle = SessionBundle.load(str(d))
    sb = _FakeSB()
    sb._diff_out = diff
    ok, details = asyncio.run(verify_rebuild(sb, bundle, 0))
    assert ok
    assert details["paths_match"]


def test_truncate_transcript_prefix():
    lines = [
        '{"type":"assistant"}',
        '{"type":"tool_result","id":"1"}',
        '{"type":"assistant"}',
        '{"type":"tool_result","id":"2"}',
        '{"type":"assistant"}',
    ]
    text = "\n".join(lines)
    out = truncate_transcript_prefix(text, 0)
    assert '"id":"1"' in out.replace(" ", "")
    assert '"id":"2"' not in out.replace(" ", "")
