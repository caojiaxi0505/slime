"""Tests for agent_runtime.install_toolchain modes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from examples.claudecode_ags.agent_runtime import install_toolchain
from tests.claudecode_ags.fake_sandbox import FakeSandbox


def test_install_toolchain_skip(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_TOOLCHAIN_MODE", "skip")
    sb = FakeSandbox()
    asyncio.run(install_toolchain(sb))
    assert sb.cmds == []


def test_install_toolchain_default_is_cos(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_TOOLCHAIN_MODE", raising=False)
    monkeypatch.setenv("SLIME_AGENT_COS_MOUNT", "/mnt/code_agent")
    monkeypatch.setenv("SLIME_AGENT_COS_NODE_PACKAGE", "node.tar.xz")
    monkeypatch.setenv("SLIME_AGENT_COS_CC_PACKAGE", "cc.tar.gz")
    monkeypatch.setenv("SLIME_AGENT_COS_NODE_DIR", "/opt/node-cos")
    monkeypatch.setenv("SLIME_AGENT_COS_CC_DIR", "/opt/cc-cos")
    sb = FakeSandbox()
    asyncio.run(install_toolchain(sb))
    assert len(sb.cmds) == 1
    cmd = sb.cmds[0]
    assert "/mnt/code_agent/node.tar.xz" in cmd
    assert "/mnt/code_agent/cc.tar.gz" in cmd
    assert "tar -xJf" in cmd
    assert "tar -xzf" in cmd
    assert "ln -sf /opt/node-cos/bin/node /usr/local/bin/node" in cmd
    assert "ln -sf /opt/cc-cos/bin/claude /usr/local/bin/claude" in cmd


def test_install_toolchain_cos_explicit(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_TOOLCHAIN_MODE", "cos")
    monkeypatch.setenv("SLIME_AGENT_COS_MOUNT", "/data/cos/")
    monkeypatch.setenv("SLIME_AGENT_COS_NODE_PACKAGE", "n.tar.xz")
    monkeypatch.setenv("SLIME_AGENT_COS_CC_PACKAGE", "c.tar.gz")
    sb = FakeSandbox()
    asyncio.run(install_toolchain(sb))
    cmd = sb.cmds[0]
    assert "/data/cos/n.tar.xz" in cmd
    assert "/data/cos/c.tar.gz" in cmd


def test_install_toolchain_tarball_requires_paths(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_TOOLCHAIN_MODE", "tarball")
    monkeypatch.delenv("SLIME_AGENT_NODE_TARBALL", raising=False)
    monkeypatch.delenv("SLIME_AGENT_CC_TARBALL", raising=False)
    with pytest.raises(RuntimeError, match="TARBALL"):
        asyncio.run(install_toolchain(FakeSandbox()))


def test_install_toolchain_tarball_calls_install_npm_cli(monkeypatch, tmp_path: Path):
    node = tmp_path / "node.tar"
    cc = tmp_path / "cc.tgz"
    node.write_text("n")
    cc.write_text("c")
    monkeypatch.setenv("SLIME_AGENT_TOOLCHAIN_MODE", "tarball")
    monkeypatch.setenv("SLIME_AGENT_NODE_TARBALL", str(node))
    monkeypatch.setenv("SLIME_AGENT_CC_TARBALL", str(cc))
    mock_install = AsyncMock()
    with patch("examples.claudecode_ags.agent_runtime.install_npm_cli", mock_install):
        asyncio.run(install_toolchain(FakeSandbox()))
    mock_install.assert_awaited_once()
    kwargs = mock_install.await_args.kwargs
    assert kwargs["node_runtime"] == Path(node)
    assert kwargs["npm_package"] == Path(cc)


def test_install_toolchain_unknown_mode(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_TOOLCHAIN_MODE", "download")
    with pytest.raises(ValueError, match="Unknown"):
        asyncio.run(install_toolchain(FakeSandbox()))
