import pytest

from slime.agent.sandbox import make_sandbox, sandbox_backend_from_env


def test_backend_from_env_default_e2b(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_SANDBOX_BACKEND", raising=False)
    assert sandbox_backend_from_env() == "e2b"


def test_backend_from_env_ags(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_SANDBOX_BACKEND", "ags")
    assert sandbox_backend_from_env() == "ags"


def test_make_sandbox_unknown_raises(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_SANDBOX_BACKEND", "nope")
    with pytest.raises(ValueError, match="Unknown sandbox backend"):
        make_sandbox("img:tag")


def test_import_sandbox_ags_without_credentials():
    import slime.agent.sandbox_ags as sandbox_ags

    assert hasattr(sandbox_ags, "AGSSandbox")
