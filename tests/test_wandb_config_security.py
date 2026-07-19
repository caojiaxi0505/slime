from types import SimpleNamespace

from slime.utils import wandb_utils
from slime.utils.wandb_utils import _compute_config_for_logging, _wandb_lineage_kwargs_from_env


def test_wandb_config_redacts_key_and_records_hybrid_objective(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_HYBRID_K", "8")
    monkeypatch.setenv("STEP_GRPO_BRANCH_LOSS_WEIGHT", "1.5")
    monkeypatch.setenv("STEP_GRPO_STAGE2_LOSS_SCOPE", "first_turn")
    args = SimpleNamespace(wandb_key="secret-value", use_critic=False)
    config = _compute_config_for_logging(args)
    assert config["wandb_key"] == "<redacted>"
    assert "secret-value" not in repr(config)
    assert config["step_grpo/hybrid_k"] == 8
    assert config["step_grpo/branch_loss_weight"] == 1.5
    assert config["step_grpo/stage2_loss_scope"] == "first_turn"
    assert config["env_vars"]["STEP_GRPO_STAGE2_LOSS_SCOPE"] == "first_turn"


def test_coding_agent_metrics_use_rollout_step(monkeypatch):
    definitions = {}

    def fake_define_metric(name, **kwargs):
        definitions[name] = kwargs

    monkeypatch.setattr(wandb_utils.wandb, "define_metric", fake_define_metric)
    wandb_utils._init_wandb_common()

    for namespace in ("outcome", "traj", "behavior", "resume", "task"):
        assert definitions[f"{namespace}/*"] == {"step_metric": "rollout/step"}


def test_wandb_fork_from_is_forwarded_without_resume(monkeypatch):
    monkeypatch.setenv("WANDB_FORK_FROM", "parent?_step=59")
    monkeypatch.setenv("WANDB_RESUME", "auto")
    monkeypatch.setenv("WANDB_RUN_ID", "stale-id")
    assert _wandb_lineage_kwargs_from_env() == {"fork_from": "parent?_step=59"}


def test_wandb_fork_and_rewind_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("WANDB_FORK_FROM", "parent?_step=59")
    monkeypatch.setenv("WANDB_RESUME_FROM", "parent?_step=59")
    try:
        _wandb_lineage_kwargs_from_env()
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected conflicting W&B lineage settings to fail")
