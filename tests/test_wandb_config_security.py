from types import SimpleNamespace

from slime.utils import wandb_utils
from slime.utils.wandb_utils import _compute_config_for_logging


def test_wandb_config_redacts_key_and_records_hybrid_objective(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_HYBRID_K", "8")
    monkeypatch.setenv("STEP_GRPO_BRANCH_LOSS_WEIGHT", "1.5")
    args = SimpleNamespace(wandb_key="secret-value", use_critic=False)
    config = _compute_config_for_logging(args)
    assert config["wandb_key"] == "<redacted>"
    assert "secret-value" not in repr(config)
    assert config["step_grpo/hybrid_k"] == 8
    assert config["step_grpo/branch_loss_weight"] == 1.5


def test_coding_agent_metrics_use_rollout_step(monkeypatch):
    definitions = {}

    def fake_define_metric(name, **kwargs):
        definitions[name] = kwargs

    monkeypatch.setattr(wandb_utils.wandb, "define_metric", fake_define_metric)
    wandb_utils._init_wandb_common()

    for namespace in ("outcome", "traj", "behavior", "resume", "task"):
        assert definitions[f"{namespace}/*"] == {"step_metric": "rollout/step"}
