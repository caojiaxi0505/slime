from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "examples" / "claudecode_ags" / "launch"


def test_launchers_do_not_serialize_wandb_credentials():
    paths = [
        LAUNCH / "run_hybrid_1node_debug.sh",
        LAUNCH / "run_grpo_1node_debug.sh",
        *LAUNCH.glob("*_job/submit_job.sh"),
        *LAUNCH.glob("*_job/pytorchjob.yaml.template"),
    ]
    for path in paths:
        text = path.read_text()
        assert "--wandb-key" not in text, path
        assert "WANDB_KEY" not in text, path


def test_pytorchjobs_inject_wandb_api_key_from_secret():
    templates = [
        LAUNCH / "hybrid_1node_job" / "pytorchjob.yaml.template",
        LAUNCH / "hybrid_2node_job" / "pytorchjob.yaml.template",
        LAUNCH / "grpo_1node_job" / "pytorchjob.yaml.template",
        LAUNCH / "grpo_2node_job" / "pytorchjob.yaml.template",
        LAUNCH / "swe484_eval_job" / "pytorchjob.yaml.template",
    ]
    for path in templates:
        text = path.read_text()
        assert "name: WANDB_API_KEY" in text, path
        assert "secretKeyRef:" in text, path
        assert "name: ${WANDB_SECRET_NAME}" in text, path
        assert "key: ${WANDB_SECRET_KEY}" in text, path
