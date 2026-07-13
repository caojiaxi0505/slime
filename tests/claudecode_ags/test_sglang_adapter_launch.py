"""Offline checks for sglang_adapter launch artifacts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "examples" / "claudecode_ags" / "launch" / "sglang_adapter"


def test_templates_have_namespace_and_alb():
    dep = (LAUNCH / "deployment.yaml.template").read_text(encoding="utf-8")
    ing = (LAUNCH / "service-ingress.yaml.template").read_text(encoding="utf-8")
    assert "${K8S_NAMESPACE}" in dep
    assert "nvidia.com/gpu" in dep
    assert "youtu-sn2-007" in dep or "${FSX_PVC_NAME}" in dep
    assert "alb.ingress.kubernetes.io/scheme: internet-facing" in ing
    assert "alb.ingress.kubernetes.io/healthcheck-path: /health" in ing
    assert "${SERVICE_NAME}" in ing


def test_submit_deploy_help():
    script = LAUNCH / "submit_deploy.sh"
    proc = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "HF_CHECKPOINT" in proc.stdout
    assert "sn5-system-intern" in proc.stdout


def test_submit_deploy_dry_run():
    env = {
        **os.environ,
        "HF_CHECKPOINT": "/mnt/sn-007/jiaxicao/fake-model",
        "K8S_NAMESPACE": "sn5-system-intern",
    }
    proc = subprocess.run(
        ["bash", str(LAUNCH / "submit_deploy.sh"), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(LAUNCH),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "kind: Deployment" in proc.stdout
    assert "kind: Ingress" in proc.stdout
    assert "sn5-system-intern" in proc.stdout
    assert "jiaxicao-cc-ags-adapter" in proc.stdout
