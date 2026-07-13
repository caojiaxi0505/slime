"""Offline checks for Kaniko launch artifacts (no cluster / docker)."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KANIKO_DIR = REPO_ROOT / "examples" / "claudecode_ags" / "launch" / "kaniko"
DOCKERFILE_KANIKO = REPO_ROOT / "docker" / "Dockerfile.kaniko"


def test_job_template_has_required_placeholders():
    text = (KANIKO_DIR / "job.yaml.template").read_text(encoding="utf-8")
    assert "${K8S_NAMESPACE}" in text
    assert "--dockerfile=${DOCKERFILE}" in text
    assert "--destination=${ECR_URI}:${IMAGE_TAG}" in text
    assert "--context=dir://${BUILD_CONTEXT}" in text
    assert "persistentVolumeClaim" in text
    assert "${FSX_PVC_NAME}" in text
    assert "${DOCKER_CONFIG_SECRET}" in text
    assert "restartPolicy: Never" in text
    assert "dexterzhou" not in text


def test_submit_script_has_no_foreign_user_defaults():
    text = (KANIKO_DIR / "submit_build.sh").read_text(encoding="utf-8")
    assert "dexterzhou" not in text
    assert 'K8S_NAMESPACE="${K8S_NAMESPACE:-sn5-system-intern}"' in text


def test_dockerfile_kaniko_copies_local_slime_not_upstream_clone():
    text = DOCKERFILE_KANIKO.read_text(encoding="utf-8")
    assert "COPY . /root/slime" in text
    assert "git clone https://github.com/THUDM/slime.git" not in text
    assert "ENABLE_EFA" in text
    assert "install-efa-in-container.sh" in text


def test_submit_build_help():
    script = KANIKO_DIR / "submit_build.sh"
    assert script.is_file()
    proc = subprocess.run(
        ["bash", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert "Kaniko" in out or "kaniko" in out.lower()
    assert "docker build" in out.lower() or "Local docker" in out
    assert "sn5-system-intern" in out


def test_submit_build_dry_run_renders_namespace():
    """ENABLE_EFA=0 dry-run should emit YAML with namespace and destination."""
    import os

    script = KANIKO_DIR / "submit_build.sh"
    env = {
        **os.environ,
        "ENABLE_EFA": "0",
        "AWS_ACCOUNT_ID": "123456789012",
        "IMAGE_TAG": "test-tag-dryrun",
        "BUILD_CONTEXT": str(REPO_ROOT),
    }
    # Avoid requiring live aws/kubectl for dry-run beyond what's in script
    proc = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(KANIKO_DIR),
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "sn5-system-intern" in proc.stdout
    assert "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:test-tag-dryrun" in proc.stdout
    assert "docker/Dockerfile.kaniko" in proc.stdout
