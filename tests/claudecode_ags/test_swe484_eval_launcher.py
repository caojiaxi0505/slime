from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "examples" / "claudecode_ags" / "launch" / "swe484_eval_job"
RUN_GRPO = ROOT / "examples" / "claudecode_ags" / "launch" / "run_grpo_1node_debug.sh"
LOAD_ENV = ROOT / "examples" / "claudecode_ags" / "env" / "load_env.sh"


def test_launcher_pins_and_audits_official_swebench():
    submit = (LAUNCH / "submit_job.sh").read_text()
    template = (LAUNCH / "pytorchjob.yaml.template").read_text()

    assert 'SLIME_SWEBENCH_VERSION="${SLIME_SWEBENCH_VERSION:-4.1.0}"' in submit
    assert '"swebench==${SLIME_SWEBENCH_VERSION}"' in template
    assert "swe_eval.audit preflight" in template
    assert "swe_eval.audit postflight" in template


def test_launcher_supports_priority_and_a_safe_scheduling_gate():
    submit = (LAUNCH / "submit_job.sh").read_text()
    template = (LAUNCH / "pytorchjob.yaml.template").read_text()

    assert "K8S_PRIORITY_CLASS" in submit
    assert "K8S_SCHEDULING_GATE" in submit
    assert "K8S_TARGET_NODE" in submit
    assert "${K8S_SCHEDULING_SPEC}" in template
    assert "${K8S_TARGET_NODE_AFFINITY}" in template
    assert "matchFields:" in submit
    assert "key: metadata.name" in submit


def test_eval_prompt_limit_is_independent_from_training_prompt_limit():
    run_script = RUN_GRPO.read_text()

    assert 'EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-${MAX_CONTEXT_LEN}}"' in run_script
    assert '--eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"' in run_script


def test_eval_launcher_has_explicit_30m_reference_agent_context():
    submit = (LAUNCH / "submit_job.sh").read_text()
    template = (LAUNCH / "pytorchjob.yaml.template").read_text()

    assert 'SLIME_CC_TIME_BUDGET_SEC="${SLIME_CC_TIME_BUDGET_SEC:-1800}"' in submit
    assert 'CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"' in submit
    assert 'SLIME_CC_INITIAL_INPUT_MODE="${SLIME_CC_INITIAL_INPUT_MODE:-positional}"' in submit
    assert "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue." in submit
    assert "--disable-slash-commands" in submit
    assert "--disallowedTools" in submit
    assert 'value: "${SLIME_CC_TIME_BUDGET_SEC}"' in template
    assert 'value: "${CLAUDE_CODE_MAX_OUTPUT_TOKENS}"' in template
    assert 'value: "${EVAL_MAX_PROMPT_LEN}"' in template
    assert 'SLIME_CC_EVAL_GUARD_SEC="${SLIME_CC_EVAL_GUARD_SEC:-$((SLIME_CC_EVAL_TIMEOUT_SEC + 180))}"' in submit
    assert 'SLIME_CC_EVAL_INFRA_RETRIES="${SLIME_CC_EVAL_INFRA_RETRIES:-1}"' in submit
    assert 'SLIME_CC_GENERATE_GUARD_SEC="${SLIME_CC_GENERATE_GUARD_SEC:-2700}"' in submit
    assert 'SLIME_CC_EVAL_CONCURRENCY="${SLIME_CC_EVAL_CONCURRENCY:-64}"' in submit
    assert 'value: "${SLIME_CC_EVAL_GUARD_SEC}"' in template
    assert 'value: "${SLIME_CC_EVAL_INFRA_RETRIES}"' in template
    assert 'value: "${SLIME_CC_GENERATE_GUARD_SEC}"' in template
    assert 'value: "${SLIME_CC_EVAL_CONCURRENCY}"' in template
    assert 'SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC="${SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC:-2700}"' in submit


def test_eval_context_overrides_survive_shared_env_loader():
    loader = LOAD_ENV.read_text()

    for key in (
        "SLIME_CC_AGENT_PROMPT",
        "SLIME_CC_INITIAL_INPUT_MODE",
        "SLIME_CC_EXTRA_ARGS_JSON",
        "SLIME_CC_EVAL_CONCURRENCY",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
    ):
        assert key in loader
