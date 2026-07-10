# Post-implementation checklist vs design spec

Branch: `feature/cc-ags-swe` · Task 10 audit · 2026-07-10

- [x] **Path A only; no SWE-Agent tree added** — `examples/claudecode_ags/` only; no `swe_agent` / rollout_buffer Path B code added in this branch.
- [x] **make_sandbox + sandbox_ags** — `slime/agent/sandbox.py` factory + `sandbox_ags.py`; unit tests cover `ags|e2b` dispatch and unknown-backend error.
- [x] **Segmented adapter: subagent/wipe/final** — `slime/agent/adapters/anthropic_segmented.py` with 5 routing tests in `test_segment_select.py`.
- [x] **fan_out even split + shared rollout_id** — `slime/agent/segment_trajectory.py` `fan_out_sample_segments`; covered by `test_fan_out.py`.
- [x] **fanout_grpo sum-then-GRPO-broadcast** — `slime/rollout/fanout_grpo.py`; covered by `test_fanout_grpo.py`.
- [x] **binary default reward; pluggable path** — `rewards/default.py` + `--custom-cc-reward-function-path`; wiring tests in `test_generate_reward_wiring.py` / `test_reward_binary.py`.
- [x] **two env files; no alias layer** — `env/claude_code.env` + `env/slime_ags.env.example`; `load_env.sh` sources both with no rename/`SWE_*`→`CLAUDE_*` mapping (example fallback when `slime_ags.env` absent is documented).
- [x] **agent_runtime not in core** — lives under `examples/claudecode_ags/agent_runtime.py` only.
- [x] **no Step-GRPO** — no `step_grpo` / `step_reconstruct` code paths in branch.
- [x] **official AnthropicAdapter / TrajectoryManager untouched in behavior** — zero diff vs `main` on `anthropic.py` and `trajectory.py`; segmented path uses new `anthropic_segmented.py` + `segment_trajectory.py`; imports verified.

## Partial / deferred (honest)

| Item | Status | Note |
|------|--------|------|
| COS toolchain install | **deferred** | `agent_runtime.install_toolchain` supports `skip` and `tarball` only; `SLIME_AGENT_COS_*` not implemented. |
| Live AGS integration test | **deferred** | Sandbox tests are factory/import-level; no end-to-end AGS create/exec/eval against real backend. |
| SWE-bench graders | **partial** | `swe_eval/simple_cmd.py` only (shell `eval_cmd` exit code); no dedicated swebench / rebench / scaleswe graders yet. |
| Generate E2E smoke | **partial** | `test_generate_reward_wiring.py` mocks sandbox/adapter; no full rollout integration test. |
| Launch script | **partial** | `launch/run_grpo_example.sh` is a dry-run skeleton (`RUN=1` to execute); not validated on HyperPod/ALB. |

## Test run (Task 10)

```bash
.venv/bin/python -m pytest tests/claudecode_ags/ -v
# 18 passed
```
