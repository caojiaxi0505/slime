# CC + AGS + SWE Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Path A (Claude Code + AGS + SWE eval + GRPO) into upstream `slime` with low core intrusion, pluggable binary reward, and no SWE-Agent / legacy env aliases.

**Architecture:** Core owns sandbox factory + AGS backend, segmented Anthropic adapter, `fan_out` + `fanout_grpo`. Examples own `claudecode_ags` orchestration (`generate`, `agent_runtime`, `swe_eval`, `rewards`, two env files, launch). Rewrite from semantics — do not copy `slime-tencent` defensive/compat style.

**Tech Stack:** Python 3, Ray, SGLang, Megatron (existing slime), Tencent AGS via SWE-ReX, Claude Code CLI in sandbox.

**Spec:** `docs/superpowers/specs/2026-07-10-cc-ags-swe-refactor-design.md`

**Style rules (every task):**
- One canonical env name only (from the two env files); no `SWE_*` / `VERL_*` fallback chains
- No multi-field boolean compatibility piles
- Prefer short direct functions over wrapper layers
- Read tencent only for behavior; rewrite cleanly

---

## File map

| Path | Responsibility |
|---|---|
| `slime/agent/sandbox.py` | Keep Protocol + E2B; add thin `make_sandbox()` / `sandbox_backend_from_env()` |
| `slime/agent/sandbox_ags.py` | AGS backend only (`SLIME_AGENT_AGS_*`) |
| `slime/agent/segment_trajectory.py` | `TokenSegment`, `fan_out_sample_segments` (new file — avoids breaking official `TrajectoryManager`) |
| `slime/agent/adapters/anthropic_segmented.py` | Segmented CC adapter (`subagent`/`wipe`/`final`); keep official `anthropic.py` for E2B |
| `slime/rollout/fanout_grpo.py` | Fan-out-safe GRPO post_process |
| `slime/utils/arguments.py` | Add `--custom-cc-reward-function-path` |
| `examples/claudecode_ags/**` | Business entry: generate, agent_runtime, rewards, swe_eval, env, launch, README |
| `tests/claudecode_ags/**` | Unit tests for the above |

**Note vs spec paths:** Spec listed changes inside `trajectory.py` / `anthropic.py`. This plan uses **new sibling modules** so official `examples/coding_agent_rl` keeps working unchanged. Behavior matches the spec.

---

### Task 1: Branch + test scaffolding

**Files:**
- Create: `tests/claudecode_ags/__init__.py`

- [ ] **Step 1: Create branch from clean main**

```bash
cd /mnt/sn-007/jiaxicao/code/slime
git fetch origin
git switch main
git pull --ff-only origin main   # if available
git switch -c feature/cc-ags-swe
```

Expected: on `feature/cc-ags-swe`, clean tree except already-committed design/plan docs.

- [ ] **Step 2: Add test package**

```bash
mkdir -p tests/claudecode_ags
touch tests/claudecode_ags/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add tests/claudecode_ags/__init__.py
git commit -m "chore: scaffold claudecode_ags test package"
```

---

### Task 2: Binary reward plugin

**Files:**
- Create: `examples/claudecode_ags/__init__.py`
- Create: `examples/claudecode_ags/rewards/__init__.py`
- Create: `examples/claudecode_ags/rewards/default.py`
- Create: `tests/claudecode_ags/test_reward_binary.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/claudecode_ags/test_reward_binary.py
from examples.claudecode_ags.rewards.default import compose


def test_binary_resolved_true():
    reward, details = compose(base_eval={"resolved": True}, sample=None)
    assert reward == 1.0
    assert details["mode"] == "binary"


def test_binary_resolved_false():
    reward, details = compose(base_eval={"resolved": False}, sample=None)
    assert reward == 0.0
    assert details["mode"] == "binary"
```

- [ ] **Step 2: Run tests — expect FAIL (import error)**

```bash
cd /mnt/sn-007/jiaxicao/code/slime
python -m pytest tests/claudecode_ags/test_reward_binary.py -v
```

Expected: FAIL with `ModuleNotFoundError` or import error.

- [ ] **Step 3: Implement binary compose (keep it tiny)**

```python
# examples/claudecode_ags/rewards/default.py
"""Default CC reward: binary resolved → {0,1}."""

from __future__ import annotations

from typing import Any


def compose(*, base_eval: dict[str, Any], sample: Any = None, args: Any = None) -> tuple[float, dict[str, Any]]:
    del sample, args
    resolved = bool(base_eval["resolved"])
    return (1.0 if resolved else 0.0), {"mode": "binary", "resolved": resolved}
```

Also add empty `__init__.py` files under `examples/claudecode_ags/` and `rewards/`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/claudecode_ags/test_reward_binary.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/claudecode_ags/rewards tests/claudecode_ags/test_reward_binary.py
git commit -m "feat(claudecode_ags): add binary reward plugin"
```

---

### Task 3: `fan_out_sample_segments`

**Files:**
- Create: `slime/agent/segment_trajectory.py`
- Create: `tests/claudecode_ags/test_fan_out.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/claudecode_ags/test_fan_out.py
from slime.agent.segment_trajectory import TokenSegment, fan_out_sample_segments
from slime.utils.types import Sample


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def test_fan_out_splits_reward_and_shares_rollout_id():
    sample = Sample(index=7, group_index=3, prompt="p", metadata={})
    segs = [
        TokenSegment(
            prompt_ids=[1],
            response_ids=[2, 3],
            loss_mask=[1, 1],
            rollout_log_probs=[0.0, 0.0],
            metadata={"segment_kind": "wipe"},
        ),
        TokenSegment(
            prompt_ids=[1],
            response_ids=[4],
            loss_mask=[1],
            rollout_log_probs=[0.0],
            metadata={"segment_kind": "final"},
        ),
    ]
    out = fan_out_sample_segments(sample, segs, reward=1.0, tokenizer=_Tok())
    assert len(out) == 2
    assert out[0].reward == 0.5
    assert out[1].reward == 0.5
    assert out[0].rollout_id == 7
    assert out[1].rollout_id == 7
    assert out[0].metadata["num_segments"] == 2
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/claudecode_ags/test_fan_out.py -v
```

- [ ] **Step 3: Implement**

```python
# slime/agent/segment_trajectory.py
"""Segment fan-out helpers for compact/subagent rollouts."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from slime.utils.types import Sample


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def write_segment_to_sample(sample: Sample, segment: TokenSegment, reward: float, tokenizer) -> None:
    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.loss_mask = list(segment.loss_mask)
    sample.rollout_log_probs = list(segment.rollout_log_probs)
    sample.response = tokenizer.decode(segment.response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED


def fan_out_sample_segments(
    sample: Sample,
    segments: list[TokenSegment],
    reward: float,
    tokenizer,
    *,
    metadata: dict[str, Any] | None = None,
    rollout_id: int | None = None,
) -> list[Sample]:
    """One Sample per segment; split reward evenly; share rollout_id."""
    k = len(segments)
    if k == 0:
        return []
    per = float(reward) / k
    shared = sample.index if rollout_id is None else rollout_id
    base_md = {**(sample.metadata or {}), **(metadata or {})}
    out: list[Sample] = []
    for i, segment in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        write_segment_to_sample(sub, segment, per, tokenizer)
        sub.rollout_id = shared
        sub.metadata = {
            **base_md,
            **(segment.metadata or {}),
            "segment_idx": i,
            "num_segments": k,
        }
        out.append(sub)
    return out
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/claudecode_ags/test_fan_out.py -v
```

- [ ] **Step 5: Commit**

```bash
git add slime/agent/segment_trajectory.py tests/claudecode_ags/test_fan_out.py
git commit -m "feat(agent): add segment fan-out helpers for CC rollouts"
```

---

### Task 4: `fanout_grpo.post_process_rewards`

**Files:**
- Create: `slime/rollout/fanout_grpo.py`
- Create: `tests/claudecode_ags/test_fanout_grpo.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/claudecode_ags/test_fanout_grpo.py
from types import SimpleNamespace

from slime.rollout.fanout_grpo import post_process_rewards
from slime.utils.types import Sample


def _s(group_index, index, reward):
    return Sample(group_index=group_index, index=index, reward=reward, prompt="p")


def test_sums_segments_then_centers_across_repeats():
    # One prompt, two repeats. Repeat0 has two segments 0.5+0.5=1; repeat1 has 0.
    samples = [
        _s(0, 10, 0.5),
        _s(0, 10, 0.5),
        _s(0, 11, 0.0),
    ]
    args = SimpleNamespace(grpo_std_normalization=False, advantage_estimator="grpo")
    raw, adv = post_process_rewards(args, samples)
    assert raw == [0.5, 0.5, 0.0]
    # episode rewards [1.0, 0.0] → mean 0.5 → advantages [0.5, -0.5]
    assert adv[0] == adv[1] == 0.5
    assert adv[2] == -0.5
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/claudecode_ags/test_fanout_grpo.py -v
```

- [ ] **Step 3: Implement (rewrite; do not copy swe_agent_grpo_std)**

```python
# slime/rollout/fanout_grpo.py
"""GRPO/GSPO reward post-process that is safe under segment fan-out."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from slime.utils.types import Sample


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    flat: list[Sample] = []
    for item in samples:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    raw = [float(s.get_reward_value(args)) for s in flat]
    if not flat:
        return raw, raw

    by_group: dict[Any, list[tuple[int, Sample]]] = defaultdict(list)
    for i, s in enumerate(flat):
        gkey = s.group_index if s.group_index is not None else s.index
        by_group[gkey].append((i, s))

    advantages = [0.0] * len(flat)
    use_std = bool(getattr(args, "grpo_std_normalization", True)) and getattr(args, "advantage_estimator", "grpo") in (
        "grpo",
        "gspo",
    )

    for entries in by_group.values():
        episode_reward: dict[Any, float] = {}
        positions: dict[Any, list[int]] = defaultdict(list)
        for pos, s in entries:
            rkey = s.index if s.index is not None else id(s)
            positions[rkey].append(pos)
            episode_reward[rkey] = episode_reward.get(rkey, 0.0) + float(s.get_reward_value(args))

        keys = list(episode_reward.keys())
        tensor = torch.tensor([episode_reward[k] for k in keys], dtype=torch.float)
        centered = tensor - tensor.mean()
        if use_std and tensor.numel() > 1:
            centered = centered / (centered.std(unbiased=False) + 1e-6)
        adv_map = {k: float(centered[i].item()) for i, k in enumerate(keys)}
        for rkey, pos_list in positions.items():
            a = adv_map[rkey]
            for pos in pos_list:
                advantages[pos] = a

    return raw, advantages
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/claudecode_ags/test_fanout_grpo.py -v
```

- [ ] **Step 5: Commit**

```bash
git add slime/rollout/fanout_grpo.py tests/claudecode_ags/test_fanout_grpo.py
git commit -m "feat(rollout): add fan-out-safe GRPO post_process"
```

---

### Task 5: Sandbox factory + AGS module

**Files:**
- Modify: `slime/agent/sandbox.py` (add factory only; keep E2B)
- Create: `slime/agent/sandbox_ags.py`
- Create: `tests/claudecode_ags/test_sandbox_factory.py`

- [ ] **Step 1: Write failing factory tests**

```python
# tests/claudecode_ags/test_sandbox_factory.py
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/claudecode_ags/test_sandbox_factory.py -v
```

- [ ] **Step 3: Add factory to `sandbox.py`**

Append (do not add env alias lists):

```python
def sandbox_backend_from_env(default: str = "e2b") -> str:
    return (os.environ.get("SLIME_AGENT_SANDBOX_BACKEND") or default).strip().lower()


def make_sandbox(image: str, *, backend: str | None = None):
    """Construct a sandbox for ``image`` using ``backend`` or env."""
    kind = (backend or sandbox_backend_from_env()).strip().lower()
    if kind == "e2b":
        return E2BSandbox(image)
    if kind == "ags":
        from slime.agent.sandbox_ags import AGSSandbox

        return AGSSandbox(image)
    raise ValueError(f"Unknown sandbox backend: {kind!r}")
```

- [ ] **Step 4: Create lean `sandbox_ags.py`**

Implement `AGSSandbox` against the `Sandbox` Protocol:

- Read **only** `SLIME_AGENT_AGS_*` and `SLIME_AGENT_AGS_ENV_FILE` (load KEY=VALUE file once)
- Methods: `__aenter__` / `__aexit__` / `exec` / `write_file` / `read_file`
- Use SWE-ReX Tencent AGS deployment (same capability as tencent, **rewrite** — no multi-name `_getenv` chains, no hardcoded account defaults in-repo; require env for secrets/tool_id/role)
- On missing required env: raise clear `RuntimeError` listing the missing `SLIME_AGENT_AGS_*` keys
- Import of the module must succeed without credentials; failure only on enter/start

Reference behavior only: `slime-tencent/slime/agent/sandbox.py` `AGSSandbox` — do not paste it.

Minimal required env (module docstring):
`SLIME_AGENT_AGS_ENV_FILE` or (`SLIME_AGENT_AGS_SECRET_ID` + `SLIME_AGENT_AGS_SECRET_KEY`), plus `SLIME_AGENT_AGS_TOOL_ID`, `SLIME_AGENT_AGS_REGION`, `SLIME_AGENT_AGS_DOMAIN`, `SLIME_AGENT_AGS_ROLE_ARN`, `SLIME_AGENT_AGS_HTTP_ENDPOINT`, optional CPU/memory/timeout/mount via `SLIME_AGENT_AGS_*`, and `SLIME_AGENT_AGS_SWE_REX_ROOT`.

- [ ] **Step 5: Run factory tests PASS**

```bash
python -m pytest tests/claudecode_ags/test_sandbox_factory.py -v
```

- [ ] **Step 6: Commit**

```bash
git add slime/agent/sandbox.py slime/agent/sandbox_ags.py tests/claudecode_ags/test_sandbox_factory.py
git commit -m "feat(agent): add AGS sandbox backend and make_sandbox factory"
```

---

### Task 6: Segmented Anthropic adapter

**Files:**
- Create: `slime/agent/adapters/anthropic_segmented.py`
- Modify: `slime/agent/adapters/__init__.py` (export `SegmentedAnthropicAdapter`)
- Create: `tests/claudecode_ags/test_segment_select.py`

**Behavior to preserve:**
- Session has `main` chain + optional `active_sub`
- Classify turn as `new` / `append` / `wipe`; wipe freezes prior turns as `wipe`
- Subagent tools `Task`/`Agent` open sub chain; closing tool_result freezes `subagent`
- `finish_session` freezes remaining sub as `subagent`, main as `final`
- Return `list[TokenSegment]` (build via turn merge in this module or small helpers next to `segment_trajectory`)

- [ ] **Step 1: Unit-test pure routing helpers**

Write tests for select/wipe/subagent close using fake Session + message hashes — no HTTP server. Extract pure functions for testability.

Cases:
1. Empty main + first request → `new` on main
2. Prefix-continuing messages → `append`
3. Divergent messages on main → `wipe` segment recorded
4. After Task tool_use, non-continuing messages route to sub
5. tool_result for pending dispatch closes sub → `subagent` segment

- [ ] **Step 2: Implement routing + HTTP turn path + `finish_session`**

Prefer reusing `slime.agent.adapters.common` SGLang helpers and `slime.agent.parsing` rather than duplicating clients. Add `/health` only if needed for ALB. Skip tencent debug/vllm-bridge sprawl unless a concrete failure requires it.

- [ ] **Step 3: Run unit tests PASS**

```bash
python -m pytest tests/claudecode_ags/test_segment_select.py -v
```

- [ ] **Step 4: Commit**

```bash
git add slime/agent/adapters/anthropic_segmented.py slime/agent/adapters/__init__.py tests/claudecode_ags/test_segment_select.py
git commit -m "feat(agent): add segmented Anthropic adapter for CC subagent/compact"
```

---

### Task 7: `swe_eval` + `agent_runtime`

**Files:**
- Create: `examples/claudecode_ags/swe_eval/__init__.py`
- Create: `examples/claudecode_ags/swe_eval/base.py`
- Create: `examples/claudecode_ags/swe_eval/simple_cmd.py`
- Create: `examples/claudecode_ags/agent_runtime.py`
- Create: `tests/claudecode_ags/fake_sandbox.py`
- Create: `tests/claudecode_ags/test_agent_runtime_env.py`

- [ ] **Step 1: Fake sandbox**

```python
# tests/claudecode_ags/fake_sandbox.py
class FakeSandbox:
    def __init__(self):
        self.files: dict[str, str] = {}
        self.cmds: list[str] = []
        self.sandbox_id = "fake"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False):
        self.cmds.append(cmd)
        return 0, "", ""

    async def write_file(self, path, content, *, user="root"):
        self.files[path] = content if isinstance(content, str) else content.decode()

    async def read_file(self, path, *, user="root"):
        return self.files.get(path, "")
```

- [ ] **Step 2: EvalResult + simple_cmd grader**

```python
# examples/claudecode_ags/swe_eval/base.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class EvalResult:
    resolved: bool
    applied_cleanly: bool
    details: dict[str, Any] = field(default_factory=dict)
```

`simple_cmd.py`: run `eval_cmd` in a fresh sandbox; `resolved = (exit_code == 0)`.

Add swebench/rebench later as separate modules with the same `EvalResult` shape. Thresholds only from `SLIME_CC_REWARD_F2P_THRESHOLD` / `SLIME_CC_REWARD_P2P_THRESHOLD` — no VERL aliases.

- [ ] **Step 3: `agent_runtime.py` public API**

```python
async def prepare_workspace(sb, *, workdir: str, problem_statement: str) -> None: ...
async def install_toolchain(sb) -> None: ...
async def run_claude(sb, *, workdir: str, prompt: str, env: dict[str, str], time_budget_sec: int) -> dict: ...
async def git_diff(sb, *, workdir: str) -> str: ...
```

Unit-test: `prepare_workspace` writes `PROBLEM_STATEMENT.md`; `run_claude` uses only the provided `env` dict (caller loads `claude_code.env`).

- [ ] **Step 4: Tests PASS + commit**

```bash
python -m pytest tests/claudecode_ags/test_agent_runtime_env.py -v
git add examples/claudecode_ags/agent_runtime.py examples/claudecode_ags/swe_eval tests/claudecode_ags/fake_sandbox.py tests/claudecode_ags/test_agent_runtime_env.py
git commit -m "feat(claudecode_ags): add agent_runtime and swe_eval stubs"
```

---

### Task 8: `generate.py` + reward CLI arg

**Files:**
- Create: `examples/claudecode_ags/generate.py`
- Modify: `slime/utils/arguments.py`
- Create: `tests/claudecode_ags/test_generate_reward_wiring.py`

- [ ] **Step 1: Add argparse near `--custom-generate-function-path`**

```python
parser.add_argument(
    "--custom-cc-reward-function-path",
    type=str,
    default="examples.claudecode_ags.rewards.default.compose",
    help="module.fn for CC reward: compose(*, base_eval, sample=None, args=None) -> (float, dict)",
)
```

- [ ] **Step 2: Implement `generate(args, sample, sampling_params, evaluation=False)`**

1. Parse metadata (`image`, `workdir`, `problem_statement`, grader fields)
2. Start `SegmentedAnthropicAdapter` session + threaded aiohttp (`slime.agent.aiohttp_threaded` if present)
3. `async with make_sandbox(image) as sb:` → agent_runtime prepare/install/run
4. `git_diff` → second sandbox eval → `EvalResult`
5. `load_function(args.custom_cc_reward_function_path)` → `R, details`
6. `finish_session` → `fan_out_sample_segments`
7. Return `list[Sample]`

Env: `SLIME_CC_TIME_BUDGET_SEC`, `SLIME_CC_EVAL_TIMEOUT_SEC`, `SLIME_CC_GENERATE_GUARD_SEC`, `SLIME_ADAPTER_PUBLIC_URL`, `SLIME_ADAPTER_BIND_HOST`, `SLIME_ADAPTER_PORT`.

- [ ] **Step 3: Mocked unit test for reward + fan_out wiring**

- [ ] **Step 4: Commit**

```bash
git add examples/claudecode_ags/generate.py slime/utils/arguments.py tests/claudecode_ags/test_generate_reward_wiring.py
git commit -m "feat(claudecode_ags): wire generate orchestration and pluggable reward"
```

---

### Task 9: Env files + launch + README

**Files:**
- Create: `examples/claudecode_ags/env/claude_code.env`
- Create: `examples/claudecode_ags/env/slime_ags.env.example`
- Create: `examples/claudecode_ags/env/load_env.sh`
- Create: `examples/claudecode_ags/launch/run_grpo_example.sh`
- Create: `examples/claudecode_ags/README.md`
- Optional: gitignore `examples/claudecode_ags/env/slime_ags.env`

- [ ] **Step 1: `claude_code.env`** — official CC vars only

- [ ] **Step 2: `slime_ags.env.example`** — all `SLIME_*` from spec §8.3

- [ ] **Step 3: `load_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck disable=SC1091
source "${DIR}/claude_code.env"
if [[ -f "${DIR}/slime_ags.env" ]]; then
  # shellcheck disable=SC1091
  source "${DIR}/slime_ags.env"
elif [[ -f "${DIR}/slime_ags.env.example" ]]; then
  echo "WARNING: using slime_ags.env.example; copy to slime_ags.env for real runs" >&2
  # shellcheck disable=SC1091
  source "${DIR}/slime_ags.env.example"
fi
set +a
```

No rename/alias logic.

- [ ] **Step 4: Launch skeleton** with:

```text
--custom-generate-function-path examples.claudecode_ags.generate.generate
--custom-cc-reward-function-path examples.claudecode_ags.rewards.default.compose
--custom-reward-post-process-path slime.rollout.fanout_grpo.post_process_rewards
```

- [ ] **Step 5: README** — call chain, two env files, how to swap reward, link to design spec

- [ ] **Step 6: Commit**

```bash
git add examples/claudecode_ags/env examples/claudecode_ags/launch examples/claudecode_ags/README.md
git commit -m "docs(claudecode_ags): add env files, launch skeleton, and README"
```

---

### Task 10: Spec compliance + regression

- [ ] **Step 1: Run all new unit tests**

```bash
python -m pytest tests/claudecode_ags/ -v
```

Expected: all PASS

- [ ] **Step 2: Official imports still work**

```bash
python -c "from slime.agent.adapters import AnthropicAdapter; from slime.agent.trajectory import TrajectoryManager; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Checklist**

- [ ] Path A only; no SWE-Agent tree added
- [ ] `make_sandbox` + `sandbox_ags`
- [ ] Segmented adapter: subagent/wipe/final
- [ ] fan_out even split + shared rollout_id
- [ ] fanout_grpo sum-then-GRPO-broadcast
- [ ] binary default reward; pluggable path
- [ ] two env files; no alias layer
- [ ] `agent_runtime` not in core
- [ ] no Step-GRPO
- [ ] official `AnthropicAdapter` / `TrajectoryManager` untouched in behavior

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| AGS + factory | Task 5 |
| Segmented adapter / final vs wipe | Task 6 |
| fan_out R/K + rollout_id | Task 3 |
| fanout_grpo | Task 4 |
| binary pluggable reward | Task 2, 8 |
| agent_runtime in examples | Task 7 |
| two env files, no aliases | Task 9 |
| generate orchestration | Task 8 |
| No SWE-Agent / Step-GRPO | Task 1 from clean main; Task 10 |
| Style: rewrite not copy | Style rules + Tasks 5–7 |
| Official coding_agent_rl intact | Sibling modules |

**Intentional path deltas:** `segment_trajectory.py` and `anthropic_segmented.py` instead of overwriting upstream `trajectory.py` / `anthropic.py`.

---

## Out of scope (follow-ups)

- Full swebench/rebench/scaleswe graders beyond `simple_cmd` (+ one production grader next)
- Live AGS CI (needs secrets)
- Step-GRPO
- Full HyperPod/ALB launch parity with every tencent `run_*.sh`
