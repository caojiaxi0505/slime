---
name: cos-cc-node-toolchain
description: Guide for wiring Claude Code and Node into slime Claude-Code/AGS training or evaluation jobs through a COS-mounted toolchain. Use when adding or modifying GRPO, Hybrid step-GRPO, SWE eval, AGS sandbox, Claude Code, Node, /mnt/code_agent, SLIME_AGENT_TOOLCHAIN_MODE=cos, or related launch templates.
---

# COS Claude Code / Node Toolchain

Use this skill when a slime Path A job needs Claude Code inside AGS sandboxes.

## Core Idea

Do not assume the training image or AGS runtime image already contains the right `node`, `npm`, `npx`, or `claude`.

For cluster jobs, install them inside each AGS sandbox from COS:

- COS mount inside sandbox: `/mnt/code_agent`
- Node package: `node-v20.18.1-linux-x64.tar.xz`
- Claude Code package: `cc-prefix-2.1.104-linux-x64.tar.gz`
- install target:
  - Node: `/opt/node-cos`
  - Claude Code: `/opt/cc-cos`
- public executable path used by code:
  - `/usr/local/bin/claude`

Implementation lives in `examples/claudecode_ags/agent_runtime.py`.

## When Adding a New Training or Eval Job

Add these env vars to every pod role that may create AGS sandboxes, usually both Master and Worker:

```yaml
- name: SLIME_AGENT_TOOLCHAIN_MODE
  value: "cos"
- name: SLIME_AGENT_COS_MOUNT
  value: "/mnt/code_agent"
- name: SLIME_AGENT_COS_NODE_PACKAGE
  value: "node-v20.18.1-linux-x64.tar.xz"
- name: SLIME_AGENT_COS_CC_PACKAGE
  value: "cc-prefix-2.1.104-linux-x64.tar.gz"
```

Also ensure the AGS tool used by the job exposes the COS directory as `/mnt/code_agent`, preferably read-only. If the AGS tool does not mount `/mnt/code_agent`, the env vars above are not enough.

## Where to Wire It

Use existing templates as references:

- GRPO 2-node: `examples/claudecode_ags/launch/grpo_2node_job/pytorchjob.yaml.template`
- Hybrid 2-node: `examples/claudecode_ags/launch/hybrid_2node_job/pytorchjob.yaml.template`
- SWE484 eval: `examples/claudecode_ags/launch/swe484_eval_job/pytorchjob.yaml.template`
- shared runtime: `examples/claudecode_ags/agent_runtime.py`
- env preservation / fallback: `examples/claudecode_ags/env/load_env.sh`

If creating a wrapper submit script, prefer exporting values through env and reusing the existing base submit script instead of copying the whole template.

## Required Runtime Calls

Before launching Claude Code in a sandbox, call:

```python
await agent_runtime.install_toolchain(sb)
```

Existing call sites include:

- GRPO rollout: `examples/claudecode_ags/generate.py`
- Hybrid Stage-1 / Stage-2: `examples/claudecode_ags/step_reconstruct/live_runners.py`
- SWE eval attempts: `examples/claudecode_ags/eval/attempts.py`
- AGS smoke: `examples/claudecode_ags/smoke/ags_smoke.py`

If adding a new path that starts Claude Code, add the call there too.

## What `install_toolchain()` Does

In `cos` mode it:

1. checks that both packages are readable under `/mnt/code_agent`;
2. deletes and recreates `/opt/node-cos` and `/opt/cc-cos`;
3. extracts Node with `tar -xJf ... --strip-components=1`;
4. extracts Claude Code with `tar -xzf ...`;
5. symlinks:
   - `/usr/local/bin/node`
   - `/usr/local/bin/npm`
   - `/usr/local/bin/npx`
   - `/usr/local/bin/claude`
6. verifies:
   - `node --version`
   - `npm --version`
   - `claude --version`

Do not silently ignore installation failures. A missing or broken toolchain should make that sandbox fail visibly.

## Validation Checklist

For a new job or AGS tool, validate with a smoke task before long training:

```bash
test -r /mnt/code_agent/node-v20.18.1-linux-x64.tar.xz
test -r /mnt/code_agent/cc-prefix-2.1.104-linux-x64.tar.gz
node --version
npm --version
claude --version
ls -l /usr/local/bin/claude
```

Expected result:

- both COS packages are readable;
- `node`, `npm`, and `claude` print versions;
- `/usr/local/bin/claude` points into `/opt/cc-cos/bin/claude`.

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `test -r /mnt/code_agent/...` fails | AGS tool did not mount COS, or package was deleted | restore COS package; use correct AGS tool |
| `tar -xJf` fails | Node package missing/corrupt/wrong format | re-upload Node tarball |
| `tar -xzf` fails | Claude Code package missing/corrupt/wrong format | re-upload Claude Code prefix tarball |
| `claude --version` fails | bad Claude package or Node not linked | rebuild package and smoke test |
| rollout suddenly has many infra failures | new sandboxes cannot install toolchain | stop/resume after COS is restored |

## Do Not

- Do not overwrite COS packages while training or eval jobs are running.
- Do not reuse an AGS tool unless it is known to mount `/mnt/code_agent` correctly.
- Do not add a fallback to fresh system `claude`; that breaks reproducibility.
- Do not hide install failures by returning success or zero reward.

## Further Reference

For the longer human-facing runbook, see:

- `docs/superpowers/runbooks/2026-07-24-cos-cc-node-toolchain.md`

