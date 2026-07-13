# SGLang adapter 部署 + L2 冒烟 — 操作手册

日期：2026-07-11  
设计：[2026-07-11-sglang-adapter-l2-smoke-design.md](../specs/2026-07-11-sglang-adapter-l2-smoke-design.md)  
计划：[2026-07-11-sglang-adapter-l2-smoke.md](../plans/2026-07-11-sglang-adapter-l2-smoke.md)

## 1. 涉及文件

| 路径 | 作用 |
|------|------|
| `examples/claudecode_ags/launch/sglang_adapter/submit_deploy.sh` | 提交 Deployment + Service + Ingress |
| `examples/claudecode_ags/launch/sglang_adapter/entrypoint.sh` | 容器内起 SGLang → adapter |
| `examples/claudecode_ags/launch/sglang_adapter/serve_adapter.py` | SegmentedAnthropicAdapter |
| `examples/claudecode_ags/smoke/ags_smoke.py` | `--level 0\|1\|2` |
| `examples/claudecode_ags/env/slime_ags.env` | 手动填 `SLIME_ADAPTER_PUBLIC_URL` |

默认镜像：`085995317762.dkr.ecr.ap-southeast-3.amazonaws.com/sn5/jiaxicao/slime:cc-ags-swe-20260710-171230`

## 2. Phase 1 — 怎么部署

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/sglang_adapter

HF_CHECKPOINT=/mnt/sn-007/jiaxicao/<your-hf-model> \
  NUM_GPUS=1 TP_SIZE=1 \
  ./submit_deploy.sh

kubectl -n sn5-system-intern get pods -l app=jiaxicao-cc-ags-adapter -w
kubectl -n sn5-system-intern logs -f deploy/jiaxicao-cc-ags-adapter
kubectl -n sn5-system-intern get ingress jiaxicao-cc-ags-adapter -w
```

ADDRESS 出现后：

```bash
export SLIME_ADAPTER_PUBLIC_URL=http://<ADDRESS>
curl -fsS "$SLIME_ADAPTER_PUBLIC_URL/health"
# 写入 examples/claudecode_ags/env/slime_ags.env（勿把密钥提交进 git）
```

删除：

```bash
./submit_deploy.sh --delete
```

## 3. Phase 2 — 怎么跑 L2

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe
source examples/claudecode_ags/env/load_env.sh

.venv/bin/python -m examples.claudecode_ags.smoke.ags_smoke \
  --level 2 \
  --dataset-type swebench_verified \
  --data-path /mnt/sn-007/jiaxicao/datasets/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --instance-id astropy__astropy-12907 \
  --allow-unresolved
```

## 4. 怎么监控

- Pod：`kubectl -n sn5-system-intern logs -f deploy/jiaxicao-cc-ags-adapter`
- Ingress：`kubectl -n sn5-system-intern get ingress jiaxicao-cc-ags-adapter -o wide`
- L2：看 `[ags-smoke]` 日志；health_gate → workspace_init → claude_exit → diff_chars → resolved

## 5. 验收

| 阶段 | 标准 |
|------|------|
| Phase 1 | Pod Ready；`curl $SLIME_ADAPTER_PUBLIC_URL/health` → 200 |
| Phase 2 | 链路跑完无 exit 2；`--allow-unresolved` 下 exit 0 即链路通过；`resolved=True` 为加分 |
