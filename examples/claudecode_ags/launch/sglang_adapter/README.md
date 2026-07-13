# SGLang + adapter deploy (Phase 1)

独立部署推理栈 + 专属 ALB，供 L2 / 训练回调使用。完整操作见：

→ [`docs/superpowers/runbooks/2026-07-11-sglang-adapter-l2-ops.md`](../../../../docs/superpowers/runbooks/2026-07-11-sglang-adapter-l2-ops.md)

## Quick start

```bash
cd examples/claudecode_ags/launch/sglang_adapter

HF_CHECKPOINT=/mnt/sn-007/jiaxicao/path/to/hf_model \
  ./submit_deploy.sh --dry-run

HF_CHECKPOINT=/mnt/sn-007/jiaxicao/path/to/hf_model \
  NUM_GPUS=1 TP_SIZE=1 \
  ./submit_deploy.sh

kubectl -n sn5-system-intern get ingress jiaxicao-cc-ags-adapter -w
# ADDRESS 出现后：
export SLIME_ADAPTER_PUBLIC_URL=http://<ADDRESS>
curl -fsS "$SLIME_ADAPTER_PUBLIC_URL/health"
# 手动写入 env/slime_ags.env
```

## Hard rules

- 仅 `kubectl -n sn5-system-intern`
- 镜像默认用已推 ECR tag（可用 `IMAGE_URI` 覆盖）
- **不**自动改 `slime_ags.env`；URL 手动粘贴
- 构建 Job **不**申请 EFA；本部署默认单机推理，GPU 可配

## Layout

| 文件 | 作用 |
|------|------|
| `entrypoint.sh` | 起 SGLang → adapter |
| `serve_adapter.py` | SegmentedAnthropicAdapter |
| `deployment.yaml.template` | GPU Deployment |
| `service-ingress.yaml.template` | Service + ALB Ingress |
| `submit_deploy.sh` | 渲染 apply |
