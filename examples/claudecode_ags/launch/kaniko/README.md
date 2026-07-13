# Kaniko → ECR full slime image

操作手册（怎么执行 / 涉及文件 / 怎么监控 / 怎么验收）：

→ [`docs/superpowers/runbooks/2026-07-11-kaniko-ecr-slime-image-ops.md`](../../../../docs/superpowers/runbooks/2026-07-11-kaniko-ecr-slime-image-ops.md)

## Quick start

```bash
cd /mnt/sn-007/jiaxicao/code/slime/.worktrees/cc-ags-swe/examples/claudecode_ags/launch/kaniko

./submit_build.sh --dry-run          # 预览 YAML
./submit_build.sh                    # 提交（默认 EFA 用户态 + 大资源）

# 监控
kubectl -n sn5-system-intern logs -f job/<JOB_NAME>
kubectl -n sn5-system-intern wait --for=condition=complete --timeout=6h job/<JOB_NAME>

# 验收
aws ecr describe-images --region ap-southeast-3 \
  --repository-name sn5/jiaxicao/slime \
  --image-ids imageTag=<IMAGE_TAG>
```

## Hard rules

- 禁止本机 `docker build` / `docker push`
- 仅 `kubectl -n sn5-system-intern`
- 构建 **不** 申请 EFA 设备；只装用户态。训练再申请 EFA

## Layout

| 文件 | 作用 |
|------|------|
| `submit_build.sh` | 渲染 + apply |
| `job.yaml.template` | Job 模板 |
| `../../../../docker/Dockerfile.kaniko` | 镜像配方 |
| `../../../../docker/efa/*.sh` | EFA 用户态安装 |
