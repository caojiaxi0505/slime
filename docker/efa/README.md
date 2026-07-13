# EFA scripts for `Dockerfile.kaniko`

Multi-node HyperPod training needs EFA userland inside the image (see HyperPod training guide §5).

## In-tree scripts

This directory contains:

- `install-efa-in-container.sh`
- `fix-efa-conflict.sh`

Canonical location: `docker/efa/` under the slime tree (and the Path A worktree).  
Do **not** keep a separate `code/efa_install/` copy — use this directory, or set `EFA_SCRIPTS_DIR` only when copying from another checkout into the build context.

`ENABLE_EFA=1` (default) requires these two files to be present here before the Kaniko Job runs.

## Skip EFA

For a single-node / non-EFA image only:

```bash
ENABLE_EFA=0 ./submit_build.sh
```

That image will not be suitable for multi-node EFA RDMA training.
