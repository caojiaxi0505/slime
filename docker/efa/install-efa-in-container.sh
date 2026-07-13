#!/usr/bin/env bash
# ------------------------------------------------------------------
# 在容器内部一键安装 AWS EFA + OpenMPI 依赖
# 用法:  sudo ./install-efa-in-container.sh  [-y] [-v <efa_version>]
# ------------------------------------------------------------------
set -euo pipefail

EFA_VERSION="${EFA_VERSION:-1.48.0}"
SKIP_CONFIRM=false
while getopts "yv:" opt; do
  case $opt in
    y) SKIP_CONFIRM=true ;;
    v) EFA_VERSION="$OPTARG" ;;
    *) echo "Usage: $0 [-y] [-v <efa_version>]"; exit 1 ;;
  esac
done

# 颜色辅助
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

# 确认在容器内（Kaniko 等构建环境可能没有 /.dockerenv）
if [[ ! -f /.dockerenv && ! -f /proc/1/cgroup ]] || ! grep -Eq 'docker|lxc|oci|containerd|kubepods' /proc/1/cgroup 2>/dev/null; then
  echo -e "${RED}警告：未检测到典型容器环境标记，继续安装（镜像构建场景常见）。${NC}"
  if [[ "$SKIP_CONFIRM" != true ]]; then
    read -rp "仍要继续吗? [y/N] " ans
    [[ "${ans:-N}" == [Yy] ]] || exit 1
  fi
fi

echo -e "${GREEN}==> 安装 EFA ${EFA_VERSION} (容器模式)${NC}"

# 1. 换 Ubuntu 22.04 官方源
cat > /etc/apt/sources.list <<'EOF'
deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
EOF

# 2. 安装依赖
apt-get update && apt-get install -y \
  curl ninja-build autoconf build-essential pciutils environment-modules \
  tcl tcl-dev libnl-3-dev libnl-route-3-dev dmidecode ethtool iproute2 \
  libevent-dev libhwloc-dev openssh-server openssh-client systemd udev \
  && rm -rf /var/lib/apt/lists/*

# 3. SSH 配置（容器内常用）
mkdir -p /var/run/sshd
sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config
echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config
sed -i 's/#\(StrictModes \).*/\1no/g'   /etc/ssh/sshd_config

# 4. 清理可能与 EFA 冲突的 HPC-X
rm -rf /opt/hpcx /usr/local/mpi /etc/ld.so.conf.d/hpcx.conf
ldconfig

# 5. 下载并安装 EFA
INSTALLER="aws-efa-installer-${EFA_VERSION}.tar.gz"
cd /tmp
curl -fsSL -O "https://efa-installer.amazonaws.com/${INSTALLER}"
tar -xf "${INSTALLER}"
cd aws-efa-installer
./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc

# 6. 写入环境变量
cat > /etc/profile.d/efa.sh <<EOF
# EFA/OpenMPI environment
export PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:\$PATH"
export LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"

# OpenMPI
export OMPI_MCA_pml=^ucx
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent
export OPAL_PREFIX=/opt/amazon/openmpi

# EFA/NCCL
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_PROTO=simple
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi-tuner.so
export PMIX_MCA_gds=hash
EOF

# 7. 当前 shell 立即生效
# shellcheck source=/dev/null
source /etc/profile.d/efa.sh

# 8. 验证（镜像构建时通常无 EFA 硬件，只检查用户态是否装上）
echo -e "${GREEN}==> 安装完成，验证中...${NC}"
if command -v fi_info &>/dev/null; then
  if fi_info -p efa -t FI_EP_RDM &>/dev/null; then
    echo -e "${GREEN}EFA 运行正常，fi_info 检测到 efa provider。${NC}"
  else
    echo -e "${GREEN}EFA 用户态已安装；当前环境无 EFA 设备（镜像构建/无硬件时正常），跳过 fi_info 硬件检测。${NC}"
    ls /opt/amazon/efa/lib >/dev/null
    test -x /opt/amazon/efa/bin/fi_info || test -x /opt/amazon/openmpi/bin/fi_info || command -v fi_info
  fi
else
  echo -e "${RED}fi_info 未安装，EFA 用户态可能不完整。${NC}"
  exit 1
fi

echo -e "${GREEN}全部步骤执行完毕，可重启容器或重新登录 shell 使环境变量全局生效。${NC}"
