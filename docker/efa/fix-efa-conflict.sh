#!/usr/bin/env bash
# ------------------------------------------------------------------
# 继续/重做 EFA 安装，解决 libpmix-aws 与系统 openmpi 冲突
# 用法:  sudo bash fix-efa-conflict.sh  [-y] [-v <efa_version>]
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

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

# 1. 先干掉冲突包（没有也返回 0）
echo -e "${GREEN}--> 移除系统自带 openmpi / pmix 避免文件冲突${NC}"
apt-get remove -y '^openmpi.*' '^prte.*' '^libpmix.*' || true
apt-get autoremove -y

# 2. 如果之前 EFA 装了一半，先清理残局（依赖冲突时尽量多卸）
if dpkg -l | grep -E 'libfabric|libpmix|openmpi.*-aws|efa-profile' &>/dev/null; then
  echo -e "${GREEN}--> 检测到残留 EFA 包，重新安装${NC}"
  # shellcheck disable=SC2046
  dpkg -P $(dpkg -l | awk '/libfabric|libpmix|openmpi.*-aws|efa-profile|openmpi40-aws|openmpi50-aws|libnccl-ofi/ {print $2}') 2>/dev/null || true
  apt-get -f install -y || true
fi

# 3. 重新执行 EFA installer（利用本地已下载的 tarball）
INSTALLER_TAR="/tmp/aws-efa-installer-${EFA_VERSION}.tar.gz"
INSTALLER_DIR="/tmp/aws-efa-installer"

if [[ ! -d "$INSTALLER_DIR" ]]; then
  [[ -f "$INSTALLER_TAR" ]] || curl -fsSL -o "$INSTALLER_TAR" \
       "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz"
  tar -xf "$INSTALLER_TAR" -C /tmp
fi

cd "$INSTALLER_DIR"
# Note: efa_installer.sh has no --force-overwrite; resolve conflicts via purge above.
./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc

# 4. 写入环境变量（如已存在则追加）
cat > /etc/profile.d/efa.sh <<'EOF'
# EFA/OpenMPI environment
export PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH"
export LD_LIBRARY_PATH="/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

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

# 5. 立即生效
source /etc/profile.d/efa.sh

# 6. 验证（镜像构建时通常无 EFA 硬件，只检查用户态是否装上）
echo -e "${GREEN}--> 验证 EFA 安装${NC}"
if command -v fi_info &>/dev/null; then
  if fi_info -p efa -t FI_EP_RDM &>/dev/null; then
    echo -e "${GREEN}fi_info 检测到 efa provider，安装成功！${NC}"
  else
    echo -e "${GREEN}EFA 用户态已安装；当前环境无 EFA 设备（镜像构建时正常）。${NC}"
    ls /opt/amazon/efa/lib >/dev/null
  fi
else
  echo -e "${RED}fi_info 未安装，EFA 用户态可能不完整。${NC}"
  exit 1
fi
