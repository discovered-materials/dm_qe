#!/usr/bin/env bash
# setup_node.sh — the toolchain of the published timings, on an Ubuntu 22.04 machine with NVIDIA
# driver 580 or newer (the Deep Learning VM image of cloud/create_vm.sh has it). Run as root; idempotent.
#
#   NVHPC 25.9 from NVIDIA's apt repository: nvfortran, CUDA 13.0, HPC-X Open MPI   (the GPU builds)
#   gfortran, Open MPI, FFTW3, OpenBLAS from Ubuntu (22.04: 11.4, 4.1.2, 3.3.8, 0.3.20)  (CPU-only builds)
#   CMake 3.31.6, the binary release, in benchmarks/tools (QE's CMake wants 3.20 or newer)
#   rsync, time, git, curl, python3 for the scripts
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
NVHPC_V=25.9; NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/$NVHPC_V; CMAKE_V=3.31.6
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
echo "== packages"
apt-get update -qq
apt-get install -y -qq build-essential gfortran libopenmpi-dev openmpi-bin libfftw3-dev libopenblas-dev \
    rsync time git curl python3 ca-certificates gnupg > /dev/null
if [ ! -d "$NVHPC" ]; then
  echo "== NVHPC $NVHPC_V (several GB, 10-20 minutes)"
  curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
      | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' \
      > /etc/apt/sources.list.d/nvhpc.list
  apt-get update -qq
  apt-get install -y -qq "nvhpc-${NVHPC_V/./-}" > /dev/null
fi
OWNER=${SUDO_USER:-root}
install -d -o "$OWNER" "$ROOT/tools"
if [ ! -x "$ROOT/tools/cmake-$CMAKE_V-linux-x86_64/bin/cmake" ]; then
  echo "== CMake $CMAKE_V"
  curl -fsSL "https://github.com/Kitware/CMake/releases/download/v$CMAKE_V/cmake-$CMAKE_V-linux-x86_64.tar.gz" | tar -xzf - -C "$ROOT/tools"
  chown -R "$OWNER" "$ROOT/tools/cmake-$CMAKE_V-linux-x86_64"
fi
echo "== $(. /etc/os-release; echo "$PRETTY_NAME") $(uname -r), $(nproc) vCPUs, $(free -g | awk '/Mem/{print $2}') GB, /dev/shm $(df -h /dev/shm | awk 'NR==2{print $2}')"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | sed 's/^ */   /' || echo "   no nvidia-smi: install the NVIDIA driver first"
echo "   nvfortran $($NVHPC/compilers/bin/nvfortran --version | sed -n 2p | awk '{print $2}'), CUDA $(ls $NVHPC/cuda | grep -E '^[0-9]' | tr '\n' ' ')"
echo "   HPC-X $($NVHPC/comm_libs/mpi/bin/mpirun --version | head -1)"
echo "   $(gfortran --version | head -1); $(/usr/bin/mpirun --version | head -1); fftw3 $(dpkg-query -W -f='${Version}' libfftw3-dev); openblas $(dpkg-query -W -f='${Version}' libopenblas-dev)"
echo "   $($ROOT/tools/cmake-$CMAKE_V-linux-x86_64/bin/cmake --version | head -1)"
echo "SETUP OK"
