# The toolchain of cloud/setup_node.sh as a container: NVIDIA's NVHPC 25.9 image (nvfortran, CUDA,
# HPC-X Open MPI, under the same /opt/nvidia/hpc_sdk paths, so scripts/env_nvhpc.sh and build.sh work
# unchanged) plus the Ubuntu packages of the CPU-only builds and CMake 3.31.6. See README.md, "Container".
FROM nvcr.io/nvidia/nvhpc:25.9-devel-cuda_multi-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        build-essential gfortran libopenmpi-dev openmpi-bin libfftw3-dev libopenblas-dev \
        rsync time git curl python3 ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/*
ARG CMAKE_V=3.31.6
RUN mkdir -p /opt/qe-tools && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_V}/cmake-${CMAKE_V}-linux-x86_64.tar.gz \
        | tar -xzf - -C /opt/qe-tools
ENV PATH=/opt/qe-tools/cmake-3.31.6-linux-x86_64/bin:$PATH \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
WORKDIR /work
