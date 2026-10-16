#!/bin/bash
# Build pw.x / ph.x from a source tree, with the flags of every published timing.
#
#   build.sh <source-dir> <build-dir> <flavor> [target ...]        (default target: pw ph)
#     gpu      nvfortran + OpenACC + CUDA Fortran + HPC-X MPI + OpenMP, sm_90 (H100)
#     cpu-gnu  gfortran + Open MPI + OpenMP + FFTW3 + OpenBLAS (every tree must build without CUDA too)
#
# Stock and modified trees are always built with the same flavor. BUILD_JOBS (default: all cores).
# CMake: benchmarks/tools/cmake-3.31.6-linux-x86_64 if cloud/setup_node.sh put it there, else the PATH.
# The first configure of a tree clones QE's external/ submodules: configure one tree at a time.
set -euo pipefail
SRC=$(realpath "$1"); BLD=$(realpath -m "$2"); FLAVOR=$3; shift 3
TARGETS=("${@:-pw}"); [ $# -eq 0 ] && TARGETS=(pw ph)
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
for c in "$ROOT"/tools/cmake-*-linux-x86_64/bin; do [ -d "$c" ] && export PATH=$c:$PATH; done
mkdir -p "$BLD"
case "$FLAVOR" in
  gpu)
    source "$HERE/env_nvhpc.sh"
    CFG=(-DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc
         "-DQE_GPU=openacc;cuda" -DQE_GPU_ARCHS=${QE_GPU_ARCHS:-sm_90}
         -DQE_ENABLE_MPI=ON -DQE_ENABLE_OPENMP=ON
         -DBLAS_LIBRARIES=$NVHPC_ROOT/compilers/lib/libblas.so
         -DLAPACK_LIBRARIES=$NVHPC_ROOT/compilers/lib/liblapack.so\;$NVHPC_ROOT/compilers/lib/libblas.so)
    ;;
  cpu-gnu)
    CFG=(-DCMAKE_Fortran_COMPILER=/usr/bin/mpif90 -DCMAKE_C_COMPILER=/usr/bin/mpicc
         -DQE_ENABLE_MPI=ON -DQE_ENABLE_OPENMP=ON -DQE_FFTW_VENDOR=FFTW3 -DBLA_VENDOR=OpenBLAS
         "-DCMAKE_Fortran_FLAGS=-march=native")
    ;;
  *) echo "unknown flavor $FLAVOR (gpu | cpu-gnu)" >&2; exit 2;;
esac
if [ ! -f "$BLD/CMakeCache.txt" ]; then
  cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=Release -DQE_ENABLE_TEST=OFF "${CFG[@]}"
fi
cmake --build "$BLD" -j "${BUILD_JOBS:-$(nproc)}" --target "${TARGETS[@]}"
