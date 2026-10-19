# Source this for the toolchain of every GPU build and run: NVHPC 25.9 (nvfortran, CUDA 13.0, HPC-X Open MPI).
# NVHPC_ROOT may be set beforehand (another version or location); the timings were taken with 25.9.
export NVHPC_ROOT=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.9}
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64:${LD_LIBRARY_PATH:-}
