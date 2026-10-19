#!/bin/bash
# Run pw.x or ph.x on GPUs the way the published timings were made: one MPI rank per GPU, OpenMP
# threads on that GPU's share of the cores, the NVHPC / HPC-X environment of scripts/env_nvhpc.sh.
#
#   run_gpu.sh <build-dir> pw|ph <input> [-o <output>] [program arguments, e.g. -nk 2]
#
#   GPUS   comma-separated GPU ids to use, one rank each (default 0)
#   OMP    OpenMP threads per rank (default: the physical cores of one GPU's share minus one; the
#          machine's vCPUs are split evenly over its GPUs, SMT siblings not counted: 12 on an a3-highgpu)
#   PIN=0  do not pin the ranks to their GPUs' cores (taskset)
#   QE_*   every variable named QE_… in the environment is passed to the ranks (the run-time switches
#          of the GPU work, e.g. QE_PH_DEV=0 for the stock path)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 3 ] || { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
BLD=$(realpath "$1"); PROG=$2; INPUT=$3; shift 3
OUT=""; if [ "${1:-}" = -o ]; then OUT=$2; shift 2; fi
EXE=$BLD/bin/$PROG.x; [ -x "$EXE" ] || { echo "no $EXE" >&2; exit 2; }
source "$HERE/env_nvhpc.sh"
GPUS=${GPUS:-0}; IFS=, read -ra G <<< "$GPUS"; NR=${#G[@]}
NGPU_TOTAL=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') || NGPU_TOTAL=1
CPG=$(( $(nproc) / NGPU_TOTAL / 2 )); [ "$CPG" -ge 2 ] || CPG=$(( $(nproc) / NGPU_TOTAL )); [ "$CPG" -ge 1 ] || CPG=1
export CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=${OMP:-$(( CPG > 1 ? CPG - 1 : 1 ))}
CORES=""; for g in "${G[@]}"; do CORES+="$((g*CPG))-$((g*CPG+CPG-1)),"; done; CORES=${CORES%,}
CMD=(mpirun -np "$NR" --bind-to none -x CUDA_VISIBLE_DEVICES -x OMP_NUM_THREADS -x LD_LIBRARY_PATH)
[ -n "${ESPRESSO_PSEUDO:-}" ] && CMD+=(-x ESPRESSO_PSEUDO)
for v in "${!QE_@}"; do CMD+=(-x "$v"); done
CMD+=("$EXE" "$@" -i "$INPUT")
[ "${PIN:-1}" = 0 ] || CMD=(taskset -c "$CORES" "${CMD[@]}")
if [ -n "$OUT" ]; then exec "${CMD[@]}" > "$OUT"; else exec "${CMD[@]}"; fi
