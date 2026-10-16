#!/bin/bash
# Run one ph.x benchmark case on an exclusive GPU with pinned cores.
#
#   run_ph.sh <build-dir> <case-dir> <run-dir> [extra ph.x args, e.g. -nk 4]
#
# <case-dir> holds pw.in, ph.in and case.meta (scf=<key>). The ground state is computed once per
# <key> by the pw.x of SCF_BUILD and cached in $PH_SCRATCH/scf/<key>; every ph.x run starts from a
# private copy of it, so stock and modified ph.x read identical wavefunctions. Scratch is in
# /dev/shm; only text (ph.out, dyn files, clocks) goes to <run-dir>. The pseudopotentials are
# benchmarks/ph/pseudo (scripts/fetch_pseudos.sh), passed to pw.x through ESPRESSO_PSEUDO.
#
# Environment (same values for stock and modified runs):
#   SCF_BUILD  build whose pw.x makes the ground state (default benchmarks/build/stock-gpu)
#   NGPU   GPUs to lock (1)      NMPI  MPI ranks (NGPU)     OMP  threads per rank (cores per GPU - 1)
#   GPUS   candidate GPU ids in order of preference (default: every GPU, highest id first)
#   PH_SCRATCH  scratch root (/dev/shm/qe_ph)       KEEP_SCRATCH=1  keep the run's scratch directory
#   PH_ENV      names of extra environment variables to forward to ph.x (the QE_PH_* switches)
#   PH_TIMEOUT  seconds after which ph.x itself is terminated (exit code 124)
#   PH_IN       ph.x input instead of <case-dir>/ph.in
#   PH_OUT_FROM start from a copy of this outdir instead of the cached ground state; PH_OUT_TO keeps
#               the outdir there afterwards
#
# GPUs are locked with flock on /tmp/qe_gpu_<id>.lock, so concurrent runs never share a device. The
# physical cores are split evenly over the GPUs (a3-highgpu machines: 26 vCPUs = 13 cores per H100);
# GPU g owns cores [g*CPG, (g+1)*CPG-1] and the run is pinned there with one core left to the system.
set -euo pipefail
BLD=$(realpath "$1"); CASE=$(realpath "$2"); RUN=$(realpath -m "$3"); shift 3
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
NGPU_TOTAL=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') || NGPU_TOTAL=1
CPG=$(( $(nproc) / NGPU_TOTAL / 2 )); [ "$CPG" -ge 2 ] || CPG=$(( $(nproc) / NGPU_TOTAL )); [ "$CPG" -ge 1 ] || CPG=1
NGPU=${NGPU:-1}; NMPI=${NMPI:-$NGPU}; OMP=${OMP:-$(( CPG > 1 ? CPG - 1 : 1 ))}
SCF_BUILD=$(realpath "${SCF_BUILD:-$ROOT/build/stock-gpu}")
PH_SCRATCH=${PH_SCRATCH:-/dev/shm/qe_ph}
export ESPRESSO_PSEUDO=${ESPRESSO_PSEUDO:-$ROOT/ph/pseudo}
PHX=$BLD/bin/ph.x
[ -x "$PHX" ] || { echo "no ph.x in $BLD/bin" >&2; exit 2; }
[ -x "$SCF_BUILD/bin/pw.x" ] || { echo "no pw.x in $SCF_BUILD/bin (SCF_BUILD)" >&2; exit 2; }
[ -f "$CASE/ph.in" ] && [ -f "$CASE/pw.in" ] || { echo "no pw.in/ph.in in $CASE" >&2; exit 2; }
KEY=$(sed -n 's/^scf=//p' "$CASE/case.meta" 2>/dev/null || true); KEY=${KEY:-$(basename "$CASE")}

# ---- lock GPUs (highest ids first) -----------------------------------------------------------
CAND=(); for ((g=NGPU_TOTAL-1; g>=0; g--)); do CAND+=("$g"); done
[ -n "${GPUS:-}" ] && IFS=, read -ra CAND <<< "$GPUS"
[ "$NGPU" -ge 1 ] && [ "$NGPU" -le "${#CAND[@]}" ] || { echo "NGPU=$NGPU: only ${#CAND[@]} GPUs to choose from" >&2; exit 2; }
exec {qfd}>/tmp/qe_gpu_queue.lock
flock "$qfd"
while true; do
  LOCKED=(); FDS=()
  for g in "${CAND[@]}"; do
    exec {fd}>"/tmp/qe_gpu_$g.lock"
    if flock -n "$fd"; then LOCKED+=("$g"); FDS+=("$fd"); else eval "exec $fd>&-"; fi
    [ "${#LOCKED[@]}" -eq "$NGPU" ] && break
  done
  [ "${#LOCKED[@]}" -eq "$NGPU" ] && break
  for fd in "${FDS[@]}"; do eval "exec $fd>&-"; done
  sleep 1
done
eval "exec $qfd>&-"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${LOCKED[*]}")
CORES=""; for g in "${LOCKED[@]}"; do CORES+="$((g*CPG))-$((g*CPG+CPG-1)),"; done; CORES=${CORES%,}

source "$HERE/env_nvhpc.sh"
export OMP_NUM_THREADS=$OMP OMP_PROC_BIND=${OMP_PROC_BIND:-false}
MPIRUN=(mpirun -np "$NMPI" --bind-to none -x CUDA_VISIBLE_DEVICES -x OMP_NUM_THREADS -x LD_LIBRARY_PATH -x ESPRESSO_PSEUDO)
for v in ${PH_ENV:-}; do [ -n "${!v:-}" ] && MPIRUN+=(-x "$v"); done

# ---- ground state, once per key --------------------------------------------------------------
SCF=$PH_SCRATCH/scf/$KEY
exec {sfd}>"/tmp/qe_ph_scf_$KEY.lock"; flock "$sfd"
if [ ! -f "$SCF/DONE" ] || ! cmp -s "$CASE/pw.in" "$SCF/pw.in"; then
  rm -rf "$SCF"; mkdir -p "$SCF"; cp "$CASE/pw.in" "$SCF/pw.in"
  ( cd "$SCF" && taskset -c "$CORES" mpirun -np 1 --bind-to none -x CUDA_VISIBLE_DEVICES \
      -x OMP_NUM_THREADS -x LD_LIBRARY_PATH -x ESPRESSO_PSEUDO "$SCF_BUILD/bin/pw.x" -i pw.in > pw.out 2> pw.err ) || true
  if grep -q "convergence has been achieved" "$SCF/pw.out" && grep -q "JOB DONE" "$SCF/pw.out"; then
    touch "$SCF/DONE"
  else
    echo "SCF failed for $KEY: see $SCF/pw.out" >&2; exit 3
  fi
fi

# ---- ph.x --------------------------------------------------------------------------------------
# the scratch directory goes away however the script ends (KEEP_SCRATCH=1 keeps it); a signal
# also ends ph.x
WORK=$PH_SCRATCH/run/$(basename "$RUN").$$
CHILD=""
killtree() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do killtree "$p"; done; kill -TERM "$1" 2>/dev/null || true; }
cleanup() { [ "${KEEP_SCRATCH:-0}" = 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT
trap '[ -n "$CHILD" ] && killtree "$CHILD"; exit 143' TERM INT HUP
rm -rf "$WORK"; mkdir -p "$WORK" "$RUN"
cp -r "${PH_OUT_FROM:-$SCF/out}" "$WORK/out"; cp "${PH_IN:-$CASE/ph.in}" "$WORK/ph.in"
eval "exec $sfd>&-"
rm -f "$RUN"/dyn* "$RUN"/ph.out "$RUN"/ph.err
cp "$WORK/ph.in" "$CASE/pw.in" "$RUN/"; cp "$SCF/pw.out" "$RUN/scf.out"
{
  echo "# build=$BLD case=$CASE scf=$KEY scf_build=$SCF_BUILD"
  echo "# NGPU=$NGPU GPUS=$CUDA_VISIBLE_DEVICES NMPI=$NMPI OMP=$OMP CORES=$CORES args=$*"
  echo "# env: $(for v in ${PH_ENV:-}; do echo -n "$v=${!v:-} "; done)"
  echo "# work=$WORK start=$(date -Is) host=$(hostname)"
} > "$RUN/run.meta"
TMO=(); [ -n "${PH_TIMEOUT:-}" ] && TMO=(timeout -s TERM -k 30 "$PH_TIMEOUT")
TIMEV=(); [ -x /usr/bin/time ] && TIMEV=(/usr/bin/time -v -o "$RUN/time.log")
set +e
( cd "$WORK" && exec "${TIMEV[@]}" taskset -c "$CORES" "${TMO[@]}" "${MPIRUN[@]}" "$PHX" "$@" -i ph.in \
    > "$RUN/ph.out" 2> "$RUN/ph.err" ) &
CHILD=$!
wait "$CHILD"
RC=$?
CHILD=""
set -e
cp "$WORK"/dyn* "$RUN/" 2>/dev/null || true
if [ -n "${PH_OUT_TO:-}" ]; then rm -rf "$PH_OUT_TO"; mkdir -p "$(dirname "$PH_OUT_TO")"; cp -r "$WORK/out" "$PH_OUT_TO"; fi
echo "# end=$(date -Is) rc=$RC" >> "$RUN/run.meta"
grep -E "^\s+PHONON\s+:" "$RUN/ph.out" | tail -1 || true
exit $RC
