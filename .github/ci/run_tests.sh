#!/bin/bash
# The test job of .github/workflows/gpu-tests.yml, runnable by hand on any machine with the toolchain:
# build pw.x and ph.x, run test-suite categories, compare the failures with known-failures.txt.
#
#   bash .github/ci/run_tests.sh                       (from the top of a checkout)
#
#   CI_CATEGORIES  test-suite categories or directories (default "pw_all ph_all"; a directory such as
#                  pw_scf runs as a custom test)
#   CI_FLAVOR      gpu (default) or cpu: which build and which mpirun
#   CI_BUILD_DIR   where to build (default /opt/qe-ci/build/<flavor> if writable, else build/ci-<flavor>);
#                  a directory that persists between runs makes the build incremental
#   CI_SKIP_BUILD  1: the build directory is ready
#   CI_KNOWN       the known-failure list (default .github/ci/known-failures.txt)
#   QE_GPU_ARCHS   GPU architecture of the build (default: read from nvidia-smi, e.g. sm_89 for an L4)
# Logs and the summary go to ci-logs/; the exit code is that of compare_failures.py (0: no new failure).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"     # the checkout (a plain tree works too)
FLAVOR=${CI_FLAVOR:-gpu}; CATS=${CI_CATEGORIES:-pw_all ph_all}; KNOWN=${CI_KNOWN:-$ROOT/.github/ci/known-failures.txt}
LOGS=$ROOT/ci-logs; rm -rf "$LOGS"; mkdir -p "$LOGS"
if [ -z "${CI_BUILD_DIR:-}" ]; then
  if [ -w /opt/qe-ci/build ] || mkdir -p /opt/qe-ci/build 2>/dev/null; then CI_BUILD_DIR=/opt/qe-ci/build/$FLAVOR; else CI_BUILD_DIR=$ROOT/build/ci-$FLAVOR; fi
fi
BLD=$(realpath -m "$CI_BUILD_DIR")
echo "== $FLAVOR build in $BLD ($(nproc) cores)"
if [ "$FLAVOR" = gpu ]; then
  if [ -z "${QE_GPU_ARCHS:-}" ]; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    export QE_GPU_ARCHS=sm_${cc:-90}
  fi
  echo "   GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1), arch $QE_GPU_ARCHS"
  BUILD_FLAVOR=gpu
else
  BUILD_FLAVOR=cpu-gnu
fi
if [ "${CI_SKIP_BUILD:-0}" != 1 ]; then
  t0=$(date +%s)
  BUILD_JOBS=${BUILD_JOBS:-$(nproc)} gpu-quickstart/scripts/build.sh "$ROOT" "$BLD" $BUILD_FLAVOR pw ph > "$LOGS/build-$FLAVOR.log" 2>&1 \
    || { tail -n 40 "$LOGS/build-$FLAVOR.log"; echo "BUILD FAILED (ci-logs/build-$FLAVOR.log)"; exit 1; }
  echo "   built in $(( $(date +%s) - t0 )) s"
fi
[ -x "$BLD/bin/pw.x" ] && [ -x "$BLD/bin/ph.x" ] || { echo "no pw.x / ph.x in $BLD/bin"; exit 1; }

# ---- the test-suite ------------------------------------------------------------------------
T=$ROOT/test-suite
export ESPRESSO_ROOT=$ROOT ESPRESSO_BUILD=$BLD ESPRESSO_PSEUDO=$ROOT/pseudo TESTCODE_DIR=$T/testcode \
       NETWORK_PSEUDO=https://pseudopotentials.quantum-espresso.org/upf_files/ OMP_NUM_THREADS=1 QE_USE_MPI=1
if [ "$FLAVOR" = gpu ]; then
  # executables of an HPC-X build start only under mpirun; the wrapper also forwards QE_* switches
  source gpu-quickstart/scripts/env_nvhpc.sh
  W=$LOGS/mpiwrap; mkdir -p "$W"
  printf '#!/bin/bash\nargs=(); for v in "${!QE_@}"; do args+=(-x "$v"); done\nexec %s/comm_libs/mpi/bin/mpirun "${args[@]}" "$@"\n' "$NVHPC_ROOT" > "$W/mpirun"
  chmod 755 "$W/mpirun"; export PATH=$W:$PATH; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi
( cd "$T" && make prolog > "$LOGS/prolog.log" 2>&1 ) || { cat "$LOGS/prolog.log"; echo "test-suite prolog failed"; exit 1; }
for c in $CATS; do ( cd "$T" && ./check_pseudo.sh "${c%%_*}_" >> "$LOGS/prolog.log" 2>&1 ); done
# categories run at the same time (their test directories are disjoint; the jobs are one rank
# with one thread, and a single GPU holds all of them): the wall time is that of the longest.
# CI_SERIAL=1 runs them one after the other.
rc_all=0; t0=$(date +%s)
run_cat() {   # <category of jobconfig, or a test directory: testcode takes either as --category>
  local c=$1 t1; t1=$(date +%s)
  ( cd "$T" && timeout "${CI_TIMEOUT:-5400}" python3 "$T/testcode/bin/testcode.py" --verbose --category="$c" > "$LOGS/$c.log" 2>&1 )
  echo "== $c: $(( $(date +%s) - t1 )) s, $(grep -E 'All done' "$LOGS/$c.log" | tail -1)"
}
for c in $CATS; do
  if [ "${CI_SERIAL:-0}" = 1 ]; then run_cat "$c"; else run_cat "$c" & fi
done
wait
for c in $CATS; do
  grep -q -E "All done" "$LOGS/$c.log" || { echo "== $c: testcode ended without a summary: $(tail -n 3 "$LOGS/$c.log" | tr '\n' ' ')"; rc_all=1; }
done
echo "== test-suite: $(( $(date +%s) - t0 )) s"
python3 .github/ci/compare_failures.py --known "$KNOWN" --summary "$LOGS/summary.md" "$LOGS"/*.log; rc=$?
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && cat "$LOGS/summary.md" >> "$GITHUB_STEP_SUMMARY"
[ "$rc_all" = 0 ] || exit 1
exit $rc
