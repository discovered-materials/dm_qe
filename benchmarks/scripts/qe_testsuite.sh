#!/bin/bash
# The QE test-suite (testcode) with a GPU build. Stock develop passes 86 of the 89 ph_* jobs (ph_metal
# al.elph, ph_multipole and ph_raman fail before any change); a pull request must give the same set.
#
#   qe_testsuite.sh <tree> <tag> [category] [gpu]
#
#   <tree>     name under benchmarks/wt/ or a path; the build is benchmarks/build/<tree>-gpu
#   category   testcode category, default ph_all (pw_all, ph_restart, ph_ahc_bas, … see test-suite/jobconfig)
#   gpu        CUDA device, default 0
#   output     benchmarks/runs/ph/<tag>/tests-<tree>-<category>.log (the summary is its last lines),
#              tests-<tree>-prolog.log, tests-<tree>.status (running, then "done rc=N")
#
# Executables of an HPC-X build only start under mpirun, so the suite runs with QE_USE_MPI=1 through
# benchmarks/tools/mpiwrap/mpirun, which forwards every QE_* switch (set e.g. QE_PH_DEV=0 to test the
# stock path of a PR build). Pseudopotentials the jobs need are downloaded once into <tree>/pseudo.
# The suite writes into <tree>/test-suite: one category of a tree at a time.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd); cd "$ROOT"
[ $# -ge 2 ] || { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
TREE=$1; TAG=$2; CAT=${3:-ph_all}; GPU=${4:-0}
SRC=$([ -d "wt/$TREE" ] && echo "$ROOT/wt/$TREE" || realpath "$TREE"); N=$(basename "$TREE")
BLD=$ROOT/build/$N-gpu; [ -x "$BLD/bin/pw.x" ] || { echo "no $BLD/bin/pw.x (scripts/build.sh <tree> build/$N-gpu gpu)" >&2; exit 2; }
T=$SRC/test-suite; OUT=$ROOT/runs/ph/$TAG; mkdir -p "$OUT"
ST=$OUT/tests-$N.status; echo running > "$ST"
source "$HERE/env_nvhpc.sh"
if [ ! -x "$ROOT/tools/mpiwrap/mpirun" ]; then
  mkdir -p "$ROOT/tools/mpiwrap"
  cat > "$ROOT/tools/mpiwrap/mpirun" <<W
#!/bin/bash
# mpirun of HPC-X with every QE_* environment switch forwarded to the ranks
args=(); while IFS='=' read -r v _; do args+=(-x "\$v"); done < <(env | grep -E '^QE_[A-Z0-9_]+=')
exec $NVHPC_ROOT/comm_libs/mpi/bin/mpirun "\${args[@]}" "\$@"
W
  chmod 755 "$ROOT/tools/mpiwrap/mpirun"
fi
export PATH=$ROOT/tools/mpiwrap:$PATH
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 QE_USE_MPI=1
export ESPRESSO_ROOT=$SRC ESPRESSO_BUILD=$BLD ESPRESSO_PSEUDO=$SRC/pseudo TESTCODE_DIR=$T/testcode \
       NETWORK_PSEUDO=https://pseudopotentials.quantum-espresso.org/upf_files/
mkdir -p "$ESPRESSO_PSEUDO"
( cd "$T" && make prolog > "$OUT/tests-$N-prolog.log" 2>&1 && ./check_pseudo.sh "${CAT%%_*}_" >> "$OUT/tests-$N-prolog.log" 2>&1 ) \
  || { echo "prolog-failed" > "$ST"; exit 1; }
( cd "$T" && timeout "${TEST_TIMEOUT:-7200}" python3 "$T/testcode/bin/testcode.py" --verbose --category="$CAT" > "$OUT/tests-$N-$CAT.log" 2>&1 ); rc=$?
grep -E "All done|Failed tests" "$OUT/tests-$N-$CAT.log" | head -3
echo "done rc=$rc" > "$ST"
exit $rc
