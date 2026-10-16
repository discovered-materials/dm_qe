#!/bin/bash
# Run a set of ph.x benchmark cases, at most JOBS at a time (each on its own GPU).
#
#   ph_suite.sh <build-dir> <run-root> [case ...]      (default: every case in benchmarks/ph/cases)
#
# Results go to benchmarks/runs/ph/<run-root>/<case>. JOBS defaults to the number of GPUs; TIMEOUT
# (seconds, default 7200) bounds each ph.x run (not the wait for a GPU or for the ground state).
# The last line is "SUITE DONE <run-root>", with the number of failed cases if any (exit code 1).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
BLD=$1; TAG=$2; shift 2
CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=($(ls "$ROOT/ph/cases"))
JOBS=${JOBS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')}; [ "${JOBS:-0}" -ge 1 ] || JOBS=1
TIMEOUT=${TIMEOUT:-7200}
OUT=$ROOT/runs/ph/$TAG; mkdir -p "$OUT"
: > "$OUT/suite.status"
for c in "${CASES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  (
    t0=$(date +%s)
    PH_TIMEOUT=$TIMEOUT "$HERE/run_ph.sh" "$BLD" "$ROOT/ph/cases/$c" "$OUT/$c" > "$OUT/$c.log" 2>&1
    rc=$?
    echo "$c $rc" >> "$OUT/suite.status"
    echo "$c rc=$rc elapsed=$(( $(date +%s) - t0 ))s $(tail -1 "$OUT/$c.log" | tr -s ' ')"
  ) &
done
wait
NFAIL=$(awk '$2 != 0' "$OUT/suite.status" | wc -l)
if [ "$NFAIL" -gt 0 ]; then echo "SUITE DONE $TAG: $NFAIL failed"; exit 1; fi
echo "SUITE DONE $TAG"
