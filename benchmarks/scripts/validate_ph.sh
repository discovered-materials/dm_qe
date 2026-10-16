#!/bin/bash
# Validate a ph.x source tree against a stock tree: builds, the 15 reduced cases in four modes, the
# 13 benchmark cases, comparison tables. This is how the GPU pull requests of this repository were
# validated (benchmarks/README.md).
#
#   validate_ph.sh <stock> <pr> <tag> [--steps build,mini,bench] [--mini "m_a m_b"] [--bench "a b"]
#                  [--no-cpu] [--jobs N]
#
#   <stock>, <pr>  source trees: names under benchmarks/wt/ or paths (git worktrees are convenient)
#   builds         benchmarks/build/<stock>-gpu, <pr>-gpu and, unless --no-cpu, <pr>-cpu-gnu (gfortran,
#                  no CUDA: every tree must compile there too); one tree configured at a time
#   <tag>          benchmarks/runs/ph/<tag>/: stock-mini, strict-mini, default-mini, off-mini, bench,
#                  stock-bench, compare-{strict,default,off,bench}.txt, validate.log, validate.status
#
# Modes of the reduced cases: the stock build; the PR build in strict mode (QE_PH_SP=0 QE_PH_NZ=0,
# the arithmetic path of stock), in default mode, and with everything off (QE_PH_DEV=0
# QE_BETA_CACHE=0). One case per GPU, one MPI process with 12 threads, the ground state made once by
# the stock pw.x and shared. The benchmarks take about 4 hours of stock ph.x time on 4 GPUs (15 h on
# one): run it detached (nohup / tmux). validate.status is the current step, then DONE or <step>-failed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd); cd "$ROOT"
[ $# -ge 3 ] || { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
STOCK=$1; PR=$2; TAG=$3; shift 3
STEPS=build,mini,bench; MINI=""; BENCH=""; CPU=1; JOBS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS=$2; shift 2;; --mini) MINI=$2; shift 2;; --bench) BENCH=$2; shift 2;;
    --no-cpu) CPU=0; shift;; --jobs) JOBS=$2; shift 2;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
tree() { [ -d "wt/$1" ] && echo "wt/$1" || realpath "$1"; }
SRC_STOCK=$(tree "$STOCK"); SRC_PR=$(tree "$PR"); NS=$(basename "$STOCK"); NP=$(basename "$PR")
OUT=runs/ph/$TAG; mkdir -p "$OUT"
LOG=$OUT/validate.log; STATUS=$OUT/validate.status
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
step() { echo "$1" > "$STATUS"; log "== $1"; }
fail() { step "$1-failed"; log "see $2"; exit 1; }
has() { case ",$STEPS," in *,$1,*) return 0;; esac; return 1; }
log "validate_ph.sh $STOCK $PR $TAG steps=$STEPS host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')"
[ -n "$MINI" ]  || MINI=$(ls ph/cases | grep '^m_')
[ -n "$BENCH" ] || BENCH=$(ls ph/cases | grep -v '^m_')
[ -d ph/pseudo ] && ls ph/pseudo/*.upf > /dev/null 2>&1 || scripts/fetch_pseudos.sh >> "$LOG" 2>&1 || fail pseudopotentials "$LOG"

if has build; then
  mkdir -p build
  for spec in "$NS gpu $SRC_STOCK" "$NP gpu $SRC_PR" "$NP cpu-gnu $SRC_PR"; do
    set -- $spec; name=$1; flavor=$2; src=$3
    [ "$flavor" = cpu-gnu ] && [ "$CPU" = 0 ] && continue
    step "build-$name-$flavor"
    scripts/build.sh "$src" "build/$name-$flavor" "$flavor" pw ph > "build/$name-$flavor.log" 2>&1 \
      || fail "build-$name-$flavor" "build/$name-$flavor.log"
    log "build/$name-$flavor: $(tail -1 build/$name-$flavor.log)"
  done
fi
[ -x "build/$NS-gpu/bin/ph.x" ] && [ -x "build/$NP-gpu/bin/ph.x" ] || fail "builds" "build/$NS-gpu, build/$NP-gpu"

export SCF_BUILD=$ROOT/build/$NS-gpu
[ -n "$JOBS" ] && export JOBS
suite() { # <label> <build> <run-root> <cases> [env…]
  local label=$1 bld=$2 run=$3 cases=$4; shift 4
  step "$label"
  env "$@" scripts/ph_suite.sh "$bld" "$TAG/$run" $cases > "$OUT/$run.suite.log" 2>&1; log "$label rc=$? ($(tail -1 "$OUT/$run.suite.log"))"
}
if has mini; then
  export TIMEOUT=3600
  suite mini-stock   "build/$NS-gpu" stock-mini   "$MINI"
  suite mini-strict  "build/$NP-gpu" strict-mini  "$MINI" QE_PH_SP=0 QE_PH_NZ=0 PH_ENV="QE_PH_SP QE_PH_NZ"
  suite mini-default "build/$NP-gpu" default-mini "$MINI"
  suite mini-off     "build/$NP-gpu" off-mini     "$MINI" QE_PH_DEV=0 QE_BETA_CACHE=0 PH_ENV="QE_PH_DEV QE_BETA_CACHE"
  step compare-mini
  for t in strict default off; do python3 scripts/ph_compare.py --suite "$OUT/stock-mini" "$OUT/$t-mini" > "$OUT/compare-$t.txt" 2>&1; done
  log "compare-mini: $(tail -1 "$OUT/compare-strict.txt")"
fi
if has bench; then
  export TIMEOUT=7200
  suite bench-pr    "build/$NP-gpu" bench       "$BENCH"
  suite bench-stock "build/$NS-gpu" stock-bench "$BENCH"
  step compare-bench
  python3 scripts/ph_compare.py --suite "$OUT/stock-bench" "$OUT/bench" > "$OUT/compare-bench.txt" 2>&1
  log "compare-bench: $(tail -1 "$OUT/compare-bench.txt")"
fi
step DONE
