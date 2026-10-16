---
name: gpu-benchmarks
description: Reproduce or extend the GPU timings of this repository's pull requests (ph.x DFPT series) — set up a machine or container with the published toolchain, build a stock and a modified tree, run the reduced cases, the benchmarks and the QE test-suite, and compare. Use for validating a GPU change, measuring a speedup, or checking a result against the published tables.
---

# GPU benchmarks

The utility is `benchmarks/`; its `README.md` has the machine, the toolchain, the protocol, the
cases, the published numbers and how to read a comparison. Work from `benchmarks/`.

```bash
sudo cloud/setup_node.sh                      # toolchain on an Ubuntu 22.04 + NVIDIA 580 machine (idempotent)
                                              # or: docker build -t qe-nvhpc:25.9 cloud && docker run --rm -it --gpus all --ipc=host --pid=host -v "$PWD/..":/work -w /work/benchmarks qe-nvhpc:25.9
scripts/fetch_pseudos.sh                      # the 12 pseudopotentials, checksums as published
git worktree add wt/stock develop; git worktree add wt/pr <branch>
scripts/validate_ph.sh stock pr <tag> [--steps build,mini,bench] [--mini "…"] [--no-cpu]   # detached; hours with the benchmarks
cat runs/ph/<tag>/validate.status; cat runs/ph/<tag>/compare-*.txt
scripts/qe_testsuite.sh stock <tag>; scripts/qe_testsuite.sh pr <tag>                     # 86 of 89 expected for both
SCF_BUILD=build/stock-gpu scripts/run_ph.sh build/pr-gpu ph/cases/<case> runs/ph/<tag>/<case>   # one case
python3 scripts/ph_compare.py --suite <ref-root> <new-root>
```

What to report for a pull request: the `compare-bench.txt` table (stock and new seconds,
speedup, DFPT iterations, max Δfreq, Δdyn, Δε∞, ΔZ*), the strict-mode line of
`compare-strict.txt` (identical digits where no non-scf step runs), default and off modes, and the
two `All done` lines of the test-suite logs. Pull requests #6 to #10 are the model.

Things to know: configure one tree at a time (QE's CMake clones `external/` on the first
configure); the test-suite runs one category of a tree at a time; scratch is `/dev/shm`, results
are under `runs/`; a flex-start VM bills while it runs and stops by itself at its run limit.
