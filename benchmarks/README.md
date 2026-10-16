# GPU benchmarks

How the timings of this repository's GPU work were made, and how to make them again on your own
machine: the `ph.x` (DFPT phonons) series of pull requests #6 to #10 was measured here against
stock `develop`, case by case, and every number in those pull requests comes out of the scripts
in this directory. The same scripts validate the next change: build a stock tree and a modified
tree the same way, run the same inputs from the same ground state, and compare results and
times.

Everything is relative to this directory (`benchmarks/`): sources under `wt/`, builds under
`build/`, results under `runs/ph/<tag>/`, pseudopotentials under `ph/pseudo/`. All of those are
produced locally and ignored by git.

## The machine

The published numbers come from one Google Compute Engine `a3-highgpu-4g` (4x NVIDIA H100 80 GB
HBM3, Intel Sapphire Rapids, 104 vCPUs, 936 GB RAM), running the Deep Learning VM image
`common-cu129-ubuntu-2204-nvidia-580-v20261001` (Ubuntu 22.04, NVIDIA driver 580.178.04). Each
timing is one `ph.x` on one GPU, so a machine with one H100 (`a3-highgpu-1g`) gives the same
numbers one case at a time. `cloud/create_vm.sh` creates such a VM in your project (H100 capacity
usually comes through flex-start scheduling: the VM queues for capacity, runs for a bounded time,
then stops; the script says so). Any other host with an H100 and a driver that supports CUDA 13
works the same way; a different GPU gives different times, the results should still agree.

## Toolchain

Two ways to the same compilers and libraries:

* **On the machine** (what produced the numbers): `sudo cloud/setup_node.sh`, idempotent. NVHPC
  25.9 from NVIDIA's apt repository (nvfortran, CUDA 13.0, HPC-X Open MPI) for the GPU builds;
  gfortran 11.4, Open MPI 4.1.2, FFTW 3.3.8 and OpenBLAS 0.3.20 (Ubuntu 22.04 packages) for the
  CPU-only builds every tree must also pass; CMake 3.31.6 into `tools/`. About 20 minutes, most of
  it NVHPC.
* **In a container** (`cloud/Dockerfile`): NVIDIA's NVHPC 25.9 image plus the same Ubuntu packages
  and CMake. On a host with docker and the NVIDIA container toolkit (the Deep Learning VM image has
  the toolkit; `sudo apt-get install -y docker.io && sudo nvidia-ctk runtime configure
  --runtime=docker && sudo systemctl restart docker` finishes the setup):

  ```bash
  docker build -t qe-nvhpc:25.9 benchmarks/cloud
  docker run --rm -it --gpus all --ipc=host --pid=host -v "$PWD":/work -w /work/benchmarks qe-nvhpc:25.9
  ```

  run from the top of a clone; the repository is `/work`, `--ipc=host` gives the container the
  host's `/dev/shm` (the scratch of every run) and `--pid=host` lets the run scripts pin cores.
  Inside, every command below works unchanged. Checked on the measured machine: a `ph.x` built in
  the container gives the bare-metal results and times (two reduced cases: identical digits,
  0.7 s and 1.6 s in both).

Flags, for both: `scripts/build.sh` (`gpu`: OpenACC + CUDA Fortran for `sm_90`, MPI, OpenMP, NVHPC
BLAS/LAPACK; `cpu-gnu`: `-march=native`, FFTW3, OpenBLAS). `scripts/env_nvhpc.sh` is the
environment of every GPU build and run.

## Running

```bash
git clone <this repository> dm_qe && cd dm_qe/benchmarks
sudo cloud/setup_node.sh                       # or the container above
scripts/fetch_pseudos.sh                       # 12 files from pseudo-dojo.org and the QE site, checksums checked
git worktree add wt/stock develop              # the stock tree
git worktree add wt/pr <branch>                # the tree to validate
scripts/validate_ph.sh stock pr <tag>          # hours: run it detached (nohup, tmux)
cat runs/ph/<tag>/validate.status              # the current step, DONE, or <step>-failed
```

`validate_ph.sh` builds `build/stock-gpu`, `build/pr-gpu` and `build/pr-cpu-gnu` (one configure at
a time: the first configure of a tree clones QE's `external/` submodules), then runs the 15
reduced cases with the stock build and with the PR build in three modes, the 13 benchmark cases
with both builds, and writes the comparison tables `compare-strict.txt`, `compare-default.txt`,
`compare-off.txt` and `compare-bench.txt`. With four GPUs the builds take about 30 minutes, the
reduced cases 15, the benchmarks 3 hours (almost all of it stock `ph.x`); with one GPU, four times
the run times. `--steps mini` skips the benchmarks, `--mini "m_si_nc_gamma m_si_nc_q"` is a
two-minute check that machine, builds and scripts are in order.

The QE test-suite, with the GPU build of a tree (`ph_all`: 89 jobs, 15 to 20 minutes):

```bash
scripts/qe_testsuite.sh stock <tag>            # develop passes 86 of 89; the other three fail before any change
scripts/qe_testsuite.sh pr <tag>               # must give the same set; tests-pr-ph_all.log ends with the summary
```

One case, one build, by hand (an exclusive GPU, pinned cores, the cached ground state):

```bash
SCF_BUILD=build/stock-gpu scripts/run_ph.sh build/pr-gpu ph/cases/m_si_nc_q runs/ph/x/m_si_nc_q
QE_PH_SP=0 QE_PH_NZ=0 PH_ENV="QE_PH_SP QE_PH_NZ" scripts/run_ph.sh build/pr-gpu ph/cases/m_si_nc_q runs/ph/y/m_si_nc_q
NGPU=4 NMPI=4 scripts/run_ph.sh build/pr-gpu ph/cases/si_nc_disp runs/ph/z/si_nc_disp -nk 4
python3 scripts/ph_compare.py runs/ph/x/m_si_nc_q runs/ph/y/m_si_nc_q      # or --suite <root> <root>
python3 scripts/ph_clocks.py runs/ph/x/m_si_nc_q/ph.out                    # the clocks of one run
```

## The protocol

What makes two runs comparable, and what the scripts enforce:

* Stock and modified trees are built with the same flavor and flags, and read the same ground
  state: `run_ph.sh` makes it once per case with the stock `pw.x` (`SCF_BUILD`) and gives every
  `ph.x` run a private copy.
* One `ph.x` per GPU: one MPI process, 12 OpenMP threads, pinned to the 13 physical cores that
  belong to that GPU (the vCPUs of the machine divided by its GPUs; `run_ph.sh` computes the
  split), the GPU locked for the run. Four cases run at once on the 4-GPU machine, each on its own
  GPU and cores. Nothing else runs on the machine.
* Scratch in `/dev/shm`; only the text output, the dynamical matrices and the clocks are kept.
* The time is the `PHONON ... WALL` clock of `ph.x`: everything `ph.x` does, the non-scf step at
  k+q included; the ground state is not timed.
* Spread between two runs of the same binary: a few per cent in wall time; 1e-5 cm⁻¹ in
  frequencies, up to 3e-3 cm⁻¹ on acoustic modes at Γ in mixed precision (`QE_PH_SP=0` is
  reproducible to the digit except in spin-polarized GGA, where a 1e-5 cm⁻¹ spread remains).

## The cases

`ph/cases/<name>/{pw.in,ph.in,case.meta}`. Cases that share a ground state name the same `scf`
key in `case.meta`, so it is computed once. Pseudopotentials: PseudoDojo NC SR v0.4 PBE standard,
pslibrary 1.0.0 PBEsol PAW and ultrasoft (`scripts/fetch_pseudos.sh`, the checksums of the files
used are in the script).

| benchmark case | what is computed |
|---|---|
| `aln_paw_eps` | AlN (4 atoms, PAW), ε∞ and Z* at Γ |
| `aln_paw_gamma` | AlN PAW, ε∞, Z* and phonons at Γ |
| `aln_paw_q` | AlN PAW, phonons at a low-symmetry q |
| `c_paw_q` | diamond PAW, 11×11×11 k, low-symmetry q |
| `si_us_disp` | Si ultrasoft, 4×4×4 q dispersion (8 q) |
| `si_nc_disp` | Si norm-conserving, 4×4×4 q dispersion |
| `mgo_nc_disp` | MgO norm-conserving, ε∞, Z*, 4×4×4 q dispersion |
| `mos2_nc_2d_m` | MoS₂ monolayer, 2D Coulomb cutoff, q = M |
| `al_paw_metal_x` | Al metal PAW, 16×16×16 k, q = X |
| `al_paw_metal_q` | Al metal PAW, low-symmetry q (4096 k-points) |
| `si54_us_gamma` | 54-atom Si cell, ultrasoft, 6 of 162 modes |
| `nv64_nc_lsda_gamma` | NV⁻ centre in 63-atom diamond, LSDA, 4 modes |
| `si216_nc_gamma` | 216-atom Si, norm-conserving, 2 modes |

The 15 reduced cases `m_*` are the same physics at a size that runs in seconds (Si NC and
ultrasoft at Γ, X, a low-symmetry q and a 2×2×2 dispersion; AlN PAW at Γ and q; MgO NC at Γ; Al
metal at Γ, X and q; Ni LSDA at X; the NV⁻ cell with one representation).

## What to expect

Stock `develop` (6e773a63) against the head of the `ph.x` series (#10), one H100 each, the
benchmark thresholds of the inputs:

| case | stock (s) | #10 (s) | speedup | DFPT iterations | max Δfreq (cm⁻¹) | Δdyn (rel) |
|---|---:|---:|---:|---|---:|---:|
| `al_paw_metal_q` | 1038.2 | 15.0 | 69x | 30/30 | 2.7e-4 | 2.3e-6 |
| `al_paw_metal_x` | 55.8 | 2.5 | 22x | 11/11 | 5.7e-3 | 2.2e-5 |
| `aln_paw_eps` | 52.2 | 4.1 | 12.7x | 8/8 | ε∞ 1.7e-7, Z* 2.0e-5 | |
| `aln_paw_gamma` | 158.7 | 6.4 | 25x | 70/69 | 0.60 (acoustic at Γ) | 1.0e-5 |
| `aln_paw_q` | 1639.5 | 25.5 | 64x | 132/132 | 5.1e-3 | 3.0e-5 |
| `c_paw_q` | 1256.4 | 11.6 | 108x | 62/66 | 5.3e-2 | 3.9e-5 |
| `mgo_nc_disp` | 882.4 | 14.1 | 62x | 244/244 | 0.11 | 1.2e-4 |
| `mos2_nc_2d_m` | 231.7 | 13.8 | 16.8x | 73/72 | 0.59 | 1.7e-3 |
| `nv64_nc_lsda_gamma` | 77.2 | 6.2 | 12.5x | 43/43 | 3.6e-2 | 7.5e-5 |
| `si216_nc_gamma` | 366.1 | 34.9 | 10.5x | 24/24 | 9.3e-3 | 6.2e-5 |
| `si54_us_gamma` | 3840.0 | 55.6 | 69x | 66/66 | (6 of 162 modes, no frequencies) | |
| `si_nc_disp` | 532.5 | 7.8 | 69x | 227/230 | 0.22 | 5.8e-5 |
| `si_us_disp` | 779.3 | 25.0 | 31x | 212/212 | 1.0 (acoustic at Γ) | 2.2e-4 |
| **total** | **10,910** | **222.5** | **49x** (geometric mean 33.6x) | | | |

Reduced cases, same builds: strict mode (`QE_PH_SP=0 QE_PH_NZ=0`) gives the DFPT iteration
counts of stock in all 15 and identical digits in the 8 cases without a non-scf step, within
1.6e-3 cm⁻¹ at q ≠ 0 (the rounding of the batched non-scf step); 624.8 s → 30.0 s. Default mode:
25.8 s, differences at the convergence error of the calculation (up to 1.4 cm⁻¹ on acoustic modes
at Γ). Everything off (`QE_PH_DEV=0 QE_BETA_CACHE=0`): stock digits in 12 of 15 cases at stock
speed. Test-suite: 86 of 89 for both.

How to read differences: at the loose thresholds of these inputs the new preconditioner changes
the path of the self-consistency, and the dynamical matrix differs from stock by the convergence
error (about 1e-4 relative, up to 1 cm⁻¹ on acoustic modes at Γ; stock is as far from the
converged answer). At `tr2_ph = 1e-19` the two agree to 1e-6 relative with the same iteration
counts. `compare-strict.txt` is the exactness check: same iterations, same digits where no
non-scf step runs.

## Switches

Every piece of the series has a run-time switch, read from the environment, and falls back to
the stock routine: `QE_PH_DEV=0` (the whole device engine), `QE_BETA_CACHE=0`, `QE_USAUG=0`,
`QE_PH_DYN=0`, `QE_PH_NSCF_DEV=0`; inside the engine `QE_PH_SP=0` (double precision only),
`QE_PH_NZ=0` (no empty bands in the preconditioner), `QE_PH_LOWG=0` (stock preconditioner),
`QE_PH_REAL=0` (complex arithmetic at k = -k); self-checks `QE_PH_CHECK_RES=1`,
`QE_USAUG_CHECK=1`, `QE_PH_DYNCHECK=1`, `QE_PAW_DPOT_CHECK=1`. The pull requests describe each.
`run_ph.sh` forwards the variables named in `PH_ENV` to `ph.x`; `qe_testsuite.sh` forwards every
`QE_*` variable.
