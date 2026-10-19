# GPU quickstart

From nothing to a GPU build of this repository's `pw.x` and `ph.x` and a first calculation, in
the configuration the GPU pull requests (the `ph.x` series, #6 to #10) were measured in: a
Google Compute Engine machine with H100s, NVIDIA's HPC SDK, the build flags, and one example
whose results and times you can compare with the ones below. Everything is relative to this
directory; `tools/`, `build/` and the example's outputs are produced locally and ignored by git.

## The machine

The published numbers come from a GCE `a3-highgpu-4g` (4x NVIDIA H100 80 GB HBM3, Sapphire
Rapids, 104 vCPUs, 936 GB RAM) running the Deep Learning VM image
`common-cu129-ubuntu-2204-nvidia-580-v20261001` (Ubuntu 22.04, NVIDIA driver 580.178.04). A
calculation uses one GPU, so `a3-highgpu-1g` (one H100, 26 vCPUs) gives the same times at a
quarter of the price. `cloud/create_vm.sh` creates such a VM in your own project with your own
`gcloud` login:

```bash
VM=qe-h100 ZONE=us-central1-a MACHINE_TYPE=a3-highgpu-1g gpu-quickstart/cloud/create_vm.sh
gcloud compute ssh qe-h100 --zone us-central1-a
```

H100 capacity usually comes through flex-start scheduling: the VM waits in a queue (minutes to
hours), runs for a bounded time, then stops by itself; its boot disk stays. It bills while it
runs: `gcloud compute instances stop` when you are done. Any other host with an H100 and a driver
that supports CUDA 13 works the same way; another GPU gives other times, the results should still
agree.

## The toolchain

Two ways to the same compilers and libraries:

* **On the machine** (what produced the numbers): `sudo cloud/setup_node.sh`, idempotent, about
  20 minutes. NVHPC 25.9 from NVIDIA's apt repository (nvfortran, CUDA 13.0, HPC-X Open MPI);
  gfortran, Open MPI, FFTW3 and OpenBLAS from Ubuntu for a CPU-only build; CMake 3.31.6 into
  `tools/`.
* **In a container** (`cloud/Dockerfile`): NVIDIA's NVHPC 25.9 image plus the same packages and
  CMake. On a host with docker and the NVIDIA container toolkit (the Deep Learning VM image has
  the toolkit; `sudo apt-get install -y docker.io && sudo nvidia-ctk runtime configure
  --runtime=docker && sudo systemctl restart docker` finishes the setup), from the top of a clone:

  ```bash
  docker build -t qe-nvhpc:25.9 gpu-quickstart/cloud
  docker run --rm -it --gpus all --ipc=host --pid=host -v "$PWD":/work -w /work/gpu-quickstart qe-nvhpc:25.9
  ```

  Inside, every command below works unchanged (`--ipc=host` for `/dev/shm`, `--pid=host` so the
  run script can pin cores). A `ph.x` built in the container gave the bare-metal results and times.

`scripts/env_nvhpc.sh` is the environment of every GPU build and run (`NVHPC_ROOT` can point at
another installation).

## Build

```bash
scripts/build.sh .. build/gpu gpu            # pw.x and ph.x of this checkout: build/gpu/bin
scripts/build.sh .. build/cpu cpu-gnu        # the same without CUDA (gfortran), if you want it
```

`gpu` is nvfortran with OpenACC and CUDA Fortran for `sm_90`, MPI and OpenMP, NVHPC's BLAS and
LAPACK: the flags of every published number. The first configure of a tree clones QE's
`external/` submodules (one tree at a time); a build takes a few minutes with all cores
(`BUILD_JOBS` to limit it). Any source tree can be given instead of `..` (a git worktree of a
branch, for instance) and any build directory.

## Run

```bash
scripts/run_gpu.sh build/gpu pw pw.in -o pw.out        # one rank on GPU 0, OpenMP threads on its cores
scripts/run_gpu.sh build/gpu ph ph.in -o ph.out
GPUS=0,1 scripts/run_gpu.sh build/gpu ph ph.in -o ph.out -nk 2      # two GPUs, one k-point pool each
QE_PH_DEV=0 scripts/run_gpu.sh build/gpu ph ph.in -o ph-stock.out   # the stock ph.x path of the same binary
```

`run_gpu.sh` sets the environment, starts one MPI rank per GPU named in `GPUS` under HPC-X
`mpirun`, gives each rank the OpenMP threads of its GPU's share of the cores (the machine's vCPUs
divided by its GPUs, SMT siblings not counted, minus one: 12 on an a3-highgpu), pins the ranks
there (`PIN=0` not to), and passes every `QE_*` variable of your environment to the program.
`OMP` overrides the thread count. The `ieee_*` warnings nvfortran prints at the end of a run are
harmless.

## The example: Si, ultrasoft, phonons

`example/si`: a two-atom silicon cell with a pslibrary ultrasoft pseudopotential (fetched from the
QE site and checksummed), 4x4x4 k-points; the ground state, then the dielectric constant and the
phonons at Γ, then the phonons at a low-symmetry q.

```bash
cd example/si && ./fetch_pseudo.sh
../../scripts/run_gpu.sh ../../build/gpu pw pw.in -o pw.out
../../scripts/run_gpu.sh ../../build/gpu ph ph-gamma.in -o ph-gamma.out
../../scripts/run_gpu.sh ../../build/gpu ph ph-q.in -o ph-q.out
grep -E '^!' pw.out; grep -A3 'Dielectric constant in cartesian' ph-gamma.out; grep 'freq (' ph-gamma.out ph-q.out
grep -E 'PWSCF.*WALL|PHONON.*WALL' *.out
```

What it gives on one H100 (measured 2026-10-16 with `run_gpu.sh`, stock `develop` 6e773a63
against the head of the `ph.x` series, #10):

| run | result | stock | with the series |
|---|---|---:|---:|
| `pw.x`, ground state | E = −22.66315051 Ry | 1.4 s | 1.4 s |
| `ph.x` at Γ, `epsil` | ε∞ = 24.8327; optical modes 519.30 / 519.31 cm⁻¹ | 10.3 s | 1.1 s |
| `ph.x` at q = (0.3, 0.2, 0.1) | 106.7, 122.8, 192.7, 484.3, 490.7, 496.4 cm⁻¹ | 59.6 s | 1.9 s |

The times are the `PWSCF` / `PHONON ... WALL` clocks. Results agree to the printed digits
(q-point modes to 0.1 cm⁻¹, ε∞ to 1e-8); the three acoustic modes at Γ are not forced to zero
(no acoustic sum rule here; `dynmat.x` applies one) and sit near −20 cm⁻¹ in both. Differences
between the two codes at these thresholds (`tr2_ph = 1e-14`) are the convergence error of the
calculation, explained in #7; `QE_PH_SP=0 QE_PH_NZ=0` makes the series follow the arithmetic path
of stock, `QE_PH_DEV=0` runs the stock routines.

## Switches

Every piece of the GPU work has a run-time switch, read from the environment, and falls back to
the stock routine: `QE_PH_DEV=0` (the whole DFPT device engine), `QE_BETA_CACHE=0`, `QE_USAUG=0`,
`QE_PH_DYN=0`, `QE_PH_NSCF_DEV=0`; inside the engine `QE_PH_SP=0` (double precision only),
`QE_PH_NZ=0`, `QE_PH_LOWG=0`, `QE_PH_REAL=0`; self-checks `QE_PH_CHECK_RES=1`, `QE_USAUG_CHECK=1`,
`QE_PH_DYNCHECK=1`, `QE_PAW_DPOT_CHECK=1`. The pull requests describe each; what they cover and
what still runs the stock path (noncollinear and spin-orbit, DFT+U, FFT grids distributed over
MPI ranks, …) is in their *Limits* sections.
