# GPU tests: the gate on `develop`

`.github/workflows/gpu-tests.yml` runs on every pull request into `develop` (and on pushes to
it, weekly, and by hand):

| job | where | what | time |
|---|---|---|---|
| `cpu-build` | GitHub-hosted | build without CUDA (gfortran, Open MPI, FFTW, OpenBLAS), test directories `pw_scf` and `ph_base` | ~30 min |
| `start-runner` | GitHub-hosted | starts the GPU runner VM in our Google Cloud project | 1 min |
| `gpu-tests` | self-hosted GPU runner | GPU build with the published flags, then the test-suite categories `pw_all` and `ph_all`, one per GPU at the same time, compared with `known-failures.txt` | ~1 h (two A100s), ~2 h (one) |

A test that fails and is not in `known-failures.txt` fails `gpu-tests`; so does a category whose
run ends without testcode's summary. A listed test that passes is reported in the job summary so
the list can be trimmed. The list is the state of stock `develop` on the runner (its GPU, the
toolchain), not an allowance for new failures: a pull request that fixes one removes it.

Branch protection on `develop` should require the checks `CPU-only build and quick tests` and
`GPU build and test-suite` (Settings → Branches → `develop` → require status checks).

## The runner

A self-hosted runner with the labels `gpu` and `dm-qe`, on a machine with an NVIDIA GPU, the
toolchain of `gpu-quickstart/cloud/setup_node.sh` and the runner service of
`.github/ci/runner-setup.sh`. Ours is a Google Compute Engine `a2-highgpu-2g` (24 vCPUs, two A100
40 GB, `dmqe-ci-a100` in `us-central1-c`), off between runs: the workflow starts it, the runner's
idle watchdog powers it off 20 minutes after the last job (a stopped VM bills only its disk). The
weekly schedule keeps the runner registered (GitHub removes a runner that has not connected for
14 days). The GPU has to be a data-centre part with real double-precision throughput: on an L4
one noncollinear SCF of `ph_all` alone took 30 minutes (seconds on an A100 or H100), which puts
the suite far outside the hour. Two GPUs, one per category: on a single A100 the two
categories slow each other down and the job takes over 1.5 h.

To make one, on an Ubuntu 22.04 machine with the NVIDIA driver (the Deep Learning VM image of
`gpu-quickstart/cloud/create_vm.sh` with `MACHINE_TYPE=a2-highgpu-2g`, or any other):

```bash
sudo gpu-quickstart/cloud/setup_node.sh                 # NVHPC 25.9, the gfortran set, CMake
sudo .github/ci/runner-setup.sh <registration-token>    # Settings -> Actions -> Runners -> New self-hosted runner
```

The workflow finds the machine through three repository settings: the secret `GCP_SA_KEY` (the
JSON key of a service account allowed to start the instance) and the variables `GCP_PROJECT`, `GPU_RUNNER_VM` (default `dmqe-ci-a100`), `GPU_RUNNER_ZONE` (default
`us-central1-c`), `GPU_RUNNER_MACHINE_TYPE` (default `a2-highgpu-2g`) and `GPU_RUNNER_FALLBACK_TYPE`
(default `a2-highgpu-1g`). Without the secret the start job does nothing and `gpu-tests` waits for
whatever runner is online. GPU capacity in a zone comes and goes, so the start job retries for about
25 minutes and, when no two-GPU machine is free, resizes the stopped VM to the one-GPU type and runs
there (the job then takes about two hours instead of one; the next run tries the two-GPU type again).
When neither is available the job fails with "no GPU capacity": re-run it later.

The GPU build is for the runner's GPU (`run_tests.sh` reads the compute capability: `sm_80` on an
A100, `sm_90` on an H100, `sm_89` on an L4); results are compared with the test-suite's references with its
tolerances, so the architecture does not matter to pass or fail, only to the time.

## By hand

The same script runs anywhere the toolchain is:

```bash
bash .github/ci/run_tests.sh                                   # GPU build, pw_all and ph_all
CI_CATEGORIES="ph_all" bash .github/ci/run_tests.sh            # one category
CI_FLAVOR=cpu CI_CATEGORIES="pw_scf ph_base" bash .github/ci/run_tests.sh
QE_PH_DEV=0 CI_SKIP_BUILD=1 bash .github/ci/run_tests.sh      # the stock path of the same GPU binary
```

Logs land in `ci-logs/` (one per category, the build log, `summary.md`); the job uploads them as
an artifact. `compare_failures.py` is the judge.
