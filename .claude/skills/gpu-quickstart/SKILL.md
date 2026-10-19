---
name: gpu-quickstart
description: Get a GPU machine (or container) with this repository's published toolchain, build pw.x and ph.x with the published flags, and run a calculation on H100s the way the GPU pull requests were measured — including a Si phonon example with reference results and times. Use when someone wants to run this code on a GPU, set up a cloud VM for it, or check a GPU build.
---

# GPU quickstart

Everything is in `gpu-quickstart/`; its `README.md` has the machine, the toolchain, the build,
the run helper, the example with reference numbers, and the run-time switches. Work from
`gpu-quickstart/`.

```bash
VM=qe-h100 ZONE=us-central1-a MACHINE_TYPE=a3-highgpu-1g cloud/create_vm.sh   # a GCE VM in the user's project (flex-start)
sudo cloud/setup_node.sh                      # toolchain on an Ubuntu 22.04 + NVIDIA 580 machine (idempotent, ~20 min)
# or: docker build -t qe-nvhpc:25.9 cloud && docker run --rm -it --gpus all --ipc=host --pid=host -v "$PWD/..":/work -w /work/gpu-quickstart qe-nvhpc:25.9
scripts/build.sh .. build/gpu gpu             # pw.x, ph.x with the published flags (a few minutes)
scripts/run_gpu.sh build/gpu pw <input> -o <out>     # one rank per GPU in GPUS (default 0), threads on its cores, QE_* forwarded
cd example/si && ./fetch_pseudo.sh && ../../scripts/run_gpu.sh ../../build/gpu pw pw.in -o pw.out && ../../scripts/run_gpu.sh ../../build/gpu ph ph-q.in -o ph-q.out
grep 'PHONON.*WALL' ph-q.out                  # README: 1.9 s on one H100 with the series, 59.6 s stock; frequencies listed there
```

Things to know: configure one source tree at a time (QE's CMake clones `external/` on the first
configure); `QE_PH_DEV=0` runs the stock path of the same binary; a flex-start VM bills while it
runs and stops by itself at its run limit; the full benchmark set and validation drivers behind
the pull requests' tables are not in this repository.
