<img src="logo-discovered-materials.png" alt="Discovered Materials" width="480">

This is Discovered Materials' fork of Quantum ESPRESSO with CUDA-specific accelerations to the
hybrid functionals (HSE06 and PBE0) and to `ph.x`. [`gpu-quickstart/`](gpu-quickstart/README.md)
is how to build and run it on NVIDIA GPUs, with a first calculation to compare against. This fork
is under active development, so expect the code and these numbers to change as the work continues.

## Speedups

Here's a summary of the speedups we've shown for `ph.x`:

| case | what is computed | stock CPU | stock GPU | optimized | vs stock GPU | vs stock CPU |
|---|---|---:|---:|---:|---:|---:|
| `aln_paw_eps` | AlN (4 atoms, PAW), ε∞ and Z\* at Γ (the κ-pipeline call) | 39.0 | 67.7 | 4.1 | 16.5x | 9.5x |
| `aln_paw_gamma` | AlN PAW, ε∞, Z\* and phonons at Γ | 142.2 | 204.8 | 6.3 | 33x | 23x |
| `aln_paw_q` | AlN PAW, low-symmetry q | 719.8 | 1842.9 | 25.4 | 73x | 28x |
| `c_paw_q` | diamond PAW, 11×11×11 k, low-symmetry q | 113.9 | 1270.7 | 11.8 | 108x | 9.7x |
| `si_us_disp` | Si ultrasoft, 4×4×4 q dispersion | 498.8 | 794.6 | 25.6 | 31x | 19.5x |
| `si_nc_disp` | Si norm-conserving, 4×4×4 q dispersion | 64.0 | 531.0 | 7.5 | 71x | 8.5x |
| `mgo_nc_disp` | MgO, ε∞, Z\*, 4×4×4 q dispersion | 209.2 | 877.9 | 13.9 | 63x | 15.1x |
| `mos2_nc_2d_m` | MoS2 monolayer, q = M | 352.3 | 234.3 | 13.6 | 17.2x | 26x |
| `al_paw_metal_x` | Al metal PAW, 16×16×16 k, q = X | 6.5 | 58.5 | 2.4 | 24x | 2.7x |
| `al_paw_metal_q` | Al metal PAW, 4096 k-points | 46.4 | 1060.2 | 14.9 | 71x | 3.1x |
| `si54_us_gamma` | 54-atom Si, ultrasoft, 6 of 162 modes | not run | 4380 | 57.8 | 76x | – |
| `nv64_nc_lsda_gamma` | NV⁻ in 63-atom diamond, LSDA, 4 modes | 178.9 | 77.2 | 6.1 | 12.7x | 29x |
| `si216_nc_gamma` | 216-atom Si, 2 modes | 1806 | 384.0 | 32.2 | 11.9x | 56x |
| **geometric mean speedup** | 13 cases (12 with a CPU reference) | | | | **36x** | **14x** |

Speedups observed across benchmarks for phonon (DFPT) calculations.

Here's a summary of the speedups on hybrid functional calculations:

| case | stock | optimized, same path | optimized, default | stock, 4 GPUs | optimized, 4 GPUs |
|---|---:|---:|---:|---:|---:|
| 01 Si HSE | 54.8 | 3.3 (16.5x) | 2.8 (19x) | 21.3 | 1.1 |
| 02 GaAs HSE | 539.2 | 26.3 (20.5x) | 19.6 (28x) | 186.5 | 5.6 |
| 03 MgO PBE0 vc-relax | 450.4 | 18.9 (24x) | 19.2 (23x) | cannot run | 6.1 |
| 04 NiO AFM HSE | 4015 | 124.7 (32x) | 60.8 (66x) | 1237 | 17.1 |
| 05 CsPbI3 HSE+SOC | 601.2 | 61.2 (9.8x) | 31.6 (19x) | 298.1 | 12.7 |
| 06 MoS2 HSE | 425.1 | 26.8 (16x) | 15.2 (28x) | 168.7 | 5.3 |
| 07 Si-63 vacancy relax | 720.8 | 183.8 (3.9x) | 98.0 (7.4x) | 311.8 | 42.6 |
| 08 Si-216 PBE0 | 281.4 | 97.4 (2.9x) | 45.7 (6.2x) | cannot run | 26.5 |
| 09 64 H2O PBE0 MD | 1103.8 | 316.8 (3.5x) | 201.8 (5.5x) | cannot run | 118.7 |
| 10 C60 B3LYP | 93.5 | 37.5 (2.5x) | 24.4 (3.8x) | cannot run | 22.7 |

Speedups observed across benchmarks for hybrid functional calculations.

## Getting started

[`gpu-quickstart/`](gpu-quickstart/README.md) covers the machine, the toolchain (NVIDIA HPC SDK
25.9), the build and a first calculation with reference results. In short, on an Ubuntu 22.04
machine with an NVIDIA GPU and driver:

```bash
cd gpu-quickstart
sudo cloud/setup_node.sh                            # the toolchain (or use cloud/Dockerfile)
scripts/build.sh .. build/gpu gpu                   # pw.x and ph.x in build/gpu/bin
scripts/run_gpu.sh build/gpu pw pw.in -o pw.out     # one MPI rank per GPU
scripts/run_gpu.sh build/gpu ph ph.in -o ph.out
```

## Switching the accelerations off

The GPU code paths can be switched off from the environment, so results can be compared with stock
Quantum ESPRESSO using the same binary: `QE_PH_DEV=0` runs the stock `ph.x` routines and
`QE_EXX_FAST=0` the original exact-exchange operator of `pw.x`. The finer `ph.x` switches are
listed in [`gpu-quickstart/README.md`](gpu-quickstart/README.md#switches).

## What runs on the GPU

- `ph.x`: collinear calculations with norm-conserving, ultrasoft or PAW pseudopotentials,
  insulators and metals, at any q. Noncollinear magnetism and spin-orbit, DFT+U and FFT grids
  distributed over MPI ranks run the stock routines. Use one GPU per MPI process.
- Hybrid functionals in `pw.x`: the batched exchange operator serves collinear k-point runs on one
  band group with an undistributed FFT grid; other runs use the existing GPU routines of Quantum
  ESPRESSO.

## Contributing

Bug reports, questions and pull requests about the GPU work are welcome here: open an issue or a
pull request on this repository. Changes to the rest of Quantum ESPRESSO belong upstream, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Like Quantum ESPRESSO, this fork is distributed under the GNU General Public License, version 2 or
(at your option) any later version. See [`License`](License).

## Citing

If you use this code in published work, please cite Quantum ESPRESSO as its developers ask
(<https://www.quantum-espresso.org/quote>) and link to this repository.

## Acknowledgements

This work builds on [Quantum ESPRESSO](https://www.quantum-espresso.org), developed by the
Quantum ESPRESSO Foundation and its many contributors. Upstream Quantum ESPRESSO's README is in
[`QE_README.md`](QE_README.md).
