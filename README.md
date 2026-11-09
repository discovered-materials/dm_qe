<img src="logo-discovered-materials.png" alt="Discovered Materials" width="480">

This is Discovered Materials' fork of Quantum ESPRESSO with CUDA-specific accelerations to the
hybrid functionals (HSE06 and PBE0) and to `ph.x`. [`gpu-quickstart/`](gpu-quickstart/README.md)
is how to build and run it on NVIDIA GPUs, with a first calculation to compare against. This fork
is under active development, so expect the code and these numbers to change as the work continues.

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

Speedups observed across benchmarks for phonon (DFPT) calculations. The silicon example in
`gpu-quickstart/` is a small `ph.x` case you can reproduce yourself, with reference results and
times.

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

Everything below is upstream Quantum ESPRESSO's README.

---

![q-e-logo](logo.jpg)

This is the distribution of the Quantum ESPRESSO suite of codes (ESPRESSO:
opEn-Source Package for Research in Electronic Structure, Simulation, and
Optimization)

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

## USAGE
Quick installation instructions for CPU-based machines. For GPU execution, see
file [README_GPU.md](README_GPU.md). Go to the directory where this file is. 

Using "make"
(`[]` means "optional"):
```
./configure [options]
make all
```
"make" alone prints a list of acceptable targets. Optionally,
`make -jN` runs parallel compilation on `N` processors.
Link to binaries are found in bin/.

Using "CMake" (v.3.14 or later):

```
mkdir ./build
cd ./build
cmake -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc [-DCMAKE_INSTALL_PREFIX=/path/to/install] ..
make [-jN]
[make install]
```
Although CMake has the capability to guess compilers, it is strongly recommended to specify
the intended compilers or MPI compiler wrappers as `CMAKE_Fortran_COMPILER` and `CMAKE_C_COMPILER`.
"make" builds all targets. Link to binaries are found in build/bin.
If `make install` is invoked, directory `CMAKE_INSTALL_PREFIX`
is prepended onto all install directories.

For more information, see the general documentation in directory Doc/, 
package-specific documentation in \*/Doc/, and the web site 
http://www.quantum-espresso.org/. Technical documentation for users and
developers 
can be found on [Wiki page on gitlab](https://gitlab.com/QEF/q-e/-/wikis/home).

## PACKAGES

- PWscf: structural optimisation and molecular dynamics on the electronic ground state, with self-consistent solution of DFT equations;
- CP: Car-Parrinello molecular dynamics;
- PHonon: vibrational and dielectric properties from DFPT (Density-Functional Perturbation Theory);
- TD-DFPT: spectra from Time-dependent DFPT;
- HP: calculation of Hubbard parameters from DFPT;
- EPW: calculation of electron-phonon coefficients, carrier transport, phonon-limited superconductivity and phonon-assisted optical processes;
- PWCOND: ballistic transport;
- XSpectra: calculation of X-ray absorption spectra;
- PWneb: reaction pathways and transition states with the Nudged Elastic Band method;
- GWL: many-body perturbation theory in the GW approach using ultra-localised Wannier functions and Lanczos chains;
- QEHeat: energy current in insulators for thermal transport calculations in DFT.
- KCW: Koopmans-compliant functionals in a Wannier representation

## Modular libraries
The following libraries have been isolated and partially encapsulated in view of their release for usage in other codes as well:

- UtilXlib: performing basic MPI handling, error handling, timing handling.
- FFTXlib: parallel (MPI and OpenMP) distributed three-dimensional FFTs, performing also load-balanced distribution of data (plane waves, G-vectors and real-space grids) across processors.
- LAXlib: parallel distributed dense-matrix diagonalization, using ELPA, SCALapack, or a custom algorithm.
- KS Solvers: parallel iterative diagonalization for the Kohn-Sham Hamiltonian (represented as an operator),using block Davidson and band-by-band or block Conjugate-Gradient algorithms.
- LRlib: performs a variety of tasks connected with (time-dependent) DFPT, to be used also in connection with Many-Body Perturbation Theory.
- upflib: pseudopotential-related code.
- devXlib: low-level utilities for GPU execution

## Contributing
Quantum ESPRESSO is an open project: contributions are welcome.
Read the [Contribution Guidelines](CONTRIBUTING.md) to see how you
can contribute.

## LICENSE

All the material included in this distribution is free software;
you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

These programs are distributed in the hope that they will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
675 Mass Ave, Cambridge, MA 02139, USA.
