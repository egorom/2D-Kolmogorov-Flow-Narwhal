# Narwhal: Viscoelastic Kolmogorov Flow Solver

**Narwhal** is a Python project that implements a 2D Kolmogorov flow solver for viscoelastic fluids using the Stokes–Oldroyd-B model.  
It is built on top of the **Dedalus** spectral PDE framework.

This repository contains the simulation code, analysis scripts, and example configurations needed to reproduce key flow regimes explored in the viscoelastic Kolmogorov flow literature.

---

## Overview

In periodic parallel shear flows, viscoelastic fluids governed by the Stokes–Oldroyd-B equations can undergo rich dynamics, including oscillatory solutions and transitions to elastic turbulence. The **Narwhal** project aims to:

- Implement a flexible spectral solver for 2D Kolmogorov flows.
- Use Dedalus to discretize and integrate the governing equations.
- Provide configuration files and scripts for parameter studies and benchmarking.

This work builds on theoretical and numerical studies of viscoelastic flow instabilities and elastic turbulence in Kolmogorov-type forcing  
([Phys. Rev. Fluids 10, L041301 (2025)](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.10.L041301)).


---

## Background

The underlying physics of the problem involves:

- A 2D periodic shear (Kolmogorov) forcing profile.
- A viscoelastic fluid model described by the Stokes–Oldroyd-B constitutive relation.
- Spectrally accurate computation of derivatives and flows via Dedalus.

For detailed context on the viscoelastic Kolmogorov flow dynamics, see the relevant literature:  
*Jeffrey Nichols, Robert D. Guy, and Becca Thomases, “Period-doubling route to chaos in viscoelastic Kolmogorov flow,” Phys. Rev. Fluids 10, L041301 (2025).*:contentReference[oaicite:4]{index=4}

---

## Dependencies

Before using this code, install the following:

- Python 3.9+  
- [Dedalus](https://dedalus-project.readthedocs.io/en/latest/), a flexible Python framework for solving differential equations using spectral methods
- MPI for parallel execution (`mpi4py`)
- FFTW, HDF5 with Python bindings (`h5py`)
- NumPy, SciPy, Matplotlib

You can install Dedalus and its dependencies using Conda (recommended):

```bash
conda install -c conda-forge dedalus mpi4py h5py numpy scipy matplotlib
