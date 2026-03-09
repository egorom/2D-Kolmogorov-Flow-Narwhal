## Narwhal — Copilot instructions (concise)

This file gives focused, actionable notes for AI coding agents working in this repository so you can be productive quickly.

- Project purpose: 2D Kolmogorov flow solver for viscoelastic fluids (Stokes–Oldroyd-B) built on the Dedalus spectral PDE framework. Main solver lives in `solvers/`.

- Quick run (serial):
  - Ensure Python 3.9+ and dependencies (Dedalus, mpi4py, h5py, numpy, scipy, matplotlib) are installed (Conda recommended).
  - Example serial run: `python solvers/SOB_narwhal.py`
  - Example MPI run: `mpiexec -n 4 python solvers/SOB_narwhal.py` (scripts use `MPI.COMM_WORLD`, and plotting is gated to rank 0).

- Key files and directories to read first:
  - `solvers/SOB_narwhal.py` — primary demo: parameter settings, Dedalus LBVP/IVP usage, snapshot saving, plotting (Narwhal plot). Many patterns and idioms used across the repo are here.
  - `solvers/*.py` — other solver variants and plotting helpers (e.g. `plot_kolmogorov.py`, `PlotFinal.py`).
  - `validation/` — small sanity-check scripts (`navier_stokes_test.py`, `steady_stokes.py`) useful for unit-like runs and expected behaviors.
  - `snapshots/`, `Wi_*/` — produced data (HDF5 and NPZ) and example outputs; useful to inspect expected snapshot layout and filenames.

- Project-specific patterns and conventions (important to follow):
  - Dedalus objects: fields use backends `.g` for grid data and `change_scales()` is called before reading or saving fields. Use `evaluate()` after field algebra when you need concrete arrays.
    - Example: `cxx.change_scales(1); txx = (alpha_p * (cxx - 1.0)).evaluate(); txx.change_scales(1)`
  - Solver construction: LBVP (Stokes) uses `d3.LBVP([...], namespace=locals())` and `build_solver()` → `solve()`. Time-dependent conformation tensor uses `d3.IVP([...])` and `build_solver(d3.RK443)` + `csolver.step(dt)`.
  - Snapshots: `csolver.evaluator.add_file_handler("snapshots", sim_dt=0.5, max_writes=None)` — these produce HDF5 outputs (see `snapshots/`). Plotting also writes PNGs in top-level when run on rank 0.
  - MPI-awareness: code uses `MPI.COMM_WORLD`; plotting and prints are gated to rank 0. Local grid shapes may differ in parallel runs — be careful when gathering or assuming full arrays.
  - Parameterization: many nondimensional parameters are set at the top of `SOB_narwhal.py` (e.g., `lam`, `n_kolm`, `nu`, `maxvel`, `Wi`) — tests and experiments modify these directly in the script.

- Developer workflows (discovered from repository):
  - Typical experiment: edit parameters in `solvers/SOB_narwhal.py`, run serial or MPI, inspect `snapshots/` HDF5 and generated `narwhal_t*.png` files. The plotting function is serial-only (rank 0), so use small MPI ranks for plotting.
  - Reproducing examples: `Wi_*` folders contain NPZ snapshot sequences that match parameters in the top of the solver; use them as reference data for plotting or regression checks.
  - Quick checks: open `validation/` scripts to run small, fast checks (these are not formal tests but useful smoke checks).

- Common gotchas to watch for in edits:
  - Forgetting `change_scales(1)` before accessing `.g` leads to unexpected shapes/aliases in Dedalus. Always follow the pattern in `SOB_narwhal.py`.
  - In parallel, fields are distributed — any code assuming full Nx×Ny arrays must gather or be rank-aware.
  - Plot code uses `matplotlib.use('Agg')` and writes PNGs; avoid interactive plotting expectations.
  - The code prints a SPD check: `tr_max = np.max(cxx['g'] + cyy['g'])` and signals `⚠ SPD` if non-positive — monitor this for stability concerns.

- Where to add tests/edits safely:
  - Small refactors around helper functions (plotting, snapshot naming) are low risk. Avoid changing solver numerics without numerical validation against `Wi_*` snapshots.

- Useful searches when coding: `change_scales`, `.g]`, `evaluator.add_file_handler`, `d3.LBVP`, `d3.IVP`, `build_solver`, `evaluate()` — these locate the important Dedalus idioms across the repo.

If any part of this is unclear or you'd like more detail (for example, a short snippet showing how to gather fields in MPI, or a checklist for adding a new experiment), tell me where to expand and I will iterate.
