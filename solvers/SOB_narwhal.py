import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import time
import sys
import os
import subprocess
import shutil

# -------------------------
# Run options
# -------------------------
PARALLEL         = True    # Set True to relaunch under mpiexec -n N_PROCS if not already parallel
N_PROCS          = 4       # Number of MPI ranks to use when PARALLEL=True
TIME_SCALING     = False   # Set True to run timing checkpoints instead of full simulation
TIME_CHECKPOINTS = [10.0, 20.0]

# -------------------------
# Self-relaunch under mpiexec if PARALLEL=True and we're only on 1 rank
# -------------------------
if PARALLEL and MPI.COMM_WORLD.Get_size() == 1:
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if mpiexec is None:
        py_bin = os.path.dirname(sys.executable)
        for candidate in ["mpiexec", "mpirun"]:
            full = os.path.join(py_bin, candidate)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                mpiexec = full
                break
    if mpiexec is None:
        print("ERROR: could not find mpiexec or mpirun. "
              "Try: conda install -c conda-forge openmpi")
        sys.exit(1)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    cmd = [mpiexec, "-n", str(N_PROCS), sys.executable] + sys.argv
    print(f"Relaunching with: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)

# If we get here, we're either already running under mpiexec, or PARALLEL=False
comm = MPI.COMM_WORLD
if comm.rank == 0:
    print(f"Running on {comm.size} rank(s)  |  "
          f"{'PARALLEL' if comm.size > 1 else 'SERIAL'}  |  "
          f"{'TIMING MODE' if TIME_SCALING else 'SIMULATION MODE'}")

# -------------------------
# Parameters
# -------------------------
aspect  = 4
Lx      = 2*np.pi
Ly      = Lx / aspect
Nx      = 128
Ny      = Nx // aspect
dealias = 3/2

n_kolm  = 4
lam     = 1.695
xi      = 0.5
maxvel  = 4.0

kappa_c = 1e-3
tauR    = lam
alpha_p = xi / lam

amp     = maxvel * n_kolm**2 * (1 + xi / (1 + lam*n_kolm**2))

dt_step = 0.005
t_end   = 80.0
plot_interval = 2.0

Wi = lam * maxvel * n_kolm

# -------------------------
# Analytical laminar solution
# -------------------------
def analytical_solution(y_arr):
    ux_sol   = maxvel * np.sin(n_kolm * y_arr)
    denom    = 1/lam**2 + 5*n_kolm**2/lam + 4**2*n_kolm**4
    C11amp   = 2*maxvel**2*n_kolm**2 / denom
    C11const = 2*lam*C11amp*n_kolm**2 + 1.0
    C11_sol  = C11amp * np.cos(n_kolm * y_arr)**2 + C11const
    C12_sol  = (maxvel*n_kolm*lam) / (1 + lam*n_kolm**2) * np.cos(n_kolm * y_arr)
    C22_sol  = np.ones_like(y_arr)
    return ux_sol, C11_sol, C12_sol, C22_sol

# -------------------------
# Domain
# -------------------------
coords = d3.CartesianCoordinates('x', 'y')
dist   = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

dx_op = lambda F: d3.Differentiate(F, coords['x'])
dy_op = lambda F: d3.Differentiate(F, coords['y'])

# -------------------------
# Stokes unknowns
# -------------------------
u   = dist.VectorField(coords, name='u',   bases=(xb, yb))
p   = dist.Field(name='p',   bases=(xb, yb))
ux0 = dist.Field(name='ux0')
uy0 = dist.Field(name='uy0')
p0  = dist.Field(name='p0')

ux   = u @ ex;  uy   = u @ ey
ux_x = dx_op(ux);  ux_y = dy_op(ux)
uy_x = dx_op(uy);  uy_y = dy_op(uy)

# -------------------------
# Forcing
# -------------------------
f0 = dist.VectorField(coords, name='f0', bases=(xb, yb))
f0['g'][0] = amp * np.sin(n_kolm * y)
f0['g'][1] = 0.0

f_total = dist.VectorField(coords, name='f_total', bases=(xb, yb))
f_total['g'][0] = f0['g'][0]
f_total['g'][1] = f0['g'][1]

# -------------------------
# Conformation tensor IC
# -------------------------
cxx = dist.Field(name='cxx', bases=(xb, yb))
cxy = dist.Field(name='cxy', bases=(xb, yb))
cyy = dist.Field(name='cyy', bases=(xb, yb))

_, C11_init, C12_init, _ = analytical_solution(y)

rng      = np.random.default_rng(seed=42)
C11max   = np.max(C11_init)
pert_amp = 1e-3 * C11max

# Use local grid shapes so this works correctly under MPI decomposition
local_Nx, local_Ny = x.shape[0], y.shape[1]
X = x * np.ones((1, local_Ny))
Y = y * np.ones((local_Nx, 1))
pert = np.zeros((local_Nx, local_Ny))
for kx_p in range(1, 5):
    for ky_p in range(1, 5):
        phase = rng.uniform(0, 2*np.pi)
        pert += np.sin(kx_p * X + phase) * np.cos(ky_p * Y)
pert = pert_amp * pert / np.max(np.abs(pert))

cxx['g'] = C11_init + pert
cxy['g'] = C12_init
cyy['g'] = 1.0

# -------------------------
# Stokes LBVP
# -------------------------
stokes = d3.LBVP([u, p, ux0, uy0, p0], namespace=locals())
stokes.add_equation("d3.div(d3.grad(u)) - d3.grad(p) + ux0*ex + uy0*ey = -f_total")
stokes.add_equation("d3.div(u) + p0 = 0")
stokes.add_equation("d3.integ(p) = 0")
stokes.add_equation("d3.integ(u@ex) = 0")
stokes.add_equation("d3.integ(u@ey) = 0")
stokes_solver = stokes.build_solver()
stokes_solver.solve()

# -------------------------
# Conformation IVP
# -------------------------
cprob = d3.IVP([cxx, cxy, cyy], namespace=locals())

cprob.add_equation(
    "dt(cxx) - kappa_c*d3.div(d3.grad(cxx)) = -(u@d3.grad(cxx))"
    " + 2*(ux_x*cxx + ux_y*cxy) - (1/tauR)*(cxx - 1)"
)
cprob.add_equation(
    "dt(cxy) - kappa_c*d3.div(d3.grad(cxy)) = -(u@d3.grad(cxy))"
    " + (ux_x*cxy + ux_y*cyy + cxx*uy_x + cxy*uy_y) - (1/tauR)*cxy"
)
cprob.add_equation(
    "dt(cyy) - kappa_c*d3.div(d3.grad(cyy)) = -(u@d3.grad(cyy))"
    " + 2*(uy_x*cxy + uy_y*cyy) - (1/tauR)*(cyy - 1)"
)

csolver = cprob.build_solver(d3.RK443)
csolver.stop_sim_time = t_end

# -------------------------
# Snapshot handler (suppressed in timing mode)
# -------------------------
if not TIME_SCALING:
    snap = csolver.evaluator.add_file_handler("snapshots", sim_dt=0.5, max_writes=None)
    snap.add_task(u @ ex, name="ux")
    snap.add_task(u @ ey, name="uy")
    snap.add_task(cxx,    name="cxx")
    snap.add_task(cxy,    name="cxy")
    snap.add_task(cyy,    name="cyy")

# -------------------------
# Polymer stress update
# -------------------------
def update_forcing_from_C():
    f_total.change_scales(1);  f0.change_scales(1)
    cxx.change_scales(1);  cxy.change_scales(1);  cyy.change_scales(1)

    txx = (alpha_p * (cxx - 1.0)).evaluate();  txx.change_scales(1)
    txy = (alpha_p *  cxy        ).evaluate();  txy.change_scales(1)
    tyy = (alpha_p * (cyy - 1.0)).evaluate();  tyy.change_scales(1)

    divtau_x = (dx_op(txx) + dy_op(txy)).evaluate();  divtau_x.change_scales(1)
    divtau_y = (dx_op(txy) + dy_op(tyy)).evaluate();  divtau_y.change_scales(1)

    f_total['g'][0] = f0['g'][0] + divtau_x['g']
    f_total['g'][1] = f0['g'][1] + divtau_y['g']

# -------------------------
# 2D snapshot plot — MPI-safe gather before plotting
# -------------------------
def save_2d_plot(t_now):
    # All ranks must participate in the gather — this is a collective call
    u.change_scales(1)
    cxx.change_scales(1);  cxy.change_scales(1);  cyy.change_scales(1)

    ux_gathered  = comm.gather(u['g'][0],  root=0)
    uy_gathered  = comm.gather(u['g'][1],  root=0)
    cxx_gathered = comm.gather(cxx['g'],   root=0)
    cxy_gathered = comm.gather(cxy['g'],   root=0)
    cyy_gathered = comm.gather(cyy['g'],   root=0)

    # Only rank 0 does the plotting
    if comm.rank != 0:
        return

    # Reassemble full fields along y-axis
    ux_g  = np.concatenate(ux_gathered,  axis=1)
    uy_g  = np.concatenate(uy_gathered,  axis=1)
    cxx_g = np.concatenate(cxx_gathered, axis=1)
    cxy_g = np.concatenate(cxy_gathered, axis=1)
    cyy_g = np.concatenate(cyy_gathered, axis=1)
    trC_g = cxx_g + cyy_g

    # Full global axes for plotting
    x_plot = np.linspace(0, Lx, Nx, endpoint=False)
    y_plot = np.linspace(0, Ly, Ny, endpoint=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"Kolmogorov flow  —  t = {t_now:.2f}   "
        f"λ={lam}, Wi={Wi:.1f}, n={n_kolm}",
        fontsize=14
    )

    def xanom(f):
        return f - np.mean(f, axis=0, keepdims=True)

    def pcolor(ax, data, title, cmap='RdBu_r', diverging=True):
        if diverging:
            vmax = np.max(np.abs(data))
            vmin = -vmax
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmax > 0 else None
        else:
            vmin, vmax = data.min(), data.max()
            norm = None
        im = ax.pcolormesh(x_plot, y_plot, data.T, cmap=cmap, norm=norm,
                           vmin=None if diverging else vmin,
                           vmax=None if diverging else vmax,
                           shading='auto')
        ax.set_xlabel('x');  ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pcolor(axes[0, 0], xanom(ux_g),  "$u_x'$",   cmap='RdBu_r', diverging=True)
    pcolor(axes[0, 1], uy_g,         '$u_y$',     cmap='RdBu_r', diverging=True)
    pcolor(axes[0, 2], xanom(trC_g), "Tr$(C)'$",  cmap='RdBu_r', diverging=True)
    pcolor(axes[1, 0], xanom(cxx_g), "$C_{11}'$", cmap='RdBu_r', diverging=True)
    pcolor(axes[1, 1], xanom(cxy_g), "$C_{12}'$", cmap='RdBu_r', diverging=True)
    pcolor(axes[1, 2], xanom(cyy_g), "$C_{22}'$", cmap='RdBu_r', diverging=True)

    plt.tight_layout()
    fname = f"narwhal_t{t_now:05.2f}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  [saved: {fname}]")

# -------------------------
# Time loop
# -------------------------
t  = 0.0
it = 0

if TIME_SCALING:
    # --- Timing mode ---
    timing_results = {}
    checkpoint_idx = 0
    wall_start     = time.perf_counter()

    if comm.rank == 0:
        print(f"\n=== Timing mode: checkpoints at {TIME_CHECKPOINTS} ===\n")

    while t < t_end - 1e-14:
        update_forcing_from_C()
        stokes_solver.solve()
        csolver.step(dt_step)
        t  += dt_step
        it += 1

        # Heartbeat every 200 steps so you know it's running
        if it % 200 == 0 and comm.rank == 0:
            elapsed = time.perf_counter() - wall_start
            print(f"  t={t:.2f}  elapsed={elapsed:.1f}s  ({elapsed/it*1000:.2f} ms/step)")

        if checkpoint_idx < len(TIME_CHECKPOINTS) and t >= TIME_CHECKPOINTS[checkpoint_idx] - 1e-10:
            elapsed = time.perf_counter() - wall_start
            timing_results[TIME_CHECKPOINTS[checkpoint_idx]] = elapsed
            if comm.rank == 0:
                print(f"  Checkpoint t={TIME_CHECKPOINTS[checkpoint_idx]:.1f}: "
                      f"{elapsed:.2f}s  ({elapsed/60:.2f} min)  "
                      f"[{it} steps,  {elapsed/it*1000:.3f} ms/step]")
            checkpoint_idx += 1

        if checkpoint_idx >= len(TIME_CHECKPOINTS):
            break

    if comm.rank == 0 and len(timing_results) >= 2:
        chk = TIME_CHECKPOINTS
        t1, t2   = chk[0], chk[1]
        w1, w2   = timing_results[t1], timing_results[t2]
        expected = t2 / t1
        actual   = w2 / w1
        print(f"\n=== Scaling summary ===")
        print(f"  t=0 → {t1:.1f}:  {w1:.2f}s")
        print(f"  t=0 → {t2:.1f}:  {w2:.2f}s")
        print(f"  Expected ratio (linear): {expected:.2f}x")
        print(f"  Actual ratio:            {actual:.3f}x")
        verdict = "✓ linear" if abs(actual - expected) < 0.05 * expected else "⚠ drifting"
        print(f"  {verdict}")

else:
    # --- Normal simulation mode ---
    save_2d_plot(t)
    next_plot = plot_interval

    while t < t_end - 1e-14:
        update_forcing_from_C()
        stokes_solver.solve()
        csolver.step(dt_step)
        t  += dt_step
        it += 1

        if it % 200 == 0 and comm.rank == 0:
            cxx.change_scales(1);  cyy.change_scales(1)
            tr_max = np.max(cxx['g'] + cyy['g'])
            spd_ok = "OK" if tr_max > 0 else "⚠ SPD"
            print(f"t={t:.3f}  max(TrC)={tr_max:.3e}  [{spd_ok}]")

        if t >= next_plot - 1e-10:
            save_2d_plot(t)
            next_plot += plot_interval

    save_2d_plot(t)

    if comm.rank == 0:
        print(f"\n=== Done  t={t:.3f} ===")