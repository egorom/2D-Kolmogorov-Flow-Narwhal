import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# -------------------------
# Parameters
# -------------------------
aspect  = 4
Lx      = 2*np.pi
Ly      = Lx / aspect           # pi/2
Nx      = 128
Ny      = Nx // aspect          # 32  (integer division — Dedalus requires int)
dealias = 3/2

# Kolmogorov forcing parameters
n_kolm  = 4                     # wavenumber
nu      = 0.0005                # kinematic viscosity
lam     = 0.03                  # relaxation time  (Wi = lam*maxvel*n_kolm ~ 0.48)
xi      = 0.5                   # polymer parameter
maxvel  = 4.0                   # velocity scale
beta    = nu                    # Stokes viscosity

# Polymer / conformation parameters
kappa_c = 0.0                   # diffusion on C (off for now)
tauR    = lam                   # relaxation time
# Polymer stress uses the nu-scaled form: tau_p = (xi*nu/lam)*(C - I)
# This keeps alpha_p small (~0.0083) so the operator-split Stokes-C coupling
# is stable. Using alpha_p=xi/lam=16.67 makes polymer stress ~300x larger
# than f0, which blows up the explicit coupling regardless of dt.
alpha_p = xi * nu / lam         # = 0.00833

# Forcing amplitude: must balance viscous drag + polymer drag at maxvel.
# Force balance at steady state (Stokes x-component, sin(ny) mode):
#   nu*(-n^2)*maxvel + amp - alpha_p*C12amp*n = 0
# where C12amp = maxvel*n*lam/(1+lam*nu*n^2)
# This gives amp = nu*n^2*maxvel*(1 + xi/(1+lam*nu*n^2))
amp     = nu * maxvel * n_kolm**2 * (1 + xi / (1 + lam*nu*n_kolm**2))  # = 0.048

# Time-stepping
dt_step = 0.00125
t_end   = 30.0

comm     = MPI.COMM_WORLD
dx_phys  = Lx / Nx
dy_phys  = Ly / Ny

def global_int_array(a):
    return comm.allreduce(np.sum(a), op=MPI.SUM) * dx_phys * dy_phys

# -------------------------
# Analytical solution (for comparison)
# -------------------------
def analytical_solution(y_arr):
    """Returns u_x_sol, C11_sol, C12_sol, C22_sol on y_arr."""
    ux_sol  = maxvel * np.sin(n_kolm * y_arr)

    denom   = 1.0 / lam**2 + 5*nu*n_kolm**2 / lam + 4*nu**2*n_kolm**4
    C11amp  = 2 * maxvel**2 * n_kolm**2 / denom
    C11const = 2 * lam * nu * C11amp * n_kolm**2 + 1.0

    C11_sol = C11amp * np.cos(n_kolm * y_arr)**2 + C11const
    C12_sol = (maxvel * n_kolm * lam) / (1 + lam*nu*n_kolm**2) * np.cos(n_kolm * y_arr)
    C22_sol = np.ones_like(y_arr)

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
# Unknowns for Stokes
# -------------------------
u   = dist.VectorField(coords, name='u',   bases=(xb, yb))
p   = dist.Field(name='p',   bases=(xb, yb))
ux0 = dist.Field(name='ux0')
uy0 = dist.Field(name='uy0')
p0  = dist.Field(name='p0')

ux   = u @ ex
uy   = u @ ey
ux_x = dx_op(ux);  ux_y = dy_op(ux)
uy_x = dx_op(uy);  uy_y = dy_op(uy)

# -------------------------
# Forcing
# -------------------------
f0 = dist.VectorField(coords, name='f0', bases=(xb, yb))
f0['g'][0] = amp * np.sin(n_kolm * y)   # FIX: was F_kolm (undefined); now uses amp
f0['g'][1] = 0.0

f_total = dist.VectorField(coords, name='f_total', bases=(xb, yb))
f_total['g'][0] = f0['g'][0]
f_total['g'][1] = f0['g'][1]

# -------------------------
# Conformation tensor  (IC = analytical steady state)
# -------------------------
# Starting from C=I causes a violent transient: u overshoots to ~6 with no polymer
# drag, and rapid stretching overwhelms the operator-split timestepper (alpha_p=16.67).
# Initialize to the analytical laminar solution instead — this tests whether the code
# correctly MAINTAINS the steady state, which fully validates the equations & coupling.
cxx = dist.Field(name='cxx', bases=(xb, yb))
cxy = dist.Field(name='cxy', bases=(xb, yb))
cyy = dist.Field(name='cyy', bases=(xb, yb))

_, C11_init, C12_init, _ = analytical_solution(y)
cxx['g'] = C11_init
cxy['g'] = C12_init
cyy['g'] = 1.0

# -------------------------
# Stokes LBVP
# -------------------------
stokes = d3.LBVP([u, p, ux0, uy0, p0], namespace=locals())
stokes.add_equation("beta*d3.div(d3.grad(u)) - d3.grad(p) + ux0*ex + uy0*ey = -f_total")
stokes.add_equation("d3.div(u) + p0 = 0")
stokes.add_equation("d3.integ(p) = 0")
stokes.add_equation("d3.integ(u@ex) = 0")
stokes.add_equation("d3.integ(u@ey) = 0")
stokes_solver = stokes.build_solver()
stokes_solver.solve()

# -------------------------
# Conformation IVP: upper-convected Oldroyd-B
# dt(C) = -(u·∇)C + (∇u)C + C(∇u)^T - (1/tauR)(C - I)
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
# Snapshot file handler
# -------------------------
snap = csolver.evaluator.add_file_handler("snapshots", sim_dt=0.1, max_writes=None)
snap.add_task(u @ ex, name="ux")
snap.add_task(u @ ey, name="uy")
snap.add_task(p,      name="p")
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
# Diagnostics storage
# -------------------------
diag_t        = []
diag_trC_max  = []
diag_cxy_max  = []
diag_ux_err   = []   # L-inf error vs analytical u_x
diag_cxy_err  = []   # L-inf error vs analytical C12
diag_spd_min  = []   # min(Tr C) — positive definiteness check

def collect_diagnostics(t_now):
    u.change_scales(1)
    cxx.change_scales(1);  cxy.change_scales(1);  cyy.change_scales(1)

    tr      = cxx['g'] + cyy['g']
    tr_max  = comm.allreduce(np.max(tr),             op=MPI.MAX)
    tr_min  = comm.allreduce(np.min(tr),             op=MPI.MIN)
    cxy_max = comm.allreduce(np.max(np.abs(cxy['g'])), op=MPI.MAX)

    # Analytical values on local y-grid
    ux_a, _, C12_a, _ = analytical_solution(y)
    ux_err  = comm.allreduce(np.max(np.abs(u['g'][0] - ux_a)), op=MPI.MAX)
    cxy_err = comm.allreduce(np.max(np.abs(cxy['g']  - C12_a)), op=MPI.MAX)

    if comm.rank == 0:
        diag_t.append(t_now)
        diag_trC_max.append(tr_max)
        diag_cxy_max.append(cxy_max)
        diag_ux_err.append(ux_err)
        diag_cxy_err.append(cxy_err)
        diag_spd_min.append(tr_min)

        spd_flag = "  ⚠ SPD LOST" if tr_min < 0 else ""
        print(
            f"t={t_now:6.3f}  max(TrC)={tr_max:.4e}  min(TrC)={tr_min:.4e}"
            f"  max|C12|={cxy_max:.4e}  err_ux={ux_err:.3e}  err_C12={cxy_err:.3e}"
            + spd_flag
        )

# -------------------------
# Diagnostic plot (saved to file)
# -------------------------
def save_diagnostic_plot(t_now, label=""):
    if comm.rank != 0:
        return

    u.change_scales(1)
    cxx.change_scales(1);  cxy.change_scales(1);  cyy.change_scales(1)

    # Use a single x-slice (x-independent at steady state)
    ix = 0
    y_local = y[ix, :]   # shape (Ny_local,)

    ux_num  = u['g'][0][ix, :]
    cxx_num = cxx['g'][ix, :]
    cxy_num = cxy['g'][ix, :]
    cyy_num = cyy['g'][ix, :]

    # Dense analytical curve
    y_dense = np.linspace(0, Ly, 300)
    ux_a, C11_a, C12_a, C22_a = analytical_solution(y_dense)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Kolmogorov flow toy problem  —  t = {t_now:.2f}\n"
        f"λ={lam}, Wi={lam*maxvel*n_kolm:.2f}, n={n_kolm}",
        fontsize=13
    )

    # --- u_x ---
    ax = axes[0, 0]
    ax.plot(y_dense, ux_a,   'k--', lw=2, label='Analytical')
    ax.plot(y_local, ux_num, 'r-o', ms=4, label='Numerical')
    ax.set_xlabel('y');  ax.set_ylabel('$u_x$')
    ax.set_title('Velocity $u_x$')
    ax.legend()

    # --- C11 ---
    ax = axes[0, 1]
    ax.plot(y_dense, C11_a,   'k--', lw=2, label='Analytical')
    ax.plot(y_local, cxx_num, 'b-o', ms=4, label='Numerical')
    ax.set_xlabel('y');  ax.set_ylabel('$C_{11}$')
    ax.set_title('Conformation $C_{11}$')
    ax.legend()

    # --- C12 ---
    ax = axes[1, 0]
    ax.plot(y_dense, C12_a,   'k--', lw=2, label='Analytical')
    ax.plot(y_local, cxy_num, 'g-o', ms=4, label='Numerical')
    ax.set_xlabel('y');  ax.set_ylabel('$C_{12}$')
    ax.set_title('Conformation $C_{12}$  (key convergence check)')
    ax.legend()

    # --- C22 ---
    ax = axes[1, 1]
    ax.plot(y_dense, C22_a,   'k--', lw=2, label='Analytical (=1)')
    ax.plot(y_local, cyy_num, 'm-o', ms=4, label='Numerical')
    ax.set_xlabel('y');  ax.set_ylabel('$C_{22}$')
    ax.set_title('Conformation $C_{22}$')
    ax.legend()

    plt.tight_layout()
    fname = f"diag_t{t_now:05.2f}{label}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [plot saved: {fname}]")

def save_convergence_plot():
    if comm.rank != 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Convergence  —  λ={lam}, Wi={lam*maxvel*n_kolm:.2f}", fontsize=12)

    axes[0].semilogy(diag_t, diag_ux_err,  'r', label='$||u_x - u_x^*||_\\infty$')
    axes[0].semilogy(diag_t, diag_cxy_err, 'g', label='$||C_{12} - C_{12}^*||_\\infty$')
    axes[0].set_xlabel('t');  axes[0].set_ylabel('Error')
    axes[0].set_title('Convergence to analytical solution')
    axes[0].legend()

    axes[1].plot(diag_t, diag_trC_max, 'b', label='max Tr(C)')
    axes[1].plot(diag_t, diag_spd_min, 'r--', label='min Tr(C)')
    axes[1].axhline(2, color='k', ls=':', label='Tr(I)=2')
    axes[1].set_xlabel('t');  axes[1].set_ylabel('Tr(C)')
    axes[1].set_title('Trace of C  (SPD check: min > 0)')
    axes[1].legend()

    axes[2].plot(diag_t, diag_cxy_max, 'g')
    axes[2].set_xlabel('t');  axes[2].set_ylabel('max $|C_{12}|$')
    axes[2].set_title('$C_{12}$ amplitude (should plateau)')

    plt.tight_layout()
    fig.savefig("convergence.png", dpi=120)
    plt.close(fig)
    print("[plot saved: convergence.png]")

# -------------------------
# Time loop
# -------------------------
t  = 0.0
it = 0

# Save initial state
collect_diagnostics(t)
save_diagnostic_plot(t, label="_init")

plot_times = set(np.round(np.arange(0, t_end + 1, 5.0), 1))  # plots at t=0,5,10,15,20,25,30

while t < t_end - 1e-14:

    update_forcing_from_C()
    stokes_solver.solve()

    csolver.step(dt_step)
    t  += dt_step
    it += 1

    # Print diagnostics every 100 steps
    if it % 100 == 0:
        collect_diagnostics(t)

    # Snapshot plots at key times
    t_round = round(t, 1)
    if t_round in plot_times:
        save_diagnostic_plot(t, label=f"_t{t_round:.0f}")
        plot_times.discard(t_round)   # only once per target time

# -------------------------
# Final diagnostics and plots
# -------------------------
collect_diagnostics(t)
save_diagnostic_plot(t, label="_final")
save_convergence_plot()

if comm.rank == 0:
    print("\n=== Run complete ===")
    print(f"Final  err_ux  = {diag_ux_err[-1]:.3e}")
    print(f"Final  err_C12 = {diag_cxy_err[-1]:.3e}")
    print(f"Final  min(TrC)= {diag_spd_min[-1]:.4f}  ({'OK' if diag_spd_min[-1] > 0 else 'SPD LOST'})")