import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -------------------------
# Parameters
# -------------------------
aspect  = 4
Lx      = 2*np.pi
Ly      = Lx / aspect           # pi/2
Nx      = 128
Ny      = Nx // aspect          # 32
dealias = 3/2

n_kolm  = 4
nu      = 0.0005
lam     = 1.695                 
xi      = 0.5
maxvel  = 4.0
beta    = nu

kappa_c = 1e-3  # polymer diffusion coefficient (small but nonzero for numerical stability)  
tauR    = lam
alpha_p = xi * nu / lam

amp     = nu * maxvel * n_kolm**2 * (1 + xi / (1 + lam*nu*n_kolm**2))

dt_step = 0.005                 # matches MATLAB dt for Nx=128
t_end   = 40.0
plot_interval = 1.0             # save a 2D snapshot every 1 time unit

comm    = MPI.COMM_WORLD
Wi      = lam * maxvel * n_kolm

# -------------------------
# Analytical laminar solution (used only as IC — system will depart from it)
# -------------------------
def analytical_solution(y_arr):
    ux_sol   = maxvel * np.sin(n_kolm * y_arr)
    denom    = 1/lam**2 + 5*nu*n_kolm**2/lam + 4*nu**2*n_kolm**4
    C11amp   = 2*maxvel**2*n_kolm**2 / denom
    C11const = 2*lam*nu*C11amp*n_kolm**2 + 1.0
    C11_sol  = C11amp * np.cos(n_kolm * y_arr)**2 + C11const
    C12_sol  = (maxvel*n_kolm*lam) / (1 + lam*nu*n_kolm**2) * np.cos(n_kolm * y_arr)
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
# Conformation tensor — IC = analytical laminar solution
# -------------------------
cxx = dist.Field(name='cxx', bases=(xb, yb))
cxy = dist.Field(name='cxy', bases=(xb, yb))
cyy = dist.Field(name='cyy', bases=(xb, yb))

_, C11_init, C12_init, _ = analytical_solution(y)


rng   = np.random.default_rng(seed=42)
C11max = np.max(C11_init)
pert_amp = 1e-3 * C11max  # 0.1% of peak C11 — small enough to avoid immediate blowup


X = x * np.ones((1, Ny))   # (Nx, Ny)
Y = y * np.ones((Nx, 1))   # (Nx, Ny)
pert = np.zeros((Nx, Ny))
for kx_p in range(1, 5):
    for ky_p in range(1, 5):
        phase = rng.uniform(0, 2*np.pi)
        pert += np.sin(kx_p * X + phase) * np.cos(ky_p * Y)
pert = pert_amp * pert / np.max(np.abs(pert))   # normalise to pert_amp

cxx['g'] = C11_init + pert   # perturb C11 only
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
# Dedalus snapshot handler
# -------------------------
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
# 2D snapshot plot — the Narwhal plot
# -------------------------
def save_2d_plot(t_now):
    if comm.rank != 0:
        return

    u.change_scales(1)
    cxx.change_scales(1);  cxy.change_scales(1);  cyy.change_scales(1)

    # Gather full 2D fields (shape: Nx x Ny)
    ux_g   = u['g'][0]
    uy_g   = u['g'][1]
    cxx_g  = cxx['g']
    cxy_g  = cxy['g']
    cyy_g  = cyy['g']
    trC_g  = cxx_g + cyy_g

    # Global x/y arrays for plotting (local grids are full since serial)
    X = x[:, 0]   # shape (Nx,)
    Y = y[0, :]   # shape (Ny,)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        f"Kolmogorov flow  —  t = {t_now:.2f}   "
        f"λ={lam}, Wi={Wi:.1f}, n={n_kolm}",
        fontsize=14
    )

    # Subtract x-average (mean over x at each y) from every field.
    # This removes the y-uniform background — whatever it currently is —
    # and reveals only the x-varying Narwhal anomaly, consistently for all fields.
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
        im = ax.pcolormesh(X, Y, data.T, cmap=cmap, norm=norm,
                           vmin=None if diverging else vmin,
                           vmax=None if diverging else vmax,
                           shading='auto')
        ax.set_xlabel('x');  ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pcolor(axes[0, 0], xanom(ux_g),  "$u_x' $",          cmap='RdBu_r', diverging=True)
    pcolor(axes[0, 1], uy_g,         '$u_y$',             cmap='RdBu_r', diverging=True)
    pcolor(axes[0, 2], xanom(trC_g), "Tr$(C)'$",          cmap='RdBu_r', diverging=True)
    pcolor(axes[1, 0], xanom(cxx_g), "$C_{11}'$",         cmap='RdBu_r', diverging=True)
    pcolor(axes[1, 1], xanom(cxy_g), "$C_{12}'$",         cmap='RdBu_r', diverging=True)
    pcolor(axes[1, 2], xanom(cyy_g), "$C_{22}'$",         cmap='RdBu_r', diverging=True)

    plt.tight_layout()
    fname = f"narwhal_t{t_now:05.2f}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  [saved: {fname}]")

# -------------------------
# Time loop
# -------------------------
t   = 0.0
it  = 0

save_2d_plot(t)   # t=0 snapshot

next_plot = plot_interval

while t < t_end - 1e-14:

    update_forcing_from_C()
    stokes_solver.solve()
    csolver.step(dt_step)
    t  += dt_step
    it += 1

    # Console print every 200 steps
    if it % 200 == 0 and comm.rank == 0:
        cxx.change_scales(1);  cyy.change_scales(1)
        tr_max = np.max(cxx['g'] + cyy['g'])
        spd_ok = "OK" if tr_max > 0 else "⚠ SPD"
        print(f"t={t:.3f}  max(TrC)={tr_max:.3e}  [{spd_ok}]")

    # 2D plot every plot_interval time units
    if t >= next_plot - 1e-10:
        save_2d_plot(t)
        next_plot += plot_interval

# Final snapshot
save_2d_plot(t)

if comm.rank == 0:
    print(f"\n=== Done  t={t:.3f} ===")