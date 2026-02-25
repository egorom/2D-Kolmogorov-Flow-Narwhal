import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)

# Parameters
Lamin = 1.695
branch = 4

nper = 4
aspect = 4
nu = 0.0005
lam = Lamin
maxvel = 4.0
xi = 0.5
Wi = 16 * lam
beta = 1.0

Nx = 512
Ny = int(Nx / aspect)
Lx = 4
Ly = Lx / aspect
dt = 0.01 / (2 ** (np.log2(Nx) - 6))

amp = maxvel * (nper ** 2) * (1 + xi / (1 + lam * nu * (nper ** 2)))

if branch in [3, 4, 5]:
    runtime = 20
elif branch == 6:
    runtime = 40

# Bases
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=3/2)

KX, KY = np.meshgrid(xbasis.wavenumbers, ybasis.wavenumbers, indexing='ij')
k_max = np.max(np.sqrt(KX**2 + KY**2))
k_cut = 0.66 * k_max

# Fields
u = dist.VectorField(coords, bases=(xbasis, ybasis))
p = dist.Field(bases=(xbasis, ybasis))

C_xx = dist.Field(bases=(xbasis, ybasis))
C_xy = dist.Field(bases=(xbasis, ybasis))
C_yy = dist.Field(bases=(xbasis, ybasis))

# Tau fields for gauge fixing
tau_p = dist.Field()
tau_u = dist.VectorField(coords)

x = dist.local_grid(xbasis)
y = dist.local_grid(ybasis)

# External forcing
f = dist.VectorField(coords, bases=(xbasis, ybasis))
f['g'][0] = amp * np.sin(nper * y)
f['g'][1] = 0

lap = d3.Laplacian
grad = d3.Gradient
div = d3.Divergence

# Single IVP for everything
problem = d3.IVP([u, p, tau_p, tau_u, C_xx, C_xy, C_yy], namespace=locals())

# Momentum equation (Stokes + polymer stress divergence)
problem.add_equation(
    "dt(u) + grad(p) - nu*lap(u) + tau_u"
    " = f"
    " - u@grad(u)"
    " + grad(C_xx)@coords['x'] * coords['x']"  # this needs to be done differently - see below
)

# We need to express div(C) as the forcing. Let's use substitution approach:
# f_poly_x = dx(C_xx) + dy(C_xy)
# f_poly_y = dx(C_xy) + dy(C_yy)
# In Dedalus 3 we can use d3.div on a tensor, but since C is symmetric
# we define it component-wise using Differentiate

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

# Redefine problem cleanly with explicit polymer stress divergence
problem = d3.IVP([u, p, tau_p, tau_u, C_xx, C_xy, C_yy], namespace=locals())

# Momentum
problem.add_equation(
    "dt(u) + grad(p) - nu*lap(u) + tau_u = f - u@grad(u)"
)

# Divergence-free
problem.add_equation("div(u) + tau_p = 0")

# Pressure gauge
problem.add_equation("integ(p) = 0")

# Velocity gauge
problem.add_equation("integ(u) = 0")

# Conformation tensor equations
problem.add_equation(
    "dt(C_xx) + u@grad(C_xx)"
    " - 2*dx(u@coords['x'])*C_xx"
    " - 2*dy(u@coords['x'])*C_xy"
    " = -(C_xx - 1)/Wi"
)

problem.add_equation(
    "dt(C_xy) + u@grad(C_xy)"
    " - dx(u@coords['x'])*C_xy"
    " - dy(u@coords['y'])*C_xy"
    " - dy(u@coords['x'])*C_xx"
    " - dx(u@coords['y'])*C_yy"
    " = -C_xy/Wi"
)

problem.add_equation(
    "dt(C_yy) + u@grad(C_yy)"
    " - 2*dy(u@coords['y'])*C_yy"
    " - 2*dx(u@coords['y'])*C_xy"
    " = -(C_yy - 1)/Wi"
)

solver = problem.build_solver(d3.RK443)

# -----------------------------------------
# Initial Conditions
# -----------------------------------------
u.change_scales(1)
u['g'][0] = maxvel * np.sin(nper * y)
u['g'][1] = 0

C11amp = 2 * maxvel**2 * nper**2 / (1/lam**2 + (5*nu*nper**2)/lam + 4*nu**2*nper**4)
C11const = 2 * lam * nu * C11amp * nper**2 + 1
C_xx['g'] = C11amp * (np.cos(nper * y))**2 + C11const
C_xy['g'] = (maxvel * nper * lam) / (1 + lam*nu*nper**2) * np.cos(nper * y)
C_yy['g'] = 1.0

solver.stop_sim_time = runtime
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# -----------------------------------------
# Diagnostics & Saving Setup
# -----------------------------------------
saves_norm_per_unit_time = 20
saves_per_unit_time = 20

savetr = int(1 / (saves_norm_per_unit_time * dt))
saveit = int(1 / (saves_per_unit_time * dt))

norms = []

outdir = f"Wi_{Wi:.3f}"
os.makedirs(outdir, exist_ok=True)


def compute_norms(u, C_xx, C_xy, C_yy):
    u.change_scales(1)
    C_xx.change_scales(1)
    C_yy.change_scales(1)

    ux = u['g'][0]
    uy = u['g'][1]

    KE = 0.5 * np.sum(ux**2 + uy**2) * (Lx/Nx) * (Ly/Ny)
    TrC = C_xx['g'] + C_yy['g']
    TrC_int = np.sum(TrC) * (Lx/Nx) * (Ly/Ny)

    return KE, TrC_int


def save_snapshot(n, t, u, C_xx, C_xy, C_yy):
    u.change_scales(1)
    C_xx.change_scales(1)
    C_xy.change_scales(1)
    C_yy.change_scales(1)

    np.savez(
        f"{outdir}/snapshot_{n:06d}.npz",
        t=t,
        u=u['g'].copy(),
        C_xx=C_xx['g'].copy(),
        C_xy=C_xy['g'].copy(),
        C_yy=C_yy['g'].copy(),
        x=x,
        y=y,
    )


def apply_filter(field, k_cut):
    field.change_scales(1)
    c = field['c']

    kx = xbasis.wavenumbers
    ky = ybasis.wavenumbers
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    mask = (K > k_cut)
    c[mask, ...] = 0.0
    field['c'] = c


def check_spd(C_xx, C_xy, C_yy):
    C_xx.change_scales(1)
    C_xy.change_scales(1)
    C_yy.change_scales(1)

    Cxx = C_xx['g']
    Cxy = C_xy['g']
    Cyy = C_yy['g']

    Tr = Cxx + Cyy
    Det = Cxx*Cyy - Cxy**2

    return np.min(Tr), np.min(Det)


# -----------------------------------------
# Time loop
# -----------------------------------------
n = 0

while solver.proceed:
    solver.step(dt)
    n += 1

    apply_filter(C_xx, k_cut)
    apply_filter(C_xy, k_cut)
    apply_filter(C_yy, k_cut)
    apply_filter(u, k_cut)

    if n % savetr == 0:
        t = solver.sim_time
        min_Tr, min_Det = check_spd(C_xx, C_xy, C_yy)
        if min_Tr <= 0 or min_Det <= 0:
            print(f"SPD violation at t={t:.3f}: min Tr={min_Tr:.3e}, min Det={min_Det:.3e}")

        KE, TrC = compute_norms(u, C_xx, C_xy, C_yy)
        norms.append([t, KE, TrC])
        print(f"t={t:.3f}, KE={KE:.6e}, TrC={TrC:.6e}")

    if n % saveit == 0:
        save_snapshot(n, solver.sim_time, u, C_xx, C_xy, C_yy)


# -----------------------------------------
# Post-run Plots
# -----------------------------------------
norms = np.array(norms)
t_arr = norms[:, 0]
KE_arr = norms[:, 1]
TrC_arr = norms[:, 2]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t_arr, KE_arr, 'b-')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Kinetic Energy')
axes[0].set_title('Kinetic Energy vs Time')
axes[0].grid(True)

axes[1].plot(t_arr, TrC_arr, 'r-')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Tr(C) integrated')
axes[1].set_title('Integrated Trace of Conformation Tensor vs Time')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"{outdir}/norms.png", dpi=150)
plt.show()

# Load and plot final snapshot
snapshot_files = sorted([f for f in os.listdir(outdir) if f.startswith('snapshot_')])
if snapshot_files:
    last = np.load(f"{outdir}/{snapshot_files[-1]}")
    t_final = float(last['t'])
    u_final = last['u']
    C_xx_final = last['C_xx']
    C_xy_final = last['C_xy']
    C_yy_final = last['C_yy']
    x_grid = last['x']
    y_grid = last['y']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Final State at t={t_final:.3f}', fontsize=14)

    im0 = axes[0, 0].pcolormesh(x_grid, y_grid.T, u_final[0].T, cmap='RdBu_r', shading='auto')
    axes[0, 0].set_title('u_x')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(x_grid, y_grid.T, u_final[1].T, cmap='RdBu_r', shading='auto')
    axes[0, 1].set_title('u_y')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    TrC_final = C_xx_final + C_yy_final
    im2 = axes[0, 2].pcolormesh(x_grid, y_grid.T, TrC_final.T, cmap='viridis', shading='auto')
    axes[0, 2].set_title('Tr(C) = C_xx + C_yy')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].pcolormesh(x_grid, y_grid.T, C_xx_final.T, cmap='viridis', shading='auto')
    axes[1, 0].set_title('C_xx')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(x_grid, y_grid.T, C_xy_final.T, cmap='RdBu_r', shading='auto')
    axes[1, 1].set_title('C_xy')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(x_grid, y_grid.T, C_yy_final.T, cmap='viridis', shading='auto')
    axes[1, 2].set_title('C_yy')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(f"{outdir}/final_snapshot.png", dpi=150)
    plt.show()