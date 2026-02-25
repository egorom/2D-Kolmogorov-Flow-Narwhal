import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

# =============================================================================
# Parameters
# =============================================================================
Lamin  = 0.85
#branch = 1

nper   = 4
aspect = 4
nu     = 0.0005
lam    = Lamin
maxvel = 4.0
xi     = 0.5
Wi     = 16 * lam
      

Nx = 512
Ny = int(Nx / aspect)
Lx = 4      
Ly = Lx / aspect
dt = 0.01 / (2 ** (np.log2(Nx) - 6))

amp = maxvel * (nper**2) * (1 + xi / (1 + lam * nu * nper**2))

runtime=10*lam #for testing!
#if branch in [3, 4, 5]:
#    runtime = 20
#elif branch == 6:
#    runtime = 40

# =============================================================================
# Bases & distributor
# =============================================================================
coords = d3.CartesianCoordinates('x', 'y')
dist   = d3.Distributor(coords, dtype=np.float64)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=3/2)

# =============================================================================
# Exponential filter  (Hou & Li 2007, matching MATLAB expfilter2D)
# Built once in coefficient space on the non-dealiased grid
# =============================================================================
kx_1d = xbasis.wavenumbers   # shape (Nx,)
ky_1d = ybasis.wavenumbers   # shape (Ny,)
KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='ij')

a_filt = 36
m_filt = 36
kxmax  = (2 * np.pi / Lx) * (Nx / 2)
kymax  = (2 * np.pi / Ly) * (Ny / 2)
exp_filter = (np.exp(-a_filt * (np.abs(KX) / kxmax) ** m_filt) *
              np.exp(-a_filt * (np.abs(KY) / kymax) ** m_filt))   # (Nx, Ny)

# =============================================================================
# Fields
# =============================================================================
x = dist.local_grid(xbasis)
y = dist.local_grid(ybasis)

# Conformation tensor — evolved by IVP
C_xx = dist.Field(bases=(xbasis, ybasis))
C_xy = dist.Field(bases=(xbasis, ybasis))
C_yy = dist.Field(bases=(xbasis, ybasis))

# Scalar velocity components — updated from LBVP each step, used as
# frozen coefficients in the conformation IVP
ux = dist.Field(bases=(xbasis, ybasis))
uy = dist.Field(bases=(xbasis, ybasis))

# Polymer stress divergence — updated each step before IVP advance
f_poly = dist.VectorField(coords, bases=(xbasis, ybasis))

# =============================================================================
# Differentiation shorthands
# =============================================================================
ddx = lambda A: d3.Differentiate(A, coords['x'])
ddy = lambda A: d3.Differentiate(A, coords['y'])

# =============================================================================
# LBVP for Stokes solve  (replaces IVP momentum equation)
# Solved fresh each timestep given updated f_poly
# Matches MATLAB Stokes_solve_hat: instantaneous, no dt(u)
# =============================================================================
u_stokes   = dist.VectorField(coords, bases=(xbasis, ybasis))
p_stokes   = dist.Field(bases=(xbasis, ybasis))
tau_p_lbvp = dist.Field()
tau_u_lbvp = dist.VectorField(coords)

f_ext = dist.VectorField(coords, bases=(xbasis, ybasis))
f_ext['g'][0] = amp * np.sin(nper * y)
f_ext['g'][1] = 0.0

# Remove tau_u_lbvp from the LBVP variables and namespace
lbvp = d3.LBVP([u_stokes, p_stokes, tau_p_lbvp], namespace={
    'u':      u_stokes,
    'p':      p_stokes,
    'tau_p':  tau_p_lbvp,
    'f_ext':  f_ext,
    'f_poly': f_poly,
    'nu':     nu,
    'grad':   d3.Gradient,
    'div':    d3.Divergence,
    'lap':    d3.Laplacian,
    'integ':  d3.integ,
})

lbvp.add_equation("grad(p) - nu*lap(u) = f_ext + f_poly")
lbvp.add_equation("div(u) + tau_p = 0")
lbvp.add_equation("integ(p) = 0")

stokes_solver = lbvp.build_solver()

# =============================================================================
# IVP for conformation tensor only
# ux, uy enter as frozen scalar fields updated each step from the LBVP
# 
# =============================================================================

ivp_ns = {
    'dt':    d3.TimeDerivative,
    'lap':   d3.Laplacian,
    'ddx':   ddx,
    'ddy':   ddy,
    'ux':    ux,
    'uy':    uy,
    'C_xx':  C_xx,
    'C_xy':  C_xy,
    'C_yy':  C_yy,
    'Wi':    Wi,
    'nu':    nu,          # <-- add this
    'integ': d3.integ,
}

# IVP only solves for C_xx, C_xy, C_yy
conf_problem = d3.IVP([C_xx, C_xy, C_yy], namespace=ivp_ns)

# Equations without tau terms:
conf_problem.add_equation(
    "dt(C_xx) + ux*ddx(C_xx) + uy*ddy(C_xx)"
    " - 2*ddx(ux)*C_xx - 2*ddy(ux)*C_xy"
    " - nu*lap(C_xx)"
    " = -(C_xx - 1)/Wi"
)
conf_problem.add_equation(
    "dt(C_xy) + ux*ddx(C_xy) + uy*ddy(C_xy)"
    " - ddx(ux)*C_xy - ddy(uy)*C_xy"
    " - ddy(ux)*C_xx - ddx(uy)*C_yy"
    " - nu*lap(C_xy)"
    " = -C_xy/Wi"
)
conf_problem.add_equation(
    "dt(C_yy) + ux*ddx(C_yy) + uy*ddy(C_yy)"
    " - 2*ddy(uy)*C_yy - 2*ddx(uy)*C_xy"
    " - nu*lap(C_yy)"
    " = -(C_yy - 1)/Wi"
)


conf_solver = conf_problem.build_solver(d3.RK443)

# =============================================================================
# Initial conditions
# =============================================================================
C11amp   = (2 * maxvel**2 * nper**2 /
            (1/lam**2 + (5*nu*nper**2)/lam + 4*nu**2*nper**4))
C11const = 2 * lam * nu * C11amp * nper**2 + 1

C_xx.change_scales(1)
C_xy.change_scales(1)
C_yy.change_scales(1)
#C_xx['g'] = C11amp * np.cos(nper * y)**2 + C11const
#C_xy['g'] = (maxvel * nper * lam) / (1 + lam*nu*nper**2) * np.cos(nper * y)
#C_yy['g'] = 1.0

#for testing purposes!
C_xx['g'] = 1.0   # C = I
C_xy['g'] = 0.0
C_yy['g'] = 1.0

# Initial Stokes solve to get u consistent with initial C
# First compute f_poly from initial C, then solve
def update_f_poly():
    """
    Compute polymer stress divergence with xi/lam prefactor.
    Matches MATLAB get_polyforce_hat: Scale = xi/lam
    div(sigma_p) = (xi/lam) * [dC_xx/dx + dC_xy/dy,  dC_xy/dx + dC_yy/dy]
    """
    scale = xi / lam
    dCxx_dx = ddx(C_xx).evaluate(); dCxx_dx.change_scales(1)
    dCxy_dy = ddy(C_xy).evaluate(); dCxy_dy.change_scales(1)
    dCxy_dx = ddx(C_xy).evaluate(); dCxy_dx.change_scales(1)
    dCyy_dy = ddy(C_yy).evaluate(); dCyy_dy.change_scales(1)
    f_poly.change_scales(1)
    f_poly['g'][0] = scale * (dCxx_dx['g'] + dCxy_dy['g'])
    f_poly['g'][1] = scale * (dCxy_dx['g'] + dCyy_dy['g'])

def run_stokes():
    stokes_solver.solve()
    u_stokes.change_scales(1)
    ux.change_scales(1)
    uy.change_scales(1)
    ux['g'] = u_stokes['g'][0].copy()
    uy['g'] = u_stokes['g'][1].copy()
    ux.change_scales(3/2)   # force Dedalus to recompute coefficients
    ux.change_scales(1)
    uy.change_scales(3/2)
    uy.change_scales(1)

update_f_poly()
run_stokes()
u_stokes.change_scales(1)
print("u_stokes max:", np.max(np.abs(u_stokes['g'][0])))
# Should be close to maxvel = 4.0

conf_solver.stop_sim_time  = runtime
conf_solver.stop_wall_time = np.inf
conf_solver.stop_iteration = np.inf

# =============================================================================
# Diagnostics & saving setup
# =============================================================================
saves_norm_per_unit_time = 20
saves_per_unit_time      = 20

savetr = int(1 / (saves_norm_per_unit_time * dt))
saveit = int(1 / (saves_per_unit_time * dt))

norms  = []
outdir = f"Wi_{Wi:.3f}"
os.makedirs(outdir, exist_ok=True)


def apply_filter_inplace(field):
    """Apply exponential filter to a Dedalus field in coefficient space (in-place)."""
    field.change_scales(1)
    c = field['c']
    if c.ndim == 2:
        c *= exp_filter
    elif c.ndim == 3:
        c *= exp_filter[np.newaxis, :, :]
    field['c'] = c


def compute_norms():
    C_xx.change_scales(1); C_yy.change_scales(1)
    u_stokes.change_scales(1)
    dx_phys = Lx / Nx
    dy_phys = Ly / Ny
    KE      = 0.5 * np.sum(u_stokes['g'][0]**2 + u_stokes['g'][1]**2) * dx_phys * dy_phys
    TrC_int = np.sum(C_xx['g'] + C_yy['g']) * dx_phys * dy_phys
    return KE, TrC_int


def check_spd():
    C_xx.change_scales(1); C_xy.change_scales(1); C_yy.change_scales(1)
    Tr  = C_xx['g'] + C_yy['g']
    Det = C_xx['g'] * C_yy['g'] - C_xy['g']**2
    return np.min(Tr), np.min(Det)


def save_snapshot(n, t):
    u_stokes.change_scales(1)
    C_xx.change_scales(1); C_xy.change_scales(1); C_yy.change_scales(1)
    np.savez(
        f"{outdir}/snapshot_{n:06d}.npz",
        t=t,
        u=u_stokes['g'].copy(),
        C_xx=C_xx['g'].copy(),
        C_xy=C_xy['g'].copy(),
        C_yy=C_yy['g'].copy(),
        x=x, y=y,
    )


# =============================================================================
# Time loop
#   1. RK443 advance conformation (ux/uy frozen from previous Stokes solve)
#   2. Update polymer force from new C
#   3. Stokes solve for new u
#   4. Diagnostics / saving (filter applied to diagnostic copies only)
# =============================================================================
n = 0

while conf_solver.proceed:

    # --- 1. Advance conformation by one RK443 step ---
    conf_solver.step(dt)
    n += 1

    # --- 2. Update polymer force with xi/lam prefactor ---
    update_f_poly()

    # --- 3. Stokes solve — updates u_stokes, ux, uy ---
    run_stokes()

    # --- 4a. Diagnostics on save steps ---
    if n % savetr == 0:
        t = conf_solver.sim_time
        min_Tr, min_Det = check_spd()
        if min_Tr <= 0 or min_Det <= 0:
            print(f"SPD violation at t={t:.3f}: min Tr={min_Tr:.3e}, min Det={min_Det:.3e}")
        KE, TrC = compute_norms()
        norms.append([t, KE, TrC])
        print(f"t={t:.3f}, KE={KE:.6e}, TrC={TrC:.6e}")

    # --- 4b. Snapshot saves ---
    if n % saveit == 0:
        save_snapshot(n, conf_solver.sim_time)

# =============================================================================
# Validation: check conformation matches analytical steady state
# =============================================================================
C_xx.change_scales(1)
C_xy.change_scales(1)
C_yy.change_scales(1)

# Analytical steady state
C_xx_sol = C11amp * np.cos(nper * y)**2 + C11const
C_xy_sol = (maxvel * nper * lam) / (1 + lam*nu*nper**2) * np.cos(nper * y)
C_yy_sol = np.ones_like(C_xx_sol)

# Pointwise absolute errors
err_xx = np.abs(C_xx['g'] - C_xx_sol)
err_xy = np.abs(C_xy['g'] - C_xy_sol)
err_yy = np.abs(C_yy['g'] - C_yy_sol)

# L-inf and L2 errors
dx_phys = Lx / Nx
dy_phys = Ly / Ny

print("\n--- Conformation tensor convergence check ---")
for name, err in [("C_xx", err_xx), ("C_xy", err_xy), ("C_yy", err_yy)]:
    linf = np.max(err)
    l2   = np.sqrt(np.sum(err**2) * dx_phys * dy_phys)
    print(f"  {name}:  L_inf = {linf:.4e},  L2 = {l2:.4e}")

# Overall pass/fail — L_inf < 1% of max value of solution
tol = 0.01
max_sol = max(np.max(np.abs(C_xx_sol)), np.max(np.abs(C_xy_sol)), 1.0)
passed = (np.max(err_xx) < tol * max_sol and
          np.max(err_xy) < tol * max_sol and
          np.max(err_yy) < tol * max_sol)
print(f"\n  Tolerance: {tol*100:.1f}% of max solution amplitude")
print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

# Plot error fields
# Use meshgrid for plotting since x, y are 1D from dist.local_grid
XX, YY = np.meshgrid(x.ravel(), y.ravel(), indexing='ij')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f'Conformation error vs analytical steady state (t={conf_solver.sim_time:.2f})', fontsize=13)

for ax, err, name in zip(axes, [err_xx, err_xy, err_yy], ['C_xx', 'C_xy', 'C_yy']):
    im = ax.pcolormesh(XX, YY, err, cmap='viridis', shading='auto')
    ax.set_title(f'|{name} - {name}_sol|')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

# =============================================================================
# Post-run plots
# =============================================================================
norms   = np.array(norms)
t_arr   = norms[:, 0]
KE_arr  = norms[:, 1]
TrC_arr = norms[:, 2]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t_arr, KE_arr, 'b-')
axes[0].set_xlabel('Time');  axes[0].set_ylabel('Kinetic Energy')
axes[0].set_title('Kinetic Energy vs Time');  axes[0].grid(True)

axes[1].plot(t_arr, TrC_arr, 'r-')
axes[1].set_xlabel('Time');  axes[1].set_ylabel('Tr(C) integrated')
axes[1].set_title('Integrated Trace of Conformation Tensor vs Time');  axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"{outdir}/norms.png", dpi=150)
plt.show()

# Final snapshot plot
snapshot_files = sorted([f for f in os.listdir(outdir) if f.startswith('snapshot_')])
if snapshot_files:
    last       = np.load(f"{outdir}/{snapshot_files[-1]}")
    t_final    = float(last['t'])
    u_pl       = last['u']
    C_xx_pl    = last['C_xx']
    C_xy_pl    = last['C_xy']
    C_yy_pl    = last['C_yy']
    x_grid     = last['x']
    y_grid     = last['y']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Final State at t={t_final:.3f}', fontsize=14)

    XX_snap, YY_snap = np.meshgrid(x_grid.ravel(), y_grid.ravel(), indexing='ij')

    def pplot(ax, data, title, cmap='RdBu_r'):
        im = ax.pcolormesh(XX_snap, YY_snap, data, cmap=cmap, shading='auto')
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

    pplot(axes[0, 0], u_pl[0],              'u_x')
    pplot(axes[0, 1], u_pl[1],              'u_y')
    pplot(axes[0, 2], C_xx_pl + C_yy_pl,   'Tr(C)', cmap='viridis')
    pplot(axes[1, 0], C_xx_pl,              'C_xx',  cmap='viridis')
    pplot(axes[1, 1], C_xy_pl,              'C_xy')
    pplot(axes[1, 2], C_yy_pl,              'C_yy',  cmap='viridis')

    for ax in axes.flat:
        ax.set_xlabel('x');  ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"{outdir}/final_snapshot.png", dpi=150)
    plt.show()
