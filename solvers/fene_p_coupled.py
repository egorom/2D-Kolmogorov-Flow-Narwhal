"""
Dedalus simulation of a 2D periodic incompressible viscoelastic (FENE-P)
Kolmogorov flow. Fixed typos and equation syntax from prior script.
"""

import numpy as np
import os
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# =============================================================
# parameters (from your script)
# =============================================================
Lamin = 1.695     # lambda (polymer relaxation time)
branch = 4

params = {}
params['nper'] = 4
params['aspect'] = 4
params['nu'] = 0.0005
params['lam'] = Lamin
params['maxvel'] = 4.0
params['xi'] = 0.5

# grid/time
Nx = 512
Ny = int(Nx / params['aspect'])
Lx = 2.0 * np.pi
Ly = Lx / params['aspect']
dt = 0.01 / (2 ** (np.log2(Nx) - 6))

if branch in [3, 4, 5]:
    runtime = 20
elif branch == 6:
    runtime = 40

Nt = int(np.ceil(runtime / dt))

out_dir = f'./Wi{16*Lamin:1.3f}'
os.makedirs(out_dir, exist_ok=True)

# ==========================================================
# DEDALUS setup (fixed spellings)
# ==========================================================
dealias = 3/2
timestepper = d3.RK222
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
# --- FIX: use correct keyword names `bounds` and `dealias`
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
Cxx = dist.Field(name='Cxx', bases=(xbasis, ybasis))
Cxy = dist.Field(name='Cxy', bases=(xbasis, ybasis))
Cyy = dist.Field(name='Cyy', bases=(xbasis, ybasis))

# Local grids / unit vectors
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# ==========================================================
# Physical parameters for FENE-P (from your choices)
# ==========================================================
Re = 1.0 / params['nu']
We = params['lam']
L2 = 100.0
beta = 0.9

# Kolmogorov forcing (fixed nper**2 in formula)
amp = params['maxvel'] * (params['nper'] ** 2) * (1
      + params['xi'] / (1 + params['lam'] * params['nu'] * (params['nper'] ** 2)))
fx_field = dist.Field(name='fx', bases=(xbasis, ybasis))
fy_field = dist.Field(name='fy', bases=(xbasis, ybasis))
fx_field['g'] = amp * np.sin(params['nper'] * y)
fy_field['g'] = 0.0 * y
fx = fx_field  # symbolic name for Dedalus
fy = fy_field


# =========================================================
# FENE-P constitutive helper expressions (Peterlin f(C))
# =========================================================
# Note: these are symbolic names in the namespace used by Dedalus
# Peterlin factor as a symbolic field (Dedalus automatically handles field arithmetic)
fC = (L2 - 3)/(L2 - (Cxx + Cyy))  # ok as symbolic field

# Polymer stress terms as symbolic expressions
tau_xx_expr = (fC*Cxx - 1)/We
tau_xy_expr = (fC*Cxy)/We
tau_yy_expr = (fC*Cyy - 1)/We


# =========================================================
# Build IVP: use componentwise momentum equations for clarity
# =========================================================
problem = d3.IVP([u, p, Cxx, Cxy, Cyy], namespace=locals())

nu_eff = beta / Re  # solvent viscous part

#momentum
problem.add_equation("dt(u@ex) + dx(p) - nu_eff*lap(u@ex) = - u@grad(u@ex) + (1-beta)/Re*(dx(tau_xx_expr) + dy(tau_xy_expr)) + fx")
problem.add_equation("dt(u@ey) + dy(p) - nu_eff*lap(u@ey) = - u@grad(u@ey) + (1-beta)/Re*(dx(tau_xy_expr) + dy(tau_yy_expr)) + fy")

# Incompressibility
problem.add_equation("div(u) = 0")

# Conformation tensor evolution (upper-convected + Peterlin relaxation)
problem.add_equation("dt(Cxx) + u@grad(Cxx) = 2*Cxx*dx(u@ex) + 2*Cxy*dy(u@ex) - (fC*Cxx - 1)/We")
problem.add_equation("dt(Cxy) + u@grad(Cxy) = Cxy*dx(u@ex) + Cyy*dy(u@ex) + Cxx*dx(u@ey) + Cxy*dy(u@ey) - fC*Cxy/We")
problem.add_equation("dt(Cyy) + u@grad(Cyy) = 2*Cxy*dx(u@ey) + 2*Cyy*dy(u@ey) - (fC*Cyy - 1)/We")

# Pressure gauge
problem.add_equation("integ(p) = 0")

# ============================================================
# Solver
# ============================================================
solver = problem.build_solver(timestepper)
solver.stop_sim_time = runtime

# ============================================================
# Initial conditions (same Kolmogorov base flow as MATLAB)
# ============================================================
u['g'][0] = params['maxvel'] * np.sin(params['nper'] * y)  # u_x
u['g'][1] = 0.0  # u_y

Cxx['g'] = 1.0
Cxy['g'] = 0.0
Cyy['g'] = 1.0

# ============================================================
# Analysis output
# ============================================================
snapshots = solver.evaluator.add_file_handler(os.path.join(out_dir, 'snapshots'), sim_dt=1.0, max_writes=50)
snapshots.add_tasks([u, p, Cxx, Cxy, Cyy], name='fields')

# ============================================================
# CFL
# ============================================================
CFL = d3.CFL(solver, initial_dt=dt, cadence=10, safety=0.2, threshold=0.1, max_change=1.5, min_change=0.5)
CFL.add_velocity(u)

# ============================================================
# Main loop
# ============================================================
try:
    logger.info('Starting FENE-P Kolmogorov flow simulation (fixed script)')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 50 == 0:
            logger.info(f"Iter {solver.iteration}, time {solver.sim_time:.3f}, dt {timestep:.3e}")
except Exception as e:
    logger.error('Simulation crashed: ' + str(e))
    raise
finally:
    solver.log_stats()
