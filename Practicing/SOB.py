import numpy as np
import os
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

# ===============================
# Parameters
# ===============================
Lamin = 1.695
branch = 4

params = {
    'nper': 4,
    'aspect': 4,
    'nu': 0.0005,
    'lam': Lamin,
    'maxvel': 4.0,
    'xi': 0.5
}

# Grid and time
Nx = 512
Ny = int(Nx / params['aspect'])
Lx = 2*np.pi
Ly = Lx / params['aspect']
dt = 0.01 / (2**(np.log2(Nx)-6))

RunTime = 20 if branch in [3,4,5] else 40
Nt = int(np.ceil(RunTime / dt))

out_dir = f'./Wi{16*Lamin:1.3f}'
os.makedirs(out_dir, exist_ok=True)

# ===============================
# Dedalus domain
# ===============================
dealias = 3/2
dtype = np.float64
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0,Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0,Ly), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))
Cxx = dist.Field(name='Cxx', bases=(xbasis,ybasis))
Cxy = dist.Field(name='Cxy', bases=(xbasis,ybasis))
Cyy = dist.Field(name='Cyy', bases=(xbasis,ybasis))

x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# Physical parameters
Re = 1.0 / params['nu']
We = params['lam']
beta = 0.9
nu_eff = beta / Re

# Forcing as Field
amp = params['maxvel'] * params['nper']**2 * (1 + params['xi'] / (1 + params['lam']*params['nu']*params['nper']**2))
fx = dist.Field(name='fx', bases=(xbasis,ybasis))
fy = dist.Field(name='fy', bases=(xbasis,ybasis))
fx['g'] = amp*np.sin(params['nper']*y)
fy['g'] = 0.0*y

# ===============================
# Oldroyd-B stress components
# ===============================
tau_xx = dist.Field(name='tau_xx', bases=(xbasis,ybasis))
tau_xy = dist.Field(name='tau_xy', bases=(xbasis,ybasis))
tau_yy = dist.Field(name='tau_yy', bases=(xbasis,ybasis))

# ===============================
# Build IVP
# ===============================
problem = d3.IVP([u,p,Cxx,Cxy,Cyy], namespace=locals())

# Define stress in namespace
problem.namespace['tau_xx'] = (Cxx-1)/We
problem.namespace['tau_xy'] = Cxy/We
problem.namespace['tau_yy'] = (Cyy-1)/We

# Momentum (Stokes-Oldroyd-B)
problem.add_equation("-p.dx + nu_eff*(u@ex).lap + (1-beta)*(tau_xx.dx + tau_xy.dy) + fx = 0")
problem.add_equation("-p.dy + nu_eff*(u@ey).lap + (1-beta)*(tau_xy.dx + tau_yy.dy) + fy = 0")

# Incompressibility
problem.add_equation("u.div = 0")

# Upper-convected Oldroyd-B stress evolution
problem.add_equation("dt(Cxx) + u@grad(Cxx) - 2*Cxx*dx(u@ex) - 2*Cxy*dy(u@ex) + (Cxx-1)/We = 0")
problem.add_equation("dt(Cxy) + u@grad(Cxy) - Cxy*dx(u@ex) - Cyy*dy(u@ex) - Cxx*dx(u@ey) - Cxy*dy(u@ey) + Cxy/We = 0")
problem.add_equation("dt(Cyy) + u@grad(Cyy) - 2*Cxy*dx(u@ey) - 2*Cyy*dy(u@ey) + (Cyy-1)/We = 0")

# Pressure gauge
problem.add_equation("integ(p) = 0")

# ===============================
# Solver setup
# ===============================
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = RunTime

# Initial conditions
u['g'][0] = params['maxvel']*np.sin(params['nper']*y)
u['g'][1] = 0.0
Cxx['g'] = 1.0
Cxy['g'] = 0.0
Cyy['g'] = 1.0

# ===============================
# CFL (optional)
# ===============================
CFL = d3.CFL(solver, initial_dt=dt, cadence=10, safety=0.2)
CFL.add_velocity(u)

# ===============================
# Main loop
# ===============================
try:
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
except Exception as e:
    print("Simulation crashed:", e)
finally:
    print("Simulation finished at t =", solver.sim_time)
