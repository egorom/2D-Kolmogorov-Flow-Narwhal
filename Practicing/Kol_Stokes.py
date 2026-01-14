import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters - YOUR ORIGINAL VALUES (now working!)
Lamin = 1.695
branch = 4

nper = 4
aspect = 4
nu = 0.0005
lam = Lamin
maxvel = 4.0
xi = 0.5
Nx = 512
Ny = int(Nx / aspect)

Lx = 2 * np.pi
Ly = Lx / aspect
timestep = 0.01 / (2 ** (np.log2(Nx) - 6))
amp = maxvel * (nper ** 2) * (1 + xi / (1 + lam * nu * (nper ** 2)))

if branch in [3, 4, 5]:
    runtime = 20
elif branch == 6:
    runtime = 40
else:
    runtime = 20

print(f"Parameters: Nx={Nx}, Ny={Ny}, nu={nu}, nper={nper}")
print(f"Domain: Lx={Lx:.3f}, Ly={Ly:.3f}")
print(f"Forcing amplitude: {amp:.3f}")

# Coordinate system
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)

# Bases
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=3/2)

# Fields -  Include tau_p for the pressure gauge
p = dist.Field(name='p', bases=(xbasis, ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
tau_p = dist.Field(name='tau_p')  # Tau field for pressure gauge
#tau is a system for imposing b.c.s using polynomail spectral methods
#introduces degrees of freedom that allow the problem to be solved exactly over polynomials
#accomodates the singular equations that can arise

#no time dependende forcing term
#no boundary conditions for stokes 
#grad p +del u +f=0, gradu=0
#u=-1/c^2 sin(cy) in x direction
#try running navier stokes
#try running with low reynolds number
#keep forcing, test boundary conditions
#periodic in x and walls in y

# Forcing
x = dist.local_grid(xbasis)
y = dist.local_grid(ybasis)
f = dist.VectorField(coords, bases=(xbasis, ybasis), name='f')
f['g'][0] = amp * np.sin(nper * y)
f['g'][1] = 0

print("\n=== Setting up problem ===")

# fully periodic incompressible flow:
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) - nu*lap(u) + grad(p) = f")
problem.add_equation("div(u) + tau_p = 0")  # tau_p added to divergence
problem.add_equation("integ(p) = 0")  # Pressure gauge

#stokes on periodic boundary with fourier space
#code navier-stokes with high viscosity
#no time dependence 

# Build solver
print("Building solver...")
solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = runtime

# Initial conditions
u['g'][0] = 0 #starting near the forcing condition
u['g'][1] = 0.0 #initial y condition
p['g'] = 0.0 #initial pressure condition

print("Starting time evolution...")

# Time stepping
try:
    iteration_count = 0
    while solver.proceed:
        solver.step(timestep)
        iteration_count += 1
        
        if iteration_count % 100 == 0:
            u.change_scales(1) #represent u on the physical grid with no padding
            u_mag = np.sqrt(u['g'][0]**2 + u['g'][1]**2) #computes velocity magnitude
            logger.info(f"Iteration {iteration_count}, Time {solver.sim_time:.4f}, Max |u|: {np.max(u_mag):.4f}")
            
except Exception as e:
    logger.error('Exception raised, triggering end of main loop.') #stops for CFL instability, NaNs, linear solver failure, etc.
    raise
finally:
    solver.log_stats() #prints stats

print("\n" + "="*60)
print("Simulation complete!")
print(f"Final time: {solver.sim_time}")
print(f"Iterations: {iteration_count}")
print("="*60)

# Final visualization
u.change_scales(1)
p.change_scales(1)

x_plot = xbasis.global_grid(dist, scale=1)
y_plot = ybasis.global_grid(dist, scale=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Velocity x-component
im1 = axes[0].pcolormesh(x_plot.ravel(), y_plot.ravel(), u['g'][0].T, 
                          cmap='RdBu_r', shading='auto')
axes[0].set_title(f'u_x velocity (t={solver.sim_time:.2f})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal')
plt.colorbar(im1, ax=axes[0])

# Velocity y-component
im2 = axes[1].pcolormesh(x_plot.ravel(), y_plot.ravel(), u['g'][1].T, 
                          cmap='RdBu_r', shading='auto')
axes[1].set_title(f'u_y velocity (t={solver.sim_time:.2f})')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_aspect('equal')
plt.colorbar(im2, ax=axes[1])

# Pressure
im3 = axes[2].pcolormesh(x_plot.ravel(), y_plot.ravel(), p['g'].T, 
                          cmap='RdBu_r', shading='auto')
axes[2].set_title(f'Pressure (t={solver.sim_time:.2f})')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_aspect('equal')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('kolmogorov_flow.png', dpi=150, bbox_inches='tight')
print("\nSaved figure to kolmogorov_flow.png")

# Check solution quality
u_mag = np.sqrt(u['g'][0]**2 + u['g'][1]**2)
print(f"\nSolution statistics:")
print(f"  Max velocity magnitude: {np.max(u_mag):.4f}")
print(f"  Mean velocity magnitude: {np.mean(u_mag):.4f}")
print(f"  Max pressure: {np.max(p['g']):.4f}")
print(f"  Min pressure: {np.min(p['g']):.4f}")
print(f"  Mean pressure: {np.mean(p['g']):.6e} (should be ~0)")

plt.show()