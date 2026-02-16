import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

#provide the names to use for each coordinate
coords=d3.CartesianCoordinates('x','y')

#Distributor objects direct the parallel decomposition of fields and problems and are needed for all problems
dist=d3.Distributor(coords,dtype=np.float64) 

#Parameters
Lamin = 1.695     # lambda (polymer relaxation time)
branch = 4

nper=4
aspect=4
nu= 0.0005
lam= Lamin
maxvel= 4.0
xi = 0.5

Nx = 512
Ny = int(Nx /aspect)
Lx = 2*np.pi
Ly = Lx / aspect
timestep = 0.01 / (2 ** (np.log2(Nx) - 6))

amp = maxvel* (nper ** 2) * (1+ xi / (1 + lam * nu * (nper ** 2)))

if branch in [3, 4, 5]:
    runtime = 20
elif branch == 6:
    runtime = 40

#Bases
#dealias for nonlinear terms
xbasis=d3.RealFourier(coords['x'], size=Nx, bounds=(0,Lx), dealias=3/2)
ybasis=d3.RealFourier(coords['y'], size=Ny, bounds=(0,Ly), dealias=3/2)

#Fields
u=dist.VectorField(coords, bases=(xbasis, ybasis))
p=dist.Field(bases=(xbasis, ybasis))

x=dist.local_grid(xbasis)
y=dist.local_grid(ybasis)

f=dist.VectorField(coords, bases=(xbasis, ybasis))
f['g'][0]=amp * np.sin(nper * y) #x forcing
f['g'][1]=0

#problem=d3.IVP([u,p], namespace=locals())
lap=d3.Laplacian
grad=d3.Gradient
div=d3.Divergence


tau_p=dist.Field()
tau_u=dist.VectorField(coords)
problem = d3.IVP([u, p, tau_p, tau_u], namespace=locals())

problem.add_equation("dt(u)+grad(p)- nu*lap(u) + tau_u = f-u@grad(u)") #time-dependent stokes
problem.add_equation("div(u)+tau_p=0")
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(u) = 0")



solver=problem.build_solver(d3.RK443) #why this RK?
u['g'][0] = 0
u['g'][1] = 0


solver.stop_sim_time=runtime
solver.stop_wall_time=np.inf
solver.stop_iteration=np.inf

while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 50==0:
        div_u=div(u).evaluate()['g']
        print(f"t={solver.sim_time:.3f}, max div(u)={np.max(np.abs(div_u)):.2e}")

y1d=y[0,:]
u_slice=u['g'][0,0,:]
plt.figure()
plt.plot(y1d, u_slice, label='numerical')
plt.plot(y1d, (amp/(nu*nper**2))*np.sin(nper*y1d), '--', label='analytical')
plt.legend()
plt.xlabel('y')
plt.ylabel('u_x')
plt.show()

div_u=div(u).evaluate()['g']
print("max div(u)=", np.max(np.abs(div_u)))