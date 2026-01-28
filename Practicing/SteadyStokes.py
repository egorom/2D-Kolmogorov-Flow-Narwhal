import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

#matplotlib inline
#config InlineBackend.figure_format='retina'

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

#Plot to see if anything is working
#x=xbasis.global_grid(dist, scale=1)
#y=ybasis.global_grid(dist, scale=1)
#X,Y=np.meshgrid(x,y, indexing='ij')
#plt.figure(figsize=(4,4))
#plt.scatter(X.flatten(), Y.flatten(), s=2)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Physical grid (Fourier)')
#plt.gca().set_aspect('equal')
#plt.show()

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
problem = d3.LBVP([u, p, tau_p, tau_u], namespace=locals())
#it interprets the following as an operator equation, discretizes spectrally in x and y and enforces incompressibility as a constraint
problem.add_equation("- nu*lap(u) + grad(p) + tau_u = f") #time-dependent stokes
problem.add_equation("div(u)+tau_p=0")
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(u) = 0")


#Steady Stokes instead
#problem=d3.LBVP([u,p], namespace=locals())
#problem.add_equation("-nu*lap(u)+grad(p)=f")
#problem.add_equation("integ(p)=0")

solver=problem.build_solver() #why this RK?
solver.solve()

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