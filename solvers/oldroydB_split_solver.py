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
Wi=16*lam
beta=1.0

Nx = 512
Ny = int(Nx /aspect)
Lx = 4
Ly = Lx / aspect
dt = 0.01 / (2 ** (np.log2(Nx) - 6))


amp = maxvel* (nper ** 2) * (1+ xi / (1 + lam * nu * (nper ** 2)))

if branch in [3, 4, 5]:
    runtime = 20
elif branch == 6:
    runtime = 40

#Bases
#dealias for nonlinear terms
xbasis=d3.RealFourier(coords['x'], size=Nx, bounds=(0,Lx), dealias=3/2)
#ybasis=d3.Chebyshev(coords['y'], size=Ny, bounds=(0,Ly), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=3/2)

k_max = np.max(np.sqrt(xbasis.wavenumbers**2 + ybasis.wavenumbers**2))
k_cut = 0.66 * k_max


#Fields
u=dist.VectorField(coords, bases=(xbasis, ybasis))
p=dist.Field(bases=(xbasis, ybasis))

C_xx = dist.Field(bases=(xbasis, ybasis))
C_xy = dist.Field(bases=(xbasis, ybasis))
C_yy = dist.Field(bases=(xbasis, ybasis))

x=dist.local_grid(xbasis)
y=dist.local_grid(ybasis)

f=dist.VectorField(coords, bases=(xbasis, ybasis))
f['g'][0]=amp * np.sin(nper * y) #x forcing
f['g'][1]=0


#make the boundaries walls instead of periodic
#add fene-p later 

#problem=d3.IVP([u,p], namespace=locals())
lap=d3.Laplacian
grad=d3.Gradient
div=d3.Divergence


tau_p=dist.Field()
tau_u=dist.VectorField(coords)

#Define problem
conf_problem=d3.IVP([C_xx, C_xy, C_yy], namespace=locals())

conf_problem.add_equation(
    "dt(C_xx) + u@grad(C_xx)"
    " - 2*(dx(u@coords['x']))*C_xx"
    " - 2*(dy(u@coords['x']))*C_xy"
    " = -(C_xx - 1)/Wi"
)

conf_problem.add_equation(
    "dt(C_xy) + u@grad(C_xy)"
    " - (dx(u@coords['x']))*C_xy"
    " - (dy(u@coords['y']))*C_xy"
    " - (dy(u@coords['x']))*C_xx"
    " - (dx(u@coords['y']))*C_yy"
    " = -C_xy/Wi"
)

conf_problem.add_equation(
    "dt(C_yy) + u@grad(C_yy)"
    " - 2*(dy(u@coords['y']))*C_yy"
    " - 2*(dx(u@coords['y']))*C_xy"
    " = -(C_yy - 1)/Wi"
)

conf_solver = conf_problem.build_solver(d3.RK443)

#Stokes solver
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

f_poly = d3.Vector(dx(C_xx) + dy(C_xy),
                   dx(C_xy) + dy(C_yy))

stokes = d3.LBVP([u, p], namespace=locals())

stokes.add_equation("-nu*lap(u) + grad(p) = f + f_poly") #momentum equation

stokes.add_equation("div(u) = 0")
stokes.add_equation("integ(p) = 0")
#stokes.add_equation("integ(u) = 0")
stokes_solver = stokes.build_solver()

# -----------------------------------------
# Initial Conditions
# -----------------------------------------
u['g'][0] = maxvel*np.sin(nper*y)
u['g'][1] = 0

# Analytic conformation 
C11amp = 2*maxvel**2 * nper**2 / (1/lam**2 + (5*nu*nper**2)/lam + 4*nu**2*nper**4) 
C11const = 2*lam*nu*C11amp*nper**2 + 1 
C_xx['g'] = C11amp * (np.cos(nper * y))**2 + C11const 
C_xy['g'] = (maxvel * nper * lam) / (1 + lam*nu*nper**2) * np.cos(nper * y) 
C_yy['g'] = 1.0

# -----------------------------------------
# Diagnostics & Saving Setup
# -----------------------------------------

# How often to save (in time units)
saves_norm_per_unit_time = 20
saves_per_unit_time = 20

savetr = int(1 / (saves_norm_per_unit_time * dt))
saveit = int(1 / (saves_per_unit_time * dt))

# Storage arrays (similar to MATLAB "norms")
norms = []

# Output directory
import os
outdir = f"Wi_{Wi:.3f}"
os.makedirs(outdir, exist_ok=True)

# Helper: compute norms (analogous to MATLAB get_norms)
def compute_norms(u, C_xx, C_xy, C_yy):
    u.change_scales(1)
    C_xx.change_scales(1)
    C_xy.change_scales(1)
    C_yy.change_scales(1)

    ux = u['g'][0]
    uy = u['g'][1]

    KE = 0.5 * np.sum(ux**2 + uy**2) * (Lx/Nx) * (Ly/Ny)
    TrC = C_xx['g'] + C_yy['g']
    TrC_int = np.sum(TrC) * (Lx/Nx) * (Ly/Ny)

    return KE, TrC_int

# Helper: save snapshot
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

    min_Tr = np.min(Tr)
    min_Det = np.min(Det)

    return min_Tr, min_Det




# Time loop
n = 0
t = 0.0

while t < runtime:
    stokes_solver.solve()
    conf_solver.step(dt)

    apply_filter(C_xx, k_cut)
    apply_filter(C_xy, k_cut)
    apply_filter(C_yy, k_cut)
    apply_filter(u,   k_cut)

    t += dt
    n += 1

    if n % savetr == 0:
        min_Tr, min_Det = check_spd(C_xx, C_xy, C_yy)
        if min_Tr <= 0 or min_Det <= 0:
            print(f"SPD violation at t={t:.3f}: min Tr={min_Tr:.3e}, min Det={min_Det:.3e}")

        KE, TrC = compute_norms(u, C_xx, C_xy, C_yy)
        norms.append([t, KE, TrC])
        print(f"t={t:.3f}, KE={KE:.6e}, TrC={TrC:.6e}")

    if n % saveit == 0:
        save_snapshot(n, t, u, C_xx, C_xy, C_yy)



