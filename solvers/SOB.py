import numpy as np
import dedalus.public as d3
from mpi4py import MPI

# -------------------------
# Parameters
# -------------------------
Lx = 2*np.pi
Ly = 2*np.pi
Nx = Ny = 512
beta = 1.0 
dealias = 3/2


# Linear conformation stepping-stone params
kappa_c = 0     # diffusion on C components
tauR    = .3       # relaxation time (C -> I)
dt_step = .00125 #5e-4   # small enough for Nx=1024 i used: grid.dt = (.01/(2^(log2(grid.Nx)-6)));  
t_end   = max(2,10*tauR)

# Coupling strength for polymer stress in Stokes forcing:
alpha_p = .5/tauR   # 

comm = MPI.COMM_WORLD

dx_phys = Lx / Nx
dy_phys = Ly / Ny

def global_int_array(a):
    # integral over domain of array a (assumes a is at scale=1 grid)
    return comm.allreduce(np.sum(a), op=MPI.SUM) * dx_phys * dy_phys

# -------------------------
# Domain
# -------------------------
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

dx = lambda F: d3.Differentiate(F, coords['x'])
dy = lambda F: d3.Differentiate(F, coords['y'])
grad = d3.grad
div  = d3.div

def cfl_number(u):
    u.change_scales(1)
    umax = comm.allreduce(np.max(np.abs(u['g'][0])), op=MPI.MAX)
    vmax = comm.allreduce(np.max(np.abs(u['g'][1])), op=MPI.MAX)
    return dt_step * max(umax/dx_phys, vmax/dy_phys)

# -------------------------
# Unknowns for Stokes
# -------------------------
u  = dist.VectorField(coords, name='u', bases=(xb, yb))
p  = dist.Field(name='p', bases=(xb, yb))

ux0 = dist.Field(name='ux0')  # gauges (constants)
uy0 = dist.Field(name='uy0')
p0  = dist.Field(name='p0')

# Velocity components (scalar fields)
ux = u@ex
uy = u@ey

# Velocity gradient entries (symbolic operators; always reflect current u)
ux_x = dx(ux)
ux_y = dy(ux)
uy_x = dx(uy)
uy_y = dy(uy)


# -------------------------
# Forcing: 4-roll-mill
# -------------------------
# Base forcing: 4-roll-mill (fixed in time)
f0 = dist.VectorField(coords, name='f0', bases=(xb, yb))
f0['g'][0] =  -2*np.sin(x)*np.cos(y)
f0['g'][1] =   2*np.cos(x)*np.sin(y)

# Total forcing used by Stokes each step:
# f_total = f0 + div(tau_p)
f_total = dist.VectorField(coords, name='f_total', bases=(xb, yb))
f_total['g'][0] = f0['g'][0]
f_total['g'][1] = f0['g'][1]



# -------------------------
# Conformation tensor components (2D symmetric)
# -------------------------
cxx = dist.Field(name='cxx', bases=(xb, yb))
cxy = dist.Field(name='cxy', bases=(xb, yb))
cyy = dist.Field(name='cyy', bases=(xb, yb))

# Initial condition: identity + blob
blob = np.exp(-((x-np.pi)**2 + (y-np.pi/2)**2)/(0.3**2))
cxx['g'] = 1.0# + 0.2*blob
cxy['g'] = 0.0
cyy['g'] = 1.0#+ 0.2*blob

# -------------------------
# Stokes LBVP
# -------------------------
stokes = d3.LBVP([u, p, ux0, uy0, p0], namespace=locals())
stokes.add_equation("beta*div(grad(u)) - grad(p) + ux0*ex + uy0*ey = -f_total")
stokes.add_equation("div(u) + p0 = 0")
stokes.add_equation("integ(p) = 0")
stokes.add_equation("integ(u@ex) = 0")
stokes.add_equation("integ(u@ey) = 0")
stokes_solver = stokes.build_solver()
stokes_solver.solve()


# -------------------------
# Conformation IVP: upper-convected Oldroyd-B (with diffusion + linear relaxation)
# dt(C) - kappa ΔC = -(u·∇C) + (∇u)C + C(∇u)^T - (1/tauR)(C-I)
# -------------------------
cprob = d3.IVP([cxx, cxy, cyy], namespace=locals())

cprob.add_equation(
    "dt(cxx) - kappa_c*div(grad(cxx)) = -(u@grad(cxx))"
    " + 2*(ux_x*cxx + ux_y*cxy) - (1/tauR)*(cxx - 1)"
)

cprob.add_equation(
    "dt(cxy) - kappa_c*div(grad(cxy)) = -(u@grad(cxy))"
    " + (ux_x*cxy + ux_y*cyy + cxx*uy_x + cxy*uy_y) - (1/tauR)*cxy"
)

cprob.add_equation(
    "dt(cyy) - kappa_c*div(grad(cyy)) = -(u@grad(cyy))"
    " + 2*(uy_x*cxy + uy_y*cyy) - (1/tauR)*(cyy - 1)"
)

csolver = cprob.build_solver(d3.RK443)
csolver.stop_sim_time = t_end


def update_forcing_from_C():
    """
    Update Stokes forcing:
      f_total = f0 + div(tau_p),
    where tau_p = alpha_p * (C - I).
    With the Stokes equation
      beta Δu - ∇p = -f_total,
    this corresponds to force balance
      beta Δu - ∇p + div(tau_p) + f0 = 0.
    """
    # Ensure consistent grid scale for array access
    f_total.change_scales(1)
    f0.change_scales(1)
    cxx.change_scales(1); cxy.change_scales(1); cyy.change_scales(1)

    # tau_p components (2D symmetric): alpha_p*(C - I)
    txx = (alpha_p*(cxx - 1.0)).evaluate()
    txy = (alpha_p*(cxy)).evaluate()
    tyy = (alpha_p*(cyy - 1.0)).evaluate()
    txx.change_scales(1); txy.change_scales(1); tyy.change_scales(1)

    # div(tau_p) = [d_x txx + d_y txy,  d_x txy + d_y tyy]
    divtau_x = (dx(txx) + dy(txy)).evaluate()
    divtau_y = (dx(txy) + dy(tyy)).evaluate()
    divtau_x.change_scales(1); divtau_y.change_scales(1)

    # f_total = f0 + div(tau_p)
    f_total['g'][0] = f0['g'][0] + divtau_x['g']
    f_total['g'][1] = f0['g'][1] + divtau_y['g']


# -------------------------
# Diagnostics
# -------------------------
def trC_mean():
    cxx.change_scales(1); cyy.change_scales(1)
    tr = cxx['g'] + cyy['g']
    return global_int_array(tr) / (Lx*Ly)

def trC_L2_sq_int():
    cxx.change_scales(1); cyy.change_scales(1)
    tr = cxx['g'] + cyy['g']
    return global_int_array(tr*tr)

def trC_grad_sq_int():
    # ∫ |∇(cxx+cyy)|^2
    trF = (cxx + cyy).evaluate()
    trx = dx(trF).evaluate(); trx.change_scales(1)
    try_ = dy(trF).evaluate(); try_.change_scales(1)
    return global_int_array(trx['g']**2 + try_['g']**2)

# write every 1.0 units of simulation time
snap = csolver.evaluator.add_file_handler("snapshots", sim_dt=.1, max_writes=None)

snap.add_task(u@ex, name="ux")
snap.add_task(u@ey, name="uy")
snap.add_task(p,    name="p")
snap.add_task(cxx,  name="cxx")
snap.add_task(cxy,  name="cxy")
snap.add_task(cyy,  name="cyy")



# -------------------------
# Time loop
# -------------------------
t = 0.0
it = 0
prev_tr2 = None

while t < t_end - 1e-14:
    # --- Coupling: update forcing from current C, then solve Stokes for u,p ---
    update_forcing_from_C()
    stokes_solver.solve()


    csolver.step(dt_step)
    t += dt_step
    it += 1

    if it % 100 == 0:
    # max(trC) and max(|cxy|)
        cxx.change_scales(1); cxy.change_scales(1); cyy.change_scales(1)

        tr = cxx['g'] + cyy['g']
        tr_max  = comm.allreduce(np.max(tr), op=MPI.MAX)

        cxy_abs_max = comm.allreduce(np.max(np.abs(cxy['g'])), op=MPI.MAX)

        if comm.rank == 0:
            print(f"[C] it={it:5d} t={t:.4f}  max(trC)={tr_max:.6e}  max(|cxy|)={cxy_abs_max:.6e}")