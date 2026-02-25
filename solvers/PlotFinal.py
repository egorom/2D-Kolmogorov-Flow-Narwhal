import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

outdir = f"Wi_27.120"
snapshot_files = sorted([f for f in os.listdir(outdir) if f.startswith('snapshot_')])

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
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].pcolormesh(x_grid, y_grid.T, u_final[1].T, cmap='RdBu_r', shading='auto')
axes[0, 1].set_title('u_y')
plt.colorbar(im1, ax=axes[0, 1])

TrC_final = C_xx_final + C_yy_final
im2 = axes[0, 2].pcolormesh(x_grid, y_grid.T, TrC_final.T, cmap='viridis', shading='auto')
axes[0, 2].set_title('Tr(C)')
plt.colorbar(im2, ax=axes[0, 2])

im3 = axes[1, 0].pcolormesh(x_grid, y_grid.T, C_xx_final.T, cmap='viridis', shading='auto')
axes[1, 0].set_title('C_xx')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].pcolormesh(x_grid, y_grid.T, C_xy_final.T, cmap='RdBu_r', shading='auto')
axes[1, 1].set_title('C_xy')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].pcolormesh(x_grid, y_grid.T, C_yy_final.T, cmap='viridis', shading='auto')
axes[1, 2].set_title('C_yy')
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.flat:
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(f"{outdir}/final_snapshot.png", dpi=150)
plt.show()