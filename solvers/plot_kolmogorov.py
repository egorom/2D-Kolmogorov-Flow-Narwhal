"""
plot_kolmogorov.py
------------------
Post-processing script for kolmogorov_flow.py snapshots.

Usage (single process, after MPI run):
    python plot_kolmogorov.py

Requires:
    dedalus, h5py, numpy, matplotlib

Steps performed:
  1. Merge parallel HDF5 snapshot files using the Dedalus CLI.
  2. Plot every saved time-frame as a 6-panel overview figure.
  3. Save a dedicated cxy "Narwhal" figure for each frame.
  4. Save a time-series summary plot.
  5. Optionally stitch frames into animations (requires ffmpeg).
"""

import os
import sys
import subprocess
import shutil
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SNAP_DIR    = Path("snapshots")        # written by the simulation
MERGED_DIR  = Path("snapshots_merged") # merged output lands here
PLOT_DIR    = Path("plots")
NARWHAL_DIR = Path("plots_narwhal")

for d in [MERGED_DIR, PLOT_DIR, NARWHAL_DIR]:
    d.mkdir(exist_ok=True)


# ── 1. Merge parallel files ───────────────────────────────────────────────────
def merge_snapshots():
    """
    Dedalus 3 writes one sub-directory per output set:
        snapshots/snapshots_s1/   snapshots/snapshots_s2/  …
    each containing one HDF5 file per MPI rank.

    We merge using the Dedalus CLI:
        python -m dedalus merge_procs <set_dir>
    which is stable across all recent Dedalus 3 versions and avoids
    the changing Python post API (merge_sets / merge_setup etc.).
    """
    sets = sorted([p for p in SNAP_DIR.glob("snapshots_s*") if p.is_dir()])

    if not sets:
        # Single-process run: flat .h5 files live directly in SNAP_DIR
        h5files = sorted(SNAP_DIR.glob("*.h5"))
        if h5files:
            print(f"Found {len(h5files)} flat .h5 file(s) – no merge needed.")
            return h5files
        print(f"No snapshot data found in '{SNAP_DIR}/'. Run the simulation first.")
        return None

    print(f"Merging {len(sets)} snapshot set(s) via 'python -m dedalus merge_procs' …")
    merged_files = []

    for s in sets:
        out = MERGED_DIR / (s.name + ".h5")

        if out.exists():
            print(f"  {out.name} already merged – skipping.")
            merged_files.append(out)
            continue

        result = subprocess.run(
            [sys.executable, "-m", "dedalus", "merge_procs", str(s)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"merge_procs failed for {s}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # The merged file is written inside the set dir with the set's name
        written = sorted(s.glob("*.h5"))
        if not written:
            raise FileNotFoundError(
                f"merge_procs ran but produced no .h5 inside {s}"
            )

        shutil.copy(written[0], out)
        print(f"  {s.name}  →  {out}")
        merged_files.append(out)

    print(f"All sets merged into {MERGED_DIR}/")
    return merged_files


# ── 2. Load all frames ────────────────────────────────────────────────────────
def load_all_frames(h5files):
    """
    Returns:
        arrays : dict  field -> np.ndarray  shape (n_frames, Nx, Ny)
        times  : np.ndarray  shape (n_frames,)
    """
    fields = ["ux", "uy", "p", "cxx", "cxy", "cyy"]
    arrays = {f: [] for f in fields}
    times  = []

    for hf in sorted(h5files):
        with h5py.File(hf, "r") as f:
            t = f["scales"]["sim_time"][:]
            for i in range(len(t)):
                times.append(t[i])
                for field in fields:
                    arr = np.squeeze(f["tasks"][field][i])
                    arrays[field].append(arr)

    times = np.array(times)
    for field in fields:
        arrays[field] = np.array(arrays[field])   # (n_frames, Nx, Ny)

    # Sort by simulation time across files
    order = np.argsort(times)
    times = times[order]
    for field in fields:
        arrays[field] = arrays[field][order]

    print(f"Loaded {len(times)} frames  |  fields: {fields}")
    return arrays, times


# ── 3. Six-panel overview ─────────────────────────────────────────────────────
def plot_overview(arrays, times, frame_idx, save_dir):
    t      = times[frame_idx]
    Nx, Ny = arrays["ux"].shape[1:]
    x      = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y      = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y   = np.meshgrid(x, y, indexing="ij")

    panels = [
        ("ux",  r"$u_x$",       "RdBu_r",  True),
        ("uy",  r"$u_y$",       "RdBu_r",  True),
        ("p",   r"$p$",         "RdBu_r",  True),
        ("cxx", r"$C_{xx}$",    "inferno", False),
        ("cxy", r"$C_{xy}$",    "RdBu_r",  True),
        ("cyy", r"$C_{yy}$",    "inferno", False),
    ]

    fig = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
    fig.suptitle(
        f"Kolmogorov viscoelastic flow  —  t = {t:.3f}",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    for idx, (key, label, cmap, diverge) in enumerate(panels):
        ax  = fig.add_subplot(gs[idx // 3, idx % 3])
        dat = arrays[key][frame_idx]

        if diverge:
            vmax = max(float(np.abs(dat).max()), 1e-12)
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        else:
            norm = None

        im = ax.pcolormesh(X, Y, dat, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(label, color="white", fontsize=11)
        ax.set_xlabel("x", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("y", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_facecolor("#111111")

    fname = save_dir / f"overview_{frame_idx:04d}.png"
    fig.savefig(fname, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


# ── 4. Narwhal figure ─────────────────────────────────────────────────────────
def plot_narwhal(arrays, times, frame_idx, save_dir):
    """
    Dedicated cxy plot with velocity direction arrows.
    The 'Narwhal' is the sharp, asymmetric stress horn visible in cxy
    when the viscoelastic instability is active.
    """
    t      = times[frame_idx]
    Nx, Ny = arrays["cxy"].shape[1:]
    x      = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y      = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y   = np.meshgrid(x, y, indexing="ij")

    cxy = arrays["cxy"][frame_idx]
    ux  = arrays["ux"][frame_idx]
    uy  = arrays["uy"][frame_idx]

    vmax = max(float(np.abs(cxy).max()), 1e-12)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#06080f")
    ax.set_facecolor("#06080f")

    im = ax.pcolormesh(X, Y, cxy, cmap="seismic", norm=norm,
                       shading="auto", rasterized=True, zorder=1)

    # Velocity direction arrows (downsampled for clarity)
    stride = max(1, Nx // 32)
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    UX = ux[::stride, ::stride]
    UY = uy[::stride, ::stride]
    spd = np.sqrt(UX**2 + UY**2) + 1e-12
    ax.quiver(Xs, Ys, UX/spd, UY/spd,
              color="white", alpha=0.35, scale=40, width=0.003,
              headwidth=3, headlength=4, zorder=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$C_{xy}$", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        r"$C_{xy}$ — Narwhal structure" + f"\n"
        f"t = {t:.3f}   max|$C_{{xy}}$| = {float(np.abs(cxy).max()):.4f}",
        color="white", fontsize=12, pad=10
    )
    ax.set_xlabel("x", color="#cccccc", fontsize=10)
    ax.set_ylabel("y", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)

    fig.tight_layout()
    fname = save_dir / f"narwhal_{frame_idx:04d}.png"
    fig.savefig(fname, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


# ── 5. Time-series summary ────────────────────────────────────────────────────
def plot_timeseries(arrays, times, save_dir):
    max_cxy  = np.array([float(np.abs(arrays["cxy"][i]).max())  for i in range(len(times))])
    mean_trC = np.array([(arrays["cxx"][i] + arrays["cyy"][i]).mean() for i in range(len(times))])
    max_ux   = np.array([float(np.abs(arrays["ux"][i]).max())   for i in range(len(times))])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor="#0d0d0d", sharex=True)
    fig.suptitle("Kolmogorov viscoelastic flow — diagnostics",
                 color="white", fontsize=13, fontweight="bold")

    specs = [
        (axes[0], max_cxy,  r"max $|C_{xy}|$",  "#e05c5c"),
        (axes[1], mean_trC, r"mean tr($C$)",     "#5ca8e0"),
        (axes[2], max_ux,   r"max $|u_x|$",      "#5ce07a"),
    ]
    for ax, data, label, color in specs:
        ax.plot(times, data, color=color, lw=1.8)
        ax.set_ylabel(label, color=color, fontsize=10)
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#888888")
        ax.grid(color="#2a2a2a", lw=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    axes[-1].set_xlabel("Simulation time", color="#aaaaaa", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = save_dir / "timeseries.png"
    fig.savefig(fname, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved time-series → {fname}")
    return fname


# ── 6. Optional ffmpeg animation ──────────────────────────────────────────────
def make_animation(frame_dir, pattern, output_name, fps=8):
    cmd = (
        f"ffmpeg -y -framerate {fps} "
        f"-i {frame_dir}/{pattern} "
        f"-vcodec libx264 -pix_fmt yuv420p "
        f"-crf 20 {output_name} 2>/dev/null"
    )
    ret = os.system(cmd)
    if ret == 0:
        print(f"Animation saved → {output_name}")
    else:
        print("ffmpeg not available or failed – skipping animation.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Merge parallel files
    h5files = merge_snapshots()
    if h5files is None:
        raise SystemExit("No snapshot files to process.")

    # 2. Load all frames
    arrays, times = load_all_frames(h5files)
    n_frames = len(times)
    print(f"Total frames: {n_frames}")

    # 3. Time-series (fast, do first)
    plot_timeseries(arrays, times, PLOT_DIR)

    # 4. Per-frame plots
    print("Rendering frames …")
    for i in range(n_frames):
        plot_overview(arrays, times, i, PLOT_DIR)
        plot_narwhal(arrays, times, i, NARWHAL_DIR)
        if (i + 1) % 5 == 0 or i == n_frames - 1:
            print(f"  frame {i+1}/{n_frames}  t={times[i]:.3f}")

    print(f"\nOverview frames → {PLOT_DIR}/")
    print(f"Narwhal frames  → {NARWHAL_DIR}/")

    