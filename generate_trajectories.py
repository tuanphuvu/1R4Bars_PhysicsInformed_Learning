# generate_trajectories.py

# Generates 8 sinusoidal reference trajectories for end-effector simulation:
#   2 amplitudes  × 4 wave counts = 8 trajectories
#
#   Amplitudes : 2.0 cm,  2.5 cm
#   Wave counts: 4,  5,  6,  7
#
# Method
# ------
#   - Alpha shape (concave hull) for true workspace boundary
#   - Generalize line = radial midpoint between inner/outer workspace boundary
#     at each angular direction from the workspace centre
#   - Sinusoidal oscillation in the radial direction along the generalize arc
#   - Points outside the workspace are discarded
#
# Output
# ------
#   trajectories_ref/traj_amp<A>cm_<N>waves.csv   (8 files)
#   figures/reference_trajectories.png

import csv
import os
import sys
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dynamics.params   import Q1_MIN, Q1_MAX, Q2_MIN, Q2_MAX
from dynamics.matrices import fk_G4

os.makedirs("trajectories_ref", exist_ok=True)
os.makedirs("figures",          exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
AMPLITUDES    = [0.020, 0.025]   # metres  →  2.0 cm, 2.5 cm
WAVE_COUNTS   = [4, 5, 6, 7]
N_PTS         = 600              # sample points along the generalize spline
N_GRID        = 200              # joint-space sweep resolution (per axis)
ALPHA_SHAPE   = 25               # alpha shape tightness — tune between 15–40
N_ANGLE_BINS  = 150              # angular bins for radial-midpoint computation
MIN_THICKNESS = 0.005            # discard angular bins thinner than this (m)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — JOINT-SPACE SWEEP  →  workspace point cloud
# ══════════════════════════════════════════════════════════════════════════
print("[1/4] Sweeping joint space...")
q1_lin = np.linspace(Q1_MIN, Q1_MAX, N_GRID)
q2_lin = np.linspace(Q2_MIN, Q2_MAX, N_GRID)
ws_pts = np.array([fk_G4(np.array([q1, q2]))
                   for q1 in q1_lin
                   for q2 in q2_lin])
print(f"      {len(ws_pts):,} workspace points")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — ALPHA SHAPE  →  concave (crescent-accurate) workspace boundary
# ══════════════════════════════════════════════════════════════════════════
def _alpha_shape_polygons(pts, alpha):
    """
    Compute alpha shape of pts using Delaunay triangulation.
    Triangles with circumradius > 1/alpha are removed, exposing the
    true concave boundary instead of the convex hull.

    Returns a list of vertex arrays (ordered polygons).
    """
    tri = Delaunay(pts)
    ia, ib, ic = tri.simplices[:, 0], tri.simplices[:, 1], tri.simplices[:, 2]

    a = np.linalg.norm(pts[ia] - pts[ib], axis=1)
    b = np.linalg.norm(pts[ib] - pts[ic], axis=1)
    c = np.linalg.norm(pts[ic] - pts[ia], axis=1)
    s    = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s-a) * (s-b) * (s-c), 1e-30))
    cr   = (a * b * c) / (4.0 * area)             # circumradius

    kept = tri.simplices[cr < 1.0 / alpha]         # keep small-circumradius triangles
    if len(kept) == 0:
        return []

    # Boundary edges = edges shared by exactly one triangle
    all_edges = np.concatenate(
        [kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    cnt       = Counter(map(tuple, all_edges))
    bnd_edges = [(u, v) for (u, v), n in cnt.items() if n == 1]

    # Walk the boundary graph → ordered polygon(s)
    adj = {}
    for u, v in bnd_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited, polygons = set(), []
    for start in adj:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        prev, curr = None, start
        while True:
            nxt = next(
                (nb for nb in adj[curr] if nb != prev and nb not in visited),
                None)
            if nxt is None:
                break
            chain.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt
        if len(chain) >= 3:
            polygons.append(pts[chain])
    return polygons


def _poly_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


print(f"[2/4] Building alpha shape boundary (alpha={ALPHA_SHAPE})...")
polys = _alpha_shape_polygons(ws_pts[::3], ALPHA_SHAPE)
if not polys:
    raise RuntimeError(
        f"Alpha shape is empty — try lowering ALPHA_SHAPE (currently {ALPHA_SHAPE}).")

main_poly       = max(polys, key=_poly_area)
boundary_closed = np.vstack([main_poly, main_poly[0]])   # closed for plotting
ws_path         = MplPath(boundary_closed)               # for point-in-poly tests
ws_center       = ws_pts.mean(axis=0)
cx, cy          = ws_center

print(f"      {len(main_poly)} boundary vertices")
print(f"      area  = {_poly_area(main_poly) * 1e4:.1f} cm²")
print(f"      centre = ({cx:.4f}, {cy:.4f}) m")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — GENERALIZE LINE  (radial midpoint approach)
# ══════════════════════════════════════════════════════════════════════════
# At each angular direction θ from the workspace centre:
#   r_inner = nearest workspace point along that ray
#   r_outer = farthest workspace point along that ray
#   r_mid   = (r_inner + r_outer) / 2   ← centre of workspace thickness
# This is independent of point density, so it correctly represents the
# geometric centre of the crescent at each angle.
print("[3/4] Computing generalize line (radial midpoint)...")

dx_ws = ws_pts[:, 0] - cx
dy_ws = ws_pts[:, 1] - cy
r_ws  = np.sqrt(dx_ws**2 + dy_ws**2)
th_ws = np.arctan2(dy_ws, dx_ws)

theta_edges = np.linspace(th_ws.min() - 1e-9, th_ws.max() + 1e-9, N_ANGLE_BINS + 1)
theta_idx   = np.digitize(th_ws, theta_edges)

bin_theta, bin_r_mid = [], []
for b in range(1, N_ANGLE_BINS + 1):
    mask = theta_idx == b
    if mask.sum() < 3:
        continue
    r_lo, r_hi = r_ws[mask].min(), r_ws[mask].max()
    if (r_hi - r_lo) < MIN_THICKNESS:    # discard edge bins too thin to be reliable
        continue
    bin_theta.append(th_ws[mask].mean())
    bin_r_mid.append((r_lo + r_hi) / 2.0)

bin_theta = np.array(bin_theta)
bin_r_mid = np.array(bin_r_mid)
sort_idx  = np.argsort(bin_theta)
bin_theta = bin_theta[sort_idx]
bin_r_mid = bin_r_mid[sort_idx]

samp_x = cx + bin_r_mid * np.cos(bin_theta)
samp_y = cy + bin_r_mid * np.sin(bin_theta)

# Parametric spline through the midpoint samples
tck, _ = splprep([samp_x, samp_y], s=5e-5, k=3)
u_fine  = np.linspace(0, 1, N_PTS)
gx, gy  = splev(u_fine, tck)

# Polar form of the generalize line (needed for radial sinusoid)
gr  = np.sqrt((gx - cx)**2 + (gy - cy)**2)
gth = np.arctan2(gy - cy, gx - cx)

# Cumulative arc length → normalized phase parameter φ ∈ [0, 1]
ds   = np.sqrt(np.diff(gx)**2 + np.diff(gy)**2)
s_g  = np.concatenate([[0], np.cumsum(ds)])
phi  = s_g / s_g[-1]   # wave phase: n_waves complete oscillations over arc

print(f"      {len(gx)} generalize points | total arc = {s_g[-1]*100:.1f} cm")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE, SAVE, AND PLOT ALL 8 TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════
print("[4/4] Generating and saving trajectories...\n")

# Visual style:  colour = wave count,  line style = amplitude
WAVE_COLOR = {4: '#1f77b4', 5: '#2ca02c', 6: '#d62728', 7: '#9467bd', 9: '#ff7f0e'}
AMP_LINEST  = {0.020: ('-',  2.0), 0.025: ('--', 2.0)}
AMP_LABEL   = {0.020: '2.0', 0.025: '2.5'}

fig, ax = plt.subplots(figsize=(11, 9))

# ── Background: point cloud + boundary ───────────────────────────────────
ax.scatter(ws_pts[::10, 0], ws_pts[::10, 1],
           c='lightgray', s=1.5, alpha=0.22, zorder=1, label='_nolegend_')
ax.fill(boundary_closed[:, 0], boundary_closed[:, 1],
        alpha=0.07, color='steelblue', zorder=0)
ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
        color='k', lw=1.8, zorder=3, label='Workspace boundary')
ax.plot(gx, gy, color='dimgray', lw=1.2, ls=':', zorder=4,
        label='Generalize line')
ax.plot(cx, cy, 'k*', ms=10, zorder=10, label='WS centre')

# ── Loop: generate → save → plot ─────────────────────────────────────────
summary = []

for amp in AMPLITUDES:
    for n_waves in WAVE_COUNTS:

        # 1. Sinusoidal oscillation in the radial direction
        r_sin = gr + amp * np.sin(2.0 * np.pi * n_waves * phi)
        x_sin = cx + r_sin * np.cos(gth)
        y_sin = cy + r_sin * np.sin(gth)

        # 2. Discard points outside the workspace
        inside = ws_path.contains_points(np.column_stack((x_sin, y_sin)))
        x_t = x_sin[inside]
        y_t = y_sin[inside]
        t_t = u_fine[inside]        # spline parameter at kept points

        # 3. Arc length along the trajectory
        diffs   = np.sqrt(np.diff(x_t)**2 + np.diff(y_t)**2)
        arc_t   = np.concatenate([[0.0], np.cumsum(diffs)])

        # 4. Save CSV
        amp_tag = f"{amp * 100:.1f}".replace('.', 'p')          # "2p0" / "2p5"
        fname   = f"traj_amp{amp_tag}cm_{n_waves}waves.csv"
        fpath   = os.path.join("trajectories_ref", fname)

        with open(fpath, 'w', newline='') as f:
            f.write("# Sinusoidal end-effector reference trajectory\n")
            f.write(f"# amplitude_m    : {amp:.4f}\n")
            f.write(f"# amplitude_cm   : {amp * 100:.1f}\n")
            f.write(f"# n_waves        : {n_waves}\n")
            f.write(f"# ws_centre_m    : ({cx:.5f}, {cy:.5f})\n")
            f.write(f"# n_points       : {len(x_t)}\n")
            f.write(f"# arc_length_m   : {arc_t[-1]:.5f}\n")
            f.write(f"# arc_length_cm  : {arc_t[-1] * 100:.3f}\n")
            writer = csv.writer(f)
            writer.writerow(["t", "x_m", "y_m", "arc_length_m"])
            for row in zip(t_t, x_t, y_t, arc_t):
                writer.writerow([f"{v:.7f}" for v in row])

        summary.append((fname, len(x_t), arc_t[-1] * 100))

        # 5. Plot
        color    = WAVE_COLOR[n_waves]
        ls, lw   = AMP_LINEST[amp]
        ax.plot(x_t, y_t, color=color, ls=ls, lw=lw,
                alpha=0.88, zorder=5 + n_waves,
                label=f"A={amp*100:.1f}cm  n={n_waves}")

        print(f"  ✓  {fname:<38}  {len(x_t):>4} pts   arc={arc_t[-1]*100:.1f} cm")

# --- final_testing trajectory 2.5 cm, 9 waves ---
amp = 0.025
n_waves = 9
r_sin = gr + amp * np.sin(2.0 * np.pi * n_waves * phi)
x_sin = cx + r_sin * np.cos(gth)
y_sin = cy + r_sin * np.sin(gth)
inside = ws_path.contains_points(np.column_stack((x_sin, y_sin)))
x_t = x_sin[inside]
y_t = y_sin[inside]
t_t = u_fine[inside]
diffs = np.sqrt(np.diff(x_t)**2 + np.diff(y_t)**2)
arc_t = np.concatenate([[0.0], np.cumsum(diffs)])
amp_tag = f"{amp * 100:.1f}".replace('.', 'p')   # "2p5"
fname = f"traj_amp{amp_tag}cm_{n_waves}waves.csv"
fpath = os.path.join("trajectories_ref", fname)

with open(fpath, 'w', newline='') as f:
    f.write("# Sinusoidal end-effector reference trajectory\n")
    f.write(f"# amplitude_m    : {amp:.4f}\n")
    f.write(f"# amplitude_cm   : {amp * 100:.1f}\n")
    f.write(f"# n_waves        : {n_waves}\n")
    f.write(f"# ws_centre_m    : ({cx:.5f}, {cy:.5f})\n")
    f.write(f"# n_points       : {len(x_t)}\n")
    f.write(f"# arc_length_m   : {arc_t[-1]:.5f}\n")
    f.write(f"# arc_length_cm  : {arc_t[-1] * 100:.3f}\n")
    writer = csv.writer(f)
    writer.writerow(["t", "x_m", "y_m", "arc_length_m"])
    for row in zip(t_t, x_t, y_t, arc_t):
        writer.writerow([f"{v:.7f}" for v in row])

summary.append((fname, len(x_t), arc_t[-1] * 100))


color = WAVE_COLOR[n_waves]
ls, lw = AMP_LINEST[amp]
ax.plot(x_t, y_t, color=color, ls=ls, lw=lw, alpha=0.88, zorder=5+n_waves,
        label=f"A={amp*100:.1f}cm  n={n_waves}")

print(f"  ✓  {fname:<38}  {len(x_t):>4} pts   arc={arc_t[-1]*100:.1f} cm")
# ══════════════════════════════════════════════════════════════════════════
# FIGURE STYLING & SAVE
# ══════════════════════════════════════════════════════════════════════════
# Legend:  infrastructure items first, then grouped by colour and line-style
legend_items = [
    Line2D([0], [0], color='k',       lw=1.8,          label='Workspace boundary'),
    Line2D([0], [0], color='dimgray', lw=1.2, ls=':',  label='Generalize line'),
    Line2D([0], [0], color='none',                      label=''),
    Line2D([0], [0], color='none',
           label='  color = wave count   ·   ─── A=2.0 cm   ╌╌╌ A=2.5 cm'),
]
for n in WAVE_COUNTS:
    legend_items.append(
        Line2D([0], [0], color=WAVE_COLOR[n], lw=2.2,
               label=f'    {n} waves'))

ax.legend(handles=legend_items, fontsize=9, loc='upper left',
          framealpha=0.92, handlelength=3.0, borderpad=0.9,
          labelspacing=0.45)

ax.set_aspect('equal')
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title(
    "8 Reference Sinusoidal Trajectories — End-Effector Workspace\n"
    f"Amplitudes: 2.0 cm & 2.5 cm   ·   Waves: 4, 5, 6, 7   "
    f"·   Alpha shape (α = {ALPHA_SHAPE})",
    fontsize=11)
ax.grid(True, alpha=0.18)

plt.tight_layout()
fig_path = "figures/reference_trajectories.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')

# ══════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  FILES SAVED  →  trajectories_ref/")
print("═" * 65)
print(f"  {'Filename':<40} {'Pts':>5}   {'Arc length':>11}")
print("  " + "─" * 61)
for fname, n_pts, arc_cm in summary:
    print(f"  {fname:<40} {n_pts:>5}    {arc_cm:>8.1f} cm")
print("═" * 65)
print(f"\n  Figure  →  {fig_path}")
print("  Done.\n")

plt.show()