"""
Workspace analysis for the 2-DOF rotating four-bar linkage.

Computes and visualises:
1. End-effector reachable workspace in Cartesian and joint space.
2. Alpha-shape concave hull of the workspace boundary.
3. Generalise line (radial midpoint curve) through the workspace.
4. Sinusoidal reference trajectory oscillating along the radial direction.

Run:
    python workspace_analysis.py

Outputs:
    figures/workspace_analysis.png
    figures/sinusoidal_trajectory_spline.png
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dynamics.params   import a, b, L, e, h4, Q1_MIN, Q1_MAX, Q2_MIN, Q2_MAX
from dynamics.matrices import fk_G4

os.makedirs("figures", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1 — JOINT SPACE GRID
# ══════════════════════════════════════════════════════════════════════════
print("Computing workspace (joint space grid)...")
N      = 200
q1_arr = np.linspace(Q1_MIN, Q1_MAX, N)
q2_arr = np.linspace(Q2_MIN, Q2_MAX, N)

G4_pts, q_pts = [], []
for q1 in q1_arr:
    for q2 in q2_arr:
        q = np.array([q1, q2])
        G4_pts.append(fk_G4(q))
        q_pts.append(q)

G4_pts = np.array(G4_pts)   # (N², 2)
q_pts  = np.array(q_pts)    # (N², 2)
print(f"  {len(G4_pts):,} workspace points")


# ══════════════════════════════════════════════════════════════════════════
# 2 — ALPHA SHAPE (concave hull)
# ══════════════════════════════════════════════════════════════════════════
# Higher α → tighter boundary (may split into multiple polygons if too large).
# Lower  α → looser boundary (approaches the convex hull).
ALPHA = 25


def _compute_alpha_boundary(pts, alpha):
    """
    Compute the alpha-shape boundary of a 2-D point cloud.

    Builds a Delaunay triangulation, discards triangles whose circumradius
    exceeds 1/alpha, and returns the boundary edges ordered into polygons.

    Parameters
    ----------
    pts : np.ndarray, shape (M, 2)
    alpha : float

    Returns
    -------
    polygons : list of np.ndarray, each shape (K, 2)
    """
    tri = Delaunay(pts)
    ia, ib, ic = tri.simplices[:,0], tri.simplices[:,1], tri.simplices[:,2]

    # Vectorised circumradius — O(n_triangles)
    a_len = np.linalg.norm(pts[ia] - pts[ib], axis=1)
    b_len = np.linalg.norm(pts[ib] - pts[ic], axis=1)
    c_len = np.linalg.norm(pts[ic] - pts[ia], axis=1)
    s     = (a_len + b_len + c_len) / 2.0
    area  = np.sqrt(np.maximum(s*(s-a_len)*(s-b_len)*(s-c_len), 1e-30))
    circum_r = (a_len * b_len * c_len) / (4.0 * area)

    # Keep only triangles with small circumradius
    kept = tri.simplices[circum_r < 1.0 / alpha]
    if len(kept) == 0:
        return []

    # Boundary edge = edge appearing exactly once
    all_edges = np.concatenate(
        [kept[:,[0,1]], kept[:,[1,2]], kept[:,[2,0]]], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    cnt       = Counter(map(tuple, all_edges))
    boundary  = np.array([e for e, n in cnt.items() if n == 1])

    if len(boundary) == 0:
        return []

    # Traverse graph to order boundary edges into polygon(s)
    adj = {}
    for (u, v) in boundary:
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
            nxt = next((nb for nb in adj[curr]
                        if nb != prev and nb not in visited), None)
            if nxt is None:
                break
            chain.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt
        if len(chain) >= 3:
            polygons.append(pts[chain])

    return polygons


def _poly_area(verts):
    """Polygon area via the Shoelace formula."""
    x, y = verts[:,0], verts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))


print(f"Computing alpha shape (alpha={ALPHA})...")
# Use stride=3 for speed (~13k points vs 40k) — sufficient for accurate boundary
polys = _compute_alpha_boundary(G4_pts[::3], ALPHA)

if not polys:
    raise RuntimeError(
        f"Alpha shape is empty (alpha={ALPHA}). Try reducing ALPHA (e.g. 15).")

# Keep the largest polygon as the main workspace boundary
main_poly       = max(polys, key=_poly_area)
WS_area         = _poly_area(main_poly)
WS_center       = G4_pts.mean(axis=0)
boundary_closed = np.vstack([main_poly, main_poly[0]])

# matplotlib Path for fast point-in-polygon queries (C-extension ray casting)
ws_mpl_path = MplPath(boundary_closed)

print(f"  Alpha shape: {len(main_poly)} boundary vertices")
print(f"  Area  : {WS_area*1e4:.1f} cm²")
print(f"  Centre: ({WS_center[0]:.3f}, {WS_center[1]:.3f}) m")
print(f"  x range: [{G4_pts[:,0].min():.3f}, {G4_pts[:,0].max():.3f}] m")
print(f"  y range: [{G4_pts[:,1].min():.3f}, {G4_pts[:,1].max():.3f}] m")


# ══════════════════════════════════════════════════════════════════════════
# 3 — WORKSPACE PLOTS
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(
    "Workspace Analysis — G4 End-Effector\n"
    f"q1 ∈ [{np.degrees(Q1_MIN):.0f}°, {np.degrees(Q1_MAX):.0f}°]  "
    f"q2 ∈ [{np.degrees(Q2_MIN):.0f}°, {np.degrees(Q2_MAX):.0f}°]",
    fontsize=12)

# ── Plot 1: Cartesian workspace ───────────────────────────────────────────
ax = axes[0]
sc = ax.scatter(G4_pts[::5,0], G4_pts[::5,1],
                c=np.degrees(q_pts[::5,1]),
                cmap='RdYlGn', s=2, alpha=0.4, zorder=1)
cb = plt.colorbar(sc, ax=ax, shrink=0.75)
cb.set_label('q2 (°)', fontsize=9)

ax.fill(boundary_closed[:,0], boundary_closed[:,1],
        alpha=0.12, color='steelblue', zorder=0)
ax.plot(boundary_closed[:,0], boundary_closed[:,1],
        'k-', lw=1.5, zorder=3, label=f'Alpha boundary (α={ALPHA})')
ax.plot(*WS_center, 'k+', ms=10, zorder=5, label='WS centre')

ax.set_aspect('equal')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Workspace (colour = q2)\nAlpha-shape boundary')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.15)

# ── Plot 2: Joint space ───────────────────────────────────────────────────
ax = axes[1]
dist = np.linalg.norm(G4_pts - WS_center, axis=1)
sc2  = ax.scatter(np.degrees(q_pts[::5,0]),
                  np.degrees(q_pts[::5,1]),
                  c=dist[::5]*100,
                  cmap='plasma', s=2, alpha=0.5)
cb2 = plt.colorbar(sc2, ax=ax, shrink=0.75)
cb2.set_label('|G4 − centre| (cm)', fontsize=9)

rect = Rectangle(
    (np.degrees(Q1_MIN), np.degrees(Q2_MIN)),
    np.degrees(Q1_MAX - Q1_MIN), np.degrees(Q2_MAX - Q2_MIN),
    lw=2, edgecolor='red', facecolor='none',
    ls='--', label='Joint limits', zorder=5)
ax.add_patch(rect)
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.axvline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel('q1 (°)'); ax.set_ylabel('q2 (°)')
ax.set_title('Joint space\n(colour = distance from workspace centre)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)
ax.set_xlim(-100, 40); ax.set_ylim(-60, 60)

plt.tight_layout()
plt.savefig("figures/workspace_analysis.png", dpi=150, bbox_inches='tight')
print("\nSaved → figures/workspace_analysis.png")


# ══════════════════════════════════════════════════════════════════════════
# 4 — GENERALISE LINE: RADIAL MIDPOINT CURVE
# ══════════════════════════════════════════════════════════════════════════
# For each angular direction from the workspace centre, take the midpoint
# between the inner (r_min) and outer (r_max) workspace boundary.
# Independent of point-cloud density → more robust than the median.

print("\nComputing generalise line via radial midpoint...")

cx_ws, cy_ws = WS_center

dx_all    = G4_pts[:,0] - cx_ws
dy_all    = G4_pts[:,1] - cy_ws
r_all     = np.sqrt(dx_all**2 + dy_all**2)
theta_all = np.arctan2(dy_all, dx_all)

N_ANGLE_BINS = 150
theta_edges  = np.linspace(theta_all.min() - 1e-9,
                            theta_all.max() + 1e-9,
                            N_ANGLE_BINS + 1)
theta_idx    = np.digitize(theta_all, theta_edges)

gen_theta, gen_r_mid               = [], []
gen_r_min_arr, gen_r_max_arr       = [], []

for b_idx in range(1, N_ANGLE_BINS + 1):
    mask = theta_idx == b_idx
    if mask.sum() < 3:
        continue
    r_min = r_all[mask].min()
    r_max = r_all[mask].max()
    # Discard bins with negligible radial thickness (workspace edge artefacts)
    if (r_max - r_min) < 0.005:
        continue
    gen_theta.append(theta_all[mask].mean())
    gen_r_mid.append((r_min + r_max) / 2.0)
    gen_r_min_arr.append(r_min)
    gen_r_max_arr.append(r_max)

gen_theta     = np.array(gen_theta)
gen_r_mid     = np.array(gen_r_mid)
gen_r_min_arr = np.array(gen_r_min_arr)
gen_r_max_arr = np.array(gen_r_max_arr)

# Sort by angle
sort_idx      = np.argsort(gen_theta)
gen_theta     = gen_theta[sort_idx]
gen_r_mid     = gen_r_mid[sort_idx]
gen_r_min_arr = gen_r_min_arr[sort_idx]
gen_r_max_arr = gen_r_max_arr[sort_idx]

# Cartesian coordinates of midpoint samples
gen_x_samp = cx_ws + gen_r_mid * np.cos(gen_theta)
gen_y_samp = cy_ws + gen_r_mid * np.sin(gen_theta)

# Fit parametric spline through midpoint samples
tck, u      = splprep([gen_x_samp, gen_y_samp], s=5e-5, k=3)
u_fine      = np.linspace(0, 1, 500)
x_gen_line, y_gen_line = splev(u_fine, tck)

inner_x = cx_ws + gen_r_min_arr * np.cos(gen_theta)
inner_y = cy_ws + gen_r_min_arr * np.sin(gen_theta)
outer_x = cx_ws + gen_r_max_arr * np.cos(gen_theta)
outer_y = cy_ws + gen_r_max_arr * np.sin(gen_theta)

print(f"  {len(gen_theta)} valid angle bins")

# ── Sinusoidal trajectory along radial direction ──────────────────────────
n_waves   = 4
amplitude = 0.025   # m

r_gen_fine     = np.sqrt((x_gen_line - cx_ws)**2 + (y_gen_line - cy_ws)**2)
theta_gen_fine = np.arctan2(y_gen_line - cy_ws, x_gen_line - cx_ws)

r_sin = r_gen_fine + amplitude * np.sin(2 * np.pi * n_waves * u_fine)
x_sin = cx_ws + r_sin * np.cos(theta_gen_fine)
y_sin = cy_ws + r_sin * np.sin(theta_gen_fine)

# Batch check inside workspace
pts_sin     = np.column_stack((x_sin, y_sin))
inside_mask = ws_mpl_path.contains_points(pts_sin)
x_valid     = x_sin[inside_mask]
y_valid     = y_sin[inside_mask]
print(f"  Retained {len(x_valid)}/{len(x_sin)} sinusoidal points inside workspace")

# ── Figure ────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle(
    "Generalise Line: Radial Midpoint  +  Sinusoidal Trajectory Along Radius",
    fontsize=13)

# Sub-plot A: construction explanation
ax = axes2[0]
ax.scatter(G4_pts[::10,0], G4_pts[::10,1],
           c='lightgray', s=2, alpha=0.4, label='Point cloud', zorder=1)
ax.fill(boundary_closed[:,0], boundary_closed[:,1],
        alpha=0.08, color='steelblue', zorder=0)
ax.plot(boundary_closed[:,0], boundary_closed[:,1],
        'k-', lw=1.5, label=f'Alpha boundary (α={ALPHA})', zorder=3)
ax.plot(inner_x, inner_y, 'c.', ms=4, alpha=0.6, label='r_min (inner)', zorder=4)
ax.plot(outer_x, outer_y, 'm.', ms=4, alpha=0.6, label='r_max (outer)', zorder=4)
ax.scatter(gen_x_samp, gen_y_samp, s=25, c='blue', zorder=5, alpha=0.7,
           label='Midpoint samples')
ax.plot(x_gen_line, y_gen_line, 'b-', lw=2.5, alpha=0.9,
        label='Generalise spline', zorder=6)

# Illustrative radial segments
for i in range(0, len(gen_theta), max(1, len(gen_theta)//12)):
    ax.annotate('', xy=(outer_x[i], outer_y[i]),
                xytext=(inner_x[i], inner_y[i]),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.5))
    ax.plot(gen_x_samp[i], gen_y_samp[i], 'b|', ms=8, zorder=7)

ax.plot(cx_ws, cy_ws, 'k*', ms=12, zorder=8, label='WS centre')
ax.set_aspect('equal')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Construction: midpoint(r_min, r_max) at each angle')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.2)

# Sub-plot B: sinusoidal trajectory
ax = axes2[1]
ax.scatter(G4_pts[::10,0], G4_pts[::10,1],
           c='lightgray', s=2, alpha=0.4, label='Point cloud', zorder=1)
ax.fill(boundary_closed[:,0], boundary_closed[:,1],
        alpha=0.08, color='steelblue', zorder=0)
ax.plot(boundary_closed[:,0], boundary_closed[:,1],
        'k-', lw=1.5, label=f'Alpha boundary (α={ALPHA})', zorder=3)
ax.plot(x_gen_line, y_gen_line, 'b--', lw=2, alpha=0.8,
        label='Generalise line (radial midpoint)', zorder=4)
ax.plot(x_valid, y_valid, 'r-', lw=2.5,
        label=f'Sinusoidal traj ({n_waves} waves, A={amplitude*100:.1f} cm)',
        zorder=5)
ax.plot(cx_ws, cy_ws, 'k*', ms=12, zorder=6, label='WS centre')
ax.set_aspect('equal')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'Sinusoidal trajectory ({n_waves} waves, A={amplitude*100:.1f} cm)\n'
             'Oscillates along radial direction')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/sinusoidal_trajectory_spline.png", dpi=150, bbox_inches='tight')
print("Saved → figures/sinusoidal_trajectory_spline.png")
plt.show()