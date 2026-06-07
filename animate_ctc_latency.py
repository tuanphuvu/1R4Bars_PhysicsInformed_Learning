

# ┌─────────────────────────────────────────────────────────────────────┐
# │  THIS IS A TORQUE-CONTROLLED DYNAMICS SIMULATION   │
# │                                                                     │
# │  Control law (CTC — Computed Torque Control):                       │
# │    τ = M(q)·[q̈_d + Kd·ė + Kp·e] + C(q,q̇)·q̇ + g(q) − Q_F             │
# │                                                                     │
# │  Actual motion integrates  M·q̈ = τ + Q_F − C·q̇ − g  via RK45.       │
# │  q_sim ≠ q_des  when tracking error is non-zero.                    │
# │  Torques and Cartesian error shown in real time.                    │
# │                                                                     │
# │  EXTERNAL FORCE F(t):  magnitude ∈ [5, 15] N                        │
# │                         angle    ∈ [−90°, +30°]                     │ 
# │                         rate of change: random (slow ↔ fast)        │
# └─────────────────────────────────────────────────────────────────────┘
#
# ══════════════════════════════════════════════════════════════════════
# COLOUR SCHEME  (4 distinct components)
#   ① WHITE   #e8e8f0  — Body 1  (triangle O–O₂–O₃)
#   ② TEAL    #5bc8af  — Parallel links  (O₂→A lower,  O₃→B upper)
#   ③ ORANGE  #f4a261  — Coupler / end-effector  (A–G₄–B  H-bracket)
#   ④ CORAL   #e76f51  — Prismatic constraint  Lp  (Bp→G₃, dashed)
# ══════════════════════════════════════════════════════════════════════

import csv, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamics.params    import a, b, L, e, h4, xB, yB, Force_sensor_latency, create_time_varying_force
from dynamics.matrices  import (mass_matrix, coriolis_qdot, gravity_vector,
                                 qf_external, fk_G4)
from kinematics.ik_solver import ik_solve, _q_init_analytical

# ══════════════════════════════════════════════════════════════════════
# CONFIG  ← change TRAJ_FILE to pick any saved trajectory
# ══════════════════════════════════════════════════════════════════════
TRAJ_FILE = "trajectories_ref/traj_amp2p0cm_5waves.csv"

# All available options:
#   traj_amp2p0cm_4waves.csv   traj_amp2p5cm_4waves.csv
#   traj_amp2p0cm_5waves.csv   traj_amp2p5cm_5waves.csv
#   traj_amp2p0cm_6waves.csv   traj_amp2p5cm_6waves.csv
#   traj_amp2p0cm_7waves.csv   traj_amp2p5cm_7waves.csv  traj_amp2p5cm_9waves.csv

EE_SPEED  = 0.05    # m/s — end-effector Cartesian speed (controls duration)
                    # smaller → slower / more frames / heavier simulation

Kp, Kd    = 400.0, 40.0   # CTC PD gains (joint space)
FORCE_DELAY = Force_sensor_latency  # s delay between F(t) update and its effect on dynamics
INTERVAL  = 20             # ms between animation frames  (50 fps)
TRAIL_LEN = 150            # number of G4 trail points to keep

# ══════════════════════════════════════════════════════════════════════
# COLOURS
# ══════════════════════════════════════════════════════════════════════
C_BG      = '#0d1117'   # canvas background
C_PANEL   = '#161b22'   # subplot background
C_BODY1   = '#e8e8f0'   # ① white   — Body 1 triangle
C_LINK    = '#5bc8af'   # ② teal    — Parallel links
C_COUPLER = '#f4a261'   # ③ orange  — Coupler / H-bracket
C_PRIS    = "#e75151"   # ④ coral   — Prismatic constraint (Lp)
C_TRAIL   = '#ffd166'   # G4 trail / star
C_DES     = '#4a9eff'   # desired path (dashed)
C_ERR     = '#c77dff'   # tracking error plot
C_JOINT   = '#ffffff'   # joint circles
C_TEXT    = '#aab2c0'   # axis labels / ticks
C_GRID    = '#2a3040'   # grid / spines
C_FORCE   = '#ff6b6b'   # external force arrow

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD TRAJECTORY
# ══════════════════════════════════════════════════════════════════════
def load_trajectory(filepath):
    rows, meta = [], {}
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if line.startswith('#'):
                if ':' in line:
                    k, v = line[1:].split(':', 1)
                    meta[k.strip()] = v.strip()
                continue
            if not line or line.startswith('t,'):
                continue
            rows.append([float(v) for v in line.split(',')])
    d = np.array(rows)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3], meta


print(f"\n{'─'*60}")
print(f"  Loading  {TRAJ_FILE}")
t_param, x_ee, y_ee, arc_m, meta = load_trajectory(TRAJ_FILE)
amp_cm  = meta.get('amplitude_cm', '?')
n_waves = meta.get('n_waves', '?')
arc_cm  = arc_m[-1] * 100
T_total = arc_m[-1] / EE_SPEED

print(f"  {len(x_ee)} waypoints | arc = {arc_cm:.1f} cm")
print(f"  A = {amp_cm} cm  |  {n_waves} waves")
print(f"  Duration at {EE_SPEED*100:.0f} cm/s  →  {T_total:.2f} s")

t_phys = arc_m / EE_SPEED

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — INVERSE KINEMATICS
# ══════════════════════════════════════════════════════════════════════
print("  Running IK on trajectory waypoints...")
q_des_list, t_ik_list = [], []
q_prev = _q_init_analytical(np.array([x_ee[0], y_ee[0]]))

for xi, yi, ti in zip(x_ee, y_ee, t_phys):
    q_sol, ok = ik_solve(np.array([xi, yi]), q_prev, max_iter=300, tol=1e-7)
    if ok:
        q_des_list.append(q_sol)
        t_ik_list.append(ti)
        q_prev = q_sol

q_des = np.array(q_des_list)
t_ik  = np.array(t_ik_list)
print(f"  IK: {len(q_des)}/{len(x_ee)} points solved  |  t ∈ [0, {t_ik[-1]:.2f}] s")

if len(q_des) < 10:
    raise RuntimeError("IK solved fewer than 10 points — check TRAJ_FILE and robot params.")

cs = CubicSpline(t_ik, q_des)

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — TIME-VARYING EXTERNAL FORCE
# ══════════════════════════════════════════════════════════════════════
# F(t): magnitude ∈ [5, 15] N, angle ∈ [−90°, +30°]
# Exponential segment widths → varying change of F(t) (slow ↔ fast)
# CubicSpline 
F_func = create_time_varying_force(
    t_ik[0], t_ik[-1],
    mag_range=(5.0, 15.0),
    angle_deg_range=(-90.0, 30.0),
    n_segs=30
)
print(f"  Time-varying F(t) created: |F| ∈ [5,15] N  θ ∈ [-90°, +30°]")

# F_delayed(t) = F(t - FORCE_DELAY), t < FORCE_DELAY -> F(0)
def F_delayed(t):
    t_eff = max(t - FORCE_DELAY, 0.0)
    return F_func(t_eff)

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — CTC DYNAMICS SIMULATION
# ══════════════════════════════════════════════════════════════════════
def ode_ctc(t, state):
    q, dq  = state[:2], state[2:]
    q_d    = cs(t);   dq_d = cs(t, 1);   ddq_d = cs(t, 2)
    e_q    = q_d - q; de_q = dq_d - dq
    M      = mass_matrix(q[1])
    Cq     = coriolis_qdot(q[1], dq)
    G      = gravity_vector(q)
    
    QF_actual = qf_external(q, F_func(t))
    
    QF_meas = qf_external(q, F_delayed(t))

    tau = M @ (ddq_d + Kd*de_q + Kp*e_q) + Cq + G - QF_meas

    ddq = np.linalg.solve(M, tau + QF_actual - Cq - G)
    return np.concatenate([dq, ddq])
    return np.concatenate([dq, ddq])

print("  Integrating dynamics (RK45)...")
sol = solve_ivp(
    ode_ctc,
    [t_ik[0], t_ik[-1]],
    np.concatenate([q_des[0], cs(t_ik[0], 1)]),
    t_eval=t_ik, method='RK45', rtol=1e-8, atol=1e-10
)

# Pre-compute F at all t_eval points
F_log  = np.array([F_func(t) for t in t_ik])   # shape (N, 2)
q_sim  = sol.y[:2].T

tau1_log = np.zeros(len(t_ik))
tau2_log = np.zeros(len(t_ik))
err_mm   = np.zeros(len(t_ik))

for i in range(len(t_ik)):
    q    = q_sim[i];   dq   = sol.y[2:, i]
    q_d  = cs(t_ik[i]); dq_d = cs(t_ik[i], 1); ddq_d = cs(t_ik[i], 2)
    e_q  = q_d - q;    de_q = dq_d - dq
    M    = mass_matrix(q[1])
    Cq   = coriolis_qdot(q[1], dq)
    G    = gravity_vector(q)
    QF   = qf_external(q, F_log[i])      
    tau1_log[i], tau2_log[i] = M @ (ddq_d + Kd*de_q + Kp*e_q) + Cq + G - QF
    err_mm[i] = np.linalg.norm(fk_G4(q_sim[i]) - fk_G4(cs(t_ik[i]))) * 1000

print(f"  Max |τ₁| = {np.abs(tau1_log).max():.1f} N·m  "
      f"  Max |τ₂| = {np.abs(tau2_log).max():.1f} N·m")
print(f"  Max Cartesian error = {err_mm.max():.4f} mm")
print(f"{'─'*60}\n")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — FORWARD KINEMATICS  (all named points)
# ══════════════════════════════════════════════════════════════════════
def fk_all(q):
    q1, q2 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)

    def rot(xb, yb):
        return np.array([c1*xb - s1*yb, s1*xb + c1*yb])

    return {
        'O' : np.array([0.0, 0.0]),
        'O2': rot(a,           -e/2),
        'O3': rot(a,           +e/2),
        'G1': rot(a,            0.0),
        'G2': rot(a + b*c2,    -e/2 + b*s2),
        'G3': rot(a + b*c2,    +e/2 + b*s2),
        'A' : rot(a + L*c2,    -e/2 + L*s2),
        'B' : rot(a + L*c2,    +e/2 + L*s2),
        'G4': rot(a + L*c2 + h4,       L*s2),
        'Bp': rot(xB, yB),
    }


def _perp(p1, p2, length=0.018):
    """Unit vector perpendicular to (p2-p1), scaled to `length`."""
    d  = p2 - p1
    n  = np.array([-d[1], d[0]])
    n /= (np.linalg.norm(n) + 1e-12)
    return n * length

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — FIGURE LAYOUT
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 7.5), facecolor=C_BG)

gs  = fig.add_gridspec(3, 2, width_ratios=[2.4, 1],
                        hspace=0.40, wspace=0.30,
                        left=0.05, right=0.97, top=0.93, bottom=0.09)

ax       = fig.add_subplot(gs[:, 0])   # mechanism  (full height, left)
ax2      = fig.add_subplot(gs[0, 1])   # torques    (top right)
ax_force = fig.add_subplot(gs[1, 1])   # force      (middle right)
ax3      = fig.add_subplot(gs[2, 1])   # error      (bottom right)

# ── Unified dark theme ─────────────────────────────────────────
for _ax in (ax, ax2, ax_force, ax3):
    _ax.set_facecolor(C_PANEL)
    _ax.tick_params(colors=C_TEXT, labelsize=8)
    for sp in _ax.spines.values():
        sp.set_color(C_GRID)
    _ax.grid(True, color=C_GRID, lw=0.5, alpha=0.6)

# ── Mechanism axis ─────────────────────────────────────────────
G4_xy = np.array([fk_all(q_sim[i])['G4'] for i in range(len(t_ik))])
pad   = 0.14
_x_min = min(-0.07, G4_xy[:, 0].min()) - pad
_x_max = max( 0.60, G4_xy[:, 0].max()) + pad
# Expand x-range by 1.5x around its centre
_x_c = 0.5 * (_x_min + _x_max)
_x_w = (_x_max - _x_min) * 1.5
ax.set_xlim(_x_c - 0.5 * _x_w, _x_c + 0.5 * _x_w)
ax.set_ylim(G4_xy[:, 1].min() - pad,
            G4_xy[:, 1].max() + pad)
ax.set_aspect('equal')
ax.set_xlabel('x  (m)', color=C_TEXT, fontsize=9)
ax.set_ylabel('y  (m)', color=C_TEXT, fontsize=9)
ax.set_title(
    f'2-DOF Parallelogram Four-Bar  —  CTC + Time-Varying F(t)\n'
    f'A = {amp_cm} cm  │  {n_waves} waves  │  Kp = {Kp:.0f}   Kd = {Kd:.0f}'
    f'  │  |F(t)| ∈ [5, 15] N   θ ∈ [−90°, +30°]',
    color='white', fontsize=10, pad=10, fontweight='bold')

# Desired path (reference, static)
ax.plot(x_ee, y_ee, '--', color=C_DES, lw=1.1, alpha=0.55, label='Desired path')

# Ground symbol at O
ax.plot(0, 0, 's', color='#606070', ms=9, zorder=5)
ax.plot([-0.035, 0.035], [-0.014, -0.014], color='#606070', lw=2.0)
for dx_g in np.linspace(-0.030, 0.030, 6):
    ax.plot([dx_g, dx_g - 0.012], [-0.014, -0.026], color='#606070', lw=1.0)

# ── Legend patches (component colours) ────────────────────────
_leg_kw = dict(facecolor=C_BG, edgecolor=C_GRID, labelcolor='white',
               fontsize=8, loc='upper left')
_patches = [
    Line2D([0], [0], color=C_BODY1,   lw=3, label='① Body 1  (O–O₂–O₃)'),
    Line2D([0], [0], color=C_LINK,    lw=3, label='② Parallel links  (O₂→A, O₃→B)'),
    Line2D([0], [0], color=C_COUPLER, lw=3, label='③ Coupler / H-bracket  (A–G₄–B)'),
    Line2D([0], [0], color=C_PRIS,    lw=2, ls='--', label='④ Prismatic Lp  (Bₚ→G₃)'),
    Line2D([0], [0], color=C_DES,     lw=1.2, ls='--', label='Desired path'),
    Line2D([0], [0], color=C_FORCE,   lw=2, label='F(t)  time-varying'),
]
ax.legend(handles=_patches, **_leg_kw)

time_text = ax.text(0.02, 0.03, '', transform=ax.transAxes,
                    color='white', fontsize=9, va='bottom',
                    fontfamily='monospace')

# Force text moved to bottom-left (will display current Fx, Fy)
lbl_Fext = ax.text(0.02, 0.06, '', transform=ax.transAxes,
                   color=C_FORCE, fontsize=9, fontweight='bold', zorder=10,
                   fontfamily='monospace')

# ══════════════════════════════════════════════════════════════════════
# LINK ARTISTS — 4 colour-coded components
# ══════════════════════════════════════════════════════════════════════
_LK = dict(solid_capstyle='round', solid_joinstyle='round', zorder=3)

# ① Body 1  — white triangle  O–O₂–O₃
ln_body1,   = ax.plot([], [], color=C_BODY1,   lw=3.0, **_LK)
ln_O3_Bp,   = ax.plot([], [], color=C_BODY1,   lw=2.5, **_LK)
ln_j_Bp,    = ax.plot([], [], color=C_BODY1,   lw=2.5, **_LK)

# ② Parallel links — teal
ln_lower,   = ax.plot([], [], color=C_LINK,    lw=3.0, **_LK)   # O₂ → A
ln_upper,   = ax.plot([], [], color=C_LINK,    lw=3.0, **_LK)   # O₃ → B

# ③ Coupler H-bracket — orange
ln_coup_vert,  = ax.plot([], [], color=C_COUPLER, lw=3.0, **_LK)
ln_coup_armA,  = ax.plot([], [], color=C_COUPLER, lw=2.5, **_LK)
ln_coup_armB,  = ax.plot([], [], color=C_COUPLER, lw=2.5, **_LK)
ln_midAB_G4,   = ax.plot([], [], color=C_COUPLER, lw=2.0, **_LK)
ln_EF,         = ax.plot([], [], color=C_COUPLER, lw=2.0, **_LK)
ln_E_perp,     = ax.plot([], [], color=C_COUPLER, lw=1.8, **_LK)
ln_F_perp,     = ax.plot([], [], color=C_COUPLER, lw=1.8, **_LK)

# ④ Prismatic constraint — coral dashed
ln_pris,    = ax.plot([], [], color=C_PRIS,    lw=2.0, ls='--',
                       zorder=3, dash_capstyle='round')

# ── Joints (white circles) ─────────────────────────────────────
pt_joints,  = ax.plot([], [], 'o', color=C_JOINT,   ms=7,  zorder=6,
                       markeredgecolor='#333344', markeredgewidth=1.0)

# G4 end-effector star
pt_G4,      = ax.plot([], [], '*', color=C_TRAIL,   ms=14, zorder=8,
                       markeredgecolor='#aa7722', markeredgewidth=0.8)

# Bp fixed prismatic base
pt_Bp,      = ax.plot([], [], 's', color=C_PRIS,    ms=6,  zorder=6)

# ── G4 trail ──────────────────────────────────────────────────
trail_x, trail_y = [], []
ln_trail,   = ax.plot([], [], '-', color=C_TRAIL, lw=1.5, alpha=0.55, zorder=4)

# ── Node labels ────────────────────────────────────────────────
_LBL_KW = dict(fontsize=7.5, color='#d0d8e8', zorder=9,
               fontfamily='sans-serif',
               bbox=dict(boxstyle='round,pad=0.15', fc=C_BG, alpha=0.75, ec='none'))
lbl_O2  = ax.text(0, 0, ' O₂', **_LBL_KW)
lbl_O3  = ax.text(0, 0, ' O₃', **_LBL_KW)
lbl_G1  = ax.text(0, 0, ' G₁', **_LBL_KW)
lbl_G2  = ax.text(0, 0, ' G₂', **_LBL_KW)
lbl_G3  = ax.text(0, 0, ' G₃', **_LBL_KW)
lbl_G4l = ax.text(0, 0, ' G₄', color=C_TRAIL,
                  fontsize=8, fontweight='bold', zorder=9,
                  bbox=dict(boxstyle='round,pad=0.15', fc=C_BG, alpha=0.75, ec='none'))
lbl_A   = ax.text(0, 0, ' A',  **_LBL_KW)
lbl_B   = ax.text(0, 0, ' B',  **_LBL_KW)
lbl_Lp  = ax.text(0, 0, ' Lp', color=C_PRIS, fontsize=7.5, zorder=9,
                  bbox=dict(boxstyle='round,pad=0.15', fc=C_BG, alpha=0.75, ec='none'))

# ── External force arrow at G4 ─────────────────────────────────
# F_SCALE: m/N — 10 N → 0.10 m
F_SCALE  = 0.0150

arrow_F  = ax.annotate(
    '', xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle='->', color=C_FORCE, lw=2.0,
                    mutation_scale=14),
    zorder=10)
# lbl_Fext defined near bottom-left as axes text

# ══════════════════════════════════════════════════════════════════════
# TORQUE SUBPLOT (top right)
# ══════════════════════════════════════════════════════════════════════
tau_max = max(np.abs(tau1_log).max(), np.abs(tau2_log).max()) * 1.2
ax2.set_xlim(0, t_ik[-1]); ax2.set_ylim(-tau_max, tau_max)
ax2.set_xlabel('t  (s)',    color=C_TEXT, fontsize=8)
ax2.set_ylabel('τ  (N·m)', color=C_TEXT, fontsize=8)
ax2.set_title('Control Torques', color='white', fontsize=9, fontweight='bold')
ax2.axhline(0, color=C_GRID, lw=0.8)

ln_tau1, = ax2.plot([], [], color=C_LINK,    lw=1.4, label='τ₁  (revolute)')
ln_tau2, = ax2.plot([], [], color=C_COUPLER, lw=1.4, label='τ₂  (prismatic)')
ax2.legend(fontsize=7.5, facecolor=C_PANEL, edgecolor=C_GRID,
           labelcolor='white', loc='upper right')
t_line2  = ax2.axvline(0, color='white', lw=0.9, alpha=0.5)
tau_info = ax2.text(0.03, 0.95, '', transform=ax2.transAxes,
                    color='white', fontsize=8, va='top', fontfamily='monospace')

# ══════════════════════════════════════════════════════════════════════
# FORCE SUBPLOT (middle right)
# ══════════════════════════════════════════════════════════════════════
f_peak = np.abs(F_log).max()
f_max = f_peak * 1.2 if f_peak > 0 else 1.0
ax_force.set_xlim(0, t_ik[-1]); ax_force.set_ylim(-f_max, f_max)
ax_force.set_xlabel('t  (s)',    color=C_TEXT, fontsize=8)
ax_force.set_ylabel('F  (N)',    color=C_TEXT, fontsize=8)
ax_force.set_title('External Force at G₄', color='white', fontsize=9, fontweight='bold')
ax_force.axhline(0, color=C_GRID, lw=0.8)

ln_Fx, = ax_force.plot([], [], color=C_LINK,    lw=1.4, label='F_x')
ln_Fy, = ax_force.plot([], [], color=C_COUPLER, lw=1.4, label='F_y')
ax_force.legend(fontsize=7.0, facecolor=C_PANEL, edgecolor=C_GRID,
                labelcolor='white', loc='upper right')
t_lineF  = ax_force.axvline(0, color='white', lw=0.9, alpha=0.5)
force_info = ax_force.text(0.03, 0.95, '', transform=ax_force.transAxes,
                           color='white', fontsize=8, va='top', fontfamily='monospace')

# ══════════════════════════════════════════════════════════════════════
# TRACKING ERROR SUBPLOT (bottom right)
# ══════════════════════════════════════════════════════════════════════
err_ceil = err_mm.max() * 1.35 + 1e-4
ax3.set_xlim(0, t_ik[-1]); ax3.set_ylim(0, err_ceil)
ax3.set_xlabel('t  (s)',    color=C_TEXT, fontsize=8)
ax3.set_ylabel('‖e‖  (mm)', color=C_TEXT, fontsize=8)
ax3.set_title('Cartesian Tracking Error  (G₄)', color='white',
              fontsize=9, fontweight='bold')
ax3.axhline(err_mm.max(), color=C_FORCE, lw=0.8, ls=':', alpha=0.7)
ax3.text(0.98, 0.95, f'max = {err_mm.max():.4f} mm',
         transform=ax3.transAxes, color=C_FORCE, fontsize=8,
         va='top', ha='right')

ln_err,  = ax3.plot([], [], color=C_ERR, lw=1.4, label='‖G₄_sim − G₄_des‖')
ax3.legend(fontsize=7.5, facecolor=C_PANEL, edgecolor=C_GRID,
           labelcolor='white', loc='upper right')
t_line3  = ax3.axvline(0, color='white', lw=0.9, alpha=0.5)
err_info = ax3.text(0.03, 0.68, '', transform=ax3.transAxes,
                    color='white', fontsize=8, va='top', fontfamily='monospace')

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — ANIMATION
# ══════════════════════════════════════════════════════════════════════
def update(frame):
    i   = frame % len(t_ik)
    q   = q_sim[i]
    t   = t_ik[i]
    pts = fk_all(q)

    A, B, G4 = pts['A'], pts['B'], pts['G4']

    # ① Body 1  — white triangle  O → O₂ → O₃ → O
    ln_body1.set_data(
        [pts['O'][0],  pts['O2'][0], pts['O3'][0], pts['O'][0]],
        [pts['O'][1],  pts['O2'][1], pts['O3'][1], pts['O'][1]])

    # ② Parallel links (teal)
    ln_lower.set_data([pts['O2'][0], A[0]], [pts['O2'][1], A[1]])
    ln_upper.set_data([pts['O3'][0], B[0]], [pts['O3'][1], B[1]])

    # ③ Coupler H-bracket (orange)
    ln_coup_vert.set_data([A[0], B[0]], [A[1], B[1]])
    pv = _perp(A, B, length=0.022)
    sign = 1 if np.dot(G4 - (A+B)/2, pv) > 0 else -1
    pv   = pv * sign
    ln_coup_armA.set_data([A[0], A[0] + pv[0]], [A[1], A[1] + pv[1]])
    ln_coup_armB.set_data([B[0], B[0] + pv[0]], [B[1], B[1] + pv[1]])
    m_ab = 0.5 * (A + B)
    ln_midAB_G4.set_data([m_ab[0], G4[0]], [m_ab[1], G4[1]])
    AB = B - A
    ab_len = np.linalg.norm(AB)
    if ab_len < 1e-12:
        u = np.array([1.0, 0.0])
    else:
        u = AB / ab_len
    L_EF   = 0.5 * ab_len
    half_L = 0.5 * L_EF
    E      = G4 + half_L * u
    F_pt   = G4 - half_L * u          # NOTE: renamed F→F_pt (avoid clash with F_log)
    ln_EF.set_data([E[0], F_pt[0]], [E[1], F_pt[1]])
    perp = np.array([-u[1], u[0]])
    ln_E_perp.set_data([E[0]    - L_EF * perp[0], E[0]],
                       [E[1]    - L_EF * perp[1], E[1]])
    ln_F_perp.set_data([F_pt[0] - L_EF * perp[0], F_pt[0]],
                       [F_pt[1] - L_EF * perp[1], F_pt[1]])

    # ④ Prismatic Lp  (coral dashed)  Bp → G₃
    ln_pris.set_data([pts['Bp'][0], pts['G3'][0]],
                     [pts['Bp'][1], pts['G3'][1]])
    lbl_Lp.set_position(((pts['Bp'][0]+pts['G3'][0])/2 + 0.01,
                          (pts['Bp'][1]+pts['G3'][1])/2))

    # Additional Body1 segments
    ln_O3_Bp.set_data([pts['O3'][0], pts['Bp'][0]],
                      [pts['O3'][1], pts['Bp'][1]])
    j_pt = 0.5 * (pts['O'] + pts['O3'])
    ln_j_Bp.set_data([j_pt[0], pts['Bp'][0]], [j_pt[1], pts['Bp'][1]])

    # Joints
    jx = [pts['O2'][0], pts['O3'][0], A[0], B[0],
          pts['G2'][0], pts['G3'][0]]
    jy = [pts['O2'][1], pts['O3'][1], A[1], B[1],
          pts['G2'][1], pts['G3'][1]]
    pt_joints.set_data(jx, jy)

    pt_G4.set_data([G4[0]], [G4[1]])
    pt_Bp.set_data([pts['Bp'][0]], [pts['Bp'][1]])

    # Node labels
    lbl_O2.set_position( (pts['O2'][0]+0.005, pts['O2'][1]-0.025))
    lbl_O3.set_position( (pts['O3'][0]+0.005, pts['O3'][1]+0.010))
    lbl_G1.set_position( (pts['G1'][0]+0.010, pts['G1'][1]))
    lbl_G2.set_position( (pts['G2'][0]+0.008, pts['G2'][1]-0.022))
    lbl_G3.set_position( (pts['G3'][0]+0.008, pts['G3'][1]+0.010))
    lbl_G4l.set_position((G4[0]+0.012,        G4[1]+0.012))
    lbl_A.set_position(  (A[0]+0.008,          A[1]-0.022))
    lbl_B.set_position(  (B[0]+0.008,          B[1]+0.010))

    # ── External force arrow — time-varying ──────────────────────
    Fi     = F_log[i]                                   # cached, no recompute
    F_norm = np.linalg.norm(Fi)
    F_ang  = np.degrees(np.arctan2(Fi[1], Fi[0]))
    f_tip  = G4 + F_SCALE * Fi                          # length ∝ magnitude
    arrow_F.set_position(f_tip)
    arrow_F.xy = G4
    # update bottom-left text with current force components and magnitude
    lbl_Fext.set_text(f'F = ({Fi[0]:+.2f}, {Fi[1]:+.2f}) N  | |F| = {F_norm:.1f} N  θ = {F_ang:.0f}°')

    # Trail
    trail_x.append(G4[0]); trail_y.append(G4[1])
    if len(trail_x) > TRAIL_LEN:
        trail_x.pop(0); trail_y.pop(0)
    ln_trail.set_data(trail_x, trail_y)

    # Time text (bottom-left) — only shows time/percentage
    pct = 100.0 * t / t_ik[-1]
    time_text.set_text(f't = {t:.2f} s   [{pct:3.0f}%]')

    # Torques
    ln_tau1.set_data(t_ik[:i+1], tau1_log[:i+1])
    ln_tau2.set_data(t_ik[:i+1], tau2_log[:i+1])
    t_line2.set_xdata([t, t])
    tau_info.set_text(f'τ₁ = {tau1_log[i]:+.1f}\nτ₂ = {tau2_log[i]:+.1f}  N·m')
    # Update force subplot lines/info
    ln_Fx.set_data(t_ik[:i+1], F_log[:i+1, 0])
    ln_Fy.set_data(t_ik[:i+1], F_log[:i+1, 1])
    t_lineF.set_xdata([t, t])
    force_info.set_text(f'Fx = {Fi[0]:+.2f} N\nFy = {Fi[1]:+.2f} N')

    # Cartesian error
    ln_err.set_data(t_ik[:i+1], err_mm[:i+1])
    t_line3.set_xdata([t, t])
    err_info.set_text(f'{err_mm[i]:.4f} mm')

    return (ln_body1, ln_O3_Bp, ln_j_Bp,
            ln_lower, ln_upper,
            ln_coup_vert, ln_coup_armA, ln_coup_armB, ln_midAB_G4,
            ln_EF, ln_E_perp, ln_F_perp,
            ln_pris,
            pt_joints, pt_G4, pt_Bp, ln_trail,
            lbl_O2, lbl_O3, lbl_G1, lbl_G2, lbl_G3, lbl_G4l,
            lbl_A, lbl_B, lbl_Lp,
            arrow_F, lbl_Fext, time_text,
            ln_tau1, ln_tau2, t_line2, tau_info,
            ln_Fx, ln_Fy, t_lineF, force_info,
            ln_err, t_line3, err_info)


anim = FuncAnimation(
    fig, update,
    frames=len(t_ik) * 999,
    interval=INTERVAL,
    blit=True
)

plt.tight_layout(pad=1.5)
print("Showing animation — close window to exit.\n")
plt.show()