# validating_controllers.py
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  CLOSED-LOOP CONTROLLER COMPARISON                                   │
# │                                                                      │
# │  Trajectory : traj_amp2p5cm_9waves.csv  (trajectories_ref/)         │
# │                                                                      │
# │  3 controllers — same robot, same F(t), same initial state:          │
# │                                                                      │
# │  ① CTC  — Computed Torque Control (ground-truth parameters)          │
# │      τ = M_true·[q̈_d + Kd·ė + Kp·e] + C_true·q̇ + G_true − QF         │
# │                                                                      │
# │  ② PINN — CTC structure with Physics‑informed learned parameters     │
# │      τ = M_pinn·[q̈_d + Kd·ė + Kp·e] + C_pinn·q̇ + G_pinn − QF         │
# │                                                                      │
# │  ③ MLP  — Black-box torque prediction                                │
# │      τ = MLP(q, q̇, q̈_d, F_sensor)                                   │
# │                                                                      │
# │  TRUE ROBOT DYNAMICS are always ground-truth:                        │
# │      M_true·q̈ = τ + QF_actual − C_true·q̇ − G_true                    │
# │                                                                      │
# └──────────────────────────────────────────────────────────────────────┘

import itertools
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines    import Line2D
from scipy.integrate     import solve_ivp
from scipy.interpolate   import CubicSpline

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found.  pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamics.params   import (a, b, L, e as e_link, h4, xB, yB, g,
                                Force_sensor_latency,
                                create_time_varying_force)
from dynamics.matrices import (mass_matrix, coriolis_qdot, gravity_vector,
                                qf_external, fk_G4)
from kinematics.ik_solver import ik_solve, _q_init_analytical

os.makedirs("figures", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
TRAJ_FILE   = "trajectories_ref/traj_amp2p5cm_9waves.csv"
EE_SPEED    = 0.05        # m/s
Kp, Kd      = 400.0, 40.0
FORCE_DELAY = Force_sensor_latency   # s

INTERVAL  = 20     # ms per animation frame (50 fps)
TRAIL_LEN = 130    

# ── Dark theme ──────────────────────────────────────────────────────────
C_BG    = '#0d1117'
C_PANEL = '#161b22'
C_GRID  = '#2a3040'
C_TEXT  = '#aab2c0'
C_CTC   = '#4a9eff'   # blue   — CTC
C_PINN  = '#5bc8af'   # teal   — PINN
C_MLP   = '#f4a261'   # orange — MLP
C_DES   = '#ffffff'   # desired path
C_FORCE = '#ff6b6b'   # force arrow / magnitude
C_TRAIL = '#ffd166'   # gold for G4 star and trail
C_PRIS  = '#e75151'   # coral for prismatic constraint

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════
def load_trajectory(filepath):
    rows, meta = [], {}
    with open(filepath) as f:
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
    return d[:,0], d[:,1], d[:,2], d[:,3], meta

print(f"\n{'─'*64}")
print(f"  Loading  {TRAJ_FILE}")
if not os.path.exists(TRAJ_FILE):
    raise FileNotFoundError(f"{TRAJ_FILE} not found.")

_, x_ee, y_ee, arc_m, meta = load_trajectory(TRAJ_FILE)
t_phys = arc_m / EE_SPEED
print(f"  {len(x_ee)} waypoints  |  arc = {arc_m[-1]*100:.1f} cm  |  "
      f"T = {t_phys[-1]:.2f} s")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — INVERSE KINEMATICS  →  CubicSpline reference q_d(t)
# ══════════════════════════════════════════════════════════════════════════
print("  Running IK...")
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
cs    = CubicSpline(t_ik, q_des)   # q_d(t), q̇_d(t), q̈_d(t)
t_eval = t_ik                       # evaluation grid = IK waypoints
N      = len(t_eval)
print(f"  IK: {len(q_des)}/{len(x_ee)} solved  |  t ∈ [0, {t_ik[-1]:.2f}] s")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — TIME-VARYING EXTERNAL FORCE
# ══════════════════════════════════════════════════════════════════════════
# F_func(t)    = true force at time t
# F_delayed(t) = F_func(t − FORCE_DELAY) = sensor reading at time t
# Both controllers receive F_delayed.
# True dynamics uses F_func(t).
F_func = create_time_varying_force(
    t_ik[0], t_ik[-1],
    mag_range=(5.0, 15.0),
    angle_deg_range=(-90.0, 30.0),
    n_segs=30
)
def F_delayed(t):
    return F_func(max(t - FORCE_DELAY, 0.0))

# Cache for plotting and animation (sensor readings = delayed)
F_log = np.array([F_delayed(t) for t in t_eval])   # (N, 2)
print(f"  Force: |F| ∈ [5,15] N  θ ∈ [-90°,+30°]  "
      f"delay = {FORCE_DELAY*1000:.1f} ms")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — LOAD PINN LEARNED PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
# PINN controller = CTC structure with learned θ.
# No PyTorch needed at runtime — just the 5 float values.
if not os.path.exists("models/pinn_results.pt"):
    raise FileNotFoundError(
        "models/pinn_results.pt not found — run train_pinn.py first.")

pinn_data = torch.load("models/pinn_results.pt", map_location='cpu')
th_p      = pinn_data['theta_learned']
K1_p    = float(th_p['K1'])
K2_p    = float(th_p['K2'])
M11_0_p = float(th_p['M11_0'])
meff1_p = float(th_p['meff1'])
meff2_p = float(th_p['meff2'])
_g_val  = float(g)

#  physics functions (numpy, identical equations to PhysicsModel)
def pinn_M(q2):
    c2 = np.cos(q2)
    return np.array([[M11_0_p + 2*K1_p*c2, K2_p + K1_p*c2],
                     [K2_p + K1_p*c2,      K2_p           ]])

def pinn_Cq(q2, dq):
    s2 = np.sin(q2)
    return np.array([-K1_p*s2*(2*dq[0]*dq[1] + dq[1]**2),
                      K1_p*s2*dq[0]**2])

def pinn_G(q):
    c1  = np.cos(q[0])
    c12 = np.cos(q[0]+q[1])
    return _g_val * np.array([meff1_p*c1 + meff2_p*c12, meff2_p*c12])

print(f"  PINN θ: K1={K1_p:.5f}  K2={K2_p:.5f}  M11_0={M11_0_p:.5f}  "
      f"meff1={meff1_p:.5f}  meff2={meff2_p:.5f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — LOAD MLP
# ══════════════════════════════════════════════════════════════════════════
# MLP input:  [q₁, q₂, dq₁, dq₂, ddq₁, ddq₂, Fx, Fy]  (8 features)

for path in ["models/mlp_baseline.pt", "models/mlp_norm.npz"]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run train_baseline.py first.")

# Redefine architecture
_HIDDEN = 128
_LAYERS = 3

class MLPBaseline(nn.Module):
    def __init__(self, in_dim=8, hidden=_HIDDEN, n_layers=_LAYERS, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

mlp = MLPBaseline()
mlp.load_state_dict(torch.load("models/mlp_baseline.pt", map_location='cpu'))
mlp.eval()

norm   = np.load("models/mlp_norm.npz")
X_mean = norm['X_mean'].astype(np.float32)
X_std  = norm['X_std'].astype(np.float32)
Y_mean = norm['Y_mean'].astype(np.float32)
Y_std  = norm['Y_std'].astype(np.float32)

def mlp_predict(q, dq, ddq, F):
    """Predict τ from physical inputs using trained MLP."""
    x  = np.hstack([q, dq, ddq, F]).astype(np.float32)
    xn = (x - X_mean) / X_std
    with torch.no_grad():
        yn = mlp(torch.from_numpy(xn[None])).numpy()[0]
    return (yn * Y_std + Y_mean).astype(np.float64)

n_mlp_params = sum(p.numel() for p in mlp.parameters())
print(f"  MLP loaded: {n_mlp_params:,} parameters  (8 → 128×3 → 2)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — RUN 3 CLOSED-LOOP SIMULATIONS (RK45)
# ══════════════════════════════════════════════════════════════════════════
q0  = q_des[0]
dq0 = cs(t_ik[0], 1)
y0  = np.concatenate([q0, dq0])

# ── ① CTC ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*64}")
print("  [1/3] CTC  (true parameters)...")

def ode_ctc(t, state):
    q, dq   = state[:2], state[2:]
    q_d     = cs(t);   dq_d  = cs(t,1);  ddq_d = cs(t,2)
    eq, deq = q_d - q, dq_d - dq
    M       = mass_matrix(q[1])
    Cq      = coriolis_qdot(q[1], dq)
    G       = gravity_vector(q)
    QF_act  = qf_external(q, F_func(t))
    QF_meas = qf_external(q, F_delayed(t))
    tau     = M @ (ddq_d + Kd*deq + Kp*eq) + Cq + G - QF_meas
    ddq     = np.linalg.solve(M, tau + QF_act - Cq - G)
    return np.concatenate([dq, ddq])

sol_ctc = solve_ivp(ode_ctc, [t_ik[0], t_ik[-1]], y0,
                    t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
q_ctc  = sol_ctc.y[:2].T
dq_ctc = sol_ctc.y[2:].T
print(f"     status={sol_ctc.status}  ({'success' if sol_ctc.success else 'FAILED'})")

# ── ② PINN ────────────────────────────────────────────────────────────────
print("  [2/3] PINN  (learned parameters, CTC structure)...")

def ode_pinn(t, state):
    q, dq   = state[:2], state[2:]
    q_d     = cs(t);   dq_d  = cs(t,1);  ddq_d = cs(t,2)
    eq, deq = q_d - q, dq_d - dq
    # Controller uses PINN-learned matrices
    M_p     = pinn_M(q[1])
    Cq_p    = pinn_Cq(q[1], dq)
    G_p     = pinn_G(q)
    QF_meas = qf_external(q, F_delayed(t))
    tau     = M_p @ (ddq_d + Kd*deq + Kp*eq) + Cq_p + G_p - QF_meas
    # True robot dynamics
    M_t     = mass_matrix(q[1])
    Cq_t    = coriolis_qdot(q[1], dq)
    G_t     = gravity_vector(q)
    QF_act  = qf_external(q, F_func(t))
    ddq     = np.linalg.solve(M_t, tau + QF_act - Cq_t - G_t)
    return np.concatenate([dq, ddq])

sol_pinn = solve_ivp(ode_pinn, [t_ik[0], t_ik[-1]], y0,
                     t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
q_pinn  = sol_pinn.y[:2].T
dq_pinn = sol_pinn.y[2:].T
print(f"     status={sol_pinn.status}  ({'success' if sol_pinn.success else 'FAILED'})")

# ── ③ MLP ─────────────────────────────────────────────────────────────────
print("  [3/3] MLP  (fixed-step RK4, batch-efficient)...")

def mlp_dynamics(q, dq, t):
    ddq_d  = cs(t, 2)
    F_s    = F_delayed(t)
    tau    = np.clip(mlp_predict(q, dq, ddq_d, F_s), -500.0, 500.0)
    M_t    = mass_matrix(q[1])
    Cq_t   = coriolis_qdot(q[1], dq)
    G_t    = gravity_vector(q)
    QF_act = qf_external(q, F_func(t))
    ddq    = np.linalg.solve(M_t, tau + QF_act - Cq_t - G_t)
    return dq.copy(), ddq

q_mlp_list  = [q0.copy()]
dq_mlp_list = [dq0.copy()]
q_c, dq_c   = q0.copy(), dq0.copy()

for i in range(1, N):
    dt_i  = t_eval[i] - t_eval[i-1]
    t_cur = t_eval[i-1]

    k1q, k1dq = mlp_dynamics(q_c,                  dq_c,                  t_cur)
    k2q, k2dq = mlp_dynamics(q_c + 0.5*dt_i*k1q,  dq_c + 0.5*dt_i*k1dq, t_cur + 0.5*dt_i)
    k3q, k3dq = mlp_dynamics(q_c + 0.5*dt_i*k2q,  dq_c + 0.5*dt_i*k2dq, t_cur + 0.5*dt_i)
    k4q, k4dq = mlp_dynamics(q_c + dt_i*k3q,       dq_c + dt_i*k3dq,      t_cur + dt_i)

    q_c  += (dt_i / 6) * (k1q  + 2*k2q  + 2*k3q  + k4q)
    dq_c += (dt_i / 6) * (k1dq + 2*k2dq + 2*k3dq + k4dq)

    if np.any(~np.isfinite(q_c)) or np.any(~np.isfinite(dq_c)):
        print(f"     !! MLP diverged at t={t_eval[i]:.2f}s — frozen to last valid state", flush=True)
        q_c  = q_mlp_list[-1].copy()
        dq_c = dq_mlp_list[-1].copy()

    q_mlp_list.append(q_c.copy())
    dq_mlp_list.append(dq_c.copy())

    if i % max(1, N // 10) == 0:
        print(f"     {100*i//N}%  (t={t_eval[i]:.1f}s)", flush=True)

q_mlp  = np.array(q_mlp_list)
dq_mlp = np.array(dq_mlp_list)
print("     Done.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — POST-PROCESS  (torques, FK, errors)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  Post-computing torques and errors...")

def recompute_tau_ctc(q_arr, dq_arr):
    tau = np.zeros((N, 2))
    for i, t in enumerate(t_eval):
        q, dq   = q_arr[i], dq_arr[i]
        q_d     = cs(t);   dq_d  = cs(t,1);  ddq_d = cs(t,2)
        eq, deq = q_d-q, dq_d-dq
        M       = mass_matrix(q[1])
        Cq      = coriolis_qdot(q[1], dq)
        G       = gravity_vector(q)
        QF_meas = qf_external(q, F_delayed(t))
        tau[i]  = M @ (ddq_d + Kd*deq + Kp*eq) + Cq + G - QF_meas
    return tau

def recompute_tau_pinn(q_arr, dq_arr):
    tau = np.zeros((N, 2))
    for i, t in enumerate(t_eval):
        q, dq   = q_arr[i], dq_arr[i]
        q_d     = cs(t);   dq_d  = cs(t,1);  ddq_d = cs(t,2)
        eq, deq = q_d-q, dq_d-dq
        M_p     = pinn_M(q[1])
        Cq_p    = pinn_Cq(q[1], dq)
        G_p     = pinn_G(q)
        QF_meas = qf_external(q, F_delayed(t))
        tau[i]  = M_p @ (ddq_d + Kd*deq + Kp*eq) + Cq_p + G_p - QF_meas
    return tau

def recompute_tau_mlp(q_arr, dq_arr):
    tau = np.zeros((N, 2))
    for i, t in enumerate(t_eval):
        tau[i] = mlp_predict(q_arr[i], dq_arr[i], cs(t,2), F_delayed(t))
    return tau

tau_ctc  = recompute_tau_ctc(q_ctc,   dq_ctc)
tau_pinn = recompute_tau_pinn(q_pinn,  dq_pinn)
tau_mlp  = recompute_tau_mlp(q_mlp,   dq_mlp)

# G4 Cartesian positions
G4_des  = np.array([fk_G4(cs(t))       for t in t_eval])
G4_ctc  = np.array([fk_G4(q_ctc[i])   for i in range(N)])
G4_pinn = np.array([fk_G4(q_pinn[i])  for i in range(N)])
G4_mlp  = np.array([fk_G4(q_mlp[i])   for i in range(N)])

# Tracking error (mm)
err_ctc  = np.linalg.norm(G4_ctc  - G4_des, axis=1) * 1000
err_pinn = np.linalg.norm(G4_pinn - G4_des, axis=1) * 1000
err_mlp  = np.linalg.norm(G4_mlp  - G4_des, axis=1) * 1000

rmse_ctc  = float(np.sqrt(np.mean(err_ctc **2)))
rmse_pinn = float(np.sqrt(np.mean(err_pinn**2)))
rmse_mlp  = float(np.sqrt(np.mean(err_mlp **2)))

F_mag = np.linalg.norm(F_log, axis=1)
F_ang = np.degrees(np.arctan2(F_log[:,1], F_log[:,0]))

# ── Results table ──────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  {'Controller':<10} {'RMSE (mm)':>12} {'max err (mm)':>14} {'τ₁ max':>10} {'τ₂ max':>10}")
print(f"  {'─'*60}")
for name, err, tau in [('CTC',  err_ctc,  tau_ctc),
                       ('PINN', err_pinn, tau_pinn),
                       ('MLP',  err_mlp,  tau_mlp)]:
    rmse_v = float(np.sqrt(np.mean(err**2)))
    print(f"  {name:<10} {rmse_v:>12.4f} {err.max():>14.4f} "
          f"{np.abs(tau[:,0]).max():>10.2f} {np.abs(tau[:,1]).max():>10.2f}  N·m")
print(f"{'═'*64}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — STATIC FIGURES (unchanged)
# ══════════════════════════════════════════════════════════════════════════
def style(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(C_GRID)
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.6)
    return ax

fig_s = plt.figure(figsize=(18, 12), facecolor=C_BG)
fig_s.suptitle(
    f"Controller Comparison — traj_amp2p5cm_9waves  |  "
    f"Kp={Kp:.0f}  Kd={Kd:.0f}  |F|∈[5,15]N  delay={FORCE_DELAY*1000:.0f}ms",
    color='white', fontsize=12, fontweight='bold')
gs_s = gridspec.GridSpec(3, 3, figure=fig_s,
                          hspace=0.48, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)

# ── (0,0) RMSE bar ────────────────────────────────────────────────────────
ax = style(fig_s.add_subplot(gs_s[0, 0]))
_names = ['CTC', 'PINN', 'MLP']
_rmses = [rmse_ctc, rmse_pinn, rmse_mlp]
_cols  = [C_CTC, C_PINN, C_MLP]
bars = ax.bar(_names, _rmses, color=_cols, edgecolor='#1e2230', linewidth=0.8)
for bar, val in zip(bars, _rmses):
    ax.text(bar.get_x()+bar.get_width()/2, val+max(_rmses)*0.015,
            f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9,
            fontweight='bold')
ax.set_ylabel('RMSE  (mm)', color=C_TEXT, fontsize=9)
ax.set_title('Tracking RMSE  (G₄)', color='white', fontsize=10, fontweight='bold')

# ── (0,1) G4 trajectory overlay ───────────────────────────────────────────
ax = style(fig_s.add_subplot(gs_s[0, 1]))
ax.plot(x_ee, y_ee, '--', color=C_DES, lw=1.0, alpha=0.4, label='Desired')
ax.plot(G4_ctc[:,0],  G4_ctc[:,1],  color=C_CTC,  lw=1.6, label='CTC')
ax.plot(G4_pinn[:,0], G4_pinn[:,1], color=C_PINN, lw=1.4, ls='--', label='PINN')
ax.plot(G4_mlp[:,0],  G4_mlp[:,1],  color=C_MLP,  lw=1.4, ls=':',  label='MLP')
ax.set_xlabel('x (m)', color=C_TEXT, fontsize=8)
ax.set_ylabel('y (m)', color=C_TEXT, fontsize=8)
ax.set_aspect('equal')
ax.set_title('G₄ Trajectory Overlay', color='white', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white')

# ── (0,2) Tracking error over time ────────────────────────────────────────
ax = style(fig_s.add_subplot(gs_s[0, 2]))
ax.plot(t_eval, err_ctc,  color=C_CTC,  lw=1.5, label=f'CTC  {rmse_ctc:.4f} mm')
ax.plot(t_eval, err_pinn, color=C_PINN, lw=1.5, ls='--', label=f'PINN {rmse_pinn:.4f} mm')
ax.plot(t_eval, err_mlp,  color=C_MLP,  lw=1.5, ls=':', label=f'MLP  {rmse_mlp:.4f} mm')
ax.set_xlabel('t (s)', color=C_TEXT, fontsize=8)
ax.set_ylabel('‖e‖ (mm)', color=C_TEXT, fontsize=8)
ax.set_title('Cartesian Tracking Error', color='white', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white')

# ── (1,:2) τ₁ full time series ────────────────────────────────────────────
ax = style(fig_s.add_subplot(gs_s[1, :2]))
ax.plot(t_eval, tau_ctc[:,0],  color=C_CTC,  lw=1.4, label='CTC')
ax.plot(t_eval, tau_pinn[:,0], color=C_PINN, lw=1.4, ls='--', label='PINN')
ax.plot(t_eval, tau_mlp[:,0],  color=C_MLP,  lw=1.4, ls=':', label='MLP')
ax.axhline(0, color=C_GRID, lw=0.7)
ax.set_xlabel('t (s)', color=C_TEXT, fontsize=8)
ax.set_ylabel('τ₁ (N·m)', color=C_TEXT, fontsize=8)
ax.set_title('τ₁ — Joint 1 Torque', color='white', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white', ncol=3)

# ── (1,2) Force F(t) ──────────────────────────────────────────────────────
ax  = style(fig_s.add_subplot(gs_s[1, 2]))
ax2 = ax.twinx()
ax.plot(t_eval,  F_mag, color=C_FORCE, lw=1.5, label='|F|')
ax.fill_between(t_eval, F_mag, alpha=0.12, color=C_FORCE)
ax2.plot(t_eval, F_ang, color='#c77dff', lw=1.2, ls='--', label='θ')
ax.axhline(5,  color=C_FORCE, lw=0.7, ls=':', alpha=0.5)
ax.axhline(15, color=C_FORCE, lw=0.7, ls=':', alpha=0.5)
ax.set_xlabel('t (s)', color=C_TEXT, fontsize=8)
ax.set_ylabel('|F(t)| (N)', color=C_TEXT, fontsize=8)
ax2.set_ylabel('θ (°)', color='#c77dff', fontsize=8)
ax2.tick_params(colors='#c77dff', labelsize=8)
ax.set_title('External Force F(t)', color='white', fontsize=10, fontweight='bold')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white')

# ── (2,:2) τ₂ full time series ────────────────────────────────────────────
ax = style(fig_s.add_subplot(gs_s[2, :2]))
ax.plot(t_eval, tau_ctc[:,1],  color=C_CTC,  lw=1.4, label='CTC')
ax.plot(t_eval, tau_pinn[:,1], color=C_PINN, lw=1.4, ls='--', label='PINN')
ax.plot(t_eval, tau_mlp[:,1],  color=C_MLP,  lw=1.4, ls=':', label='MLP')
ax.axhline(0, color=C_GRID, lw=0.7)
ax.set_xlabel('t (s)', color=C_TEXT, fontsize=8)
ax.set_ylabel('τ₂ (N·m)', color=C_TEXT, fontsize=8)
ax.set_title('τ₂ — Joint 2 Torque', color='white', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white', ncol=3)

# ── (2,2) PINN parameter recovery ─────────────────────────────────────────
ax = style(fig_s.add_subplot(gs_s[2, 2]))
th_true = pinn_data.get('theta_true', {})
if th_true:
    p_names = ['K1', 'K2', 'M11_0', 'meff1', 'meff2']
    p_errs  = [abs(th_p[k] - th_true[k]) / (abs(th_true[k]) + 1e-8) * 100
               for k in p_names]
    bar_c   = ['#2ca02c' if e<2 else '#ff7f0e' if e<5 else '#d62728'
               for e in p_errs]
    ax.barh(p_names, p_errs, color=bar_c, edgecolor='#1e2230', linewidth=0.6)
    for j, (pn, pe) in enumerate(zip(p_names, p_errs)):
        ax.text(pe+0.05, j, f'{pe:.2f}%', va='center', color='white', fontsize=8)
    ax.axvline(2, color='#2ca02c', ls='--', lw=1.1, alpha=0.8, label='<2% ok')
    ax.axvline(5, color='#ff7f0e', ls='--', lw=1.1, alpha=0.8, label='<5% !!')
    ax.set_xlabel('Recovery error (%)', color=C_TEXT, fontsize=8)
    ax.legend(fontsize=7.5, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white')
ax.set_title('PINN Parameter Recovery', color='white', fontsize=10, fontweight='bold')

plt.savefig("figures/validation_static.png", dpi=150, bbox_inches='tight')
print("\n  Static figure → figures/validation_static.png")
plt.show(block=False)
plt.pause(0.5)

# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — ANIMATION WITH FULL ROBOT STRUCTURE
# ══════════════════════════════════════════════════════════════════════════
# Full forward kinematics (all points of the 4‑body robot)
def fk_all(q):
    q1, q2 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    def rot(xb, yb):
        return np.array([c1*xb - s1*yb, s1*xb + c1*yb])
    return {
        'O'  : np.array([0.0, 0.0]),
        'O2' : rot(a,           -e_link/2),
        'O3' : rot(a,           +e_link/2),
        'G1' : rot(a,            0.0),
        'G2' : rot(a + b*c2,    -e_link/2 + b*s2),
        'G3' : rot(a + b*c2,    +e_link/2 + b*s2),
        'A'  : rot(a + L*c2,    -e_link/2 + L*s2),
        'B'  : rot(a + L*c2,    +e_link/2 + L*s2),
        'G4' : rot(a + L*c2 + h4,       L*s2),
        'Bp' : rot(xB, yB),
    }

def _perp(p1, p2, length=0.018):
    """Unit vector perpendicular to (p2-p1), scaled to `length`."""
    d  = p2 - p1
    n  = np.array([-d[1], d[0]])
    n /= (np.linalg.norm(n) + 1e-12)
    return n * length

# Layout: left = robot mechanism (large), right = scrolling subplots
fig_a = plt.figure(figsize=(18, 9), facecolor=C_BG)
fig_a.suptitle(
    "Closed‑loop simulation — CTC  (full robot)  vs  PINN  vs  MLP",
    color='white', fontsize=12, fontweight='bold', y=0.98)

gs_a = gridspec.GridSpec(4, 2, figure=fig_a,
                          width_ratios=[2.2, 1.0],
                          hspace=0.42, wspace=0.28,
                          left=0.05, right=0.97, top=0.94, bottom=0.07)

ax_r   = fig_a.add_subplot(gs_a[:, 0])    # robot (full height)
ax_t1  = fig_a.add_subplot(gs_a[0, 1])   # τ₁
ax_t2  = fig_a.add_subplot(gs_a[1, 1])   # τ₂
ax_f   = fig_a.add_subplot(gs_a[2, 1])   # |F|
ax_err = fig_a.add_subplot(gs_a[3, 1])   # error

for _ax in (ax_r, ax_t1, ax_t2, ax_f, ax_err):
    _ax.set_facecolor(C_PANEL)
    _ax.tick_params(colors=C_TEXT, labelsize=7.5)
    for sp in _ax.spines.values(): sp.set_color(C_GRID)
    _ax.grid(True, color=C_GRID, lw=0.5, alpha=0.5)

# ── Robot axis ─────────────────────────────────────────────────────────────
all_G4 = np.vstack([G4_ctc, G4_pinn, G4_mlp])
pad = 0.13
ax_r.set_xlim(np.nanmin(all_G4[:,0])-pad, np.nanmax(all_G4[:,0])+pad)
ax_r.set_ylim(np.nanmin(all_G4[:,1])-pad, np.nanmax(all_G4[:,1])+pad)
ax_r.set_aspect('equal')
ax_r.set_xlabel('x  (m)', color=C_TEXT, fontsize=9)
ax_r.set_ylabel('y  (m)', color=C_TEXT, fontsize=9)
ax_r.set_title(
    f'Kp={Kp:.0f}  Kd={Kd:.0f}  '
    f'|F(t)| ∈ [5,15] N  θ ∈ [-90°,+30°]  delay={FORCE_DELAY*1000:.0f} ms',
    color=C_TEXT, fontsize=8.5)

# Desired path
ax_r.plot(x_ee, y_ee, '--', color=C_DES, lw=0.9, alpha=0.35, zorder=1)

# Ground symbol
ax_r.plot(0, 0, 's', color='#606070', ms=8, zorder=5)
ax_r.plot([-0.03,0.03], [-0.014,-0.014], color='#606070', lw=2)
for dx_g in np.linspace(-0.025, 0.025, 5):
    ax_r.plot([dx_g, dx_g-0.01], [-0.014,-0.024], color='#606070', lw=1)

# Legend
ax_r.legend(handles=[
    Line2D([0],[0], color=C_CTC,  lw=2.2,            label=f'CTC   (RMSE={rmse_ctc:.4f} mm)'),
    Line2D([0],[0], color=C_PINN, lw=2.0, ls='--',   label=f'PINN  (RMSE={rmse_pinn:.4f} mm)'),
    Line2D([0],[0], color=C_MLP,  lw=2.0, ls=':',    label=f'MLP   (RMSE={rmse_mlp:.4f} mm)'),
    Line2D([0],[0], color=C_DES,  lw=1.0, ls='--',   label='Desired path'),
    Line2D([0],[0], color=C_FORCE, lw=2.0,            label='F(t) sensor'),
], fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor='white', loc='upper left')

# ── Mechanism artists (full 4‑body robot, drawn from CTC state) ──────────
_LK = dict(solid_capstyle='round', solid_joinstyle='round', zorder=3)

# ① Body 1 (white triangle)
ln_body1,   = ax_r.plot([], [], color='#e8e8f0', lw=2.5, **_LK)
ln_O3_Bp,   = ax_r.plot([], [], color='#e8e8f0', lw=2.0, **_LK)
ln_j_Bp,    = ax_r.plot([], [], color='#e8e8f0', lw=2.0, **_LK)

# ② Parallel links (teal)
ln_lower,   = ax_r.plot([], [], color='#5bc8af', lw=2.5, **_LK)   # O₂ → A
ln_upper,   = ax_r.plot([], [], color='#5bc8af', lw=2.5, **_LK)   # O₃ → B

# ③ Coupler (orange) — H‑bracket + EF segment
ln_coup_vert,  = ax_r.plot([], [], color='#f4a261', lw=2.5, **_LK)
ln_coup_armA,  = ax_r.plot([], [], color='#f4a261', lw=2.0, **_LK)
ln_coup_armB,  = ax_r.plot([], [], color='#f4a261', lw=2.0, **_LK)
ln_midAB_G4,   = ax_r.plot([], [], color='#f4a261', lw=2.0, **_LK)
ln_EF,         = ax_r.plot([], [], color='#f4a261', lw=2.0, **_LK)
ln_E_perp,     = ax_r.plot([], [], color='#f4a261', lw=1.8, **_LK)
ln_F_perp,     = ax_r.plot([], [], color='#f4a261', lw=1.8, **_LK)

# ④ Prismatic constraint (coral dashed)
ln_pris,    = ax_r.plot([], [], color=C_PRIS, lw=2.0, ls='--',
                         zorder=3, dash_capstyle='round')

# Joint circles
pt_joints,  = ax_r.plot([], [], 'o', color='white', ms=5, zorder=6,
                         markeredgecolor='#333', markeredgewidth=0.7)

# G₄ markers for three controllers
pt_ctc,  = ax_r.plot([], [], '*', color=C_CTC,  ms=14, zorder=9,
                      markeredgecolor='#1a3a5c', markeredgewidth=0.8)
pt_pinn, = ax_r.plot([], [], 'o', color=C_PINN, ms=10, zorder=9,
                      markeredgecolor='#0d4030', markeredgewidth=0.8)
pt_mlp,  = ax_r.plot([], [], 'D', color=C_MLP,  ms=8,  zorder=9,
                      markeredgecolor='#5c3010', markeredgewidth=0.8)
pt_des,  = ax_r.plot([], [], '+', color=C_DES,  ms=9, mew=1.5, zorder=7, alpha=0.7)

# Bp fixed point (prismatic base)
pt_Bp,      = ax_r.plot([], [], 's', color=C_PRIS, ms=6, zorder=6)

# Trails for G₄
tr = {k: ([], []) for k in ('ctc','pinn','mlp')}
ln_tr_ctc,  = ax_r.plot([], [], '-', color=C_CTC,  lw=1.6, alpha=0.55, zorder=4)
ln_tr_pinn, = ax_r.plot([], [], '-', color=C_PINN, lw=1.4, alpha=0.55, zorder=4)
ln_tr_mlp,  = ax_r.plot([], [], '-', color=C_MLP,  lw=1.4, alpha=0.55, zorder=4)

# Force arrow at CTC G₄
arrow_F = ax_r.annotate('', xy=(0,0), xytext=(0,0),
                         arrowprops=dict(arrowstyle='->', color=C_FORCE,
                                         lw=2.0, mutation_scale=14), zorder=10)
F_SCALE  = 0.012
lbl_F    = ax_r.text(0.02, 0.04, '', transform=ax_r.transAxes,
                     color=C_FORCE, fontsize=8, fontfamily='monospace')
t_txt    = ax_r.text(0.02, 0.08, '', transform=ax_r.transAxes,
                     color='white', fontsize=9, fontfamily='monospace')

# ── Node labels ───────────────────────────────────────────────────────────
_LBL_KW = dict(fontsize=7.5, color='#d0d8e8', zorder=9,
               fontfamily='sans-serif',
               bbox=dict(boxstyle='round,pad=0.15', fc=C_BG, alpha=0.75, ec='none'))
lbl_O2  = ax_r.text(0, 0, ' O₂', **_LBL_KW)
lbl_O3  = ax_r.text(0, 0, ' O₃', **_LBL_KW)
lbl_G1  = ax_r.text(0, 0, ' G₁', **_LBL_KW)
lbl_G2  = ax_r.text(0, 0, ' G₂', **_LBL_KW)
lbl_G3  = ax_r.text(0, 0, ' G₃', **_LBL_KW)
lbl_G4l = ax_r.text(0, 0, ' G₄', color=C_TRAIL,
                    fontsize=8, fontweight='bold', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc=C_BG, alpha=0.75, ec='none'))
lbl_A   = ax_r.text(0, 0, ' A',  **_LBL_KW)
lbl_B   = ax_r.text(0, 0, ' B',  **_LBL_KW)
lbl_Lp  = ax_r.text(0, 0, ' Lp', color=C_PRIS, fontsize=7.5, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc=C_BG, alpha=0.75, ec='none'))

# ── Time‑series axes (right side) ─────────────────────────────────────────
tau_lim = (max(np.abs(tau_ctc).max(), np.abs(tau_pinn).max(),
               np.abs(tau_mlp).max()) * 1.25)
err_lim = max(err_ctc.max(), err_pinn.max(), err_mlp.max()) * 1.35 + 0.001

for _ax, _title in [(ax_t1, 'τ₁  (N·m)'),
                    (ax_t2, 'τ₂  (N·m)')]:
    _ax.set_xlim(0, t_ik[-1]);  _ax.set_ylim(-tau_lim, tau_lim)
    _ax.axhline(0, color=C_GRID, lw=0.7)
    _ax.set_title(_title, color='white', fontsize=8.5, fontweight='bold')

ax_f.set_xlim(0, t_ik[-1]);   ax_f.set_ylim(0, 17)
ax_f.set_title('|F(t)|  (N)', color='white', fontsize=8.5, fontweight='bold')
ax_err.set_xlim(0, t_ik[-1]); ax_err.set_ylim(0, err_lim)
ax_err.set_title('‖e‖  (mm)', color='white', fontsize=8.5, fontweight='bold')
ax_err.set_xlabel('t  (s)', color=C_TEXT, fontsize=8)

_so = dict(lw=1.4)
ln_t1_ctc,  = ax_t1.plot([], [], color=C_CTC,  **_so)
ln_t1_pinn, = ax_t1.plot([], [], color=C_PINN, ls='--', **_so)
ln_t1_mlp,  = ax_t1.plot([], [], color=C_MLP,  ls=':',  lw=1.7)
ln_t2_ctc,  = ax_t2.plot([], [], color=C_CTC,  **_so)
ln_t2_pinn, = ax_t2.plot([], [], color=C_PINN, ls='--', **_so)
ln_t2_mlp,  = ax_t2.plot([], [], color=C_MLP,  ls=':',  lw=1.7)
ln_fmag,    = ax_f.plot([],  [], color=C_FORCE, lw=1.5)
ln_e_ctc,   = ax_err.plot([], [], color=C_CTC,  **_so)
ln_e_pinn,  = ax_err.plot([], [], color=C_PINN, ls='--', **_so)
ln_e_mlp,   = ax_err.plot([], [], color=C_MLP,  ls=':',  lw=1.7)

tl_t1  = ax_t1.axvline(0,  color='white', lw=0.8, alpha=0.35)
tl_t2  = ax_t2.axvline(0,  color='white', lw=0.8, alpha=0.35)
tl_f   = ax_f.axvline(0,   color='white', lw=0.8, alpha=0.35)
tl_err = ax_err.axvline(0, color='white', lw=0.8, alpha=0.35)

# ── Animation update function ─────────────────────────────────────────────
def update(frame):
    i   = frame % N
    t   = t_eval[i]
    pts = fk_all(q_ctc[i])      # CTC state defines the full robot structure

    # ① Body 1 — white triangle O‑O₂‑O₃
    ln_body1.set_data(
        [pts['O'][0],  pts['O2'][0], pts['O3'][0], pts['O'][0]],
        [pts['O'][1],  pts['O2'][1], pts['O3'][1], pts['O'][1]])
    ln_O3_Bp.set_data([pts['O3'][0], pts['Bp'][0]], [pts['O3'][1], pts['Bp'][1]])
    j_pt = 0.5 * (pts['O'] + pts['O3'])
    ln_j_Bp.set_data([j_pt[0], pts['Bp'][0]], [j_pt[1], pts['Bp'][1]])

    # ② Parallel links (teal)
    ln_lower.set_data([pts['O2'][0], pts['A'][0]], [pts['O2'][1], pts['A'][1]])
    ln_upper.set_data([pts['O3'][0], pts['B'][0]], [pts['O3'][1], pts['B'][1]])

    # ③ Coupler H‑bracket (orange)
    A, B, G4 = pts['A'], pts['B'], pts['G4']
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
    F_pt   = G4 - half_L * u          # rename to F_pt (avoid clash with F_log)
    ln_EF.set_data([E[0], F_pt[0]], [E[1], F_pt[1]])
    perp = np.array([-u[1], u[0]])
    ln_E_perp.set_data([E[0]    - L_EF * perp[0], E[0]],
                       [E[1]    - L_EF * perp[1], E[1]])
    ln_F_perp.set_data([F_pt[0] - L_EF * perp[0], F_pt[0]],
                       [F_pt[1] - L_EF * perp[1], F_pt[1]])

    # ④ Prismatic constraint (coral dashed)
    ln_pris.set_data([pts['Bp'][0], pts['G3'][0]], [pts['Bp'][1], pts['G3'][1]])
    lbl_Lp.set_position(((pts['Bp'][0]+pts['G3'][0])/2 + 0.01,
                          (pts['Bp'][1]+pts['G3'][1])/2))

    # Joint circles
    jx = [pts['O2'][0], pts['O3'][0], A[0], B[0], pts['G2'][0], pts['G3'][0]]
    jy = [pts['O2'][1], pts['O3'][1], A[1], B[1], pts['G2'][1], pts['G3'][1]]
    pt_joints.set_data(jx, jy)
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

    # G₄ markers (overlay from all three controllers)
    pt_ctc.set_data( [G4_ctc[i,0]],  [G4_ctc[i,1]])
    pt_pinn.set_data([G4_pinn[i,0]], [G4_pinn[i,1]])
    pt_mlp.set_data( [G4_mlp[i,0]],  [G4_mlp[i,1]])
    pt_des.set_data( [G4_des[i,0]],  [G4_des[i,1]])

    # G₄ trails
    for key, G4_arr in [('ctc', G4_ctc), ('pinn', G4_pinn), ('mlp', G4_mlp)]:
        tx, ty = tr[key]
        tx.append(G4_arr[i,0]); ty.append(G4_arr[i,1])
        if len(tx) > TRAIL_LEN: tx.pop(0); ty.pop(0)
    ln_tr_ctc.set_data(*tr['ctc'])
    ln_tr_pinn.set_data(*tr['pinn'])
    ln_tr_mlp.set_data(*tr['mlp'])

    # Force arrow at CTC G₄
    Fi   = F_log[i]
    Fmag = np.linalg.norm(Fi)
    Fang = np.degrees(np.arctan2(Fi[1], Fi[0]))
    ftip = pts['G4'] + F_SCALE * Fi
    arrow_F.set_position(ftip); arrow_F.xy = pts['G4']
    lbl_F.set_text(f'F=({Fi[0]:+.1f}, {Fi[1]:+.1f}) N  '
                   f'|F|={Fmag:.1f} N  θ={Fang:.0f}°')
    t_txt.set_text(f't = {t:.2f} s   [{100*t/t_ik[-1]:.0f}%]')

    # Update time‑series plots
    sl = slice(0, i+1)
    tv = t_eval[sl]
    ln_t1_ctc.set_data(tv, tau_ctc[sl,0]);   ln_t1_pinn.set_data(tv, tau_pinn[sl,0])
    ln_t1_mlp.set_data(tv, tau_mlp[sl,0])
    ln_t2_ctc.set_data(tv, tau_ctc[sl,1]);   ln_t2_pinn.set_data(tv, tau_pinn[sl,1])
    ln_t2_mlp.set_data(tv, tau_mlp[sl,1])
    ln_fmag.set_data(tv, F_mag[sl])
    ln_e_ctc.set_data(tv,  err_ctc[sl]);     ln_e_pinn.set_data(tv, err_pinn[sl])
    ln_e_mlp.set_data(tv,  err_mlp[sl])
    for tl in (tl_t1, tl_t2, tl_f, tl_err):
        tl.set_xdata([t, t])

    return (ln_body1, ln_O3_Bp, ln_j_Bp,
            ln_lower, ln_upper,
            ln_coup_vert, ln_coup_armA, ln_coup_armB, ln_midAB_G4,
            ln_EF, ln_E_perp, ln_F_perp,
            ln_pris,
            pt_joints, pt_Bp,
            pt_ctc, pt_pinn, pt_mlp, pt_des,
            ln_tr_ctc, ln_tr_pinn, ln_tr_mlp,
            lbl_O2, lbl_O3, lbl_G1, lbl_G2, lbl_G3, lbl_G4l, lbl_A, lbl_B, lbl_Lp,
            arrow_F, lbl_F, t_txt,
            ln_t1_ctc, ln_t1_pinn, ln_t1_mlp,
            ln_t2_ctc, ln_t2_pinn, ln_t2_mlp,
            ln_fmag,
            ln_e_ctc, ln_e_pinn, ln_e_mlp,
            tl_t1, tl_t2, tl_f, tl_err)

anim = FuncAnimation(fig_a, update,
                     frames=itertools.cycle(range(N)),
                     interval=INTERVAL,
                     blit=True,
                     cache_frame_data=False)

plt.tight_layout(pad=1.2)
print("Showing animation with full 4‑body robot structure — close window to exit.\n")
plt.show()