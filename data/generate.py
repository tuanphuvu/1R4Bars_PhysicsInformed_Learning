# data/generate.py
# Generate dataset from 8 sinusoidal reference trajectories in workspace via CTC simulation.
#
# Input  : trajectories_ref/*.csv   (from analysis/generate_trajectories.py)
# Output : data/saved/train.npz, data/saved/ood_test.npz
#
# Pipeline per trajectory:
#   CSV (x_ee, y_ee, arc) → t_phys = arc / EE_SPEED
#   → IK warm-chain       → q_des(t)
#   → CubicSpline         → q̇_d(t), q̈_d(t)
#   → F_func(t)           → time-varying external force  (random C² profile)
#   → CTC ODE (RK45)      → q_sim, dq_sim   at uniform DT
#                            · actual dynamics  : F_func(t)          (true force)
#                            · control law      : F_func(t−delay)    (delayed measure)
#   → recompute τ         → ddq from spline of q_sim
#   → save (q, dq, ddq, τ, G4_sim, G4_des, F_log)
#
# ── Force model ────────────────────────────────────────────────────────────
#   |F(t)| ∈ [5, 15] N     angle ∈ [−90°, +30°]
#   Exponential segment widths → rate of change varies (slow ↔ fast)
#   CubicSpline → C² continuous (smooth, no kinks)
#   FORCE_DELAY = Force_sensor_latency   (imported from dynamics.params)
# ──────────────────────────────────────────────────────────────────────────

import numpy as np
import os
from scipy.integrate   import solve_ivp
from scipy.interpolate import CubicSpline

from dynamics.matrices  import (mass_matrix, coriolis_qdot, gravity_vector,
                                 inverse_dynamics, fk_G4, qf_external)
from dynamics.params    import Force_sensor_latency, create_time_varying_force
from kinematics.ik_solver import ik_solve, _q_init_analytical

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
EE_SPEED    = 0.05    # m/s — arc_length / EE_SPEED = physical time
Kp, Kd      = 400.0, 40.0
DT          = 0.01    # s — uniform sample interval in the final dataset
TRAJ_DIR    = "trajectories_ref"

# Force parameters — mirror simulate_sinusoidal.py exactly
FORCE_DELAY     = Force_sensor_latency   # s  (controller sees delayed force)
FORCE_MAG_RANGE = (5.0, 15.0)           # N
FORCE_ANG_RANGE = (-90.0, 30.0)         # degrees
FORCE_N_SEGS    = 30                    # number of random segments

# ── Train/OOD split by amplitude ──────────────────────────────────────────
TRAIN_TRAJS = [
    "traj_amp2p0cm_4waves.csv",
    "traj_amp2p0cm_5waves.csv",
    "traj_amp2p0cm_6waves.csv",
    "traj_amp2p0cm_7waves.csv",
]

OOD_TRAJS = [
    "traj_amp2p5cm_4waves.csv",
    "traj_amp2p5cm_5waves.csv",
    "traj_amp2p5cm_6waves.csv",
    "traj_amp2p5cm_7waves.csv",
]

# ══════════════════════════════════════════════════════════════════════════
# CSV LOADER
# ══════════════════════════════════════════════════════════════════════════
def load_trajectory(filepath):
    """
    Parse a trajectories_ref CSV.

    Returns
    -------
    t_param : (N,)   spline parameter [0, 1]
    x_ee    : (N,)   end-effector x   [m]
    y_ee    : (N,)   end-effector y   [m]
    arc_m   : (N,)   cumulative arc length [m]
    meta    : dict   header key-value pairs
    """
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

# ══════════════════════════════════════════════════════════════════════════
# SIMULATION  (one trajectory)
# ══════════════════════════════════════════════════════════════════════════
def simulate_from_trajectory(traj_filename, ee_speed=EE_SPEED, dt=DT):
    """
    Full CTC simulation for one sinusoidal reference trajectory
    with time-varying external force and sensor latency.

    Control law (CTC):
        τ  = M(q)·[q̈_d + Kd·ė + Kp·e] + C(q,q̇)·q̇ + g(q) − QF_meas
    where:
        QF_actual = qf_external(q, F_func(t))           — true force on robot
        QF_meas   = qf_external(q, F_func(t − delay))   — delayed sensor reading

    Actual dynamics:
        M·q̈ = τ + QF_actual − C·q̇ − g

    Steps
    -----
    1.  Load (x_ee, y_ee, arc_m) from CSV
    2.  t_phys  = arc_m / ee_speed         (constant Cartesian speed)
    3.  IK warm-chain: (x,y) → q_des(t),  skip IK failures
    4.  CubicSpline  q_d(t)  →  q̇_d, q̈_d
    5.  Create time-varying F_func(t) for this trajectory's time range
    6.  RK45 integration of CTC ODE (delayed force in controller)
    7.  Resample at uniform dt
    8.  Cache F_log at t_eval (true force, no delay)
    9.  Recompute τ at each sample
    10. CubicSpline of q_sim  → ddq_sim
    11. G4_sim = fk_G4(q_sim),  G4_des = fk_G4(q_d)

    Returns
    -------
    t_eval  : (N,)
    q_sim   : (N, 2)
    dq_sim  : (N, 2)
    ddq_sim : (N, 2)
    tau_sim : (N, 2)
    G4_sim  : (N, 2)
    G4_des  : (N, 2)
    F_log : (N, 2)   delayed force reading F(t − delay) — matches sensor input to controller
    """
    traj_path = os.path.join(TRAJ_DIR, traj_filename)
    print(f"\n  [{traj_filename}]")

    # ── 1. Load CSV ───────────────────────────────────────────────────────
    t_param, x_ee, y_ee, arc_m, meta = load_trajectory(traj_path)
    amp_tag  = meta.get('amplitude_cm', '?')
    wave_tag = meta.get('n_waves',      '?')
    print(f"     A={amp_tag} cm   n={wave_tag} waves   "
          f"arc={arc_m[-1]*100:.1f} cm   T={arc_m[-1]/ee_speed:.2f} s")

    # ── 2. Physical time ──────────────────────────────────────────────────
    t_phys = arc_m / ee_speed

    # ── 3. IK warm-chain ─────────────────────────────────────────────────
    q_des_list, t_ik_list = [], []
    q_prev = _q_init_analytical(np.array([x_ee[0], y_ee[0]]))

    for xi, yi, ti in zip(x_ee, y_ee, t_phys):
        q_sol, ok = ik_solve(np.array([xi, yi]), q_prev,
                             max_iter=300, tol=1e-7)
        if ok:
            q_des_list.append(q_sol)
            t_ik_list.append(ti)
            q_prev = q_sol

    q_des    = np.array(q_des_list)
    t_ik     = np.array(t_ik_list)
    ik_rate  = 100.0 * len(q_des) / len(x_ee)
    print(f"     IK: {len(q_des)}/{len(x_ee)} solved ({ik_rate:.0f}%)")

    if len(q_des) < 20:
        raise RuntimeError(
            f"Only {len(q_des)} IK solutions — check robot params / trajectory.")

    # ── 4. Desired trajectory spline ──────────────────────────────────────
    cs = CubicSpline(t_ik, q_des)   # gives q_d(t), q̇_d(t), q̈_d(t)

    # ── 5. Time-varying external force ────────────────────────────────────
    # One independent random profile per trajectory.
    # F_func  : t → [Fx, Fy]  (true force, no delay)
    # F_delayed: F(t − FORCE_DELAY), clamped at t=0
    F_func = create_time_varying_force(
        t_ik[0], t_ik[-1],
        mag_range=FORCE_MAG_RANGE,
        angle_deg_range=FORCE_ANG_RANGE,
        n_segs=FORCE_N_SEGS,
    )
    print(f"     F(t): |F| ∈ [{FORCE_MAG_RANGE[0]}, {FORCE_MAG_RANGE[1]}] N   "
          f"θ ∈ [{FORCE_ANG_RANGE[0]}°, {FORCE_ANG_RANGE[1]}°]   "
          f"delay = {FORCE_DELAY*1000:.1f} ms")

    def F_delayed(t):
        """Controller's force reading: F evaluated at (t − FORCE_DELAY)."""
        t_eff = max(t - FORCE_DELAY, 0.0)
        return F_func(t_eff)

    # ── 6. CTC ODE ────────────────────────────────────────────────────────
    def ode_ctc(t, state):
        q, dq  = state[:2], state[2:]
        q_d    = cs(t);   dq_d  = cs(t, 1);   ddq_d = cs(t, 2)
        e_q    = q_d - q; de_q  = dq_d - dq
        M      = mass_matrix(q[1])
        Cq     = coriolis_qdot(q[1], dq)
        G      = gravity_vector(q)

        # Actual force acting on the robot right now
        QF_actual = qf_external(q, F_func(t))
        # What the controller *thinks* the force is (delayed measurement)
        QF_meas   = qf_external(q, F_delayed(t))

        tau  = M @ (ddq_d + Kd*de_q + Kp*e_q) + Cq + G - QF_meas
        ddq  = np.linalg.solve(M, tau + QF_actual - Cq - G)
        return np.concatenate([dq, ddq])

    t_eval = np.arange(t_ik[0], t_ik[-1], dt)

    sol = solve_ivp(
        ode_ctc,
        [t_ik[0], t_ik[-1]],
        np.concatenate([q_des[0], cs(t_ik[0], 1)]),
        t_eval=t_eval,
        method='RK45', rtol=1e-8, atol=1e-10,
    )
    q_sim  = sol.y[:2].T    # (N, 2)
    dq_sim = sol.y[2:].T    # (N, 2)

    # ── 7. Cache F_log at uniform samples (delayed — what sensor reports) ──
    F_log = np.array([F_func(max(t - FORCE_DELAY, 0.0)) for t in t_eval])

    # ── 8. Recompute τ at uniform samples ─────────────────────────────────
    # Controller uses *delayed* force; we reconstruct it from F_log.
    tau_sim = np.zeros((len(t_eval), 2))
    for i, t in enumerate(t_eval):
        q, dq  = q_sim[i], dq_sim[i]
        q_d    = cs(t);   dq_d  = cs(t, 1);   ddq_d = cs(t, 2)
        e_q    = q_d - q; de_q  = dq_d - dq
        M      = mass_matrix(q[1])
        Cq     = coriolis_qdot(q[1], dq)
        G      = gravity_vector(q)

        # Reconstruct the delayed force index for this sample
        t_eff  = max(t - FORCE_DELAY, 0.0)
        F_meas = F_func(t_eff)                        # delayed measurement
        QF_meas = qf_external(q, F_meas)

        tau_sim[i] = M @ (ddq_d + Kd*de_q + Kp*e_q) + Cq + G - QF_meas

    # ── 9. ddq from spline of q_sim ───────────────────────────────────────
    cs_sim  = CubicSpline(t_eval, q_sim)
    ddq_sim = cs_sim(t_eval, 2)

    # ── 10. G4 positions ──────────────────────────────────────────────────
    G4_sim = np.array([fk_G4(q_sim[i])      for i in range(len(t_eval))])
    G4_des = np.array([fk_G4(cs(t_eval[i])) for i in range(len(t_eval))])

    err_mm = np.linalg.norm(G4_sim - G4_des, axis=1) * 1000
    flag   = "[ok]" if err_mm.max() < 1.0 else "[!!]"
    print(f"     {flag} tracking: max={err_mm.max():.4f} mm   "
          f"mean={err_mm.mean():.4f} mm   N={len(t_eval)} samples")

    return t_eval, q_sim, dq_sim, ddq_sim, tau_sim, G4_sim, G4_des, F_log

# ══════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════
def build_dataset(traj_filenames, ee_speed=EE_SPEED, dt=DT):
    """
    Run CTC simulation for each file in traj_filenames, concatenate results.

    Returns dict with keys: q, dq, ddq, tau, G4_sim, G4_des, F_log
    """
    all_q, all_dq, all_ddq, all_tau = [], [], [], []
    all_G4_sim, all_G4_des, all_F = [], [], []
    completed = []

    for fname in traj_filenames:
        try:
            _, q, dq, ddq, tau, G4_sim, G4_des, F_log = simulate_from_trajectory(
                fname, ee_speed=ee_speed, dt=dt)
        except Exception as exc:
            print(f"     'x'  SKIP {fname} — {exc}")
            continue

        all_q.append(q);        all_dq.append(dq)
        all_ddq.append(ddq);    all_tau.append(tau)
        all_G4_sim.append(G4_sim); all_G4_des.append(G4_des)
        all_F.append(F_log)
        completed.append(fname)

    if not all_q:
        raise RuntimeError("No trajectory simulated successfully.")

    print(f"\n  Completed: {completed}")
    return {
        'q'     : np.vstack(all_q),
        'dq'    : np.vstack(all_dq),
        'ddq'   : np.vstack(all_ddq),
        'tau'   : np.vstack(all_tau),
        'G4_sim': np.vstack(all_G4_sim),
        'G4_des': np.vstack(all_G4_des),
        'F_log' : np.vstack(all_F),
    }

# ══════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════
def verify_dataset(data, name):
    """
    Physics consistency check:
        inverse_dynamics(q, dq, ddq, F)  ≈  saved τ

    With time-varying force, each sample uses its own F_log[i].
    In a well-behaved simulation the residual is dominated by the PD
    correction term Kp·e + Kd·ė (≈ 0 when tracking is good), so RMSE
    should be well below 2 N·m.  A large RMSE indicates a pipeline bug.
    """
    N       = len(data['q'])
    N_check = min(500, N)
    idx     = np.random.default_rng(0).choice(N, N_check, replace=False)

    tau_check = np.array([
        inverse_dynamics(data['q'][i], data['dq'][i], data['ddq'][i],
                         data['F_log'][i])
        for i in idx
    ])
    rmse    = np.sqrt(np.mean((tau_check - data['tau'][idx]) ** 2))
    tau_abs = np.abs(data['tau'])
    F_log   = data['F_log']

    print(f"\n  ── {name} sanity ──────────────────────────────────")
    print(f"  Samples       : {N:,}")
    print(f"  Sanity RMSE   : {rmse:.4f} N·m   (expect < 2.0)")
    print(f"  |τ₁| max      : {tau_abs[:, 0].max():.2f} N·m")
    print(f"  |τ₂| max      : {tau_abs[:, 1].max():.2f} N·m")
    print(f"  q₁ range      : [{np.degrees(data['q'][:, 0].min()):+.1f}°, "
          f"{np.degrees(data['q'][:, 0].max()):+.1f}°]")
    print(f"  q₂ range      : [{np.degrees(data['q'][:, 1].min()):+.1f}°, "
          f"{np.degrees(data['q'][:, 1].max()):+.1f}°]")
    print(f"  |dq| max      : {np.abs(data['dq']).max():.3f} rad/s")
    print(f"  |F| range     : [{np.linalg.norm(F_log, axis=1).min():.2f}, "
          f"{np.linalg.norm(F_log, axis=1).max():.2f}] N")
    G4_err  = np.linalg.norm(data['G4_sim'] - data['G4_des'], axis=1) * 1000
    print(f"  G4 err max    : {G4_err.max():.4f} mm")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("data/saved", exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    print("=" * 64)
    print(" TRAIN dataset  (A = 2.0 cm,  waves = 4 / 5 / 6 / 7)")
    print("=" * 64)
    train = build_dataset(TRAIN_TRAJS)
    np.savez("data/saved/train.npz", **train)
    verify_dataset(train, "TRAIN")
    print(f"\n  → Saved: data/saved/train.npz")

    # ── OOD test ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(" OOD TEST dataset  (A = 2.5 cm,  unseen amplitude)")
    print("=" * 64)
    ood = build_dataset(OOD_TRAJS)
    np.savez("data/saved/ood_test.npz", **ood)
    verify_dataset(ood, "OOD TEST")
    print(f"\n  → Saved: data/saved/ood_test.npz")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(" Dataset complete.")
    print(" Keys  : q, dq, ddq, tau, G4_sim, G4_des, F_log")
    print(f" EE_SPEED={EE_SPEED} m/s   DT={DT} s   Kp={Kp}   Kd={Kd}")
    print(f" Force delay = {FORCE_DELAY*1000:.1f} ms   "
          f"|F| ∈ {FORCE_MAG_RANGE} N   θ ∈ {FORCE_ANG_RANGE}°")
    print("=" * 64)