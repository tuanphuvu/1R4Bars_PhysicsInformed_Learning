#Physics_informed_model.py

# ┌─────────────────────────────────────────────────────────────────────┐
# │  PHYSICS-INFORMED PARAMETER IDENTIFICATION                          │
# │                                                                     │
# │  What is learned (θ = 5 parameters):                               │
# │    K1, K2, M11_0   — composite inertia constants (mass matrix)     │
# │    meff1, meff2    — effective mass×length products (gravity)       │
# │                                                                     │
# │    PINN knows HOW F enters the equations (via Jacobian JᵀF);       │
# │    MLP treats F as just another feature without structure.          │
# │                                                                     │
# │  What is fixed (geometry, measurable with ruler):                  │
# │    a, b, L, e, h4, g                                               │
# │                                                                     │
# │    Dynamics equations only depend on their composite combinations.  │
# │    Even with perfect infinite data you cannot recover m1, ms, m4   │
# │    separately — only K1, K2, M11_0, meff1, meff2.                  │
# │    This is the OBSERVABILITY property of robot dynamics.            │
# │                                                                     │
# │  Model structure:                            │
# │    τ_pred = M(q; K1, K2, M11_0)·ddq                               │
# │           + C(q, dq; K1)·dq                                        │
# │           + G(q; meff1, meff2)                                      │
# │           − QF(q; F_sensor)         ← F is input, not learned       │
# └─────────────────────────────────────────────────────────────────────┘
#
# PIPELINE
# ────────
#  1.  Load train.npz, ood_test.npz  (keys: q, dq, ddq, tau, F_log)
#  2.  Build inputs: (Q, DQ, DDQ, F)  — all in physical units (rad, N...)
#  3.  Initialise θ with deliberate error to test identification
#  4.  Full-batch L-BFGS optimisation (7 params → exact gradients clean)
#  5.  Evaluate τ RMSE on Train / Val / OOD
#  6.  Report parameter recovery accuracy vs ground truth
#
# Output:
#   models/pinn_physics.pt   — best checkpoint
#   models/pinn_results.pt   — theta + RMSE results for comparison script
#   figures/pinn_training.png

import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found.  Install: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Known geometry─────────────────────────────────
from dynamics.params import a, b, L, e, h4, g

# ── True θ values — used ONLY for comparison after training ──────────────
# In real deployment these would be unknown.
# In simulation we know them to verify PINN recovery accuracy.
from dynamics.params import (K1    as K1_TRUE,
                              K2    as K2_TRUE,
                              M11_0 as M11_0_TRUE,
                              meff1 as meff1_TRUE,
                              meff2 as meff2_TRUE)

os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
DATA_FRACTION = 0.1    # 1.0 = full dataset of~14000

LR       = 1.0         # L-BFGS step size (line search controls actual step)
N_EPOCHS = 2000        # physics identification needs more epochs than MLP
PATIENCE = 100         # more patience: params converge slowly near optimum
VAL_FRAC = 0.15
SEED     = 42

# ── Initial guess for each parameter ──────────────────────────────────────
# Simulates "rough prior knowledge from CAD / datasheet with errors".
# scale < 1 = underestimate,  scale > 1 = overestimate.
THETA_INIT_SCALE = {
    'K1'   : 0.2,   # 30% underestimate
    'K2'   : 0.20,   # 50% underestimate
    'M11_0': 0.3,   # 25% underestimate
    'meff1': 1.25,   # 25% overestimate
    'meff2': 0.80,   # 20% underestimate
}

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════


print("\nLoading datasets...")
for path in ["data/saved/train.npz", "data/saved/ood_test.npz"]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} — run data/generate_dataset.py first.")

train_raw = np.load("data/saved/train.npz")
ood_raw   = np.load("data/saved/ood_test.npz")

for name, ds in [("TRAIN", train_raw), ("OOD", ood_raw)]:
    if 'F_log' not in ds:
        raise KeyError(
            f"'{name}' missing 'F_log' — "
            "regenerate with updated generate_dataset.py")

def extract(ds):
    """Return (q, dq, ddq, tau, F) all as float32 arrays."""
    return (ds['q'   ].astype(np.float32),
            ds['dq'  ].astype(np.float32),
            ds['ddq' ].astype(np.float32),
            ds['tau' ].astype(np.float32),
            ds['F_log'].astype(np.float32))

q_all, dq_all, ddq_all, tau_all, F_all = extract(train_raw)
q_ood, dq_ood, ddq_ood, tau_ood, F_ood = extract(ood_raw)

# Optional subsampling (OOD always full for fair evaluation)
if DATA_FRACTION < 1.0:
    n_keep = int(len(q_all) * DATA_FRACTION)
    idx    = np.random.default_rng(SEED+1).choice(len(q_all), n_keep, replace=False)
    q_all, dq_all, ddq_all, tau_all, F_all = (
        q_all[idx], dq_all[idx], ddq_all[idx], tau_all[idx], F_all[idx])
    print(f"  ⚠  Reduced to {DATA_FRACTION*100:.0f}%: {n_keep:,} samples")

print(f"  Train total: {len(q_all):,}   OOD: {len(q_ood):,}")
print(f"  Input: (q, dq, ddq, F)  →  8 physical values  (same as MLP)")

# ── Train / Val split ──────────────────────────────────────────────────────
rng   = np.random.default_rng(SEED)
perm  = rng.permutation(len(q_all))
n_val = int(len(q_all) * VAL_FRAC)
i_val, i_tr = perm[:n_val], perm[n_val:]

def split(arr): return arr[i_tr], arr[i_val]

q_tr,   q_val   = split(q_all)
dq_tr,  dq_val  = split(dq_all)
ddq_tr, ddq_val = split(ddq_all)
tau_tr, tau_val = split(tau_all)
F_tr,   F_val   = split(F_all)
print(f"  Train: {len(q_tr):,}   Val: {len(q_val):,}")

def T(x): return torch.from_numpy(x).to(DEVICE)

Q_tr,  DQ_tr,  DDQ_tr,  TAU_tr,  F_TR  = T(q_tr),  T(dq_tr),  T(ddq_tr),  T(tau_tr),  T(F_tr)
Q_val, DQ_val, DDQ_val, TAU_val, F_VAL = T(q_val), T(dq_val), T(ddq_val), T(tau_val), T(F_val)
Q_ood, DQ_ood, DDQ_ood, TAU_ood, F_OOD = T(q_ood), T(dq_ood), T(ddq_ood), T(tau_ood), T(F_ood)

# Loss scale: normalise by τ std so τ₁ and τ₂ contribute equally
tau_scale = torch.tensor(tau_tr.std(0) + 1e-8, dtype=torch.float32, device=DEVICE)

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — PHYSICS MODEL
# ══════════════════════════════════════════════════════════════════════════
# The "network" is the inverse dynamics equation, not a stack of linear layers.
# Learnable: θ = {K1, K2, M11_0, meff1, meff2}  — 5 inertial parameters
# Given as input: F = [Fx, Fy] from sensor at each sample
#
# Positive parameters use log-parameterisation:
#   stored as log(θ),  forward uses exp(log_θ)
#   → Guarantees positivity (no negative inertia) throughout optimisation.
#   → Better-conditioned gradient landscape (multiplicative, not additive).

_a  = float(a)
_L  = float(L)
_h4 = float(h4)
_g  = float(g)

class PhysicsModel(nn.Module):
    """
    Batched inverse dynamics with time-varying external force input.

        τ = M(q; K1,K2,M11_0)·ddq
          + C(q,dq; K1)·dq
          + G(q; meff1,meff2)
          − QF(q; F_input)
    Learnable parameters
    θ = {K1, K2, M11_0, meff1, meff2}  (5 parameters)
    """
    def __init__(self, scale: dict):
        super().__init__()

        def _lp(true_val, s):
            return nn.Parameter(torch.tensor(
                np.log(max(true_val * s, 1e-8)), dtype=torch.float32))

        self.log_K1    = _lp(K1_TRUE,    scale['K1'])
        self.log_K2    = _lp(K2_TRUE,    scale['K2'])
        self.log_M11_0 = _lp(M11_0_TRUE, scale['M11_0'])
        self.log_meff1 = _lp(meff1_TRUE, scale['meff1'])
        self.log_meff2 = _lp(meff2_TRUE, scale['meff2'])

    @property
    def K1(self):    return torch.exp(self.log_K1)
    @property
    def K2(self):    return torch.exp(self.log_K2)
    @property
    def M11_0(self): return torch.exp(self.log_M11_0)
    @property
    def meff1(self): return torch.exp(self.log_meff1)
    @property
    def meff2(self): return torch.exp(self.log_meff2)

    def forward(self, q, dq, ddq, F):
        """
        Parameters
        ----------
        q, dq, ddq : (N, 2)  joint state in physical units  [rad, rad/s, rad/s²]
        F          : (N, 2)  external force at G4  [N]  — sensor reading

        Returns
        -------
        tau_pred   : (N, 2)  predicted control torques  [N·m]
        """
        q1,   q2   = q[:,0],   q[:,1]
        dq1,  dq2  = dq[:,0],  dq[:,1]
        ddq1, ddq2 = ddq[:,0], ddq[:,1]
        Fx,   Fy   = F[:,0],   F[:,1]

        c1  = torch.cos(q1);   s1  = torch.sin(q1)
        c2  = torch.cos(q2);   s2  = torch.sin(q2)
        c12 = torch.cos(q1+q2); s12 = torch.sin(q1+q2)

        # ── Mass matrix elements ─────────────────────────────────────
        # M = [[M11_0 + 2K1c2,  K2+K1c2],
        #      [K2+K1c2,        K2     ]]
        M11 = self.M11_0 + 2.0*self.K1*c2
        M12 = self.K2    +     self.K1*c2
        M22 = self.K2

        # ── Coriolis·q̇  ─────────────────────────────────────────────
        # C·q̇ = [−K1·s2·(2·dq1·dq2 + dq2²),
        #          K1·s2·dq1²             ]
        C1 = -self.K1 * s2 * (2.0*dq1*dq2 + dq2**2)
        C2 =  self.K1 * s2 * dq1**2

        # ── Gravity ──────────────────────────────────────────────────
        # G = g·[meff1·c1 + meff2·c12,
        #         meff2·c12           ]
        G1 = _g * (self.meff1*c1 + self.meff2*c12)
        G2 = _g *  self.meff2*c12

        # ── Generalised external forces: QF = Jᵀ·F ──────────────────
        # G4 Jacobian (partial, 2×2 for the two DOF):
        #   J_row1 = [−(a + L·c2 + h4)·s1 − L·s2·c1,
        #              (a + L·c2 + h4)·c1 − L·s2·s1 ]
        #   J_row2 = [−L·s12,    L·c12              ]
        arm = _a + _L*c2 + _h4
        QF1 = (-arm*s1 - _L*s2*c1)*Fx + ( arm*c1 - _L*s2*s1)*Fy
        QF2 = (-_L*s12            )*Fx + (_L*c12             )*Fy

        # ── τ = M·ddq + C·q̇ + G − QF ─────────────────────────────
        tau1 = M11*ddq1 + M12*ddq2 + C1 + G1 - QF1
        tau2 = M12*ddq1 + M22*ddq2 + C2 + G2 - QF2

        return torch.stack([tau1, tau2], dim=1)

    def get_theta(self) -> dict:
        return {k: getattr(self, k).item()
                for k in ['K1','K2','M11_0','meff1','meff2']}


model = PhysicsModel(THETA_INIT_SCALE).to(DEVICE)

# ── Ground-truth θ for comparison ────────────────────────────────────────
theta_true = {
    'K1'   : float(K1_TRUE),
    'K2'   : float(K2_TRUE),
    'M11_0': float(M11_0_TRUE),
    'meff1': float(meff1_TRUE),
    'meff2': float(meff2_TRUE),
}
theta_init = model.get_theta()

print(f"\nPhysics model — {len(theta_true)} learnable parameters "
      f"(F treated as sensor input):")
print(f"\n  {'Param':<8} {'True':>12} {'Init':>12} {'Init error':>12}")
print("  " + "─"*50)
for k, v_true in theta_true.items():
    v_init = theta_init[k]
    pct    = abs(v_init - v_true) / (abs(v_true) + 1e-8) * 100
    print(f"  {k:<8} {v_true:>12.5f} {v_init:>12.5f} {pct:>10.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINING  (full-batch L-BFGS)
# ══════════════════════════════════════════════════════════════════════════

# Physics loss = normalised MSE:
#   L = mean( ((τ_pred − τ_true) / τ_std)² )
# Normalisation ensures τ₁ and τ₂ contribute equally regardless of scale.

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=LR, max_iter=20,
    history_size=100,
    line_search_fn='strong_wolfe')

def physics_loss(pred, true):
    return ((pred - true) / tau_scale).pow(2).mean()

train_losses, val_losses = [], []
theta_history = {k: [] for k in theta_true}
best_val      = float('inf')
patience_cnt  = 0
t0            = time.time()

print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'improved':>9}")
print("─" * 52)

def make_closure():
    optimizer.zero_grad()
    loss = physics_loss(model(Q_tr, DQ_tr, DDQ_tr, F_TR), TAU_tr)
    loss.backward()
    return loss

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    loss_tr = optimizer.step(make_closure).detach().item()

    model.eval()
    with torch.no_grad():
        loss_val = physics_loss(
            model(Q_val, DQ_val, DDQ_val, F_VAL), TAU_val).item()

    train_losses.append(loss_tr)
    val_losses.append(loss_val)

    for k, v in model.get_theta().items():
        theta_history[k].append(v)

    if loss_val < best_val:
        best_val = loss_val
        torch.save(model.state_dict(), "models/pinn_physics.pt")
        patience_cnt = 0
        star = "★"
    else:
        patience_cnt += 1
        star = ""

    if epoch % 100 == 0 or epoch == 1:
        print(f"{epoch:>6}  {loss_tr:>12.6f}  {loss_val:>12.6f}  {star}")

    if patience_cnt >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}")
        break

print(f"\n  Training done in {time.time()-t0:.1f}s  "
      f"|  Best val loss = {best_val:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════


model.load_state_dict(torch.load("models/pinn_physics.pt", map_location=DEVICE))
model.eval()

def evaluate_rmse(Q, DQ, DDQ, F, TAU, label=""):
    with torch.no_grad():
        pred = model(Q, DQ, DDQ, F).cpu().numpy()
    true      = TAU.cpu().numpy()
    err       = pred - true
    per_joint = np.sqrt(np.mean(err**2, axis=0))
    total     = float(np.sqrt(np.mean(err**2)))
    return per_joint, total, pred

pj_tr,  r_tr,  Yp_tr  = evaluate_rmse(Q_tr,  DQ_tr,  DDQ_tr,  F_TR,  TAU_tr)
pj_val, r_val, Yp_val = evaluate_rmse(Q_val, DQ_val, DDQ_val, F_VAL, TAU_val)
pj_ood, r_ood, Yp_ood = evaluate_rmse(Q_ood, DQ_ood, DDQ_ood, F_OOD, TAU_ood)

# ── τ RMSE table ──────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  PINN — τ Prediction RMSE  (input: q, dq, ddq, F_sensor)")
print("═"*70)
print(f"  {'Split':<14} {'τ₁ RMSE':>10} {'τ₂ RMSE':>10} {'Total':>12}")
print("  " + "─"*56)
print(f"  {'Train':<14} {pj_tr[0]:>10.4f} {pj_tr[1]:>10.4f} {r_tr:>12.4f}  N·m")
print(f"  {'Validation':<14} {pj_val[0]:>10.4f} {pj_val[1]:>10.4f} {r_val:>12.4f}  N·m")
print(f"  {'OOD Test':<14} {pj_ood[0]:>10.4f} {pj_ood[1]:>10.4f} {r_ood:>12.4f}  N·m")
print("═"*70)
print(f"  OOD / Train ratio  :  {r_ood/r_tr:.2f}×")


# ── Parameter recovery ────────────────────────────────────────────────────
theta_learned = model.get_theta()

print("\n" + "═"*70)
print("  PINN — Physical Parameter Recovery")
print("═"*70)
print(f"  {'Param':<8} {'True':>12} {'Init':>12} {'Learned':>12} {'Error %':>10}")
print("  " + "─"*60)
for k, v_true in theta_true.items():
    v_init    = theta_init[k]
    v_learned = theta_learned[k]
    err_pct   = abs(v_learned - v_true) / (abs(v_true) + 1e-8) * 100
    status    = "[ok]" if err_pct < 5 else "[!!]"
    print(f"  {k:<8} {v_true:>12.5f} {v_init:>12.5f} {v_learned:>12.5f} "
          f"{err_pct:>9.2f}%  {status}")
print("═"*70)

# ── Save results ──────────────────────────────────────────────────────────
torch.save({
    'theta_true'   : theta_true,
    'theta_init'   : theta_init,
    'theta_learned': theta_learned,
    'r_tr' : r_tr,  'r_val': r_val,  'r_ood': r_ood,
    'pj_tr': pj_tr.tolist(),
    'pj_val': pj_val.tolist(),
    'pj_ood': pj_ood.tolist(),
}, "models/pinn_results.pt")

print(f"\n  Saved: models/pinn_physics.pt")
print(f"  Saved: models/pinn_results.pt")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — FIGURES  (2 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════
epochs_x = np.arange(1, len(train_losses)+1)
fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    f"PINN — Physics Parameter Identification\n"
    f"θ = {{K1, K2, M11_0, meff1, meff2}}  |  F treated as sensor input  |  "
    f"data = {DATA_FRACTION*100:.0f}%  ({len(q_tr):,} train samples)",
    fontsize=11, fontweight='bold')
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

c_tr  = '#4a9eff'
c_val = '#5bc8af'
c_ood = '#ff6b6b'

# ── (0,0) Training curves ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.semilogy(epochs_x, train_losses, color=c_tr,  lw=1.5, label='Train (norm. MSE)')
ax.semilogy(epochs_x, val_losses,   color=c_val, lw=1.5, label='Val  (norm. MSE)')
ax.set_xlabel('Epoch');  ax.set_ylabel('Normalised MSE')
ax.set_title('Training Curves  (L-BFGS)')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

# ── (0,1) Inertia parameter convergence ──────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
colors_K = {'K1': '#1f77b4', 'K2': '#2ca02c', 'M11_0': '#d62728'}
for k, col in colors_K.items():
    ratio = np.array(theta_history[k]) / theta_true[k]
    ax.plot(epochs_x, ratio, color=col, lw=1.8, label=k)
ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5, label='True (=1.0)')
ax.set_xlabel('Epoch');  ax.set_ylabel('θ_learned / θ_true')
ax.set_title('Inertia Constants Convergence\n(perfect recovery → 1.0)')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 1.7)

# ── (0,2) Gravity parameter convergence ──────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
colors_m = {'meff1': '#9467bd', 'meff2': '#8c564b'}
for k, col in colors_m.items():
    ratio = np.array(theta_history[k]) / theta_true[k]
    ax.plot(epochs_x, ratio, color=col, lw=1.8, label=k)
ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5, label='True (=1.0)')
ax.set_xlabel('Epoch');  ax.set_ylabel('θ_learned / θ_true')
ax.set_title('Gravity Parameters Convergence\n(perfect recovery → 1.0)')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 1.7)

# ── (1,0) τ₁ true vs predicted ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
lim = max(np.abs(tau_tr[:,0]).max(), np.abs(tau_ood[:,0]).max()) * 1.05
ax.scatter(tau_tr[:,0],  Yp_tr[:,0],  s=1, alpha=0.25, color=c_tr,  label='Train')
ax.scatter(tau_ood[:,0], Yp_ood[:,0], s=2, alpha=0.45, color=c_ood, label='OOD')
ax.plot([-lim,lim], [-lim,lim], 'k--', lw=1, label='Perfect')
ax.set_xlabel('τ₁ true (N·m)');  ax.set_ylabel('τ₁ predicted (N·m)')
ax.set_title(f'τ₁  |  Train={pj_tr[0]:.4f}  OOD={pj_ood[0]:.4f} N·m')
ax.legend(fontsize=8, markerscale=6);  ax.grid(True, alpha=0.3)

# ── (1,1) τ₂ true vs predicted ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
lim = max(np.abs(tau_tr[:,1]).max(), np.abs(tau_ood[:,1]).max()) * 1.05
ax.scatter(tau_tr[:,1],  Yp_tr[:,1],  s=1, alpha=0.25, color=c_tr,  label='Train')
ax.scatter(tau_ood[:,1], Yp_ood[:,1], s=2, alpha=0.45, color=c_ood, label='OOD')
ax.plot([-lim,lim], [-lim,lim], 'k--', lw=1, label='Perfect')
ax.set_xlabel('τ₂ true (N·m)');  ax.set_ylabel('τ₂ predicted (N·m)')
ax.set_title(f'τ₂  |  Train={pj_tr[1]:.4f}  OOD={pj_ood[1]:.4f} N·m')
ax.legend(fontsize=8, markerscale=6);  ax.grid(True, alpha=0.3)

# ── (1,2) Parameter recovery bar chart ───────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
p_names = list(theta_true.keys())
p_errs  = [abs(theta_learned[k] - theta_true[k]) / (abs(theta_true[k]) + 1e-8) * 100
           for k in p_names]
bar_colors = ['#2ca02c' if e < 2 else '#ff7f0e' if e < 5 else '#d62728'
              for e in p_errs]
ax.barh(p_names, p_errs, color=bar_colors, edgecolor='white', linewidth=0.6)
ax.axvline(2.0, color='green',  ls='--', lw=1.2, alpha=0.8, label='< 2% [ok]')
ax.axvline(5.0, color='orange', ls='--', lw=1.2, alpha=0.8, label='< 5% [!!]')
# Annotate each bar
for i, (name, err) in enumerate(zip(p_names, p_errs)):
    ax.text(err + 0.1, i, f'{err:.2f}%', va='center', fontsize=9)
ax.set_xlabel('Parameter recovery error (%)')
ax.set_title('Parameter Recovery Accuracy\n(lower = PINN learned correct physics)')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='x')

plt.savefig("figures/pinn_training.png", dpi=150, bbox_inches='tight')
print(f"\n  Figure → figures/pinn_training.png")
plt.show()