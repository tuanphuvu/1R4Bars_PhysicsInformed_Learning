#MLP.py

#  f(q, dq, ddq, F_sensor) → τ
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  PURPOSE                                                         │
# │                                                                  │
# │  This is NOT a model used in production.                         │
# │                                                                  │
# │  WHAT THE MLP LEARNS                                             │
# │  ─────────────────────────────────────────────────────────────  │
# │  Input  (8 features): [q₁, q₂, dq₁, dq₂, ddq₁, ddq₂, Fx, Fy] │
# │  Output (2 values)  : [τ₁, τ₂]                                  │
# │                                                                  │
# │    τ = M(q)·ddq + C(q,dq)·dq + g(q) − QF(q, F_sensor)         │
# │                                                                  │
# │  F_sensor is used to train
# └──────────────────────────────────────────────────────────────────┘
#
# PIPELINE
# ────────
#  1.  Load train.npz, ood_test.npz  (keys: q, dq, ddq, tau, F_log)
#  2.  Build features X = [q, dq, ddq, F_log]  →  8 columns
#  3.  Normalise X and Y using TRAIN statistics only
#  4.  Train MLP with Adam + ReduceLROnPlateau + early stopping
#  5.  Evaluate on Train / Val / OOD in original N·m units
#  6.  Save checkpoint + normalisation stats (reused by Physics_informed_model)
#
# Output:
#   models/mlp_baseline.pt    — best checkpoint (lowest val loss)
#   models/mlp_norm.npz       — normalisation statistics (reuse for Physics_informed_model)
#   figures/mlp_training.png  — training curves + prediction scatter

import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch not found.  Install: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
DATA_FRACTION = 0.1   # 1.0 = full dataset of ~ 14000
HIDDEN_DIM    = 128   # neurons per hidden layer
N_LAYERS      = 3     # number of hidden layers
BATCH_SIZE    = 512
LR            = 1e-3
N_EPOCHS      = 300
PATIENCE      = 30    # early stopping patience (epochs)
VAL_FRAC      = 0.15  # fraction of TRAIN used as validation
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
# Feature set (8 inputs)
# ─────────────────────
#   q₁, q₂       — joint positions    [rad]
#   dq₁, dq₂     — joint velocities   [rad/s]
#   ddq₁, ddq₂   — joint accel.       [rad/s²]  (from spline of q_sim)
#   Fx, Fy        — external force     [N]  (sensor reading = F_log, true force)
#
# The equation of motion is:
#   τ = M(q)·ddq + C(q,dq)·dq + g(q) − QF(q, F)

print("\nLoading datasets...")
for path in ["data/saved/train.npz", "data/saved/ood_test.npz"]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run data/generate_dataset.py first.")

train_raw = np.load("data/saved/train.npz")
ood_raw   = np.load("data/saved/ood_test.npz")

# Verify F_log is present
for name, ds in [("TRAIN", train_raw), ("OOD", ood_raw)]:
    if 'F_log' not in ds:
        raise KeyError(
            f"'{name}' dataset missing 'F_log' — "
            "regenerate with updated generate_dataset.py")

def build_XY(ds):
    """
    Build feature matrix and target vector.

    X : (N, 8)  [q₁, q₂, dq₁, dq₂, ddq₁, ddq₂, Fx, Fy]
    Y : (N, 2)  [τ₁, τ₂]
    """
    X = np.hstack([
        ds['q'],        # (N, 2)  joint positions
        ds['dq'],       # (N, 2)  joint velocities
        ds['ddq'],      # (N, 2)  joint accelerations
        ds['F_log'],    # (N, 2)  external force (sensor reading)
    ]).astype(np.float32)
    Y = ds['tau'].astype(np.float32)   # (N, 2)  control torques
    return X, Y

X_all, Y_all = build_XY(train_raw)
X_ood, Y_ood = build_XY(ood_raw)

print(f"  TRAIN total : {len(X_all):,} samples   shape X={X_all.shape}  Y={Y_all.shape}")
print(f"  OOD         : {len(X_ood):,} samples")
print(f"  Input columns: [q₁ q₂ | dq₁ dq₂ | ddq₁ ddq₂ | Fx Fy]  →  8 features")

# Optional subsampling (OOD always stays full for fair evaluation)
if DATA_FRACTION < 1.0:
    n_keep   = int(len(X_all) * DATA_FRACTION)
    idx_keep = np.random.default_rng(SEED+1).choice(
                   len(X_all), n_keep, replace=False)
    X_all, Y_all = X_all[idx_keep], Y_all[idx_keep]
    print(f"  ⚠  Reduced to {DATA_FRACTION*100:.0f}%: {n_keep:,} samples")

# ── Train / Validation split ──────────────────────────────────────────────
rng   = np.random.default_rng(SEED)
perm  = rng.permutation(len(X_all))
n_val = int(len(X_all) * VAL_FRAC)
idx_val, idx_tr = perm[:n_val], perm[n_val:]

X_tr,  Y_tr  = X_all[idx_tr],  Y_all[idx_tr]
X_val, Y_val = X_all[idx_val], Y_all[idx_val]
print(f"\n  Split → Train: {len(X_tr):,}   Val: {len(X_val):,}   OOD: {len(X_ood):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — NORMALISATION
# ══════════════════════════════════════════════════════════════════════════
# Why normalise all 8 features?
# ──────────────────────────────
# Feature scales differ greatly:
#   q     ∈ [−π/2, π/4]  ≈  ±1.5
#   dq    ∈ [−2,   2]    ≈  ±2
#   ddq   ∈ [−50,  50]   ≈  ±50   ← 30× larger than q
#   F     ∈ [−15,  15]   ≈  ±15
#   τ     ∈ [−30,  30]   ≈  ±30
# Without normalisation the gradient is dominated by ddq/τ columns,
# causing slow convergence or divergence on the others.
#
# Statistics are computed on TRAIN only — never on val or OOD.
# The same stats are saved so Physics_informed_model can reuse them for a fair comparison.

X_mean = X_tr.mean(0);  X_std = X_tr.std(0) + 1e-8
Y_mean = Y_tr.mean(0);  Y_std = Y_tr.std(0) + 1e-8

def norm_X(X):     return (X - X_mean) / X_std
def norm_Y(Y):     return (Y - Y_mean) / Y_std
def denorm_Y(Yn):  return Yn * Y_std + Y_mean

# Feature names for reporting
FEAT_NAMES = ['q₁', 'q₂', 'dq₁', 'dq₂', 'ddq₁', 'ddq₂', 'Fx', 'Fy']
print("\n  Normalisation (TRAIN statistics):")
print(f"  {'Feature':<8} {'mean':>10} {'std':>10}")
print("  " + "─"*30)
for name, mu, sig in zip(FEAT_NAMES, X_mean, X_std):
    print(f"  {name:<8} {mu:>10.4f} {sig:>10.4f}")

np.savez("models/mlp_norm.npz",
         X_mean=X_mean, X_std=X_std,
         Y_mean=Y_mean, Y_std=Y_std,
         feat_names=np.array(FEAT_NAMES))
print("  → Saved: models/mlp_norm.npz  (reuse for Physics_informed_model comparison)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — DATASET / DATALOADER
# ══════════════════════════════════════════════════════════════════════════
def to_tensors(X, Y=None):
    Xt = torch.from_numpy(norm_X(X)).to(DEVICE)
    if Y is not None:
        return Xt, torch.from_numpy(norm_Y(Y)).to(DEVICE)
    return Xt

X_tr_t,  Y_tr_t  = to_tensors(X_tr,  Y_tr)
X_val_t, Y_val_t = to_tensors(X_val, Y_val)
X_ood_t          = to_tensors(X_ood)

train_loader = DataLoader(
    TensorDataset(X_tr_t, Y_tr_t),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════
# Architecture:  Linear(8→128) → [BN → ReLU → Linear] × N_LAYERS → Linear(128→2)
#
# BatchNorm is used because the 8 input features span very different scales
# even after global normalisation; BN stabilises per-layer activations.
# The architecture is intentionally simple — this is a baseline, not a

class MLPBaseline(nn.Module):
    """
    Fully-connected network with BatchNorm.

    Input  : 8 features  [q, dq, ddq, F_sensor]
    Output : 2 torques   [τ₁, τ₂]
    """
    def __init__(self, in_dim=8, hidden=HIDDEN_DIM,
                 n_layers=N_LAYERS, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden),
                  nn.BatchNorm1d(hidden),
                  nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden),
                       nn.BatchNorm1d(hidden),
                       nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model     = MLPBaseline().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# ReduceLROnPlateau: halve LR if val loss stalls for 10 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {N_LAYERS} hidden layers × {HIDDEN_DIM} neurons  "
      f"= {n_params:,} parameters")
print(f"  Input  (8): [q₁ q₂  dq₁ dq₂  ddq₁ ddq₂  Fx Fy]")
print(f"  Output (2): [τ₁ τ₂]")
print(f"  Batch size: {BATCH_SIZE}   LR: {LR}   Epochs: {N_EPOCHS}   Patience: {PATIENCE}\n")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
# Pipeline per epoch:
#   model.train()  → BN uses batch statistics  → mini-batch Adam step
#   model.eval()   → BN uses running stats     → val loss (no grad)
#   scheduler.step(val_loss)                   → LR decay on plateau
#   save checkpoint if val loss improves       → best model = lowest val
#   early stopping if no improvement ≥ PATIENCE epochs

train_losses, val_losses = [], []
best_val_loss  = float('inf')
patience_count = 0
t0 = time.time()

print(f"{'Epoch':>6}  {'Train MSE':>11}  {'Val MSE':>11}  {'LR':>10}")
print("─" * 50)

for epoch in range(1, N_EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────
    model.train()
    batch_losses = []
    for Xb, Yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), Yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    tr_loss = float(np.mean(batch_losses))

    # ── Validate ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), Y_val_t).item()

    scheduler.step(val_loss)
    train_losses.append(tr_loss)
    val_losses.append(val_loss)

    # ── Checkpoint ────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        torch.save(model.state_dict(), "models/mlp_baseline.pt")
        patience_count = 0
        improved = "★"
    else:
        patience_count += 1
        improved = ""

    if epoch % 20 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6}  {tr_loss:>11.5f}  {val_loss:>11.5f}  "
              f"{lr_now:>10.2e}  {improved}")

    if patience_count >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}  "
              f"(no improvement for {PATIENCE} epochs)")
        break

print(f"\n  Training complete in {time.time()-t0:.1f}s  "
      f"|  Best val MSE = {best_val_loss:.5f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════
# Load best checkpoint and evaluate in original N·m (denormalised).
# Three splits:
#   Train  — lower bound (in-distribution, seen during training)
#   Val    — unbiased in-distribution estimate (same distribution, not seen)
#   OOD    — unseen amplitude A=2.5cm  (tests generalisation)
#

model.load_state_dict(torch.load("models/mlp_baseline.pt", map_location=DEVICE))
model.eval()

def evaluate(X_raw, Y_raw, label=""):
    """RMSE per joint and total, in original N·m (denormalised)."""
    Xt = to_tensors(X_raw)
    with torch.no_grad():
        Yn_pred = model(Xt).cpu().numpy()
    Y_pred   = denorm_Y(Yn_pred)
    err      = Y_pred - Y_raw
    rmse_j   = np.sqrt(np.mean(err**2, axis=0))   # (2,)
    rmse_tot = float(np.sqrt(np.mean(err**2)))
    return rmse_tot, rmse_j, Y_pred

r_tr,  pj_tr,  Yp_tr  = evaluate(X_tr,  Y_tr,  "Train")
r_val, pj_val, Yp_val = evaluate(X_val, Y_val, "Val")
r_ood, pj_ood, Yp_ood = evaluate(X_ood, Y_ood, "OOD")

# ── Results table ─────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  MLP BASELINE — Final Results  (8 inputs: q, dq, ddq, F_sensor)")
print("═"*70)
print(f"  {'Split':<14} {'τ₁ RMSE':>10} {'τ₂ RMSE':>10} {'Total RMSE':>12}")
print("  " + "─"*48)
print(f"  {'Train':<14} {pj_tr[0]:>10.4f} {pj_tr[1]:>10.4f} {r_tr:>12.4f}  N·m")
print(f"  {'Validation':<14} {pj_val[0]:>10.4f} {pj_val[1]:>10.4f} {r_val:>12.4f}  N·m")
print(f"  {'OOD Test':<14} {pj_ood[0]:>10.4f} {pj_ood[1]:>10.4f} {r_ood:>12.4f}  N·m")
print("═"*70)
print(f"  OOD / Train ratio  :  {r_ood/r_tr:.1f}×")
print()
print("  Interpretation")
print("  ─────────────────────────────────────────────────────────────")
print("  OOD ratio >> 1 is expected and confirms the need for Physic-informed ML.")
print("═"*70)
print(f"\n  Saved:  models/mlp_baseline.pt")
print(f"  Saved:  models/mlp_norm.npz    (feat_names, X_mean/std, Y_mean/std)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — FIGURES
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    f"MLP Baseline  |  Input: [q, dq, ddq, F_sensor]  (8 features)\n"
    f"hidden={HIDDEN_DIM}  layers={N_LAYERS}  params={n_params:,}  "
    f"Kp={400}  Kd={40}",
    fontsize=11, fontweight='bold')

c_tr  = '#4a9eff'
c_ood = '#ff6b6b'
c_val = '#5bc8af'

# ── (0,0) Training curves ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.semilogy(train_losses, color=c_tr,  lw=1.5, label='Train MSE (norm.)')
ax.semilogy(val_losses,   color=c_val, lw=1.5, label='Val MSE (norm.)')
ax.set_xlabel('Epoch');  ax.set_ylabel('MSE (normalised, log scale)')
ax.set_title('Training Curves')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

# ── (0,1) τ₁ true vs predicted ───────────────────────────────────────────
ax = axes[0, 1]
lim = max(np.abs(Y_tr[:,0]).max(), np.abs(Y_ood[:,0]).max()) * 1.05
ax.scatter(Y_tr[:,0],  Yp_tr[:,0],  s=1, alpha=0.2, color=c_tr,  label='Train')
ax.scatter(Y_ood[:,0], Yp_ood[:,0], s=2, alpha=0.4, color=c_ood, label='OOD')
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='Perfect')
ax.set_xlabel('τ₁ true (N·m)');  ax.set_ylabel('τ₁ predicted (N·m)')
ax.set_title(f'τ₁  |  Train={pj_tr[0]:.3f}  OOD={pj_ood[0]:.3f} N·m')
ax.legend(fontsize=8, markerscale=6);  ax.grid(True, alpha=0.3)

# ── (0,2) τ₂ true vs predicted ───────────────────────────────────────────
ax = axes[0, 2]
lim = max(np.abs(Y_tr[:,1]).max(), np.abs(Y_ood[:,1]).max()) * 1.05
ax.scatter(Y_tr[:,1],  Yp_tr[:,1],  s=1, alpha=0.2, color=c_tr,  label='Train')
ax.scatter(Y_ood[:,1], Yp_ood[:,1], s=2, alpha=0.4, color=c_ood, label='OOD')
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='Perfect')
ax.set_xlabel('τ₂ true (N·m)');  ax.set_ylabel('τ₂ predicted (N·m)')
ax.set_title(f'τ₂  |  Train={pj_tr[1]:.3f}  OOD={pj_ood[1]:.3f} N·m')
ax.legend(fontsize=8, markerscale=6);  ax.grid(True, alpha=0.3)

# ── (1,0) Error distribution: Train vs OOD ────────────────────────────────
ax = axes[1, 0]
err_tr_all  = (Yp_tr  - Y_tr).ravel()
err_ood_all = (Yp_ood - Y_ood).ravel()
bins = np.linspace(
    min(err_tr_all.min(), err_ood_all.min()),
    max(err_tr_all.max(), err_ood_all.max()),
    80)
ax.hist(err_tr_all,  bins=bins, color=c_tr,  alpha=0.6, density=True,
        label=f'Train (std={err_tr_all.std():.3f})')
ax.hist(err_ood_all, bins=bins, color=c_ood, alpha=0.6, density=True,
        label=f'OOD   (std={err_ood_all.std():.3f})')
ax.set_xlabel('τ error  (N·m)');  ax.set_ylabel('density')
ax.set_title('Prediction Error Distribution')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

# ── (1,1) Feature importance proxy: weight norm of first layer ────────────
# the L2 norm of input weights per feature reflects how much each
# input was used. NOT a rigorous attribution but useful for sanity check —
# if Fx/Fy weights ≈ 0 the model ignored the force input.
ax = axes[1, 1]
first_layer = list(model.net.children())[0]   # first nn.Linear (8 → hidden)
w_norms = first_layer.weight.detach().cpu().numpy()   # (hidden, 8)
feat_importance = np.linalg.norm(w_norms, axis=0)     # (8,)
colors_bar = [c_tr]*6 + ['#f4a261']*2                 # orange for F columns
bars = ax.bar(FEAT_NAMES, feat_importance, color=colors_bar, edgecolor='white',
              linewidth=0.6)
ax.set_ylabel('‖w‖₂  (input weight norm)')
ax.set_title('Input Feature Usage\n(first-layer weight norms)')
ax.grid(True, alpha=0.3, axis='y')
# Annotate F columns
for i in (6, 7):
    ax.text(i, feat_importance[i] + feat_importance.max()*0.02,
            f'{feat_importance[i]:.2f}', ha='center', fontsize=8, color='#f4a261')

# ── (1,2) Summary bar: RMSE comparison ────────────────────────────────────
ax = axes[1, 2]
labels   = ['Train', 'Validation', 'OOD Test']
rmse_all = [r_tr, r_val, r_ood]
bar_colors = [c_tr, c_val, c_ood]
bars = ax.bar(labels, rmse_all, color=bar_colors, edgecolor='white', linewidth=0.6)
for bar, val in zip(bars, rmse_all):
    ax.text(bar.get_x() + bar.get_width()/2, val + max(rmse_all)*0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('RMSE  (N·m)')
ax.set_title(f'RMSE Summary\nOOD/Train = {r_ood/r_tr:.1f}×')
ax.grid(True, alpha=0.3, axis='y')
# Reference line at best train RMSE
ax.axhline(r_tr, color='white', lw=0.8, ls='--', alpha=0.6, label='Train level')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/mlp_training.png", dpi=150, bbox_inches='tight')
print(f"\n  Figure → figures/mlp_training.png")
plt.show()