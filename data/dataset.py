"""
PyTorch Dataset wrapper for the four-bar linkage inverse-dynamics data.

Loads a .npz file produced by data/generate.py and exposes
(X, Y) pairs for use with a DataLoader.

Input features X  (8 columns):
    [q1, q2, dq1, dq2, ddq1, ddq2, Fx, Fy]

Target Y  (2 columns):
    [τ1, τ2]

The external force F_log = [Fx, Fy] MUST be included in X.
Without it, the same kinematic state (q, dq, ddq) maps to different
torques at different times because F(t) varies — making the regression
ill-posed.

Notes
-----
This class computes normalisation statistics from the loaded dataset.
For a fair train/OOD comparison, always normalise OOD data with the
statistics computed from the TRAINING set (not from OOD itself).
See MLP.py for the recommended two-dataset workflow.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicsDataset(Dataset):
    """
    Dataset of (state, torque) pairs for the four-bar linkage.

    Parameters
    ----------
    npz_path : str
        Path to a .npz file with keys: q, dq, ddq, tau, F_log.
    normalize : bool, optional
        If True, z-score normalise X and Y using statistics from
        this dataset (default True).
        Set to False when applying external statistics (e.g. training stats
        to an OOD set).
    x_mean, x_std, y_mean, y_std : np.ndarray or None, optional
        External normalisation statistics.  When provided, these are used
        instead of computing statistics from the loaded data, and the
        `normalize` flag is ignored.  Pass the training-set statistics when
        wrapping an OOD dataset.
    """

    #: Input feature names — used for logging and sanity checks.
    FEAT_NAMES = ['q1', 'q2', 'dq1', 'dq2', 'ddq1', 'ddq2', 'Fx', 'Fy']

    def __init__(self, npz_path: str, normalize: bool = True,
                 x_mean=None, x_std=None, y_mean=None, y_std=None):
        data = np.load(npz_path)

        # ── Verify required keys ──────────────────────────────────────────
        required = {'q', 'dq', 'ddq', 'tau', 'F_log'}
        missing  = required - set(data.keys())
        if missing:
            raise KeyError(
                f"{npz_path} is missing keys: {missing}. "
                "Regenerate with data/generate.py."
            )

        # ── Build feature matrix (8 inputs) ──────────────────────────────
        # Input order must match MLP.py and PINN.py exactly.
        X = np.hstack([
            data['q'],      # (N, 2)  joint positions   [rad]
            data['dq'],     # (N, 2)  joint velocities  [rad/s]
            data['ddq'],    # (N, 2)  joint accels      [rad/s²]
            data['F_log'],  # (N, 2)  external force    [N]  ← REQUIRED
        ]).astype(np.float32)
        Y = data['tau'].astype(np.float32)   # (N, 2)  torques [N·m]

        # ── Normalisation ─────────────────────────────────────────────────
        if x_mean is not None:
            # External statistics supplied — use them directly
            self.x_mean = x_mean.astype(np.float32)
            self.x_std  = x_std.astype(np.float32)
            self.y_mean = y_mean.astype(np.float32)
            self.y_std  = y_std.astype(np.float32)
        else:
            # Compute from this dataset
            self.x_mean = X.mean(0).astype(np.float32)
            self.x_std  = (X.std(0) + 1e-8).astype(np.float32)
            self.y_mean = Y.mean(0).astype(np.float32)
            self.y_std  = (Y.std(0) + 1e-8).astype(np.float32)

        if normalize or x_mean is not None:
            X = (X - self.x_mean) / self.x_std
            Y = (Y - self.y_mean) / self.y_std

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]