"""
The IK problem: given a desired end-effector position p = [x, y],
find joint angles q = [q1, q2] such that fk_G4(q) = p.

Two public functions are used by the data-generation and validation pipelines:

    _q_init_analytical(p_des)  — closed-form initial guess
    ik_solve(p_des, q_init)    — Newton-Raphson iteration with damped Jacobian
"""

import numpy as np
from dynamics.matrices import fk_G4, jacobian_G4
from dynamics.params import a, L, h4


# ── Analytical initial guess ─────────────────────────────────────────────

def _q_init_analytical(p_des: np.ndarray) -> np.ndarray:
    """
    Compute a closed-form initial guess for IK using the distance equation.

    Derivation:
        |G4|² = (a + h4)² + L² + 2·(a + h4)·L·cos(q2)
    Solve for q2, then recover q1 from the direction of p_des.

    Parameters
    ----------
    p_des : np.ndarray, shape (2,)
        Desired end-effector position [x, y] in metres.

    Returns
    -------
    q_init : np.ndarray, shape (2,)
        Initial guess [q1, q2] in radians.
    """
    px, py   = p_des
    p_norm   = np.linalg.norm(p_des)

    # q2 from distance equation
    cos_q2 = (p_norm**2 - ((a + h4)**2 + L**2)) / (2 * (a + h4) * L)
    cos_q2 = np.clip(cos_q2, -0.99, 0.99)
    q2_init = np.arccos(cos_q2)   # principal value in [0, π]

    # q1 from direction of p_des
    phi     = np.arctan2(L * np.sin(q2_init), (a + h4) + L * cos_q2)
    q1_init = np.arctan2(py, px) - phi

    return np.array([q1_init, q2_init])


# ── Iterative IK solver ───────────────────────────────────────────────────

def ik_solve(p_des: np.ndarray, q_init: np.ndarray,
             max_iter: int = 200, tol: float = 1e-7,
             alpha: float = 0.5) -> tuple:
    """
    Iterative inverse kinematics via damped least-squares (Levenberg–Marquardt).

    Solves  fk_G4(q) = p_des  by iterating:
        J_pinv = Jᵀ · (J·Jᵀ + λ·I)⁻¹     (damped pseudo-inverse)
        q ← q + α · J_pinv · (p_des − G4(q))

    The damping factor λ prevents numerical blow-up near Jacobian singularities.

    Parameters
    ----------
    p_des : np.ndarray, shape (2,)
        Desired end-effector position [x, y] in metres.
    q_init : np.ndarray, shape (2,)
        Initial joint angle guess [q1, q2] in radians.
        Use _q_init_analytical(p_des) for the first call; thereafter
        pass the previous solution for warm-starting (much faster).
    max_iter : int, optional
        Maximum Newton–Raphson iterations (default 200).
    tol : float, optional
        Convergence threshold on Cartesian error ‖p_des − G4(q)‖ [m]
        (default 1e-7).
    alpha : float, optional
        Step size in (0, 1].  Reduce if the solver oscillates (default 0.5).

    Returns
    -------
    q : np.ndarray, shape (2,)
        Converged joint angles [q1, q2] in radians.
    converged : bool
        True if ‖error‖ < tol was reached within max_iter steps.
    """
    q   = q_init.copy()
    lam = 1e-4   # damping coefficient

    for _ in range(max_iter):
        err = p_des - fk_G4(q)
        if np.linalg.norm(err) < tol:
            return q, True
        J      = jacobian_G4(q)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(2))
        q     += alpha * J_pinv @ err

    return q, False   # did not converge