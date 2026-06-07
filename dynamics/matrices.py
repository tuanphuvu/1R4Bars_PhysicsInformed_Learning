"""
Lagrangian dynamics matrices for the 2-DOF rotating parallelogram four-bar linkage.

Implements the standard manipulator form:
    M(q2) q̈  +  C(q2, q̇) q̇  +  G(q)  =  τ  +  Q_F

where:
    M  — symmetric positive-definite inertia matrix 
    C  — Coriolis/centrifugal matrix via Christoffel symbols 
    G  — gravitational torque vector
    Q_F — generalised force from external load at G4

All expressions follow the technical report (Vu, 2025/2026), which
derives the model from first principles using Lagrangian mechanics.

Notes
-----
The mass matrix depends only on q2 (not q1), because the parallelogram
constraint S1 = 0, S2 = 1 eliminates the base-angle dependence from all
inertia terms.
"""

import numpy as np
from dynamics.params import K1, K2, M11_0, meff1, meff2, g, a, b, L, e, h4, F_EXT


# ── Mass matrix ───────────────────────────────────────────────────────────

def mass_matrix(q2: float) -> np.ndarray:
    """
    Inertia matrix M(q2), shape (2, 2).

    M = [[M11_0 + 2·K1·cos(q2),  K2 + K1·cos(q2)],
         [K2 + K1·cos(q2),        K2             ]]

    M is symmetric positive-definite for all q2 when the parameters
    satisfy the condition K2·(M11_0 − K2) > K1².

    Parameters
    ----------
    q2 : float
        Relative link angle [rad].

    Returns
    -------
    M : np.ndarray, shape (2, 2)
    """
    c2 = np.cos(q2)
    return np.array([[M11_0 + 2 * K1 * c2,  K2 + K1 * c2],
                     [K2 + K1 * c2,          K2           ]])


# ── Coriolis / centrifugal term ───────────────────────────────────────────

def coriolis_qdot(q2: float, dq: np.ndarray) -> np.ndarray:
    """
    Coriolis and centrifugal term C(q2, q̇) · q̇, shape (2,).

    (C·q̇)₁ = −K1·sin(q2)·(2·q̇1·q̇2 + q̇2²)    
    (C·q̇)₂ = +K1·sin(q2)·q̇1²                   

    Parameters
    ----------
    q2 : float
        Relative link angle [rad].
    dq : np.ndarray, shape (2,)
        Joint velocities [rad/s].

    Returns
    -------
    Cqdot : np.ndarray, shape (2,)
    """
    dq1, dq2 = dq
    s2 = np.sin(q2)
    return np.array([-K1 * s2 * (2 * dq1 * dq2 + dq2**2),
                      K1 * s2 * dq1**2])


# ── Gravity vector ────────────────────────────────────────────────────────

def gravity_vector(q: np.ndarray) -> np.ndarray:
    """
    Gravitational torque vector G(q), shape (2,).

    G1 = g·[meff1·cos(q1) + meff2·cos(q1+q2)]    
    G2 = g·[meff2·cos(q1+q2)]                     

    Structurally identical to a 2-link planar serial arm, a direct
    consequence of the parallelogram constraint (Sec. 4.3 of the report).

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles [q1, q2] in radians.

    Returns
    -------
    G : np.ndarray, shape (2,)
    """
    q1, q2 = q
    return g * np.array([meff1 * np.cos(q1) + meff2 * np.cos(q1 + q2),
                          meff2 * np.cos(q1 + q2)])


# ── Forward kinematics of G4 ─────────────────────────────────────────────

def fk_G4(q: np.ndarray) -> np.ndarray:
    """
    Cartesian position of the end-effector G4 in the inertial frame .

    G4_body = (a + L·cos(q2) + h4,  L·sin(q2))
    G4_inertial = R(q1) · G4_body

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles [q1, q2] in radians.

    Returns
    -------
    G4 : np.ndarray, shape (2,)
        Position [x, y] in metres.
    """
    q1, q2 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    xB = a + L * np.cos(q2) + h4
    yB = L * np.sin(q2)
    return np.array([c1 * xB - s1 * yB,
                     s1 * xB + c1 * yB])


# ── Jacobian of G4 ───────────────────────────────────────────────────────

def jacobian_G4(q: np.ndarray) -> np.ndarray:
    """
    Geometric Jacobian J_G4 = ∂G4_inertial/∂q, shape (2, 2)
    Column 1: ∂G4/∂q1 (base-rotation contribution)
    Column 2: ∂G4/∂q2 (link-angle contribution)

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles [q1, q2] in radians.

    Returns
    -------
    J : np.ndarray, shape (2, 2)
    """
    q1, q2 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    arm = a + L * c2 + h4   # effective radius for base rotation

    J1 = np.array([-arm * s1 - L * s2 * c1,
                    arm * c1 - L * s2 * s1])   # ∂G4/∂q1
    J2 = np.array([-L * s12,
                    L * c12])                    # ∂G4/∂q2 
    return np.column_stack([J1, J2])             # shape (2, 2)


# ── Generalised force from external load ─────────────────────────────────

def qf_external(q: np.ndarray, F: np.ndarray = F_EXT) -> np.ndarray:
    """
    Generalised forces Q_F from an external force F = [Fx, Fy] at G4.

    Q_F = J_G4ᵀ · F 

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles [q1, q2] in radians.
    F : np.ndarray, shape (2,), optional
        External force [Fx, Fy] in Newtons.  Defaults to the constant
        F_EXT defined in params.py.

    Returns
    -------
    QF : np.ndarray, shape (2,)
        Generalised forces [N·m].
    """
    J = jacobian_G4(q)
    return J.T @ F


# ── Inverse dynamics ──────────────────────────────────────────────────────

def inverse_dynamics(q: np.ndarray, dq: np.ndarray,
                     ddq: np.ndarray, F: np.ndarray = F_EXT) -> np.ndarray:
    """
    Compute joint torques from the full inverse dynamics:

        τ = M(q2)·q̈ + C(q2, q̇)·q̇ + G(q) − Q_F(q, F)

    Parameters
    ----------
    q, dq, ddq : np.ndarray, shape (2,)
        Joint position [rad], velocity [rad/s], acceleration [rad/s²].
    F : np.ndarray, shape (2,), optional
        External force at G4 [N].  Defaults to F_EXT.

    Returns
    -------
    tau : np.ndarray, shape (2,)
        Joint torques [N·m].
    """
    M  = mass_matrix(q[1])
    Cq = coriolis_qdot(q[1], dq)
    G  = gravity_vector(q)
    QF = qf_external(q, F)
    return M @ ddq + Cq + G - QF


# ── Forward dynamics ──────────────────────────────────────────────────────

def forward_dynamics(q: np.ndarray, dq: np.ndarray,
                     tau: np.ndarray, F: np.ndarray = F_EXT) -> np.ndarray:
    """
    Compute joint accelerations from the forward dynamics:

        q̈ = M⁻¹(τ + Q_F − C·q̇ − G)   

    Parameters
    ----------
    q, dq : np.ndarray, shape (2,)
        Joint position [rad] and velocity [rad/s].
    tau : np.ndarray, shape (2,)
        Applied joint torques [N·m].
    F : np.ndarray, shape (2,), optional
        External force at G4 [N].  Defaults to F_EXT.

    Returns
    -------
    ddq : np.ndarray, shape (2,)
        Joint accelerations [rad/s²].
    """
    M  = mass_matrix(q[1])
    Cq = coriolis_qdot(q[1], dq)
    G  = gravity_vector(q)
    QF = qf_external(q, F)
    return np.linalg.solve(M, tau + QF - Cq - G)