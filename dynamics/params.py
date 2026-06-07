"""
Physical parameters and composite Lagrangian constants for the 2-DOF
rotating parallelogram four-bar linkage.


    K1, K2, M11_0   — mass matrix constants 
    meff1, meff2    — effective mass parameters for gravity

Unit system: SI  (metres, kilograms, seconds, radians)
Gravitational acceleration: g = 9.81 m/s²

Notes
-----
The composite constants K1, K2, M11_0, meff1, meff2 are the only
observable combinations from torque measurements.  Individual body
parameters (m1, ms, m4, I1, Is, I4) cannot be recovered separately —
only their composites appear in the equations of motion.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

# ── Individual body parameters ────────────────────────────────────────────
m1  = 1.0    # kg   — Body 1 (rotating arm)
I1  = 0.05   # kg·m² — Body 1 moment of inertia about G1
ms  = 0.50   # kg   — mass of each side link (bodies 2 and 3)
Is  = 0.01   # kg·m² — side-link moment of inertia about its own CoM
m4  = 1.00   # kg   — coupler (body 4)
I4  = 0.01   # kg·m² — coupler moment of inertia about G4

# ── Geometry ──────────────────────────────────────────────────────────────
a   = 0.30   # m   — distance from O to G1 (Body 1 CoM)
b   = 0.13   # m   — distance from pivot to CoM along each side link
c   = 0.15   # m   — distance from CoM to distal end of side link
L   = b + c  # m   — total side-link length (= 0.28 m)
e   = 0.15   # m   — length of segment O2–O3 (coupler span)
h4  = 0.05   # m   — perpendicular offset from midpoint of AB to G4
g   = 9.81   # m/s² — gravitational acceleration

# ── Hardware joint limits ─────────────────────────────────────────────────
Q1_MIN = np.radians(-90.0)   # rad  — lower limit of base joint
Q1_MAX = np.radians( 30.0)   # rad  — upper limit of base joint
Q2_MIN = np.radians(-50.0)   # rad  — lower limit of relative link angle
Q2_MAX = np.radians( 50.0)   # rad  — upper limit of relative link angle

# ── Force sensor latency ──────────────────────────────────────────────────
Force_sensor_latency = 0.005   # s — delay between force update and controller

# ── Prismatic actuator attachment point (body-1 frame) ───────────────────
xB  = a + 0.05          # m — x-coordinate of attachment point Bp
yB  = 1.5 * e           # m — y-coordinate of attachment point Bp
dx  = xB - a            # m — offset Δx = xB − a
dy  = yB - e / 2        # m — offset Δy = yB − e/2

# ── Constant external force ───────────────────────────────────────────────
# A 10 N load applied at −30° from the horizontal at end-effector G4.
F_MAG   = 10.0                            # N
F_ANGLE = np.radians(-30.0)               # rad
F_EXT   = np.array([F_MAG * np.cos(F_ANGLE),   # Fx ≈ +8.660 N
                    F_MAG * np.sin(F_ANGLE)])   # Fy ≈ −5.000 N


def create_time_varying_force(
        t_min: float,
        t_max: float,
        mag_range: tuple = (5.0, 15.0),
        angle_deg_range: tuple = (-90.0, 30.0),
        n_segs: int = 9,
        seed=None,
):
    """
    Generate a continuously time-varying external force F(t) = [Fx(t), Fy(t)].

    The magnitude and direction each follow independent Pchip splines through
    random nodes with exponentially distributed widths, producing segments that
    alternate between fast and slow changes.

    Parameters
    ----------
    t_min, t_max : float
        Time interval [s].
    mag_range : (float, float)
        Min and max force magnitude [N].
    angle_deg_range : (float, float)
        Min and max force angle [degrees].
    n_segs : int
        Number of random segments (more → faster variation).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    F_func : callable
        F_func(t) → np.ndarray of shape (2,) giving [Fx, Fy] at time t.

    Notes
    -----
    Magnitude and angle are interpolated separately with PchipInterpolator
    to guarantee that |F(t)| stays within mag_range without overshoot.
    Interpolating Fx and Fy directly with a cubic spline can produce large
    overshoots between nodes, violating the intended bounds.
    """
    rng = np.random.default_rng(seed)

    # Exponentially distributed segment widths → mix of fast and slow segments
    widths      = rng.exponential(scale=1.0, size=n_segs)
    widths      = widths / widths.sum() * (t_max - t_min)
    t_nodes     = np.concatenate(([t_min], t_min + np.cumsum(widths)))
    t_nodes[-1] = t_max

    mags    = rng.uniform(*mag_range,                      size=len(t_nodes))
    angs_r  = np.deg2rad(rng.uniform(*angle_deg_range,     size=len(t_nodes)))

    mag_interp = PchipInterpolator(t_nodes, mags)
    ang_interp = PchipInterpolator(t_nodes, angs_r)

    def F_func(t: float) -> np.ndarray:
        m = float(mag_interp(t))
        a = float(ang_interp(t))
        return np.array([m * np.cos(a), m * np.sin(a)])

    return F_func


# ── Composite Lagrangian constants ───────────────────
K1     = 2 * ms * a * b + m4 * (a + h4) * L         
K2     = 2 * ms * b**2 + 2 * Is + m4 * L**2         
M11_0  = (m1 * a**2 + I1
          + 2 * ms * (a**2 + b**2 + e**2 / 4) + 2 * Is
          + m4 * ((a + h4)**2 + L**2) + I4)         
meff1  = (m1 + 2 * ms + m4) * a + m4 * h4      
meff2  = 2 * ms * b + m4 * L                   


if __name__ == "__main__":
    print(f"K1    = {K1:.5f}  kg·m²")
    print(f"K2    = {K2:.5f}  kg·m²")
    print(f"M11_0 = {M11_0:.5f}  kg·m²")
    print(f"meff1 = {meff1:.5f}  kg·m")
    print(f"meff2 = {meff2:.5f}  kg·m")
    print(f"F_EXT = [{F_EXT[0]:.3f}, {F_EXT[1]:.3f}] N")
    pd_ok = K2 * (M11_0 - K2) > K1**2
    print(f"\nPositive-definite check")
    print(f"  K2·(M11_0 − K2) = {K2*(M11_0-K2):.4f}  >  K1² = {K1**2:.4f}  →  {pd_ok}")
    if not pd_ok:
        print("  WARNING: mass matrix is NOT positive definite — check parameters!")