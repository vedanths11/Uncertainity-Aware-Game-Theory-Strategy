import numpy as np
from scipy.integrate import solve_ivp

"""
Clohessy-Wilshire (Hill) Equations
    Assuming circular orbit (e=0) and small relative seperation w.r.t orbital semi major axis
"""

def hcw_prop(state, control, t, n):
    x0, y0, z0, vx0, vy0, vz0 = state
    a_x, a_y, a_z = control

    c, s = np.cos(n*t), np.sin(n*t)

    # Derived in derivations/hcw_dynamics.ipynb
    x = (a_x + 4*n**2*x0 + 2*n*(a_y*t + vy0) + (-2*a_y + n*vx0)*s - (a_x + 3*n**2*x0 + 2*n*vy0)*c)/n**2
    y = -2*a_x*t/n + 2*a_x*s/n**2 - 3*a_y*t**2/2 - 4*a_y*c/n**2 + 4*a_y/n**2 - 6*n*t*x0 - 3*t*vy0 + 6*x0*s + y0 + 2*vx0*c/n - 2*vx0/n + 4*vy0*s/n
    z = a_z/n**2 + vz0*s/n + (-a_z + n**2*z0)*c/n**2

    vx = (2*a_y*n + n*(-2*a_y + n*vx0)*c + n*(a_x + 3*n**2*x0 + 2*n*vy0)*s)/n**2
    vy = 2*a_x*c/n - 2*a_x/n - 3*a_y*t + 4*a_y*s/n + 6*n*x0*c - 6*n*x0 - 2*vx0*s + 4*vy0*c - 3*vy0
    vz = vz0*c - (-a_z + n**2*z0)*s/n

    return np.array([x, y, z, vx, vy, vz])

def hcw_state_space_prop(state, control, dt, n):
    c = np.cos(dt * n)
    s = np.sin(dt * n)

    # Derived in derivations/hcw_dynamics.ipynb
    A = np.array([
        [4 - 3*c, 0, 0, s/n, 2*(1-c)/n, 0],
        [-6*dt*n + 6*s, 1, 0, 2*(c-1)/n, -3*dt + 4*s/n, 0],
        [0, 0, c, 0, 0, s/n],
        [3*n*s, 0, 0, c, 2*s, 0],
        [6*n*(c-1), 0, 0, -2*s, 4*c-3, 0],
        [0, 0, -n*s, 0, 0, c]
    ], dtype=float)

    B = np.array([
        [(1-c)/n**2, 2*(dt*n-s)/n**2, 0],
        [2*(-dt*n+s)/n**2, (-3*dt**2*n**2 - 8*c + 8)/(2*n**2), 0],
        [0, 0, (1-c)/n**2],
        [s/n, 2*(1-c)/n, 0],
        [2*(c-1)/n, -3*dt + 4*s/n, 0],
        [0, 0, s/n]
    ], dtype=float)

    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)

    return A @ state + B @ control

"""
Non linear equations of motion propagation, LVLH Frame, discrete control input every timestep
"""

def truth_dynamics(state, a, e, mu, control):
    x, y, z, xd, yd, zd, nu = state
    ax, ay, az = control

    # Target orbit
    p = a * (1 - e**2)
    h = np.sqrt(mu * p)

    rT = p / (1 + e * np.cos(nu))
    rT_dot = (mu / h) * e * np.sin(nu)

    theta_dot = h / rT**2
    theta_ddot = -2 * rT_dot * theta_dot / rT

    # Chaser distance from Earth's center
    rC = np.sqrt((rT + x)**2 + y**2 + z**2)

    # Nonlinear relative accelerations
    xdd = (
        2 * theta_dot * yd
        + theta_ddot * y
        + theta_dot**2 * x
        + mu / rT**2
        - mu * (rT + x) / rC**3
        + ax
    )

    ydd = (
        -2 * theta_dot * xd
        - theta_ddot * x
        + theta_dot**2 * y
        - mu * y / rC**3
        + ay
    )

    zdd = -mu * z / rC**3 + az

    nu_dot = theta_dot

    return np.array([
        xd, yd, zd,
        xdd, ydd, zdd,
        nu_dot
    ])


def truth_model(rel_state0, nu0, tgrid, a, e, mu, controller, u_max=None):

    # Initial Relative State
    rel_state0 = np.asarray(rel_state0, dtype=float)
    tgrid = np.asarray(tgrid, dtype=float)

    # Augmented initial state with true anomaly
    state = np.append(rel_state0, nu0)

    N = len(tgrid)
    history = np.zeros((N, 6))
    nu_hist = np.zeros(N)
    controls = np.zeros((N - 1, 3))

    history[0] = state[:6]
    nu_hist[0] = nu0

    for k in range(N - 1):
        tk = tgrid[k]
        tk1 = tgrid[k + 1]

        # Calculate active control from the current state
        control = np.asarray(
            controller(state[:6].copy(), tk),
            dtype=float
        )

        # Actuator saturation
        if u_max is not None:
            control = np.clip(control, -u_max, u_max)

        controls[k] = control

        # Propagate nonlinear dynamics over this timestep. Control is held constant during this interval (Fair assumption for truth)
        sol = solve_ivp(
            truth_dynamics,
            (tk, tk1),
            state,
            args=(a, e, mu, control),
            method="DOP853",
            rtol=1e-10,
            atol=1e-12
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        state = sol.y[:, -1]

        history[k + 1] = state[:6]
        nu_hist[k + 1] = state[6]

    return history, nu_hist, controls