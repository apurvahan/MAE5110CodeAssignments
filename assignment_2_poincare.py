import numpy as np

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator


def integrate_until_any_guard(state0, guards, params, dt=0.001, t_max=5.0, bisect_tol=1e-10):
    state = np.array(state0, dtype=float)
    t = 0.0
    while t < t_max:
        next_state = integrator.step(model.dynamics, t, state, dt, params)
        for gi, guard_fn in enumerate(guards):
            if guard_fn(state, next_state, params):
                t_lo, x_lo = t, state
                t_hi, x_hi = t + dt, next_state
                for _ in range(60):
                    t_mid = 0.5 * (t_lo + t_hi)
                    x_mid = integrator.step(model.dynamics, t_lo, x_lo, t_mid - t_lo, params)
                    if guard_fn(x_lo, x_mid, params):
                        t_hi, x_hi = t_mid, x_mid
                    else:
                        t_lo, x_lo = t_mid, x_mid
                return gi, t_hi, x_hi
        state, t = next_state, t + dt
    return None, None, state


def poincare_step(thetadot_k, alpha, params, dt=0.001, t_max=5.0):
    p = dict(params)
    p["angle_of_attack"] = alpha
    p["ankle_torque"] = 0.0  

    
    which, _, x_impact = integrate_until_any_guard(
        [0.0, thetadot_k], [model.event_guard], p, dt=dt, t_max=t_max
    )
    if which is None:
        return None  

    x_reset = model.event_dynamics(x_impact, p)
    theta_start = x_reset[0]

    def success_guard(prev, nxt, params):
        return prev[0] < 0 <= nxt[0]

    def fail_guard(prev, nxt, params, theta_start=theta_start):
        return prev[0] >= theta_start > nxt[0]

    which, _, x_event = integrate_until_any_guard(
        x_reset, [success_guard, fail_guard], p, dt=dt, t_max=t_max
    )
    if which == 0:
        return x_event[1]  
    return None  # 