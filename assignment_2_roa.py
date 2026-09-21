import numpy as np
import matplotlib.pyplot as plt

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator



def plot_ankle_roa(thetas, thetadots, labels, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    extent = [thetas[0], thetas[-1], thetadots[0], thetadots[-1]]
    ax.imshow(labels, origin="lower", extent=extent, aspect="auto",
              cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    ax.set_title("RoA of feedback-linearization ankle controller\n(blue=diverges, red=converges to upright)")
    ax.axhline(0, color="k", lw=0.4, alpha=0.5)
    ax.axvline(0, color="k", lw=0.4, alpha=0.5)
    return ax



def closed_loop_outcome(state0, params, dt=0.001, t_max=8.0, tol=1e-3, bound=np.pi / 2):
    """Run the closed-loop (controller active) system from state0. Returns
    'converged' (reached the origin within tol by t_max), 'diverged'
    (theta left [-bound, bound]), or 'unresolved' (neither, by t_max --
    should be rare if t_max is long enough; worth checking for in your report)."""
    state = np.array(state0, dtype=float)
    p = dict(params)  # local copy so we don't mutate the caller's dict
    t = 0.0
    while t < t_max:
        p["ankle_torque"] = model.compute_ankle_torque(state, p)
        state = integrator.step(model.dynamics, t, state, dt, p)
        t += dt
        if abs(state[0]) > bound:
            return "diverged"
    return "converged" if (abs(state[0]) < tol and abs(state[1]) < tol) else "unresolved"
 
 
def sweep_ankle_roa(params, theta_range=(-0.2, 0.2), thetadot_range=(-0.6, 0.6),
                     n_theta=35, n_thetadot=35, **kwargs):
    thetas = np.linspace(*theta_range, n_theta)
    thetadots = np.linspace(*thetadot_range, n_thetadot)
    labels = np.zeros((n_thetadot, n_theta), dtype=int)
    for i, td in enumerate(thetadots):
        for j, th in enumerate(thetas):
            outcome = closed_loop_outcome([th, td], params, **kwargs)
            labels[i, j] = 1 if outcome == "converged" else 0
    return thetas, thetadots, labels
 
 
def build_roa_lookup(params, theta_range=(-0.2, 0.2), thetadot_range=(-0.6, 0.6),
                      n_theta=35, n_thetadot=35, **kwargs):
    thetas, thetadots, labels = sweep_ankle_roa(
        params, theta_range, thetadot_range, n_theta, n_thetadot, **kwargs
    )
    return dict(thetas=thetas, thetadots=thetadots, labels=labels)
 
 
def reached_stabilizing_roa(state, roa_lookup):
    theta, thetadot = state
    thetas, thetadots, labels = roa_lookup["thetas"], roa_lookup["thetadots"], roa_lookup["labels"]
    if not (thetas[0] <= theta <= thetas[-1] and thetadots[0] <= thetadot <= thetadots[-1]):
        return False
    j = np.argmin(np.abs(thetas - theta))
    i = np.argmin(np.abs(thetadots - thetadot))
    return bool(labels[i, j])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    extent = [thetas[0], thetas[-1], thetadots[0], thetadots[-1]]
    ax.imshow(labels, origin="lower", extent=extent, aspect="auto",
              cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    ax.set_title("RoA of feedback-linearization ankle controller\n(blue=diverges, red=converges to upright)")
    ax.axhline(0, color="k", lw=0.4, alpha=0.5)
    ax.axvline(0, color="k", lw=0.4, alpha=0.5)
    return ax
 
 
if __name__ == "__main__":
    params = dict(gravity=9.81, length=1.0, mass=1.0, incline=0.06,
                  angle_of_attack=np.pi / 8, ankle_torque=0.0)
 
    import time
    t0 = time.time()
    thetas, thetadots, labels = sweep_ankle_roa(params)
    print(f"sweep took {time.time()-t0:.1f}s, fraction converging: {labels.mean():.3f}")
 
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_ankle_roa(thetas, thetadots, labels, ax=ax)
    fig.tight_layout()
    fig.savefig("ankle_roa.png", dpi=150)
    print("saved ankle_roa.png")
 