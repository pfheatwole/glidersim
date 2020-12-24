import time

from IPython import embed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np

from scipy.interpolate import interp1d

import hook3
import pfh.glidersim as gsim


# ---------------------------------------------------------------------------
# Simple wind "models"


class CircularThermal:

    def __init__(self, px, py, mag, radius5, t_start=0):
        """
        Parameters
        ----------
        px, py : float [m]
            The x and y coordinates of the thermal center
        mag : float [m/s]
            The magnitude of the thermal center
        radius95 : float [m]
            The distance from the center where the magnitude has dropped to 5%
        """
        self.c = np.array([px, py])
        self.mag = mag
        self.R = -radius5**2 / np.log(0.05)
        self.t_start = t_start

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        d2 = ((self.c - r[..., :2])**2).sum(axis=1)
        wind = np.zeros(r.shape)
        if t > self.t_start:
            wind[..., 2] = self.mag * np.exp(-d2 / self.R)
        return wind


class HorizontalShear:
    """
    Increasing vertical wind when traveling north. Transitions from 0 to `mag`
    as a sigmoid function. The transition is stretch using `smooth`.
    """
    def __init__(self, x_start, mag, smooth, t_start):
        self.x_start = x_start
        self.mag = mag
        self.smooth = smooth
        self.t_start = t_start

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        d = r[..., 0] - self.x_start
        wind = np.zeros(r.shape)
        if t > self.t_start:
            wind[..., 2] = (  # Sigmoid
                self.mag * np.exp(d / self.smooth) / (np.exp(d / self.smooth) + 1)
            )
        return wind


class LateralGust:
    """
    Adds an east-west gust. Linear rampump.
    """
    def __init__(self, t_start, t_ramp, t_duration, mag):
        t0 = 0
        t1 = t_start  # Start the ramp-up
        t2 = t1 + t_ramp  # Start the hold
        t3 = t2 + t_duration  # Start the ramp-down
        t4 = t3 + t_ramp  # Finish the ramp down
        times = [t0, t1, t2, t3, t4]
        values = [0, 0, mag, mag, 0]
        self._func = interp1d(times, values, bounds_error=False, fill_value=0)

    def __call__(self, t, r):
        wind = np.zeros(r.shape)
        wind[..., 1] = self._func(t)
        return wind


# ---------------------------------------------------------------------------


def linear_control(pairs):
    """
    Helper funtion to build linear interpolators for control inputs.

    The input is a sequence of tuples encoding  `(duration, value)`. An initial
    value can be set with a leading `(0, initial_value)` tuple. To "hold" a
    value, use `None` to repeat the previous value.

    For example, to ramp from 0 to 0.5 over the initial 15 seconds, then
    transition to 0.75 over a period of 2 seconds, hold for 10 seconds, then
    decrease to 0 over 10 seconds:

        pairs = [(15, 0.5), (2, 0.75), (10, None), (10, 0)]

    Parameters
    ----------
    pairs : list of 2-tuples of float
        Each tuple is (duration, value).
    """
    durations = np.array([t[0] for t in pairs])
    values = [t[1] for t in pairs]
    assert all(durations >= 0)
    for n, v in enumerate(values):  # Use `None` for "hold previous value"
        values[n] = v if v is not None else values[n - 1]
    times = np.cumsum(durations)
    c = interp1d(times, values, fill_value=(values[0], values[-1]), bounds_error=False)
    return c


def main():
    # Build the glider
    wing = hook3.build_hook3()
    harness = gsim.harness.Spherical(
        mass=75, z_riser=0.5, S=0.55, CD=0.8, kappa_w=0.1,
    )
    glider_6a = gsim.paraglider.Paraglider6a(wing, harness)
    glider_6b = gsim.paraglider.Paraglider6b(wing, harness)
    glider_6c = gsim.paraglider.Paraglider6c(wing, harness)
    glider_9a = gsim.paraglider.Paraglider9a(wing, harness)
    glider_9b = gsim.paraglider.Paraglider9b(wing, harness)
    glider_9c = gsim.paraglider.Paraglider9c(wing, harness)
    rho_air = 1.2

    # -----------------------------------------------------------------------
    # Define the initial state for both models

    # Precomputed equilibrium states
    equilibrium_6a = {
        "Theta_b2e": [0, np.deg2rad(2.170), 0],
        "v_R2e": [9.8595, 0, 1.2184],  # In body coordinates (frd)
    }

    equilibrium_9a = {
        "Theta_b2e": [0, np.deg2rad(2.6169), 0],
        "Theta_p2b": [0, np.deg2rad(-4.588), 0],
        "v_R2e": [9.7167, 0, 1.1938],  # In body coordinates (frd)
    }

    # Optional: recompute the equilibrium state
    # equilibrium_6a = glider_6a.equilibrium_state(
    #     delta_a=0,
    #     delta_b=0,
    #     alpha_0=np.deg2rad(9),
    #     theta_0=np.deg2rad(3),
    #     v_0=10,
    #     rho_air=1.2,
    # )
    # equilibrium_9a = glider_9a.equilibrium_state(
    #     delta_a=0,
    #     delta_b=0,
    #     alpha_0=np.deg2rad(9),
    #     theta_0=np.deg2rad(3),
    #     v_0=10,
    #     rho_air=1.2,
    # )

    q_b2e_6a = gsim.orientation.euler_to_quaternion(equilibrium_6a["Theta_b2e"])
    state_6a = np.empty(1, dtype=gsim.simulator.Dynamics6a.state_dtype)
    state_6a["q_b2e"] = q_b2e_6a
    state_6a["omega_b2e"] = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
    state_6a["r_R2O"] = [0, 0, 0]
    state_6a["v_R2e"] = gsim.orientation.quaternion_rotate(
        q_b2e_6a * [-1, 1, 1, 1], equilibrium_6a["v_R2e"],
    )

    q_b2e_9a = gsim.orientation.euler_to_quaternion(equilibrium_9a["Theta_b2e"])
    state_9a = np.empty(1, dtype=gsim.simulator.Dynamics9a.state_dtype)
    state_9a["q_b2e"] = q_b2e_9a
    q_p2b_9a = gsim.orientation.euler_to_quaternion(equilibrium_9a["Theta_p2b"])
    state_9a["q_p2e"] = gsim.orientation.quaternion_product(q_b2e_9a, q_p2b_9a)
    state_9a["omega_b2e"] = [0, 0, 0]
    state_9a["omega_p2e"] = [0, 0, 0]
    state_9a["r_R2O"] = [0, 0, 0]
    state_9a["v_R2e"] = gsim.orientation.quaternion_rotate(
        q_b2e_9a * [-1, 1, 1, 1], equilibrium_9a["v_R2e"],
    )

    # Optional: arbitrary modifications:
    # state_9a["q_b2e"] = np.array([1, 0, 0, 0])
    # state_9a["q_p2b"] = np.array([1, 0, 0, 0])

    # -----------------------------------------------------------------------
    # Build a test scenario
    #
    # FIXME: move these into "scenario" functions

    # Scenario: zero inputs
    # delta_a = 0.00
    # delta_bl = 0.0
    # delta_br = 0.0
    # T = 60

    # Scenario: constant inputs
    delta_a = 0.0
    delta_bl = 0.0
    delta_br = 0.0
    delta_w = 0.0
    T = 60

    # Scenario: short right turn
    # delta_a = 0.0
    # delta_bl = 0.0
    # delta_br = linear_control([(5, 0), (1, 0.80), (5, None), (2, 0)])
    # delta_w = linear_control([(2, 0), (3, 0.80),])
    # T = 20

    # Scenario: continuous right turn
    # delta_a = 0.0
    # delta_bl = 0.0
    # delta_br = linear_control([(10, 0), (3, 0.5),])
    # delta_w = linear_control([(2, 0), (2, 1.00),])
    # T = 60

    # Scenario: speedbar off-on-off
    # delta_a = linear_control([(2, 0), (5, 1.0), (10, None), (5, 0.0)])
    # delta_bl = 0.0
    # delta_br = 0.0
    # T = 30

    # Scenario: roll-yaw coupling w/ accelerator
    # delta_a = linear_control([(2, 0), 3, 1)])
    # delta_bl = 0.0
    # delta_br = linear_control([(10, 0), (8, 0.75)])  # Stress test: s/8/15
    # T = 60

    # Scenario: roll-yaw coupling with 5s brake pulse
    # delta_a = 1.0
    # delta_bl = 0.0
    # delta_br = linear_control([(20, 0), (2, 0.65), (5, None), (1, 0)])
    # T = 60

    # Scenario: smooth roll right then roll left
    # delta_a = 0.0
    # delta_br = linear_control([(2, 0), (2, 0.5), (10, None), (2, 0)])
    # delta_bl = linear_control([(16, 0), (3, 0.5)])
    # T = 60

    # Scenario: multiple figure-8
    # delta_a = 0.0
    # duration = 30  # Seconds per half-cycle
    # N_cycles = 2  # Total number of full cycles
    # on = [(2.0, 0.75), (duration - 2.0, None)]  # Braking on
    # off = [(1.0, 0), (duration - 1.0, None)]  # Braking off
    # delta_br = linear_control([(2, 0), *([*on, *off] * N_cycles)])
    # delta_bl = linear_control([(2, 0), *([*off, *on] * N_cycles)])
    # T = N_cycles * duration * 2

    # -----------------------------------------------------------------------
    # Add some wind

    v_W2e = None

    # Use with no brakes
    # v_W2e = CircularThermal(10*30, 0, -3, 10, t_start=30)
    # v_W2e = CircularThermal(10*30, 4, -3, 10, t_start=0)
    # v_W2e = CircularThermal(10*30, 5, 2, 15)  # Sink!

    # Use with `delta_br = 0.75`
    # v_W2e = CircularThermal(21, 65, -3, 10, t_start=30)
    # v_W2e = CircularThermal(21, 71, -3, 10, t_start=30)
    # v_W2e = CircularThermal(21, 78, -1.0, 8, t_start=30)
    # v_W2e = CircularThermal(21, 65, 2, 10, t_start=30)  # Sink!

    # v_W2e = HorizontalShear(10*30, -4, 25, 0)

    # v_W2e = LateralGust(2, 1, 3, 10*1.6/3.6)

    # -----------------------------------------------------------------------
    # Build the dynamics models

    common_args = (rho_air, delta_a, delta_bl, delta_br, delta_w, v_W2e)
    model_6a = gsim.simulator.Dynamics6a(glider_6a, *common_args)
    model_6b = gsim.simulator.Dynamics6a(glider_6b, *common_args)
    model_6c = gsim.simulator.Dynamics6a(glider_6c, *common_args)
    model_9a = gsim.simulator.Dynamics9a(glider_9a, *common_args)
    model_9b = gsim.simulator.Dynamics9a(glider_9b, *common_args)
    model_9c = gsim.simulator.Dynamics9a(glider_9c, *common_args)

    # Choose which model to run
    state0, model = state_6a, model_6a
    # state0, model = state_6a, model_6b  # Same state as model_6a
    # state0, model = state_6a, model_6c  # Same state as model_6a
    # state0, model = state_9a, model_9a
    # state0, model = state_9a, model_9b  # Same state as model_9a
    # state0, model = state_9a, model_9c  # Same state as model_9a

    # embed()
    # 1/0

    # -----------------------------------------------------------------------
    # Run the simulation

    Theta_b2e = gsim.orientation.quaternion_to_euler(state0["q_b2e"])

    print("Preparing the simulation.")
    print("Initial state:")
    print("  Theta_b2e:", np.rad2deg(Theta_b2e).round(4))
    if "q_p2b" in state0.dtype.names:
        Theta_p2b = gsim.orientation.quaternion_to_euler(state0["q_p2b"])[0]
        print("  Theta_p2b:", np.rad2deg(Theta_p2b).round(4))
    print("  omega_b2e:", state0["omega_b2e"][0].round(4))
    if "omega_p2e" in state0.dtype.names:
        print("  omega_p2e:", state0["omega_p2e"][0].round(4))
    print("      r_R2O:", state0["r_R2O"][0].round(4))
    print("      v_R2e:", state0["v_R2e"][0].round(4))

    t_start = time.perf_counter()
    dt = 0.10  # Time step for the `path` trajectory
    times, path = gsim.simulator.simulate(model, state0, dt=dt, T=T)

    # -----------------------------------------------------------------------
    # Extra values for verification/debugging

    K = len(times)
    if np.isscalar(delta_a):
        r_LE2R = -model.glider.wing.r_R2LE(delta_a)
    else:
        r_LE2R = -model.glider.wing.r_R2LE(delta_a(times))
    q_e2b = path["q_b2e"] * [1, -1, -1, -1]  # Applies C_ned/frd
    r_LE2O = path["r_R2O"] + gsim.orientation.quaternion_rotate(q_e2b, r_LE2R)
    v_LE2O = path["v_R2e"] + gsim.orientation.quaternion_rotate(
        q_e2b, np.cross(path["omega_b2e"], r_LE2R)
    )
    v_frd = gsim.orientation.quaternion_rotate(path["q_b2e"], path["v_R2e"])

    if "q_p2e" in path.dtype.names:  # 9 DoF model
        # FIXME: vectorize `gsim.orientation.quaternion_product`
        q_b2p = [
            gsim.orientation.quaternion_product(
                path["q_p2e"][k] * [-1, 1, 1, 1],
                path["q_b2e"][k],
            )
            for k in range(K)
        ]
        q_b2p = np.asarray(q_b2p)

        # FIXME: assumes the payload has only one control point (r_P2R^p)
        r_P2O = path["r_R2O"] + gsim.orientation.quaternion_rotate(
            q_e2b,
            gsim.orientation.quaternion_rotate(
                q_b2p,
                model.glider.payload.control_points(),
            ),
        )

        q_p2b = q_b2p * [-1, 1, 1, 1]
        Theta_p2b = gsim.orientation.quaternion_to_euler(q_p2b)
        Theta_p2e = gsim.orientation.quaternion_to_euler(path["q_p2e"])

    else:  # 6 DoF model
        r_P2O = path["r_R2O"] + gsim.orientation.quaternion_rotate(
            q_e2b, model.glider.payload.control_points(),
        )

    # Euler derivatives (Stevens Eq:1.4-4)
    Theta_b2e = gsim.orientation.quaternion_to_euler(path["q_b2e"])
    _0, _1 = np.zeros(K), np.ones(K)
    sp, st, sg = np.sin(Theta_b2e.T)
    cp, ct, cg = np.cos(Theta_b2e.T)
    tp, tt, tg = np.tan(Theta_b2e.T)
    T = np.array([[_1, sp * tt, cp * tt], [_0, cp, -sp], [_0, sp / ct, cp / ct]])
    T = np.moveaxis(T, -1, 0)
    Theta_b2e_dot = np.einsum("kij,kj->ki", T, path["omega_b2e"])

    print("\nRe-running the dynamics to get the accelerations")
    N = len(times)
    derivatives = np.empty((N,), dtype=model.state_dtype)
    params = {"solution": None}  # Is modified by `model.dynamics`
    pf = path.view(float).reshape((N, -1))  # Ugly hack...
    for n in range(N):
        print(f"\r{n}/{N}", end="")
        derivatives[n] = model.dynamics(times[n], pf[n], params).view(model.state_dtype)
    print()

    t_stop = time.perf_counter()
    print(f"\nTotal time: {t_stop - t_start:.2f}\n")

    print("Final state:", path[-1])

    # -----------------------------------------------------------------------
    # Plots

    # 3D Plot: Position over time
    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca(projection='3d')
    ax.invert_yaxis()
    ax.invert_zaxis()
    lpp = 0.25  # Line-plotting period [sec]
    for t in range(0, K, int(lpp / dt)):  # Draw connecting lines every `lpp` seconds
        p1, p2 = path["r_R2O"][t], r_LE2O[t]  # Risers -> wing central LE
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=0.5, c='k')

        p1, p2 = path["r_R2O"][t], r_P2O[t]  # Risers -> payload
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=0.5, c='k')
    ax.plot(path["r_R2O"].T[0], path["r_R2O"].T[1], path["r_R2O"].T[2], label="risers")
    ax.plot(r_LE2O.T[0], r_LE2O.T[1], r_LE2O.T[2], label="LE0")
    ax.plot(r_P2O.T[0], r_P2O.T[1], r_P2O.T[2], label="payload", lw=0.5, c='r')
    ax.legend()
    gsim.plots._set_axes_equal(ax)
    # plt.show()

    # Plot: velocity vs Time
    # mag_v_LE0 = np.linalg.norm(v_LE0, axis=1)
    # mag_v_frd = np.linalg.norm(v_frd, axis=1)
    # ax = plt.gca()
    # ax.plot(times, mag_v_LE0, marker='.', lw=0.75, label="v_LE0")
    # ax.plot(times, mag_v_frd, marker='.', lw=0.75, label="v_frd")
    # ax.set_ylim(0, max(mag_v_LE0.max(), mag_v_frd.max()) * 1.1)
    # ax.legend()
    # plt.show()

    # Plot: omega vs Time
    # plt.plot(times, np.rad2deg(path["omega_b2e"]))
    # plt.ylabel("omega [deg]")
    # plt.show()

    fig, ax = plt.subplots(3)
    ax[0].plot(times, np.rad2deg(Theta_b2e))
    ax[1].plot(times, np.rad2deg(path["omega_b2e"]))
    ax[2].plot(times, np.rad2deg(derivatives["omega_b2e"]))
    ax[0].set_ylabel("Theta_b2e [deg]")
    ax[1].set_ylabel("omega_b2e [deg]")
    ax[2].set_ylabel("alpha_b2e [deg]")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    plt.show()

    embed()


if __name__ == "__main__":
    main()
