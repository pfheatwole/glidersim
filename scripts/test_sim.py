import time

from IPython import embed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import scipy.integrate
from scipy.interpolate import interp1d

import hook3
from pfh.glidersim import quaternion
from pfh.glidersim.util import cross3
from pfh.glidersim.plots import _set_axes_equal


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


class GliderSim:
    state_dtype = [
        ("q", float, (4,)),
        ("p", float, (3,)),
        ("v", float, (3,)),
        ("omega", float, (3,)),
    ]

    def __init__(self, glider, rho_air, delta_a=0, delta_bl=0, delta_br=0):
        self.glider = glider

        if callable(rho_air):
            self.rho_air = rho_air
        elif np.isscalar(rho_air):
            self.rho_air = lambda t: rho_air
        else:
            raise ValueError("`rho_air` must be a scalar or callable")

        if callable(delta_a):
            self.delta_a = delta_a
        elif np.isscalar(delta_a):
            self.delta_a = lambda t: delta_a
        else:
            raise ValueError("`delta_a` must be a scalar or callable")

        if callable(delta_bl):
            self.delta_bl = delta_bl
        elif np.isscalar(delta_bl):
            self.delta_bl = lambda t: delta_bl
        else:
            raise ValueError("`delta_bl` must be a scalar or callable")

        if callable(delta_br):
            self.delta_br = delta_br
        elif np.isscalar(delta_br):
            self.delta_br = lambda t: delta_br
        else:
            raise ValueError("`delta_br` must be a scalar or callable")

    def dynamics(self, t, y, params):
        """The state dynamics for the model

        Matches the `f(t, y, *params)` signature for scipy.integrate.ode

        Parameters
        ----------
        t : float [s]
            Time
        y : ndarray of float, shape (N,)
            The array of N state components
        params : dictionary
            Any extra non-state parameters for computing the dynamics. Be aware
            that 'solution' is an in-out parameter: solutions for the current
            Gamma distribution are passed forward (output) to be used as the
            proposal to the next time step.

        Returns
        -------
        x_dot : ndarray of float, shape (N,)
            The array of N state component derivatives
        """
        x = y.view(self.state_dtype)[0]  # The integrator uses a flat array

        # cps_frd = self.glider.control_points(delta_a)  # In body coordinates
        # cps = x["p"] + quaternion.apply_quaternion_rotation(x["q"], cps_frd)
        # v_w2e = self.wind(t, cps)  # Lookup the wind at each `ned` coordinate
        v_w2e = np.array([0, 0, 0])  # FIXME: implement wind lookups
        v_w2e = quaternion.apply_quaternion_rotation(x["q"], v_w2e)

        g = 9.8 * quaternion.apply_quaternion_rotation(x["q"], [0, 0, 1])
        # g = [0, 0, 0]  # Disable the gravity force

        v_frd = quaternion.apply_quaternion_rotation(x["q"], x["v"])
        a_frd, alpha_frd, solution = self.glider.accelerations(
            v_frd,
            x["omega"],
            g,
            rho_air=self.rho_air(t),
            delta_a=self.delta_a(t),
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            v_w2e=v_w2e,
            reference_solution=params["solution"],
        )

        # FIXME: what if Phillips fails? How do I abort gracefully?

        q_inv = x["q"] * [1, -1, -1, -1]  # Encodes `C_ned/frd`
        a_ned = quaternion.apply_quaternion_rotation(q_inv, a_frd)

        # `v_frd` (aka `UVW`) is the velocity of the reference point, which for
        # the Paraglider I've defined to be the origin (the midpoint between
        # the risers). Because the Paraglider rotates about it's center of
        # mass, not the origin, you need to include the angular velocity of the
        # paraglider when computing UVW.

        # Quaternion derivative
        #  * ref: Stevens, Eq:1.8-15, p51 (65)
        P, Q, R = x["omega"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_dot = 0.5 * Omega @ x["q"]

        x_dot = np.empty(1, self.state_dtype)
        x_dot["q"] = q_dot
        x_dot["p"] = x["v"]
        x_dot["v"] = a_ned
        x_dot["omega"] = alpha_frd

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


# ---------------------------------------------------------------------------


def simulate(model, state0, T=10, T0=0, dt=0.5, first_step=0.25, max_step=0.5):
    """

    Parameters
    ----------
    model
        The model that provides `dynamics` and `state_dtype`
    state0 : model.state_dtype
        The initial state
    T : float [seconds]
        The total simulation time
    T0 : float [seconds]
        The start time of the simulation. Useful for models with time varying
        behavior (eg, wind fields).
    dt : float [seconds]
        The simulation step size. This determines the time separation of each
        point in the state trajectory, but the RK4 integrator is free to use
        a different step size internally.

    Returns
    -------
    times : array of float, shape (K+1,) [seconds]
        The timestamp of each solution
    path : array of `model.state_dtype`, shape (K+1,)
        The state trajectory.
    """

    num_steps = int(np.ceil(T / dt)) + 1  # Include the initial state
    times = np.zeros(num_steps)  # The simulation times
    path = np.empty(num_steps, dtype=model.state_dtype)
    path[0] = state0

    solver = scipy.integrate.ode(model.dynamics)
    solver.set_integrator("dopri5", rtol=1e-5, first_step=0.25, max_step=0.5)
    solver.set_initial_value(state0.view(float))
    solver.set_f_params({"solution": None})  # Is modified by `model.dynamics`

    t_start = time.perf_counter()
    msg = ""
    k = 1  # Number of completed states (including the initial state)
    print("\nRunning the simulation.")
    try:
        while solver.successful() and k < num_steps:
            if k % 25 == 0:  # Update every 25 iterations
                avg_rate = (k - 1) / (time.perf_counter() - t_start)  # k=0 was free
                rem = (num_steps - k) / avg_rate  # Time remaining in seconds
                msg = f"ETA: {int(rem // 60)}m{int(rem % 60):02d}s"
            print(f"\rStep: {k} (t = {k*dt:.2f}). {msg}", end="")

            # WARNING: `solver.integrate` returns a *reference* to `_y`
            #          Modifying `state` modifies `solver.y` directly.
            # FIXME: normalizing `q` is a leaky abstraction
            # FIXME: Is that valid for all `integrate` methods (eg, Adams)?
            #        Using `solver.set_initial_value` would reset the
            #        integrator, but that's not what we want here.
            state = solver.integrate(solver.t + dt).view(model.state_dtype)
            state["q"] /= np.sqrt((state["q"] ** 2).sum())  # Normalize `q`
            path[k] = state  # Makes a copy of `solver._y`
            times[k] = solver.t
            k += 1
    except RuntimeError:  # The model blew up
        # FIXME: refine this idea
        print("\n--- Simulation failed. Terminating. ---")
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted. ---")
    finally:
        if k < num_steps:  # Truncate if the simulation did not complete
            times = times[:k]
            path = path[:k]

    print(f"\nTotal simulation time: {time.perf_counter() - t_start}\n")

    return times, path


def main():
    # -----------------------------------------------------------------------
    # Build the glider

    glider = hook3.build_hook3()
    rho_air = 1.2

    # -----------------------------------------------------------------------
    # Define the initial state

    # Option 1: Arbitrary state (this should be equilibrium Hook 3)
    alpha = np.deg2rad(8.86313992)
    beta = np.deg2rad(0)
    theta = np.deg2rad(2.06783323)
    v_mag = 10.32649163

    # Option 2: Approximate equilibrium state (neglects harness moment)
    # alpha, theta, v, _ = glider.equilibrium_glide(
    #     delta_a=0.0,
    #     delta_b=0,
    #     v_eq_proposal=10,
    #     rho_air=rho_air
    # )
    # beta = 0

    # Option 3: Equilibrium code (slow, but more accurate)
    # alpha, theta, v, _ = glider.equilibrium_glide2(
    #     delta_a=0,
    #     delta_b=0,
    #     alpha_0=np.deg2rad(9),
    #     theta_0=np.deg2rad(3),
    #     v_0=10,
    #     rho_air=1.2,
    # )
    # beta = 0

    v_R2e = v_mag * np.asarray(
        [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
    )
    omega_b2e = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]  # [rad/sec]
    euler = [np.deg2rad(0), theta, np.deg2rad(0)]  # [phi, theta, gamma]
    q = quaternion.euler_to_quaternion(euler)  # Encodes C_frd/ned
    q_inv = q * [1, -1, -1, -1]  # Encodes C_ned/frd

    # Define the initial state
    state0 = np.empty(1, dtype=GliderSim.state_dtype)
    state0["q"] = q
    state0["p"] = [0, 0, 0]
    state0["v"] = quaternion.apply_quaternion_rotation(q_inv, v_R2e)
    state0["omega"] = omega_b2e

    # -----------------------------------------------------------------------
    # Build a test scenario
    #
    # FIXME: move these into "scenario" functions

    # Scenario: continuous right turn
    delta_a = 0.0
    delta_bl = 0.0
    delta_br = linear_control([(5, 0), (5, 0.75)])
    T = 60

    # Scenario: roll-yaw coupling w/ accelerator
    # delta_a = 1.0
    # delta_bl = 0.0
    # delta_br = linear_control([(20, 0), (8, 0.65)])  # Stress test: s/8/15
    # T = 60

    # Scenario: roll-yaw coupling with 5s brake pulse
    # delta_a = 1.0
    # delta_bl = 0.0
    # delta_br = linear_control([(20, 0), (2, 0.65), (5, None), (1, 0)])
    # T = 60

    model = GliderSim(glider, rho_air, delta_a, delta_bl, delta_br)

    # -----------------------------------------------------------------------
    # Run the simulation

    print("Preparing the simulation.")
    print("Initial state:")
    print("      q: ", state0["q"].round(4))
    print("  euler: ", np.rad2deg(euler).round(4))
    print("      p: ", state0["p"].round(4))
    print("      v: ", state0["v"].round(4))
    print("  omega: ", state0["omega"].round(4))

    # Run the simulation
    dt = 0.1
    times, path = simulate(model, state0, dt=dt, T=T)

    # -----------------------------------------------------------------------
    # Extra values for verification/debugging

    k = len(times)
    q_inv = path["q"] * [1, -1, -1, -1]  # Applies C_ned/frd
    eulers = quaternion.quaternion_to_euler(path["q"])  # [phi, theta, gamma]
    cps = model.glider.wing.control_points(0)  # Wing control points in frd
    cp0 = cps[len(cps) // 2]  # The central control point in frd
    p_cp0 = path["p"] + quaternion.apply_quaternion_rotation(q_inv, cp0)
    v_cp0 = path["v"] + quaternion.apply_quaternion_rotation(q_inv, cross3(path["omega"], cp0))
    v_frd = quaternion.apply_quaternion_rotation(path["q"], path["v"])

    # Euler derivatives (Stevens Eq:1.4-4)
    _0, _1 = np.zeros(k), np.ones(k)
    sp, st, sg = np.sin(eulers.T)
    cp, ct, cg = np.cos(eulers.T)
    tp, tt, tg = np.tan(eulers.T)
    T = np.array([[_1, sp * tt, cp * tt], [_0, cp, -sp], [_0, sp / ct, cp / ct]])
    T = np.moveaxis(T, -1, 0)
    euler_dot = np.einsum("kij,kj->ki", T, path["omega"])

    # FIXME: these energy terms need review. Some are only for the harness,
    #        some are only for the wing. Not sure how useful they are anyway,
    #        since lots of energy is lost to the air mass.
    # delta_PE = 9.8 * 75 * -path["p"].T[2]
    # KE_trans = 0.5 * 75 * np.linalg.norm(path["v"], axis=1)**2
    # KE_rot = 0.5 * np.einsum("ij,kj->k", model.J, path["omega"]**2)
    # delta_E = delta_PE + (KE_trans - KE_trans[0]) + KE_rot

    # -----------------------------------------------------------------------
    # Plots

    # 3D Plot: Position over time
    ax = plt.gca(projection='3d')
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.plot(path["p"].T[0], path["p"].T[1], path["p"].T[2], label="p_risers")
    ax.plot(p_cp0.T[0], p_cp0.T[1], p_cp0.T[2], label="p_cp0")
    for t in range(0, k, int(1 / dt)):  # Draw connecting lines once per second
        p1, p2 = path["p"][t], p_cp0[t]
        ax.plot([p1.T[0], p2.T[0]], [p1.T[1], p2.T[1]], [p1.T[2], p2.T[2]], lw=0.5, c='k')
    ax.legend()
    _set_axes_equal(ax)
    plt.show()

    # Plot: velocity vs Time
    # mag_v_cp0 = np.linalg.norm(v_cp0, axis=1)
    # mag_v_frd = np.linalg.norm(v_frd, axis=1)
    # ax = plt.gca()
    # ax.plot(times, mag_v_cp0, marker='.', lw=0.75, label="v_cp0")
    # ax.plot(times, mag_v_frd, marker='.', lw=0.75, label="v_frd")
    # ax.set_ylim(0, max(mag_v_cp0.max(), mag_v_frd.max()) * 1.1)
    # ax.legend()
    # plt.show()

    # Plot: omega vs Time
    # plt.plot(times, np.rad2deg(path["omega"]))
    # plt.ylabel("omega [deg]")
    # plt.show()

    embed()


if __name__ == "__main__":
    main()
