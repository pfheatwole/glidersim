import time

from IPython import embed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import scipy.integrate

import hook3
from pfh.glidersim import quaternion
from pfh.glidersim.util import cross3
from pfh.glidersim.plots import _set_axes_equal


class GliderSim:
    state_dtype = [
        ("q", float, (4,)),
        ("p", float, (3,)),
        ("v", float, (3,)),
        ("omega", float, (3,)),
    ]

    def __init__(self, glider, rho_air):
        self.glider = glider

        # FIXME: assumes J is constant. This is okay if rho_air is constant,
        #        and delta_a is zero (no weight shift deformations)
        self.J = glider.wing.inertia(rho_air=rho_air, N=5000)
        self.J_inv = np.linalg.inv(self.J)

        if np.isscalar(rho_air):
            self.rho_air = rho_air
        else:
            raise ValueError("Non-scalar rho_air is not yet supported")

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

        # Determine the environmental conditions
        # rho_air = self.rho_air(t, x["p"])
        rho_air = self.rho_air  # FIXME: support air density functions

        delta_a, delta_br, delta_bl = 0, 0, 0  # FIXME: time-varying input

        q_inv = x["q"] * [1, -1, -1, -1]  # for frd->ned
        # cps_frd = self.glider.control_points(delta_a)  # In body coordinates
        # cps = x["p"] + quaternion.apply_quaternion_rotation(x["q"], cps_frd)
        # v_w2e = self.wind(t, cps)  # Lookup the wind at each `ned` coordinate
        v_w2e = 0  # FIXME: implement wind lookups

        g = 9.8 * quaternion.apply_quaternion_rotation(x["q"], [0, 0, 1])
        # g = [0, 0, 0]  # Disable the gravity force

        # FIXME: Paraglider should return accelerations directly, not forces.
        #        The Glider might want to utilize appparent inertia, etc.
        v_frd = quaternion.apply_quaternion_rotation(x["q"], x["v"])
        F, M, solution = self.glider.forces_and_moments(
            v_frd,
            x["omega"],
            g,
            rho_air=rho_air,
            delta_a=delta_a,
            delta_bl=delta_bl,
            delta_br=delta_br,
            reference_solution=params["solution"],
        )

        # FIXME: what if Phillips fails? How do I abort gracefully?

        # Translational acceleration of the cm
        a_frd = F / self.glider.harness.mass  # FIXME: crude, incomplete
        a_ned = quaternion.apply_quaternion_rotation(q_inv, a_frd)

        # Angular acceleration of the body relative to the ned frame
        #  * ref: Stevens, Eq:1.7-5, p36 (50)
        alpha = self.J_inv @ (M - cross3(x["omega"], self.J @ x["omega"]))

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
        x_dot["omega"] = alpha

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


# ---------------------------------------------------------------------------


def simulate(model, state0, T=10, T0=0, dt=0.5, first_step=0.25, max_step=0.5):
    """

    Parameters
    ----------
    model
        The model that provides `dynamics`
    state0
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
    times : ndarray, shape (K+1, N)
    path : ndarray, shape (K+1, N)
        K : the number of discrete time values
        N : the number of state variables
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

    # For debugging
    q_inv = path["q"] * [1, -1, -1, -1]  # Applies C_ned/frd
    eulers = np.rad2deg(quaternion.quaternion_to_euler(path["q"]))  # [role, pitch, yaw] == [phi, theta, gamma]
    cps = model.glider.control_points(0)  # Control points in FRD (wing+harness)
    cp0 = cps[(cps.shape[0] - 1) // 2]  # The central control point in frd
    v_frd = quaternion.apply_quaternion_rotation(path["q"], path["v"])
    p_cp0 = path["p"] + quaternion.apply_quaternion_rotation(q_inv, cp0)
    v_cp0 = path["v"] + cross3(path["omega"], cp0)

    ax = plt.gca(projection='3d')
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.plot(path["p"].T[0], path["p"].T[1], path["p"].T[2], label="p_risers")
    ax.plot(p_cp0.T[0], p_cp0.T[1], p_cp0.T[2], label="p_cp0")
    for t in range(0, k, int(1/dt)):  # Draw connecting lines once per second
        p1, p2 = path["p"][t], p_cp0[t]
        ax.plot([p1.T[0], p2.T[0]], [p1.T[1], p2.T[1]], [p1.T[2], p2.T[2]], lw=0.5, c='k')
    ax.legend()
    _set_axes_equal(ax)
    plt.show()

    # ax = plt.gca()
    # mag_v_cp0 = np.linalg.norm(v_cp0, axis=1)
    # mag_v_frd = np.linalg.norm(v_frd, axis=1)
    # ax.plot(times, mag_v_cp0, marker='.', lw=0.75, label="v_cp0")
    # ax.plot(times, mag_v_frd, marker='.', lw=0.75, label="v_frd")
    # ax.set_ylim(0, max(mag_v_cp0.max(), mag_v_frd.max()) * 1.1)
    # ax.legend()
    # plt.show()

    # plt.plot(times, np.rad2deg(path["omega"]))
    # plt.ylabel("omega [deg]")


    embed()

    return times, path


def main():
    glider = hook3.build_hook3()
    rho_air = 1.2
    model = GliderSim(glider, rho_air=rho_air)

    # Start the wing in an equilibrium state (assuming zero wind)
    # alpha, Theta, V = glider.equilibrium_glide(0, 0, rho_air=rho_air)

    # Start the wing in a random state (non-equilibrium)
    alpha = np.deg2rad(8.5)
    beta = np.deg2rad(0)
    Theta = np.deg2rad(3)
    V = 10.5

    # Build some data
    UVW = V * np.asarray(
        [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)]
    )
    PQR = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]  # omega [rad/sec]
    euler = [np.deg2rad(0), Theta, np.deg2rad(0)]  # [phi, theta, gamma]

    # q = np.array([1, 0, 0, 0])  # The identity quaternion (zero attitude)
    q = quaternion.euler_to_quaternion(euler)  # Encodes C_frd/ned
    q_inv = q * [1, -1, -1, -1]  # Encodes C_ned/frd

    # Define the initial state
    state0 = np.empty(1, dtype=GliderSim.state_dtype)
    state0["q"] = q
    state0["p"] = [0, 0, 0]
    state0["v"] = quaternion.apply_quaternion_rotation(q_inv, UVW)
    state0["omega"] = PQR

    print("Preparing the simulation.")
    print("Initial state:")
    print("      q: ", state0["q"].round(4))
    print("  euler: ", np.rad2deg(euler).round(4))
    print("      p: ", state0["p"].round(4))
    print("      v: ", state0["v"].round(4))
    print("  omega: ", state0["omega"].round(4))

    # times05, path05 = simulate(model, state0, dt=0.05, T=300)
    # times10, path10 = simulate(model, state0, dt=0.10, T=300)
    times25, path25 = simulate(model, state0, dt=0.25, T=300)
    # times50, path50 = simulate(model, state0, dt=0.50, T=300)

    # For verification purposes
    # delta_PE = 9.8 * 75 * -path["p"].T[2]  # PE = mgh
    # KE_trans = 1/2 * 75 * np.linalg.norm(path["v"], axis=1)**2  # translational KE
    # KE_rot = 1/2 * np.einsum("ij,kj->k", J, path["omega"]**2)
    # delta_E = delta_PE + (KE_trans - KE_trans[0]) + KE_rot

    # embed()


if __name__ == "__main__":
    main()
