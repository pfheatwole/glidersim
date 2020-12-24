"""FIXME: add module docstring."""

import time

import numpy as np
import scipy.integrate

from pfh.glidersim import orientation


# ---------------------------------------------------------------------------
# Dynamics Models


class Dynamics6a:
    """Defines the state dynamics for a 6 DoF paraglider model."""

    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("r_R2O", float, (3,)),  # The position of `R` in ned
        ("v_R2e", float, (3,)),  # The velocity of `R` in ned
    ]

    def __init__(
        self,
        glider,
        rho_air,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        v_W2e=None,
    ):
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

        if callable(delta_w):
            self.delta_w = delta_w
        elif np.isscalar(delta_w):
            self.delta_w = lambda t: delta_w
        else:
            raise ValueError("`delta_w` must be a scalar or callable")

        if callable(v_W2e):
            self.v_W2e = v_W2e
        elif v_W2e is None:
            self.v_W2e = lambda t, r: np.zeros_like(r)
        else:
            raise ValueError("`v_W2e` must be a callable")

    def cleanup(self, state, t):
        # FIXME: hack that runs after each integration step. Assumes it can
        #        modify the integrator state directly.
        state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())  # Normalize

    def dynamics(self, t, y, params):
        """
        Compute the state derivatives from the model.

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
        q_e2b = x["q_b2e"] * [1, -1, -1, -1]  # Encodes `C_ned/frd`

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2R = self.glider.control_points(delta_a, delta_w)  # In body frd
        r_CP2O = x["r_R2O"] + orientation.quaternion_rotate(q_e2b, r_CP2R)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        a_R2e, alpha_b2e, solution = self.glider.accelerations(
            orientation.quaternion_rotate(x["q_b2e"], x["v_R2e"]),
            x["omega_b2e"],
            orientation.quaternion_rotate(x["q_b2e"], [0, 0, 9.8]),
            rho_air=self.rho_air(t),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            v_W2e=orientation.quaternion_rotate(x["q_b2e"], v_W2e),
            r_CP2R=r_CP2R,
            reference_solution=params["solution"],
        )

        # FIXME: what if Phillips fails? How do I abort gracefully?

        # Quaternion derivative
        #  * ref: Stevens, Eq:1.8-15, p51 (65)
        P, Q, R = x["omega_b2e"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_dot = 0.5 * Omega @ x["q_b2e"]

        x_dot = np.empty(1, self.state_dtype)
        x_dot["q_b2e"] = q_dot
        x_dot["r_R2O"] = x["v_R2e"]
        x_dot["v_R2e"] = orientation.quaternion_rotate(q_e2b, a_R2e)
        x_dot["omega_b2e"] = alpha_b2e

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


class Dynamics9a:
    """Defines the state dynamics for a 9 DoF paraglider model."""

    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("q_p2e", float, (4,)),  # Orientation: payload/earth
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("omega_p2e", float, (3,)),  # Angular velocity of the payload in payload frd
        ("r_R2O", float, (3,)),  # The position of `R` in ned
        ("v_R2e", float, (3,)),  # The velocity of `R` in ned
    ]

    def __init__(
        self,
        glider,
        rho_air,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        v_W2e=None,
    ):
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

        if callable(delta_w):
            self.delta_w = delta_w
        elif np.isscalar(delta_w):
            self.delta_w = lambda t: delta_w
        else:
            raise ValueError("`delta_w` must be a scalar or callable")

        if callable(v_W2e):
            self.v_W2e = v_W2e
        elif v_W2e is None:
            self.v_W2e = lambda t, r: np.zeros_like(r)
        else:
            raise ValueError("`v_W2e` must be a callable")

    def cleanup(self, state, t):
        # FIXME: hack that runs after each integration step. Assumes it can
        #        modify the integrator state directly.
        state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())  # Normalize
        state["q_p2e"] /= np.sqrt((state["q_p2e"] ** 2).sum())  # Normalize

    def dynamics(self, t, y, params):
        """
        Compute the state derivatives from the model.

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
        q_e2b = x["q_b2e"] * [-1, 1, 1, 1]  # Encodes `C_ned/frd`
        Theta_p2b = orientation.quaternion_to_euler(
            orientation.quaternion_product(q_e2b, x["q_p2e"])
        )

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2R = self.glider.control_points(Theta_p2b, delta_a, delta_w)  # In body frd
        r_CP2O = x["r_R2O"] + orientation.quaternion_rotate(q_e2b, r_CP2R)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        a_R2e, alpha_b2e, alpha_p2e, solution = self.glider.accelerations(
            orientation.quaternion_rotate(x["q_b2e"], x["v_R2e"]),
            x["omega_b2e"],
            x["omega_p2e"],
            Theta_p2b,  # FIXME: design review the call signature
            orientation.quaternion_rotate(x["q_b2e"], [0, 0, 9.8]),
            rho_air=self.rho_air(t),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            v_W2e=orientation.quaternion_rotate(x["q_b2e"], v_W2e),
            r_CP2R=r_CP2R,
            reference_solution=params["solution"],
        )

        # FIXME: what if Phillips fails? How do I abort gracefully?

        # Quaternion derivatives
        #  * ref: Stevens, Eq:1.8-15, p51 (65)
        P, Q, R = x["omega_b2e"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_b2e_dot = 0.5 * Omega @ x["q_b2e"]

        P, Q, R = x["omega_p2e"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_p2e_dot = 0.5 * Omega @ x["q_p2e"]

        x_dot = np.empty(1, self.state_dtype)
        x_dot["q_b2e"] = q_b2e_dot
        x_dot["q_p2e"] = q_p2e_dot
        x_dot["omega_b2e"] = alpha_b2e
        x_dot["omega_p2e"] = alpha_p2e
        x_dot["r_R2O"] = x["v_R2e"]
        x_dot["v_R2e"] = orientation.quaternion_rotate(q_e2b, a_R2e)

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


# ---------------------------------------------------------------------------
# Simulator


def simulate(model, state0, T=10, T0=0, dt=0.5, first_step=0.25, max_step=0.5):
    """
    Generate a state trajectory using the model dynamics.

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

    Theta_b2e = orientation.quaternion_to_euler(state0["q_b2e"])[0]

    print("\nPreparing the simulation...\n")
    print("Initial state:")
    print("  Theta_b2e:", np.rad2deg(Theta_b2e).round(4))
    if "q_p2b" in state0.dtype.names:
        Theta_p2b = orientation.quaternion_to_euler(state0["q_p2b"])[0]
        print("  Theta_p2b:", np.rad2deg(Theta_p2b).round(4))
    print("  omega_b2e:", state0["omega_b2e"][0].round(4))
    if "omega_p2e" in state0.dtype.names:
        print("  omega_p2e:", state0["omega_p2e"][0].round(4))
    print("      r_R2O:", state0["r_R2O"][0].round(4))
    print("      v_R2e:", state0["v_R2e"][0].round(4))
    print()

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
    print("Running the simulation...")
    try:
        while solver.successful() and k < num_steps:
            if k % 25 == 0:  # Update every 25 iterations
                avg_rate = (k - 1) / (time.perf_counter() - t_start)  # k=0 was free
                rem = (num_steps - k) / avg_rate  # Time remaining in seconds
                msg = f"ETA: {int(rem // 60)}m{int(rem % 60):02d}s"
            print(f"\rStep: {k} (t = {k*dt:.2f}). {msg}", end="")

            # WARNING: `solver.integrate` returns a *reference* to `_y`, so
            #          modifying `state` modifies `solver._y` directly.
            # FIXME: Is that valid for all `integrate` methods (eg, Adams)?
            #        Using `solver.set_initial_value` would reset the
            #        integrator, but that's not what we want here.
            state = solver.integrate(solver.t + dt).view(model.state_dtype)
            model.cleanup(state, solver.t)  # Modifies `solver._y`!!
            path[k] = state  # Makes a copy of `solver._y`
            times[k] = solver.t
            k += 1
    except RuntimeError as e:  # The model blew up
        # FIXME: refine this idea
        print(f"\n--- Simulation failed: {type(e).__name__}:", e)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted. ---")

    # Truncate if the simulation did not complete
    if k < num_steps:
        times = times[:k]
        path = path[:k]

    print(f"\nTotal simulation time: {time.perf_counter() - t_start}\n")

    return times, path


def recompute_derivatives(model, times, path):
    """
    Re-run the dynamics to access the state derivatives (accelerations).

    In `simulate` the derivatives are internal, but it can be helpful to know
    the accelerations at each time step. This is a bit of a kludge, but it's
    fast so good enough for now.

    Parameters
    ----------
    model : dynamics model
        For now, this is either Dynamics6a or Dynamics9a
    times : array of float
        The time value at each step.
    path : array of model.state_dtype
        The state of the simulation at each step.

    Returns
    -------
    derivatives : array of model.state_dtype
        The derivatives of the state variables at each step.
    """
    print("\nRe-running the dynamics to get the accelerations")
    N = len(times)
    derivatives = np.empty((N,), dtype=model.state_dtype)
    params = {"solution": None}  # Is modified by `model.dynamics`
    pf = path.view(float).reshape((N, -1))  # Ugly hack...
    for n in range(N):
        print(f"\r{n}/{N}", end="")
        derivatives[n] = model.dynamics(times[n], pf[n], params).view(model.state_dtype)
    print()

    return derivatives
