"""FIXME: add module docstring."""

import time

import numpy as np
import scipy.integrate

from pfh.glidersim import orientation
from pfh.glidersim.util import _broadcast_shapes  # FIXME: stopgap


__all__ = [
    "Dynamics6a",
    "Dynamics9a",
    "simulate",
    "prettyprint_state",
    "recompute_derivatives",
]


def __dir__():
    return __all__


# ---------------------------------------------------------------------------
# Dynamics Models


class Dynamics6a:
    """Defines the state dynamics for a 6 DoF paraglider model."""

    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("r_RM2O", float, (3,)),  # The position of `RM` in ned
        ("v_RM2e", float, (3,)),  # The velocity of `RM` in ned
    ]

    def __init__(
        self,
        paraglider,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
    ):
        self.paraglider = paraglider

        def _wrap(name, val):
            if callable(val):
                return val
            elif np.isscalar(val):
                return lambda t: np.full(np.shape(t), val)
            else:
                raise ValueError(f"`{name}` must be a scalar or callable")

        self.delta_a = _wrap("delta_a", delta_a)
        self.delta_bl = _wrap("delta_bl", delta_bl)
        self.delta_br = _wrap("delta_br", delta_br)
        self.delta_w = _wrap("delta_w", delta_w)
        self.rho_air = _wrap("rho_air", rho_air)

        if callable(v_W2e):
            self.v_W2e = v_W2e
        elif np.shape(v_W2e) == (3,):
            v_W2e = np.asarray(v_W2e, dtype=float)
            # FIXME: kludgy, assumes r.shape[-1] == 3
            self.v_W2e = lambda t, r: np.broadcast_to(
                v_W2e,
                (*_broadcast_shapes(np.shape(t), np.shape(r)[:-1]), 3),
            )
        else:
            raise ValueError("`v_W2e` must be a callable or 3-tuple of float")

    def cleanup(self, state, t):
        # FIXME: hack that runs after each integration step. Assumes it can
        #        modify the integrator state directly.
        state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())  # Normalize

    def dynamics(self, t, state, params):
        """
        Compute the state derivatives from the model.

        Parameters
        ----------
        t : float [s]
            Time
        state : Dynamics6a.state_dtype
            The current state
        params : dictionary
            Any extra non-state parameters for computing the dynamics. Be aware
            that 'solution' is an in-out parameter: solutions for the current
            Gamma distribution are passed forward (output) to be used as the
            proposal to the next time step.

        Returns
        -------
        state_dot : Dynamics6a.state_dtype
            The state derivatives
        """
        q_e2b = state["q_b2e"] * [1, -1, -1, -1]  # Encodes `C_ned/frd`

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2RM = self.paraglider.control_points(delta_a, delta_w)  # In body frd
        r_CP2O = state["r_RM2O"] + orientation.quaternion_rotate(q_e2b, r_CP2RM)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        a_RM2e, alpha_b2e, solution = self.paraglider.accelerations(
            orientation.quaternion_rotate(state["q_b2e"], state["v_RM2e"]),
            state["omega_b2e"],
            orientation.quaternion_rotate(state["q_b2e"], [0, 0, 9.8]),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            rho_air=self.rho_air(t),
            v_W2e=orientation.quaternion_rotate(state["q_b2e"], v_W2e),
            r_CP2RM=r_CP2RM,
            reference_solution=params["solution"],
        )

        # FIXME: what if Phillips fails? How do I abort gracefully?

        # Quaternion derivative
        #  * ref: Stevens, Eq:1.8-15, p51 (65)
        P, Q, R = state["omega_b2e"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_dot = 0.5 * Omega @ state["q_b2e"]

        state_dot = np.empty(1, self.state_dtype)
        state_dot["q_b2e"] = q_dot
        state_dot["r_RM2O"] = state["v_RM2e"]
        state_dot["v_RM2e"] = orientation.quaternion_rotate(q_e2b, a_RM2e)
        state_dot["omega_b2e"] = alpha_b2e

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return state_dot

    def starting_equilibrium(self):
        """
        Compute the equilibrium state at `t = 0` assuming uniform local wind.

        In this case, "equilibrium" means "non-accelerating and no sideslip".
        In a uniform wind field, this steady-state definition requires
        symmetric brakes and no weight shift.

        Equilibrium is first calculated assuming zero wind. Steady-state is
        established by relative wind, so the local wind vector is merely an
        offset from the zero-wind steady-state. The non-zero local wind is
        included by adding it to the equilibrium `v_RM2e`.

        Returns
        -------
        gsim.simulator.Dynamics6a.state_dtype
            The equilibrium state
        """
        if not np.isclose(self.delta_bl(0), self.delta_br(0)):
            raise ValueError(
                "Asymmetric brake inputs at t=0. Unable to calculate equilibrium."
            )
        if not np.isclose(self.delta_w(0), 0):
            raise ValueError(
                "Non-zero weight shift control input. Unable to calculate equilibrium."
            )
        glider_eq = self.paraglider.equilibrium_state(
            delta_a=self.delta_a(0),
            delta_b=self.delta_bl(0),  # delta_bl == delta_br
            rho_air=self.rho_air(0),
        )
        q_b2e = orientation.euler_to_quaternion(glider_eq["Theta_b2e"])
        state = np.empty(1, dtype=self.state_dtype)[0]
        state["q_b2e"] = q_b2e
        state["omega_b2e"] = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
        state["r_RM2O"] = [0, 0, 0]
        state["v_RM2e"] = orientation.quaternion_rotate(
            q_b2e * [-1, 1, 1, 1],
            glider_eq["v_RM2e"],
        )
        state["v_RM2e"] += self.v_W2e(t=0, r=state["r_RM2O"])
        return state


class Dynamics9a:
    """Defines the state dynamics for a 9 DoF paraglider model."""

    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("q_p2e", float, (4,)),  # Orientation: payload/earth
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("omega_p2e", float, (3,)),  # Angular velocity of the payload in payload frd
        ("r_RM2O", float, (3,)),  # The position of `RM` in ned
        ("v_RM2e", float, (3,)),  # The velocity of `RM` in ned
    ]

    def __init__(
        self,
        paraglider,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
    ):
        self.paraglider = paraglider

        def _wrap(name, val):
            if callable(val):
                return val
            elif np.isscalar(val):
                return lambda t: val
            else:
                raise ValueError(f"`{name}` must be a scalar or callable")

        self.rho_air = _wrap("rho_air", rho_air)
        self.delta_a = _wrap("delta_a", delta_a)
        self.delta_bl = _wrap("delta_bl", delta_bl)
        self.delta_br = _wrap("delta_br", delta_br)
        self.delta_w = _wrap("delta_w", delta_w)

        if callable(v_W2e):
            self.v_W2e = v_W2e
        elif np.shape(v_W2e) == (3,):
            v_W2e = np.asarray(v_W2e, dtype=float)
            # FIXME: kludgy, assumes r.shape[-1] == 3
            self.v_W2e = lambda t, r: np.broadcast_to(
                v_W2e,
                (*_broadcast_shapes(np.shape(t), np.shape(r)[:-1]), 3),
            )
        else:
            raise ValueError("`v_W2e` must be a callable or 3-tuple of float")

    def cleanup(self, state, t):
        # FIXME: hack that runs after each integration step. Assumes it can
        #        modify the integrator state directly.
        state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())  # Normalize
        state["q_p2e"] /= np.sqrt((state["q_p2e"] ** 2).sum())  # Normalize

    def dynamics(self, t, state, params):
        """
        Compute the state derivatives from the model.

        Parameters
        ----------
        t : float [s]
            Time
        state : Dynamics9a.state_dtype
            The current state
        params : dictionary
            Any extra non-state parameters for computing the dynamics. Be aware
            that 'solution' is an in-out parameter: solutions for the current
            Gamma distribution are passed forward (output) to be used as the
            proposal to the next time step.

        Returns
        -------
        state_dot : Dynamics9a.state_dtype
            The state derivatives
        """
        q_e2b = state["q_b2e"] * [-1, 1, 1, 1]  # Encodes `C_ned/frd`
        Theta_p2b = orientation.quaternion_to_euler(
            orientation.quaternion_product(q_e2b, state["q_p2e"])
        )

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2RM = self.paraglider.control_points(Theta_p2b, delta_a, delta_w)  # In body frd
        r_CP2O = state["r_RM2O"] + orientation.quaternion_rotate(q_e2b, r_CP2RM)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        a_RM2e, alpha_b2e, alpha_p2e, solution = self.paraglider.accelerations(
            orientation.quaternion_rotate(state["q_b2e"], state["v_RM2e"]),
            state["omega_b2e"],
            state["omega_p2e"],
            Theta_p2b,  # FIXME: design review the call signature
            orientation.quaternion_rotate(state["q_b2e"], [0, 0, 9.8]),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            rho_air=self.rho_air(t),
            v_W2e=orientation.quaternion_rotate(state["q_b2e"], v_W2e),
            r_CP2RM=r_CP2RM,
            reference_solution=params["solution"],
        )

        # FIXME: what if Phillips fails? How do I abort gracefully?

        # Quaternion derivatives
        #  * ref: Stevens, Eq:1.8-15, p51 (65)
        P, Q, R = state["omega_b2e"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_b2e_dot = 0.5 * Omega @ state["q_b2e"]

        P, Q, R = state["omega_p2e"]
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_p2e_dot = 0.5 * Omega @ state["q_p2e"]

        state_dot = np.empty(1, self.state_dtype)
        state_dot["q_b2e"] = q_b2e_dot
        state_dot["q_p2e"] = q_p2e_dot
        state_dot["omega_b2e"] = alpha_b2e
        state_dot["omega_p2e"] = alpha_p2e
        state_dot["r_RM2O"] = state["v_RM2e"]
        state_dot["v_RM2e"] = orientation.quaternion_rotate(q_e2b, a_RM2e)

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return state_dot

    def starting_equilibrium(self):
        """
        Compute the equilibrium state at `t = 0` assuming uniform local wind.

        In this case, "equilibrium" means "non-accelerating and no sideslip".
        In a uniform wind field, this steady-state definition requires
        symmetric brakes and no weight shift.

        Equilibrium is first calculated assuming zero wind. Steady-state is
        established by relative wind, so the local wind vector is merely an
        offset from the zero-wind steady-state. The non-zero local wind is
        included by adding it to the equilibrium `v_RM2e`.

        Returns
        -------
        gsim.simulator.Dynamics9a.state_dtype
            The equilibrium state
        """
        if not np.isclose(self.delta_bl(0), self.delta_br(0)):
            raise ValueError(
                "Asymmetric brake inputs at t=0. Unable to calculate equilibrium."
            )
        if not np.isclose(self.delta_w(0), 0):
            raise ValueError(
                "Non-zero weight shift control input. Unable to calculate equilibrium."
            )
        glider_eq = self.paraglider.equilibrium_state(
            delta_a=self.delta_a(0),
            delta_b=self.delta_bl(0),  # delta_bl == delta_br
            rho_air=self.rho_air(0),
        )
        q_b2e = orientation.euler_to_quaternion(glider_eq["Theta_b2e"])
        state = np.empty(1, dtype=self.state_dtype)[0]
        state["q_b2e"] = q_b2e
        q_p2b = orientation.euler_to_quaternion(glider_eq["Theta_p2b"])
        state["q_p2e"] = orientation.quaternion_product(q_b2e, q_p2b)
        state["omega_b2e"] = [0, 0, 0]
        state["omega_p2e"] = [0, 0, 0]
        state["r_RM2O"] = [0, 0, 0]
        state["v_RM2e"] = orientation.quaternion_rotate(
            q_b2e * [-1, 1, 1, 1],
            glider_eq["v_RM2e"],
        )
        state["v_RM2e"] += self.v_W2e(t=0, r=state["r_RM2O"])
        return state


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
    states : array of `model.state_dtype`, shape (K+1,)
        The state trajectory.
    """

    K = int(np.ceil(T / dt)) + 1  # Total number of states in the output
    times = np.zeros(K)  # Simulation timestamps [sec]
    states = np.empty(K, dtype=model.state_dtype)
    states[0] = state0

    def _flattened_dynamics(t, y, params):
        # Adapter function: the integrator requires a flat array of float
        state_dot = model.dynamics(t, y.view(model.state_dtype)[0], params)
        return state_dot.view(float)

    solver = scipy.integrate.ode(_flattened_dynamics)
    solver.set_integrator("dopri5", rtol=1e-5, first_step=0.25, max_step=0.5)
    solver.set_initial_value(state0.flatten().view(float))
    solver.set_f_params({"solution": None})  # Is modified by `model.dynamics`

    t_start = time.perf_counter()
    msg = ""
    k = 1  # Number of completed states (including the initial state)
    print("Running the simulation...")
    try:
        while solver.successful() and k < K:
            if k % 25 == 0:  # Update every 25 iterations
                avg_rate = (k - 1) / (time.perf_counter() - t_start)  # k=0 was free
                rem = (K - k) / avg_rate  # Time remaining in seconds
                msg = f"ETA: {int(rem // 60)}m{int(rem % 60):02d}s"
            print(f"\rStep: {k}/{K} (t = {k*dt:.2f}). {msg}", end="")

            # WARNING: `solver.integrate` returns a *reference* to `_y`, so
            #          modifying `state` modifies `solver._y` directly.
            # FIXME: Is that valid for all `integrate` methods (eg, Adams)?
            #        Using `solver.set_initial_value` would reset the
            #        integrator, but that's not what we want here.
            state = solver.integrate(solver.t + dt).view(model.state_dtype)
            model.cleanup(state, solver.t)  # Modifies `solver._y`!!
            states[k] = state  # Makes a copy of `solver._y`
            times[k] = solver.t
            k += 1
    except RuntimeError as e:  # The model blew up
        # FIXME: refine this idea
        print(f"\n--- Simulation failed: {type(e).__name__}:", e)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted. ---")

    # Truncate if the simulation did not complete
    if k < K:
        times = times[:k]
        states = states[:k]

    print(f"\nTotal simulation time: {time.perf_counter() - t_start:.2f}")

    return times, states


def recompute_derivatives(model, times, states):
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
    states : array of model.state_dtype
        The state of the simulation at each step.

    Returns
    -------
    derivatives : array of model.state_dtype
        The derivatives of the state variables at each step.
    """
    print("Re-running the dynamics to get the accelerations")
    K = len(times)
    derivatives = np.empty((K,), dtype=model.state_dtype)
    params = {"solution": None}  # Is modified by `model.dynamics`
    for k in range(K):
        print(f"\rStep: {k}/{K}", end="")
        derivatives[k] = model.dynamics(times[k], states[k], params)
    print()

    return derivatives


def prettyprint_state(state, header=None, footer=None):
    """
    Pretty-print the `state_dtype` for `Dynamics6a` and `Dynamics9a`.

    Parameters
    ----------
    state : Dynamics6a.state_dtype or Dynamics9a.state_dtype
        The state to pretty-print.
    header : string, optional
        A string to print on a separate line preceding the states.
    footer : string, optional
        A string to print on a separate line after the states.

    Notes
    -----
    Don't rely on this function. It's here because I currently find it useful
    in some scripting, but overall I'm not a fan of hard-coding this
    information. Then again, I'm in crunch mode, so...
    """
    # FIXME: Review the existence/design of this function
    Theta_b2e = orientation.quaternion_to_euler(state["q_b2e"])
    with np.printoptions(precision=4, suppress=True):
        if header is not None:
            print(header)
        print("  Theta_b2e:", np.rad2deg(Theta_b2e))
        if "q_p2e" in state.dtype.names:
            Theta_p2e = orientation.quaternion_to_euler(state["q_p2e"])
            print("  Theta_p2e:", np.rad2deg(Theta_p2e))
        print("  omega_b2e:", state["omega_b2e"])
        if "omega_p2e" in state.dtype.names:
            print("  omega_p2e:", state["omega_p2e"])
        print("     r_RM2O:", state["r_RM2O"])
        print("     v_RM2e:", state["v_RM2e"])
        if footer is not None:
            print(footer)
