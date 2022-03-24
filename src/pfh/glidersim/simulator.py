"""
Models for simulating flight trajectories.

FIXME: explain *state dynamics* models
"""

from __future__ import annotations

import abc
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np
import scipy.integrate

from pfh.glidersim import orientation
from pfh.glidersim.util import cross3


if TYPE_CHECKING:
    from pfh.glidersim.paraglider import (
        ParagliderSystemDynamics6a,
        ParagliderSystemDynamics9a,
    )


__all__ = [
    "StateDynamics",
    "ParagliderStateDynamics6a",
    "ParagliderStateDynamics9a",
    "simulate",
    "prettyprint_state",
    "recompute_derivatives",
]


def __dir__():
    return __all__


# ---------------------------------------------------------------------------
# Dynamics Models


@runtime_checkable
class StateDynamics(Protocol):
    """
    Interface for classes that implement a StateDynamics model.

    StateDynamics are used to `simulate` state trajectories.
    """

    state_dtype: Any  # FIXME: declare properly and use it for type hinting

    @abc.abstractclassmethod
    def extra_arguments(self):
        """
        Any additional arguments used to compute the state derivatives.

        Returns
        -------
        tuple
            Additional arguments that the integrator should pass to the
            `StateDynamics.derivatives` method.
        """

    @abc.abstractmethod
    def cleanup(self, t: float, state) -> None:
        """
        Perform any necessary cleanup after each integration step.

        NOTE: mutating `state` will modify the integrator's internal state

        Parameters
        ----------
        t : float [s]
            The timestamp of the completed integration step.
        state : model.state_dtype
            The `state_dtype` of the state dynamics model.
        """

    @abc.abstractmethod
    def derivatives(self, t: float, state, *args):
        """
        Compute the state derivatives given a specific state at time `t`.

        The inputs to the system at time `t` are given the by the control
        functions internal to the model.

        Parameters
        ----------
        t : float [s]
            The current time
        state : model.state_dtype
            The current state
        args : tuple
            Extra arguments for computing the derivatives as defined by
            `StateDynamics.extra_arguments`.

        Returns
        -------
        state_dot : model.state_dtype
            The state derivatives
        """


class ParagliderStateDynamics6a(StateDynamics):
    """
    State dynamics for a 6 DoF paraglider model.

    Implements the choice of state variables and their associated dynamics
    outlined in :external+thesis:doc:`state_dynamics`.
    """

    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("r_RM2O", float, (3,)),  # The position of `RM` in ned
        ("v_RM2e", float, (3,)),  # The velocity of `RM` in ned
    ]

    def __init__(
        self,
        paraglider: ParagliderSystemDynamics6a,
        delta_a: float | Callable = 0,
        delta_bl: float | Callable = 0,
        delta_br: float | Callable = 0,
        delta_w: float | Callable = 0,
        rho_air: float | Callable = 1.225,
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
            v_W2e = np.asfarray(v_W2e)
            # FIXME: kludgy, assumes r.shape[-1] == 3
            self.v_W2e = lambda t, r: np.broadcast_to(
                v_W2e,
                (*np.broadcast_shapes(np.shape(t), np.shape(r)[:-1]), 3),
            )
        else:
            raise ValueError("`v_W2e` must be a callable or 3-tuple of float")

    def extra_arguments(self):
        # Note that `solution` is an in-out parameter to `derivatives`: each
        # aerodynamic solution is saved so it can be used as the proposal
        # solution at the next step.
        return ({"solution": None},)

    def cleanup(self, t: float, state):
        state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())  # Normalize

    def derivatives(self, t, state, params):
        q_e2b = state["q_b2e"] * [1, -1, -1, -1]  # Encodes `C_ned/frd`

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2RM = self.paraglider.r_CP2RM(delta_a, delta_w)  # In body frd
        r_CP2O = state["r_RM2O"] + orientation.quaternion_rotate(q_e2b, r_CP2RM)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        v_RM2e = orientation.quaternion_rotate(state["q_b2e"], state["v_RM2e"])
        a_RM2e, alpha_b2e, solution = self.paraglider.accelerations(
            v_RM2e,
            state["omega_b2e"],
            orientation.quaternion_rotate(state["q_b2e"], [0, 0, 9.8]),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            rho_air=self.rho_air(t),
            v_W2e=orientation.quaternion_rotate(state["q_b2e"], v_W2e),
            reference_solution=params["solution"],
        )
        a_RM2e += cross3(state["omega_b2e"], v_RM2e)  # In frame F_e

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
        gsim.simulator.ParagliderStateDynamics6a.state_dtype
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
        state["omega_b2e"] = [0, 0, 0]
        state["r_RM2O"] = [0, 0, 0]
        state["v_RM2e"] = orientation.quaternion_rotate(
            q_b2e * [1, -1, -1, -1],
            glider_eq["v_RM2e"],
        )
        state["v_RM2e"] += self.v_W2e(t=0, r=state["r_RM2O"])
        return state


class ParagliderStateDynamics9a(StateDynamics):
    """
    State dynamics for a 9 DoF paraglider model.

    Implements the choice of state variables and their associated dynamics
    outlined in :external+thesis:doc:`state_dynamics`.
    """

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
        paraglider: ParagliderSystemDynamics9a,
        delta_a: float | Callable = 0,
        delta_bl: float | Callable = 0,
        delta_br: float | Callable = 0,
        delta_w: float | Callable = 0,
        rho_air: float | Callable = 1.225,
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
            v_W2e = np.asfarray(v_W2e)
            # FIXME: kludgy, assumes r.shape[-1] == 3
            self.v_W2e = lambda t, r: np.broadcast_to(
                v_W2e,
                (*np.broadcast_shapes(np.shape(t), np.shape(r)[:-1]), 3),
            )
        else:
            raise ValueError("`v_W2e` must be a callable or 3-tuple of float")

    def extra_arguments(self):
        # Note that `solution` is an in-out parameter to `derivatives`: each
        # aerodynamic solution is saved so it can be used as the proposal
        # solution at the next step.
        return ({"solution": None},)

    def cleanup(self, t, state):
        state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())  # Normalize
        state["q_p2e"] /= np.sqrt((state["q_p2e"] ** 2).sum())  # Normalize

    def derivatives(self, t, state, params):
        q_e2b = state["q_b2e"] * [1, -1, -1, -1]  # Encodes `C_ned/frd`
        Theta_p2b = orientation.quaternion_to_euler(
            orientation.quaternion_product(q_e2b, state["q_p2e"])
        )

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2RM = self.paraglider.r_CP2RM(Theta_p2b, delta_a, delta_w)  # In body frd
        r_CP2O = state["r_RM2O"] + orientation.quaternion_rotate(q_e2b, r_CP2RM)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        v_RM2e = orientation.quaternion_rotate(state["q_b2e"], state["v_RM2e"])
        a_RM2e, alpha_b2e, alpha_p2e, solution = self.paraglider.accelerations(
            v_RM2e,
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
            reference_solution=params["solution"],
        )
        a_RM2e += cross3(state["omega_b2e"], v_RM2e)  # In frame F_e

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
        gsim.simulator.ParagliderStateDynamics9a.state_dtype
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
            q_b2e * [1, -1, -1, -1],
            glider_eq["v_RM2e"],
        )
        state["v_RM2e"] += self.v_W2e(t=0, r=state["r_RM2O"])
        return state


# ---------------------------------------------------------------------------
# Simulator


def simulate(
    model: StateDynamics,
    state0,
    T: float,
    dt: float,
    t0: float = 0.0,
):
    """
    Generate a state trajectory using the model's state derivatives.

    Parameters
    ----------
    model : StateDynamics
        The model that provides `state_dtype` and `derivatives`
    state0 : model.state_dtype
        The initial state
    T : float [seconds]
        The total simulation time
    dt : float [seconds]
        The simulation step size. This determines the time separation of each
        point in the state trajectory, but the RK4 integrator is free to use
        a different step size internally.
    t0 : float [seconds], optional
        The start time of the simulation. Useful for models with time varying
        behavior (eg, wind fields).

    Returns
    -------
    times : array of float, shape (K+1,) [seconds]
        The timestamp of each solution
    states : array of `model.state_dtype`, shape (K+1,)
        The state trajectory.
    """
    K = int(np.ceil(T / dt)) + 1  # Total number of states in the output
    times = np.empty(K)  # Simulation timestamps [sec]
    states = np.empty(K, dtype=model.state_dtype)
    times[0] = t0
    states[0] = state0

    def _flattened_derivatives(t, y, params):
        # Adapter function: the integrator requires a flat array of float
        state_dot = model.derivatives(t, y.view(model.state_dtype)[0], params)
        return state_dot.view(float)

    solver = scipy.integrate.ode(_flattened_derivatives)
    solver.set_integrator("dopri5", rtol=1e-5)
    solver.set_initial_value(state0.flatten().view(float), t0)
    solver.set_f_params(*model.extra_arguments())

    t_start = time.perf_counter()
    msg = ""
    k = 1  # Number of completed states (including the initial state)
    print("Running the simulation...")
    is_interactive = sys.stdout.isatty()
    try:
        while solver.successful() and k < K:
            if is_interactive:
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
            model.cleanup(solver.t, state)  # Modifies `solver._y`!!
            states[k] = state  # Makes a copy of `solver._y`
            times[k] = solver.t
            k += 1
    except RuntimeError as e:  # The model blew up
        print(f"\n--- Simulation failed: {type(e).__name__}:", e)
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted. ---")
    if is_interactive:
        print()

    # Truncate if the simulation did not complete
    if k < K:
        times = times[:k]
        states = states[:k]

    print(f"Total simulation time: {time.perf_counter() - t_start:.2f}")

    return times, states


def recompute_derivatives(model: StateDynamics, times, states):
    """
    Re-run the dynamics to access the state derivatives (accelerations).

    In `simulate` the derivatives are internal, but it can be helpful to know
    the accelerations at each time step. This is a bit of a kludge, but it's
    fast so good enough for now.

    Parameters
    ----------
    model : StateDynamics
        The state dynamics model that generated the derivatives.
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
    is_interactive = sys.stdout.isatty()
    for k in range(K):
        if is_interactive:
            print(f"\rStep: {k}/{K}", end="")
        derivatives[k] = model.derivatives(times[k], states[k], params)
    if is_interactive:
        print()

    return derivatives.view(model.state_dtype)


def prettyprint_state(state, header: str = None, footer: str = None) -> None:
    """
    Pretty-print a single instance of a StateDynamics.state_dtype

    Parameters
    ----------
    state : StateDynamics.state_dtype
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
