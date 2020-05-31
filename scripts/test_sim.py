import time

from IPython import embed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np

import scipy.integrate
from scipy.interpolate import interp1d

import hook3
import pfh.glidersim as gsim
from pfh.glidersim import quaternion


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


class Dynamics6a:
    """Defines the state dynamics for a 6 DoF paraglider model."""

    # FIXME: I dislike this notation. It confuses the reference frames with
    #        the coordinate systems embedded in those frames.
    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("r_R2O", float, (3,)),  # The position of `R` in ned
        ("v_R2e", float, (3,)),  # The velocity of `R` in ned
    ]

    def __init__(
        self, glider, rho_air, delta_a=0, delta_bl=0, delta_br=0, delta_w=0, v_W2e=None,
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
        r_CP2O = x["r_R2O"] + quaternion.apply_quaternion_rotation(q_e2b, r_CP2R)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        a_R2e, alpha_b2e, solution = self.glider.accelerations(
            quaternion.apply_quaternion_rotation(x["q_b2e"], x["v_R2e"]),
            x["omega_b2e"],
            quaternion.apply_quaternion_rotation(x["q_b2e"], [0, 0, 9.8]),
            rho_air=self.rho_air(t),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            v_W2e=quaternion.apply_quaternion_rotation(x["q_b2e"], v_W2e),
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
        x_dot["v_R2e"] = quaternion.apply_quaternion_rotation(q_e2b, a_R2e)
        x_dot["omega_b2e"] = alpha_b2e

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


class Dynamics9a:
    """Defines the state dynamics for a 9 DoF paraglider model."""

    # FIXME: I dislike this notation. It confuses the reference frames with
    #        the coordinate systems embedded in those frames.
    state_dtype = [
        ("q_b2e", float, (4,)),  # Orientation: body/earth
        ("q_p2b", float, (4,)),  # Orientation: payload/body
        ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
        ("omega_p2e", float, (3,)),  # Angular velocity of the payload in payload frd
        ("r_R2O", float, (3,)),  # The position of `R` in ned
        ("v_R2e", float, (3,)),  # The velocity of `R` in ned
    ]

    def __init__(
        self, glider, rho_air, delta_a=0, delta_bl=0, delta_br=0, delta_w=0, v_W2e=None,
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
        state["q_p2b"] /= np.sqrt((state["q_p2b"] ** 2).sum())  # Normalize

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
        Theta_p2b = quaternion.quaternion_to_euler(x["q_p2b"])

        delta_a = self.delta_a(t)
        delta_w = self.delta_w(t)
        r_CP2R = self.glider.control_points(Theta_p2b, delta_a, delta_w)  # In body frd
        r_CP2O = x["r_R2O"] + quaternion.apply_quaternion_rotation(q_e2b, r_CP2R)
        v_W2e = self.v_W2e(t, r_CP2O)  # Wind vectors at each ned coordinate

        a_R2e, alpha_b2e, alpha_p2e, solution = self.glider.accelerations(
            quaternion.apply_quaternion_rotation(x["q_b2e"], x["v_R2e"]),
            x["omega_b2e"],
            x["omega_p2e"],
            Theta_p2b,  # FIXME: design review the call signature
            quaternion.apply_quaternion_rotation(x["q_b2e"], [0, 0, 9.8]),
            rho_air=self.rho_air(t),
            delta_a=delta_a,
            delta_bl=self.delta_bl(t),
            delta_br=self.delta_br(t),
            delta_w=delta_w,
            v_W2e=quaternion.apply_quaternion_rotation(x["q_b2e"], v_W2e),
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

        C_p2b = quaternion.quaternion_to_dcm(x["q_p2b"])
        omega_p2b = x["omega_p2e"] - C_p2b @ x["omega_b2e"]

        P, Q, R = omega_p2b
        # fmt: off
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        # fmt: on
        q_p2b_dot = 0.5 * Omega @ x["q_p2b"]

        x_dot = np.empty(1, self.state_dtype)
        x_dot["q_b2e"] = q_b2e_dot
        x_dot["q_p2b"] = q_p2b_dot
        x_dot["omega_b2e"] = alpha_b2e
        x_dot["omega_p2e"] = alpha_p2e
        x_dot["r_R2O"] = x["v_R2e"]
        x_dot["v_R2e"] = quaternion.apply_quaternion_rotation(q_e2b, a_R2e)

        # Use the solution as the reference_solution at the next time step
        params["solution"] = solution  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


# ---------------------------------------------------------------------------


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
    finally:
        if k < num_steps:  # Truncate if the simulation did not complete
            times = times[:k]
            path = path[:k]

    print(f"\nTotal simulation time: {time.perf_counter() - t_start}\n")

    return times, path


def main():
    # -----------------------------------------------------------------------
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
    rho_air = 1.2

    # -----------------------------------------------------------------------
    # Define the initial state for both models

    # Precomputed equilibrium states in body coordinates
    equilibrium_6a = {
        "Theta_b2e": [0, np.deg2rad(2.1676), 0],
        "v_R2e": [9.8783, 0, 1.2173],
    }

    equilibrium_9a = {
        "Theta_b2e": [0, np.deg2rad(2.6173), 0],
        "Theta_p2b": [0, np.deg2rad(-4.588), 0],
        "v_R2e": [9.739, 0, 1.1916],
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

    q_b2e_6a = quaternion.euler_to_quaternion(equilibrium_6a["Theta_b2e"])
    state_6a = np.empty(1, dtype=Dynamics6a.state_dtype)
    state_6a["q_b2e"] = q_b2e_6a
    state_6a["omega_b2e"] = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
    state_6a["r_R2O"] = [0, 0, 0]
    state_6a["v_R2e"] = quaternion.apply_quaternion_rotation(
        q_b2e_6a * [-1, 1, 1, 1], equilibrium_6a["v_R2e"],
    )

    q_b2e_9a = quaternion.euler_to_quaternion(equilibrium_9a["Theta_b2e"])
    state_9a = np.empty(1, dtype=Dynamics9a.state_dtype)
    state_9a["q_b2e"] = q_b2e_9a
    state_9a["q_p2b"] = quaternion.euler_to_quaternion(equilibrium_9a["Theta_p2b"])
    state_9a["omega_b2e"] = [0, 0, 0]
    state_9a["omega_p2e"] = [0, 0, 0]
    state_9a["r_R2O"] = [0, 0, 0]
    state_9a["v_R2e"] = quaternion.apply_quaternion_rotation(
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
    # delta_a = linear_control([(0, 0), (3, 0.75)])
    # delta_bl = linear_control([(0, 0), (3, 0.75)])
    # delta_br = linear_control([(0, 0), (3, 0.75)])
    T = 60

    # Scenario: short right turn
    # delta_a = 0.0
    # delta_bl = 0.0
    # delta_br = linear_control([(2, 0), (1, 0.80), (5, None), (2, 0)])
    # T = 20

    # Scenario: continuous right turn
    # delta_a = 0.0
    # delta_bl = 0.0
    # delta_br = linear_control([(2, 0), (3, 0.75),])
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
    # duration = 25  # Seconds per half-cycle
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
    model_6a = Dynamics6a(glider_6a, *common_args)
    model_6b = Dynamics6a(glider_6b, *common_args)
    model_6c = Dynamics6a(glider_6c, *common_args)
    model_9a = Dynamics9a(glider_9a, *common_args)
    model_9b = Dynamics9a(glider_9b, *common_args)

    # Choose which model to run
    # state0, model = state_6a, model_6a
    state0, model = state_6a, model_6b  # Same state as model_6a
    # state0, model = state_6a, model_6c  # Same state as model_6a
    # state0, model = state_9a, model_9a
    # state0, model = state_9a, model_9b  # Same state as model_9a

    # embed()
    # 1/0

    # -----------------------------------------------------------------------
    # Run the simulation

    Theta_b2e = quaternion.quaternion_to_euler(state0["q_b2e"])

    print("Preparing the simulation.")
    print("Initial state:")
    print("  Theta_b2e:", np.rad2deg(Theta_b2e).round(4))
    if "q_p2b" in state0.dtype.names:
        Theta_p2b = quaternion.quaternion_to_euler(state0["q_p2b"])[0]
        print("  Theta_p2b:", np.rad2deg(Theta_p2b).round(4))
    print("  omega_b2e:", state0["omega_b2e"][0].round(4))
    if "omega_p2e" in state0.dtype.names:
        print("  omega_p2e:", state0["omega_p2e"][0].round(4))
    print("      r_R2O:", state0["r_R2O"][0].round(4))
    print("      v_R2e:", state0["v_R2e"][0].round(4))

    dt = 0.10  # Time step for the `path` trajectory
    times, path = simulate(model, state0, dt=dt, T=T)

    # -----------------------------------------------------------------------
    # Extra values for verification/debugging

    K = len(times)
    LE0 = model.glider.wing.canopy_origin(0)  # FIXME: ignores `delta_a(t)`
    q_e2b = path["q_b2e"] * [1, -1, -1, -1]  # Applies C_ned/frd
    r_LE0 = path["r_R2O"] + quaternion.apply_quaternion_rotation(q_e2b, LE0)
    v_LE0 = path["v_R2e"] + quaternion.apply_quaternion_rotation(
        q_e2b, np.cross(path["omega_b2e"], LE0)
    )
    v_frd = quaternion.apply_quaternion_rotation(path["q_b2e"], path["v_R2e"])

    if "q_p2b" in path.dtype.names:  # 9 DoF model
        q_p2e = np.asarray([quaternion.quaternion_product(path["q_b2e"][k], path["q_p2b"][k]) for k in range(K)])
        Theta_p2b = quaternion.quaternion_to_euler(path["q_p2b"])  # [phi, theta, gamma]
        Theta_p2e = quaternion.quaternion_to_euler(q_p2e)  # FIXME: verify!
        r_P2O = path["r_R2O"] + quaternion.apply_quaternion_rotation(
            q_e2b,
            quaternion.apply_quaternion_rotation(
                path["q_p2b"] * [1, -1, -1, -1], model.glider.payload.control_points(),
            ),
        )
    else:  # 6 DoF model
        r_P2O = path["r_R2O"] + quaternion.apply_quaternion_rotation(
            q_e2b, model.glider.payload.control_points(),
        )

    # Euler derivatives (Stevens Eq:1.4-4)
    Theta_b2e = quaternion.quaternion_to_euler(path["q_b2e"])  # [phi, theta, gamma]
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

    # -----------------------------------------------------------------------
    # Plots

    # 3D Plot: Position over time
    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca(projection='3d')
    ax.invert_yaxis()
    ax.invert_zaxis()
    lpp = 0.25  # Line-plotting period [sec]
    for t in range(0, K, int(lpp / dt)):  # Draw connecting lines every `lpp` seconds
        p1, p2 = path["r_R2O"][t], r_LE0[t]  # Risers -> wing central LE
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=0.5, c='k')

        p1, p2 = path["r_R2O"][t], r_P2O[t]  # Risers -> payload
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=0.5, c='k')
    ax.plot(path["r_R2O"].T[0], path["r_R2O"].T[1], path["r_R2O"].T[2], label="risers")
    ax.plot(r_LE0.T[0], r_LE0.T[1], r_LE0.T[2], label="LE0")
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
