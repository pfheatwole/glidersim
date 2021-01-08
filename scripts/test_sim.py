import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.interpolate import interp1d

import pfh.glidersim as gsim


# ---------------------------------------------------------------------------
# Simple wind "models"


class CircularThermal:
    """
    Functor to create circular thermals at specific <x,y> coordinates.

    Parameters
    ----------
    px, py : float [m]
        The x and y coordinates of the thermal center
    mag : float [m/s]
        The magnitude of the thermal center
    radius95 : float [m]
        The distance from the center where the magnitude has dropped to 5%
    """

    def __init__(self, px, py, mag, radius5, t_start=0):
        self.c = np.array([px, py])
        self.mag = mag
        self.R = -(radius5 ** 2) / np.log(0.05)
        self.t_start = t_start

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        d2 = ((self.c - r[..., :2]) ** 2).sum(axis=1)
        wind = np.zeros(r.shape)
        if t > self.t_start:
            wind[..., 2] = self.mag * np.exp(-d2 / self.R)
        return wind


class HorizontalShear:
    """
    Functor to create increasing vertical wind when traveling north.

    Transitions from 0 to `mag` as a sigmoid function. The transition is
    stretch using `smooth`.

    Parameters
    ----------
    x_start : float [m]
        Northerly position to begin the sigmoid transition.
    mag : float [m/s]
        The peak vertical windspeed.
    smooth : float
        Scaling factor to stretch the transition. FIXME: explain (I forget!)
    t_start : float [sec]
        The time at which to enable this wind component.
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
    Functor to create a global east-to-west gust with linear ramps up and down.

    Parameters
    ----------
    t_start : float [sec]
        Time to start the ramp up.
    t_ramp : float [sec]
        Time duration for the linear ramps up/down to/from peak magnitude.
    t_duration : float [sec]
        Time to hold the maximum magnitude gust.
    mag : float [m/s]
        The peak gust magnitude.
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
# Build a set "scenario" inputs


def linear_control(pairs):
    """
    Helper function to build linear interpolators for control inputs.

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


def zero_controls(T=20):
    """Scenario: zero_inputs."""
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": 0,
        "delta_w": 0,
        "v_W2e": None,
    }
    return inputs, T


def symmetric_brakes_fast_on():
    """Scenario: zero_inputs."""
    t_start = 2
    t_rise = 0.5
    braking = linear_control([(t_start, 0), (t_rise, 1)])
    inputs = {
        "delta_a": 0,
        "delta_bl": braking,
        "delta_br": braking,
        "delta_w": 0,
        "v_W2e": None,
    }
    T = 10
    return inputs, T


def symmetric_brakes_fast_off():
    """Scenario: zero_inputs."""
    t_start = 2
    t_rise = 3
    t_hold = 10
    t_fall = 0.5
    braking = linear_control([(t_start, 0), (t_rise, 1), (t_hold, None), (t_fall, 0)])
    inputs = {
        "delta_a": 0,
        "delta_bl": braking,
        "delta_br": braking,
        "delta_w": 0,
        "v_W2e": None,
    }
    T = t_start + t_rise + t_hold + t_fall + 10
    return inputs, T


def symmetric_brakes_fast_on_off():
    """Scenario: zero_inputs."""
    t_start = 2
    t_rise = 2
    t_hold = 2
    t_fall = 0.5
    braking = linear_control([
        (t_start, 0),
        (t_rise, 1),
        (t_hold, None),
        (t_fall, 0)
    ])
    inputs = {
        "delta_a": 0,
        "delta_bl": braking,
        "delta_br": braking,
        "delta_w": 0,
        "v_W2e": None,
    }
    T = t_start + t_rise + t_hold + t_fall + 10
    return inputs, T


def short_right_turn_without_weightshift():
    """Scenario: short right turn."""
    t_start = 2
    t_rise = 4
    t_hold = 5
    t_fall = 1
    t_settle = 5
    mag = 1
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": linear_control([
            (t_start, 0),
            (t_rise, mag),
            (t_hold, None),
            (t_fall, 0),
        ]),
        "delta_w": 0,
        "v_W2e": None,
    }
    T = t_start + t_rise + t_hold + t_fall + t_settle
    return inputs, T


def short_right_turn_with_weightshift():
    """Scenario: short right turn."""
    t_start = 2
    t_warmup = 10
    t_rise = 1
    t_hold = 5
    t_fall = 1
    t_settle = 5
    mag = 0.75
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": linear_control([
            (t_start + t_warmup, 0),
            (t_rise, mag),
            (t_hold, None),
            (t_fall, 0),
        ]),
        "delta_w": linear_control([(t_start, 0), (2, 0.75)]),
        "v_W2e": None,
    }
    T = t_start + t_rise + t_hold + t_fall + t_settle
    return inputs, T


def continuous_right_turn_without_weightshift(mag=0.75, T=60):
    """Scenario: continuous right turn without weightshift."""
    t_start = 5
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": linear_control([(t_start, 0), (3, mag)]),
        "delta_w": 0,
        "v_W2e": None,
    }
    T = 60
    return inputs, T


def continuous_right_turn_with_weightshift(mag=0.75):
    """Scenario: continuous right turn with weightshift."""
    t_start = 2
    t_warmup = 5
    t_rise = 1
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": linear_control([(t_start + t_rise + t_warmup, 0), (t_rise, mag)]),
        "delta_w": linear_control([(t_start, 0), (t_rise, 1.00)]),
        "v_W2e": None,
    }
    T = 60
    return inputs, T


def thermal_zero_controls(py=0, mag=-3, radius5=10):
    """
    Place a thermal in the path of a glider flying hands-up.

    Parameters
    ----------
    py : float [m]
        The y-axis (easterly) offset of the thermal
    mag : float [m/s]
        The strength of the thermal core.
    radius5 : float [m]
        The distance at which the thermal strength has reduced to 5%.
    """
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": 0,
        "delta_w": 0,
        "v_W2e": CircularThermal(
            px=10 * 10,  # At 10m/s, roughly 10 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_start=0,
        )
    }
    T = 20
    return inputs, T


def centered_thermal_with_symmetric_brake(py=0, mag=-3, radius5=10):
    """
    Place a thermal in the path of a glider flying with symmetric brakes.

    Parameters
    ----------
    py : float [m]
        The y-axis (easterly) offset of the thermal
    mag : float [m/s]
        The strength of the thermal core.
    radius5 : float [m]
        The distance at which the thermal strength has reduced to 5%.
    """
    t_start = 2
    t_rise = 2
    mag = 0.75
    inputs = {
        "delta_a": 0,
        "delta_bl": linear_control([(t_start, 0), (t_rise, mag)]),
        "delta_br": linear_control([(t_start, 0), (t_rise, mag)]),
        "delta_w": 0,
        "v_W2e": CircularThermal(
            px=10 * 10,  # At 10m/s, roughly 10 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_start=0,
        )
    }
    T = 20
    return inputs, T


def centered_thermal_with_accelerator(py=0, mag=-3, radius5=10):
    """
    Place a thermal in the path of a glider flying with accelerator applied.

    Parameters
    ----------
    py : float [m]
        The y-axis (easterly) offset of the thermal
    mag : float [m/s]
        The strength of the thermal core.
    radius5 : float [m]
        The distance at which the thermal strength has reduced to 5%.
    """
    t_start = 2
    t_rise = 2
    mag = 0.75
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": 0,
        "delta_w": linear_control([(t_start, 0), (t_rise, mag)]),
        "v_W2e": CircularThermal(
            px=10 * 10,  # At 10m/s, roughly 10 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_start=0,
        )
    }
    T = 20
    return inputs, T


def roll_yaw_coupling_with_accelerator():
    """
    Scenario: roll-yaw coupling w/ accelerator.

    Purpose: observe how accelerator increases roll-yaw coupling when brakes
    are applied.

    Notes: not sure how representative this is of a real wing since a real wing
    experiences distorions (eg, profile flattening) when the accelerator is
    applied.
    """
    t_start = 2
    t_warmup = 5
    t_rise = 1.5
    t_hold = 3
    t_fall = 1.5
    inputs = {
        "delta_a": linear_control([(t_start, 0), (t_rise, 0.75)]),
        "delta_bl": 0,
        "delta_br": linear_control([
            (t_start + t_warmup, 0),
            (t_rise, 0.75),
            (t_hold, None),
            (t_fall, 0)
        ]),
        "delta_w": 0,
        "v_W2e": None,
    }
    T = t_start + t_warmup + t_hold + t_fall + 5
    return inputs, T


def roll_right_then_left():
    """Scenario: smooth roll right then roll left."""
    inputs = {
        "delta_a": 0,
        "delta_br": linear_control([(2, 0), (2, 0.5), (10, None), (2, 0)]),
        "delta_bl": linear_control([(16, 0), (3, 0.5)]),
        "delta_w": 0,
        "v_W2e": None,
    }
    T = 20
    return inputs, T


def figure_8s(N_cycles=2, duration=30, mag=0.75):
    """
    Scenario: multiple figure-8s.

    Parameters
    ----------
    N_cycles : int
        How many cycles of left+right braking.
    duration : int [sec]
        Seconds per half-cycle.
    mag : float
        Magnitude of braking applied.
    """
    on = [(2.0, mag), (duration - 2.0, None)]  # Braking on
    off = [(1.0, 0), (duration - 1.0, None)]  # Braking off
    inputs = {
        "delta_a": 0,
        "delta_br": linear_control([(2, 0), *([*on, *off] * N_cycles)]),
        "delta_bl": linear_control([(2, 0), *([*off, *on] * N_cycles)]),
        "delta_w": 0,
        "v_W2e": None,
    }
    T = N_cycles * duration * 2
    return inputs, T


def horizontal_shear_zero_controls():
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": 0,
        "delta_w": 0,
        "v_W2e": HorizontalShear(
            x_start=10 * 10,
            mag=-4,
            smooth=25,
            t_start=0,
        ),
    }
    T = 20
    return inputs, T


def lateral_gust_zero_controls():
    mag = 10  # [mph]
    inputs = {
        "delta_a": 0,
        "delta_bl": 0,
        "delta_br": 0,
        "delta_w": 0,
        "v_W2e": LateralGust(
            t_start=2,
            t_ramp=1,
            t_duration=3,
            mag=mag * 1.6 / 3.6,  # [m/s]
        ),
    }
    T = 20
    return inputs, T


# ---------------------------------------------------------------------------


def main():

    # -----------------------------------------------------------------------
    # Build a set of glider models from a common base configuration

    wing = gsim.extras.wings.build_hook3(verbose=False)
    harness = gsim.harness.Spherical(
        mass=75,
        z_riser=0.5,
        S=0.55,
        CD=0.8,
        kappa_w=0.1,
    )
    use_apparent_mass = True

    # 6 DoF models
    glider_6a = gsim.paraglider.Paraglider6a(
        wing,
        harness,
        use_apparent_mass=use_apparent_mass,
    )
    glider_6b = gsim.paraglider.Paraglider6b(wing, harness)  # No apparent mass
    glider_6c = gsim.paraglider.Paraglider6c(wing, harness)  # No apparent mass

    # Coefficients for the spring-damper connection (9DoF models)
    # FIXME: naming?
    kappa_RM = [-100, 0, -10]  # Coefficients for Theta_p2b
    kappa_RM_dot = [-50, -5, -50]  # Coefficients for dot{Theta_p2b}

    # 9 DoF models
    glider_9a = gsim.paraglider.Paraglider9a(
        wing,
        harness,
        kappa_RM=kappa_RM,
        kappa_RM_dot=kappa_RM_dot,
        use_apparent_mass=use_apparent_mass,
    )
    glider_9b = gsim.paraglider.Paraglider9b(
        wing,
        harness,
        kappa_RM=kappa_RM,
        kappa_RM_dot=kappa_RM_dot,
        # No apparent mass
    )
    glider_9c = gsim.paraglider.Paraglider9c(
        wing,
        harness,
        kappa_RM=kappa_RM,
        kappa_RM_dot=kappa_RM_dot,
        use_apparent_mass=use_apparent_mass,
    )

    # -----------------------------------------------------------------------
    # Define the initial states for both models from precomputed equilibriums

    equilibrium_6a = glider_6a.equilibrium_state()
    equilibrium_9a = glider_9a.equilibrium_state()

    q_b2e_6a = gsim.orientation.euler_to_quaternion(equilibrium_6a["Theta_b2e"])
    state_6a = np.empty(1, dtype=gsim.simulator.Dynamics6a.state_dtype)
    state_6a["q_b2e"] = q_b2e_6a
    state_6a["omega_b2e"] = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
    state_6a["r_RM2O"] = [0, 0, 0]
    state_6a["v_RM2e"] = gsim.orientation.quaternion_rotate(
        q_b2e_6a * [-1, 1, 1, 1],
        equilibrium_6a["v_RM2e"],
    )

    q_b2e_9a = gsim.orientation.euler_to_quaternion(equilibrium_9a["Theta_b2e"])
    state_9a = np.empty(1, dtype=gsim.simulator.Dynamics9a.state_dtype)
    state_9a["q_b2e"] = q_b2e_9a
    q_p2b_9a = gsim.orientation.euler_to_quaternion(equilibrium_9a["Theta_p2b"])
    state_9a["q_p2e"] = gsim.orientation.quaternion_product(q_b2e_9a, q_p2b_9a)
    state_9a["omega_b2e"] = [0, 0, 0]
    state_9a["omega_p2e"] = [0, 0, 0]
    state_9a["r_RM2O"] = [0, 0, 0]
    state_9a["v_RM2e"] = gsim.orientation.quaternion_rotate(
        q_b2e_9a * [-1, 1, 1, 1],
        equilibrium_9a["v_RM2e"],
    )

    # Optional: arbitrary modifications:
    # state_9a["q_b2e"] = np.array([1, 0, 0, 0])
    # state_9a["q_p2b"] = np.array([1, 0, 0, 0])

    # -----------------------------------------------------------------------
    # Load a test scenario

    inputs, T = zero_controls()
    # inputs, T = symmetric_brakes_fast_on()
    # inputs, T = symmetric_brakes_fast_off()
    # inputs, T = symmetric_brakes_fast_on_off()
    # inputs, T = short_right_turn_with_weightshift()
    # inputs, T = short_right_turn_without_weightshift()
    # inputs, T = continuous_right_turn_with_weightshift()
    # inputs, T = continuous_right_turn_without_weightshift()
    # inputs, T = thermal_zero_controls()
    # inputs, T = centered_thermal_with_accelerator()
    # inputs, T = centered_thermal_with_symmetric_brake()
    # inputs, T = roll_right_then_left()
    # inputs, T = roll_yaw_coupling_with_accelerator()
    # inputs, T = figure_8s()
    # inputs, T = horizontal_shear_zero_controls()
    # inputs, T = lateral_gust_zero_controls()

    # -----------------------------------------------------------------------
    # Build the dynamics models

    common_args = {"rho_air": 1.225, **inputs}

    model_6a = gsim.simulator.Dynamics6a(glider_6a, **common_args)
    model_6b = gsim.simulator.Dynamics6a(glider_6b, **common_args)
    model_6c = gsim.simulator.Dynamics6a(glider_6c, **common_args)
    model_9a = gsim.simulator.Dynamics9a(glider_9a, **common_args)
    model_9b = gsim.simulator.Dynamics9a(glider_9b, **common_args)
    model_9c = gsim.simulator.Dynamics9a(glider_9c, **common_args)

    # Choose which model to run
    state0, model = state_6a, model_6a
    # state0, model = state_6a, model_6b  # Same state as model_6a
    # state0, model = state_6a, model_6c  # Same state as model_6a
    # state0, model = state_9a, model_9a
    # state0, model = state_9a, model_9b  # Same state as model_9a
    # state0, model = state_9a, model_9c  # Same state as model_9a

    # -----------------------------------------------------------------------
    # Run the simulation

    t_start = time.perf_counter()
    dt = 0.10  # Time step for the `path` trajectory
    times, path = gsim.simulator.simulate(model, state0, dt=dt, T=T)

    # -----------------------------------------------------------------------
    # Extra values for verification/debugging

    K = len(times)
    if np.isscalar(inputs["delta_a"]):
        r_LE2RM = -model.glider.wing.r_RM2LE(inputs["delta_a"])
    else:
        r_LE2RM = -model.glider.wing.r_RM2LE(inputs["delta_a"](times))
    q_e2b = path["q_b2e"] * [1, -1, -1, -1]  # Applies C_ned/frd
    r_LE2O = path["r_RM2O"] + gsim.orientation.quaternion_rotate(q_e2b, r_LE2RM)
    v_LE2O = path["v_RM2e"] + gsim.orientation.quaternion_rotate(
        q_e2b, np.cross(path["omega_b2e"], r_LE2RM)
    )
    v_frd = gsim.orientation.quaternion_rotate(path["q_b2e"], path["v_RM2e"])

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

        # FIXME: assumes the payload has only one control point (r_P2RM^p)
        r_P2O = path["r_RM2O"] + gsim.orientation.quaternion_rotate(
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
        r_P2O = path["r_RM2O"] + gsim.orientation.quaternion_rotate(
            q_e2b,
            model.glider.payload.control_points(),
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

    derivatives = gsim.simulator.recompute_derivatives(model, times, path)

    t_stop = time.perf_counter()
    print(f"\nTotal time: {t_stop - t_start:.2f}\n")

    print("Final state:", path[-1])

    # -----------------------------------------------------------------------
    # Plots

    # 3D Plot: Position over time
    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca(projection="3d")
    ax.invert_yaxis()
    ax.invert_zaxis()
    lpp = 0.25  # Line-plotting period [sec]
    for t in range(0, K, int(lpp / dt)):  # Draw connecting lines every `lpp` seconds
        p1, p2 = path["r_RM2O"][t], r_LE2O[t]  # Risers -> wing central LE
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=0.5, c="k")

        p1, p2 = path["r_RM2O"][t], r_P2O[t]  # Risers -> payload
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=0.5, c="k")
    ax.plot(path["r_RM2O"].T[0], path["r_RM2O"].T[1], path["r_RM2O"].T[2], label="risers")
    ax.plot(r_LE2O.T[0], r_LE2O.T[1], r_LE2O.T[2], label="LE0")
    ax.plot(r_P2O.T[0], r_P2O.T[1], r_P2O.T[2], label="payload", lw=0.5, c="r")
    ax.legend()
    gsim.plots._set_axes_equal(ax)

    fig, ax = plt.subplots(3, figsize=(10, 10))
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

    breakpoint()


if __name__ == "__main__":
    main()
