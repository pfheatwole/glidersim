"""Scratch file for developing simulator-related code."""

import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import pfh.glidersim as gsim
from pfh.glidersim.extras import simulation


# ---------------------------------------------------------------------------
# Build a set "scenario" inputs


def zero_controls(T=5):
    """Scenario: zero_inputs."""
    return {}, T


def symmetric_brakes_fast_on(delta_b):
    """Scenario: zero_inputs."""
    t_warmup = 2
    t_rise = 0.5
    t_settle = 5
    braking = simulation.linear_control([(t_warmup, 0), (t_rise, delta_b)])
    inputs = {
        "delta_bl": braking,
        "delta_br": braking,
    }
    T = t_warmup + t_rise + t_settle
    return inputs, T


def symmetric_brakes_fast_off(delta_b):
    """Scenario: zero_inputs."""
    t_warmup = 2
    t_fall = 0.5
    t_settle = 5
    braking = simulation.linear_control([(0, delta_b), (t_warmup, None), (t_fall, 0)])
    inputs = {
        "delta_bl": braking,
        "delta_br": braking,
    }
    T = t_warmup + t_fall + t_settle
    return inputs, T


def symmetric_brakes_fast_on_off(delta_b):
    """Scenario: zero_inputs."""
    t_warmup = 2
    t_rise = 0.5
    t_hold = 2
    t_fall = 0.5
    t_settle = 5
    braking = simulation.linear_control(
        [(t_warmup, 0), (t_rise, delta_b), (t_hold, None), (t_fall, 0)],
    )
    inputs = {
        "delta_bl": braking,
        "delta_br": braking,
    }
    T = t_warmup + t_rise + t_hold + t_fall + t_settle
    return inputs, T


def short_right_turn(delta_br, delta_w, t_hold=5):
    """Scenario: short right turn."""
    t_warmup = 2
    t_rise_b = 2
    t_rise_w = 1
    t_hold_w = 5  # Hold time for initial weight shift
    t_fall = 1
    t_settle = 5
    inputs = {
        "delta_br": simulation.linear_control(
            [
                (t_warmup + t_rise_w + t_hold_w, 0),
                (t_rise_b, delta_br),
                (t_hold, None),
                (t_fall, 0),
            ]
        ),
        "delta_w": simulation.linear_control(
            [
                (t_warmup, 0),
                (t_rise_w, delta_w),
                (t_hold_w + t_rise_b + t_hold, None),
                (t_fall, 0),
            ]
        ),
    }
    T = t_warmup + t_rise_w + t_hold_w + +t_rise_b + t_hold + t_fall + t_settle
    return inputs, T


def continuous_right_turn(delta_w, delta_br):
    t_warmup = 2
    t_rise = 2
    t_hold = 5
    inputs = {
        "delta_br": simulation.linear_control(
            [(t_warmup + t_rise + t_hold, 0), (t_rise, delta_br)],
        ),
        "delta_w": simulation.linear_control([(t_warmup, 0), (t_rise, delta_w)]),
    }
    T = 60
    return inputs, T


def roll_right_then_left():
    """Scenario: smooth roll right then roll left."""
    inputs = {
        "delta_br": simulation.linear_control([(2, 0), (2, 0.75), (10, None), (2, 0)]),
        "delta_bl": simulation.linear_control([(16, 0), (3, 0.75)]),
    }
    T = 30
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
        "delta_a": simulation.linear_control([(t_start, 0), (t_rise, 0.75)]),
        "delta_br": simulation.linear_control(
            [
                (t_start + t_warmup, 0),
                (t_rise, 0.75),
                (t_hold, None),
                (t_fall, 0),
            ]
        ),
    }
    T = t_start + t_warmup + t_hold + t_fall + 5
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
        "delta_br": simulation.linear_control([(2, 0), *([*on, *off] * N_cycles)]),
        "delta_bl": simulation.linear_control([(2, 0), *([*off, *on] * N_cycles)]),
    }
    T = N_cycles * duration * 2
    return inputs, T


def ramping_headwind(t_rise=10, mag=-20):
    _mag = simulation.linear_control([(2, 0), (t_rise, mag)])

    def _headwind(t, r):
        v_W2e = np.zeros(
            (*gsim.util._broadcast_shapes(np.shape(t), np.shape(r)[:-1]), 3),
        )
        v_W2e[..., 0] = _mag(t)
        return v_W2e

    inputs = {
        "v_W2e": _headwind,
    }
    T = 20
    return inputs, T


def centered_thermal(delta_a=0, delta_b=0, py=0, mag=-3, radius5=10):
    """
    Place a thermal in the path of a glider.

    The default is that the wing will hit the thermal dead-center, but `py`
    can be used to shift the thermal and create an indirect hit.

    Parameters
    ----------
    delta_a : float [percentage]
        The amount of accelerator.
    delta_b : float [percentage]
        The amount of symmetric brake.
    py : float [m]
        The y-axis (easterly) offset of the thermal
    mag : float [m/s]
        The strength of the thermal core.
    radius5 : float [m]
        The distance at which the thermal strength has reduced to 5%.
    """
    inputs = {
        "delta_a": delta_a,
        "delta_bl": delta_b,
        "delta_br": delta_b,
        "v_W2e": simulation.CircularThermal(
            px=10 * 10,  # At 10m/s, roughly 10 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_enable=0,
        ),
    }
    T = 20
    return inputs, T


def horizontal_shear(delta_a=0, delta_b=0):
    inputs = {
        "delta_a": delta_a,
        "delta_bl": delta_b,
        "delta_br": delta_b,
        "v_W2e": simulation.HorizontalShear(
            x_start=10 * 10,
            mag=-4,
            smooth=25,
            t_enable=0,
        ),
    }
    T = 20
    return inputs, T


def lateral_gust(delta_a=0, delta_b=0):
    mag = 10  # [mph]
    inputs = {
        "delta_a": delta_a,
        "delta_bl": delta_b,
        "delta_br": delta_b,
        "v_W2e": simulation.LateralGust(
            t_start=2,
            t_ramp=3,
            t_duration=3,
            mag=mag * 1.6 / 3.6,  # [m/s]
        ),
    }
    T = 15
    return inputs, T


# ---------------------------------------------------------------------------


def build_paragliders(use_apparent_mass=True):
    """Build a set of glider models from a common base configuration."""
    wing = gsim.extras.wings.niviuk_hook3(size=23, verbose=False)
    harness = gsim.paraglider_harness.Spherical(
        mass=75,
        z_riser=0.5,
        S=0.55,
        CD=0.8,
        kappa_w=0.15,
    )

    # 6 DoF models
    paraglider_6a = gsim.paraglider.ParagliderSystemDynamics6a(
        wing,
        harness,
        use_apparent_mass=use_apparent_mass,
    )
    # 6b and 6c do not account for apparent mass
    paraglider_6b = gsim.paraglider.ParagliderSystemDynamics6b(wing, harness)
    paraglider_6c = gsim.paraglider.ParagliderSystemDynamics6c(wing, harness)

    # Coefficients for the spring-damper connection (9DoF models)
    # FIXME: naming?
    kappa_RM = [-100, 0, -10]  # Coefficients for Theta_p2b
    kappa_RM_dot = [-50, -5, -50]  # Coefficients for dot{Theta_p2b}

    # 9 DoF models
    paraglider_9a = gsim.paraglider.ParagliderSystemDynamics9a(
        wing,
        harness,
        kappa_RM=kappa_RM,
        kappa_RM_dot=kappa_RM_dot,
        use_apparent_mass=use_apparent_mass,
    )
    paraglider_9b = gsim.paraglider.ParagliderSystemDynamics9b(
        wing,
        harness,
        kappa_RM=kappa_RM,
        kappa_RM_dot=kappa_RM_dot,
        # No apparent mass
    )
    paraglider_9c = gsim.paraglider.ParagliderSystemDynamics9c(
        wing,
        harness,
        kappa_RM=kappa_RM,
        kappa_RM_dot=kappa_RM_dot,
        use_apparent_mass=use_apparent_mass,
    )

    return {
        "6a": paraglider_6a,
        "6b": paraglider_6b,
        "6c": paraglider_6c,
        "9a": paraglider_9a,
        "9b": paraglider_9b,
        "9c": paraglider_9c,
    }


def main():

    use_apparent_mass = True
    # use_apparent_mass = False
    paragliders = build_paragliders(use_apparent_mass=use_apparent_mass)

    # -----------------------------------------------------------------------
    # Load a test scenario

    inputs, T = zero_controls()
    # inputs, T = symmetric_brakes_fast_on(delta_b=1)
    # inputs, T = symmetric_brakes_fast_off(delta_b=1)
    # inputs, T = symmetric_brakes_fast_on_off(delta_b=1)
    # inputs, T = short_right_turn(delta_br=0.85, delta_w=0, t_hold=5)
    # inputs, T = short_right_turn(delta_br=0.85, delta_w=1, t_hold=5)
    # inputs, T = short_right_turn(delta_br=0.85, delta_w=0, t_hold=20)
    # inputs, T = short_right_turn(delta_br=0.85, delta_w=1, t_hold=20)
    # inputs, T = continuous_right_turn(delta_br=0.85, delta_w=0)
    # inputs, T = continuous_right_turn(delta_br=0.85, delta_w=1)
    # inputs, T = roll_right_then_left()
    # inputs, T = roll_yaw_coupling_with_accelerator()
    # inputs, T = figure_8s()
    # inputs, T = ramping_headwind()
    # inputs, T = centered_thermal()
    # inputs, T = centered_thermal(delta_a=1)
    # inputs, T = centered_thermal(delta_b=1)
    # inputs, T = centered_thermal(delta_b=0.85, py=10, radius5=20)
    # inputs, T = horizontal_shear()
    # inputs, T = horizontal_shear(delta_a=1)
    # inputs, T = horizontal_shear(delta_b=1)
    # inputs, T = lateral_gust()
    # inputs, T = lateral_gust(delta_a=1)
    # inputs, T = lateral_gust(delta_b=1)

    # -----------------------------------------------------------------------
    # Build a dynamics model and simulate the scenario

    sim_parameters = {  # Default scenario
        "delta_a": 0.0,
        "delta_bl": 0.0,
        "delta_br": 0.0,
        "delta_w": 0.0,
        "rho_air": 1.225,
        "v_W2e": (0, 0, 0),
    }
    sim_parameters.update(inputs)

    paraglider = paragliders["6a"]
    # paraglider = paragliders["6b"]  # Same state dynamics as 6a
    # paraglider = paragliders["6c"]  # Same state dynamics as 6a
    # paraglider = paragliders["9a"]
    # paraglider = paragliders["9b"]  # Same state dynamics as 9a
    # paraglider = paragliders["9c"]  # Same state dynamics as 9a
    if isinstance(paraglider, gsim.paraglider.ParagliderSystemDynamics6a):
        model = gsim.simulator.ParagliderStateDynamics6a(
            paraglider,
            **sim_parameters,
        )
    elif isinstance(paraglider, gsim.paraglider.ParagliderSystemDynamics9a):
        model = gsim.simulator.ParagliderStateDynamics9a(
            paraglider,
            **sim_parameters,
        )
    else:
        raise RuntimeError(f"Unsupported paraglider model {type(paraglider)}")

    print("\nPreparing the simulation...\n")
    modelname = str(type(paraglider)).split(".")[-1].split("'")[0]
    print(f"Model: {modelname}(use_apparent_mass={use_apparent_mass})")
    state0 = model.starting_equilibrium()
    gsim.simulator.prettyprint_state(state0, "Initial state:", "")
    t_start = time.perf_counter()
    dt = 0.25  # Time step for the sequence of `states`
    times, states = gsim.simulator.simulate(model, state0, dt=dt, T=T)
    states_dot = gsim.simulator.recompute_derivatives(model, times, states)
    t_stop = time.perf_counter()
    print(f"\nTotal time: {t_stop - t_start:.2f}\n")
    gsim.simulator.prettyprint_state(states[-1], "Final state:", "")

    # -----------------------------------------------------------------------
    # Extra values for verification/debugging

    Theta_b2e = gsim.orientation.quaternion_to_euler(states["q_b2e"])
    Theta_b2e_dot = gsim.extras.simulation.compute_euler_derivatives(
        Theta_b2e,
        states["omega_b2e"],
    )

    # -----------------------------------------------------------------------
    # Plots

    # 3D Plot: Position over time
    points = gsim.extras.simulation.sample_paraglider_positions(model, states, times)
    gsim.extras.plots.plot_3d_simulation_path(**points, show=False)

    # Plot: orientation (note: `omega_b2e` != `Theta_b2e_dot`)
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].plot(times, np.rad2deg(Theta_b2e))
    ax[1].plot(times, np.rad2deg(states["omega_b2e"]))
    ax[2].plot(times, np.rad2deg(states_dot["omega_b2e"]))
    ax[0].set_ylabel("Theta_b2e [deg]")
    ax[1].set_ylabel("omega_b2e [deg]")
    ax[2].set_ylabel("alpha_b2e [deg]")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    plt.show()

    # breakpoint()


if __name__ == "__main__":
    main()
