import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

import pfh.glidersim as gsim
from pfh.glidersim.extras import simulation


# ---------------------------------------------------------------------------
# Build a set "scenario" inputs


def zero_controls(T=5):
    """Scenario: zero_inputs."""
    return {}, T


def symmetric_brakes_fast_on():
    """Scenario: zero_inputs."""
    t_warmup = 2
    t_rise = 0.5
    braking = simulation.linear_control([(t_warmup, 0), (t_rise, 1)])
    inputs = {
        "delta_bl": braking,
        "delta_br": braking,
    }
    T = 10
    return inputs, T


def symmetric_brakes_fast_off():
    """Scenario: zero_inputs."""
    t_warmup = 2
    t_fall = 0.5
    braking = simulation.linear_control([(0, 1), (t_warmup, None), (t_fall, 0)])
    inputs = {
        "delta_bl": braking,
        "delta_br": braking,
    }
    T = t_warmup + t_fall + 10
    return inputs, T


def symmetric_brakes_fast_on_off():
    """Scenario: zero_inputs."""
    t_warmup = 2
    t_rise = 1
    t_hold = 2
    t_fall = 0.5
    braking = simulation.linear_control([
        (t_warmup, 0),
        (t_rise, 1),
        (t_hold, None),
        (t_fall, 0)
    ])
    inputs = {
        "delta_bl": braking,
        "delta_br": braking,
    }
    T = t_warmup + t_rise + t_hold + t_fall + 10
    return inputs, T


def short_right_turn_with_weightshift():
    """Scenario: short right turn."""
    t_warmup = 2
    t_rise_b = 1
    t_rise_w = 1
    t_hold = 5
    t_fall = 1
    t_settle = 5
    mag = 0.75
    inputs = {
        "delta_br": simulation.linear_control([
            (t_warmup + t_rise_w + t_hold, 0),
            (t_rise_b, mag),
            (t_hold, None),
            (t_fall, 0),
        ]),
        "delta_w": simulation.linear_control([
            (t_warmup, 0),
            (t_rise_w, 0.75),
            (t_hold + t_rise_b + t_hold, None),
            (t_fall, 0),
        ]),
    }
    T = t_warmup + t_rise_w + t_hold + + t_rise_b + t_hold + t_fall + t_settle
    return inputs, T


def short_right_turn_without_weightshift():
    """Scenario: short right turn."""
    t_warmup = 2
    t_rise = 4
    t_hold = 5
    t_fall = 1
    t_settle = 5
    mag = 0.75
    inputs = {
        "delta_br": simulation.linear_control([
            (t_warmup, 0),
            (t_rise, mag),
            (t_hold, None),
            (t_fall, 0),
        ]),
    }
    T = t_warmup + t_rise + t_hold + t_fall + t_settle
    return inputs, T


def continuous_right_turn_with_weightshift(mag=0.75):
    """Scenario: continuous right turn with weightshift."""
    t_warmup = 2
    t_rise = 1
    t_hold = 5
    inputs = {
        "delta_br": simulation.linear_control([(t_warmup + t_rise + t_hold, 0), (t_rise, mag)]),
        "delta_w": simulation.linear_control([(t_warmup, 0), (t_rise, 1.00)]),
    }
    T = 60
    return inputs, T


def continuous_right_turn_without_weightshift(mag=0.75):
    """Scenario: continuous right turn without weightshift."""
    t_warmup = 5
    inputs = {
        "delta_br": simulation.linear_control([(t_warmup, 0), (3, mag)]),
    }
    T = 60
    return inputs, T


def centered_thermal_zero_controls(py=0, mag=-3, radius5=10):
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
        "v_W2e": simulation.CircularThermal(
            px=10 * 10,  # At 10m/s, roughly 10 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_enable=0,
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
    inputs = {
        "delta_a": 0.75,
        "v_W2e": simulation.CircularThermal(
            px=10 * 15,  # At 10m/s, roughly 15 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_enable=0,
        )
    }
    T = 20
    return inputs, T


def centered_thermal_with_symmetric_brake(py=0, mag=-3, radius5=15):
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
    inputs = {
        "delta_bl": 1,
        "delta_br": 1,
        "v_W2e": simulation.CircularThermal(
            px=10 * 10,  # At 10m/s, roughly 10 seconds in
            py=py,
            mag=mag,
            radius5=radius5,
            t_enable=0,
        )
    }
    T = 20
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
        "delta_br": simulation.linear_control([
            (t_start + t_warmup, 0),
            (t_rise, 0.75),
            (t_hold, None),
            (t_fall, 0)
        ]),
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


def horizontal_shear_zero_controls():
    inputs = {
        "v_W2e": simulation.HorizontalShear(
            x_start=10 * 10,
            mag=-4,
            smooth=25,
            t_enable=0,
        ),
    }
    T = 20
    return inputs, T


def lateral_gust_zero_controls():
    mag = 10  # [mph]
    inputs = {
        "v_W2e": simulation.LateralGust(
            t_start=2,
            t_ramp=1,
            t_duration=3,
            mag=mag * 1.6 / 3.6,  # [m/s]
        ),
    }
    T = 20
    return inputs, T


def lateral_gust_with_accelerator():
    mag = 10  # [mph]
    inputs = {
        "delta_a": 1,
        "v_W2e": simulation.LateralGust(
            t_start=2,
            t_ramp=1,
            t_duration=3,
            mag=mag * 1.6 / 3.6,  # [m/s]
        ),
    }
    T = 20
    return inputs, T


def lateral_gust_with_symmetric_brakes():
    mag = 10  # [mph]
    inputs = {
        "delta_bl": 0.75,
        "delta_br": 0.75,
        "v_W2e": simulation.LateralGust(
            t_start=2,
            t_ramp=3,
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
    # Load a test scenario

    inputs, T = zero_controls()
    # inputs, T = symmetric_brakes_fast_on()
    # inputs, T = symmetric_brakes_fast_off()
    # inputs, T = symmetric_brakes_fast_on_off()
    # inputs, T = short_right_turn_with_weightshift()
    # inputs, T = short_right_turn_without_weightshift()
    # inputs, T = continuous_right_turn_with_weightshift()
    # inputs, T = continuous_right_turn_without_weightshift()
    # inputs, T = centered_thermal_zero_controls()
    # inputs, T = centered_thermal_with_accelerator()
    # inputs, T = centered_thermal_with_symmetric_brake()
    # inputs, T = roll_right_then_left()
    # inputs, T = roll_yaw_coupling_with_accelerator()
    # inputs, T = figure_8s()
    # inputs, T = horizontal_shear_zero_controls()
    # inputs, T = lateral_gust_zero_controls()
    # inputs, T = lateral_gust_with_accelerator()
    # inputs, T = lateral_gust_with_symmetric_brakes()

    # -----------------------------------------------------------------------
    # Build a dynamics model and simulate the scenario

    sim_parameters = {  # Default scenario
        "delta_a": 0.0,
        "delta_bl": 0.0,
        "delta_br": 0.0,
        "delta_w": 0.0,
        "v_W2e": None,
        "rho_air": 1.225
    }
    sim_parameters.update(inputs)
    model = gsim.simulator.Dynamics6a(glider_6a, **sim_parameters)
    # model = gsim.simulator.Dynamics6a(glider_6b, **sim_parameters)
    # model = gsim.simulator.Dynamics6a(glider_6c, **sim_parameters)
    # model = gsim.simulator.Dynamics9a(glider_9a, **sim_parameters)
    # model = gsim.simulator.Dynamics9a(glider_9b, **sim_parameters)
    # model = gsim.simulator.Dynamics9a(glider_9c, **sim_parameters)

    print("\nPreparing the simulation...\n")
    state0 = model.starting_equilibrium()
    Theta_b2e = gsim.orientation.quaternion_to_euler(state0["q_b2e"])[0]
    with np.printoptions(precision=4, suppress=True):
        print("Initial state:")
        print("  Theta_b2e:", np.rad2deg(Theta_b2e))
        if "q_p2b" in state0.dtype.names:
            Theta_p2b = gsim.orientation.quaternion_to_euler(state0["q_p2b"])[0]
            print("  Theta_p2b:", np.rad2deg(Theta_p2b))
        print("  omega_b2e:", state0["omega_b2e"][0])
        if "omega_p2e" in state0.dtype.names:
            print("  omega_p2e:", state0["omega_p2e"][0])
        print("     r_RM2O:", state0["r_RM2O"][0])
        print("     v_RM2e:", state0["v_RM2e"][0])
        print()

    t_start = time.perf_counter()
    dt = 0.10  # Time step for the `path` trajectory
    times, path = gsim.simulator.simulate(model, state0, dt=dt, T=T)
    path_dot = gsim.simulator.recompute_derivatives(model, times, path)
    t_stop = time.perf_counter()
    print(f"\nTotal time: {t_stop - t_start:.2f}\n")

    stateK = path[-1]
    with np.printoptions(precision=4, suppress=True):
        print("Final state:")
        print("  Theta_b2e:", np.rad2deg(Theta_b2e))
        if "q_p2b" in stateK.dtype.names:
            Theta_p2b = gsim.orientation.quaternion_to_euler(stateK["q_p2b"])
            print("  Theta_p2b:", np.rad2deg(Theta_p2b))
        print("  omega_b2e:", stateK["omega_b2e"])
        if "omega_p2e" in state0.dtype.names:
            print("  omega_p2e:", stateK["omega_p2e"])
        print("     r_RM2O:", stateK["r_RM2O"])
        print("     v_RM2e:", stateK["v_RM2e"])
        print()

    # -----------------------------------------------------------------------
    # Extra values for verification/debugging

    K = len(times)
    if np.isscalar(sim_parameters["delta_a"]):
        r_LE2RM = -model.glider.wing.r_RM2LE(sim_parameters["delta_a"])
    else:
        r_LE2RM = -model.glider.wing.r_RM2LE(sim_parameters["delta_a"](times))
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
    ax[2].plot(times, np.rad2deg(path_dot["omega_b2e"]))
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
