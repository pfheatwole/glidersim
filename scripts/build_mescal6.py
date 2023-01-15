import matplotlib.pyplot as plt
import numpy as np

import pfh.glidersim as gsim
from pfh.glidersim import orientation

if __name__ == "__main__":
    print("\n-------------------------------------------------------------\n")

    use_6a = True
    use_6a = False  # Set to False to use Paraglider9a

    size = 'XS'

    wing = gsim.extras.wings.skywalk_mescal_6(size=size, verbose=True)

    if size == 'XS':
        harness = gsim.paraglider_harness.Spherical(
            mass=75, z_riser=0.5, S=0.55, CD=0.8, kappa_w=0.15
        )
    else:
        raise RuntimeError(f"Invalid Mescal 6 canopy size {size}")

    if use_6a:
        paraglider = gsim.paraglider.ParagliderSystemDynamics6a(wing, harness)
    else:
        paraglider = gsim.paraglider.ParagliderSystemDynamics9a(wing, harness)

    print("Computing the glider equilibrium state...\n")
    eq = paraglider.equilibrium_state()

    # Compute the residual acceleration at the given equilibrium state
    q_b2e = orientation.euler_to_quaternion(eq["Theta_b2e"])
    q_e2b = q_b2e * [1, -1, -1, -1]
    v_RM2e = orientation.quaternion_rotate(q_e2b, eq["v_RM2e"])

    if use_6a:
        a_RM2e, alpha_b2e, _ = paraglider.accelerations(
            v_RM2e=eq["v_RM2e"],
            omega_b2e=[0, 0, 0],
            g=orientation.quaternion_rotate(q_b2e, [0, 0, 9.8]),
            reference_solution=eq["reference_solution"],
        )
    else:
        a_RM2e, alpha_b2e, alpha_p2b, _ = paraglider.accelerations(
            v_RM2e=eq["v_RM2e"],
            omega_b2e=[0, 0, 0],
            omega_p2e=[0, 0, 0],
            Theta_p2b=eq["Theta_p2b"],
            g=orientation.quaternion_rotate(q_b2e, [0, 0, 9.8]),
            reference_solution=eq["reference_solution"],
        )

    print("Equilibrium state:")
    print(f"  alpha_b:     {np.rad2deg(eq['alpha_b']):>6.3f} [deg]")
    print(f"  theta_b2e:   {np.rad2deg(eq['Theta_b2e'][1]):>6.3f} [deg]")
    if use_6a is False:
        theta_p2e = eq["Theta_p2b"][1] + eq["Theta_b2e"][1]
        print(f"  theta_p2e:   {np.rad2deg(theta_p2e):>6.3f} [deg]")
    print(f"  Glide angle: {np.rad2deg(eq['gamma_b']):>6.3f} [deg]")
    print(f"  Glide ratio: {eq['glide_ratio']:>6.3f}")
    print(f"  Glide speed: {np.linalg.norm(v_RM2e):>6.3f}")
    print()
    print("Verify accelerations at equilibrium:")
    print(f"  a_RM2e:      {a_RM2e.round(4)}")
    print(f"  alpha_b2e:   {np.rad2deg(alpha_b2e).round(4)}")
    if use_6a is False:
        print(f"  alpha_p2b:   {np.rad2deg(alpha_p2b).round(4)}")

    print("\n<pausing before polar curves>\n")
    # breakpoint()

    input("Plot the polar curve?  Press any key")
    accelerating, braking = gsim.extras.compute_polars.compute_polar_data(
        paraglider,
    )

    # Just
    # gsim.extras.compute_polars.plot_wing_coefficients(paraglider)

    # -----------------------------------------------------------------------
    # Plot the curves
    fig = plt.figure()
    fig.suptitle('Mescal 6 {} Characteristics'.format(size))

    deltas_a = accelerating["delta"]
    deltas_b = braking["delta"]
    thetas_a = accelerating["theta_b"]
    thetas_b = braking["theta_b"]
    v_RM2e_a = accelerating["v_RM2e"]
    v_RM2e_b = braking["v_RM2e"]

    # alpha_b versus control input
    alpha_b_plot = fig.add_subplot(2, 2, 1) # Upper Left
    alpha_b_plot.plot(deltas_a, np.rad2deg(accelerating["alpha_b"]), "g")
    alpha_b_plot.plot(-deltas_b, np.rad2deg(braking["alpha_b"]), "r")
    alpha_b_plot.set_xlabel("Control input [%]")
    alpha_b_plot.set_ylabel("alpha_b [deg]")

    # theta_b versus control input
    theta_b_plot = fig.add_subplot(2, 2, 3) # Lower Left
    theta_b_plot.plot(deltas_a, np.rad2deg(thetas_a), "g")
    theta_b_plot.plot(-deltas_b, np.rad2deg(thetas_b), "r")
    theta_b_plot.set_xlabel("Control input [%]")
    theta_b_plot.set_ylabel("theta_b [deg]")

    # Vertical versus horizontal airspeed
    polar_plot = fig.add_subplot(2, 2, 2) # Upper Right
    polar_plot.plot(v_RM2e_a.T[0], v_RM2e_a.T[2], "g")
    polar_plot.plot(v_RM2e_b.T[0], v_RM2e_b.T[2], "r")
    # polar_plot.set_aspect("equal")
    # polar_plot.set_xlim(0, 25)
    # polar_plot.set_ylim(0, 8)
    polar_plot.invert_yaxis()
    polar_plot.set_xlabel("Horizontal airspeed [m/s]")
    polar_plot.set_ylabel("sink rate [m/s]")
    polar_plot.grid(which="both")
    polar_plot.minorticks_on()

    # Glide ratio
    glide_ratio_plot = fig.add_subplot(2, 2, 4, sharex=polar_plot) # Lower Right. Share axis with polar plot (V forward)
    glide_ratio_plot.plot(v_RM2e_a.T[0], accelerating["glide_ratio"], "g")
    glide_ratio_plot.plot(v_RM2e_b.T[0], braking["glide_ratio"], "r")
    # glide_ratio_plot.set_xlim(0, 25)
    glide_ratio_plot.set_xlabel("Horizontal airspeed [m/s]")
    glide_ratio_plot.set_ylabel("Glide ratio")
    glide_ratio_plot.grid()

    fig.tight_layout() # Make sure graphs don't overlap
    plt.show()

    # breakpoint()