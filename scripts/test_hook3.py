import matplotlib.pyplot as plt
import numpy as np

import pfh.glidersim as gsim
from pfh.glidersim import orientation


if __name__ == "__main__":

    print("\n-------------------------------------------------------------\n")

    # Note to self: the wing should weight 4.7kG in total; according to these
    # specs, and the `rho_upper`/`rho_lower` embedded in ParagliderWing, the
    # wing materials I'm accounting for total to 1.83kg, so there's a lot left
    # in the lines, risers, ribs, etc.
    wing = gsim.extras.wings.build_hook3()
    harness = gsim.harness.Spherical(
        mass=75, z_riser=0.5, S=0.55, CD=0.8, kappa_w=0.15,
    )
    glider = gsim.paraglider.Paraglider6a(wing, harness)
    # glider = gsim.paraglider.Paraglider9a(wing, harness)

    # print("Plotting the wing performance curves")
    # plot_wing_coefficients(wing)

    print("\nFinished building the glider.\n")
    # breakpoint()
    # 1/0

    print("\nComputing the glider equilibrium state...")

    approximate = True
    # approximate = False

    if approximate:  # Option 1: fast, but only approximate
        eqstate = glider.equilibrium_state2
    else:  # Option 2: currently very slow, but more accurate
        eqstate = glider.equilibrium_state

    eq = eqstate(
        delta_a=0.0,
        delta_b=0.0,
        alpha_0=np.deg2rad(9),
        theta_0=np.deg2rad(3),
        v_0=10,
        rho_air=1.2,
    )

    # Compute the residual acceleration at the given equilibrium state
    q_b2e = orientation.euler_to_quaternion(eq["Theta_b2e"])
    q_e2b = q_b2e * [-1, 1, 1, 1]
    v_RM2e = orientation.quaternion_rotate(q_e2b, eq["v_RM2e"])

    # For the `Paraglider6a` model
    a_RM2e, alpha_b2e, _ = glider.accelerations(
        v_RM2e=eq["v_RM2e"],
        omega_b2e=[0, 0, 0],
        g=orientation.quaternion_rotate(q_b2e, [0, 0, 9.8]),
        rho_air=1.2,
        reference_solution=eq["reference_solution"],
    )
    # For the `Paraglider9a` model
    # a_RM2e, alpha_b2e, alpha_p2b, _ = glider.accelerations(
    #     v_RM2e=eq["v_RM2e"],
    #     omega_b2e=[0, 0, 0],
    #     omega_p2e=[0, 0, 0],
    #     Theta_p2b=eq["Theta_p2b"],
    #     g=orientation.quaternion_rotate(q_b2e, [0, 0, 9.8]),
    #     rho_air=1.2,
    #     reference_solution=eq["reference_solution"],
    # )

    print("Equilibrium state:")
    print(f"  alpha_b:     {np.rad2deg(eq['alpha_b']):>6.3f} [deg]")
    print(f"  theta_b:     {np.rad2deg(eq['Theta_b2e'][1]):>6.3f} [deg]")
    print(f"  Glide angle: {np.rad2deg(eq['gamma_b']):>6.3f} [deg]")
    print(f"  Glide ratio: {eq['glide_ratio']:>6.3f}")
    print(f"  Glide speed: {np.linalg.norm(v_RM2e):>6.3f}")
    print()
    print("For verification of the equilibrium state:")
    print(f"  v_RM2e:      {v_RM2e.round(4)}")
    print(f"  a_RM2e:      {a_RM2e.round(4)}")
    print(f"  alpha_b2e:   {np.rad2deg(alpha_b2e).round(4)}")

    print("\n<pausing before polar curves>\n")
    breakpoint()

    input("Plot the polar curve?  Press any key")
    gsim.extras.compute_polars.plot_polar_curve(glider, approximate=approximate)
