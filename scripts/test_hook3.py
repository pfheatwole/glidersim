import matplotlib.pyplot as plt
import numpy as np

import pfh.glidersim as gsim
from pfh.glidersim import orientation


if __name__ == "__main__":

    print("\n-------------------------------------------------------------\n")

    use_6a = True
    # use_6a = False  # Disable to use Paraglider9a

    wing = gsim.extras.wings.build_hook3(verbose=True)
    harness = gsim.harness.Spherical(
        mass=75, z_riser=0.5, S=0.55, CD=0.8, kappa_w=0.15,
    )
    if use_6a:
        glider = gsim.paraglider.Paraglider6a(wing, harness)
    else:
        glider = gsim.paraglider.Paraglider9a(wing, harness)

    print("Computing the glider equilibrium state...\n")
    eq = glider.equilibrium_state()

    # Compute the residual acceleration at the given equilibrium state
    q_b2e = orientation.euler_to_quaternion(eq["Theta_b2e"])
    q_e2b = q_b2e * [-1, 1, 1, 1]
    v_RM2e = orientation.quaternion_rotate(q_e2b, eq["v_RM2e"])

    if use_6a:
        a_RM2e, alpha_b2e, _ = glider.accelerations(
            v_RM2e=eq["v_RM2e"],
            omega_b2e=[0, 0, 0],
            g=orientation.quaternion_rotate(q_b2e, [0, 0, 9.8]),
            reference_solution=eq["reference_solution"],
        )
    else:
        a_RM2e, alpha_b2e, alpha_p2b, _ = glider.accelerations(
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
    breakpoint()

    input("Plot the polar curve?  Press any key")
    gsim.extras.compute_polars.plot_polar_curve(glider)
