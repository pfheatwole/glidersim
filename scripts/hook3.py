from IPython import embed

import matplotlib.pyplot as plt

import numpy as np

import pfh.glidersim as gsim
from pfh.glidersim import quaternion


def plot_polar_curve(glider, N=21, approximate=True):
    """Compute the equilibrium conditions and plot the polar curves."""

    if approximate:  # Option 1: fast, but only approximate
        eqstate = glider.equilibrium_state2
    else:  # Option 2: currently very slow, but more accurate
        eqstate = glider.equilibrium_state

    eqs_a = []
    delta_as = np.linspace(0, 1, N)
    equilibrium = {  # Initial guesses
        "alpha_b": np.deg2rad(9),
        "Theta_b2e": [0, np.deg2rad(3), 0],
        "v_R2e": [10, 0, 1],
        "reference_solution": None,
    }
    print("Calculating equilibrium states over the range of accelerator")
    for n, da in enumerate(delta_as):
        print(f"\r{da:.2f}", end="")
        try:
            equilibrium = eqstate(
                delta_a=da,
                delta_b=0,
                alpha_0=equilibrium["alpha_b"],
                theta_0=equilibrium["Theta_b2e"][1],
                v_0=np.linalg.norm(equilibrium["v_R2e"]),
                rho_air=1.2,
                reference_solution=equilibrium["reference_solution"],
            )
        except gsim.foil.ForceEstimator.ConvergenceError:
            print("\nConvergence started failing. Aborting early.")
            delta_as = delta_as[:n]
            break
        eqs_a.append(equilibrium)
    print()

    eqs_b = []
    delta_bs = np.linspace(0, 1, N)
    equilibrium = {  # Initial guesses
        "alpha_b": np.deg2rad(9),
        "Theta_b2e": [0, np.deg2rad(3), 0],
        "v_R2e": [10, 0, 1],
        "reference_solution": None,
    }
    print("Calculating equilibrium states over the range of brake")
    for n, db in enumerate(delta_bs):
        print("\rdb: {:.2f}".format(db), end="")
        try:
            equilibrium = eqstate(
                delta_a=0,
                delta_b=db,
                alpha_0=equilibrium["alpha_b"],
                theta_0=equilibrium["Theta_b2e"][1],
                v_0=np.linalg.norm(equilibrium["v_R2e"]),
                rho_air=1.2,
                reference_solution=equilibrium["reference_solution"],
            )
        except gsim.foil.ForceEstimator.ConvergenceError:
            print("\nConvergence started failing. Aborting early.")
            delta_bs = delta_bs[:n]
            break
        eqs_b.append(equilibrium)
    print()

    # Convert the airspeeds from body coordinates to Earth coordinates
    Theta_b2e_a = np.asarray([e["Theta_b2e"] for e in eqs_a])
    Theta_b2e_b = np.asarray([e["Theta_b2e"] for e in eqs_b])
    q_e2b_a = quaternion.euler_to_quaternion(Theta_b2e_a.T).T * [-1, 1, 1, 1]
    q_e2b_b = quaternion.euler_to_quaternion(Theta_b2e_b.T).T * [-1, 1, 1, 1]
    v_R2e_a = [e["v_R2e"] for e in eqs_a]
    v_R2e_b = [e["v_R2e"] for e in eqs_b]
    v_R2e_a = quaternion.apply_quaternion_rotation(q_e2b_a, v_R2e_a)
    v_R2e_b = quaternion.apply_quaternion_rotation(q_e2b_b, v_R2e_b)

    # -----------------------------------------------------------------------
    # Plot the curves
    fig, ax = plt.subplots(2, 2)  # [[alpha_b, sink rate], [theta_b, GR]]

    # alpha_b versus control input
    alpha_b_a = [e["alpha_b"] for e in eqs_a]
    alpha_b_b = [e["alpha_b"] for e in eqs_b]
    ax[0, 0].plot(delta_as, np.rad2deg(alpha_b_a), "g")
    ax[0, 0].plot(-delta_bs, np.rad2deg(alpha_b_b), "r")
    ax[0, 0].set_xlabel("Control input [%]")
    ax[0, 0].set_ylabel("alpha_b [deg]")

    # Vertical versus horizontal airspeed
    ax[0, 1].plot(v_R2e_a.T[0], v_R2e_a.T[2], "g")
    ax[0, 1].plot(v_R2e_b.T[0], v_R2e_b.T[2], "r")
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xlim(0, 25)
    ax[0, 1].set_ylim(0, 8)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlabel("Horizontal airspeed [m/s]")
    ax[0, 1].set_ylabel("sink rate [m/s]")
    ax[0, 1].grid(which="both")
    ax[0, 1].minorticks_on()

    # theta_b versus control input
    theta_b_a = [e["Theta_b2e"][1] for e in eqs_a]
    theta_b_b = [e["Theta_b2e"][1] for e in eqs_b]
    ax[1, 0].plot(delta_as, np.rad2deg(theta_b_a), "g")
    ax[1, 0].plot(-delta_bs, np.rad2deg(theta_b_b), "r")
    ax[1, 0].set_xlabel("Control input [%]")
    ax[1, 0].set_ylabel("theta_b [deg]")

    # Glide ratio
    GR_a = [e["glide_ratio"] for e in eqs_a]
    GR_b = [e["glide_ratio"] for e in eqs_b]
    ax[1, 1].plot(v_R2e_a.T[0], GR_a, "g")
    ax[1, 1].plot(v_R2e_b.T[0], GR_b, "r")
    ax[1, 1].set_xlim(0, 25)
    ax[1, 1].set_xlabel("Horizontal airspeed [m/s]")
    ax[1, 1].set_ylabel("Glide ratio")

    plt.show()

    embed()


def plot_wing_coefficients(wing, delta_b=0, v_mag=10, rho_air=1.2):
    alphas = np.deg2rad(np.linspace(-10, 25, 50))
    Fs, Ms = [], []
    reference_solution = None
    for k, alpha in enumerate(alphas):
        print(f"\ralpha: {np.rad2deg(alpha):6.2f}", end="")
        try:
            dF, dM, reference_solution, = wing.forces_and_moments(
                delta_bl=delta_b,
                delta_br=delta_b,
                v_W2b=-v_mag * np.array([np.cos(alpha), 0, np.sin(alpha)]),
                rho_air=rho_air,
                reference_solution=reference_solution,
            )
            Fs.append(dF.sum(axis=0))
            Ms.append(dM.sum(axis=0))
        except gsim.foil.ForceEstimator.ConvergenceError:
            break
    alphas = alphas[:k]
    Fs = Fs[:k]
    Ms = Ms[:k]

    CLs, CDs, CMs = [], [], []
    for n, F in enumerate(Fs):
        L = F[0] * np.sin(alphas[n]) - F[2] * np.cos(alphas[n])
        D = -F[0] * np.cos(alphas[n]) - F[2] * np.sin(alphas[n])
        CL = L / (0.5 * rho_air * v_mag ** 2 * wing.canopy.S)
        CD = D / (0.5 * rho_air * v_mag ** 2 * wing.canopy.S)
        CM = Ms[n][1] / (
            0.5
            * rho_air
            * v_mag ** 2
            * wing.canopy.S
            * wing.canopy.chord_length(0)
        )
        CLs.append(CL)
        CDs.append(CD)
        CMs.append(CM)

    style = {"c": "k", "lw": 0.75, "ls": "-", "marker": "o", "markersize": "1.5"}
    fig, ax = plt.subplots(3, 2, figsize=(9, 8))
    ax[0, 0].plot(np.rad2deg(alphas), CLs, **style)
    ax[1, 0].plot(np.rad2deg(alphas), CDs, **style)
    ax[2, 0].plot(np.rad2deg(alphas), CMs, **style)
    plt.setp(ax[:, 0], xlabel="alpha [deg]")
    ax[0, 0].set_ylabel("Lift Coefficient")
    ax[1, 0].set_ylabel("Drag Coefficient")
    ax[2, 0].set_ylabel("Pitching Coefficient")

    ax[0, 1].plot(np.rad2deg(alphas), np.array(CLs) / np.array(CDs), **style)
    ax[0, 1].set_xlabel("alpha [deg]")
    ax[0, 1].set_ylabel("CL/CD")

    ax[1, 1].plot(CDs, CLs, **style)
    ax[1, 1].set_xlabel("CD")
    ax[1, 1].set_ylabel("CL")

    for ax in fig.axes:
        if len(ax.lines) > 0:
            ax.grid()

    fig.tight_layout()  # Prevent axes overlapping titles
    plt.show()
    embed()


def build_hook3():

    print("Building an (approximate) Niviuk Hook 3 23\n")

    # -----------------------------------------------------------------------
    # Airfoil

    print("Airfoil: NACA 24018, curving flap\n")
    airfoil_geo = gsim.airfoil.NACA(24018, convention="vertical")
    airfoil_coefs = gsim.airfoil.GridCoefficients("polars/braking_NACA24018_Xtr0.25/gridded.csv")
    delta_max = np.deg2rad(13.37)  # FIXME: magic number
    airfoil = gsim.airfoil.Airfoil(airfoil_coefs, airfoil_geo)

    # -----------------------------------------------------------------------
    # Parafoil: an approximate Niviuk Hook 3, size 23

    # True technical specs
    chord_tip, chord_root, chord_mean = 0.52, 2.58, 2.06
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat / AR_flat
    S, b, AR = 19.55, 8.84, 4.00

    chord_length = gsim.foil.elliptical_chord(
        root=chord_root / (b_flat / 2),
        tip=chord_tip / (b_flat / 2),
    )

    # The geometric torsion distribution is uncertain. In Sec. 11.4, pg 17 of
    # the manual ("Line Plan") it appears to have a roughly square-root
    # spanwise distribution with a maximum valueof 6 or so? Unfortunately that
    # distribution proves difficult for Phillips method, so here I provide an
    # "easier" alternative.
    # torsion = Parafoil.PolynomialTorsion(start=0.0, peak=6, exponent=0.75)
    torsion = gsim.foil.PolynomialTorsion(start=0.8, peak=4, exponent=2)

    chord_surface = gsim.foil.ChordSurface(
        r_x=0.75,
        x=0,
        r_yz=1.00,
        yz=gsim.foil.elliptical_arc(mean_anhedral=33, tip_anhedral=67),
        chord_length=chord_length,
        torsion=torsion,
    )

    # FIXME: add the viscous drag modifiers?
    sections = gsim.foil.FoilSections(
        airfoil=airfoil,
        intakes=gsim.foil.SimpleIntakes(0.85, -0.04, -0.09),  # FIXME: guess
    )

    canopy = gsim.foil.SimpleFoil(
        chords=chord_surface,
        sections=sections,
        # b=b,  # Option 1: Scale the using the projected span
        b_flat=b_flat,  # Option 2: Scale the using the flattened span
    )

    print("Canopy geometry:           [Target]")
    print(f"  flattened span: {canopy.b_flat:>6.3f}   [{b_flat:>6.3f}]")
    print(f"  flattened area: {canopy.S_flat:>6.3f}   [{S_flat:>6.3f}]")
    print(f"  flattened AR:   {canopy.AR_flat:>6.3f}   [{AR_flat:>6.3f}]")
    # print(f"  planform flat SMC   {canopy.SMC:>6.3f}")
    # print(f"  planform flat MAC:  {canopy.MAC:>6.3f}")

    print(f"  projected span: {canopy.b:>6.3f}   [{b:>6.3f}]")
    print(f"  projected area: {canopy.S:>6.3f}   [{S:>6.3f}]")
    print(f"  projected AR:   {canopy.AR:>6.3f}   [{AR:>6.3f}]")
    print()

    # print("Drawing the canopy")
    # gsim.plots.plot_foil(canopy, N_sections=131, flatten=False)
    # gsim.plots.plot_foil(canopy, N_sections=71, flatten=True)
    # gsim.plots.plot_foil_topdown(canopy, N_sections=51)
    # gsim.plots.plot_foil_topdown(canopy, N_sections=51, flatten=True)

    # print("\nPausing...")
    # embed()

    # Compare to the Hook 3 manual, sec 11.4 "Line Plan", page 17
    # gsim.plots.plot_foil_topdown(canopy, N_sections=77)

    # -----------------------------------------------------------------------
    # Line geometry
    #
    # The brake parameters are not based on the actual wing in any way.
    # The line drag positions are crude guess.

    line_parameters = {
        "kappa_x": 0.49,  # FIXME: Source? Trying to match `theta_eq` at trim?
        "kappa_z": 6.8 / chord_root,  # ref: "Hook 3 technical specs", pg 2
        "kappa_A": 0.11,  # Approximated from the line plan in the users manual, pg 17
        "kappa_C": 0.59,
        "kappa_a": 0.15 / chord_root,  # ref: "Hook 3 technical specs", pg 2
        "total_line_length": 213 / chord_root,  # ref: "Hook 3 technical specs", pg 2
        "average_line_diameter": 1e-3,  # Blind guess
        "line_drag_positions": np.array([[-0.5 * chord_root, -1.75, -5],
                                         [-0.5 * chord_root,  1.75, -5]]) / chord_root,
        "Cd_lines": 0.98,  # ref: Kulhánek, 2019; page 5
    }

    s_delta_start = 0.1
    s_delta_max = gsim.line_geometry.SimpleLineGeometry.minimum_s_delta_max(s_delta_start) + 1e-9
    brake_parameters = {
        "s_delta_start": s_delta_start,
        "s_delta_max": s_delta_max,
        "delta_max": delta_max,
    }

    lines = gsim.line_geometry.SimpleLineGeometry(
        **line_parameters,
        **brake_parameters,
    )

    # -----------------------------------------------------------------------
    # Wing and glider

    wing = gsim.paraglider_wing.ParagliderWing(
        lines=lines,
        canopy=canopy,
        rho_upper=39 / 1000,  # [kg/m^2]  Porcher 9017 E77A
        rho_lower=35 / 1000,  # [kg/m^2]  Dominico N20DMF
        force_estimator=gsim.foil.Phillips(canopy, v_ref_mag=10, K=31),
    )

    return wing


if __name__ == "__main__":

    print("\n-------------------------------------------------------------\n")

    # Note to self: the wing should weight 4.7kG in total; according to these
    # specs, and the `rho_upper`/`rho_lower` embedded in ParagliderWing, the
    # wing materials I'm accounting for total to 1.83kg, so there's a lot left
    # in the lines, risers, ribs, etc.
    wing = build_hook3()
    harness = gsim.harness.Spherical(
        mass=75, z_riser=0.5, S=0.55, CD=0.8, kappa_w=0.15,
    )
    glider = gsim.paraglider.Paraglider6a(wing, harness)
    # glider = gsim.paraglider.Paraglider9a(wing, harness)

    # print("Plotting the wing performance curves")
    # plot_wing_coefficients(wing)

    print("\nFinished building the glider.\n")
    # embed()
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
    q_b2e = quaternion.euler_to_quaternion(eq["Theta_b2e"])
    q_e2b = q_b2e * [-1, 1, 1, 1]
    v_R2e = quaternion.apply_quaternion_rotation(q_e2b, eq["v_R2e"])

    # For the `Paraglider6a` model
    a_R2e, alpha_b2e, _ = glider.accelerations(
        v_R2e=eq["v_R2e"],
        omega_b2e=[0, 0, 0],
        g=quaternion.apply_quaternion_rotation(q_b2e, [0, 0, 9.8]),
        rho_air=1.2,
        reference_solution=eq["reference_solution"],
    )
    # For the `Paraglider9a` model
    # a_R2e, alpha_b2e, alpha_p2b, _ = glider.accelerations(
    #     v_R2e=eq["v_R2e"],
    #     omega_b2e=[0, 0, 0],
    #     omega_p2e=[0, 0, 0],
    #     Theta_p2b=eq["Theta_p2b"],
    #     g=quaternion.apply_quaternion_rotation(q_b2e, [0, 0, 9.8]),
    #     rho_air=1.2,
    #     reference_solution=eq["reference_solution"],
    # )

    print("Equilibrium state:")
    print(f"  alpha_b:     {np.rad2deg(eq['alpha_b']):>6.3f} [deg]")
    print(f"  theta_b:     {np.rad2deg(eq['Theta_b2e'][1]):>6.3f} [deg]")
    print(f"  Glide angle: {np.rad2deg(eq['gamma_b']):>6.3f} [deg]")
    print(f"  Glide ratio: {eq['glide_ratio']:>6.3f}")
    print(f"  Glide speed: {np.linalg.norm(v_R2e):>6.3f}")
    print()
    print("For verification of the equilibrium state:")
    print(f"  v_R2e:       {v_R2e.round(4)}")
    print(f"  a_R2e:       {a_R2e.round(4)}")
    print(f"  alpha_b2e:   {np.rad2deg(alpha_b2e).round(4)}")

    print("\n<pausing before polar curves>\n")
    embed()

    input("Plot the polar curve?  Press any key")
    plot_polar_curve(glider, approximate=approximate)
