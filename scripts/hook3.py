from IPython import embed

import matplotlib.pyplot as plt

import numpy as np

import pfh.glidersim as gsim


def plot_polar_curve(glider, N=51):
    """Compute the equilibrium conditions and plot the polar curves."""
    speedbar_equilibriums = np.empty((N, 5))
    delta_as = np.linspace(0, 1, N)
    print("Calculating equilibriums over the range of speedbar")
    ref = None  # Reference solution; speeds convergence
    V_eq = 10  # Initial guess
    for n, da in enumerate(delta_as):
        print(f"\r{da:.2f}", end="")
        alpha_eq, Theta_eq, V_eq, ref = glider.equilibrium_glide(
            da, 0, V_eq_proposal=V_eq, rho_air=1.2, reference_solution=ref,
        )
        gamma_eq = alpha_eq - Theta_eq
        GR = 1 / np.tan(gamma_eq)
        speedbar_equilibriums[n] = (alpha_eq, Theta_eq, gamma_eq, GR, V_eq)
    print()

    brake_equilibriums = np.empty((N, 5))
    delta_bs = np.linspace(0, 1, N)
    print("Calculating equilibriums over the range of brake")
    ref = None
    V_eq = 10  # Initial guess
    for n, db in enumerate(delta_bs):
        print("\rdb: {:.2f}".format(db), end="")
        try:
            alpha_eq, Theta_eq, V_eq, ref = glider.equilibrium_glide(
                0, db, V_eq_proposal=V_eq, rho_air=1.2, reference_solution=ref,
            )
        except gsim.foil.ForceEstimator.ConvergenceError:
            print("\nConvergence started failing. Aborting early.")
            break
        gamma_eq = alpha_eq - Theta_eq
        GR = 1 / np.tan(gamma_eq)
        brake_equilibriums[n] = (alpha_eq, Theta_eq, gamma_eq, GR, V_eq)
    print()

    # Truncate everything after convergence failed
    delta_bs = delta_bs[:n]
    brake_equilibriums = brake_equilibriums[:n]

    # Build the polar curves
    be, se = brake_equilibriums.T, speedbar_equilibriums.T
    brake_polar = be[4] * np.array([np.cos(be[2]), -np.sin(be[2])])
    speedbar_polar = se[4] * np.array([np.cos(se[2]), -np.sin(se[2])])

    fig, ax = plt.subplots(2, 2)  # [[alpha_eq, polar curve], [Theta_eq, GR]]

    # alpha_eq
    ax[0, 0].plot(-delta_bs, np.rad2deg(brake_equilibriums.T[0]), "r")
    ax[0, 0].plot(delta_as, np.rad2deg(speedbar_equilibriums.T[0]), "g")
    ax[0, 0].set_ylabel("alpha_eq [deg]")

    # Polar curve
    #
    # For (m/s, km/h)
    # ax[0, 1].plot(3.6*brake_polar[0], brake_polar[1], 'r')
    # ax[0, 1].plot(3.6*speedbar_polar[0], speedbar_polar[1], 'g')
    # ax[0, 1].set_xlabel('airspeed [km/h]')
    #
    # For (m/s, m/s)
    ax[0, 1].plot(brake_polar[0], brake_polar[1], "r")
    ax[0, 1].plot(speedbar_polar[0], speedbar_polar[1], "g")
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xlabel("airspeed [m/s]")
    ax[0, 1].set_xlim(0, 25)
    ax[0, 1].set_ylim(-8, 0)
    ax[0, 1].set_ylabel("sink rate [m/s]")
    ax[0, 1].grid(which="both")
    ax[0, 1].minorticks_on()

    # Theta_eq
    ax[1, 0].plot(-delta_bs, np.rad2deg(brake_equilibriums.T[1]), "r")
    ax[1, 0].plot(delta_as, np.rad2deg(speedbar_equilibriums.T[1]), "g")
    ax[1, 0].set_xlabel("control input [percentage]")
    ax[1, 0].set_ylabel("Theta_eq [deg]")

    # Glide ratio
    #
    # For (m/s, km/h)
    # ax[1, 1].plot(3.6*brake_polar[0], brake_equilibriums.T[3], 'r')
    # ax[1, 1].plot(3.6*speedbar_polar[0], speedbar_equilibriums.T[3], 'g')
    # ax[1, 1].set_xlabel('airspeed [km/h]')
    #
    # For (m/s, m/s)
    ax[1, 1].plot(brake_polar[0], brake_equilibriums.T[3], "r")
    ax[1, 1].plot(speedbar_polar[0], speedbar_equilibriums.T[3], "g")
    ax[1, 1].set_xlim(0, 25)
    ax[1, 1].set_xlabel("airspeed [m/s]")
    ax[1, 1].set_ylabel("Glide ratio")

    plt.show()

    embed()


def plot_foil_coefficients(glider, delta_a=0, delta_b=0, V_mag=10, rho_air=1.2):
    alphas = np.deg2rad(np.linspace(-10, 25, 50))
    Fs, Ms = [], []
    reference_solution = None
    for k, alpha in enumerate(alphas):
        print(f"\ralpha: {np.rad2deg(alpha):6.2f}", end="")
        try:
            F, M, reference_solution, = glider.forces_and_moments(
                UVW=V_mag * np.array([np.cos(alpha), 0, np.sin(alpha)]),
                PQR=[0, 0, 0],
                g=[0, 0, 0],
                rho_air=rho_air,
                delta_a=delta_a,
                delta_bl=delta_b,
                delta_br=delta_b,
                reference_solution=reference_solution,
            )
            Fs.append(F)
            Ms.append(M)
        except gsim.foil.ForceEstimator.ConvergenceError:
            break
    alphas = alphas[:k]
    Fs = Fs[:k]
    Ms = Ms[:k]

    CLs, CDs, CMs = [], [], []
    for n, F in enumerate(Fs):
        L = F[0] * np.sin(alphas[n]) - F[2] * np.cos(alphas[n])
        D = -F[0] * np.cos(alphas[n]) - F[2] * np.sin(alphas[n])
        CL = L / (0.5 * rho_air * V_mag ** 2 * glider.wing.parafoil.S)
        CD = D / (0.5 * rho_air * V_mag ** 2 * glider.wing.parafoil.S)
        CM = Ms[n][1] / (
            0.5
            * rho_air
            * V_mag ** 2
            * glider.wing.parafoil.S
            * glider.wing.parafoil.chord_length(0)
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
    airfoil_coefs = gsim.airfoil.GridCoefficients("polars/exp_curving_24018/gridded.csv")
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
        yz=gsim.foil.elliptical_lobe(mean_anhedral=33, max_anhedral=67),
        chord_length=chord_length,
        torsion=torsion,
    )

    parafoil = gsim.foil.SimpleFoil(
        airfoil=airfoil,
        chords=chord_surface,
        # b=b,  # Option 1: Scale the using the projected span
        b_flat=b_flat,  # Option 2: Scale the using the flattened span
        intakes=gsim.foil.SimpleIntakes(0.85, -0.04, -0.09),  # FIXME: guess
    )

    print("Parafoil geometry:")
    print(f"  flattened span: {parafoil.b_flat:>6.3f}")
    print(f"  flattened area: {parafoil.S_flat:>6.3f}")
    print(f"  flattened AR:   {parafoil.AR_flat:>6.3f}")
    # print(f"  planform flat SMC   {parafoil.SMC:>6.3f}")
    # print(f"  planform flat MAC:  {parafoil.MAC:>6.3f}")

    print(f"  projected span: {parafoil.b:>6.3f}")
    print(f"  projected area: {parafoil.S:>6.3f}")
    print(f"  projected AR:   {parafoil.AR:>6.3f}")
    print()

    # print("Drawing the parafoil")
    # gsim.plots.plot_foil(parafoil, N_sections=131, flatten=False)
    # gsim.plots.plot_foil(parafoil, N_sections=71, flatten=True)
    # gsim.plots.plot_foil_topdown(parafoil, N_sections=51)
    # gsim.plots.plot_foil_topdown(parafoil, N_sections=51, flatten=True)

    # print("\nPausing...")
    # embed()

    # Compare to the Hook 3 manual, sec 11.4 "Line Plan", page 17
    # plots.plot_foil_topdown(parafoil, N_sections=77)

    # -----------------------------------------------------------------------
    # Brake geometry
    #
    # FIXME: this is completely unknown. For now this just a standin until I
    #        implement a proper line geometry and can calculate the deflection
    #        distributions from that.

    p_start = 0.10
    p_peak = gsim.brake_geometry.Cubic.p_peak_min(p_start) + 1e-9
    brakes = gsim.brake_geometry.Cubic(p_start, p_peak, delta_max)

    # -----------------------------------------------------------------------
    # Wing and glider

    wing = gsim.paraglider_wing.ParagliderWing(
        parafoil=parafoil,
        force_estimator=gsim.foil.Phillips(parafoil, V_ref_mag=10, K=31),
        brake_geo=brakes,
        d_riser=0.49,  # FIXME: Source? Trying to match `Theta_eq` at trim?
        z_riser=6.8,  # From the Hook 3 manual PDF, section 11.1
        pA=0.11,  # Approximated from the line plan in the manual PDF, page 17
        pC=0.59,
        kappa_a=0.15,  # From the Hook 3 manual
        rho_upper=39 / 1000,  # [kg/m^2]  Porcher 9017 E77A
        rho_lower=35 / 1000,  # [kg/m^2]  Dominico N20DMF
    )

    # Note to self: the wing should weight 4.7kG in total; according to these
    # specs, and the `rho_upper`/`rho_lower` embedded in ParagliderWing, the
    # wing materials I'm accounting for total to 1.83kg, so there's a lot left
    # in the lines, risers, ribs, etc.

    harness = gsim.harness.Spherical(mass=75, z_riser=0.5, S=0.55, CD=0.8)

    glider = gsim.paraglider.Paraglider(wing, harness)

    # print("Plotting the basic glider performance curves")
    # plot_foil_coefficients(glider)

    print("\nFinished building the glider.\n")
    # embed()
    # 1/0

    return glider


if __name__ == "__main__":

    print("\n-------------------------------------------------------------\n")

    glider = build_hook3()

    print("\nComputing the glider equilibrium...")
    alpha_eq, Theta_eq, V_eq, _ = glider.equilibrium_glide(
        delta_a=0.0,
        delta_b=0.0,
        V_eq_proposal=10,
        rho_air=1.2,
    )
    gamma_eq = alpha_eq - Theta_eq
    UVW = V_eq * np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
    PQR = np.array([0, 0, 0])
    g = 9.8 * np.array([-np.sin(Theta_eq), 0, np.cos(Theta_eq)])
    F, M, _, = glider.forces_and_moments(
        UVW, PQR, g=g, rho_air=1.2, delta_bl=0, delta_br=0,
    )

    print(f"  UVW:   {UVW.round(4)}")
    print(f"  F:     {F.round(4)}")
    print(f"  M:     {M.round(4)}")
    print()
    print(f"  alpha:       {np.rad2deg(np.arctan2(UVW[2], UVW[0])):>6.3f} [deg]")
    print(f"  Theta:       {np.rad2deg(Theta_eq):>6.3f} [deg]")
    print(f"  Glide angle: {np.rad2deg(gamma_eq):>6.3f} [deg]")
    print(f"  Glide ratio: {1 / np.tan(gamma_eq):>6.3f}")
    print(f"  Glide speed: {V_eq:>6.3f}")

    # Sanity check the dynamics
    # J_wing = glider.wing.inertia(rho_air=1.2, N=5000)
    # alpha_rad = np.linalg.inv(J_wing) @ M
    # print("\nAngular acceleration at equilibrium:", np.rad2deg(alpha_rad))

    print("\n<pausing before polar curves>\n")
    embed()

    input("Plot the polar curve?  Press any key")
    plot_polar_curve(glider)
