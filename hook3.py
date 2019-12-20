from IPython import embed

import matplotlib.pyplot as plt

import numpy as np

import Airfoil
import Parafoil
import BrakeGeometry
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider
import plots


def plot_polar_curve(glider, N=51):
    """Compute the equilibrium conditions and plot the polar curves."""
    speedbar_equilibriums = np.empty((N, 5))
    delta_as = np.linspace(0, 1, N)
    print("Calculating equilibriums over the range of speedbar")
    ref = None  # Reference solution; speeds convergence
    for n, da in enumerate(delta_as):
        print(f"\r{da:.2f}", end="")
        alpha_eq, Theta_eq, V_eq, ref = glider.equilibrium_glide(
            0, da, rho_air=1.2, reference_solution=ref,
        )
        gamma_eq = alpha_eq - Theta_eq
        GR = 1 / np.tan(gamma_eq)
        speedbar_equilibriums[n] = (alpha_eq, Theta_eq, gamma_eq, GR, V_eq)
    print()

    brake_equilibriums = np.empty((N, 5))
    delta_Bs = np.linspace(0, 1, N)
    print("Calculating equilibriums over the range of brake")
    ref = None
    for n, db in enumerate(delta_Bs):
        print("\rdb: {:.2f}".format(db), end="")
        try:
            alpha_eq, Theta_eq, V_eq, ref = glider.equilibrium_glide(
                db, 0, rho_air=1.2, reference_solution=ref,
            )
        except Parafoil.ForceEstimator.ConvergenceError:
            print("\nConvergence started failing. Aborting early.")
            break
        gamma_eq = alpha_eq - Theta_eq
        GR = 1 / np.tan(gamma_eq)
        brake_equilibriums[n] = (alpha_eq, Theta_eq, gamma_eq, GR, V_eq)
    print()

    # Truncate everything after convergence failed
    delta_Bs = delta_Bs[:n]
    brake_equilibriums = brake_equilibriums[:n]

    # Build the polar curves
    be, se = brake_equilibriums.T, speedbar_equilibriums.T
    brake_polar = be[4] * np.array([np.cos(be[2]), -np.sin(be[2])])
    speedbar_polar = se[4] * np.array([np.cos(se[2]), -np.sin(se[2])])

    fig, ax = plt.subplots(2, 2)  # [[alpha_eq, polar curve], [Theta_eq, GR]]

    # alpha_eq
    ax[0, 0].plot(-delta_Bs, np.rad2deg(brake_equilibriums.T[0]), "r")
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
    ax[1, 0].plot(-delta_Bs, np.rad2deg(brake_equilibriums.T[1]), "r")
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


def plot_CL_curve(glider, delta_B=0, delta_a=0, rho_air=1.2):
    alphas = np.deg2rad(np.linspace(-8, 20, 50))
    Fs, Ms = [], []
    reference_solution = None
    for alpha in alphas:
        F, M, reference_solution, = glider.forces_and_moments(
            UVW=[np.cos(alpha), 0, np.sin(alpha)],
            PQR=[0, 0, 0],
            g=[0, 0, 0],
            rho_air=rho_air,
            delta_Bl=delta_B,
            delta_Br=delta_B,
            reference_solution=reference_solution,
        )
        Fs.append(F)
        Ms.append(M)

    CLs = []
    CDs = []
    CMs = []
    for n, F in enumerate(Fs):
        L = F[0] * np.sin(alphas[n]) - F[2] * np.cos(alphas[n])
        D = -F[0] * np.cos(alphas[n]) - F[2] * np.sin(alphas[n])
        CL = 2 * L / (rho_air * glider.wing.parafoil.S)
        CD = 2 * D / (rho_air * glider.wing.parafoil.S)
        CM = (
            2
            * Ms[n][1]
            / (rho_air * glider.wing.parafoil.S * glider.wing.parafoil.chord_length(0))
        )
        CLs.append(CL)
        CDs.append(CD)
        CMs.append(CM)

    deltas = np.full_like(alphas, delta_B)
    Cls = glider.wing.parafoil.airfoil.coefficients.Cl(alphas, deltas)
    Cds = glider.wing.parafoil.airfoil.coefficients.Cd(alphas, deltas)
    Cms = glider.wing.parafoil.airfoil.coefficients.Cm(alphas, deltas)

    fig, ax = plt.subplots(3, 2, figsize=(9, 8))
    ax[0, 0].plot(np.rad2deg(alphas), CLs, label="CL")
    ax[0, 0].plot(np.rad2deg(alphas), Cls, "k--", linewidth=0.75, label="Cl")
    ax[1, 0].plot(np.rad2deg(alphas), CDs, label="CD")
    ax[1, 0].plot(np.rad2deg(alphas), Cds, "k--", linewidth=0.75, label="Cd")
    ax[2, 0].plot(np.rad2deg(alphas), CMs, label="CM")
    ax[2, 0].plot(np.rad2deg(alphas), Cms, "k--", linewidth=0.75, label="Cm")
    plt.setp(ax[:, 0], xlabel="alpha [deg]")
    ax[0, 0].set_ylabel("Lift Coefficient")
    ax[1, 0].set_ylabel("Drag Coefficient")
    ax[2, 0].set_ylabel("Pitching Coefficient")

    ax[0, 1].plot(np.rad2deg(alphas), np.array(CLs) / np.array(CDs))
    ax[0, 1].set_xlabel("alpha [deg]")
    ax[0, 1].set_ylabel("CL/CD")

    ax[1, 1].plot(CDs, CLs)
    ax[1, 1].set_xlabel("CD")
    ax[1, 1].set_ylabel("CL")

    for ax in fig.axes:
        if len(ax.lines) > 0:
            ax.grid()
        if len(ax.lines) > 1:
            ax.legend()

    fig.tight_layout()  # Prevent axes overlapping titles

    plt.show()

    embed()


def build_hook3():

    print("Building an (approximate) Niviuk Hook 3 23\n")

    # -----------------------------------------------------------------------
    # Airfoil

    print("Airfoil: NACA 24018, curving flap\n")
    airfoil_geo = Airfoil.NACA(24018, convention="british")
    airfoil_coefs = Airfoil.GridCoefficients('polars/exp_curving_24018.csv')
    delta_max = np.deg2rad(10.00)  # True value: 13.28

    # print("\nAirfoil: NACA 23015, curving flap")
    # airfoil_geo = Airfoil.NACA(23015, convention='british')
    # airfoil_coefs = Airfoil.GridCoefficients('polars/exp_curving_23015.csv')
    # delta_max = np.deg2rad(12.00)  # True value: 13.38

    airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)

    # -----------------------------------------------------------------------
    # Parafoil: an approximate Niviuk Hook 3, size 23

    # True technical specs
    chord_min, chord_max, chord_mean = 0.52, 2.58, 2.06
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat / AR_flat
    S, b, AR = 19.55, 8.84, 4.00

    # Build an approximate version. Anything not directly from the technical
    # specs is a guess to make the projected values match up.
    c_root = chord_max / (b_flat / 2)  # Proportional values
    c_tip = chord_min / (b_flat / 2)

    # The geometric torsion distribution is uncertain. In Sec. 11.4, pg 17 of
    # the manual ("Line Plan") it appears to have a roughly square-root
    # spanwise distribution with a maximum valueof 6 or so? Unfortunately that
    # distribution proves difficult for Phillips method, so here I provide an
    # "easier" alternative.
    # torsion = Parafoil.PolynomialTorsion(start=0.0, peak=6, exponent=0.75)
    torsion = Parafoil.PolynomialTorsion(start=0.8, peak=4, exponent=2)

    parafoil = Parafoil.ParafoilGeometry(
        airfoil=airfoil,
        chord_length=Parafoil.elliptical_chord(root=c_root, tip=c_tip),
        r_x=0.75,
        x=0,
        r_yz=1.00,
        yz=Parafoil.elliptical_lobe(mean_anhedral=33, max_anhedral_rate=67),
        torsion=torsion,
        intakes=Parafoil.SimpleIntakes(0.85, -0.04, -0.09),  # FIXME: guess
        b_flat=b_flat,  # Option 1: Determine the scale using the planform
        # b=b,  # Option 2: Determine the scale using the lobe
    )

    print("Parafoil geometry:")
    print(f"  planform flat span: {parafoil.b_flat:>6.3f}")
    print(f"  planform flat area: {parafoil.S_flat:>6.3f}")
    print(f"  planform flat AR:   {parafoil.AR_flat:>6.3f}")
    # print(f"  planform flat SMC   {parafoil.SMC:>6.3f}")
    # print(f"  planform flat MAC:  {parafoil.MAC:>6.3f}")

    print(f"  projected span:     {parafoil.b:>6.3f}")
    print(f"  projected area:     {parafoil.S:>6.3f}")
    print(f"  projected AR:       {parafoil.AR:>6.3f}")
    print()

    # print("Drawing the parafoil")
    # plots.plot_parafoil_planform_topdown(parafoil)
    # plots.plot_parafoil_planform(parafoil, N_sections=50)
    # plots.plot_parafoil_geo(parafoil, N_sections=131)

    # Compare to the Hook 3 manual, sec 11.4 "Line Plan", page 17
    # plots.plot_parafoil_geo_topdown(parafoil, N_sections=77)

    # -----------------------------------------------------------------------
    # Brake geometry
    #
    # FIXME: this is completely unknown. For now this just a standin until I
    #        implement a proper line geometry and can calculate the deflection
    #        distributions from that.

    p_start = 0.00
    p_peak = BrakeGeometry.Cubic.p_peak_min(p_start) + 1e-9
    brakes = BrakeGeometry.Cubic(p_start, p_peak, delta_max)

    # -----------------------------------------------------------------------
    # Wing and glider

    wing = ParagliderWing(
        parafoil,
        Parafoil.Phillips,
        brakes,
        d_riser=0.49,  # FIXME: Source? Trying to match `Theta_eq` at trim?
        z_riser=6.8,  # From the Hook 3 manual PDF, section 11.1
        pA=0.11,  # Approximated from a picture in the manual
        pC=0.59,
        kappa_a=0.15,  # From the Hook 3 manual
        rho_upper=39 / 1000,  # [kg/m^2]  Porcher 9017 E77A
        rho_lower=35 / 1000,  # [kg/m^2]  Dominico N20DMF
    )

    # Note to self: the wing should weight 4.7kG in total; according to these
    # specs, and the `rho_upper`/`rho_lower` embedded in ParagliderWing, the
    # wing materials I'm accounting for total to 1.83kg, so there's a lot left
    # in the lines, risers, ribs, etc.

    harness = Harness.Spherical(mass=75, z_riser=0.5, S=0.55, CD=0.8)

    glider = Paraglider(wing, harness)

    # print("Plotting the basic glider performance curves")
    # plot_CL_curve(glider)

    # print("\nFinished building the glider.\n")
    # embed()
    # 1/0

    return glider


if __name__ == "__main__":

    print("\n-------------------------------------------------------------\n")

    glider = build_hook3()

    print("\nComputing the wing equilibrium...")
    alpha, Theta, V, _ = glider.equilibrium_glide(0, 0, rho_air=1.2)

    print(f"  alpha: {np.rad2deg(alpha):>6.3f} [deg]")
    print(f"  Theta: {np.rad2deg(Theta):>6.3f} [deg]")
    print(f"  Speed: {V:>6.3f} [m/s]")

    print()
    print("Computing the glider equilibrium...")
    UVW = V * np.array([np.cos(alpha), 0, np.sin(alpha)])
    PQR = np.array([0, 0, 0])
    g = 9.8 * np.array([-np.sin(Theta), 0, np.cos(Theta)])
    F, M, _, = glider.forces_and_moments(
        UVW, PQR, g=g, rho_air=1.2, delta_Bl=0, delta_Br=0,
    )
    alpha_eq, Theta_eq, V_eq, _ = glider.equilibrium_glide(0, 0, rho_air=1.2)
    gamma_eq = alpha_eq - Theta_eq

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
    # J_wing = wing.inertia(rho_air=1.2, N=5000)
    # alpha_rad = np.linalg.inv(J_wing) @ M
    # print("Angular acceleration at equilibrium:", np.rad2deg(alpha_rad))

    print("\n<pausing before polar curves>\n")
    embed()

    input("Plot the polar curve?  Press any key")
    plot_polar_curve(glider)
