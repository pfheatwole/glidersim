import matplotlib.pyplot as plt

import numpy as np
from numpy import rad2deg

from IPython import embed

import Airfoil
import Parafoil

import BrakeGeometry
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider
import plots


def plot_polar_curve(glider, N=51):
    # Compute the equilibrium conditions and plot the polar curves
    speedbar_equilibriums = np.empty((N, 5))
    delta_ss = np.linspace(0, 1, N)
    print("Calculating equilibriums over the range of speedbar")
    for n, ds in enumerate(delta_ss):
        print("\rds: {:.2f}".format(ds), end="")
        alpha_eq, Theta_eq, V_eq = glider.equilibrium_glide(0, ds, rho=1.2)
        gamma_eq = alpha_eq - Theta_eq
        GR = 1/np.tan(gamma_eq)
        speedbar_equilibriums[n] = (alpha_eq, Theta_eq, gamma_eq, GR, V_eq)
    print()

    brake_equilibriums = np.empty((N, 5))
    delta_Bs = np.linspace(0, 1, N)
    print("Calculating equilibriums over the range of brake")
    for n, db in enumerate(delta_Bs):
        print("\rdb: {:.2f}".format(db), end="")
        alpha_eq, Theta_eq, V_eq = glider.equilibrium_glide(db, 0, rho=1.2)
        gamma_eq = alpha_eq - Theta_eq
        GR = 1/np.tan(gamma_eq)
        brake_equilibriums[n] = (alpha_eq, Theta_eq, gamma_eq, GR, V_eq)

    # Build the polar curves
    be, se = brake_equilibriums.T, speedbar_equilibriums.T
    brake_polar = (be[4]*np.array([np.cos(be[2]), -np.sin(be[2])]))
    speedbar_polar = se[4]*np.array([np.cos(se[2]), -np.sin(se[2])])

    fig, ax = plt.subplots(2, 2)  # [[alpha_eq, polar curve], [Theta_eq, GR]]

    # alpha_eq
    ax[0, 0].plot(-delta_Bs, rad2deg(brake_equilibriums.T[0]), 'r')
    ax[0, 0].plot(delta_ss, rad2deg(speedbar_equilibriums.T[0]), 'g')
    ax[0, 0].set_ylabel('alpha_eq [deg]')

    # Polar curve
    #
    # For (m/s, km/h)
    # ax[0, 1].plot(3.6*brake_polar[0], brake_polar[1], 'r')
    # ax[0, 1].plot(3.6*speedbar_polar[0], speedbar_polar[1], 'g')
    # ax[0, 1].set_xlabel('airspeed [km/h]')
    #
    # For (m/s, m/s)
    ax[0, 1].plot(brake_polar[0], brake_polar[1], 'r')
    ax[0, 1].plot(speedbar_polar[0], speedbar_polar[1], 'g')
    ax[0, 1].set_aspect('equal')
    ax[0, 1].set_xlabel('airspeed [m/s]')
    ax[0, 1].set_xlim(0, 25)
    ax[0, 1].set_ylim(-8, 0)
    ax[0, 1].set_ylabel('sink rate [m/s]')
    ax[0, 1].grid(which='both')
    ax[0, 1].minorticks_on()

    # Theta_eq
    ax[1, 0].plot(-delta_Bs, rad2deg(brake_equilibriums.T[1]), 'r')
    ax[1, 0].plot(delta_ss, rad2deg(speedbar_equilibriums.T[1]), 'g')
    ax[1, 0].set_xlabel('control input [percentage]')
    ax[1, 0].set_ylabel('Theta_eq [deg]')

    # Glide ratio
    #
    # For (m/s, km/h)
    # ax[1, 1].plot(3.6*brake_polar[0], brake_equilibriums.T[3], 'r')
    # ax[1, 1].plot(3.6*speedbar_polar[0], speedbar_equilibriums.T[3], 'g')
    # ax[1, 1].set_xlabel('airspeed [km/h]')
    #
    # For (m/s, m/s)
    ax[1, 1].plot(brake_polar[0], brake_equilibriums.T[3], 'r')
    ax[1, 1].plot(speedbar_polar[0], speedbar_equilibriums.T[3], 'g')
    ax[1, 1].set_xlim(0, 25)
    ax[1, 1].set_xlabel('airspeed [m/s]')
    ax[1, 1].set_ylabel('Glide ratio')

    plt.show()

    embed()


def plot_CL_curve(glider, delta_B=0, delta_S=0, rho_air=1.2):
    alphas = np.deg2rad(np.linspace(-8, 20, 50))
    Fs, Ms = [], []
    for alpha in alphas:
        g = [0, 0, 0]
        UVW = np.array([np.cos(alpha), 0, np.sin(alpha)])
        F, M, _, = glider.forces_and_moments(UVW, [0, 0, 0], g=g, rho=rho_air,
                                         delta_Bl=delta_B, delta_Br=delta_B)
        Fs.append(F)
        Ms.append(M)

    CLs = []
    CDs = []
    CMs = []
    for n, F in enumerate(Fs):
        L = F[0]*np.sin(alphas[n]) - F[2]*np.cos(alphas[n])
        D = -F[0]*np.cos(alphas[n]) - F[2]*np.sin(alphas[n])
        CL = 2*L/(rho_air * glider.wing.parafoil.S)
        CD = 2*D/(rho_air * glider.wing.parafoil.S)
        CM = 2*Ms[n][1]/(rho_air * glider.wing.parafoil.S * glider.wing.parafoil.planform.fc(0))
        CLs.append(CL)
        CDs.append(CD)
        CMs.append(CM)

    deltas = np.full_like(alphas, delta_B)
    Cls = glider.wing.parafoil.airfoil.coefficients.Cl(alphas, deltas)
    Cds = glider.wing.parafoil.airfoil.coefficients.Cd(alphas, deltas)
    Cms = glider.wing.parafoil.airfoil.coefficients.Cm(alphas, deltas)

    fig, ax = plt.subplots(3, 2, figsize=(9, 8))
    ax[0, 0].plot(np.rad2deg(alphas), CLs, label='CL')
    ax[0, 0].plot(np.rad2deg(alphas), Cls, 'k--', linewidth=0.75, label='Cl')
    ax[1, 0].plot(np.rad2deg(alphas), CDs, label='CD')
    ax[1, 0].plot(np.rad2deg(alphas), Cds, 'k--', linewidth=0.75, label='Cd')
    ax[2, 0].plot(np.rad2deg(alphas), CMs, label='CM')
    ax[2, 0].plot(np.rad2deg(alphas), Cms, 'k--', linewidth=0.75, label='Cm')
    plt.setp(ax[:, 0], xlabel='alpha [deg]')
    ax[0, 0].set_ylabel('Lift Coefficient')
    ax[1, 0].set_ylabel('Drag Coefficient')
    ax[2, 0].set_ylabel('Pitching Coefficient')

    ax[0, 1].plot(np.rad2deg(alphas), np.array(CLs)/np.array(CDs))
    ax[0, 1].set_xlabel('alpha [deg]')
    ax[0, 1].set_ylabel('CL/CD')

    ax[1, 1].plot(CDs, CLs)
    ax[1, 1].set_xlabel('CD')
    ax[1, 1].set_ylabel('CL')

    for ax in fig.axes:
        if len(ax.lines) > 0:
            ax.grid()
        if len(ax.lines) > 1:
            ax.legend()

    fig.tight_layout()  # Prevent axes overlapping titles

    plt.show()

    embed()


def build_elliptical_parafoil(b_flat, taper, dMed, sMed, airfoil,
                              SMC=None, MAC=None,
                              dMax=None, sMax=None,
                              torsion_max=0, torsion_exponent=6):

    if SMC is None and MAC is None:
        raise ValueError("One of the SMC or MAC are required")

    if dMed > 0 or (dMax is not None and dMax > 0):
        raise ValueError("dihedral must be negative")

    if sMed < 0 or (sMax is not None and sMax < 0):
        raise ValueError("sweep must be positive")  # FIXME: why?

    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print(f"Using minimum max dihedral ({dMax})")

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print(f"Using minimum max sweep ({sMax})")

    if SMC is not None:
        c0 = Parafoil.EllipticalPlanform.SMC_to_c0(SMC, taper)
    else:
        c0 = Parafoil.EllipticalPlanform.MAC_to_c0(MAC, taper)

    planform = Parafoil.EllipticalPlanform(
        b_flat, c0, taper, sMed, sMax, torsion_exponent, torsion_max)
    lobe = Parafoil.EllipticalLobe(dMed, dMax)

    return Parafoil.ParafoilGeometry(planform, lobe, airfoil)


def main():

    print('\n-------------------------------------------------------------\n')

    # -----------------------------------------------------------------------
    # Airfoil

    # print("\nAirfoil: GNULAB3, simple flap, hinge at 80%")
    # TODO: AirfoilGeometry importer for .DAT files (`gnulab3.dat`)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)

    # print("\nAirfoil: NACA2412, simple flap, hinge at 80%")
    # airfoil_geo = Airfoil.NACA4(2412)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca2412_xhinge80_yhinge_50.csv', 0.8)

    # print("\nAirfoil: NACA4412, simple flap, hinge at 80%")
    # airfoil_geo = Airfoil.NACA4(4412)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca4412_xhinge80_yhinge_50.csv', 0.8)

    # print("\nAirfoil: NACA4412, simple flap, hinge at 80%")
    # airfoil_geo = Airfoil.NACA4(4415)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca4415_xhinge80_yhinge_50.csv')

    # print("\nAirfoil: NACA4415, simple flap, hinge at 80%")
    # airfoil_geo = Airfoil.NACA4(4415)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca4415_xhinge80_yhinge_50_Re2_cleaned.csv')
    # delta_max = np.deg2rad(50)*0.99 * (1 - 0.8)   # FIXME: magic number!

    # print("\nAirfoil: NACA4415, simple flap, variable xhinge")
    # airfoil_geo = Airfoil.NACA4(4415)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca4415_fixed_flap65.csv')
    # delta_max = np.deg2rad(10)  # FIXME: magic number

    # print("\nAirfoil: NACA4418, curving flap")
    # airfoil_geo = Airfoil.NACA4(4418, open_TE=False, convention='british')
    # airfoil_coefs = Airfoil.GridCoefficients('polars/NACA4418_theta30_epsilon10_Ku4_Kl0.5_ver3.csv')
    # delta_max = np.deg2rad(10.8)  # FIXME: magic number

    # print("\nAirfoil: NACA 23015, curving flap")
    # airfoil_geo = Airfoil.NACA5(23015, convention='british')
    # airfoil_coefs = Airfoil.GridCoefficients('polars/NACA23015_theta30_epsilon10_Ku3_Kl0.5.csv')  # delta_max = 13.84
    # delta_max = np.deg2rad(13.8)  # FIXME: magic number
    # airfoil_coefs = Airfoil.GridCoefficients('polars/NACA23015_theta30_epsilon10_Ku6_Kl0.75.csv')  # delta_max = 11.75
    # delta_max = np.deg2rad(11.74)  # FIXME: magic number
    # airfoil_coefs = Airfoil.GridCoefficients('polars/NACA23015_theta30_epsilon10_Ku3_Kl0.5_N0.1.csv')  # delta_max = 15.15
    # delta_max = np.deg2rad(13)  # FIXME: magic number
    # For a Hook 3, I want max brake to produce a mininum speed of ~6.6m/s

    # print("\nAirfoil: NACA 24018, curving flap")
    # airfoil_geo = Airfoil.NACA5(24018, convention='british')
    # airfoil_coefs = Airfoil.GridCoefficients('polars/NACA24018_theta30_epsilon10_Ku3_Kl0.5.csv')  # delta_max = 13.76
    # delta_max = np.deg2rad(13.75)  # FIXME: magic number

    print("\nAirfoil: NACA 24018, curving flap")
    airfoil_geo = Airfoil.NACA5(24018, convention='british')
    airfoil_coefs = Airfoil.GridCoefficients('polars/exp_curving_24018.csv')  # delta_max = 13.38
    delta_max = np.deg2rad(13.25)  # FIXME: magic number

    # plots.plot_airfoil_geo(airfoil_geo)

    # -----------------------------------------------------------------------
    # Parafoil

    # Hook3 specs:
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat/AR_flat
    S, b, AR = 19.55, 8.84, 4.00
    V_limits_hook3 = np.array([24, 38, 52]) / 3.6  # min/trim/max in m/s

    airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)
    # parafoil = build_elliptical_parafoil(
    #     b_flat=10, SMC=2.5, taper=0.35, dMed=-25, dMax=-70,
    #     sMed=15, torsion_max=0, airfoil=airfoil)
    parafoil = build_elliptical_parafoil(   # Hook 3 (ish)
        b_flat=b_flat, SMC=SMC_flat, taper=0.35, dMed=-32, dMax=-75,
        sMed=13.5, sMax=40, torsion_max=0, airfoil=airfoil)

    print("planform flat span:", parafoil.planform.b)
    print("planform flat area:", parafoil.planform.S)
    print("planform flat AR:  ", parafoil.planform.AR)
    print("planform flat SMC: ", parafoil.planform.SMC)
    print("planform flat MAC: ", parafoil.planform.MAC)

    print("planform span:", parafoil.b)
    print("planform area:", parafoil.S)
    print("planform AR:  ", parafoil.AR)

    # print("Drawing the parafoil")
    # plots.plot_parafoil_planform_topdown(parafoil)
    # plots.plot_parafoil_planform(parafoil, N_sections=50)
    # plots.plot_parafoil_geo(parafoil, N_sections=50)

    # -----------------------------------------------------------------------
    # Brake geometry

    p_start, p_peak = 0, 0.75
    # p_start, p_peak = 0.00, BrakeGeometry.Cubic.p_peak_min(0.00)
    brakes = BrakeGeometry.Cubic(p_start, p_peak, delta_max)

    # -----------------------------------------------------------------------
    # Wing and glider

    wing = ParagliderWing(parafoil, Parafoil.Phillips, brakes,
                          d_riser=0.49, z_riser=6.8,
                          pA=0.08, pC=0.80,
                          kappa_s=0.15)

    harness = Harness.Spherical(mass=75, z_riser=0.5, S=0.55, CD=0.8)

    glider = Paraglider(wing, harness)

    # -----------------------------------------------------------------------
    # Tests
    alpha, Theta, V = glider.equilibrium_glide(0, 0, rho=1.2)
    UVW = V*np.array([np.cos(alpha), 0, np.sin(alpha)])

    print("Equilibrium condition: alpha={:.3f}, Theta={:.3f}, V={}".format(
        np.rad2deg(alpha), np.rad2deg(Theta), V))

    # P = np.deg2rad(15)
    # Q = np.deg2rad(-5)
    # R = np.deg2rad(15)  # yaw rate = 15 degrees/sec clockwise
    # PQR = np.array([P, Q, R])
    PQR = np.array([0, 0, 0])
    g = 9.8 * np.array([-np.sin(Theta), 0, np.cos(Theta)])
    F, M, _, = glider.forces_and_moments(UVW, PQR, g=g, rho=1.2,
                                     delta_Bl=0, delta_Br=0)

    print("\nGlider results:")
    print("alpha:", np.rad2deg(np.arctan2(UVW[2], UVW[0])))
    print("UVW:  ", UVW.round(4))
    print("PQR:  ", PQR.round(4))
    print("F:    ", F.round(4))
    print("M:    ", M.round(4))
    print()

    alpha_eq, Theta_eq, V_eq = glider.equilibrium_glide(0, 0, rho=1.2)
    gamma_eq = alpha_eq - Theta_eq
    print("Wing equilibrium angle of attack:", np.rad2deg(alpha_eq))

    print("Glider Theta_eq:", np.rad2deg(Theta_eq))
    print("Glider equilibrium glide angle:", np.rad2deg(gamma_eq))
    print("Glider equilibrium glide ratio:", 1/np.tan(gamma_eq))
    print("Glider equilibrium glide speed:", V_eq)

    # Dynamics
    J_wing = wing.inertia(rho_air=1.2, N=5000)
    alpha_rad = np.linalg.inv(J_wing) @ M
    print("angular acceleration in deg/s**2:", np.rad2deg(alpha_rad))

    print("\n<pausing>\n")
    embed()

    input("Plot the polar curve?  Press any key")
    plot_polar_curve(glider)


if __name__ == "__main__":
    main()
