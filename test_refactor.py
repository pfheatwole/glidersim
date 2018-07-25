from functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import minimize_scalar

from IPython import embed

import Airfoil
import Parafoil

import BrakeGeometry
from ParafoilGeometry import Elliptical
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider


def build_elliptical_geo(MAC, AR, taper, dMed, sMed, dMax=None, sMax=None,
                         torsion=0, airfoil_geo=None, sections=None):
    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print("Using minimum max dihedral ({})".format(dMax))

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print("Using minimum max sweep ({})".format(sMax))

    # Compute some missing data in reverse
    c0 = Elliptical.MAC_to_c0(MAC, taper)
    b = Elliptical.AR_to_b(c0, AR, taper)

    if sections is None:
        raise ValueError("FIXME: the `sections` parameter is mandatory")

    # FIXME: this naming is confusing. Elliptical is a geometry, not a wing
    foil_geo = Elliptical(b, c0, taper, dMed, dMax, sMed, sMax, torsion)

    return foil_geo


def plot_coefficients(coefs):
    CLs = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CLs.append(coefs.CL(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CLs):
        # ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
        ax.plot(alphas, np.full_like(alphas, deltas[n]), c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CL')
    plt.show()

    CDs = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CDs.append(coefs.CD(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CDs):
        # ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
        ax.plot(alphas, np.full_like(alphas, deltas[n]), c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CD')
    plt.show()

    CM_c4s = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CM_c4s.append(coefs.CM_c4(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CM_c4s):
        # ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
        ax.plot(alphas, np.full_like(alphas, deltas[n]), c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CM')
    plt.show()


def plot_polar(glider):
    alpha_eq = []
    for d in np.linspace(0, 1, 21)[::-1]:
        alpha_eq.append(glider.wing.alpha_eq(d, 0))
    for d in np.linspace(0, 1, 21):
        alpha_eq.append(glider.wing.alpha_eq(0, d))
    alpha_eq = np.asarray(alpha_eq)

    w_brakes = []
    for d in np.linspace(0, 1, 21)[::-1]:
        w_brakes.append(glider.equilibrium_glide(d, 0))
    w_brakes = np.asarray(w_brakes)

    w_speedbar = []
    for d in np.linspace(0, 1, 21):
        w_speedbar.append(glider.equilibrium_glide(0, d))
    w_speedbar = np.asarray(w_speedbar)

    eq = np.vstack([w_brakes, w_speedbar])

    gamma_eq = alpha_eq - eq[:, 0]

    Vx = eq[:, 1] * np.cos(gamma_eq)
    Vz = -eq[:, 1] * np.sin(gamma_eq)

    plt.plot(Vx, Vz)
    plt.show()


def main():
    airfoil_geo = Airfoil.NACA4(4412)  # For graphical purposes only

    # print("\nAirfoil: GNULAB3, simple flap, hinge at 80%")
    # airfoil_coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)

    print("\nAirfoil: NACA4412, simple flap, hinge at 80%")
    airfoil_coefs = Airfoil.GridCoefficients('polars/naca4412_xhinge80_yhinge_50.csv', 0.8)

    airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)

    sections = Parafoil.ConstantCoefficients(airfoil)

    # print("\nNACA4412 LinearCoefficients airfoil\n")
    # coefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)

    # print("\nPFD example LinearCoefficients airfoil\n")
    # coefs = Airfoil.LinearCoefficients(5.73, -2, 0.011, -0.05)

    parafoil_geo = build_elliptical_geo(
        # MAC=2.5, AR=3.9, taper=0.15, dMed=-20, dMax=-50,
        MAC=2.5, AR=3.9, taper=0.15, dMed=-25, dMax=-70,
        # MAC=2.5, AR=3.9, taper=0.15, dMed=-1, dMax=-2,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), sections=sections)

    # Same geometry, two different force estimation methods
    parafoil = Parafoil.Parafoil(parafoil_geo, sections)

    b = parafoil.geometry.b

    # brakes = BrakeGeometry.PFD(foil.geometry.b, .25, .025)  # FIXME: values?
    # brakes = BrakeGeometry.Exponential(b, .65, np.deg2rad(10))

    delta_max = np.deg2rad(50)*(1 - 0.8) * parafoil_geo.fc(b/2)
    bQuadratic = BrakeGeometry.Quadratic(b, delta_max)
    bCubic25 = BrakeGeometry.Cubic(b, 0.25, delta_max)
    bCubic45 = BrakeGeometry.Cubic(b, 0.45, delta_max)
    bCubic65 = BrakeGeometry.Cubic(b, 0.65, delta_max)
    # brakes = bQuadratic
    brakes = bCubic65

    # Build a wings with the different force estimation methods
    wing2d = ParagliderWing(parafoil, Parafoil.Phillips2D, brakes,
                            d_riser=0.5, z_riser=7,
                            kappa_s=0.4)
    wing3d = ParagliderWing(parafoil, Parafoil.Phillips, brakes,
                            d_riser=0.5, z_riser=7,
                            kappa_s=0.4)

    glider2d = Paraglider(wing2d, 75, 0.55, 0.75)
    glider3d = Paraglider(wing3d, 75, 0.55, 0.75)

    # ---------------------------------------------------------------------
    # Run some tests
    cp_y = wing2d.control_points(0)[:, 1]
    K = len(cp_y)
    V_rel = np.asarray([[10.0, 0.0, 1.0]] * K)
    V_rel[:, 0] += np.linspace(0, 1, K)**2 * 2  # spinning!

    #          Bl     Br
    # deltas = [0.00, 0.00]
    # deltas = [0.00, 0.25]
    deltas = [0.00, 0.50]
    # deltas = [0.00, 0.75]
    # deltas = [0.00, 1.00]
    # deltas = [1.00, 1.00]

    dF_2d, _ = wing2d.forces_and_moments(V_rel, *deltas)
    dF_3d, _ = wing3d.forces_and_moments(V_rel, *deltas)

    print("Plotting the forces")
    fig, ax = plt.subplots(3, sharex=True, figsize=(16, 10))
    ax[0].plot(cp_y, dF_2d[:, 0], label='2D')
    ax[0].plot(cp_y, dF_3d[:, 0], label='3D')
    ax[1].plot(cp_y, dF_2d[:, 1], label='2D')
    ax[1].plot(cp_y, dF_3d[:, 1], label='3D')
    ax[2].plot(cp_y, dF_2d[:, 2], label='2D')
    ax[2].plot(cp_y, dF_3d[:, 2], label='3D')
    ax[0].set_xlabel('spanwise position')
    ax[0].set_ylabel('Fx')
    ax[1].set_ylabel('Fy')
    ax[2].set_ylabel('Fz')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

    # ------------------------
    glider3d = Paraglider(wing3d, m_cg=70, S_cg=1, CD_cg=1)
    UVW = np.asarray([[10.0, 0.0, 1.0]] * K)
    R = 0
    # R = np.deg2rad(15)  # yaw rate = 15 degrees/sec clockwise
    PQR = np.array([0, 0, R])
    # sec_wind = glider3d.section_wind(None, UVW, PQR)

    xyz = glider3d.control_points(delta_s=0)
    dF, dM = glider3d.forces_and_moments(UVW, PQR, delta_Bl=0, delta_Br=0,
                                         xyz=xyz)
    embed()

    print("Plotting the forces")
    fig, ax = plt.subplots(3, sharex=True, figsize=(16, 10))
    ax[0].plot(xyz[:, 1], dF[:, 0])
    ax[1].plot(xyz[:, 1], dF[:, 1])
    ax[2].plot(xyz[:, 1], dF[:, 2])
    ax[0].set_xlabel('spanwise position')
    ax[0].set_ylabel('Fx')
    ax[1].set_ylabel('Fy')
    ax[2].set_ylabel('Fz')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.show()

    embed()


if __name__ == "__main__":
    main()
