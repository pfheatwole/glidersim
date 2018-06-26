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


def build_elliptical(MAC, AR, taper, dMed, sMed, dMax=None, sMax=None,
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

    # FIXME: this naming is confusing. Ellipticalg is a geometry, not a wing
    foil_geo = Elliptical(b, c0, taper, dMed, dMax, sMed, sMax, torsion)

    return Parafoil.Parafoil(foil_geo, sections)


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

    parafoil = build_elliptical(
        # MAC=2.5, AR=3.9, taper=0.15, dMed=-20, dMax=-50,
        MAC=2.5, AR=3.9, taper=0.15, dMed=-25, dMax=-70,
        # MAC=2.5, AR=3.9, taper=0.15, dMed=-1, dMax=-2,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), sections=sections)

    b = parafoil.geometry.b

    # brakes = BrakeGeometry.PFD(foil.geometry.b, .25, .025)  # FIXME: values?
    # brakes = BrakeGeometry.Exponential(b, .65, np.deg2rad(10))

    delta_max = np.deg2rad(50)*(1 - 0.8) * parafoil.geometry.fc(b/2)
    bQuadratic = BrakeGeometry.Quadratic(b, delta_max)
    bCubic25 = BrakeGeometry.Cubic(b, 0.25, delta_max)
    bCubic45 = BrakeGeometry.Cubic(b, 0.45, delta_max)
    bCubic65 = BrakeGeometry.Cubic(b, 0.65, delta_max)
    # brakes = bQuadratic
    brakes = bCubic65

    wing = ParagliderWing(parafoil, d_cg=0.5, h_cg=7, kappa_S=0.4)
    glider = Paraglider(wing, 75, 0.55, 0.75)

    print("\nFinished building the wing\n")

    # print("entering Anderson2")
    # anders = Parafoil.Anderson2(parafoil, brakes)
    # cl, cdi = anders._compute_section_coefs(0.123, 0)

    print("entering Phillips")
    phillips = Parafoil.Phillips(parafoil)

    print("entering Phillips2D")
    phillips2d = Parafoil.Phillips2D(parafoil)


    print("Testing V_inf = [10, 0, 1]")
    cp_y = phillips.cps[:, 0]
    K = len(cp_y)
    V_inf = np.asarray([[10.0, 0.0, 1.0]] * K)
    V_inf[:, 0] += np.linspace(0, 1, K)**2 * 2  # spinning!
    # delta = brakes(cp_y, 0, 0.0)
    # delta = brakes(cp_y, 0, 0.25)
    # delta = brakes(cp_y, 0, 0.5)
    delta = brakes(cp_y, 0, 1.0)

    # Gamma_2d = phillips2d.section_forces(V_inf, 0, 1)
    # Gamma_3d = phillips.find_vortex_strengths(V_inf, 0, 1)

    dF_2d, _ = phillips2d.forces_and_moments(V_inf, delta)
    dF_3d, _ = phillips.forces_and_moments(V_inf, delta)

    embed()


    # plot_coefficients(coefs)
    # embed()

    # input("\ncontinue with NACA?")
    # print("\nNACA4412 wing\n")
    # nacacoefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)
    # nacawing = build_elliptical(
    #     MAC=2.5, AR=4, taper=0.05, dMed=-30, dMax=-70,
    #     sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=nacacoefs)

    # embed()


if __name__ == "__main__":
    main()
