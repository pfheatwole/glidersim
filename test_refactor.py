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
                     torsion=0, airfoil_geo=None, coefs=None):
    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print("Using minimum max dihedral ({})".format(dMax))

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print("Using minimum max sweep ({})".format(sMax))

    # Compute some missing data in reverse
    c0 = Elliptical.MAC_to_c0(MAC, taper)
    b = Elliptical.AR_to_b(c0, AR, taper)

    # FIXME: this naming is confusing. Ellipticalg is a geometry, not a wing
    foil_geo = Elliptical(b, c0, taper, dMed, dMax, sMed, sMax, torsion)

    if airfoil_geo is None:
        airfoil_geo = Airfoil.NACA4(2415)

    if coefs is None:
        # coefs = LinearCoefficients(5.73, -5, 0.007, -0.05)  # a0, i0, D0, Cm0
        print("Using default airfoil coefficients")
        coefs = Airfoil.LinearCoefficients(5.80, -4, 0.008, -0.1)

    return Parafoil.Parafoil(foil_geo, Airfoil.Airfoil(coefs, airfoil_geo))


def plot_coefficients(coefs):
    CLs = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CLs.append(coefs.CL(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CLs):
        ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
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
        ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CD')
    plt.show()

    CMs = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CMs.append(coefs.CM(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CMs):
        ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
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
    print("\nGNULAB3 airfoil\n")
    coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)

    # print("\nNACA4412 LinearCoefficients airfoil\n")
    # coefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)

    # print("\nPFD example LinearCoefficients airfoil\n")
    # coefs = Airfoil.LinearCoefficients(5.73, -2, 0.011, -0.05)

    parafoil = build_elliptical(
        MAC=2.5, AR=3.9, taper=0.15, dMed=-25, dMax=-70,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=coefs)

    b = parafoil.geometry.b
    y = np.linspace(-b/2, b/2, 501)

    # brakes = BrakeGeometry.PFD(foil.geometry.b, .25, .025)  # FIXME: values?
    brakes = BrakeGeometry.Exponential(b, .65, np.deg2rad(10))
    parafoil_coefs = Parafoil.Coefs2D(parafoil, brakes)
    parafoil_coefs2 = Parafoil.CoefsMine(parafoil, brakes)
    parafoil_coefs3 = Parafoil.CoefsPFD(parafoil, brakes)
    parafoil_coefs4 = Parafoil.Coefs2(parafoil, brakes)
    coefs = parafoil_coefs4

    if isinstance(coefs, Parafoil.CoefsPFD):  # FIXME: HACK!
        coefs._pointwise_local_coefficients(.123, 0)

    wing = ParagliderWing(parafoil, coefs, d_cg=0.5, h_cg=7, kappa_S=0.4)
    glider = Paraglider(wing, 75, 0.55, 0.75)

    print("\nFinished building the wing\n")
    embed()

    plot_coefficients(coefs)
    embed()

    # input("\ncontinue with NACA?")
    # print("\nNACA4412 wing\n")
    # nacacoefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)
    # nacawing = build_elliptical(
    #     MAC=2.5, AR=4, taper=0.05, dMed=-30, dMax=-70,
    #     sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=nacacoefs)

    # embed()


if __name__ == "__main__":
    main()
