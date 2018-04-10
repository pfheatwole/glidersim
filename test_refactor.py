from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from IPython import embed

import Airfoil
import Parafoil

import BrakeGeometry
from ParafoilGeometry import Elliptical
from ParagliderWing import ParagliderWing


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


def total_moment(wing, delta_B, alpha):
    CL, CD, Cm_c4 = wing.parafoil_coefs._pointwise_global_coefficients(
        alpha, delta_B)

    Cx = CL*np.sin(alpha) - CD*np.cos(alpha)
    Cz = -CL*np.cos(alpha) - CD*np.sin(alpha)  # FIXME: verify, esp orientation

    MAC = wing.parafoil.geometry.MAC
    c4 = wing.parafoil.geometry.fx(0)
    kMy = Cm_c4*MAC - Cx*wing.h_cg - Cz*(c4 - (-wing.d_cg))

    # print("\ninside total_moment")
    # embed()
    # input('continue?')

    return np.abs(kMy)


def main():
    print("\nGNULAB3 wing\n")
    coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)

    # print("\nNACA4412 wing\n")
    # coefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)

    # LinearCoefficients wing
    # coefs = Airfoil.LinearCoefficients(5.73, -2, 0.011, -0.05)

    foil = build_elliptical(
        MAC=2.5, AR=3.9, taper=0.15, dMed=-25, dMax=-70,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=coefs)

    b = foil.geometry.b
    y = np.linspace(-b/2, b/2, 501)

    # brakes = BrakeGeometry.PFD(foil.geometry.b, .25, .025)  # FIXME: values?
    brakes = BrakeGeometry.Exponential(foil.geometry.b, .65, np.deg2rad(10))
    parafoil_coefs = Parafoil.Coefs2D(foil, brakes)
    parafoil_coefs2 = Parafoil.CoefsMine(foil, brakes)
    parafoil_coefs3 = Parafoil.CoefsPFD(foil, brakes)
    coefs = parafoil_coefs

    if isinstance(coefs, Parafoil.CoefsPFD):  # FIXME: HACK!
        coefs._pointwise_local_coefficients(.123, 0)

    d_cg = 0.5*foil.geometry.fc(0)  # Place the cg at 50% central chord
    print("\nd_cg:", d_cg)

    wing = ParagliderWing(foil, coefs, d_cg=d_cg, h_cg=7, kappa_a=0)

    f = partial(total_moment, wing, 0)
    alpha_min, alpha_max = np.deg2rad(-1.5), np.deg2rad(20)
    r = minimize_scalar(f, bounds=(alpha_min, alpha_max), method='Bounded').x
    print("Equilibrium condition for zero brakes:")
    print("  alpha_eq   : {}".format(np.rad2deg(r)))
    print("  Glide ratio: {}".format(1/np.tan(r)))
    print()

    # print("Testing: wing.parafoil_coefs._pointwise_global_coefficients(alpha, delta_B)")
    embed()

    # print("brakes (deltas) not implemented yet, skipping")
    # return

    deltas = np.linspace(0, 1, 250)
    results = np.zeros_like(deltas)
    for n, delta in enumerate(deltas):
        f = partial(total_moment, wing, delta)
        alpha_min, alpha_max = np.deg2rad(-1.5), np.deg2rad(20)
        results[n] = minimize_scalar(f, bounds=(alpha_min, alpha_max),
                                     method='Bounded').x

    test_alpha = np.deg2rad(8)
    global_coefs = []
    for d in deltas:
        global_coefs.append(wing.parafoil_coefs._pointwise_global_coefficients(
            test_alpha, d))
    global_coefs = np.asarray(global_coefs)


    Vx = np.cos(results)
    Vz = np.sin(results)

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
