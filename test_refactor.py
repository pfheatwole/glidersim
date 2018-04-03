import matplotlib.pyplot as plt
import numpy as np

from IPython import embed

import Airfoil
import Parafoil

from ParafoilGeometry import Elliptical
from ParagliderWing import BrakeGeometry, ParagliderWing


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


if __name__ == "__main__":
    print("\nGNULAB3 wing\n")
    coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)
    foil = build_elliptical(
        MAC=2.5, AR=3.9, taper=0.15, dMed=-25, dMax=-70,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=coefs)

    b = foil.geometry.b
    y = np.linspace(-b/2, b/2, 501)

    brakes = BrakeGeometry(foil.geometry.b, .5, .1)  # FIXME: values?
    parafoil_coefs = Parafoil.Coefs2D(foil, brakes)

    wing = ParagliderWing(foil, parafoil_coefs, d_cg=0.2, h_cg=7, kappa_a=0)

    embed()

    # input("\ncontinue with NACA?")

    # print("\nNACA4412 wing\n")
    # nacacoefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)
    # nacawing = build_elliptical(
    #     MAC=2.5, AR=4, taper=0.05, dMed=-30, dMax=-70,
    #     sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=nacacoefs)

    # embed()
