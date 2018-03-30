import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`
import numpy as np
from numpy import sin, cos, sqrt, arcsin, arctan  # noqa: F401

# from matplotlib import animation
# import mpl_toolkits.mplot3d.axes3d as p3

from IPython import embed  # noqa: F401

import Airfoil

from Wing import Wing
from WingGeometry import EllipticalWing
from Glider import Glider
from Glider import trapz


def build_elliptical(MAC, AR, taper, dMed, sMed, dMax=None, sMax=None,
                     torsion=0, airfoil_geo=None, coefs=None):
    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print("Using minimum max dihedral ({})".format(dMax))

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print("Using minimum max sweep ({})".format(sMax))

    # Compute some missing data in reverse
    c0 = EllipticalWing.MAC_to_c0(MAC, taper)
    b = EllipticalWing.AR_to_b(c0, AR, taper)

    # FIXME: this naming is confusing. EllipticalWing is a geometry, not a wing
    wing_geo = EllipticalWing(b, c0, taper, dMed, dMax, sMed, sMax, torsion)

    if airfoil_geo is None:
        airfoil_geo = Airfoil.NACA4(2415)

    if coefs is None:
        # coefs = LinearCoefficients(5.73, -5, 0.007, -0.05)  # a0, i0, D0, Cm0
        print("Using default airfoil coefficients")
        coefs = Airfoil.LinearCoefficients(5.80, -4, 0.008, -0.1)

    return Wing(wing_geo, Airfoil.Airfoil(coefs, airfoil_geo))


if __name__ == "__main__":
    print("\nGNULAB3 wing\n")
    coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)
    wing = build_elliptical(
        MAC=2.5, AR=3.9, taper=0.15, dMed=-25, dMax=-70,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=coefs)

    embed()
    input("\ncontinue with NACA?")

    print("\nNACA4412 wing\n")
    nacacoefs = Airfoil.LinearCoefficients(5.73, -2, 0.007, -0.05)
    nacawing = build_elliptical(
        MAC=2.5, AR=4, taper=0.05, dMed=-30, dMax=-70,
        sMed=5, airfoil_geo=Airfoil.NACA4(4412), coefs=nacacoefs)


    # embed()
