import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`
import numpy as np
from numpy import sin, cos, sqrt, arcsin, arctan  # noqa: F401

# from matplotlib import animation
# import mpl_toolkits.mplot3d.axes3d as p3

from IPython import embed  # noqa: F401

from Airfoil import Airfoil, LinearCoefficients, NACA4
from Wing import Wing, EllipticalWing
from Glider import Glider


def build_elliptical(MAC, AR, taper, dMed, sMed, dMax=None, sMax=None,
                     torsion=0, airfoil_geo=None, coefs=None):
    if dMax is None:
        print("Using minimum max dihedral")
        dMax = 2*dMed - 1  # ref page 48 (56)

    if sMax is None:
        print("Using minimum max sweep")
        sMax = (2*sMed) + 1  # ref page 48 (56)

    # Compute some missing data in reverse
    c0 = EllipticalWing.MAC_to_c0(MAC, taper)
    b = EllipticalWing.AR_to_b(c0, AR, taper)

    # FIXME: this naming is confusing. EllipticalWing is a geometry, not a wing
    wing_geo = EllipticalWing(b, c0, taper, dMed, dMax, sMed, sMax, torsion)

    if airfoil_geo is None:
        airfoil_geo = NACA4(2415)

    if coefs is None:
        # coefs = LinearCoefficients(5.73, -5, 0.007, -0.05)  # a0, i0, D0, Cm0
        coefs = LinearCoefficients(5.80, -4, 0.008, -0.1)  # a0, i0, D0, Cm0

    return Wing(wing_geo, Airfoil(coefs, airfoil_geo))


def find_first(arr, val):
    """Find index of the first occurence of `val` in the list-like `arr"""
    for n, v in enumerate(arr):
        if v == val:
            return n
    return None


if __name__ == "__main__":
    # Trying to recreate the graphs on PFD pg 72 (80)
    airfoil_geo = NACA4(4412)

    # FIXME: these parameters are largely unknown?
    coefs = LinearCoefficients(5.73, -2, 0.007, -0.05)
    wing = build_elliptical(
        MAC=2.3, AR=4.0, taper=0.4, dMed=-30, sMed=10, coefs=coefs)

    # from PFD p89 (97)
    coefs = LinearCoefficients(5.73, -2, 0.007, -0.05)
    wing1 = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=10, coefs=coefs)
    wing2 = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-35, sMed=10, coefs=coefs)
    wing3 = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=25, coefs=coefs)
    wing4 = build_elliptical(
        MAC=2.2, AR=4.9, taper=0.4, dMed=-20, sMed=10, coefs=coefs)
    wing5 = build_elliptical(
        MAC=2.3, AR=4.0, taper=0.6, dMed=-20, sMed=10, coefs=coefs)

    # Plot the section coefficients
    # plt.plot(ys, F_par_x/dy, 'k', label='X')
    # plt.plot(ys, F_par_y/dy, 'g', label='Y')
    # plt.plot(ys, F_par_z/dy, 'b', label='Z')
    # plt.title('Force per unit span')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    a1 = wing1.CL(.05)/(.05-wing1.i0)
    a2 = wing2.CL(.05)/(.05-wing2.i0)
    a3 = wing3.CL(.05)/(.05-wing3.i0)
    a4 = wing4.CL(.05)/(.05-wing4.i0)
    a5 = wing5.CL(.05)/(.05-wing5.i0)
    print("wing1.a:", a1)
    print("wing1.a:", a2)
    print("wing1.a:", a3)
    print("wing1.a:", a4)
    print("wing1.a:", a5)

    D01 = wing1.CD(wing1.i0)
    D02 = wing2.CD(wing2.i0)
    D03 = wing3.CD(wing3.i0)
    D04 = wing4.CD(wing4.i0)
    D05 = wing5.CD(wing5.i0)
    print("wing1.D0:", D01)
    print("wing2.D0:", D02)
    print("wing3.D0:", D03)
    print("wing4.D0:", D04)
    print("wing5.D0:", D05)
    print()

    D21 = (wing1.CD(0.05) - D01) / wing1.CL(0.05)**2
    D22 = (wing2.CD(0.05) - D02) / wing2.CL(0.05)**2
    D23 = (wing3.CD(0.05) - D03) / wing3.CL(0.05)**2
    D24 = (wing4.CD(0.05) - D04) / wing4.CL(0.05)**2
    D25 = (wing5.CD(0.05) - D05) / wing5.CL(0.05)**2
    print("wing1.D2:", D21)
    print("wing2.D2:", D22)
    print("wing3.D2:", D23)
    print("wing4.D2:", D24)
    print("wing5.D2:", D25)
    print()

    alphas = np.linspace(-1.99, 25, 1000)
    alphas_r = np.deg2rad(alphas)
    LD1 = wing1.CL(alphas_r)/wing1.CD(alphas_r)

    input("Continue?")
    embed()




    #Glider(wing, d_cg, h_cg, S_cg, Cd_cg):
    glider1 = Glider(wing1, d_cg=0.35, h_cg=7, S_cg=1, Cd_cg=0.8)
    glider2 = Glider(wing2, d_cg=0.35, h_cg=7, S_cg=1, Cd_cg=0.8)
    glider3 = Glider(wing3, d_cg=0.35, h_cg=7, S_cg=1, Cd_cg=0.8)
    glider4 = Glider(wing4, d_cg=0.35, h_cg=7, S_cg=1, Cd_cg=0.8)
    glider5 = Glider(wing5, d_cg=0.35, h_cg=7, S_cg=1, Cd_cg=0.8)


    print("\nChecking equilibrium for hands-up")
    glider1.equilibrium_parameters(0)

    input("Continue?")
    embed()

    print("\nSome plots")

    alphas = np.linspace(-1.99, 25, 1000)
    alphas_r = np.deg2rad(alphas)

    plt.plot(alphas, wing.airfoil.coefficients.Cl(alphas_r),
             'r--', lw=0.8, label='Central CL')
    plt.plot(alphas, wing.CL(alphas_r), label='Global CL')
    plt.legend()
    plt.show()

    plt.plot(alphas, wing.CD(alphas_r), label='Global CD')
    plt.plot(alphas, wing.airfoil.coefficients.Cd(np.deg2rad(alphas)),
             'r--', lw=0.8, label='Central CD')
    plt.legend()
    plt.show()

    plt.plot(wing.CL(alphas_r), wing.CD(alphas_r))
    plt.title('Cd vs Cl')
    plt.show()

    plt.plot(alphas, wing.CL(alphas_r)/wing.CD(alphas_r))
    plt.title('L/D vs Alpha')
    plt.show()

    input("Continue?")
    embed()