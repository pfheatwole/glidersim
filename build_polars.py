from functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`
import numpy as np
from numpy import sin, cos, sqrt, arcsin, arctan  # noqa: F401
from scipy.optimize import least_squares

# from matplotlib import animation
# import mpl_toolkits.mplot3d.axes3d as p3

from IPython import embed  # noqa: F401

from Airfoil import Airfoil, LinearCoefficients, NACA4
from Wing import Wing, EllipticalWing


def build_elliptical(MAC, AR, taper, dMed, sMed, dMax=None, sMax=None,
                     torsion=0, airfoil_geo=None, coefs=None, dcg=None):
    if dMax is None:
        print("Using minimum max dihedral")
        dMax = 2*dMed - 1  # ref page 48 (56)

    if sMax is None:
        print("Using minimum max sweep")
        sMax = (2*sMed) + 1  # ref page 48 (56)

    # Compute some missing data in reverse
    c0 = EllipticalWing.MAC_to_c0(MAC, taper)
    b = EllipticalWing.AR_to_b(c0, AR, taper)

    if dcg is None:
        dcg = 0.65  # FIXME: unlisted
    h0 = 7  # FIXME: unlisted
    wing_geo = EllipticalWing(
        dcg, c0, h0, dMed, dMax, b, taper, sMed, sMax, torsion=torsion)

    if airfoil_geo is None:
        airfoil_geo = NACA4(2415)

    if coefs is None:
        # coefs = LinearCoefficients(5.73, -5, 0.007, -0.05)  # a0, i0, D0, Cm0
        coefs = LinearCoefficients(5.80, -4, 0.008, -0.1)  # a0, i0, D0, Cm0

    return Wing(wing_geo, Airfoil(coefs, airfoil_geo))


def equilibrium_parameters(wing, delta):
    """Compute alpha_eq, d0, h0

    Parameters
    ----------
    wing : Wing
        The glider system that defines the wing geometry and airfoil
    delta : float
        The symmetric brake actuation, where `0 <= delta <= 1`

    Returns
    -------
    alpha_eq : float
        The equilibrium AOA for the given symmetric brakes actuation
    d0 : float
        The x-axis distance of the cg to the global AC
    h0 : float
        The z-axis distance of the cg to the global AC
    """

    # The integration points across the span
    N = 1000
    dy = wing.geometry.b/N
    y = np.linspace(-wing.geometry.b/2, wing.geometry.b/2, N,
                    endpoint=False) + dy/2
    deltas = wing.geometry.delta(y)
    thetas = wing.geometry.ftheta(y)

    def calc_d0h0(wing, alpha_eq):
        """Calculate the global AC, {d0, h0}

        These points are deterministic given alpha_eq, but are used as part
        of the optimization routine to find alpha_eq.

        ref: PFD Eqs 5.44-5.45
        """

        CL = wing.CL(alpha_eq)
        CD = wing.CD(alpha_eq)

        Cli = wing.Cl(alpha_eq)
        Cdi = wing.Cd(alpha_eq)
        alpha_i = alpha_eq*cos(deltas) + thetas

        # PFD eq 5.44
        d0 = ((Cli*cos(alpha_i) + Cdi*sin(alpha_i)) *
              wing.geometry.fx(y) * wing.geometry.fc(y) * dy).sum() / \
            ((CL*cos(alpha_eq) + CD*sin(alpha_eq)) * wing.geometry.S)

        # PFD eq 5.45
        h0 = -((Cli*sin(alpha_i) - Cdi*cos(alpha_i)) *
               wing.geometry.fz(y) * wing.geometry.fc(y) /
               cos(deltas)*dy).sum() / \
              ((CL*sin(alpha_eq) - CD*cos(alpha_eq)) * wing.geometry.S)

        print("DEBUG> d0: {}, h0: {}".format(d0, h0))

        return d0, h0

    # def calc_my(wing, d0_prime, h0_prime, alpha):
    def calc_my(wing, alpha, d0=None, h0=None):
        """Optimization target for computing alpha_eq

        Parameters
        ----------
        wing : Wing
        alpha : float
            Current guess for alpha_eq

        Returns
        -------
        My : float
            The total moment about the y-axis. Should be zero for equilibrium.
        """
        print("DEBUG> calc_my: alpha:", alpha)

        # Update {d0, h0} to track the changing alpha
        if d0 is None:
            d0, h0 = calc_d0h0(wing, alpha)

        CL = wing.CL(alpha)
        CD = wing.CD(alpha)
        Cm = wing.Cm(alpha)
        Cz = CL*cos(alpha) + CD*sin(alpha)  # PFD eq 4.76
        Cx = CL*sin(alpha) - CD*cos(alpha)  # PFD eq 4.77
        My = Cz*d0 - Cx*h0 + Cm*wing.geometry.MAC  # PFD Eq 4.78/5.37
        print("DEBUG> My: {} (alpha: {}, d0: {}, h0: {}, h0/d0: {})".format(
            My, alpha, d0, h0, h0/d0))
        return My

    def calc_my_TEST(wing, d0, h0, alpha):
        CL = wing.CL(alpha)
        CD = wing.CD(alpha)
        Cm = wing.Cm(alpha)
        Cz = CL*cos(alpha) + CD*sin(alpha)  # PFD eq 4.76
        Cx = CL*sin(alpha) - CD*cos(alpha)  # PFD eq 4.77
        My = Cz*d0 - Cx*h0 + Cm*wing.geometry.MAC  # PFD Eq 4.78/5.37
        return My

    alphas = np.linspace(-1.99, 24, 50)
    d0h0s = np.asarray([calc_d0h0(wing, np.deg2rad(a)) for a in alphas])
    Mys = np.asarray([calc_my(wing, np.deg2rad(a)) for a in alphas])
    print("d0h0s")
    input("Continue?")
    embed()

    # FIXME: Initialize alpha_eq to something reasonable
    f_alpha = partial(calc_my, wing)
    alpha_eq_prime = least_squares(f_alpha, np.deg2rad(8)).x[0]
    # alpha_eq_prime = least_squares(
    #     f_alpha, np.deg2rad(2),
    #   bounds=(np.deg2rad(1.75), np.deg2rad(2.3))).x[0]
    #   bounds=(0, np.deg2rad(15))).x[0]

    print("Finished finding alpha_eq_prime:", alpha_eq_prime)
    input("Continue?")
    embed()


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

    D21 = (wing1.CD(0.05) - wing1.CD(wing1.i0)) / (a1*(.05-wing1.i0))**2
    D22 = (wing2.CD(0.05) - wing2.CD(wing2.i0)) / (a2*(.05-wing2.i0))**2
    D23 = (wing3.CD(0.05) - wing3.CD(wing3.i0)) / (a3*(.05-wing3.i0))**2
    D24 = (wing4.CD(0.05) - wing4.CD(wing4.i0)) / (a4*(.05-wing4.i0))**2
    D25 = (wing5.CD(0.05) - wing5.CD(wing5.i0)) / (a5*(.05-wing5.i0))**2
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
    # input("Continue?")

    print("\nChecking equilibrium for hands-up")
    equilibrium_parameters(wing, 0)

    input("Continue?")
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
