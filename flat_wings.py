from IPython import embed

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.interpolate import UnivariateSpline

import airfoil
import foil
import plots


class FlaplessAirfoilCoefficients(airfoil.AirfoilCoefficients):
    """
    Use airfoil coefficients from a CSV file.

    The CSV must contain the following columns: [alpha, delta, CL, CD, Cm]

    This is similar to `Airfoil.GridCoefficients`, but it assumes that delta
    is always zero. This is convenient, since no assuptions need to be made
    for the non-existent flaps on the wind tunnel model.
    """

    def __init__(self, filename, convert_degrees=True):
        data = pd.read_csv(filename, skiprows=9)
        self.data = data

        if convert_degrees:
            data['alpha'] = np.deg2rad(data.alpha)

        self._Cl = UnivariateSpline(data[['alpha']], data.CL, s=0.001)
        self._Cd = UnivariateSpline(data[['alpha']], data.CD, s=0.0001)
        self._Cm = UnivariateSpline(data[['alpha']], data.Cm, s=0.0001)
        self._Cl_alpha = self._Cl.derivative()

    def _clean(self, alpha, val):
        # The UnivariateSpline doesn't fill `nan` outside the boundaries
        min_alpha, max_alpha = np.deg2rad(-9.9), np.deg2rad(24.9)
        mask = (alpha < min_alpha) | (alpha > max_alpha)
        val[mask] = np.nan
        return val

    def Cl(self, alpha, delta):
        return self._clean(alpha, self._Cl(alpha))

    def Cd(self, alpha, delta):
        return self._clean(alpha, self._Cd(alpha))

    def Cm(self, alpha, delta):
        return self._clean(alpha, self._Cm(alpha))

    def Cl_alpha(self, alpha, delta):
        return self._clean(alpha, self._Cl_alpha(alpha))


if __name__ == "__main__":

    # airfoil_geo = airfoil.NACA(24018, convention="british")
    # airfoil_coefs = airfoil.GridCoefficients('polars/exp_curving_24018.csv')  # delta_max = 13.38
    # delta_max = np.deg2rad(10.00)  # True max: 13.28

    airfoil_geo = airfoil.NACA(23015)
    # airfoil_coefs = FlaplessAirfoilCoefficients('polars/NACA 23015_T1_Re0.920_M0.03_N7.0_XtrTop 5%_XtrBot 5%.csv')
    airfoil_coefs = FlaplessAirfoilCoefficients('/home/peter/stupid_wing/T1_Re0.650_M0.03_N0.1_XtrTop 5%_XtrBot 5%.csv')

    airfoil = airfoil.Airfoil(coefficients=airfoil_coefs, geometry=airfoil_geo)

    # Straight
    wing1 = foil.FoilGeometry(
        airfoil=airfoil,
        chord_length=0.25,
        r_x=0,
        x=0,
        r_yz=0,
        yz=foil.FlatYZ(),
        b_flat=8,
    )
    M_ref1 = wing1.chord_xyz(0, 0)

    # Elliptical
    wing2 = foil.FoilGeometry(
        airfoil=airfoil,
        chord_length=foil.elliptical_chord(.25, .1),
        r_x=0.5,
        x=0,
        r_yz=0,
        yz=foil.FlatYZ(),
        b_flat=8,
    )
    M_ref2 = wing2.chord_xyz(0, 0.5)

    # Diagonal
    wing3 = foil.FoilGeometry(
        airfoil=airfoil,
        chord_length=0.5,
        r_x=0.5,
        x=lambda s: -np.abs(s),
        r_yz=0,
        yz=foil.FlatYZ(),
        b_flat=1,
    )
    M_ref3 = wing3.chord_xyz(0, 0.0)

    # Triangle
    wing4 = foil.FoilGeometry(
        airfoil=airfoil,
        chord_length=lambda s: 1 - np.abs(s),
        r_x=1.0,
        x=0,
        r_yz=0,
        yz=foil.FlatYZ(),
        b_flat=1,
    )
    M_ref4 = wing4.chord_xyz(0, 0.0)

    # Diamond
    wing5 = foil.FoilGeometry(
        airfoil=airfoil,
        chord_length=lambda s: 1 - np.abs(s),
        r_x=0.5,
        x=0,
        r_yz=0,
        yz=foil.FlatYZ(),
        b_flat=1,
    )
    M_ref5 = wing5.chord_xyz(0, 0.0)

    # wing, M_ref = wing1, M_ref1
    wing, M_ref = wing2, M_ref2
    # wing, M_ref = wing3, M_ref3
    # wing, M_ref = wing4, M_ref4
    # wing, M_ref = wing5, M_ref5

    plots.plot_parafoil_geo(wing, N_sections=51)

    # For a flat wing:
    #
    #   CD = CD0 + (CL**2) / (np.pi * e_0 * AR)
    #
    # Where `e_0` is the Oswald efficiency number.

    alpha = np.deg2rad(6)
    beta = np.deg2rad(5)
    V_total = 10
    UVW = V_total * np.asarray(
        [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
    )
    rho_air = 1.225

    fe = foil.Phillips(wing, alpha_ref=5)
    dF, dM, solution = fe(UVW, 0)
    F = rho_air * dF.sum(axis=0)
    M = rho_air * dM.sum(axis=0)

    M += np.cross(fe.cps - M_ref, dF).sum(axis=0)

    S = wing.S

    CX, CY, CZ = F / (.5 * rho_air * V_total ** 2 * S)
    CN = -CZ
    CM = M.T[1] / (0.5 * rho_air * V_total ** 2 * S * wing.chord_length(0))

    # From Stevens, "Aircraft Control and Simulation", pg 90 (104)
    CD = (
        -np.cos(alpha) * np.cos(beta) * CX
        - np.sin(beta) * CY
        + np.sin(alpha) * np.cos(beta) * CN
    )
    CL = np.sin(alpha) * CX + np.cos(alpha) * CN

    print()
    print(f"Force coefficients: CL={CL:.3f}, CD={CD:.3f}")
    print(
        "Moment coefficients:",
        (M / (0.5 * rho_air * V_total ** 2 * S * wing.chord_length(0))).round(3),
    )
    print()

    # FIXME: broken
    # CD0 = airfoil_coefs.Cd(alpha, 0)
    # e_0 = CL**2 / ((CD - CD0) * wing.AR * np.pi)
    # print("Oswald efficiency:", e_0)

    # plt.plot(solution['Gamma'])
    # plt.show()

    # Check the resulting section alphas
    u_inf = -np.array([np.cos(alpha), 0, np.sin(alpha)])
    v = fe._induced_velocities(u_inf)
    V, V_n, V_a, alphas = fe._local_velocities(-UVW, solution['Gamma'], v)

    embed()
