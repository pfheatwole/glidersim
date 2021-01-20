import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.interpolate import UnivariateSpline

import pfh.glidersim as gsim


class FlaplessAirfoilCoefficients(gsim.airfoil.AirfoilCoefficients):
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
            data["alpha"] = np.deg2rad(data.alpha)

        self._Cl = UnivariateSpline(data[["alpha"]], data.CL, s=0.001)
        self._Cd = UnivariateSpline(data[["alpha"]], data.CD, s=0.0001)
        self._Cm = UnivariateSpline(data[["alpha"]], data.Cm, s=0.0001)
        self._Cl_alpha = self._Cl.derivative()

    def _clean(self, alpha, val):
        # The UnivariateSpline doesn't fill `nan` outside the boundaries
        min_alpha, max_alpha = np.deg2rad(-9.9), np.deg2rad(24.9)
        mask = (alpha < min_alpha) | (alpha > max_alpha)
        val[mask] = np.nan
        return val

    def Cl(self, delta, alpha, Re):
        return self._clean(alpha, self._Cl(alpha))

    def Cd(self, delta, alpha, Re):
        return self._clean(alpha, self._Cd(alpha))

    def Cm(self, delta, alpha, Re):
        return self._clean(alpha, self._Cm(alpha))

    def Cl_alpha(self, delta, alpha, Re):
        return self._clean(alpha, self._Cl_alpha(alpha))


if __name__ == "__main__":

    # airfoil_geo = gsim.airfoil.NACA(24018, convention="vertical")
    # airfoil_coefs = gsim.airfoil.GridCoefficients("polars/exp_curving_24018.csv")

    airfoil_geo = gsim.airfoil.NACA(23015)
    airfoil_coefs = FlaplessAirfoilCoefficients(
        "/home/peter/Documents/Projects/Masters/Paraglider Model/work/xflr5_flat_elliptical_wing/T1_Re0.650_M0.03_N0.1_XtrTop 5%_XtrBot 5%.csv",
    )

    airfoil = gsim.airfoil.Airfoil(coefficients=airfoil_coefs, geometry=airfoil_geo)
    sections = gsim.foil.FoilSections(airfoil=airfoil)

    # Straight
    layout1 = gsim.foil.SectionLayout(
        c=0.25,
        r_x=0,
        x=0,
        r_yz=0,
        yz=gsim.foil.FlatYZ(),
    )
    wing1 = gsim.foil.SimpleFoil(
        layout=layout1,
        sections=sections,
        b_flat=8,
    )
    M_ref1 = wing1.surface_xyz(0, 0, surface="chord")

    # Elliptical
    layout2 = gsim.foil.SectionLayout(
        c=gsim.foil.elliptical_chord(0.25, 0.1),
        r_x=0.5,
        x=0,
        r_yz=0,
        yz=gsim.foil.FlatYZ(),
    )
    wing2 = gsim.foil.SimpleFoil(
        layout=layout2,
        sections=sections,
        b_flat=8,
    )
    M_ref2 = wing2.surface_xyz(0, 0.5, surface="chord")

    # Diagonal
    layout3 = gsim.foil.SectionLayout(
        c=0.5,
        r_x=0.5,
        x=lambda s: -np.abs(s),
        r_yz=0,
        yz=gsim.foil.FlatYZ(),
    )
    wing3 = gsim.foil.SimpleFoil(
        layout=layout3,
        sections=sections,
        b_flat=1,
    )
    M_ref3 = wing3.surface_xyz(0, 0.0, surface="chord")

    # Triangle
    layout4 = gsim.foil.SectionLayout(
        c=lambda s: 1 - np.abs(s),
        r_x=1.0,
        x=0,
        r_yz=0,
        yz=gsim.foil.FlatYZ(),
    )
    wing4 = gsim.foil.SimpleFoil(
        layout=layout4,
        sections=sections,
        b_flat=1,
    )
    M_ref4 = wing4.surface_xyz(0, 0.0, surface="chord")

    # Diamond
    layout5 = gsim.foil.SectionLayout(
        c=lambda s: 1 - np.abs(s),
        r_x=0.5,
        x=0,
        r_yz=0,
        yz=gsim.foil.FlatYZ(),
    )
    wing5 = gsim.foil.SimpleFoil(
        layout=layout5,
        sections=sections,
        b_flat=1,
    )
    M_ref5 = wing5.surface_xyz(0, 0.0, surface="chord")

    wing, M_ref = wing1, M_ref1
    # wing, M_ref = wing2, M_ref2
    # wing, M_ref = wing3, M_ref3
    # wing, M_ref = wing4, M_ref4
    # wing, M_ref = wing5, M_ref5

    gsim.extras.plots.plot_foil(wing, N_sections=51)

    # For a flat wing:
    #
    #   CD = CD0 + (CL**2) / (np.pi * e_0 * AR)
    #
    # Where `e_0` is the Oswald efficiency number.

    alpha = np.deg2rad(6)
    beta = np.deg2rad(5)
    v_mag = 10
    v_cp2w = np.asarray(
        [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
    )
    v_cp2w *= v_mag  # The Reynolds numbers are a function of the magnitude
    rho_air = 1.225

    fe = gsim.foil_aerodynamics.Phillips(wing, v_ref_mag=v_mag, alpha_ref=5)
    dF, dM, solution = fe(0, -v_cp2w, 1.2)
    F = rho_air * dF.sum(axis=0)
    M = rho_air * dM.sum(axis=0)

    M += np.cross(fe.cps - M_ref, dF).sum(axis=0)

    S = wing.S

    CX, CY, CZ = F / (0.5 * rho_air * v_mag ** 2 * S)
    CN = -CZ
    CM = M.T[1] / (0.5 * rho_air * v_mag ** 2 * S * wing.chord_length(0))

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
        (M / (0.5 * rho_air * v_mag ** 2 * S * wing.chord_length(0))).round(3),
    )
    print()

    # FIXME: broken
    # CD0 = airfoil_coefs.Cd(alpha, 0)
    # e_0 = CL**2 / ((CD - CD0) * wing.AR * np.pi)
    # print("Oswald efficiency:", e_0)

    # plt.plot(solution["Gamma"])
    # plt.show()

    # Check the resulting section alphas
    u_inf = -np.array([np.cos(alpha), 0, np.sin(alpha)])
    v = fe._induced_velocities(u_inf)
    V, V_n, V_a, alphas = fe._local_velocities(-v_cp2w, solution["Gamma"], v)

    breakpoint()
