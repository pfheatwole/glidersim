"""FIXME: add module docstring."""

from IPython import embed

import numpy as np

from scipy.integrate import simps
from scipy.optimize import root_scalar

from util import cross3


class ParagliderWing:
    """FIXME: add class docstring."""

    # FIXME: review weight shift and speedbar designs. Why use percentage-based
    #        controls?

    def __init__(self, parafoil, force_estimator, brake_geo, d_riser, z_riser,
                 pA, pC, kappa_s=0):
        """
        FIXME: add docstring.

        Parameters
        ----------
        parafoil : Parafoil.ParafoilGeometry
            The geometric shape of the lifting surface.
        force_estimator : Parafoil.ForceEstimator
            The estimation method for the aerodynamic forces and moments.
        brake_Geo : BrakeGeometry
            Section trailing edge deflections as a function of delta_s
        d_riser : float [percentage]
            The longitudinal distance from the risers to the central leading
            edge, as a percentage of the chord length.
        z_riser : float [meters]
            The vertical distance from the risers to the central chord
        pA, pC : float [percentage]
            The position of the A and C lines as a fraction of the central
            chord. The speedbar adjusts the length of the A lines, while C
            remains fixed, causing a rotation about the point `pC`.
        kappa_s : float [meters], optional
            The speed bar line length. This corresponds to the maximum change
            in the length of the lines to the leading edge.
        """
        self.parafoil = parafoil
        self.force_estimator = force_estimator(parafoil)
        self.brake_geo = brake_geo
        self.pA = pA
        self.pC = pC
        self.kappa_s = kappa_s  # FIXME: strange notation. Why `kappa`?

        # The ParagliderWing coordinate system is a shifted version of the
        # one defined by the Parafoil. The axes of both systems are parallel,
        # but the origin moves from the central leading edge to the midpoint
        # of the risers.
        self.c0 = parafoil.chord_length(0)
        foil_x = d_riser * self.c0
        foil_z = -z_riser

        # Nominal lengths of the A and C lines, and their connection distance
        self.A = np.sqrt((foil_x - self.pA * self.c0) ** 2 + foil_z ** 2)
        self.C = np.sqrt((self.pC * self.c0 - foil_x) ** 2 + foil_z ** 2)
        self.AC = (self.pC - self.pA) * self.c0

        # FIXME: relocate. Here for testing, from a Hook 3
        self.rho_upper = 40 / 1000  # kg/m2
        self.rho_lower = 35 / 1000  # kg/m2

    def forces_and_moments(self, V_cp2w, delta_Bl, delta_Br, initial_Gamma=None):
        """
        FIXME: add docstring.

        Parameters
        ----------
        V_cp2w : array of float, shape (K,3) [m/s]
            The relative velocity of each control point vs the fluid
        delta_Bl : float [percentage]
            The amount of left brake
        delta_Br : float [percentage]
            The amount of right brake
        initial_Gamma : array of float, shape (K,) [m^2/s], optional
            An initial guess for the circulation distribution, to improve
            convergence

        Returns
        -------
        dF, dM : array of float, shape (K,3) [N, N m]
            Forces and moments for each section, proportional to the air
            density in [kg/m^3]. (This function assumes an air density of 1.)
        """
        delta = self.brake_geo(self.force_estimator.s_cps, delta_Bl, delta_Br)  # FIXME: leaky, don't grab s_cps directly
        dF, dM, Gamma = self.force_estimator(V_cp2w, delta, initial_Gamma)
        return dF, dM, Gamma

    def foil_origin(self, delta_s=0):
        """
        Compute the origin of the Parafoil coordinate system in FRD.

        Parameters
        ----------
        delta_s : float or array of float, shape (N,) [percentage] (optional)
            Fraction of maximum speed bar application

        Returns
        -------
        foil_origin : array of float, shape (3,) [meters]
            The offset of the origin of the Parafoil coordinate system in
            ParagliderWing coordinates.
        """
        # Speedbar shortens the A lines, while AC and C remain fixed
        A = self.A - (delta_s * self.kappa_s)
        foil_x = (A ** 2 - self.C ** 2 + self.AC ** 2) / (2 * self.AC)
        foil_y = 0  # FIXME: not with weight shift?
        foil_z = -np.sqrt(A ** 2 - foil_x ** 2)
        foil_x += self.pA * self.c0  # Account for the start of the AC line

        return np.array([foil_x, foil_y, foil_z])

    def equilibrium_alpha(self, deltaB, deltaS):
        """FIXME: add docstring."""
        def target(alpha, deltaB, deltaS):
            cp_wing = self.control_points(deltaS)
            v_wing = np.array([np.cos(alpha), 0, np.sin(alpha)])
            dF_w, dM_w, _ = self.forces_and_moments(v_wing, deltaB, deltaB)
            dM = dM_w.sum(axis=0)
            dM += cross3(cp_wing, dF_w).sum(axis=0)
            return dM[1]  # Pitching moment

        x0, x1 = np.deg2rad([0, 9])  # FIXME: review these bounds
        res = root_scalar(target, args=(deltaB, deltaS), x0=x0, x1=x1)
        if not res.converged:
            raise RuntimeError(f"Failed to converge: {res.flag}")
        return res.root

    def control_points(self, delta_s=0):
        """
        Compute the Parafoil control points in FRD.

        FIXME: descibe/define "control points"

        Parameters
        ----------
        delta_s : float or array of float, shape (N,) [percentage] (optional)
            Fraction of maximum speed bar application

        Returns
        -------
        cps : array of floats, shape (K,3) [meters]
            The control points in ParagliderWing coordinates
        """
        cps = self.force_estimator.control_points  # In Parafoil coordinates
        return cps + self.foil_origin(delta_s)  # In Wing coordinates

    # FIXME: moved from foil. Verify and test.
    def surface_distributions(self, delta_s=0):
        """
        Compute the surface area distributions that define the inertial moments.

        The moments of inertia for the parafoil are the mass distribution of
        the air and wing material. That distribution is typically decomposed
        into the product of volumetric density and volume, but a simplification
        is to calculate the density per unit area.

        FIXME: this description is mediocre.

        Ref: "Paraglider Flight Dynamics", page 48 (56)

        Returns
        -------
        S : 3x3 matrix of float
            The surface distributions, such that `J = (p_w + p_air)*S`
        """
        N = 501
        s = np.cos(np.linspace(np.pi, 0, N))  # -1 < s < 1
        x, y, z = (self.parafoil.chord_xyz(s, 0.25) + self.foil_origin(delta_s)).T
        c = self.parafoil.chord_length(s)

        Sxx = simps((y ** 2 + z ** 2) * c, y)
        Syy = simps((3 * x ** 2 - x * c + (7 / 32) * c ** 2 + 6 * z ** 2) * c, y) / 6
        Szz = simps((3 * x ** 2 - x * c + (7 / 32) * c ** 2 + 6 * y ** 2) * c, y) / 6
        Sxy = 0
        Sxz = simps((2 * x - c / 2) * z * c, y)
        Syz = 0

        S = np.array([
            [ Sxx, -Sxy, -Sxz],
            [-Sxy,  Syy, -Syz],
            [-Sxz, -Syz,  Szz]])

        return S

    # FIXME: moved from Parafoil. Verify and test.
    def J(self, rho_air, N=2000):
        raise NotImplementedError("BROKEN!")
        S = self.geometry.surface_distributions(N=N)
        wing_air_density = rho_air * self.density_factor
        surface_density = self.wing_density + wing_air_density
        return surface_density * S

    def inertia(self, rho_air, delta_s=0, N=200):
        """Compute the 3x3 moment of inertia matrix.

        TODO: this function currently only uses `delta_s` to determine the
              shift of the cg, but in the future `delta_s` may also deform the
              ParafoilLobe (if `lobe_args` starts getting used)

        TODO: precompute the static components that don't depend on `rho_air`

        Parameters
        ----------
        rho_air : float [kg/m^3]
            Volumetric air density of the atmosphere
        delta_s : float [percentage], optional
            Percentage of speed bar application
        N : integer
            The number of points for integration across the span

        Returns
        -------
        J : array of float, shape (3,3) [kg m^2]
            The inertia matrix of the wing

            ::

                    [[ Jxx -Jxy -Jxz]
                J =  [-Jxy  Jyy -Jyz]
                     [-Jxz -Jyz  Jzz]]

        """
        p = self.parafoil.mass_properties(N=N)

        # Storing this here for now: calculate the total mass and centroid
        # upper_mass = self.rho_upper * p['upper_area']
        # air_mass = rho_air * p['volume']
        # lower_mass = self.rho_lower * p['lower_area']
        # total_mass = upper_mass + air_mass + lower_mass
        # parafoil_cm = (upper_mass * p['upper_centroid'] +
        #                air_mass * p['volume_centroid'] +
        #                lower_mass * p['lower_centroid']) / total_mass

        o = -self.foil_origin(delta_s)  # Origin is `risers->origin`
        Ru = o - p["upper_centroid"]
        Rv = o - p["volume_centroid"]
        Rl = o - p["lower_centroid"]
        Du = (Ru @ Ru) * np.eye(3) - np.outer(Ru, Ru)
        Dv = (Rv @ Rv) * np.eye(3) - np.outer(Rv, Rv)
        Dl = (Rl @ Rl) * np.eye(3) - np.outer(Rl, Rl)

        J_wing = (self.rho_upper * (p["upper_inertia"] + p["upper_area"] * Du)
                  + rho_air * (p["volume_inertia"] + p["volume"] * Dv)
                  + self.rho_lower * (p["lower_inertia"] + p["lower_area"] * Dl))

        return J_wing
