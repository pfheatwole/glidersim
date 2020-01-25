"""FIXME: add module docstring."""

import numpy as np

from scipy.integrate import simps
from scipy.optimize import root_scalar

from pfh.glidersim import foil
from pfh.glidersim.util import cross3


class ParagliderWing:
    """FIXME: add class docstring."""

    # FIXME: review weight shift and speedbar designs. Why use percentage-based
    #        controls?

    def __init__(self, parafoil, force_estimator, brake_geo, d_riser, z_riser,
                 pA, pC, kappa_a, rho_upper, rho_lower):
        """
        FIXME: add docstring.

        Parameters
        ----------
        parafoil : foil.FoilGeometry
            The geometric shape of the lifting surface.
        force_estimator : foil.ForceEstimator
            The estimation method for the aerodynamic forces and moments.
        brake_geo : BrakeGeometry
            Section trailing edge deflections as a function of delta_Bl/Br
        d_riser : float [percentage]
            The longitudinal distance from the risers to the central leading
            edge, as a percentage of the chord length.
        z_riser : float [meters]
            The vertical distance from the risers to the central chord
        pA, pC : float [percentage]
            The position of the A and C lines as a fraction of the central
            chord. The speedbar adjusts the length of the A lines, while C
            remains fixed, causing a rotation about the point `pC`.
        kappa_a : float [meters], optional
            The speed bar line length. This corresponds to the maximum change
            in the length of the lines to the leading edge.
        rho_upper, rho_lower : float [kg m^-2]
            Surface area densities of the upper and lower foil surfaces.
        """
        self.parafoil = parafoil
        self.force_estimator = force_estimator(parafoil)
        self.brake_geo = brake_geo
        self.pA = pA
        self.pC = pC
        self.kappa_a = kappa_a  # FIXME: strange notation. Why `kappa`?
        self.rho_upper = rho_upper
        self.rho_lower = rho_lower

        # The ParagliderWing coordinate system is a shifted version of the
        # one defined by the Parafoil. The axes of both systems are parallel,
        # but the origin moves from the central leading edge to the midpoint
        # of the risers.
        self.c0 = parafoil.chords.length(0)
        foil_x = d_riser * self.c0
        foil_z = -z_riser

        # Nominal lengths of the A and C lines, and their connection distance
        self.A = np.sqrt((foil_x - self.pA * self.c0) ** 2 + foil_z ** 2)
        self.C = np.sqrt((self.pC * self.c0 - foil_x) ** 2 + foil_z ** 2)
        self.AC = (self.pC - self.pA) * self.c0

    def forces_and_moments(self, V_cp2w, delta_Bl, delta_Br, reference_solution=None):
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
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        dF, dM : array of float, shape (K,3) [N, N m]
            Forces and moments for each section, proportional to the air
            density in [kg/m^3]. (This function assumes an air density of 1.)
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        delta = self.brake_geo(self.force_estimator.s_cps, delta_Bl, delta_Br)  # FIXME: leaky, don't grab s_cps directly
        dF, dM, solution = self.force_estimator(V_cp2w, delta, reference_solution)
        return dF, dM, solution

    def foil_origin(self, delta_a=0):
        """
        Compute the origin of the Parafoil coordinate system in FRD.

        Parameters
        ----------
        delta_a : float or array of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application

        Returns
        -------
        foil_origin : array of float, shape (3,) [meters]
            The offset of the origin of the Parafoil coordinate system in
            ParagliderWing coordinates.
        """
        # Speedbar shortens the A lines, while AC and C remain fixed
        A = self.A - (delta_a * self.kappa_a)
        foil_x = (A ** 2 - self.C ** 2 + self.AC ** 2) / (2 * self.AC)
        foil_y = 0  # FIXME: not with weight shift?
        foil_z = -np.sqrt(A ** 2 - foil_x ** 2)
        foil_x += self.pA * self.c0  # Account for the start of the AC line

        return np.array([foil_x, foil_y, foil_z])

    def equilibrium_alpha(self, deltaB, delta_a, reference_solution=None):
        """FIXME: add docstring."""
        def target(alpha, deltaB, delta_a, reference_solution):
            cp_wing = self.control_points(delta_a)
            v_wing = np.array([np.cos(alpha), 0, np.sin(alpha)])
            dF_w, dM_w, _ = self.forces_and_moments(
                v_wing, deltaB, deltaB, reference_solution,
            )
            M = dM_w.sum(axis=0)
            M += cross3(cp_wing, dF_w).sum(axis=0)
            return M[1]  # Wing pitching moment

        x0, x1 = np.deg2rad([8, 9])  # FIXME: review these bounds
        res = root_scalar(
            target, args=(deltaB, delta_a, reference_solution), x0=x0, x1=x1,
        )  # FIXME: add `rtol`?
        if not res.converged:
            raise foil.ForceEstimator.ConvergenceError
        return res.root

    def control_points(self, delta_a=0):
        """
        Compute the Parafoil control points in FRD.

        FIXME: descibe/define "control points"

        Parameters
        ----------
        delta_a : float or array of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application

        Returns
        -------
        cps : array of floats, shape (K,3) [meters]
            The control points in ParagliderWing coordinates
        """
        cps = self.force_estimator.control_points  # In Parafoil coordinates
        return cps + self.foil_origin(delta_a)  # In Wing coordinates

    # FIXME: moved from foil. Verify and test.
    def surface_distributions(self, delta_a=0):
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
        x, y, z = (self.parafoil.chords.xyz(s, 0.25) + self.foil_origin(delta_a)).T
        c = self.parafoil.chords.length(s)

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

    def inertia(self, rho_air, delta_a=0, N=200):
        """Compute the 3x3 moment of inertia matrix.

        TODO: this function currently only uses `delta_a` to determine the
              shift of the cg, but in the future `delta_a` may also deform the
              ParafoilLobe (if `lobe_args` starts getting used)

        TODO: precompute the static components that don't depend on `rho_air`

        Parameters
        ----------
        rho_air : float [kg/m^3]
            Volumetric air density of the atmosphere
        delta_a : float [percentage], optional
            Percentage of accelerator application
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

        o = -self.foil_origin(delta_a)  # Origin is `risers->origin`
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
