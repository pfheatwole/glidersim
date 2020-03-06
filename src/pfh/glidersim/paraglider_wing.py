"""FIXME: add module docstring."""

import numpy as np

from scipy.optimize import root_scalar

from pfh.glidersim import foil
from pfh.glidersim.util import cross3


class ParagliderWing:
    """FIXME: add class docstring."""

    def __init__(self, canopy, force_estimator, brake_geo, d_riser, z_riser,
                 pA, pC, kappa_a, rho_upper, rho_lower):
        """
        FIXME: add docstring.

        Parameters
        ----------
        canopy : foil.FoilGeometry
            The geometric shape of the lifting surface.
        force_estimator : foil.ForceEstimator
            The estimation method for the aerodynamic forces and moments.
        brake_geo : BrakeGeometry
            Section trailing edge deflections as a function of delta_bl/br
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
        self.canopy = canopy
        self.force_estimator = force_estimator
        self.brake_geo = brake_geo
        self.pA = pA
        self.pC = pC
        self.kappa_a = kappa_a  # FIXME: strange notation. Why `kappa`?
        self.rho_upper = rho_upper
        self.rho_lower = rho_lower

        # The ParagliderWing coordinate axes are parallel to the canopy axes,
        # but the origin is translated from the central leading edge of the
        # canopy to `R`, the midpoint between the two riser connections.
        self.c0 = canopy.chord_length(0)
        foil_x = d_riser * self.c0
        foil_z = -z_riser

        # Default lengths of the A and C lines, and their connection distance
        self.A = np.sqrt((foil_x - self.pA * self.c0) ** 2 + foil_z ** 2)
        self.C = np.sqrt((self.pC * self.c0 - foil_x) ** 2 + foil_z ** 2)
        self.AC = (self.pC - self.pA) * self.c0

        # Compute the mass properties in canopy coordinates
        pmp = self.canopy.mass_properties(N=5000)
        m_upper = pmp["upper_area"] * self.rho_upper
        m_lower = pmp["lower_area"] * self.rho_lower
        J_upper = pmp["upper_inertia"] * self.rho_upper
        J_lower = pmp["lower_inertia"] * self.rho_lower
        m_solid = m_upper + m_lower
        cm_solid = (m_upper * pmp['upper_centroid']
                    + m_lower * pmp['lower_centroid']) / m_solid
        Ru = cm_solid - pmp["upper_centroid"]
        Rl = cm_solid - pmp["lower_centroid"]
        Du = (Ru @ Ru) * np.eye(3) - np.outer(Ru, Ru)
        Dl = (Rl @ Rl) * np.eye(3) - np.outer(Rl, Rl)
        J_solid = J_upper + m_upper * Du + J_lower + m_lower * Dl
        self._mass_properties = {
            "m_solid": m_solid,
            "cm_solid": cm_solid,  # In canopy coordinates
            "J_solid": J_solid,
            "m_air": pmp["volume"],  # Normalized by unit air density
            "cm_air": pmp["volume_centroid"],  # In canopy coordinates
            "J_air": pmp["volume_inertia"],  # Normalized by unit air density
        }

    def forces_and_moments(self, delta_bl, delta_br, v_W2b, rho_air, reference_solution=None):
        """
        FIXME: add docstring.

        Parameters
        ----------
        delta_bl : float [percentage]
            The amount of left brake
        delta_br : float [percentage]
            The amount of right brake
        v_W2b : array of float, shape (K,3) [m/s]
            The wind vector at each control point in body frd
        rho_air : float [kg/m^3]
            The ambient air density
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        dF, dM : array of float, shape (K,3) [N, N m]
            Aerodynamic forces and moments for each section.
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        delta_f = self.brake_geo(self.force_estimator.s_cps, delta_bl, delta_br)  # FIXME: leaky, don't grab s_cps directly
        dF, dM, solution = self.force_estimator(delta_f, v_W2b, rho_air, reference_solution)
        return dF, dM, solution

    def canopy_origin(self, delta_a=0):
        """
        Compute the origin of the FoilGeometry coordinate system in frd.

        Parameters
        ----------
        delta_a : float or array of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application

        Returns
        -------
        canopy_origin : array of float, shape (3,) [meters]
            The offset of the origin of the FoilGeometry coordinate system in
            ParagliderWing coordinates.
        """
        # Speedbar shortens the A lines, while AC and C remain fixed
        A = self.A - (delta_a * self.kappa_a)
        foil_x = (A ** 2 - self.C ** 2 + self.AC ** 2) / (2 * self.AC)
        foil_y = 0
        foil_z = -np.sqrt(A ** 2 - foil_x ** 2)
        foil_x += self.pA * self.c0  # Account for the start of the AC line

        return np.array([foil_x, foil_y, foil_z])

    def equilibrium_alpha(
        self, delta_a, delta_b, v_mag, rho_air, reference_solution=None,
    ):
        """Compute the zero aerodynamic pitching moment angle of attack."""
        cp_wing = self.control_points(delta_a)

        def target(alpha):
            v_W2b = -v_mag * np.array([np.cos(alpha), 0, np.sin(alpha)])
            dF_wing, dM_wing, _ = self.forces_and_moments(
                delta_b, delta_b, v_W2b, rho_air, reference_solution,
            )
            M = dM_wing.sum(axis=0) + cross3(cp_wing, dF_wing).sum(axis=0)
            return M[1]  # Wing pitching moment
        x0, x1 = np.deg2rad([9, 6])  # FIXME: review these bounds
        res = root_scalar(target, x0=x0, x1=x1)
        if not res.converged:
            raise foil.ForceEstimator.ConvergenceError
        return res.root

    def control_points(self, delta_a=0):
        """
        Compute the FoilGeometry control points in frd.

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
        cps = self.force_estimator.control_points  # In foil coordinates
        return cps + self.canopy_origin(delta_a)  # In wing coordinates

    def mass_properties(self, rho_air, delta_a=0):
        """
        Compute the mass properties of the solid mass and enclosed air.

        Parameters
        ----------
        rho_air : float [kg/m^3]
            Air density
        delta_a : float [percentage], optional
            Percentage of accelerator application

        Returns
        -------
        dictionary
            m_solid : float [kg]
                The solid mass of the wing
            cm_solid : array of float, shape (3,) [m]
                The solid mass centroid
            J_solid : array of float, shape (3,3) [kg m^2]
                The inertia matrix of the solid mass
            m_air : float [kg m^3]
                The enclosed air mass.
            cm_air : array of float, shape (3,) [m]
                The air mass centroid
            J_air : array of float, shape (3,3) [m^2]
                The inertia matrix of the enclosed air mass.
        """
        offset = self.canopy_origin(delta_a)  # canopy origin <- wing origin
        mp = self._mass_properties.copy()
        mp["cm_solid"] = mp["cm_solid"] + offset
        mp["cm_air"] = mp["cm_air"] + offset
        mp["m_air"] = mp["m_air"] * rho_air
        mp["J_air"] = mp["J_air"] * rho_air
        return mp
