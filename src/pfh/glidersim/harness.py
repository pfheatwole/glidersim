import abc

import numpy as np


class Harness(abc.ABC):
    """
    FIXME: docstring
    """

    @abc.abstractmethod
    def control_points(self):
        """FIXME: docstring"""

    @abc.abstractmethod
    def forces_and_moments(self, v_W2h, rho_air):
        """
        Calculate the aerodynamic forces at the control points.

        Parameters
        ----------
        v_W2h : array of float, shape (K,3) [m/s]
            The wind velocity at each of the control points in harness frd.

        Returns
        -------
        dF, dM : array of float, shape (K,3) [N, N m]
            Aerodynamic forces and moments for each control point.
        """


class Spherical(Harness):
    """
    Models the harness as a point mass inside a sphere.

    The mass is concentrated at a single point, and the spherical assumption
    implies isotropic drag, characterized by a single drag coefficient.

    FIXME: finish documentation

    Notes
    -----
    Typical drag coefficient for pilot + harness:
     * Conventional:    0.8
     * Performance:     0.4

    Typical cross-sectional area for pilot + harness:
     * <80kg:           0.5
     * 80kg to 100kg:   0.6
     * >100kg:          0.7

    ref: PFD p85 (93)
    """

    def __init__(self, mass, z_riser, S, CD):
        # FIXME: the pilot is the thing that weight shifts; does the Paraglider
        #        manage that mass? Seems the right decision...

        """

        Parameters
        ----------
        mass : float [kg]
            The mass of the harness
        z_riser : float [m]
            The distance from the risers to the harness center of mass, given
            in a local frd coordinate system with the origin at the risers.
        S : float [m^2]
            The projected area of the sphere (ie, the area of a circle)
        CD : float [FIXME: units?]
            The isotropic drag coefficient
        """
        self._mass = mass
        self._z_riser = z_riser
        self._S = S
        self._CD = CD

    def control_points(self):
        return np.array([0, 0, self._z_riser])

    def forces_and_moments(self, v_W2h, rho_air):
        v2 = (v_W2h ** 2).sum()
        u_drag = v_W2h / np.sqrt(v2)  # Drag force unit vector
        dF = 0.5 * rho_air * v2 * self._S * self._CD * u_drag
        dM = np.zeros(3)
        return dF, dM

    def mass_properties(self):
        return {
            "mass": self._mass,
            "cm": np.array([0, 0, self._z_riser]),
            "J": np.zeros((3, 3)),  # FIXME: assumes a point mass
            "J_apparent": np.zeros((3, 3)),
        }
