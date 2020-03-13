import abc

import numpy as np


class Harness(abc.ABC):
    """
    FIXME: docstring
    """

    @abc.abstractmethod
    def control_points(self, delta_w):
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

    @abc.abstractmethod
    def mass_properties(self, delta_w):
        """
        FIXME: docstring
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

    def __init__(self, mass, z_riser, S, CD, kappa_w):
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
        kappa_w : float [m]
            The maximum weight shift distance
        """
        self._mass = mass
        self._z_riser = z_riser
        self._S = S
        self._CD = CD
        self._kappa_w = kappa_w  # FIXME: Strange notation to match `kappa_a`

    def control_points(self, delta_w=0):
        return np.array([0, delta_w * self._kappa_w, self._z_riser])

    def forces_and_moments(self, v_W2h, rho_air):
        v2 = (v_W2h ** 2).sum()
        u_drag = v_W2h / np.sqrt(v2)  # Drag force unit vector
        dF = 0.5 * rho_air * v2 * self._S * self._CD * u_drag
        dM = np.zeros(3)
        return dF, dM

    def mass_properties(self, delta_w=0):
        # Treats the mass as a uniform density solid sphere
        return {
            "mass": self._mass,
            "cm": self.control_points(delta_w),
            "J": (2 / 5 * self._mass * self._S / np.pi) * np.eye(3),
            "J_apparent": np.zeros((3, 3)),
        }
