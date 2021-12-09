"""FIXME: add module docstring"""

import abc
from typing import Protocol, runtime_checkable

import numpy as np


__all__ = [
    "Harness",
    "Spherical",
]


def __dir__():
    return __all__


@runtime_checkable
class Harness(Protocol):
    """Interface for classes that implement a Harness model."""

    @abc.abstractmethod
    def control_points(self, delta_w):
        """
        Compute the control points for the harness model dynamics.

        Parameters
        ----------
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)

        Returns
        -------
        r_CP2RM : float, shape (K,3) [m]
            Control points relative to the riser midpoint `RM`. Coordinates are
            in payload frd, and `K` is the number of control points for the
            harness model.
        """

    @abc.abstractmethod
    def aerodynamics(self, v_W2h, rho_air: float):
        """
        Calculate the aerodynamic forces and moments at each control point.

        Parameters
        ----------
        v_W2h : array of float, shape (K,3) [m/s]
            The wind velocity at each of the control points in harness frd.
        rho_air : float [kg/m^3]
            Air density

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
    Model a harness as a uniform density sphere.

    Coordinates use the front-right-down (frd) convention, with the origin at
    the midpoint of the two riser connections.

    Parameters
    ----------
    mass : float [kg]
        The mass of the harness
    z_riser : float [m]
        The vertical distance from `RM` to the harness center.
    S : float [m^2]
        The projected area of the sphere (ie, the area of a circle)

        Typical values for pilot + harness ([1]_):
         * <80kg:           0.5
         * 80kg to 100kg:   0.6
         * >100kg:          0.7

    CD : float
        The isotropic drag coefficient.

        Typical values for pilot + harness ([1]_):
         * Conventional:    0.8
         * Performance:     0.4

    kappa_w : float [m]
        The maximum weight shift distance

    Notes
    -----
    The spherical assumption has several effects:

    * Isotropic drag: the aerodynamic force is the same in all directions, so
      the drag coefficient is a single number. This implies that using the drag
      coefficient for a performance harness (shaped to minimize drag in the
      forward direction) will also reduce the drag from crosswind.

      Also, the aerodynamic moment for a sphere is zero, and since the
      aerodynamic force is computed at the center of mass, the net moment about
      the center of mass is always zero.

    * Isotropic inertia: neglects the fact that pilot will often extend their
      legs forward for aerodynamic efficiency, which should increase the pitch
      and yaw inertia.

    References
    ----------
    .. [1] Benedetti, Diego Muniz. "Paragliders Flight Dynamics". 2012. pg 85

    .. [2] Babinsky, Holger. "The aerodynamic performance of paragliders".
           1999. pg 422
    """

    def __init__(
        self,
        mass: float,
        z_riser: float,
        S: float,
        CD: float,
        kappa_w: float,
    ) -> None:
        self._mass = mass
        self._z_riser = z_riser
        self._S = S
        self._CD = CD
        self._kappa_w = kappa_w

    def control_points(self, delta_w=0):
        # FIXME: Weight shift probably shouldn't move the AERODYNAMIC control point
        delta_w = np.asarray(delta_w)
        r_P2RM = [
            np.zeros(delta_w.shape),
            delta_w * self._kappa_w,
            np.full(delta_w.shape, self._z_riser),
        ]
        return np.atleast_2d(np.array(r_P2RM).T)

    def aerodynamics(self, v_W2h, rho_air):
        v_W2h = np.asarray(v_W2h)
        assert v_W2h.shape == (3,)
        v2 = (v_W2h ** 2).sum()
        if not np.isclose(v2, 0.0):
            u_drag = v_W2h / np.sqrt(v2)  # Drag force unit vector
            dF = 0.5 * rho_air * v2 * self._S * self._CD * u_drag
        else:
            dF = np.zeros(3)
        dM = np.zeros(3)
        return dF, dM

    def mass_properties(self, delta_w: float = 0):
        # Treats the mass as a uniform density solid sphere
        return {
            "m_p": self._mass,
            "r_P2RM": self.control_points(delta_w)[0],
            "J_p2P": (2 / 5 * self._mass * self._S / np.pi) * np.eye(3),
        }
