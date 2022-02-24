"""Models of paraglider harnesses."""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import numpy as np

from pfh.glidersim.util import cross3


__all__ = [
    "ParagliderHarness",
    "Spherical",
]


def __dir__():
    return __all__


@runtime_checkable
class ParagliderHarness(Protocol):
    """Interface for classes that implement a ParagliderHarness model."""

    @abc.abstractmethod
    def r_CP2RM(self, delta_w):
        """
        Compute the control points for the harness model dynamics.

        Parameters
        ----------
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)

        Returns
        -------
        ndarray of float, shape (K,3) [m]
            Control points relative to the riser midpoint `RM`. Coordinates are
            in payload frd, and `K` is the number of control points for the
            harness model.
        """

    @abc.abstractmethod
    def mass_properties(self, delta_w: float, r_R2RM):
        """
        Compute inertia-related properties about a reference point.

        Parameters
        ----------
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        r_R2RM : array of float, shape (3,) [m]
            The reference point in harness frd

        Returns
        -------
        dictionary
            m_p : float [kg]
                The solid mass of the harness (and pilot)
            J_p2P : array of float, shape (3,3) [kg m^2]
                The moment of inertia matrix of the harness about its cm
            J_p2R : array of float, shape (3,3) [kg m^2]
                The moment of inertia matrix of the harness about `R`
            r_P2R : array of float, shape (3,) [m]
                The position of the harness cm relative to the reference point
            r_P2RM : array of float, shape (3,) [m]
                The position of the harness cm relative to the riser midpoint
        """

    @abc.abstractmethod
    def resultant_force(
        self,
        delta_w: float,
        v_W2h,
        rho_air: float,
        g,
        r_R2RM,
        mp: dict | None = None,
    ):
        """
        Calculate the net force and moment applied to the harness (and pilot).

        The moment is calculated with respect to a reference point `R` at some
        position relative to the harness origin `RM`.

        Parameters
        ----------
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        v_W2h : array of float, shape (K,3) [m/s]
            The wind velocity at each control point in harness frd.
        rho_air : float [kg/m^3]
            Air density
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in harness frd
        r_R2RM : array of float, shape (3,) [m]
            The reference point in harness frd about which the moment is
            calculated.
        mp : dictionary, optional
            The mass properties associated with the specified control inputs
            and reference point. Used to avoid recomputation.

        Returns
        -------
        f_h, g_h2R : array of float, shape (K,3) [N, N m]
            Net force and moment about the reference point `R`.
        """


class Spherical(ParagliderHarness):
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

    def r_CP2RM(self, delta_w=0):
        delta_w = np.asfarray(delta_w)
        r_P2RM = [
            np.zeros(delta_w.shape),
            delta_w * self._kappa_w,
            np.full(delta_w.shape, self._z_riser),
        ]
        return np.atleast_2d(np.array(r_P2RM).T)

    def mass_properties(self, delta_w, r_R2RM):
        # Treats the mass as a uniform density solid sphere
        m_p = self._mass
        J_p2P = (2 / 5 * m_p * self._S / np.pi) * np.eye(3)

        # Use the parallel axis theorem to also compute J_p2R
        r_P2RM = self.r_CP2RM(delta_w)[0]
        r_P2R = r_P2RM - r_R2RM
        D_p = (r_P2R @ r_P2R) * np.eye(3) - np.outer(r_P2R, r_P2R)
        J_p2R = J_p2P + self._mass * D_p

        return {
            "m_p": m_p,
            "J_p2P": J_p2P,
            "J_p2R": J_p2R,
            "r_P2R": r_P2R,
            "r_P2RM": r_P2RM,
        }

    def resultant_force(self, delta_w, v_W2h, rho_air, g, r_R2RM, mp=None):
        v_W2h = np.asfarray(v_W2h)
        g = np.asfarray(g)
        if mp is None:
            mp = self.mass_properties(delta_w, r_R2RM)

        if v_W2h.shape not in [(3,), (1, 3)]:
            raise ValueError("v_W2h must be a (3,) or a (1,3)")
        if v_W2h.shape == (1, 3):
            v_W2h = v_W2h[0]
        r_CP2RM = self.r_CP2RM(delta_w)[0]  # (1,3) -> (3,)

        v2 = (v_W2h**2).sum()
        if not np.isclose(v2, 0.0):
            u_drag = v_W2h / np.sqrt(v2)  # Drag force unit vector
            f_aero = 0.5 * rho_air * v2 * self._S * self._CD * u_drag
        else:
            f_aero = np.zeros(3)

        f_weight = mp["m_p"] * g
        f = f_aero + f_weight
        r_CP2R = r_CP2RM - r_R2RM
        g_h2R = cross3(r_CP2R, f_aero) + cross3(mp["r_P2R"], f_weight)
        return f, g_h2R
