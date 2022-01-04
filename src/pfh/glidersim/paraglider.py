"""Models of complete paraglider systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate
import scipy.optimize

from pfh.glidersim import foil_aerodynamics, orientation
from pfh.glidersim.util import cross3, crossmat


if TYPE_CHECKING:
    from pfh.glidersim.paraglider_harness import ParagliderHarness
    from pfh.glidersim.paraglider_wing import ParagliderWing


__all__ = [
    "ParagliderSystemDynamics6a",
    "ParagliderSystemDynamics6b",
    "ParagliderSystemDynamics6c",
    "ParagliderSystemDynamics9a",
    "ParagliderSystemDynamics9b",
    "ParagliderSystemDynamics9c",
]


def __dir__():
    return __all__


class ParagliderSystemDynamics6a:
    """
    A 6 degrees-of-freedom paraglider model; there is no relative motion
    between the wing and the harness.

    This version uses the riser connection midpoint `RM` as the reference point
    for the angular momentum, and can include the effects of apparent mass.

    Parameters
    ----------
    wing : ParagliderWing
    payload : ParagliderHarness
        The harness model includes the mass of the pilot.
    use_apparent_mass : bool, optional
        Whether to estimate the effects of apparent inertia.
    """

    def __init__(
        self,
        wing: ParagliderWing,
        payload: ParagliderHarness,
        *,
        use_apparent_mass: bool = True,
    ) -> None:
        self.wing = wing
        self.payload = payload
        self.use_apparent_mass = use_apparent_mass

    def r_CP2RM(self, delta_a: float = 0, delta_w: float = 0):
        """
        Compute the reference points for the composite Paraglider system.

        All the components of the Paraglider that experience aerodynamic forces
        need their relative wind vectors. Each component is responsible for
        creating a list of the coordinates where they need the value of the
        wind. This function then transforms them into body coordinates.

        Parameters
        ----------
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)

        Returns
        -------
        ndarray of float, shape (K,3) [m]
            The position of the control points with respect to `RM`.
        """
        r_LE2RM = -self.wing.r_RM2LE(delta_a)
        wing_cps = self.wing.r_CP2LE(delta_a=delta_a) + r_LE2RM
        payload_cps = self.payload.r_CP2RM(delta_w)
        return np.vstack((wing_cps, payload_cps))

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        g,
        delta_a: float = 0,
        delta_bl: float = 0,
        delta_br: float = 0,
        delta_w: float = 0,
        rho_air: float = 1.225,
        v_W2e=(0, 0, 0),
        reference_solution: dict | None = None,
    ):
        r"""
        Compute the translational and angular accelerations about the center of mass.

        Parameters
        ----------
        v_RM2e : array of float, shape (3,) [m/s]
            Translational velocity of `RM` in body frd coordinates, where `RM`
            is the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        rho_air : float [kg/m^3], optional
            Air density
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s], optional
            The wind relative to the earth, in body frd coordinates. If it is
            a (3,) array then the wind is uniform at every control point. If
            it is a (K,3) array then it is the vectors for each control point.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body with respect to Earth as the time
            derivative of angular velocity taken with respect to the body
            frame, expressed in body frd coordinates
            :math:`\left( ^b \dot{\omega}_{b/e}^b \right)`.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`
        """
        r_CP2RM = self.r_CP2RM(delta_a, delta_w)
        v_W2e = np.asfarray(v_W2e)
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError(
                "Number of distinct wind vectors in v_W2e do not match the "
                "number of control points",
            )
        v_RM2e = np.asfarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e.shape != (3,)")

        # -------------------------------------------------------------------
        # Compute the inertia of the body (wing + payload) with respect to the
        # riser midpoint `RM`.
        r_RM2LE = self.wing.r_RM2LE(delta_a)
        wmp = self.wing.mass_properties(rho_air, r_RM2LE)  # R = RM
        pmp = self.payload.mass_properties(delta_w, [0, 0, 0])  # R = RM
        m_b = wmp["m_b"] + pmp["m_p"]
        J_b2RM = wmp["J_b2R"] + pmp["J_p2R"]
        r_B2RM = (wmp["m_b"] * wmp["r_B2R"] + pmp["m_p"] * pmp["r_P2R"]) / m_b

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_CP2e = v_RM2e + cross3(omega_b2e, r_CP2RM)
        v_W2CP = v_W2e - v_CP2e

        # FIXME: "magic" indexing established by `self.r_CP2RM`
        v_W2CP_wing = v_W2CP[:-1]
        v_W2CP_payload = v_W2CP[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        try:
            f_b, g_b2RM, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                g,
                r_RM2LE,
                wmp,
                reference_solution,
            )
        except foil_aerodynamics.ConvergenceError:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            f_b, g_b2RM, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                g,
                r_RM2LE,
                wmp,
            )

        f_p, g_p2RM = self.payload.resultant_force(
            delta_w=delta_w,
            v_W2h=v_W2CP_payload,
            rho_air=rho_air,
            g=g,
            r_R2RM=[0, 0, 0],
            mp=pmp,
        )

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_RM2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum to the forces and moments.

        # fmt: off

        # Real mass momentums
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        p_b2e = m_b * v_B2e  # Linear
        h_b2RM = J_b2RM @ omega_b2e + m_b * cross3(r_B2RM, v_RM2e)  # Angular

        # Build the system matrices for the real mass
        A1 = [m_b * np.eye(3), -m_b * crossmat(r_B2RM)]
        A2 = [m_b * crossmat(r_B2RM), J_b2RM]
        A = np.block([A1, A2])
        B1 = f_b + f_p - cross3(omega_b2e, p_b2e)
        B2 = (  # ref: Hughes Eq:13, p58
            g_b2RM
            + g_p2RM
            - cross3(v_RM2e, p_b2e)
            - cross3(omega_b2e, h_b2RM)
        )

        if self.use_apparent_mass:
            amp = self.wing.apparent_mass_properties(
                rho_air,
                r_RM2LE,
                v_RM2e,
                omega_b2e,
            )
            A += amp["A_a2R"]  # Incorporate the apparent inertia
            B1 += (  # Apparent inertial force (Barrows Eq:61)
                -cross3(omega_b2e, amp["p_a2e"])
            )
            B2 += (  # Apparent inertial moment (Barrows Eq:64)
                -cross3(v_RM2e, amp["p_a2e"])
                - cross3(omega_b2e, amp["h_a2R"])
                + cross3(v_RM2e, amp["M_a"] @ v_RM2e)  # Remove the steady-state term
            )

        # fmt: on

        B = np.r_[B1, B2]
        x = np.linalg.solve(A, B)  # Solve for the derivatives
        a_RM2e = x[:3]  # In frame F_b
        alpha_b2e = x[3:]  # In frames F_b and F_e
        return a_RM2e, alpha_b2e, ref

    def equilibrium_state(
        self,
        delta_a: float = 0,
        delta_b: float = 0,
        rho_air: float = 1.225,
        alpha_0: float = None,
        theta_0: float = 0,
        v_0: float = 10,
        reference_solution: dict | None = None,
    ):
        """
        Compute the equilibrium glider state for given inputs.

        Assumes that the wing is symmetric about the xz-plane.

        Parameters
        ----------
        delta_a : float, optional
            Fraction of accelerator application, where `0 <= delta_a <= 1`
        delta_b : float, optional
            Fraction of symmetric brake application, where `0 <= delta_b <= 1`
        rho_air : float [kg/m^3], optional
            Air density
        alpha_0 : float [rad], optional
            An initial proposal for the body angle of attack. If no value is
            set, the wing equilibrium alpha will be used.
        theta_0 : float [rad], optional
            An initial proposal for the body pitch angle.
        v_0 : float [m/s], optional
            An initial proposal for the body airspeed.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        dictionary
            alpha_b : float [radians]
                Wing angle of attack
            gamma_b : float [radians]
                Wing glide angle
            glide_ratio : float
                Units of ground distance traveled per unit of altitude lost
            Theta_b2e : array of float, shape (3,) [radians]
                Equilibrium orientation of the body relative to Earth as a set
                of Tait-Bryan yaw-pitch-role angles.
            v_RM2e : float [m/s]
                Steady-state velocity of the riser midpoint in body coordinates
            solution : dictionary
                FIXME: docstring. See `Phillips.__call__`
        """
        if alpha_0 is None:
            alpha_0 = self.wing.equilibrium_alpha(
                delta_a=delta_a,
                delta_b=delta_b,
                v_mag=v_0,
                rho_air=rho_air,
                reference_solution=reference_solution,
            )

        _state = {
            "delta_a": delta_a,
            "delta_bl": delta_b,
            "delta_br": delta_b,
            "omega_b2e": np.zeros(3),
            "rho_air": rho_air,
            "reference_solution": reference_solution,
        }

        def _helper(x, state):
            v_x, v_z, theta_b2e = x
            a_RM2e, alpha_b2e, ref = self.accelerations(
                v_RM2e=[v_x, 0, v_z],
                g=9.8 * np.array([-np.sin(theta_b2e), 0, np.cos(theta_b2e)]),
                **state,
            )
            _state["reference_solution"] = ref
            return (a_RM2e[0], a_RM2e[2], alpha_b2e[1])

        v_x_0 = v_0 * np.cos(alpha_0)
        v_z_0 = v_0 * np.sin(alpha_0)
        x0 = np.array([v_x_0, v_z_0, theta_0])
        res = scipy.optimize.root(_helper, x0, _state)
        v_RM2e = np.array([res.x[0], 0, res.x[1]])  # In body frd
        alpha_eq = np.arctan2(v_RM2e[2], v_RM2e[0])
        theta_eq = res.x[2]
        gamma_eq = alpha_eq - theta_eq

        equilibrium = {
            "alpha_b": alpha_eq,
            "gamma_b": gamma_eq,
            "glide_ratio": 1 / np.tan(gamma_eq),
            "Theta_b2e": np.array([0, theta_eq, 0]),
            "v_RM2e": v_RM2e,
            "reference_solution": _state["reference_solution"],
        }

        return equilibrium


class ParagliderSystemDynamics6b(ParagliderSystemDynamics6a):
    """
    A 6 degrees-of-freedom paraglider model; there is no relative motion
    between the wing and the harness.

    This version uses the body center of mass `B` as the reference point for
    the angular momentum. Using the center of mass produces a decoupled linear
    system, which is easier to reason about, making this model useful for
    validating other models. The system solves for `a_B2e` which is then used
    to compute `a_RM2e`.

    This model does not support apparent mass; the apparent mass model requires
    that the reference point lies in the xz-plane, which is not the case for
    `B` during weight shift control. As a result, this model is intended to
    help validate other models involving the real mass only.

    Parameters
    ----------
    wing : ParagliderWing
    payload : ParagliderHarness
        The harness model includes the mass of the pilot.
    use_apparent_mass : bool, optional
        Whether to estimate the effects of apparent inertia.
    """

    def __init__(self, wing: ParagliderWing, payload: ParagliderHarness) -> None:
        self.wing = wing
        self.payload = payload

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        g,
        delta_a: float = 0,
        delta_bl: float = 0,
        delta_br: float = 0,
        delta_w: float = 0,
        rho_air: float = 1.225,
        v_W2e=(0, 0, 0),
        reference_solution=None,
    ):
        r"""
        Compute the translational and angular accelerations about the center of mass.

        Parameters
        ----------
        v_RM2e : array of float, shape (3,) [m/s]
            Translational velocity of `RM` in body frd coordinates, where `RM` is
            the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        rho_air : float [kg/m^3], optional
            Air density
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s], optional
            The wind relative to the earth, in body frd coordinates. If it is
            a (3,) array then the wind is uniform at every control point. If
            it is a (K,3) array then it is the vectors for each control point.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body with respect to Earth as the time
            derivative of angular velocity taken with respect to the body
            frame, expressed in body frd coordinates
            :math:`\left( ^b \dot{\omega}_{b/e}^b \right)`.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`
        """
        r_CP2RM = self.r_CP2RM(delta_a, delta_w)
        v_W2e = np.asfarray(v_W2e)
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError(
                "Number of distinct wind vectors in v_W2e do not match the "
                "number of control points",
            )
        v_RM2e = np.asfarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e.shape != (3,)")

        # -------------------------------------------------------------------
        # Compute the inertia of the body (wing + payload) with respect to its
        # center of mass `B`. Because `B` depends on the masses of both
        # components, this model must query those values and compute `B` before
        # computing properties with respect to `B`. This is wasteful, but this
        # model is mostly about checking the other models anyway.
        r_RM2LE = self.wing.r_RM2LE(delta_a)
        wmp0 = self.wing.mass_properties(rho_air, r_RM2LE)  # R = RM
        pmp0 = self.payload.mass_properties(delta_w, [0, 0, 0])  # R = RM
        m_b = wmp0["m_b"] + pmp0["m_p"]
        r_B2RM = (wmp0["m_b"] * wmp0["r_B2R"] + pmp0["m_p"] * pmp0["r_P2R"]) / m_b
        r_B2LE = r_B2RM + r_RM2LE

        # Recompute the mass properties using `B` as the reference point
        wmp = self.wing.mass_properties(rho_air, r_B2LE)  # R = B
        pmp = self.payload.mass_properties(delta_w, r_B2RM)  # R = B
        J_b2B = wmp["J_b2R"] + pmp["J_p2R"]

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_CP2e = v_RM2e + cross3(omega_b2e, r_CP2RM)
        v_W2CP = v_W2e - v_CP2e

        # FIXME: "magic" indexing established by `self.r_CP2RM`
        v_W2CP_wing = v_W2CP[:-1]
        v_W2CP_payload = v_W2CP[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        try:
            f_b, g_b2B, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                g,
                r_B2LE,
                wmp,
                reference_solution,
            )
        except foil_aerodynamics.ConvergenceError:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            f_b, g_b2B, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                g,
                r_B2LE,
                wmp,
            )

        f_p, g_p2B = self.payload.resultant_force(
            delta_w=delta_w,
            v_W2h=v_W2CP_payload,
            rho_air=rho_air,
            g=g,
            r_R2RM=r_B2RM,
            mp=pmp,
        )

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_RM2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum to the net forces and moments.

        # Real mass momentums
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        p_b2e = m_b * v_B2e  # Linear
        h_b2B = J_b2B @ omega_b2e  # Angular

        A1 = [m_b * np.eye(3), np.zeros((3, 3))]
        A2 = [np.zeros((3, 3)), J_b2B]
        A = np.block([A1, A2])
        B1 = f_b + f_p - cross3(omega_b2e, p_b2e)
        B2 = g_b2B + g_p2B - np.cross(omega_b2e, h_b2B)  # ref: Hughes Eq:13, p58
        B = np.r_[B1, B2]
        x = np.linalg.solve(A, B)  # Solve for the derivatives
        a_B2e = x[:3]  # In frame F_b
        alpha_b2e = x[3:]  # In frames F_b and F_e
        a_RM2e = a_B2e - np.cross(alpha_b2e, r_B2RM)  # In frame F_b
        return a_RM2e, alpha_b2e, ref


class ParagliderSystemDynamics6c(ParagliderSystemDynamics6a):
    """
    A 6 degrees-of-freedom paraglider model; there is no relative motion
    between the wing and the harness.

    This version uses the body center of mass `B` as the reference point for
    the angular momentum. Similar to 6b, except it solves for v_RM2e directly.

    This model does not support apparent mass; the apparent mass model requires
    that the reference point lies in the xz-plane, which is not the case for
    `B` during weight shift control. As a result, this model is intended to
    help validate other models involving the real mass only.

    Parameters
    ----------
    wing : ParagliderWing
    payload : ParagliderHarness
        The harness model includes the mass of the pilot.
    """

    def __init__(self, wing: ParagliderWing, payload: ParagliderHarness) -> None:
        self.wing = wing
        self.payload = payload

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        g,
        delta_a: float = 0,
        delta_bl: float = 0,
        delta_br: float = 0,
        delta_w: float = 0,
        rho_air: float = 1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution: dict | None = None,
    ):
        r"""
        Compute the translational and angular accelerations about the center of mass.

        Parameters
        ----------
        v_RM2e : array of float, shape (3,) [m/s]
            Translational velocity of `RM` in body frd coordinates, where `RM` is
            the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        rho_air : float [kg/m^3], optional
            Air density
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s], optional
            The wind relative to the earth, in body frd coordinates. If it is
            a (3,) array then the wind is uniform at every control point. If
            it is a (K,3) array then it is the vectors for each control point.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body with respect to Earth as the time
            derivative of angular velocity taken with respect to the body
            frame, expressed in body frd coordinates
            :math:`\left( ^b \dot{\omega}_{b/e}^b \right)`.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`
        """
        r_CP2RM = self.r_CP2RM(delta_a, delta_w)
        v_W2e = np.asfarray(v_W2e)
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError(
                "Number of distinct wind vectors in v_W2e do not match the "
                "number of control points",
            )
        v_RM2e = np.asfarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e.shape != (3,)")

        # -------------------------------------------------------------------
        # Compute the inertia of the body (wing + payload) with respect to its
        # center of mass `B`. Because `B` depends on the masses of both
        # components, this model must query those values and compute `B` before
        # computing properties with respect to `B`. This is wasteful, but this
        # model is mostly about checking the other models anyway.
        r_RM2LE = self.wing.r_RM2LE(delta_a)
        wmp0 = self.wing.mass_properties(rho_air, r_RM2LE)  # R = RM
        pmp0 = self.payload.mass_properties(delta_w, [0, 0, 0])  # R = RM
        m_b = wmp0["m_b"] + pmp0["m_p"]
        r_B2RM = (wmp0["m_b"] * wmp0["r_B2R"] + pmp0["m_p"] * pmp0["r_P2R"]) / m_b
        r_B2LE = r_B2RM + r_RM2LE

        # Recompute the mass properties using `B` as the reference point
        wmp = self.wing.mass_properties(rho_air, r_B2LE)  # R = B
        pmp = self.payload.mass_properties(delta_w, r_B2RM)  # R = B
        J_b2B = wmp["J_b2R"] + pmp["J_p2R"]

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_CP2e = v_RM2e + cross3(omega_b2e, r_CP2RM)
        v_W2CP = v_W2e - v_CP2e

        # FIXME: "magic" indexing established by `self.r_CP2RM`
        v_W2CP_wing = v_W2CP[:-1]
        v_W2CP_payload = v_W2CP[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        try:
            f_b, g_b2B, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                g,
                r_B2LE,
                wmp,
                reference_solution,
            )
        except foil_aerodynamics.ConvergenceError:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            f_b, g_b2B, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                g,
                r_B2LE,
                wmp,
            )

        f_p, g_p2B = self.payload.resultant_force(
            delta_w=delta_w,
            v_W2h=v_W2CP_payload,
            rho_air=rho_air,
            g=g,
            r_R2RM=r_B2RM,
            mp=pmp,
        )

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_RM2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum to the net forces and moments.

        # Real mass momentums
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        p_b2e = m_b * v_B2e  # Linear
        h_b2B = J_b2B @ omega_b2e  # Angular

        A1 = [m_b * np.eye(3), -m_b * crossmat(r_B2RM)]
        A2 = [np.zeros((3, 3)), J_b2B]
        A = np.block([A1, A2])
        B1 = f_b + f_p - cross3(omega_b2e, p_b2e)
        B2 = g_b2B + g_p2B - np.cross(omega_b2e, h_b2B)  # ref: Hughes Eq:13, p58
        B = np.r_[B1, B2]
        x = np.linalg.solve(A, B)  # Solve for the derivatives
        a_RM2e = x[:3]  # In frame F_b
        alpha_b2e = x[3:]  # In frames F_b and F_e
        return a_RM2e, alpha_b2e, ref


class ParagliderSystemDynamics9a:
    """
    A 9 degrees-of-freedom paraglider model, allowing rotation between the wing
    and the harness, with the connection modelled by spring-damper dynamics.

    This version uses the riser connection midpoint `RM` as the reference point
    for the angular momentum of both the body (the wing system) and the payload
    (the harness and pilot).

    Parameters
    ----------
    wing : ParagliderWing
    payload : ParagliderHarness
        The harness model includes the mass of the pilot.
    kappa_RM : array of float, shape (3,), optional
        Spring-damper coefficients for Theta_p2b (force as a linear function
        of angular displacement).
    kappa_RM_dot : array of float, shape (3,), optional
        Spring-damper coefficients for the derivative of Theta_p2b
    use_apparent_mass : bool, optional
        Whether to estimate the effects of apparent inertia. Default: True
    """

    def __init__(
        self,
        wing: ParagliderWing,
        payload: ParagliderHarness,
        kappa_RM=(0, 0, 0),
        kappa_RM_dot=(0, 0, 0),
        *,
        use_apparent_mass: bool = True,
    ) -> None:
        self.wing = wing
        self.payload = payload
        self._kappa_RM = np.asfarray(kappa_RM[:])
        self._kappa_RM_dot = np.asfarray(kappa_RM_dot[:])
        self.use_apparent_mass = use_apparent_mass

    def r_CP2RM(self, Theta_p2b, delta_a: float = 0, delta_w: float = 0):
        """
        Compute the reference points for the composite Paraglider system.

        All the components of the Paraglider that experience aerodynamic forces
        need their relative wind vectors. Each component is responsible for
        creating a list of the coordinates where they need the value of the
        wind. This function then transforms them into body coordinates.

        Parameters
        ----------
        Theta_p2b : array of float, shape (3,) [radians]
            The [phi, theta, gamma] of a yaw-pitch-roll sequence that encodes
            the relative orientation of the payload with respect to the body.
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)

        Returns
        -------
        ndarray of float, shape (K,3) [m]
            The position of the control points with respect to `RM`.
        """
        r_LE2RM = -self.wing.r_RM2LE(delta_a)
        wing_cps = self.wing.r_CP2LE(delta_a=delta_a)  # In body frd
        payload_cps = self.payload.r_CP2RM(delta_w)  # In payload frd
        C_b2p = orientation.euler_to_dcm(Theta_p2b).T
        return np.vstack((wing_cps + r_LE2RM, (C_b2p @ payload_cps.T).T))

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        omega_p2e,
        Theta_p2b,
        g,
        delta_a: float = 0,
        delta_bl: float = 0,
        delta_br: float = 0,
        delta_w: float = 0,
        rho_air: float = 1.225,
        v_W2e=(0, 0, 0),
        reference_solution: dict | None = None,
    ):
        r"""
        Compute the translational and angular accelerations about the center of mass.

        Parameters
        ----------
        v_RM2e : array of float, shape (3,) [m/s]
            Translational velocity of `RM` in body frd coordinates, where `RM` is
            the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        omega_p2e : array of float, shape (3,) [rad/s]
            Angular velocity of the payload, in payload frd coordinates.
        Theta_p2b : array of float, shape (3,) [radians]
            The [phi, theta, gamma] of a yaw-pitch-roll sequence that encodes
            the relative orientation of the payload with respect to the body.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        rho_air : float [kg/m^3], optional
            Air density
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s], optional
            The wind relative to the earth, in body frd coordinates. If it is
            a (3,) array then the wind is uniform at every control point. If
            it is a (K,3) array then it is the vectors for each control point.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body with respect to Earth as the time
            derivative of angular velocity taken with respect to the body
            frame, expressed in body frd coordinates
            :math:`\left( ^b \dot{\omega}_{b/e}^b \right)`.
        alpha_p2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the payload with respect to Earth as the
            time derivative of angular velocity taken with respect to the
            payload frame, expressed in payload frd coordinates
            :math:`\left( ^p \dot{\omega}_{p/e}^p \right)`.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`
        """
        r_CP2RM = self.r_CP2RM(Theta_p2b, delta_a, delta_w)
        v_W2e = np.asfarray(v_W2e)
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError(
                "Number of distinct wind vectors in v_W2e do not match the "
                "number of control points",
            )
        v_W2e = np.broadcast_to(v_W2e, r_CP2RM.shape)
        v_RM2e = np.asfarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e.shape != (3,)")

        C_p2b = orientation.euler_to_dcm(Theta_p2b)
        C_b2p = C_p2b.T
        omega_p2b = C_b2p @ omega_p2e - omega_b2e  # In body frd
        omega_b2p = -omega_p2b

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload about `RM`.
        r_RM2LE = self.wing.r_RM2LE(delta_a)
        wmp = self.wing.mass_properties(rho_air, r_RM2LE)  # R = RM
        m_b = wmp["m_b"]
        J_b2RM = wmp["J_b2R"]
        r_B2RM = wmp["r_B2R"]

        pmp = self.payload.mass_properties(delta_w, [0, 0, 0])  # R = RM
        m_p = pmp["m_p"]
        J_p2RM = pmp["J_p2R"]
        r_P2RM = pmp["r_P2R"]

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point. Body
        # vectors are in body frd, payload vectors are in payload frd.
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        v_P2e = C_p2b @ v_RM2e + cross3(omega_p2e, r_P2RM)

        # FIXME: "magic" indexing established by `self.r_CP2RM`
        r_CP2RM_b = r_CP2RM[:-1]
        r_CP2RM_p = C_p2b @ r_CP2RM[-1]

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2RM_b - r_B2RM)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2RM_p - r_P2RM)

        v_W2CP_b = v_W2e[:-1] - v_CP2e_b
        v_W2CP_p = C_p2b @ v_W2e[-1] - v_CP2e_p

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        try:
            f_b, g_b2RM, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                g,
                r_RM2LE,
                wmp,
                reference_solution,
            )
        except foil_aerodynamics.ConvergenceError:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            f_b, g_b2RM, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                g,
                r_RM2LE,
                wmp,
            )

        f_p, g_p2RM = self.payload.resultant_force(  # In payload frd
            delta_w=delta_w,
            v_W2h=v_W2CP_p,
            rho_air=rho_air,
            g=C_p2b @ g,
            r_R2RM=[0, 0, 0],
            mp=pmp,
        )

        # Moment at the connection point `RM` modeled as a spring+damper system
        g_RM = Theta_p2b * self._kappa_RM + omega_p2b * self._kappa_RM_dot

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_RM2e`,
        # `omega_b2e` (in body frd), `omega_p2e` (in payload frd), and the
        # internal force on the risers, `F_RM` (in body frd).

        # fmt: off

        # Real mass momentums
        p_b2e = m_b * v_B2e  # Linear
        p_p2e = m_p * v_P2e
        h_b2RM = m_b * cross3(r_B2RM, v_RM2e) + J_b2RM @ omega_b2e  # Angular
        h_p2RM = m_p * cross3(r_P2RM, C_p2b @ v_RM2e) + J_p2RM @ omega_p2e

        # Build the system matrices for the real mass. A1 and A2 are the forces
        # and moments on the body, A3 and A4 are the forces and moments on the
        # payload. The unknowns are [a_RM2e^b, alpha_b2e^b, alpha_p2e^p, F_RM^b].
        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A1 = [m_b * I3, -m_b * crossmat(r_B2RM), Z3, I3]
        A2 = [m_b * crossmat(r_B2RM), J_b2RM, Z3, Z3]
        A3 = [m_p * C_p2b, Z3, -m_p * crossmat(r_P2RM), -C_p2b]
        A4 = [m_p * crossmat(r_P2RM) @ C_p2b, Z3, J_p2RM, Z3]
        A = np.block([A1, A2, A3, A4])

        B1 = f_b - cross3(omega_b2e, p_b2e)
        B2 = (  # ref: Hughes Eq:13, p58
            g_b2RM
            - g_RM
            - cross3(v_RM2e, p_b2e)
            - cross3(omega_b2e, h_b2RM)
        )
        B3 = (
            f_p
            - m_p * C_p2b @ cross3(omega_b2p, v_RM2e)
            - cross3(omega_p2e, p_p2e)
        )
        B4 = (
            g_p2RM
            + C_p2b @ g_RM
            - cross3(C_p2b @ v_RM2e, p_p2e)
            - m_p * cross3(r_P2RM, C_p2b @ cross3(omega_b2p, v_RM2e))
            - cross3(omega_p2e, h_p2RM)
        )

        if self.use_apparent_mass:
            amp = self.wing.apparent_mass_properties(
                rho_air,
                r_RM2LE,
                v_RM2e,
                omega_b2e,
            )
            A[:6, :6] += amp["A_a2R"]  # Incorporate the apparent inertia
            B1 += (  # Apparent inertial force (Barrows Eq:61)
                -cross3(omega_b2e, amp["p_a2e"])
            )
            B2 += (  # Apparent inertial moment (Barrows Eq:64)
                -cross3(v_RM2e, amp["p_a2e"])
                - cross3(omega_b2e, amp["h_a2R"])
                + cross3(v_RM2e, amp["M_a"] @ v_RM2e)  # Remove the steady-state term
            )

        # fmt: on

        B = np.r_[B1, B2, B3, B4]
        x = np.linalg.solve(A, B)
        a_RM2e = x[:3]  # In frame F_b
        alpha_b2e = x[3:6]  # In frames F_b and F_e
        alpha_p2e = x[6:9]  # In frames F_p and F_e
        F_RM = x[9:]  # For debugging
        return a_RM2e, alpha_b2e, alpha_p2e, ref

    def equilibrium_state(
        self,
        delta_a: float = 0,
        delta_b: float = 0,
        rho_air: float = 1.225,
        alpha_0: float | None = None,
        theta_0: float = 0,
        v_0: float = 10,
        reference_solution=None,
    ):
        """
        Compute the equilibrium glider state for given inputs.

        Assumes that the wing is symmetric about the xz-plane.

        Parameters
        ----------
        delta_a : float, optional
            Fraction of accelerator application, where `0 <= delta_a <= 1`
        delta_b : float, optional
            Fraction of symmetric brake application, where `0 <= delta_b <= 1`
        rho_air : float [kg/m^3], optional
            Air density
        alpha_0 : float [rad], optional
            An initial proposal for the body angle of attack. If no value is
            set, the wing equilibrium alpha will be used.
        theta_0 : float [rad], optional
            An initial proposal for the body pitch angle.
        v_0 : float [m/s], optional
            An initial proposal for the body airspeed.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        dictionary
            alpha_b : float [radians]
                Wing angle of attack
            gamma_b : float [radians]
                Wing glide angle
            glide_ratio : float
                Units of ground distance traveled per unit of altitude lost
            Theta_b2e : array of float, shape (3,) [radians]
                Equilibrium orientation of the body relative to Earth as a set
                of Tait-Bryan yaw-pitch-role angles.
            v_RM2e : float [m/s]
                Steady-state velocity of the riser midpoint in body coordinates
            solution : dictionary
                FIXME: docstring. See `Phillips.__call__`
        """
        if alpha_0 is None:
            alpha_0 = self.wing.equilibrium_alpha(
                delta_a=delta_a,
                delta_b=delta_b,
                v_mag=v_0,
                rho_air=rho_air,
                reference_solution=reference_solution,
            )

        _state = {
            "delta_a": delta_a,
            "delta_bl": delta_b,
            "delta_br": delta_b,
            "omega_b2e": np.zeros(3),
            "omega_p2e": np.zeros(3),
            "rho_air": rho_air,
            "reference_solution": reference_solution,
        }

        def _helper(x, state):
            v_x, v_z, theta_b2e, theta_p2e = x
            theta_p2b = theta_p2e - theta_b2e
            a_RM2e, alpha_b2e, alpha_p2e, ref = self.accelerations(
                v_RM2e=[v_x, 0, v_z],
                g=9.8 * np.array([-np.sin(theta_b2e), 0, np.cos(theta_b2e)]),
                Theta_p2b=[0, theta_p2b, 0],
                **state,
            )
            _state["reference_solution"] = ref
            return (a_RM2e[0], a_RM2e[2], alpha_b2e[1], alpha_p2e[1])

        v_x_0 = v_0 * np.cos(alpha_0)
        v_z_0 = v_0 * np.sin(alpha_0)
        theta_p2e_0 = 0  # Assume the payload is hanging straight down
        x0 = np.array([v_x_0, v_z_0, theta_0, theta_p2e_0])
        res = scipy.optimize.root(_helper, x0, _state)
        v_RM2e = np.array([res.x[0], 0, res.x[1]])
        alpha_eq = np.arctan2(v_RM2e[2], v_RM2e[0])
        theta_b2e = res.x[2]
        theta_p2b = res.x[3] - res.x[2]
        gamma_eq = alpha_eq - theta_b2e

        equilibrium = {
            "alpha_b": alpha_eq,
            "gamma_b": gamma_eq,
            "glide_ratio": 1 / np.tan(gamma_eq),
            "Theta_b2e": np.array([0, theta_b2e, 0]),
            "Theta_p2b": np.array([0, theta_p2b, 0]),
            "v_RM2e": v_RM2e,
            "reference_solution": _state["reference_solution"],
        }

        return equilibrium


class ParagliderSystemDynamics9b(ParagliderSystemDynamics9a):
    """
    A 9 degrees-of-freedom paraglider model, allowing rotation between the wing
    and the harness, with the connection modelled by spring-damper dynamics.

    This model uses the body center of mass `B` as the reference point for the
    angular momentum of the body (the wing system) and the payload center of
    mass `P` for the angular momentum of the payload (the harness and pilot).

    This model does not support apparent mass. (Because it uses `B` as the
    reference point for the body dynamics, the system of equations for apparent
    mass would be in terms of `a_B2e`, which makes the equations involving the
    payload dynamics significantly messier.) Its purpose is to check for
    implementation errors in other models involving the real mass only.

    Parameters
    ----------
    wing : ParagliderWing
    payload : ParagliderHarness
        The harness model includes the mass of the pilot.
    kappa_RM : array of float, shape (3,), optional
        Spring-damper coefficients for Theta_p2b (force as a linear function
        of angular displacement).
    kappa_RM_dot : array of float, shape (3,), optional
        Spring-damper coefficients for the derivative of Theta_p2b
    """

    def __init__(
        self,
        wing: ParagliderWing,
        payload: ParagliderHarness,
        kappa_RM=(0, 0, 0),
        kappa_RM_dot=(0, 0, 0),
    ) -> None:
        self.wing = wing
        self.payload = payload
        self._kappa_RM = np.asfarray(kappa_RM[:])
        self._kappa_RM_dot = np.asfarray(kappa_RM_dot[:])

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        omega_p2e,
        Theta_p2b,
        g,
        delta_a: float = 0,
        delta_bl: float = 0,
        delta_br: float = 0,
        delta_w: float = 0,
        rho_air: float = 1.225,
        v_W2e=(0, 0, 0),
        reference_solution: dict | None = None,
    ):
        r"""
        Compute the translational and angular accelerations about the center of mass.

        Parameters
        ----------
        v_RM2e : array of float, shape (3,) [m/s]
            Translational velocity of `RM` in body frd coordinates, where `RM` is
            the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        omega_p2e : array of float, shape (3,) [rad/s]
            Angular velocity of the payload, in payload frd coordinates.
        Theta_p2b : array of float, shape (3,) [radians]
            The [phi, theta, gamma] of a yaw-pitch-roll sequence that encodes
            the relative orientation of the payload with respect to the body.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        rho_air : float [kg/m^3], optional
            Air density
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s], optional
            The wind relative to the earth, in body frd coordinates. If it is
            a (3,) array then the wind is uniform at every control point. If
            it is a (K,3) array then it is the vectors for each control point.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body with respect to Earth as the time
            derivative of angular velocity taken with respect to the body
            frame, expressed in body frd coordinates
            :math:`\left( ^b \dot{\omega}_{b/e}^b \right)`.
        alpha_p2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the payload with respect to Earth as the
            time derivative of angular velocity taken with respect to the
            payload frame, expressed in payload frd coordinates
            :math:`\left( ^p \dot{\omega}_{p/e}^p \right)`.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`
        """
        r_CP2RM = self.r_CP2RM(Theta_p2b, delta_a, delta_w)
        v_W2e = np.asfarray(v_W2e)
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError(
                "Number of distinct wind vectors in v_W2e do not match the "
                "number of control points",
            )
        v_W2e = np.broadcast_to(v_W2e, r_CP2RM.shape)
        v_RM2e = np.asfarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e.shape != (3,)")

        C_p2b = orientation.euler_to_dcm(Theta_p2b)
        C_b2p = C_p2b.T

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload with respect
        # to their centers of mass (`B` and `P`). This is wasteful since it
        # calls `mass_properties` twice just to get the centers of mass, but
        # it's simple and this model is just for validation anyway.
        r_RM2LE = self.wing.r_RM2LE(delta_a)
        wmp0 = self.wing.mass_properties(rho_air, [0, 0, 0])  # R = LE
        wmp = self.wing.mass_properties(rho_air, wmp0["r_B2R"])  # R = B
        m_b = wmp["m_b"]
        J_b2B = wmp["J_b2R"]
        r_B2RM = wmp0["r_B2R"] - r_RM2LE
        r_B2LE = r_B2RM + r_RM2LE

        pmp0 = self.payload.mass_properties(delta_w, [0, 0, 0])  # R = RM
        pmp = self.payload.mass_properties(delta_w, pmp0["r_P2R"])  # R = P
        m_p = pmp["m_p"]
        J_p2P = pmp["J_p2R"]
        r_P2RM = pmp0["r_P2R"]

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point. Body
        # vectors are in body frd, payload vectors are in payload frd.
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        v_P2e = C_p2b @ v_RM2e + cross3(omega_p2e, r_P2RM)

        # FIXME: "magic" indexing established by `self.r_CP2RM`
        r_CP2B_b = r_CP2RM[:-1] - r_B2RM
        r_CP2P_p = C_p2b @ r_CP2RM[-1] - r_P2RM

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2B_b)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2P_p)

        v_W2CP_b = v_W2e[:-1] - v_CP2e_b
        v_W2CP_p = C_p2b @ v_W2e[-1] - v_CP2e_p

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        try:
            f_b, g_b2B, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                g,
                r_B2LE,
                wmp,
                reference_solution,
            )
        except foil_aerodynamics.ConvergenceError:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            f_b, g_b2B, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                g,
                r_B2LE,
                wmp,
            )

        f_p, g_p2P = self.payload.resultant_force(  # In payload frd
            delta_w=delta_w,
            v_W2h=v_W2CP_p,
            rho_air=rho_air,
            g=C_p2b @ g,
            r_R2RM=pmp["r_P2RM"],
            mp=pmp,
        )

        # Moment at the connection point `RM` modeled as a spring+damper system
        omega_p2b = C_b2p @ omega_p2e - omega_b2e
        g_RM = Theta_p2b * self._kappa_RM + omega_p2b * self._kappa_RM_dot

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_RM2e`,
        # `omega_b2e` (in body frd), `omega_p2e` (in payload frd), and the
        # internal force on the risers, `F_RM` (in body frd).

        # fmt: off

        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A1 = [m_b * I3, -m_b * crossmat(r_B2RM), Z3, I3]
        A2 = [Z3, J_b2B, Z3, -crossmat(r_B2RM)]
        A3 = [m_p * C_p2b, Z3, -m_p * crossmat(r_P2RM), -C_p2b]
        A4 = [Z3, Z3, J_p2P, crossmat(r_P2RM) @ C_p2b]
        A = np.block([A1, A2, A3, A4])
        B1 = (
            f_b
            - m_b * cross3(omega_b2e, v_RM2e)
            - m_b * cross3(omega_b2e, cross3(omega_b2e, r_B2RM))
        )
        B2 = (  # ref: Hughes Eq:13, p58
            g_b2B
            - g_RM
            - cross3(omega_b2e, J_b2B @ omega_b2e)
        )
        B3 = (
            f_p
            - m_p * C_p2b @ cross3(omega_b2e, v_RM2e)
            - m_p * cross3(omega_p2e, cross3(omega_p2e, r_P2RM))
        )
        B4 = g_p2P + C_p2b @ g_RM - cross3(omega_p2e, J_p2P @ omega_p2e)

        # fmt: on

        B = np.r_[B1, B2, B3, B4]
        x = np.linalg.solve(A, B)
        a_RM2e = x[:3]  # In frame F_b
        alpha_b2e = x[3:6]  # In frames F_b and F_e
        alpha_p2e = x[6:9]  # In frames F_p and F_e
        F_RM = x[9:]  # For debugging
        return a_RM2e, alpha_b2e, alpha_p2e, ref


class ParagliderSystemDynamics9c(ParagliderSystemDynamics9a):
    r"""
    A 9 degrees-of-freedom paraglider model, allowing rotation between the wing
    and the harness, with the connection modelled by spring-damper dynamics.

    Similar to ParagliderSystemDynamics9a, this version uses the riser midpoint
    `RM` as the reference point for both the body and the payload. Unlike
    ParagliderSystemDynamics9a, this model computes \dot{omega_p2b} instead of
    \dot{omega_p2e} and converts. Also, note that it computes everything in
    body frd and converts omega_p2e back to payload frd at the very end.

    FIXME: although 9a and 9b agree, this model produces slightly different
    answers, which might be worth looking into.

    Parameters
    ----------
    wing : ParagliderWing
    payload : ParagliderHarness
        The harness model includes the mass of the pilot.
    kappa_RM : array of float, shape (3,), optional
        Spring-damper coefficients for Theta_p2b (force as a linear function
        of angular displacement).
    kappa_RM_dot : array of float, shape (3,), optional
        Spring-damper coefficients for the derivative of Theta_p2b
    use_apparent_mass : bool, optional
        Whether to estimate the effects of apparent inertia. Default: True
    """

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        omega_p2e,
        Theta_p2b,
        g,
        delta_a: float = 0,
        delta_bl: float = 0,
        delta_br: float = 0,
        delta_w: float = 0,
        rho_air: float = 1.225,
        v_W2e=(0, 0, 0),
        reference_solution: dict | None = None,
    ):
        r"""
        Compute the translational and angular accelerations about the center of mass.

        Parameters
        ----------
        v_RM2e : array of float, shape (3,) [m/s]
            Translational velocity of `RM` in body frd coordinates, where `RM`
            is the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        omega_p2e : array of float, shape (3,) [rad/s]
            Angular velocity of the payload, in payload frd coordinates.
        Theta_p2b : array of float, shape (3,) [radians]
            The [phi, theta, gamma] of a yaw-pitch-roll sequence that encodes
            the relative orientation of the payload with respect to the body.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_w : float [percentage]
            The fraction of weight shift, from -1 (left) to +1 (right)
        rho_air : float [kg/m^3], optional
            Air density
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s], optional
            The wind relative to the earth, in body frd coordinates. If it is
            a (3,) array then the wind is uniform at every control point. If
            it is a (K,3) array then it is the vectors for each control point.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body with respect to Earth as the time
            derivative of angular velocity taken with respect to the body
            frame, expressed in body frd coordinates
            :math:`\left( ^b \dot{\omega}_{b/e}^b \right)`.
        alpha_p2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the payload with respect to Earth as the
            time derivative of angular velocity taken with respect to the
            payload frame, expressed in payload frd coordinates
            :math:`\left( ^p \dot{\omega}_{p/e}^p \right)`.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`
        """
        r_CP2RM = self.r_CP2RM(Theta_p2b, delta_a, delta_w)
        v_W2e = np.asfarray(v_W2e)
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError(
                "Number of distinct wind vectors in v_W2e do not match the "
                "number of control points",
            )
        v_W2e = np.broadcast_to(v_W2e, r_CP2RM.shape)
        v_RM2e = np.asfarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e.shape != (3,)")

        C_p2b = orientation.euler_to_dcm(Theta_p2b)
        C_b2p = C_p2b.T
        omega_p2e = C_b2p @ omega_p2e  # Dynamics9a uses `p`, this model uses `b`

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload.
        r_RM2LE = self.wing.r_RM2LE(delta_a)
        wmp = self.wing.mass_properties(rho_air, r_RM2LE)  # R = RM
        m_b = wmp["m_s"] + wmp["m_air"]
        J_b2RM = wmp["J_b2R"]
        r_B2RM = wmp["r_B2R"]

        pmp = self.payload.mass_properties(delta_w, [0, 0, 0])  # R = RM
        m_p = pmp["m_p"]
        J_p2RM = C_b2p @ pmp["J_p2R"] @ C_p2b  # In body frd
        r_P2RM = C_b2p @ pmp["r_P2R"]  # In body frd

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point. All vectors
        # are in body frd.
        omega_p2b = omega_p2e - omega_b2e
        omega_b2p = -omega_p2b
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        v_P2e = v_RM2e + cross3(omega_p2e, r_P2RM)

        # FIXME: "magic" indexing established by `self.r_CP2RM`
        r_CP2RM_b = r_CP2RM[:-1]
        r_CP2RM_p = r_CP2RM[-1]

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2RM_b - r_B2RM)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2RM_p - r_P2RM)

        v_W2CP_b = v_W2e[:-1] - v_CP2e_b
        v_W2CP_p = v_W2e[-1] - v_CP2e_p

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        try:
            f_b, g_b2RM, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                g,
                r_RM2LE,
                wmp,
                reference_solution,
            )
        except foil_aerodynamics.ConvergenceError:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            f_b, g_b2RM, ref = self.wing.resultant_force(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                g,
                r_RM2LE,
                wmp,
            )

        f_p, g_p2RM = self.payload.resultant_force(  # In payload frd
            delta_w=delta_w,
            v_W2h=C_p2b @ v_W2CP_p,
            rho_air=rho_air,
            g=C_p2b @ g,
            r_R2RM=[0, 0, 0],
            mp=pmp,
        )

        # Moment at the connection point `RM` modeled as a spring+damper system
        g_RM = Theta_p2b * self._kappa_RM + omega_p2b * self._kappa_RM_dot

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_RM2e^b`,
        # `omega_b2e^b`, `omega_p2e^p`, and the internal force on the risers,
        # `F_RM^b`. All derivatives are from the body frame.

        # fmt: off

        # Real mass momentums
        p_b2e = m_b * v_B2e  # Linear
        p_p2e = m_p * v_P2e
        h_b2RM = m_b * cross3(r_B2RM, v_RM2e) + J_b2RM @ omega_b2e  # Angular
        h_p2RM = m_p * cross3(r_P2RM, v_RM2e) + J_p2RM @ omega_p2e

        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A1 = [m_b * I3, -m_b * crossmat(r_B2RM), Z3, I3]
        A2 = [m_b * crossmat(r_B2RM), J_b2RM, Z3, Z3]
        A3 = [m_p * I3, -m_p * crossmat(r_P2RM), -m_p * crossmat(r_P2RM), -I3]
        A4 = [m_p * crossmat(r_P2RM), J_p2RM, J_p2RM, Z3]
        A = np.block([A1, A2, A3, A4])

        B1 = f_b - cross3(omega_b2e, p_b2e)
        B2 = (  # ref: Hughes Eq:13, p58
            g_b2RM
            - g_RM
            - cross3(v_RM2e, p_b2e)
            - cross3(omega_b2e, h_b2RM)
        )
        B3 = (
            C_b2p @ f_p
            - m_p * cross3(omega_b2p, v_RM2e)
            - m_p * cross3(cross3(omega_b2p, omega_b2e), r_P2RM)
            # - m_p * cross3(cross3(omega_b2p, omega_p2e), r_P2RM)  # equivalent
            - cross3(omega_p2e, p_p2e)
        )
        B4 = (
            C_b2p @ g_p2RM
            + g_RM
            - cross3(v_RM2e, p_p2e)
            - m_p * cross3(r_P2RM, cross3(omega_b2p, v_RM2e))
            - cross3(omega_b2p, J_p2RM @ omega_p2e)
            - cross3(omega_p2e, h_p2RM)
        )

        if self.use_apparent_mass:
            amp = self.wing.apparent_mass_properties(
                rho_air,
                r_RM2LE,
                v_RM2e,
                omega_b2e,
            )
            A[:6, :6] += amp["A_a2R"]  # Incorporate the apparent inertia
            B1 += (  # Apparent inertial force (Barrows Eq:61)
                -cross3(omega_b2e, amp["p_a2e"])
            )
            B2 += (  # Apparent inertial moment (Barrows Eq:64)
                -cross3(v_RM2e, amp["p_a2e"])
                - cross3(omega_b2e, amp["h_a2R"])
                + cross3(v_RM2e, amp["M_a"] @ v_RM2e)  # Remove the steady-state term
            )

        # fmt: on

        B = np.r_[B1, B2, B3, B4]
        x = np.linalg.solve(A, B)
        a_RM2e = x[:3]  # In frame F_b
        alpha_b2e = x[3:6]  # In frames F_b and F_e
        alpha_p2b = x[6:9]  # In frames F_b and F_p
        F_RM = x[9:]  # For debugging
        alpha_p2e = alpha_p2b + alpha_b2e + cross3(omega_b2e, omega_p2b)
        alpha_p2e = C_p2b @ alpha_p2e  # In frames F_p and F_e
        return a_RM2e, alpha_b2e, alpha_p2e, ref
