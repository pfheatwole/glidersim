"""FIXME: add module docstring."""

import numpy as np
import scipy.integrate
import scipy.optimize

from pfh.glidersim import orientation
from pfh.glidersim.util import cross3, crossmat


__all__ = [
    "Paraglider6a",
    "Paraglider6b",
    "Paraglider6c",
    "Paraglider9a",
    "Paraglider9b",
    "Paraglider9c",
]


def __dir__():
    return __all__


class Paraglider6a:
    """
    A 6 degrees-of-freedom paraglider model; there is no relative motion
    between the wing and the harness.

    This version uses the riser connection midpoint `RM` as the reference point
    for the angular momentum, and includes the effects of apparent mass.

    Parameters
    ----------
    wing : ParagliderWing
    payload : Harness
        This uses a `Harness`, but since there is no model for the pilot
        the harness should include the pilot mass.
    use_apparent_mass : bool, optional
        Whether to estimate the effects of apparent inertia. Default: True
    """

    def __init__(self, wing, payload, *, use_apparent_mass=True):
        self.wing = wing
        self.payload = payload
        self.use_apparent_mass = use_apparent_mass

    def control_points(self, delta_a=0, delta_w=0):
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
        r_CP2RM : array of float, shape (N,3) [m]
            The position of the control points with respect to `RM`.
        """
        r_LE2RM = -self.wing.r_RM2LE(delta_a)
        wing_cps = self.wing.control_points(delta_a=delta_a) + r_LE2RM
        payload_cps = self.payload.control_points(delta_w)
        return np.vstack((wing_cps, payload_cps))

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        g,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

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
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2RM : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2RM` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2RM. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2RM` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body, in body frd coordinates.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        There are two use cases:

        1. Uniform global wind across the wing (v_W2e.shape == (3,))

        2. Non-uniform global wind across the wing (v_W2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2RM is None:
            # FIXME: needs a design review. The idea was that if `v_W2e` is
            #        given for each individual control point, then require the
            #        values of those control points to ensure they match the
            #        current state of the wing (including the current control
            #        inputs, `delta_a` and `delta_w`, which move the CPs). I've
            #        never liked this design.
            raise ValueError("Control point relative winds require r_CP2RM")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError("Different number of wind and r_CP2RM vectors")
        if r_CP2RM is None:
            r_CP2RM = self.control_points(delta_a, delta_w)

        v_RM2e = np.asarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        # -------------------------------------------------------------------
        # Compute the inertia matrices about the riser connection midpoint `RM`
        wmp = self.wing.mass_properties(rho_air, delta_a)
        pmp = self.payload.mass_properties(delta_w)
        m_b = wmp["m_s"] + wmp["m_air"] + pmp["m_p"]
        r_B2RM = (  # Center of mass of the body system
            wmp["m_s"] * wmp["r_S2RM"]
            + wmp["m_air"] * wmp["r_V2RM"]
            + pmp["m_p"] * pmp["r_P2RM"]
        ) / m_b
        r_S2RM = wmp["r_S2RM"]  # Displacement of the wing solid mass
        r_V2RM = wmp["r_V2RM"]  # Displacement of the wing enclosed air
        r_P2RM = pmp["r_P2RM"]  # Displacement of the payload mass
        D_s = (r_S2RM @ r_S2RM) * np.eye(3) - np.outer(r_S2RM, r_S2RM)
        D_v = (r_V2RM @ r_V2RM) * np.eye(3) - np.outer(r_V2RM, r_V2RM)
        D_p = (r_P2RM @ r_P2RM) * np.eye(3) - np.outer(r_P2RM, r_P2RM)
        J_wing2RM = (
            wmp["J_s2S"]
            + wmp["m_s"] * D_s
            + wmp["J_v2V"] * rho_air
            + wmp["m_air"] * D_v
        )
        J_p2RM = (pmp["J_p2P"] + pmp["m_p"] * D_p)

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_CP2e = v_RM2e + cross3(omega_b2e, r_CP2RM)
        v_W2CP = v_W2e - v_CP2e

        # FIXME: "magic" indexing established by `self.control_points`
        r_CP2RM_wing = r_CP2RM[:-1]
        r_CP2RM_payload = r_CP2RM[-1]
        v_W2CP_wing = v_W2CP[:-1]
        v_W2CP_payload = v_W2CP[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        try:
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_wing, rho_air, reference_solution,
            )
        except Exception:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            # breakpoint()
            # 1/0
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_wing, rho_air,
            )

        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_s"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2RM_wing, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["r_S2RM"], F_wing_weight)

        # Forces and moments of the payload
        dF_p_aero, dM_p_aero = self.payload.aerodynamics(v_W2CP_payload, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["m_p"] * g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2RM_payload, dF_p_aero).sum(axis=0)
        M_p += cross3(pmp["r_P2RM"], F_p_weight)

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_RM2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum to the forces and moments.

        # Compute the real mass momentums
        J_b2RM = J_wing2RM + J_p2RM  # Real mass inertia matrix about `RM`
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        p_B2e = m_b * v_B2e  # Linear momentum
        h_b2RM = J_b2RM @ omega_b2e + m_b * cross3(r_B2RM, v_RM2e)  # Angular momentum

        # Build the system matrices for the real mass
        A1 = [m_b * np.eye(3), -m_b * crossmat(r_B2RM)]
        A2 = [m_b * crossmat(r_B2RM), J_b2RM]
        A = np.block([A1, A2])
        B1 = (
            F_wing_aero
            + F_wing_weight
            + F_p_aero
            + F_p_weight
            - cross3(omega_b2e, p_B2e)
        )
        B2 = (  # ref: Hughes Eq:13, pg 58 (67)
            M_wing
            + M_p
            - cross3(v_RM2e, p_B2e)
            - cross3(omega_b2e, h_b2RM)
        )

        if self.use_apparent_mass:
            # Extract M_a and J_a2RM from A_a2RM (Barrows Eq:27)
            M_a = wmp["A_a2RM"][:3, :3]  # Apparent mass matrix
            J_a2RM = wmp["A_a2RM"][3:, 3:]  # Apparent angular inertia matrix
            S2 = np.diag([0, 1, 0])  # Selection matrix (Barrows Eq:15)
            S_PC2RC = crossmat(wmp["r_PC2RC"])
            S_RC2RM = crossmat(wmp["r_RC2RM"])
            p_a2e = M_a @ (  # Apparent linear momentum (Barrows Eq:16)
                v_RM2e
                - cross3(wmp["r_RC2RM"], omega_b2e)
                - crossmat(wmp["r_PC2RC"]) @ S2 @ omega_b2e
            )
            h_a2RM = (  # Apparent angular momentum (Barrows Eq:24)
                (S2 @ S_PC2RC + S_RC2RM) @ M_a @ v_RM2e + J_a2RM @ omega_b2e
            )
            A += wmp["A_a2RM"]  # Incorporate the apparent inertia
            B1 += (  # Apparent inertial force (Barrows Eq:61)
                -cross3(omega_b2e, p_a2e)
            )
            B2 += (  # Apparent inertial moment (Barrows Eq:64)
                -cross3(v_RM2e, p_a2e)
                - cross3(omega_b2e, h_a2RM)
                + cross3(v_RM2e, M_a @ v_RM2e)  # Remove the steady-state term
            )

        B = np.r_[B1, B2]

        derivatives = np.linalg.solve(A, B)
        a_RM2e = derivatives[:3]  # In frame F_b
        a_RM2e += cross3(omega_b2e, v_RM2e)  # In frame F_e
        alpha_b2e = derivatives[3:]  # In frames F_b and F_e

        return a_RM2e, alpha_b2e, ref

    def equilibrium_state(
        self,
        delta_a=0,
        delta_b=0,
        rho_air=1.225,
        alpha_0=None,
        theta_0=0,
        v_0=10,
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
                reference_solution=reference_solution
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


class Paraglider6b(Paraglider6a):
    """
    A 6 degrees-of-freedom paraglider model; there is no relative motion
    between the wing and the harness.

    This version uses the body center of mass `B` as the reference point for
    the angular momentum. It does not includes the effects of apparent mass.
    Neglecting apparent mass and using the center of mass means the linear and
    angular momentum are fully decoupled and can be solved independently. The
    system produces `a_B2e` which is then used to compute `a_RM2e`.

    Identical to 6c, except it uses `v_RM2e` for the linear momentum.

    Parameters
    ----------
    wing : ParagliderWing
    payload : Harness
        This uses a `Harness`, but since there is no model for the pilot
        the harness should include the pilot mass.
    """

    def __init__(self, wing, payload):
        self.wing = wing
        self.payload = payload

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        g,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

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
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2RM : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2RM` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2RM. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2RM` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body, in body frd coordinates.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        There are two use cases:
         1. Uniform global wind across the wing (v_W2e.shape == (3,))
         2. Non-uniform global wind across the wing (v_W2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2RM is None:
            # FIXME: needs a design review. The idea was that if `v_W2e` is
            #        given for each individual control point, then require the
            #        values of those control points to ensure they match the
            #        current state of the wing (including the current control
            #        inputs, `delta_a` and `delta_w`, which move the CPs). I've
            #        never liked this design.
            raise ValueError("Control point relative winds require r_CP2RM")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError("Different number of wind and r_CP2RM vectors")
        if r_CP2RM is None:
            r_CP2RM = self.control_points(delta_a, delta_w)

        v_RM2e = np.asarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        # -------------------------------------------------------------------
        # Compute the inertia matrices about the glider cm
        wmp = self.wing.mass_properties(rho_air, delta_a)
        pmp = self.payload.mass_properties(delta_w)
        m_b = wmp["m_s"] + wmp["m_air"] + pmp["m_p"]
        r_B2RM = (  # Center of mass of the body system
            wmp["m_s"] * wmp["r_S2RM"]
            + wmp["m_air"] * wmp["r_V2RM"]
            + pmp["m_p"] * pmp["r_P2RM"]
        ) / m_b
        r_S2B = wmp["r_S2RM"] - r_B2RM  # Displacement of the wing solid mass
        r_V2B = wmp["r_V2RM"] - r_B2RM  # Displacement of the wing enclosed air
        r_P2B = pmp["r_P2RM"] - r_B2RM  # Displacement of the payload mass
        D_s = (r_S2B @ r_S2B) * np.eye(3) - np.outer(r_S2B, r_S2B)
        D_v = (r_V2B @ r_V2B) * np.eye(3) - np.outer(r_V2B, r_V2B)
        D_p = (r_P2B @ r_P2B) * np.eye(3) - np.outer(r_P2B, r_P2B)
        J_wing2B = (
            wmp["J_s2S"]
            + wmp["m_s"] * D_s
            + wmp["J_v2V"] * rho_air
            + wmp["m_air"] * D_v
        )
        J_p2B = (pmp["J_p2P"] + pmp["m_p"] * D_p)

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_CP2e = v_RM2e + cross3(omega_b2e, r_CP2RM)
        v_W2CP = v_W2e - v_CP2e

        # FIXME: "magic" indexing established by `self.control_points`
        r_CP2B_wing = r_CP2RM[:-1] - r_B2RM
        r_CP2B_payload = r_CP2RM[-1] - r_B2RM
        v_W2CP_wing = v_W2CP[:-1]
        v_W2CP_payload = v_W2CP[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        try:
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_wing,
                rho_air,
                reference_solution,
            )
        except Exception:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            # breakpoint()
            # 1/0
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_wing, rho_air,
            )
        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_s"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2B_wing, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["r_S2RM"] - r_B2RM, F_wing_weight)

        # Forces and moments of the payload
        dF_p_aero, dM_p_aero = self.payload.aerodynamics(v_W2CP_payload, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["m_p"] * g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2B_payload, dF_p_aero).sum(axis=0)
        M_p += cross3(pmp["r_P2RM"] - r_B2RM, F_p_weight)

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_RM2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum to the net forces and moments.

        # Compute the real mass inertias
        J_b2B = J_wing2B + J_p2B  # Total inertia matrix about `B`
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        p_b2e = m_b * v_B2e  # Linear momentum
        h_b2B = J_b2B @ omega_b2e  # Angular momentum

        A1 = [m_b * np.eye(3), np.zeros((3, 3))]
        A2 = [np.zeros((3, 3)), J_b2B]
        A = np.block([A1, A2])

        B1 = (
            F_wing_aero
            + F_wing_weight
            + F_p_aero
            + F_p_weight
            - cross3(omega_b2e, p_b2e)
        )
        B2 = M_wing + M_p - np.cross(omega_b2e, h_b2B)
        B = np.r_[B1, B2]

        derivatives = np.linalg.solve(A, B)
        a_B2e = derivatives[:3]  # In frame F_b
        alpha_b2e = derivatives[3:]  # In frames F_b and F_e
        a_RM2e = a_B2e - np.cross(alpha_b2e, r_B2RM)  # In frame F_b
        a_RM2e += cross3(omega_b2e, v_RM2e)  # In frame F_e

        return a_RM2e, alpha_b2e, ref


class Paraglider6c(Paraglider6a):
    """
    A 6 degrees-of-freedom paraglider model; there is no relative motion
    between the wing and the harness.

    This version uses the body center of mass `B` as the reference point for
    the angular momentum. It does not includes the effects of apparent mass.
    Neglecting apparent mass and using the center of mass means the linear and
    angular momentum are fully decoupled and can be solved independently. The
    system produces `a_B2e` which is then used to compute `a_RM2e`.

    Identical to 6b, except it uses `v_B2e` for the linear momentum.

    Parameters
    ----------
    wing : ParagliderWing
    payload : Harness
        This uses a `Harness`, but since there is no model for the pilot
        the harness should include the pilot mass.
    """

    def __init__(self, wing, payload):
        self.wing = wing
        self.payload = payload

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        g,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

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
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2RM : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2RM` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2RM. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2RM` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body, in body frd coordinates.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        There are two use cases:
         1. Uniform global wind across the wing (v_W2e.shape == (3,))
         2. Non-uniform global wind across the wing (v_W2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2RM is None:
            # FIXME: needs a design review. The idea was that if `v_W2e` is
            #        given for each individual control point, then require the
            #        values of those control points to ensure they match the
            #        current state of the wing (including the current control
            #        inputs, `delta_a` and `delta_w`, which move the CPs). I've
            #        never liked this design.
            raise ValueError("Control point relative winds require r_CP2RM")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError("Different number of wind and r_CP2RM vectors")
        if r_CP2RM is None:
            r_CP2RM = self.control_points(delta_a, delta_w)

        v_RM2e = np.asarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        # -------------------------------------------------------------------
        # Compute the inertia matrices about the glider cm
        wmp = self.wing.mass_properties(rho_air, delta_a)
        pmp = self.payload.mass_properties(delta_w)
        m_b = wmp["m_s"] + wmp["m_air"] + pmp["m_p"]
        r_B2RM = (  # Center of mass of the body system
            wmp["m_s"] * wmp["r_S2RM"]
            + wmp["m_air"] * wmp["r_V2RM"]
            + pmp["m_p"] * pmp["r_P2RM"]
        ) / m_b
        r_S2B = wmp["r_S2RM"] - r_B2RM  # Displacement of the wing solid mass
        r_V2B = wmp["r_V2RM"] - r_B2RM  # Displacement of the wing enclosed air
        r_P2B = pmp["r_P2RM"] - r_B2RM  # Displacement of the payload mass
        D_s = (r_S2B @ r_S2B) * np.eye(3) - np.outer(r_S2B, r_S2B)
        D_v = (r_V2B @ r_V2B) * np.eye(3) - np.outer(r_V2B, r_V2B)
        D_p = (r_P2B @ r_P2B) * np.eye(3) - np.outer(r_P2B, r_P2B)
        J_wing2B = (
            wmp["J_s2S"]
            + wmp["m_s"] * D_s
            + wmp["J_v2V"] * rho_air
            + wmp["m_air"] * D_v
        )
        J_p2B = (pmp["J_p2P"] + pmp["m_p"] * D_p)

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_CP2e = v_RM2e + cross3(omega_b2e, r_CP2RM)
        v_W2CP = v_W2e - v_CP2e

        # FIXME: "magic" indexing established by `self.control_points`
        r_CP2B_wing = r_CP2RM[:-1] - r_B2RM
        r_CP2B_payload = r_CP2RM[-1] - r_B2RM
        v_W2CP_wing = v_W2CP[:-1]
        v_W2CP_payload = v_W2CP[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        try:
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_wing, rho_air, reference_solution,
            )
        except Exception:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            # breakpoint()
            # 1/0
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_wing, rho_air,
            )
        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_s"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2B_wing, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["r_S2RM"] - r_B2RM, F_wing_weight)

        # Forces and moments of the payload
        dF_p_aero, dM_p_aero = self.payload.aerodynamics(v_W2CP_payload, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["m_p"] * g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2B_payload, dF_p_aero).sum(axis=0)
        M_p += cross3(pmp["r_P2RM"] - r_B2RM, F_p_weight)

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_RM2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum to the net forces and moments.

        # Compute the real mass inertias
        J_b2B = J_wing2B + J_p2B  # Total inertia matrix about `B`
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        p_b2e = m_b * v_B2e  # Linear momentum
        h_b2B = J_b2B @ omega_b2e  # Angular momentum

        A1 = [m_b * np.eye(3), -m_b * crossmat(r_B2RM)]
        A2 = [np.zeros((3, 3)), J_b2B]
        A = np.block([A1, A2])

        B1 = (
            F_wing_aero
            + F_wing_weight
            + F_p_aero
            + F_p_weight
            - cross3(omega_b2e, p_b2e)
        )
        B2 = M_wing + M_p - np.cross(omega_b2e, h_b2B)
        B = np.r_[B1, B2]

        derivatives = np.linalg.solve(A, B)
        a_RM2e = derivatives[:3]  # In frame F_b
        a_RM2e += cross3(omega_b2e, v_RM2e)  # In frame F_e
        alpha_b2e = derivatives[3:]  # In frames F_b and F_e

        return a_RM2e, alpha_b2e, ref


class Paraglider9a:
    """
    A 9 degrees-of-freedom paraglider model, allowing rotation between the wing
    and the harness, with the connection modelled by spring-damper dynamics.

    This version uses the riser connection midpoint `RM` as the reference point
    for both the body and the payload. It includes the effects of apparent
    mass.

    Parameters
    ----------
    wing : ParagliderWing
    payload : Harness
        This uses a `Harness`, but since there is no model for the pilot
        the harness should include the pilot mass.
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
        wing,
        payload,
        kappa_RM=(0, 0, 0),
        kappa_RM_dot=(0, 0, 0),
        *,
        use_apparent_mass=True,
    ):
        self.wing = wing
        self.payload = payload
        self._kappa_RM = np.asarray(kappa_RM[:])
        self._kappa_RM_dot = np.asarray(kappa_RM_dot[:])
        self.use_apparent_mass = use_apparent_mass

    def control_points(self, Theta_p2b, delta_a=0, delta_w=0):
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
        FIXME: describe
        """
        r_LE2RM = -self.wing.r_RM2LE(delta_a)
        wing_cps = self.wing.control_points(delta_a=delta_a)  # In body frd
        payload_cps = self.payload.control_points(delta_w)  # In payload frd
        C_b2p = orientation.euler_to_dcm(Theta_p2b).T
        return np.vstack((wing_cps + r_LE2RM, (C_b2p @ payload_cps.T).T))

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        omega_p2e,
        Theta_p2b,
        g,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

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
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2RM : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2RM` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2RM. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2RM` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body, in body frd coordinates.
        alpha_p2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the payload, in payload frd coordinates.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        There are two use cases:
         1. Uniform global wind across the wing (v_W2e.shape == (3,))
         2. Non-uniform global wind across the wing (v_W2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2RM is None:
            # FIXME: needs a design review. The idea was that if `v_W2e` is
            #        given for each individual control point, then require the
            #        values of those control points to ensure they match the
            #        current state of the wing (including the current control
            #        inputs, `delta_a` and `delta_w`, which move the CPs). I've
            #        never liked this design.
            raise ValueError("Control point relative winds require r_CP2RM")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError("Different number of wind and r_CP2RM vectors")
        if r_CP2RM is None:
            r_CP2RM = self.control_points(Theta_p2b, delta_a, delta_w)

        v_W2e = np.broadcast_to(v_W2e, r_CP2RM.shape)

        v_RM2e = np.asarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        C_p2b = orientation.euler_to_dcm(Theta_p2b)
        C_b2p = C_p2b.T
        omega_p2b = C_b2p @ omega_p2e - omega_b2e  # In body frd
        omega_b2p = -omega_p2b

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload about `RM`
        wmp = self.wing.mass_properties(rho_air, delta_a)
        m_b = wmp["m_s"] + wmp["m_air"]
        r_B2RM = (  # Center of mass of the body in body frd
            wmp["m_s"] * wmp["r_S2RM"]
            + wmp["m_air"] * wmp["r_V2RM"]
        ) / m_b
        r_S2RM = wmp["r_S2RM"]  # Displacement of the wing solid mass
        r_V2RM = wmp["r_V2RM"]  # Displacement of the wing enclosed air
        D_s = (r_S2RM @ r_S2RM) * np.eye(3) - np.outer(r_S2RM, r_S2RM)
        D_v = (r_V2RM @ r_V2RM) * np.eye(3) - np.outer(r_V2RM, r_V2RM)
        J_b2RM = (
            wmp["J_s2S"]
            + wmp["m_s"] * D_s
            + wmp["J_v2V"] * rho_air
            + wmp["m_air"] * D_v
        )

        pmp = self.payload.mass_properties(delta_w)
        m_p = pmp["m_p"]
        r_P2RM = pmp["r_P2RM"]
        D_p = (r_P2RM @ r_P2RM) * np.eye(3) - np.outer(r_P2RM, r_P2RM)
        J_p2RM = pmp["J_p2P"] + pmp["m_p"] * D_p

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        #
        # Body vectors are in body frd, payload vectors are in payload frd

        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        v_P2e = C_p2b @ v_RM2e + cross3(omega_p2e, r_P2RM)

        # FIXME: "magic" indexing established by `self.control_points`
        r_CP2RM_b = r_CP2RM[:-1]
        r_CP2RM_p = C_p2b @ r_CP2RM[-1]

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2RM_b - r_B2RM)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2RM_p - r_P2RM)

        v_W2CP_b = v_W2e[:-1] - v_CP2e_b
        v_W2CP_p = C_p2b @ v_W2e[-1] - v_CP2e_p

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        try:
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                reference_solution,
            )
        except Exception:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            # breakpoint()
            # 1/0
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_b, rho_air,
            )

        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_s"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2RM_b, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["r_S2RM"], F_wing_weight)

        # Forces and moments of the payload in payload frd
        dF_p_aero, dM_p_aero = self.payload.aerodynamics(v_W2CP_p, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["m_p"] * C_p2b @ g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2RM_p, dF_p_aero).sum(axis=0)
        M_p += cross3(pmp["r_P2RM"], F_p_weight)

        # Moment at the connection point `RM` modeled as a spring+damper system
        M_RM = Theta_p2b * self._kappa_RM + omega_p2b * self._kappa_RM_dot

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_RM2e`,
        # `omega_b2e` (in body frd), `omega_p2e` (in payload frd), and the
        # internal force on the risers, `F_RM` (in body frd).

        # Compute the real mass inertias
        p_b2e = m_b * v_B2e
        p_p2e = m_p * v_P2e
        h_b2RM = m_b * cross3(r_B2RM, v_RM2e) + J_b2RM @ omega_b2e
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

        B1 = (
            F_wing_aero
            + F_wing_weight
            - cross3(omega_b2e, p_b2e)
        )
        B2 = (
            M_wing
            - M_RM
            - cross3(v_RM2e, p_b2e)
            - cross3(omega_b2e, h_b2RM)
        )
        B3 = (
            F_p_aero
            + F_p_weight
            - m_p * C_p2b @ cross3(omega_b2p, v_RM2e)
            - cross3(omega_p2e, p_p2e)
        )
        B4 = (
            M_p
            + C_p2b @ M_RM
            - cross3(C_p2b @ v_RM2e, p_p2e)
            - m_p * cross3(r_P2RM, C_p2b @ cross3(omega_b2p, v_RM2e))
            - cross3(omega_p2e, h_p2RM)
        )
        B = np.r_[B1, B2, B3, B4]

        if self.use_apparent_mass:
            # Extract M_a and J_a2RM from A_a2RM (Barrows Eq:27)
            M_a = wmp["A_a2RM"][:3, :3]  # Apparent mass matrix
            J_a2RM = wmp["A_a2RM"][3:, 3:]  # Apparent angular inertia matrix
            S2 = np.diag([0, 1, 0])  # Selection matrix (Barrows Eq:15)
            S_PC2RC = crossmat(wmp["r_PC2RC"])
            S_RC2RM = crossmat(wmp["r_RC2RM"])
            p_a2e = M_a @ (  # Apparent linear momentum (Barrows Eq:16)
                v_RM2e
                - cross3(wmp["r_RC2RM"], omega_b2e)
                - crossmat(wmp["r_PC2RC"]) @ S2 @ omega_b2e
            )
            h_a2RM = (  # Apparent angular momentum (Barrows Eq:24)
                (S2 @ S_PC2RC + S_RC2RM) @ M_a @ v_RM2e + J_a2RM @ omega_b2e
            )
            A[:6, :6] += wmp["A_a2RM"]  # Incorporate the apparent inertia
            B1 += (  # Apparent inertial force (Barrows Eq:61)
                -cross3(omega_b2e, p_a2e)
            )
            B2 += (  # Apparent inertial moment (Barrows Eq:64)
                -cross3(v_RM2e, p_a2e)
                - cross3(omega_b2e, h_a2RM)
                + cross3(v_RM2e, M_a @ v_RM2e)  # Remove the steady-state term
            )

        x = np.linalg.solve(A, B)
        a_RM2e = x[:3]  # In frame F_b
        a_RM2e += cross3(omega_b2e, v_RM2e)  # In frame F_e
        alpha_b2e = x[3:6]  # In frames F_b and F_e
        alpha_p2e = x[6:9]  # In frames F_p and F_e
        F_RM = x[9:]  # For debugging

        # breakpoint()
        # 1/0

        return a_RM2e, alpha_b2e, alpha_p2e, ref

    def equilibrium_state(
        self,
        delta_a=0,
        delta_b=0,
        rho_air=1.225,
        alpha_0=None,
        theta_0=0,
        v_0=10,
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
                reference_solution=reference_solution
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


class Paraglider9b(Paraglider9a):
    """
    A 9 degrees-of-freedom paraglider model, allowing rotation between the wing
    and the harness, with the connection modelled by spring-damper dynamics.

    This version uses the body center of mass `B` as the reference point for
    the "body" (the wing) and the payload center of mass `P` as the reference
    point for the payload. It does not include the effects of apparent mass.

    Using the centers of mass makes the derivation of the linear system of
    equations more intuitive, but it also makes using the apparent mass a bit
    more work (since `ParagliderWing` returns the apparent inertia matrix using
    `RM` as the reference point, not `B`). I suppose you could (assuming `B`
    lies in the xz-plane as is assumed by Barrows), but I just haven't done it
    yet. This class is mostly for practice and to help catch implementation
    mistakes in `Paraglider9a`.

    Parameters
    ----------
    wing : ParagliderWing
    payload : Harness
        This uses a `Harness`, but since there is no model for the pilot
        the harness should include the pilot mass.
    kappa_RM : array of float, shape (3,), optional
        Spring-damper coefficients for Theta_p2b (force as a linear function
        of angular displacement).
    kappa_RM_dot : array of float, shape (3,), optional
        Spring-damper coefficients for the derivative of Theta_p2b
    """

    def __init__(self, wing, payload, kappa_RM=(0, 0, 0), kappa_RM_dot=(0, 0, 0)):
        self.wing = wing
        self.payload = payload
        self._kappa_RM = np.asarray(kappa_RM[:])
        self._kappa_RM_dot = np.asarray(kappa_RM_dot[:])

    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        omega_p2e,
        Theta_p2b,
        g,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

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
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2RM : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2RM` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2RM. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2RM` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body, in body frd coordinates.
        alpha_p2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the payload, in payload frd coordinates.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        There are two use cases:
         1. Uniform global wind across the wing (v_W2e.shape == (3,))
         2. Non-uniform global wind across the wing (v_W2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2RM is None:
            # FIXME: needs a design review. The idea was that if `v_W2e` is
            #        given for each individual control point, then require the
            #        values of those control points to ensure they match the
            #        current state of the wing (including the current control
            #        inputs, `delta_a` and `delta_w`, which move the CPs). I've
            #        never liked this design.
            raise ValueError("Control point relative winds require r_CP2RM")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError("Different number of wind and r_CP2RM vectors")
        if r_CP2RM is None:
            r_CP2RM = self.control_points(Theta_p2b, delta_a, delta_w)

        v_W2e = np.broadcast_to(v_W2e, r_CP2RM.shape)

        v_RM2e = np.asarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        C_p2b = orientation.euler_to_dcm(Theta_p2b)
        C_b2p = C_p2b.T

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload
        wmp = self.wing.mass_properties(rho_air, delta_a)
        m_b = wmp["m_s"] + wmp["m_air"]
        r_B2RM = (  # Center of mass of the body in body frd
            wmp["m_s"] * wmp["r_S2RM"]
            + wmp["m_air"] * wmp["r_V2RM"]
        ) / m_b
        r_S2B = wmp["r_S2RM"] - r_B2RM  # Displacement of the wing solid mass
        r_V2B = wmp["r_V2RM"] - r_B2RM  # Displacement of the wing enclosed air
        D_s = (r_S2B @ r_S2B) * np.eye(3) - np.outer(r_S2B, r_S2B)
        D_v = (r_V2B @ r_V2B) * np.eye(3) - np.outer(r_V2B, r_V2B)
        J_b2B = (  # Inertia of the body about `B`
            wmp["J_s2S"]
            + wmp["m_s"] * D_s
            + wmp["J_v2V"] * rho_air
            + wmp["m_air"] * D_v
        )

        pmp = self.payload.mass_properties(delta_w)
        m_p = pmp["m_p"]
        r_P2RM = pmp["r_P2RM"]  # Center of mass of the payload in payload frd
        J_p2P = pmp["J_p2P"]  # Inertia of the payload about `P`

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        #
        # Body vectors are in body frd, payload vectors are in payload frd

        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        v_P2e = C_p2b @ v_RM2e + cross3(omega_p2e, r_P2RM)

        # FIXME: "magic" indexing established by `self.control_points`
        r_CP2B_b = r_CP2RM[:-1] - r_B2RM
        r_CP2P_p = C_p2b @ r_CP2RM[-1] - r_P2RM

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2B_b)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2P_p)

        v_W2CP_b = v_W2e[:-1] - v_CP2e_b
        v_W2CP_p = C_p2b @ v_W2e[-1] - v_CP2e_p

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        try:
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                reference_solution,
            )
        except Exception:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            # breakpoint()
            # 1/0
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_b, rho_air,
            )

        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_s"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2B_b, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["r_S2RM"] - r_B2RM, F_wing_weight)

        # Forces and moments of the payload in payload frd
        dF_p_aero, dM_p_aero = self.payload.aerodynamics(v_W2CP_p, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["m_p"] * C_p2b @ g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2P_p, dF_p_aero).sum(axis=0)
        M_p += cross3(pmp["r_P2RM"] - r_P2RM, F_p_weight)

        # Moment at the connection point `RM` modeled as a spring+damper system
        omega_p2b = C_b2p @ omega_p2e - omega_b2e
        M_RM = Theta_p2b * self._kappa_RM + omega_p2b * self._kappa_RM_dot

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_RM2e`,
        # `omega_b2e` (in body frd), `omega_p2e` (in payload frd), and the
        # internal force on the risers, `F_RM` (in body frd).

        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A1 = [m_b * I3, -m_b * crossmat(r_B2RM), Z3, I3]
        A2 = [m_p * C_p2b, Z3, -m_p * crossmat(r_P2RM), -C_p2b]
        A3 = [Z3, J_b2B, Z3, -crossmat(r_B2RM)]
        A4 = [Z3, Z3, J_p2P, crossmat(r_P2RM) @ C_p2b]
        A = np.block([A1, A2, A3, A4])

        B1 = (
            F_wing_aero
            + F_wing_weight
            - m_b * cross3(omega_b2e, v_RM2e)
            - m_b * cross3(omega_b2e, cross3(omega_b2e, r_B2RM))
        )
        B2 = (
            F_p_aero
            + F_p_weight
            - m_p * C_p2b @ cross3(omega_b2e, v_RM2e)
            - m_p * cross3(omega_p2e, cross3(omega_p2e, r_P2RM))
        )
        B3 = M_wing - M_RM - cross3(omega_b2e, J_b2B @ omega_b2e)
        B4 = M_p + C_p2b @ M_RM - cross3(omega_p2e, J_p2P @ omega_p2e)
        B = np.r_[B1, B2, B3, B4]

        x = np.linalg.solve(A, B)
        a_RM2e = x[:3]  # In frame F_b
        a_RM2e += cross3(omega_b2e, v_RM2e)  # In frame F_e
        alpha_b2e = x[3:6]  # In frames F_b and F_e
        alpha_p2e = x[6:9]  # In frames F_p and F_e
        F_RM = x[9:]  # For debugging

        # breakpoint()
        # 1/0

        return a_RM2e, alpha_b2e, alpha_p2e, ref


class Paraglider9c(Paraglider9a):
    """
    A 9 degrees-of-freedom paraglider model, allowing rotation between the wing
    and the harness, with the connection modelled by spring-damper dynamics.

    Similar to Paraglider9a, this version uses the riser connection midpoint
    `RM` as the reference point for both the body and the payload, and includes
    the effects of apparent mass. Unlike Paraglider9a, this model computes
    \dot{omega_p2b} instead of \dot{omega_p2e} and converts.

    Unfortunately it also appears to be broken; at least, it doesn't agree with
    Paraglider9a or Paraglider9b, which are mathematically less complicated so
    I tend to believe them. See the end of `accelerations` for a discussion.

    Also, note that it computes everything in body frd and converts omega_p2e
    back to payload frd at the very end.

    Parameters
    ----------
    wing : ParagliderWing
    payload : Harness
        This uses a `Harness`, but since there is no model for the pilot
        the harness should include the pilot mass.
    """
    def accelerations(
        self,
        v_RM2e,
        omega_b2e,
        omega_p2e,
        Theta_p2b,
        g,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        delta_w=0,
        rho_air=1.225,
        v_W2e=(0, 0, 0),
        r_CP2RM=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

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
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2RM : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2RM` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2RM. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2RM` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_RM2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `RM` in body frd coordinates.
        alpha_b2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the body, in body frd coordinates.
        alpha_p2e : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the payload, in payload frd coordinates.
        solution : dictionary
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        There are two use cases:
         1. Uniform global wind across the wing (v_W2e.shape == (3,))
         2. Non-uniform global wind across the wing (v_W2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2RM is None:
            # FIXME: needs a design review. The idea was that if `v_W2e` is
            #        given for each individual control point, then require the
            #        values of those control points to ensure they match the
            #        current state of the wing (including the current control
            #        inputs, `delta_a` and `delta_w`, which move the CPs). I've
            #        never liked this design.
            raise ValueError("Control point relative winds require r_CP2RM")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2RM.shape[0]:
            raise ValueError("Different number of wind and r_CP2RM vectors")
        if r_CP2RM is None:
            r_CP2RM = self.control_points(Theta_p2b, delta_a, delta_w)

        v_W2e = np.broadcast_to(v_W2e, r_CP2RM.shape)

        v_RM2e = np.asarray(v_RM2e)
        if v_RM2e.shape != (3,):
            raise ValueError("v_RM2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        C_p2b = orientation.euler_to_dcm(Theta_p2b)
        C_b2p = C_p2b.T
        omega_p2e = C_b2p @ omega_p2e  # Dynamics9a uses `p`, this model uses `b`

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload
        wmp = self.wing.mass_properties(rho_air, delta_a)
        m_b = wmp["m_s"] + wmp["m_air"]
        r_B2RM = (  # Center of mass of the body in body frd
            wmp["m_s"] * wmp["r_S2RM"]
            + wmp["m_air"] * wmp["r_V2RM"]
        ) / m_b
        r_S2RM = wmp["r_S2RM"]  # Displacement of the wing solid mass
        r_V2RM = wmp["r_V2RM"]  # Displacement of the wing enclosed air
        D_s = (r_S2RM @ r_S2RM) * np.eye(3) - np.outer(r_S2RM, r_S2RM)
        D_v = (r_V2RM @ r_V2RM) * np.eye(3) - np.outer(r_V2RM, r_V2RM)
        J_b2RM = (
            wmp["J_s2S"]
            + wmp["m_s"] * D_s
            + wmp["J_v2V"] * rho_air
            + wmp["m_air"] * D_v
        )

        pmp = self.payload.mass_properties(delta_w)
        m_p = pmp["m_p"]
        r_P2RM = pmp["r_P2RM"]  # Center of mass of the payload in payload frd
        D_p = (r_P2RM @ r_P2RM) * np.eye(3) - np.outer(r_P2RM, r_P2RM)
        J_p2RM = pmp["J_p2P"] + pmp["m_p"] * D_p  # In payload frd

        r_P2RM = C_b2p @ r_P2RM  # In body frd
        J_p2RM = C_b2p @ J_p2RM @ C_p2b  # In body frd

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        #
        # All vectors are in body frd

        omega_p2b = omega_p2e - omega_b2e
        omega_b2p = -omega_p2b
        v_B2e = v_RM2e + cross3(omega_b2e, r_B2RM)
        v_P2e = v_RM2e + cross3(omega_p2e, r_P2RM)

        # FIXME: "magic" indexing established by `self.control_points`
        r_CP2RM_b = r_CP2RM[:-1]
        r_CP2RM_p = r_CP2RM[-1]

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2RM_b - r_B2RM)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2RM_p - r_P2RM)

        v_W2CP_b = v_W2e[:-1] - v_CP2e_b
        v_W2CP_p = v_W2e[-1] - v_CP2e_p

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        try:
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a,
                delta_bl,
                delta_br,
                v_W2CP_b,
                rho_air,
                reference_solution,
            )
        except Exception:
            # Maybe it can't recover once Gamma is jacked?
            print("\nBonk! Retrying with the default reference solution")
            # breakpoint()
            # 1/0
            dF_wing_aero, dM_wing_aero, ref = self.wing.aerodynamics(
                delta_a, delta_bl, delta_br, v_W2CP_b, rho_air,
            )

        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_s"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2RM_b, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["r_S2RM"], F_wing_weight)

        # Forces and moments of the payload in payload frd
        dF_p_aero, dM_p_aero = self.payload.aerodynamics(C_p2b @ v_W2CP_p, rho_air)
        dF_p_aero = np.atleast_2d(C_b2p @ dF_p_aero)
        dM_p_aero = np.atleast_2d(C_b2p @ dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["m_p"] * g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2RM_p, dF_p_aero).sum(axis=0)
        M_p += cross3(r_P2RM, F_p_weight)

        # Moment at the connection point `RM` modeled as a spring+damper system
        M_RM = Theta_p2b * self._kappa_RM + omega_p2b * self._kappa_RM_dot

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_RM2e^b`,
        # `omega_b2e^b`, `omega_p2e^p`, and the internal force on the risers,
        # `F_RM^b`. All derivatives are from the body frame.

        # Compute the real mass inertias
        p_b2e = m_b * v_B2e
        p_p2e = m_p * v_P2e
        h_b2RM = m_b * cross3(r_B2RM, v_RM2e) + J_b2RM @ omega_b2e
        h_p2RM = m_p * cross3(r_P2RM, v_RM2e) + J_p2RM @ omega_p2e

        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A1 = [m_b * I3, -m_b * crossmat(r_B2RM), Z3, I3]
        A2 = [m_b * crossmat(r_B2RM), J_b2RM, Z3, Z3]
        A3 = [m_p * I3, -m_p * crossmat(r_P2RM), -m_p * crossmat(r_P2RM), -I3]
        A4 = [m_p * crossmat(r_P2RM), J_p2RM, J_p2RM, Z3]
        A = np.block([A1, A2, A3, A4])

        B1 = (
            F_wing_aero
            + F_wing_weight
            - cross3(omega_b2e, p_b2e)
        )
        B2 = (
            M_wing
            - M_RM
            - cross3(v_RM2e, p_b2e)
            - cross3(omega_b2e, h_b2RM)
        )
        B3 = (
            F_p_aero
            + F_p_weight
            - m_p * cross3(omega_b2p, v_RM2e)
            - m_p * cross3(cross3(omega_b2p, omega_b2e), r_P2RM)
            # - m_p * cross3(cross3(omega_b2p, omega_p2e), r_P2RM)  # equivalent
            - cross3(omega_p2e, p_p2e)
        )
        B4 = (
            M_p
            + M_RM
            - cross3(v_RM2e, p_p2e)
            - m_p * cross3(r_P2RM, cross3(omega_b2p, v_RM2e))
            - cross3(omega_b2p, J_p2RM @ omega_p2e)
            - cross3(omega_p2e, h_p2RM)
        )
        B = np.r_[B1, B2, B3, B4]

        if self.use_apparent_mass:
            # Extract M_a and J_a2RM from A_a2RM (Barrows Eq:27)
            M_a = wmp["A_a2RM"][:3, :3]  # Apparent mass matrix
            J_a2RM = wmp["A_a2RM"][3:, 3:]  # Apparent angular inertia matrix
            S2 = np.diag([0, 1, 0])  # Selection matrix (Barrows Eq:15)
            S_PC2RC = crossmat(wmp["r_PC2RC"])
            S_RC2RM = crossmat(wmp["r_RC2RM"])
            p_a2e = M_a @ (  # Apparent linear momentum (Barrows Eq:16)
                v_RM2e
                - cross3(wmp["r_RC2RM"], omega_b2e)
                - crossmat(wmp["r_PC2RC"]) @ S2 @ omega_b2e
            )
            h_a2RM = (  # Apparent angular momentum (Barrows Eq:24)
                (S2 @ S_PC2RC + S_RC2RM) @ M_a @ v_RM2e + J_a2RM @ omega_b2e
            )
            A[:6, :6] += wmp["A_a2RM"]  # Incorporate the apparent inertia
            B1 += (  # Apparent inertial force (Barrows Eq:61)
                -cross3(omega_b2e, p_a2e)
            )
            B2 += (  # Apparent inertial moment (Barrows Eq:64)
                -cross3(v_RM2e, p_a2e)
                - cross3(omega_b2e, h_a2RM)
                + cross3(v_RM2e, M_a @ v_RM2e)  # Remove the steady-state term
            )

        x = np.linalg.solve(A, B)
        a_RM2e = x[:3]  # In frame F_b
        a_RM2e += cross3(omega_b2e, v_RM2e)  # In frame F_e
        alpha_b2e = x[3:6]  # In frames F_b and F_e
        alpha_p2b = x[6:9]  # In frames F_b and F_p
        F_RM = x[9:]  # For debugging

        # Dynamics9a expects `^p dot{omega}_{p/e}^p`
        alpha_p2e = alpha_p2b + alpha_b2e + cross3(omega_b2e, omega_p2b)
        alpha_p2e = C_p2b @ alpha_p2e  # In frames F_p and F_e

        # breakpoint()
        # 1/0

        return a_RM2e, alpha_b2e, alpha_p2e, ref
