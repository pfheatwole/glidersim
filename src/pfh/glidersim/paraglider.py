"""FIXME: add module docstring."""

from IPython import embed

import numpy as np

from pfh.glidersim import quaternion
from pfh.glidersim.util import cross3

import scipy.integrate


class Paraglider6a:
    """
    A 6 DoF paraglider model; there is no relative motion between the wing and
    the harness.
    """

    def __init__(self, wing, payload):
        """
        Instantiate a Paraglider from given wing and harness.

        Parameters
        ----------
        wing : ParagliderWing
        payload : Harness
            This uses a `Harness`, but since there is no model for the pilot
            the harness should include the pilot mass.
        """
        self.wing = wing
        self.payload = payload

    def control_points(self, delta_a=0):
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

        Returns
        -------
        FIXME: describe
        """
        wing_cps = self.wing.control_points(delta_a=delta_a)
        payload_cps = self.payload.control_points()
        return np.vstack((wing_cps, payload_cps))

    def accelerations(
        self,
        v_R2e,
        omega_b2e,
        g,
        rho_air,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        v_W2e=None,
        r_CP2R=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

        Parameters
        ----------
        v_R2e : array of float, shape (3,) [m/s]
            Translational velocity of `R` in body frd coordinates, where `R` is
            the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        rho_air : float [kg/m^3]
            The ambient air density
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s]
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2R : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2R` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2R. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2R` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_R2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `R` in body frd coordinates.
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
        # FIXME: design review names. `w` is overloaded ("wind" and "wing")
        if v_W2e is None:
            v_W2e = np.array([0, 0, 0])
        else:
            v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2R is None:
            # FIXME: needs a design review. Ensure that if `v_W2e` and `r_CP2R`
            #        were computed using the same `delta_a`, if `v_W2e` was
            #        computed for the individual control points.
            raise ValueError("Control point relative winds require r_CP2R")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2R.shape[0]:
            raise ValueError("Different number of wind and r_CP2R vectors")
        if r_CP2R is None:
            r_CP2R = self.control_points(delta_a)

        v_R2e = np.asarray(v_R2e)
        if v_R2e.shape != (3,):
            raise ValueError("v_R2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        # -------------------------------------------------------------------
        # Compute the inertia matrices about the glider cm
        wmp = self.wing.mass_properties(rho_air, delta_a)
        pmp = self.payload.mass_properties()
        m_B = wmp["m_solid"] + wmp["m_air"] + pmp["mass"]
        r_B2R = (  # Center of mass of the body system
            wmp["m_solid"] * wmp["cm_solid"]
            + wmp["m_air"] * wmp["cm_air"]
            + pmp["mass"] * pmp["cm"]
        ) / m_B
        r_wsm2B = wmp["cm_solid"] - r_B2R  # Displacement of the wing solid mass
        r_wea2B = wmp["cm_air"] - r_B2R  # Displacement of the wing enclosed air
        r_P2B = pmp["cm"] - r_B2R  # Displacement of the payload mass
        Dwsm = (r_wsm2B @ r_wsm2B) * np.eye(3) - np.outer(r_wsm2B, r_wsm2B)
        Dwea = (r_wea2B @ r_wea2B) * np.eye(3) - np.outer(r_wea2B, r_wea2B)
        Dp = (r_P2B @ r_P2B) * np.eye(3) - np.outer(r_P2B, r_P2B)
        J_wing = (
            wmp["J_solid"]
            + wmp["m_solid"] * Dwsm
            + wmp["J_air"]
            + wmp["m_air"] * Dwea
        )
        J_p = (pmp["J"] + pmp["mass"] * Dp)

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        v_B2e = v_R2e + cross3(omega_b2e, r_B2R)
        v_CP2e = v_R2e + cross3(omega_b2e, r_CP2R)
        v_W2b = v_W2e - v_CP2e

        # FIXME: "magic" layout of array contents
        r_CP2B_wing = r_CP2R[:-1] - r_B2R
        r_CP2B_payload = r_CP2R[-1] - r_B2R
        v_W2b_wing = v_W2b[:-1]
        v_W2b_payload = v_W2b[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        dF_wing_aero, dM_wing_aero, ref = self.wing.forces_and_moments(
            delta_bl, delta_br, v_W2b_wing, rho_air, reference_solution,
        )
        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_solid"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2B_wing, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["cm_solid"] - r_B2R, F_wing_weight)

        # Forces and moments of the payload
        dF_p_aero, dM_p_aero = self.payload.forces_and_moments(v_W2b_payload, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["mass"] * g
        M_p = dM_p_aero.sum(axis=0)
        M_p += cross3(r_CP2B_payload, dF_p_aero).sum(axis=0)
        M_p += cross3(pmp["cm"] - r_B2R, F_p_weight)

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{v_R2e} and \dot{omega_b2e}
        #
        # Builds a system of equations by equating derivatives of translational
        # and angular momentum against the forces and moments.

        J = J_wing + J_p  # Total moment of inertia matrix about the glider cm

        A1 = [m_B * np.eye(3), -m_B * quaternion.skew(r_B2R)]
        A2 = [m_B * quaternion.skew(r_B2R), J]
        A = np.block([A1, A2])

        B1 = (
            F_wing_aero
            + F_wing_weight
            + F_p_aero
            + F_p_weight
            - m_B * cross3(omega_b2e, v_R2e)
            - m_B * cross3(omega_b2e, cross3(omega_b2e, r_B2R))
        )
        B2 = M_wing + M_p - np.cross(omega_b2e, J @ omega_b2e)
        B = np.r_[B1, B2]

        derivatives = np.linalg.solve(A, B)
        a_R2e, alpha_b2e = derivatives[:3], derivatives[3:]

        return a_R2e, alpha_b2e, ref

    def equilibrium_glide(
        self,
        delta_a,
        delta_b,
        v_eq_proposal,
        rho_air,
        N_iter=2,
        reference_solution=None,
    ):
        r"""
        Steady-state angle of attack, pitch angle, and airspeed.

        Parameters
        ----------
        delta_a : float [percentage]
            Percentage of accelerator application
        delta_b : float [percentage]
            Percentage of symmetric brake application
        v_eq_proposal : float [m/s]
            A rough guess for the equilibrium airspeed. This is required to put
            the Reynolds numbers into the proper range.
        rho_air : float [kg/m^3]
            Air density.
        N_iter : integer, optional
            Number of iterations to account for the fact that the Reynolds
            numbers (and thus the coefficients) vary with the solution for
            `v_eq`. If `v_eq_proposal` is anywhere close to accurate, then one
            or two iterations are usually sufficient. Default: 2
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        theta_eq : float [radians]
            Steady-state pitch angle
        v_eq : float [m/s]
            Steady-state airspeed
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        Calculating :math:`v_eq` takes advantage of the fact that all the
        aerodynamic forces are proportional to :math:`v^2`. Thus, by
        normalizing the forces to :math:`v = 1`, the following equation can be
        solved for :math:`v_eq` directly using:

        .. math::

            v_{eq}^2 \cdot \Sigma F_{z,aero} + m_p g \cdot \text{sin} \left( \theta \right)

        where :math:`m_p` is the mass of the payload.
        """

        v_eq = v_eq_proposal  # The initial guess
        solution = reference_solution  # Approximate solution, if available
        m_p = self.payload.mass_properties()["mass"]

        for _ in range(N_iter):
            alpha_eq = self.wing.equilibrium_alpha(
                delta_a, delta_b, v_eq, rho_air, solution,
            )
            v_W2b = -v_eq * np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
            dF_wing, dM_wing, solution = self.wing.forces_and_moments(
                delta_b, delta_b, v_W2b, rho_air, solution,
            )
            dF_p, dM_p = self.payload.forces_and_moments(v_W2b, rho_air)
            F = dF_wing.sum(axis=0) + np.atleast_2d(dF_p).sum(axis=0)
            F /= v_eq ** 2  # The equation for `v_eq` assumes `|v| == 1`

            theta_eq = np.arctan2(F[0], -F[2])

            # FIXME: neglects the weight of the wing
            weight_z = 9.8 * m_p * np.cos(theta_eq)
            v_eq = np.sqrt(-weight_z / F[2])

        return alpha_eq, theta_eq, v_eq, solution

    def equilibrium_glide2(
        self,
        delta_a,
        delta_b,
        alpha_0,
        theta_0,
        v_0,
        rho_air,
        reference_solution=None,
    ):
        """
        Compute the equilibrium state through simulation.

        Unlike `equilibrium_glide`, this uses the full model dynamics. The
        other method is very fast, but ignores things like the weight of the
        wing.

        Parameters
        ----------
        delta_a : float [percentage]
            The percentage of accelerator.
        delta_b : float [percentage]
            The percentage of symmetric brake.
        alpha_0 : float [rad], optional
            The initial proposal for angle of attack.
        theta_0 : float [rad], optional
            The initial proposal for glider pitch angle.
        v_0 : float [m/s], optional
            The initial proposal for glider airspeed.
        rho_air : float [kg/m^3]
            Air density.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        theta_eq : float [radians]
            Steady-state pitch angle
        v_eq : float [m/s]
            Steady-state airspeed
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        state_dtype = [
            ("q_b2e", float, (4,)),  # Orientation quaternion, body/earth
            ("v_R2e", float, (3,)),  # Translational velocity in body frd
            ("omega_b2e", float, (3,)),  # Angular velocity in body frd
        ]

        def dynamics(t, state, kwargs):
            x = state.view(state_dtype)[0]
            a_frd, alpha_frd, ref = self.accelerations(
                x["v_R2e"],
                x["omega_b2e"],
                quaternion.apply_quaternion_rotation(x["q_b2e"], [0, 0, 9.8]),
                **kwargs,
            )
            P, Q, R = x["omega_b2e"]
            # fmt: off
            Omega = np.array([
                [0, -P, -Q, -R],
                [P,  0,  R, -Q],
                [Q, -R,  0,  P],
                [R,  Q, -P,  0]])
            # fmt: on
            q_dot = 0.5 * Omega @ x["q_b2e"]
            x_dot = np.empty(1, state_dtype)
            x_dot["q_b2e"] = q_dot
            x_dot["v_R2e"] = a_frd
            x_dot["omega_b2e"] = alpha_frd
            kwargs["reference_solution"] = ref
            return x_dot.view(float)  # The integrator expects a flat array

        state = np.empty(1, state_dtype)
        state["q_b2e"] = quaternion.euler_to_quaternion([0, theta_0, 0])
        state["v_R2e"] = v_0 * np.array([np.cos(alpha_0), 0, np.sin(alpha_0)])
        state["omega_b2e"] = [0, 0, 0]

        dynamics_kwargs = {
            "delta_a": delta_a,
            "delta_bl": delta_b,
            "delta_br": delta_b,
            "rho_air": rho_air,
            "reference_solution": reference_solution,
        }

        solver = scipy.integrate.ode(dynamics)
        solver.set_integrator("dopri5", rtol=1e-5, max_step=0.1)
        solver.set_f_params(dynamics_kwargs)

        while True:
            state["q_b2e"] /= np.sqrt((x["q_b2e"] ** 2).sum())
            solver.set_initial_value(state.view(float))
            state = solver.integrate(1).view(state_dtype)
            state["omega_b2e"] = [0, 0, 0]  # Zero every step to avoid oscillations
            a_frd, alpha_frd, _ = self.accelerations(
                state["v_R2e"][0],
                state["omega_b2e"][0],
                quaternion.apply_quaternion_rotation(state["q_b2e"][0], [0, 0, 9.8]),
                rho_air=rho_air,
                delta_a=delta_a,
                delta_bl=delta_b,
                delta_br=delta_b,
                reference_solution=dynamics_kwargs["reference_solution"],
            )

            # FIXME: this test doesn't guarantee equilibria
            if any(abs(a_frd) > 0.001) or any(abs(alpha_frd) > 0.001):
                continue
            else:
                state = state[0]
                break

        alpha_eq = np.arctan2(*state["v_R2e"][[2, 0]])
        theta_eq = quaternion.quaternion_to_euler(state["q_b2e"])[1]
        v_eq = np.linalg.norm(state["v_R2e"])

        return alpha_eq, theta_eq, v_eq, dynamics_kwargs["reference_solution"]


class Paraglider9a:
    """
    A 9 DoF paraglider model, allowing rotation between the wing and the
    harness, with the connection modelled by spring-damper dynamics.
    """

    def __init__(self, wing, payload):
        """
        Instantiate a Paraglider from given wing and harness.

        Parameters
        ----------
        wing : ParagliderWing
        payload : Harness
            This uses a `Harness`, but since there is no model for the pilot
            the harness should include the pilot mass.
        """
        self.wing = wing
        self.payload = payload

    def control_points(self, Theta_p, delta_a=0):
        """
        Compute the reference points for the composite Paraglider system.

        All the components of the Paraglider that experience aerodynamic forces
        need their relative wind vectors. Each component is responsible for
        creating a list of the coordinates where they need the value of the
        wind. This function then transforms them into body coordinates.

        Parameters
        ----------
        Theta_p : array of float, shape (3,) [radians]
            The [phi, theta, gamma] of a yaw-pitch-roll sequence that encodes
            the relative orientation of the payload with respect to the body.
        delta_a : float [percentage]
            The fraction of maximum accelerator

        Returns
        -------
        FIXME: describe
        """
        wing_cps = self.wing.control_points(delta_a=delta_a)  # In body frd
        payload_cps = self.payload.control_points()  # In payload frd
        C_b2p = quaternion.euler_to_dcm(Theta_p).T
        return np.vstack((wing_cps, C_b2p @ payload_cps))

    def accelerations(
        self,
        v_R2e,
        omega_b2e,
        omega_p2e,
        Theta_p,
        g,
        rho_air,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        v_W2e=None,
        r_CP2R=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: the input sanitation is messy
        FIXME: review the docstring

        Parameters
        ----------
        v_R2e : array of float, shape (3,) [m/s]
            Translational velocity of `R` in body frd coordinates, where `R` is
            the midpoint between the two riser connection points.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.
        omega_p2e : array of float, shape (3,) [rad/s]
            Angular velocity of the payload, in payload frd coordinates.
        Theta_p : array of float, shape (3,) [radians]
            The [phi, theta, gamma] of a yaw-pitch-roll sequence that encodes
            the relative orientation of the payload with respect to the body.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        rho_air : float [kg/m^3]
            The ambient air density
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        v_W2e : ndarray of float, shape (3,) or (K,3) [m/s]
            The wind relative to the earth, in body frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        r_CP2R : ndarray of float, shape (K,3) [m] (optional)
            Position vectors of the control points, in body frd coordinates.
            These are optional if the wind field is uniform, but for
            non-uniform wind fields the simulator used these coordinates to
            determine the wind vectors at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really, why
            do I even have both `r_CP2R` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the r_CP2R. Using `delta_a`
            alone would be the more intuitive, but would incur extra
            computation time for finding the control points; the only point of
            `r_CP2R` is to avoid recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_R2e : array of float, shape (3,) [m/s^2]
            Translational acceleration of `R` in body frd coordinates.
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
        if v_W2e is None:
            v_W2e = np.array([0, 0, 0])
        else:
            v_W2e = np.asarray(v_W2e)
        if v_W2e.ndim > 1 and r_CP2R is None:
            # FIXME: needs a design review. Ensure that if `v_W2e` and `r_CP2R`
            #        were computed using the same `delta_a`, if `v_W2e` was
            #        computed for the individual control points.
            raise ValueError("Control point relative winds require r_CP2R")
        if v_W2e.ndim > 1 and v_W2e.shape[0] != r_CP2R.shape[0]:
            raise ValueError("Different number of wind and r_CP2R vectors")
        if r_CP2R is None:
            r_CP2R = self.control_points(Theta_p, delta_a)

        v_W2e = np.broadcast_to(v_W2e, r_CP2R.shape)

        v_R2e = np.asarray(v_R2e)
        if v_R2e.shape != (3,):
            raise ValueError("v_R2e must be a 3-vector velocity of the body cm")  # FIXME: awkward phrasing

        C_p2b = quaternion.euler_to_dcm(Theta_p)

        # -------------------------------------------------------------------
        # Compute the inertia properties of the body and payload
        wmp = self.wing.mass_properties(rho_air, delta_a)
        m_b = wmp["m_solid"] + wmp["m_air"]
        r_B2R = (  # Center of mass of the body in body frd
            wmp["m_solid"] * wmp["cm_solid"]
            + wmp["m_air"] * wmp["cm_air"]
        ) / m_b
        r_wsm2B = wmp["cm_solid"] - r_B2R  # Displacement of the wing solid mass
        r_wea2B = wmp["cm_air"] - r_B2R  # Displacement of the wing enclosed air
        Dwsm = (r_wsm2B @ r_wsm2B) * np.eye(3) - np.outer(r_wsm2B, r_wsm2B)
        Dwea = (r_wea2B @ r_wea2B) * np.eye(3) - np.outer(r_wea2B, r_wea2B)
        J_b = (  # Inertia of the body about `B`
            wmp["J_solid"]
            + wmp["m_solid"] * Dwsm
            + wmp["J_air"]
            + wmp["m_air"] * Dwea
        )

        pmp = self.payload.mass_properties()
        m_p = pmp["mass"]
        r_P2R = pmp["cm"]  # Center of mass of the payload in payload frd
        J_p = pmp["J"]  # Inertia of the payload about `P`

        # -------------------------------------------------------------------
        # Compute the relative wind vectors for each control point.
        #
        # Body vectors are in body frd, payload vectors are in payload frd

        v_B2e = v_R2e + cross3(omega_b2e, r_B2R)
        v_P2e = C_p2b @ v_R2e + cross3(omega_p2e, r_P2R)

        # FIXME: "magic" layout of array contents
        r_CP2B = r_CP2R[:-1] - r_B2R
        r_CP2P = C_p2b @ r_CP2R[-1] - r_P2R

        v_CP2e_b = v_B2e + cross3(omega_b2e, r_CP2B)
        v_CP2e_p = v_P2e + cross3(omega_p2e, r_CP2P)

        v_W2b = v_W2e[:-1] - v_CP2e_b  # Relative wind to the body control points
        v_W2p = C_p2b @ v_W2e[-1] - v_CP2e_p  # Relative wind to the payload control points

        # -------------------------------------------------------------------
        # Forces and moments of the wing in body frd
        dF_wing_aero, dM_wing_aero, ref = self.wing.forces_and_moments(
            delta_bl, delta_br, v_W2b, rho_air, reference_solution,
        )
        F_wing_aero = dF_wing_aero.sum(axis=0)
        F_wing_weight = wmp["m_solid"] * g
        M_wing = dM_wing_aero.sum(axis=0)
        M_wing += cross3(r_CP2B, dF_wing_aero).sum(axis=0)
        M_wing += cross3(wmp["cm_solid"] - r_B2R, F_wing_weight)

        # Forces and moments of the payload in payload frd
        dF_p_aero, dM_p_aero = self.payload.forces_and_moments(v_W2p, rho_air)
        dF_p_aero = np.atleast_2d(dF_p_aero)
        dM_p_aero = np.atleast_2d(dM_p_aero)
        F_p_aero = dF_p_aero.sum(axis=0)
        F_p_weight = pmp["mass"] * C_p2b @ g
        M_p = dM_p_aero.sum(axis=0)                   # FIXME: verify: should be zero
        M_p += cross3(r_CP2P, dF_p_aero).sum(axis=0)  # FIXME: verify: should be zero
        M_p += cross3(pmp["cm"] - r_P2R, F_p_weight)  # FIXME: verify: should be zero

        M_R = np.zeros(3)  # FIXME: implement proper spring+damper dynamics
        omega_p2b = omega_p2e - C_p2b @ omega_b2e
        M_R[0] += -3.0 * Theta_p[0]  # Roll restoring force
        M_R[1] += -0.0 * Theta_p[1]  # Pitch restoring force
        M_R[2] += -3.0 * Theta_p[2]  # Yaw restoring force
        M_R[0] += -1.0 * omega_p2b[0]  # Roll dampening
        M_R[1] += -1.0 * omega_p2b[1]  # Pitch dampening
        M_R[2] += -1.0 * omega_p2b[2]  # Yaw dampening

        # ------------------------------------------------------------------
        # Build a system of equations by equating the time derivatives of the
        # translation and angular momentum (with respect to the Earth) of the
        # body and payload to the forces and moments on the body and payload.
        #
        # The four unknown vectors are the time derivatives of `v_R2e`,
        # `omega_b2e` (in body frd), `omega_p2e` (in payload frd), and the
        # internal force on the risers, `F_R` (in body frd).

        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A1 = [m_b * I3, m_b * quaternion.skew(-r_B2R), Z3, I3]
        A2 = [m_p * C_p2b, Z3, m_p * quaternion.skew(-r_P2R), -C_p2b]
        A3 = [Z3, J_b, Z3, quaternion.skew(-r_B2R)]
        A4 = [Z3, Z3, J_p, quaternion.skew(r_P2R) @ C_p2b]
        A = np.block([A1, A2, A3, A4])

        B1 = (
            F_wing_aero
            + F_wing_weight
            - m_b * cross3(omega_b2e, v_R2e)
            + m_b * cross3(omega_b2e, cross3(omega_b2e, -r_B2R))
        )
        B2 = (
            F_p_aero
            + F_p_weight
            - m_p * C_p2b @ cross3(omega_b2e, v_R2e)
            + m_p * cross3(omega_p2e, cross3(omega_p2e, -r_P2R))
        )
        B3 = M_wing - M_R - cross3(omega_b2e, J_b @ omega_b2e)
        B4 = M_p + C_p2b @ M_R - cross3(omega_p2e, J_p @ omega_p2e)
        B = np.r_[B1, B2, B3, B4]

        x = np.linalg.solve(A, B)
        a_R2e = x[:3]
        alpha_b2e = x[3:6]
        alpha_p2e = x[6:9]
        F_R = x[9:]  # For debugging

        return a_R2e, alpha_b2e, alpha_p2e, ref

    def equilibrium_glide(
        self,
        delta_a,
        delta_b,
        v_eq_proposal,
        rho_air,
        N_iter=2,
        reference_solution=None,
    ):
        r"""
        Steady-state angle of attack, pitch angle, and airspeed.

        Parameters
        ----------
        delta_a : float [percentage]
            Percentage of accelerator application
        delta_b : float [percentage]
            Percentage of symmetric brake application
        v_eq_proposal : float [m/s]
            A rough guess for the equilibrium airspeed. This is required to put
            the Reynolds numbers into the proper range.
        rho_air : float [kg/m^3]
            Air density.
        N_iter : integer, optional
            Number of iterations to account for the fact that the Reynolds
            numbers (and thus the coefficients) vary with the solution for
            `v_eq`. If `v_eq_proposal` is anywhere close to accurate, then one
            or two iterations are usually sufficient. Default: 2
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        theta_eq : float [radians]
            Steady-state pitch angle
        v_eq : float [m/s]
            Steady-state airspeed
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        Calculating :math:`v_eq` takes advantage of the fact that all the
        aerodynamic forces are proportional to :math:`v^2`. Thus, by
        normalizing the forces to :math:`v = 1`, the following equation can be
        solved for :math:`v_eq` directly using:

        .. math::

            v_{eq}^2 \cdot \Sigma F_{z,aero} + m_p g \cdot \text{sin} \left( \theta \right)

        where :math:`m_p` is the mass of the payload.
        """

        v_eq = v_eq_proposal  # The initial guess
        solution = reference_solution  # Approximate solution, if available
        m_p = self.payload.mass_properties()["mass"]

        for _ in range(N_iter):
            alpha_eq = self.wing.equilibrium_alpha(
                delta_a, delta_b, v_eq, rho_air, solution,
            )
            v_W2b = -v_eq * np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
            dF_wing, dM_wing, solution = self.wing.forces_and_moments(
                delta_b, delta_b, v_W2b, rho_air, solution,
            )
            dF_p, dM_p = self.payload.forces_and_moments(v_W2b, rho_air)
            F = dF_wing.sum(axis=0) + np.atleast_2d(dF_p).sum(axis=0)
            F /= v_eq ** 2  # The equation for `v_eq` assumes `|v| == 1`

            theta_eq = np.arctan2(F[0], -F[2])

            # FIXME: neglects the weight of the wing
            weight_z = 9.8 * m_p * np.cos(theta_eq)
            v_eq = np.sqrt(-weight_z / F[2])

        return alpha_eq, theta_eq, v_eq, solution

    def equilibrium_glide2(
        self,
        delta_a,
        delta_b,
        alpha_0,
        theta_0,
        v_0,
        rho_air,
        reference_solution=None,
    ):
        """
        Compute the equilibrium state through simulation.

        Unlike `equilibrium_glide`, this uses the full model dynamics. The
        other method is very fast, but ignores things like the weight of the
        wing.

        Parameters
        ----------
        delta_a : float [percentage]
            The percentage of accelerator.
        delta_b : float [percentage]
            The percentage of symmetric brake.
        alpha_0 : float [rad], optional
            The initial proposal for angle of attack.
        theta_0 : float [rad], optional
            The initial proposal for glider pitch angle.
        v_0 : float [m/s], optional
            The initial proposal for glider airspeed.
        rho_air : float [kg/m^3]
            Air density.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        theta_eq : float [radians]
            Steady-state pitch angle
        v_eq : float [m/s]
            Steady-state airspeed
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        state_dtype = [
            ("q_b2e", float, (4,)),  # Orientation: body/earth
            ("q_p2b", float, (4,)),  # Orientation: payload/body
            ("omega_b2e", float, (3,)),  # Angular velocity of the body in body frd
            ("omega_p2e", float, (3,)),  # Angular velocity of the payload in payload frd
            ("v_R2e", float, (3,)),  # The velocity of `R` in ned
        ]

        def dynamics(t, state, kwargs):
            x = state.view(state_dtype)[0]
            q_e2b = x["q_b2e"] * [1, -1, -1, -1]
            Theta_p = quaternion.quaternion_to_euler(x["q_p2b"])
            a_R2e, alpha_b2e, alpha_p2e, solution = self.accelerations(
                x["v_R2e"],
                x["omega_b2e"],
                x["omega_p2e"],
                Theta_p,  # FIXME: design review the call signature
                quaternion.apply_quaternion_rotation(x["q_b2e"], [0, 0, 9.8]),
                **kwargs,
            )

            P, Q, R = x["omega_b2e"]
            # fmt: off
            Omega = np.array([
                [0, -P, -Q, -R],
                [P,  0,  R, -Q],
                [Q, -R,  0,  P],
                [R,  Q, -P,  0]])
            # fmt: on
            q_b2e_dot = 0.5 * Omega @ x["q_b2e"]

            omega_b2e = quaternion.apply_quaternion_rotation(x["q_p2b"], x["omega_b2e"])
            omega_p2b = x["omega_p2e"] - omega_b2e
            P, Q, R = omega_p2b
            # fmt: off
            Omega = np.array([
                [0, -P, -Q, -R],
                [P,  0,  R, -Q],
                [Q, -R,  0,  P],
                [R,  Q, -P,  0]])
            # fmt: on
            q_p2b_dot = 0.5 * Omega @ x["q_p2b"]

            x_dot = np.empty(1, state_dtype)
            x_dot["q_b2e"] = q_b2e_dot
            x_dot["q_p2b"] = q_p2b_dot
            x_dot["omega_b2e"] = alpha_b2e
            x_dot["omega_p2e"] = alpha_p2e
            x_dot["v_R2e"] = a_R2e
            kwargs["reference_solution"] = solution
            return x_dot.view(float)  # The integrator expects a flat array

        state = np.empty(1, state_dtype)
        state["q_b2e"] = quaternion.euler_to_quaternion([0, theta_0, 0])
        state["q_p2b"] = state["q_b2e"] * [1, -1, -1, -1]  # Payload aligned to gravity (straight down)  # FIXME: add theta_p_0
        state["omega_b2e"] = [0, 0, 0]
        state["omega_p2e"] = [0, 0, 0]
        state["v_R2e"] = v_0 * np.array([np.cos(alpha_0), 0, np.sin(alpha_0)])

        dynamics_kwargs = {
            "delta_a": delta_a,
            "delta_bl": delta_b,
            "delta_br": delta_b,
            "rho_air": rho_air,
            "reference_solution": reference_solution,
        }

        solver = scipy.integrate.ode(dynamics)
        solver.set_integrator("dopri5", rtol=1e-5, max_step=0.05)
        solver.set_f_params(dynamics_kwargs)

        while True:
            state["q_b2e"] /= np.sqrt((state["q_b2e"] ** 2).sum())
            state["q_p2b"] /= np.sqrt((state["q_p2b"] ** 2).sum())
            solver.set_initial_value(state.view(float))
            state = solver.integrate(0.25).view(state_dtype)
            state["omega_b2e"] = [0, 0, 0]  # Zero every step to avoid oscillations
            state["omega_p2e"] = [0, 0, 0]  # Zero every step to avoid oscillations
            Theta_p = quaternion.quaternion_to_euler(state["q_p2b"][0])
            a_R2e, alpha_b2e, alpha_p2e, solution = self.accelerations(
                state["v_R2e"][0],
                state["omega_b2e"][0],
                state["omega_p2e"][0],
                Theta_p,  # FIXME: design review the call signature
                quaternion.apply_quaternion_rotation(state["q_b2e"][0], [0, 0, 9.8]),
                **dynamics_kwargs,
            )

            # FIXME: this test doesn't guarantee equilibria
            if any(abs(a_R2e) > 0.001) or any(abs(alpha_b2e) > 0.001) or any(abs(alpha_p2e) > 0.001):
                continue
            else:
                state = state[0]
                break

        alpha_eq = np.arctan2(*state["v_R2e"][[2, 0]])
        theta_b_eq = quaternion.quaternion_to_euler(state["q_b2e"])[1]
        theta_p_eq = quaternion.quaternion_to_euler(state["q_p2b"])[1]
        v_eq = np.linalg.norm(state["v_R2e"])

        return alpha_eq, theta_b_eq, theta_p_eq, v_eq, dynamics_kwargs["reference_solution"]
