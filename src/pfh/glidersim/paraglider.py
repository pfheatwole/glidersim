"""FIXME: add module docstring."""

from IPython import embed

import numpy as np

from pfh.glidersim import quaternion
from pfh.glidersim.util import cross3

import scipy.integrate


class Paraglider:
    """
    FIXME: add a docstring.

    FIXME: warn for non-zero harness control points (this model ignores them)

    FIXME: this model assumes the glider center of mass is at the glider origin
           (where the risers attach), so the harness doesn't contribute a
           moment. I should estimate the true cm to double check the validity
           of this assumption.

    Notes
    -----
    This is a 7 DoF model: there is no relative motion between the wing and
    the glider system, except for weight shift (y-axis displacement of the cm).
    """

    def __init__(self, wing, harness):
        """
        Instantiate a Paraglider from given wing and harness.

        Parameters
        ----------
        wing : ParagliderWing
        harness : Harness
        """
        self.wing = wing
        self.harness = harness

    def control_points(self, delta_a=0):
        """
        Compute the reference points for the composite Paraglider system.

        All the components of the Paraglider that experience aerodynamic forces
        need their relative wind vectors. Each component is responsible for
        creating a list of the coordinates where they need the value of the
        wind. This function then transforms them into body coordinates.
        """
        wing_cps = self.wing.control_points(delta_a=delta_a)
        harness_cps = self.harness.control_points()
        return np.vstack((wing_cps, harness_cps))

    def accelerations(
        self,
        UVW,
        PQR,
        g,
        rho_air,
        delta_a=0,
        delta_bl=0,
        delta_br=0,
        v_w2e=None,
        xyz=None,
        reference_solution=None,
    ):
        """
        Compute the translational and angular accelerations about the center of mass.

        FIXME: needs a design review; the `xyz` parameter name in particular
        FIXME: the input sanitation is messy
        FIXME: review the docstring

        Parameters
        ----------
        UVW : array of float, shape (3,) [m/s]
            Translational velocity of the cm, in frd coordinates.
        PQR : array of float, shape (3,) [rad/s]
            Angular velocity of the cm, in frd coordinates.
        g : array of float, shape (3,) [m/s^s]
            The gravity vector
        rho_air : float [kg/m^3]
            The ambient air density
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        v_w2e : ndarray of float, shape (3,) or (K,3) [m/s]
            The wind relative to the earth, in frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        xyz : ndarray of float, shape (K,3) [meters] (optional)
            The control points, in frd coordinates. These are optional if the
            wind field is uniform, but for non-uniform wind fields the
            simulator used these coordinates to determine the wind vectors
            at each control point.

            FIXME: This docstring is wrong; they are useful if delta_a != 0,
            they have nothing to do with wind field uniformity. And really,
            why do I even have both `xyz` and `delta_a` as inputs? The only
            purpose of `delta_a` is to compute the xyz. Using `delta_a` alone
            would be the more intuitive, but would incur extra computation time
            for finding the control points; the only point of `xyz` is to avoid
            recomputing them.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        a_frd : array of float, shape (3,) [m/s^2]
            Translational acceleration of the center of mass
        alpha_frd : array of float, shape (3,) [rad/s^2]
            Angular acceleration of the center of mass

        Notes
        -----
        There are two use cases:
         1. Uniform global wind across the wing (v_w2e.shape == (3,))
         2. Non-uniform global wind across the wing (v_w2e.shape == (K,3))

        If the wind is locally uniform across the wing, then the simulator
        can pass the wind vector with no knowledge of the control points.
        If the wind is non-uniform across the wing, then the simulator will
        need the control point coordinates in order to query the global wind
        field; for the non-uniform case, the control points are a required
        parameter to eliminate their redundant computation.
        """
        # FIXME: design review names. `w` is overloaded ("wind" and "wing")
        if v_w2e is None:
            v_w2e = np.array([0, 0, 0])
        else:
            v_w2e = np.asarray(v_w2e)
        if v_w2e.ndim > 1 and xyz is None:
            # FIXME: needs a design review. Ensure that if `v_w2e` and `xyz`
            #        were computed using the same `delta_a`, if `v_w2e` was
            #        computed for the individual control points.
            raise ValueError("Control point relative winds require xyz")
        if v_w2e.ndim > 1 and v_w2e.shape[0] != xyz.shape[0]:
            raise ValueError("Different number of wind and xyz vectors")
        if xyz is None:
            xyz = self.control_points(delta_a)

        UVW = np.asarray(UVW)
        if UVW.shape != (3,):
            raise ValueError("UVW must be a 3-vector velocity of the body cm")

        # -------------------------------------------------------------------
        # Compute the inertia matrices about the glider cm
        wmp = self.wing.mass_properties(rho_air, delta_a)
        hmp = self.harness.mass_properties()
        m_g = wmp["m_solid"] + wmp["m_air"] + hmp["mass"]
        cm_g = (
            wmp["m_solid"] * wmp["cm_solid"]
            + wmp["m_air"] * wmp["cm_air"]
            + hmp["mass"] * hmp["cm"]
        ) / m_g
        Rws = wmp["cm_solid"] - cm_g  # Displacement of the wing solid mass
        Rwa = wmp["cm_air"] - cm_g  # Displacement of the wing enclosed air
        Rh = hmp["cm"] - cm_g  # Displacement of the harness
        Dws = (Rws @ Rws) * np.eye(3) - np.outer(Rws, Rws)
        Dwa = (Rwa @ Rwa) * np.eye(3) - np.outer(Rwa, Rwa)
        Dh = (Rh @ Rh) * np.eye(3) - np.outer(Rh, Rh)
        J_w = (
            wmp["J_solid"]
            + wmp["m_solid"] * Dws
            + wmp["J_air"]
            + wmp["m_air"] * Dwa
        )
        J_h = (hmp["J"] + hmp["mass"] * Dh)

        # -------------------------------------------------------------------
        # Compute the velocity of each control point relative to the air
        v_cm2w = UVW - v_w2e  # ref: ACS Eq:1.4-2, p17 (31)

        # FIXME: review this. Do the velocities and "arms" need to be wrt the
        # same point? The UVW is for the origin, but rotation happens about
        # the glider cm. I'm not sure what to do, but I suspect it should either
        # be `V_o + cross(PQR, xyz - o` or `V_cm + cross(PQR, xyz - cm)`
        #
        v_cp2w = v_cm2w + cross3(PQR, xyz - cm_g)  # ACS, Eq:1.7-14, p40 (54)

        # FIXME: "magic" layout of array contents
        cp_wing = xyz[:-1]
        cp_harness = xyz[-1]
        v_wing = v_cp2w[:-1]
        v_harness = v_cp2w[-1]

        # -------------------------------------------------------------------
        # Compute the forces and moments of the wing
        dF_w_aero, dM_w_aero, ref = self.wing.forces_and_moments(
            delta_bl, delta_br, v_wing, rho_air, reference_solution,
        )
        F_w_aero = dF_w_aero.sum(axis=0)
        F_w_weight = wmp["m_solid"] * g
        M_w = dM_w_aero.sum(axis=0)
        M_w += cross3(cp_wing - cm_g, dF_w_aero).sum(axis=0)
        M_w += cross3(wmp["cm_solid"] - cm_g, F_w_weight)

        # Forces and moments of the harness
        dF_h_aero, dM_h_aero = self.harness.forces_and_moments(v_harness, rho_air)
        dF_h_aero = np.atleast_2d(dF_h_aero)
        dM_h_aero = np.atleast_2d(dM_h_aero)
        F_h_aero = dF_h_aero.sum(axis=0)
        F_h_weight = hmp["mass"] * g
        M_h = dM_h_aero.sum(axis=0)
        M_h += cross3(cp_harness - cm_g, dF_h_aero).sum(axis=0)
        M_h += cross3(hmp["cm"] - cm_g, F_h_weight)

        # ------------------------------------------------------------------
        # Compute the accelerations \dot{PQR} and \dot{UVW}
        #
        # Builds a system of equations by equating the derivatives of angular
        # and translatational momentum against the moments and forces, and
        # rearranging terms with unknown and known factors.

        J = J_w + J_h  # Total moment of inertia matrix about the glider cm

        A1 = [np.zeros((3, 3)), m_g * np.eye(3)]
        A2 = [J, np.zeros((3, 3))]
        A = np.block([A1, A2])

        B1 = (
            F_w_aero
            + F_w_weight
            + F_h_aero
            + F_h_weight
            - m_g * cross3(PQR, UVW)
            - m_g * cross3(PQR, cross3(PQR, cm_g))
        )
        B2 = M_w + M_h - np.cross(PQR, J @ PQR)
        B = np.r_[B1, B2]

        derivatives = np.linalg.solve(A, B)
        alpha_frd, a_frd = derivatives[:3], derivatives[3:]

        return a_frd, alpha_frd, ref

    def equilibrium_glide(
        self,
        delta_a,
        delta_b,
        V_eq_proposal,
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
        V_eq_proposal : float [m/s]
            A rough guess for the equilibrium airspeed. This is required to put
            the Reynolds numbers into the proper range.
        rho_air : float [kg/m^3]
            Air density.
        N_iter : integer, optional
            Number of iterations to account for the fact that the Reynolds
            numbers (and thus the coefficients) vary with the solution for
            `V_eq`. If `V_eq_proposal` is anywhere close to accurate, then one
            or two iterations are usually sufficient. Default: 2
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        theta_eq : float [radians]
            Steady-state pitch angle
        V_eq : float [m/s]
            Steady-state airspeed
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Notes
        -----
        Calculating :math:`V_eq` takes advantage of the fact that all the
        aerodynamic forces are proportional to :math:`V^2`. Thus, by
        normalizing the forces to :math:`V = 1`, the following equation can be
        solved for :math:`V_eq` directly using:

        .. math::

            V_{eq}^2 \cdot \Sigma F_{z,aero} + mg \cdot \text{sin} \left( \theta \right)

        where `m` is the mass of the harness + pilot.
        """

        V_eq = V_eq_proposal  # The initial guess
        solution = reference_solution  # Approximate solution, if available
        m_h = self.harness.mass_properties()["mass"]

        for _ in range(N_iter):
            alpha_eq = self.wing.equilibrium_alpha(
                delta_a, delta_b, V_eq, rho_air, solution,
            )
            UVW = V_eq * np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
            dF_w, dM_w, solution = self.wing.forces_and_moments(
                delta_b, delta_b, UVW, rho_air, solution,
            )
            dF_h, dM_h = self.harness.forces_and_moments(UVW, rho_air)
            F = dF_w.sum(axis=0) + np.atleast_2d(dF_h).sum(axis=0)
            F /= V_eq ** 2  # The equation for `V_eq` assumes `V == 1`

            theta_eq = np.arctan2(F[0], -F[2])

            # FIXME: neglects the weight of the wing
            weight_z = 9.8 * m_h * np.cos(theta_eq)
            V_eq = np.sqrt(-weight_z / F[2])

        return alpha_eq, theta_eq, V_eq, solution

    def equilibrium_glide2(
        self,
        delta_a,
        delta_b,
        alpha_0,
        theta_0,
        V_0,
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
        V_0 : float [m/s], optional
            The initial proposal for glider airspeed.

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        theta_eq : float [radians]
            Steady-state pitch angle
        V_eq : float [m/s]
            Steady-state airspeed
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        state_dtype = [
            ("q", float, (4,)),  # Orientation quaternion, body/earth
            ("v", float, (3,)),  # Translational velocity in frd
            ("omega", float, (3,)),  # Angular velocity in frd
        ]

        def dynamics(t, state, kwargs):
            x = state.view(state_dtype)[0]
            x["q"] /= np.sqrt((x["q"] ** 2).sum())
            g = 9.8 * quaternion.apply_quaternion_rotation(x["q"], [0, 0, 1])
            a_frd, alpha_frd, ref = self.accelerations(x["v"], x["omega"], g, **kwargs)
            P, Q, R = x["omega"]
            # fmt: off
            Omega = np.array([
                [0, -P, -Q, -R],
                [P,  0,  R, -Q],
                [Q, -R,  0,  P],
                [R,  Q, -P,  0]])
            # fmt: on
            q_dot = 0.5 * Omega @ x["q"]
            x_dot = np.empty(1, state_dtype)
            x_dot["q"] = q_dot
            x_dot["v"] = a_frd
            x_dot["omega"] = alpha_frd
            kwargs["reference_solution"] = ref
            return x_dot.view(float)  # The integrator expects a flat array

        state = np.empty(1, state_dtype)
        state["q"] = quaternion.euler_to_quaternion([0, theta_0, 0])
        state["v"] = V_0 * np.array([np.cos(alpha_0), 0, np.sin(alpha_0)])
        state["omega"] = [0, 0, 0]

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
            solver.set_initial_value(state.view(float))
            state = solver.integrate(1).view(state_dtype)
            state["omega"] = [0, 0, 0]   # Zero every step to avoid oscillations
            g = 9.8 * quaternion.apply_quaternion_rotation(state["q"][0], [0, 0, 1])
            a_frd, alpha_frd, _ = self.accelerations(
                UVW=state["v"][0],
                PQR=state["omega"][0],
                g=g,
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

        alpha_eq = np.arctan2(*state["v"][[2, 0]])
        theta_eq = quaternion.quaternion_to_euler(state["q"])[1]
        V_eq = np.linalg.norm(state["v"])

        return alpha_eq, theta_eq, V_eq, dynamics_kwargs["reference_solution"]
