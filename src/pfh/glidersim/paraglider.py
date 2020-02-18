"""FIXME: add module docstring."""

from IPython import embed

import numpy as np

from pfh.glidersim.util import cross3


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

        # FIXME: assumes J is constant. This is okay if rho_air is constant,
        #        and delta_a is zero (no weight shift deformations)
        self.J = wing.inertia(rho_air=1.2, N=5000)
        self.J_inv = np.linalg.inv(self.J)

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

        FIXME: should this function compute ALL forces, including gravity?
        FIXME: needs a design review; the `xyz` parameter name in particular
        FIXME: the input sanitation is messy
        FIXME: review the docstring

        Parameters
        ----------
        UVW : array of float, shape (3,) [m/s]
            Translational velocity of the body, in frd coordinates.
        PQR : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in frd coordinates.
        g : array_like of float, shape (3,)
            The gravity unit vector
        rho_air : float [kg/m^3]
            The ambient air density
        delta_a : float [percentage]
            The fraction of maximum accelerator
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        v_w2e : ndarray of float, shape (3,) or (K,3)
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
        F : array of float, shape (3,) [N]
            The total aerodynamic force applied at the cg
        M : array of float, shape (3,) [N*m]
            The total aerodynamic torque applied at the cg

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
        if v_w2e is None:
            v_w2e = np.array([0, 0, 0])
        if v_w2e.ndim > 1 and xyz is None:
            raise ValueError("Control point relative winds require xyz")                # Why? I probably did this to ensure the <v_w2e and xyz are computed using the same delta_a
        if v_w2e.ndim > 1 and v_w2e.shape[0] != xyz.shape[0]:
            raise ValueError("Different number of wind and xyz vectors")
        if xyz is None:
            xyz = self.control_points(delta_a)

        UVW = np.asarray(UVW)
        if UVW.shape != (3,):
            raise ValueError("UVW must be a 3-vector velocity of the body cm")

        # Compute the velocity of each control point relative to the air
        v_cm2w = UVW - v_w2e  # ref: ACS Eq:1.4-2, p17 (31)
        v_cp2w = v_cm2w + cross3(PQR, xyz)  # ref: ACS, Eq:1.7-14, p40 (54)

        # FIXME: how does this Glider know which v_cp2w goes to the wing and
        #        which to the harness? Those components must declare how many
        #        control points they're using.
        # FIXME: design a proper method for separating the v_cp2w
        cp_wing = xyz[:-1]
        v_wing = v_cp2w[:-1]
        v_harness = v_cp2w[-1]

        # Compute the resultant force and moment about the cg
        dF_w, dM_w, ref = self.wing.forces_and_moments(
            delta_bl, delta_br, v_wing, rho_air, reference_solution,
        )
        dF_h, dM_h = self.harness.forces_and_moments(v_harness, rho_air)
        F = np.atleast_2d(dF_w).sum(axis=0) + np.atleast_2d(dF_h).sum(axis=0)
        M = np.atleast_2d(dM_w).sum(axis=0) + np.atleast_2d(dM_h).sum(axis=0)

        # Add the torque produced by the wing forces; the harness drag is
        # applied at the center of mass, and so produces no additional torque.
        cg_glider = np.array([0.01, 0, -0.25])
        M += cross3(cp_wing - cg_glider, dF_w).sum(axis=0)


        # FIXME: centroid of the harness and wing real mass
        # FIXME: moment for the wing solid mass
        wing_m_solid = 1.7751
        wing_cg_solid = self.wing.foil_origin() + [-1.3567, 0, 0.751]
        F += wing_m_solid * np.asarray(g)
        M += np.cross(wing_cg_solid, wing_m_solid * np.asarray(g))


        # FIXME: moment for the harness mass
        cg_glider = np.array([0.01, 0, -0.75])
        M_h_aero = np.cross(-cg_glider, dF_h)
        M_h_g = np.cross(-cg_glider, 75 * np.asarray(g))
        M += M_h_aero
        M += M_h_g


        # The harness also contributes a gravitational force, but since this
        # model places the cg at the harness, that force does not generate a
        # moment.
        F += self.harness.mass * np.asarray(g)  # FIXME: leaky abstraction

        # return F, M, ref

        # ------------------------------------------------------------------
        # Compute the accelerations

        J = self.J  # WRONG: `J` changes with rho_air
        J_inv = self.J_inv

        # Translational acceleration of the cm
        #
        # FIXME: review Stevens Eq:1.7-18. The `F` here includes gravity, but
        #        what about the `cross(omega, v)` term? And how does that
        #        compare to Eq:1.7-21? (It's an "alternative"? How so?)
        #
        #        Also be careful with Eq:1.7-16: `cross(omega, v)` is for the
        #        velocity of the cm, not the origin!
        #
        # a_frd = F / self.glider.harness.mass  # FIXME: Crude. Incomplete. Wrong.
        a_frd = F / (self.harness.mass + 4) - np.cross(PQR, UVW)  # Also wrong? Uses the origin, not the cm.

        # Angular acceleration of the body relative to the ned frame
        #  * ref: Stevens, Eq:1.7-5, p36 (50)
        #
        # FIXME: doesn't account for the moment from the harness to the glider cm
        alpha = J_inv @ (M - cross3(PQR, J @ PQR))

        return a_frd, alpha, ref


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
        Theta_eq : float [radians]
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

            V_{eq}^2 \cdot \Sigma F_{z,aero} + mg \cdot \text{sin} \left( \Theta \right)

        where `m` is the mass of the harness + pilot.
        """

        V_eq = V_eq_proposal  # The initial guess
        solution = reference_solution  # Approximate solution, if available

        for n in range(N_iter):
            alpha_eq = self.wing.equilibrium_alpha(
                delta_a, delta_b, V_eq, rho_air, solution
            )
            UVW = V_eq * np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
            dF_w, dM_w, solution = self.wing.forces_and_moments(
                 delta_b, delta_b, UVW, rho_air, solution,
            )
            dF_h, dM_h = self.harness.forces_and_moments(UVW, rho_air)
            F = dF_w.sum(axis=0) + np.atleast_2d(dF_h).sum(axis=0)
            F /= V_eq ** 2  # The equation for `V_eq` assumes `V == 1`

            Theta_eq = np.arctan2(F[0], -F[2])

            # FIXME: neglects the weight of the wing
            weight_z = 9.8 * self.harness.mass * np.cos(Theta_eq)
            V_eq = np.sqrt(-weight_z / F[2])

        return alpha_eq, Theta_eq, V_eq, solution
