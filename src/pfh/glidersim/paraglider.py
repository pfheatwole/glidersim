"""FIXME: add module docstring."""

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

    def forces_and_moments(self, UVW, PQR, g, rho_air,
                           delta_bl=0, delta_br=0, delta_a=0,
                           v_w2e=None, xyz=None, reference_solution=None):
        """
        Compute the aerodynamic force and moment about the center of gravity.

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
        delta_bl : float [percentage]
            The fraction of maximum left brake
        delta_br : float [percentage]
            The fraction of maximum right brake
        delta_a : float [percentage]
            The fraction of maximum accelerator
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
        M += cross3(cp_wing, dF_w).sum(axis=0)

        # The harness also contributes a gravitational force, but since this
        # model places the cg at the harness, that force does not generate a
        # moment.
        F += self.harness.mass * np.asarray(g)  # FIXME: leaky abstraction

        return F, M, ref

    def equilibrium_glide(
        self,
        delta_b,
        delta_a,
        V_eq_proposal,
        rho_air,
        N_iter=2,
        reference_solution=None,
    ):
        r"""
        Steady-state angle of attack, pitch angle, and airspeed.

        Parameters
        ----------
        delta_b : float [percentage]
            Percentage of symmetric brake application
        delta_a : float [percentage]
            Percentage of accelerator application
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
                delta_b, delta_a, V_eq, rho_air, solution
            )
            UVW = V_eq * np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
            F, M, solution = self.forces_and_moments(
                UVW,
                [0, 0, 0],  # PQR
                [0, 0, 0],  # g (don't include the weight of the harness)
                rho_air,
                delta_b,
                delta_b,
                delta_a,
                reference_solution=solution,
            )
            F /= V_eq ** 2  # The equation for `V_eq` assumes `V == 1`
            Theta_eq = np.arctan2(F[0], -F[2])

            # FIXME: neglects the weight of the wing
            weight_z = 9.8 * self.harness.mass * np.cos(Theta_eq)
            V_eq = np.sqrt(-weight_z / F[2])

        return alpha_eq, Theta_eq, V_eq, solution
