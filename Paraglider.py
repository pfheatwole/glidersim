"""FIXME: add module docstring."""

import numpy as np

from IPython import embed

from util import cross3


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

    def control_points(self, delta_s=0):
        """
        Compute the reference points for the composite Paraglider system.

        All the components of the Paraglider that experience aerodynamic forces
        need their relative wind vectors. Each component is responsible for
        creating a list of the coordinates where they need the value of the
        wind. This function then transforms them into body coordinates.
        """
        wing_cps = self.wing.control_points(delta_s=delta_s)
        harness_cps = self.harness.control_points()
        return np.vstack((wing_cps, harness_cps))

    def forces_and_moments(self, UVW, PQR, g, rho_air,
                           delta_Bl=0, delta_Br=0, delta_s=0,
                           v_w2e=None, xyz=None, initial_Gamma=None):
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
        delta_Bl : float [percentage]
            The fraction of maximum left brake
        delta_Br : float [percentage]
            The fraction of maximum right brake
        delta_s : float [percentage]
            The fraction of maximum speed bar
        v_w2e : ndarray of float, shape (3,) or (K,3)
            The wind relative to the earth, in frd coordinates. If it is a
            single vector, then the wind is uniform everywhere on the wing. If
            it is an ndarray, then it is the wind at each control point.
        xyz : ndarray of float, shape (K,3) [meters] (optional)
            The control points, in frd coordinates. These are optional if the
            wind field is uniform, but for non-uniform wind fields the
            simulator used these coordinates to determine the wind vectors
            at each control point.

            FIXME: This docstring is wrong; they are useful if delta_s != 0,
            they have nothing to do with wind field uniformity. And really,
            why do I even have both `xyz` and `delta_s` as inputs? The only
            purpose of `delta_s` is to compute the xyz. Using `delta_s` alone
            would be the more intuitive, but would incur extra computation time
            for finding the control points; the only point of `xyz` is to avoid
            recomputing them.

        initial_Gamma : array of float, shape (K,) [units?] (optional)
            An initial guess for the circulation distribution, to improve
            convergence

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
            raise ValueError("Control point relative winds require xyz")                # Why? I probably did this to ensure the <v_w2e and xyz are computed using the same delta_s
        if v_w2e.ndim > 1 and v_w2e.shape[0] != xyz.shape[0]:
            raise ValueError("Different number of wind and xyz vectors")
        if xyz is None:
            xyz = self.control_points(delta_s)

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
        dF_w, dM_w, Gamma = self.wing.forces_and_moments(v_wing, delta_Bl, delta_Br, initial_Gamma)
        dF_h, dM_h = self.harness.forces_and_moments(v_harness)
        F = np.atleast_2d(dF_w).sum(axis=0) + np.atleast_2d(dF_h).sum(axis=0)
        M = np.atleast_2d(dM_w).sum(axis=0) + np.atleast_2d(dM_h).sum(axis=0)

        # Add the torque produced by the wing forces; the harness drag is
        # applied at the center of mass, and so produces no additional torque.
        M += cross3(cp_wing, dF_w).sum(axis=0)

        # Scale the aerodynamic forces to account for the air density before
        # adding the weight of the harness
        F *= rho_air
        M *= rho_air

        # The harness also contributes a gravitational force, but since this
        # model places the cg at the harness, that force does not generate a
        # moment.
        F += self.harness.mass * np.asarray(g)  # FIXME: leaky abstraction

        return F, M, Gamma

    def equilibrium_glide(self, delta_B, delta_S, rho_air, alpha_eq=None):
        r"""
        Steady-state angle of attack, pitch angle, and airspeed.

        Parameters
        ----------
        delta_B : float [percentage]
            Percentage of symmetric brake application
        delta_S : float [percentage]
            Percentage of speed bar application
        rho_air : float [kg/m^3]
            Air density. Default value is 1.
        alpha_eq : float [radians] (optional)
            Steady-state angle of attack

        Returns
        -------
        alpha_eq : float [radians]
            Steady-state angle of attack
        Theta_eq : float [radians]
            Steady-state pitch angle
        V_eq : float [meters/second]
            Steady-state airspeed

        Notes
        -----
        Calculating :math:`V_eq` takes advantage of the fact that all the
        aerodynamic forces are proportional to :math:`V^2`. Thus, by
        calculating the forces for :math:`V = 1`, the following equation can be
        solved for :math:`V_eq` directly:

        .. math::

            V_{eq}^2 \cdot \Sigma F_{z,aero} + mg \cdot \text{sin} \left( \Theta \right)

        where `m` is the mass of the harness + pilot.
        """
        if alpha_eq is None:
            alpha_eq = self.wing.equilibrium_alpha(delta_B, delta_S)

        g = np.zeros(3)  # Don't include the weight of the harness
        V = np.array([np.cos(alpha_eq), 0, np.sin(alpha_eq)])
        F, M, _ = self.forces_and_moments(
            V, [0, 0, 0], g, rho_air, delta_B, delta_B, delta_S,
        )

        # FIXME: neglects the weight of the wing
        Theta_eq = np.arctan2(F[0], -F[2])
        V_eq = np.sqrt(-(9.8*self.harness.mass*np.cos(Theta_eq))/F[2])

        return alpha_eq, Theta_eq, V_eq
