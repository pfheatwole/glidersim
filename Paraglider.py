import numpy as np

from IPython import embed

class Paraglider:
    """


    FIXME: I need to think through the purpose of this class


    Notes
    -----
    This is a 7 DoF model: there is no relative motion between the wing and
    the glider system, except for weight shift (y-axis displacement of the cm).
    """

    def __init__(self, wing, harness):
        """
        Parameters
        ----------
        wing : ParagliderWing
        harness : Harness
        """
        self.wing = wing
        self.harness = harness

    def control_points(self, delta_s=0):
        """
        Compute the all the reference points on the Paraglider system.

        All the components of the Paraglider that experience aerodynamic forces
        need the relative wind vectors. Each component is responsible for
        creating a list of the coordinates where they need the value of the
        wind.

        Given each component's set of control points, this function then
        transforms them as necessary into body coordinates. For example:

        ```
            foil_cps = self.wing.control_points(delta_s=delta_s)
            foil_cps[:, 1] = foil_cps[:, 1] - (delta_w * self.kappa_w)
        ```

        For this simple model, where the cg is defined as the origin of the
        Parafoil, it's pretty simple. But if you get more complicated, such as
        non-spherical harness that isn't exactly at the cg, line drag, etc,
        it gets more interesting.


        Keep in mind: I'm anticipating that the simulator will query this
        function FIRST in order to determine the CP coordinates in Earth
        coordinates, so it can query the wind field at each point.

           *** THIS NEEDS A DESIGN REVIEW ***


        """
        wing_cps = self.wing.control_points(delta_s=delta_s)
        harness_cps = self.harness.control_points()
        return np.vstack((wing_cps, harness_cps))

    def section_wind(self, y, UVW, PQR, controls=None):
        # FIXME: remove the explicit `y` parameter?



        # FIXME: this is duplicated in `forces_and_moments`
        #
        #        Also, isn't this a more general idea? It doesn't just apply
        #        to the wing; it applies to the harness, and any other points
        #        that contributed an aerodynamic force. So "section wind" is
        #        really "control point relative wind", for ALL cps, not just
        #        those on the wing. This will come into play if I place the
        #        harness below the cg, for example.



        # Compute the local relative wind for parafoil sections
        #
        # FIXME: finish the docstring
        # FIXME: this is incomplete. It doesn't have a parameter for the wind.
        #        (UVW is the motion of the cg relative to the inertial frame)

        delta_s = 0  # FIXME: should be part of the control?

        # Version 1: PFD eqs 4.10-4.12, p73
        # uL = U + z*Q - y*R
        # vL = V - z*P + x*R
        # wL = W + y*P - x*Q
        #
        # Or, rewritten to highlight `v_rel = UVW + cross(PQR, xyz)`:
        # uL = U +    0 + -R*y +  Q*z
        # vL = V +  R*x +    0 + -P*z
        # wL = W + -Q*x +  P*y +    0
        # v_rel  = np.c_[uL, vL, wL]

        # Version 2: 'Aircraft Control and Simulation', Eq:1.4-2, p17 (31)
        # Assumes the ParagliderWing is mounted exactly at the cg
        # FIXME: this does not account for the orientation of the wing!!!

        # FIXME: testing

        v_b2w = UVW + 0  # FIXME: what is v_{body/wind} ?
        xyz = self.wing.control_points(delta_s)
        v_rel = v_b2w + np.cross(PQR, xyz)
        return v_rel

    def forces_and_moments(self, UVW, PQR,
                           delta_Bl=0, delta_Br=0,
                           delta_s=0,
                           v_w2e=None, xyz=None):
        """
        Compute the aerodynamic force and moment about the center of gravity.

        FIXME: how is this function useful? The simulator doesn't care about
               forces, it cares about rates-of-change of the state variables
               The key goal are the _accelerations_, not the forces.

        FIXME: should this function compute ALL forces, including gravity?
               Related: the function name should match what it does.
        FIXME: separate the section and resultant force computations?
        FIXME: needs a design review; the `xyz` parameter name in particular
        FIXME: the input sanitation is messy
        FIXME: review the docstring
        FIXME: TEST TEST TEST

        Parameters
        ----------
        UVW : array of float, shape (3,) [m/s]
            Translational velocity of the body, in frd coordinates.
        PQR : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in frd coordinates.
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
         2. Non-uniform global wind across the wing (v_w2e.shape == (3,K))

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
            raise ValueError("Control point relative winds require xyz")
        if v_w2e.ndim > 1 and v_w2e.shape[0] != xyz.shape[0]:
            raise ValueError("Different number of wind and xyz vectors")
        if xyz is None:
            xyz = self.wing.control_points(delta_s)

        if UVW.shape != (3,):
            raise ValueError("UVW must be a 3-vector velocity of the body cm")

        # Compute the velocity of each control point relative to the air
        v_cm2w = UVW - v_w2e  # ref: ACS Eq:1.4-2, p17 (31)
        v_cp2w = v_cm2w + np.cross(PQR, xyz)  # ref: ACS, Eq:1.7-14, p40 (54)

        # FIXME: how does this Glider know which v_cp2w goes to the wing and
        #        which to the harness? Those components must declare how many
        #        control points they're using.
        # FIXME: define an API for separating the v_cp2w
        v_wing = v_cp2w[:-1]
        v_harness = v_cp2w[-1]

        # Compute the resultant force and moment about the cg
        dF_w, dM_w = self.wing.forces_and_moments(v_wing, delta_Bl, delta_Br)
        dF_h, dM_h = self.harness.forces_and_moments(v_harness)
        dF = np.atleast_2d(dF_w).sum(axis=0) + np.atleast_2d(dF_h).sum(axis=0)
        dM = np.atleast_2d(dM_w).sum(axis=0) + np.atleast_2d(dM_h).sum(axis=0)

        # FIXME: compute the glider center of mass
        # FIXME: apply the forces about the cm to compute the correct moment

        return dF, dM

    # def equilibrium_glide(self, delta_B, delta_S, rho=1):
    #     """
    #     Equilibrium pitch angle and airspeed for a given brake position.
    #
    #     Parameters
    #     -----------
    #     delta_B : float [percentage]
    #         Percentage of symmetric brake application
    #     delta_S : float [percentage]
    #         Percentage of speed bar application
    #     rho : float, optional [kg/m^3]
    #         Air density. Default value is 1.
    #
    #     Returns
    #     -------
    #     Theta_eq : float [radians]
    #         Steady-state pitch angle
    #     VT_eq : float [meters/second]
    #         Steady-state airspeed
    #
    #     Notes
    #     -----
    #     It is important to note that the pitch angle is always defined as the
    #     angle between the (unbraked) central chord and the horizon. A change
    #     in the pitch angle does not necessarily indicate a equal change in
    #     the position of the wing overhead, such as when using the speed bar.
    #     """
    #
    #     # FIXME: redesign (probably in terms of forces, not coefficients?)
    #     raise NotImplementedError("This was broken by the refactored code")
    #
    #     alpha_eq = self.wing.alpha_eq(delta_B, delta_S)
    #     CL = self.wing.parafoil_coefs.CL(alpha_eq, delta_B)
    #     CD = self.wing.parafoil_coefs.CD(alpha_eq, delta_B)
    #     Cx = CL*np.sin(alpha_eq) - CD*np.cos(alpha_eq)
    #     Cz = -CL*np.cos(alpha_eq) - CD*np.sin(alpha_eq)  # FIXME: verify
    #
    #     D_cg = self.S_cg * self.CD_cg  # Total drag at the cg
    #     S = self.wing.parafoil.geometry.S
    #     g = 9.801  # FIXME: move the gravity somewhere standard
    #
    #     # PFD Eq:5.50, p121
    #     # Note the negated `Cz` versus the PFD derivation
    #     numerator = Cx - (np.cos(alpha_eq)**2 * D_cg/S)
    #     denominator = -Cz + (np.sin(alpha_eq)**2 * D_cg/S)
    #     Theta_eq = np.arctan(numerator/denominator)
    #
    #     # PFD Eq:5.51, p121
    #     numerator = self.m_cg * g * np.sin(Theta_eq)
    #     denominator = (rho/2) * (S*Cx - np.cos(alpha_eq)**2 * D_cg)
    #     VT_eq = np.sqrt(numerator/denominator)
    #
    #     return Theta_eq, VT_eq
