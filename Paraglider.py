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

    def __init__(self, wing, m_cg, S_cg, CD_cg):
        """
        Parameters
        ----------
        wing : ParagliderWing
        m_cg : float [kilograms]
            Total mass at the center of gravity
        S_cg : float [meters**2]
            Spherical surface area of the center of gravity
        CD_cg : float [N/m**2]
            Isotropic drag coefficient of the center of gravity
        """
        self.wing = wing
        self.m_cg = m_cg
        self.S_cg = S_cg  # FIXME: move into a Harness?
        self.CD_cg = CD_cg  # FIXME: move into a Harness?

    def control_points(self, delta_w=0, delta_s=0):
        # FIXME: this seems poorly thought out
        #
        #        Ah! Because it's incomplete? What about the relative wind on
        #        the harness? Or maybe at the lines? etc.
        #
        #        Each component with an aerodynamic force will need a wind
        #        vector. This might be a good spot for the Paraglider to
        #        signal all those points, which are then separated into their
        #        constituent components later.
        return self.wing.control_points(delta_w=delta_w, delta_s=delta_s)

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

        delta_w = 0  # FIXME: should be part of the control?
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
        xyz = self.wing.control_points(delta_w, delta_s)
        v_rel = v_b2w + np.cross(PQR, xyz)
        return v_rel

    def forces_and_moments(self, UVW, PQR,
                           delta_Bl=0, delta_Br=0,
                           delta_w=0, delta_s=0,
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
        delta_w : float [percentage]
            The fraction of maximum weight shift
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
            raise ValueError("Control point relative winds require xyz")
        if v_w2e.ndim > 1 and v_w2e.shape[0] != xyz.shape[0]:
            raise ValueError("Different number of wind and xyz vectors")
        if xyz is None:
            xyz = self.wing.control_points(delta_w, delta_s)

        # Compute the velocity of each control point relative to the air
        v_cm2w = UVW - v_w2e  # ref: ACS Eq:1.4-2, p17 (31)
        v_cp2w = v_cm2w + np.cross(PQR, xyz)  # ref: ACS, Eq:1.7-14, p40 (54)

        # Compute the resultant force and moment about the cg
        dF, dM = self.wing.forces_and_moments(v_cp2w, delta_Bl, delta_Br)
        return dF, dM

        # Alternative: return the resultants
        # if dM is None:
        #     print("FIXME: the wing didn't return the moments yet")
        #     dM = np.array([0, 0, 0])
        # F = np.sum(dF, axis=0)
        # M = np.sum(dM + np.cross(xyz, dF), axis=0)
        # return F, M

    def equilibrium_glide(self, delta_B, delta_S, rho=1):
        """
        Equilibrium pitch angle and airspeed for a given brake position.

        Parameters
        -----------
        delta_B : float [percentage]
            Percentage of symmetric brake application
        delta_S : float [percentage]
            Percentage of speed bar application
        rho : float, optional [kg/m^3]
            Air density. Default value is 1.

        Returns
        -------
        Theta_eq : float [radians]
            Steady-state pitch angle
        VT_eq : float [meters/second]
            Steady-state airspeed

        Notes
        -----
        It is important to note that the pitch angle is always defined as the
        angle between the (unbraked) central chord and the horizon. A change
        in the pitch angle does not necessarily indicate a equal change in the
        position of the wing overhead, such as when using the speed bar.
        """

        # FIXME: redesign (probably in terms of forces, not coefficients?)
        raise NotImplementedError("This was broken by the refactored code")

        alpha_eq = self.wing.alpha_eq(delta_B, delta_S)
        CL = self.wing.parafoil_coefs.CL(alpha_eq, delta_B)
        CD = self.wing.parafoil_coefs.CD(alpha_eq, delta_B)
        Cx = CL*np.sin(alpha_eq) - CD*np.cos(alpha_eq)
        Cz = -CL*np.cos(alpha_eq) - CD*np.sin(alpha_eq)  # FIXME: verify

        D_cg = self.S_cg * self.CD_cg  # Total drag at the cg
        S = self.wing.parafoil.geometry.S
        g = 9.801  # FIXME: move the gravity somewhere standard

        # PFD Eq:5.50, p121
        # Note the negated `Cz` versus the PFD derivation
        numerator = Cx - (np.cos(alpha_eq)**2 * D_cg/S)
        denominator = -Cz + (np.sin(alpha_eq)**2 * D_cg/S)
        Theta_eq = np.arctan(numerator/denominator)

        # PFD Eq:5.51, p121
        numerator = self.m_cg * g * np.sin(Theta_eq)
        denominator = (rho/2) * (S*Cx - np.cos(alpha_eq)**2 * D_cg)
        VT_eq = np.sqrt(numerator/denominator)

        return Theta_eq, VT_eq
