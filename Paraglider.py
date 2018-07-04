import numpy as np

from IPython import embed


class Paraglider:
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

    def section_wind(self, y, UVW, PQR, control=None):
        # Compute the local relative wind for parafoil sections
        #
        # FIXME: finish the docstring
        # FIXME: this is incomplete. It doesn't have a parameter for the wind.
        #        (UVW is the motion of the cg relative to the inertial frame)

        delta_s = None  # FIXME: should be part of the control

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
        raise NotImplementedError("FIXME: finish implementation+testing")
        v_b2w = UVW + 0  # FIXME: what is v_{body/wind} ?
        xyz = self.wing.control_points(delta_s)
        v_rel = v_b2w + np.cross(PQR, xyz)
        return v_rel

    def forces_and_moments(self, UVW, PQR, delta_Bl, delta_Br, delta_s,
                           g=None, v_w_b=None, xyz=None):
        """

        Interesting to note: you can use v_wb in two different ways:
         1. A constant wind field (the wind is uniform across the wing)
         2. A varying wind field (the wind varies across the wing)

        In both cases, you compute the relwind for each control point in the
        same way, but in the constant wind field `v_we` is a scalar, and for
        the variable wind field `v_we` is a vector, with one wind vector for
        each control point.

        This does require some extra work on the part of the simulator: for the
        variable wind vector field, the simulator will need to query the glider
        to determine the position of the control points, so it can pass in the
        correct wind values at those points. In the case of a uniform wind
        field, the value of the wind at the glider cg is sufficient, since it
        will be the same at all the control points.

        Parameters
        ----------
        v_w_b : float, or array of 3-vectors of floats, shape (K,)
            Wind vectors relative to the body in frd coordinates. If a scalar,
            then the wind is a uniform vector field; if a vector, then the wind
            is non-uniform around the paraglider, and was calculated relative
            to the control points, `xyz`.

        g : array of float, shape (3,)
            The gravity vector in frd coordinates.
            FIXME: optional?

        xyz : array of 3-vectors of floats, shape (K,) [meters] (optional)
            If provided, it contains the positions of the control points on the
            parafoil, relative to the mounting point of the parafoil, in frd
            coordinates. This is used for non-uniform wind fields, where the
            simulator used these control points for calculating the relative
            wind vectors.
        """
        if v_w_b is None:
            v_w_b = np.array([0, 0, 0])
        if v_w_b.ndim > 1 and xyz is None:
            raise ValueError("Control point relative winds require xyz")
        if v_w_b.ndim > 1 and v_w_b.shape[0] != xyz.shape[0]:
            raise ValueError("Different number of wind and xyz vectors")

        # FIXME: check that len(xyz) == len(self.parafoil.estimator.cps) ?

        # If xyz is none, then v_we is a scalar of the wind at the cg.
        # If xyz is not none, then v_we is a vector of the wind at each cp
        if xyz is None:
            xyz = self.wing.control_points(delta_s)

        # Compute the relative wind at the control points
        # Implicit in this: the wing is mounted at the Paraglider cg
        # ref: 'Aircraft Control and Simulation', Eq:1.4-2, p17 (31)
        # FIXME: verify this math
        v_cm_w = UVW - v_w_b  # `v_cm_w` is also known as `v_rel`
        v_cp_w = v_cm_w + np.cross(PQR, xyz)  # Wind at each control point

        dF, dM = self.wing.forces_and_moments(v_cp_w, delta_Bl, delta_Br)

        if g is not None:
            # FIXME: do stuff?
            pass

        # Compute the resultants
        raise NotImplementedError("FIXME: finish implementation+testing")

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
