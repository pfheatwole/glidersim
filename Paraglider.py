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
