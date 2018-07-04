# import abc

from functools import partial

import numpy as np

from scipy.optimize import minimize_scalar

from IPython import embed

from util import trapz


class ParagliderWing:
    def __init__(self, parafoil, brake_geo, d_cg, h_cg, kappa_S):
        """
        Parameters
        ----------
        parafoil : Parafoil
        d_cg : float [percentage]
            Distance of the cg from the central chord leading edge, as a
            percentage of the chord length, where 0 < d_cg < 1
        h_cg : float [meters]
            Perpendiular distance from the cg to the central chord
        kappa_S : float [meters]
            The speed bar line length. This corresponds to the maximum change
            in the length of the lines to the leading edge.
        """
        self.parafoil = parafoil
        self.brake_geo = brake_geo
        self.d_cg = d_cg  # FIXME: reparametrize. d_cg -> x_cg (absolute)
        self.h_cg = h_cg  # FIXME: rename. h_cg -> z_cg (absolute)
        self.kappa_S = kappa_S  # FIXME: strange notation. Why `kappa`?

        self.C0 = parafoil.geometry.fc(0)
        self.LE = np.sqrt(h_cg**2 + (d_cg*self.C0)**2)
        self.TE = np.sqrt(h_cg**2 + ((1-d_cg)*self.C0)**2)

        # Precompute some useful values
        self.y = parafoil.control_points[:, 1]  # Control point y-coordinates
        self.c = parafoil.geometry.fc(self.y)  # Chord lengths

        # FIXME: pre-compute the `body->local` transformation matrices here?

    def forces_and_moments(self, V_cp2w, delta_Bl, delta_Br, delta_a):
        """

        Parameters
        ----------
        V_cp2w : ndarray of floats, shape (K,3) [m/s]
            The relative velocity of each control point vs the fluid
        delta_Bl : float [percentage]
            The amount of left brake
        delta_Br : float [percentage]
            The amount of right brake

        Returns
        -------
        dF, dM : array of float, shape (K,)
            Forces and moments for each section.
        """
        delta = self.brake_geo(self.y, delta_Bl, delta_Br) / self.c
        dF, dM = self.parafoil.forces_and_moments(V_cp2w, delta)
        return dF, dM

    def cg_position(self, delta_S):
        """
        Compute {d_cg, h_cg} for a given speed bar position

        Parameters
        ----------
        delta_S : float or array of float
            Fraction of speed bar application

        Returns
        -------
        d_cg : float or array of float [percent]
            The horizontal position of the cg relative to the central leading
            edge as a fraction of the central chord.
        h_cg : float or array of float [m]
            The vertical position of the cg relative to the central chord.
        """
        c = self.parafoil.geometry.fc(0)
        delta_LE = delta_S * self.kappa_S
        d_cg = (self.TE**2 - (self.LE-delta_LE)**2 - c**2)/(-2*c**2)
        h_cg = np.sqrt((self.LE-delta_LE)**2 - (d_cg*c)**2)

        return d_cg, h_cg

    def alpha_eq(self, delta_B, delta_S):
        """

        Parameters
        ----------
        delta_B : float [percentage]
            Fraction of symmetric braking application
        delta_S : float [percentage]
            Fraction of speed bar application

        Returns
        -------
        alpha_eq : float [radians]
            The equilibrium angle of attack for the given control inputs
        """

        def moment_factor(delta_B, delta_S, alpha):
            raise RuntimeError("FIXME: broken. Wrong implementaion anyway.")
            CL = self.parafoil_coefs.CL(alpha, delta_B)
            CD = self.parafoil_coefs.CD(alpha, delta_B)
            CM_c4 = self.parafoil_coefs.CM(alpha, delta_B)

            Cx = CL*np.sin(alpha) - CD*np.cos(alpha)
            Cz = -CL*np.cos(alpha) - CD*np.sin(alpha)  # FIXME: verify

            MAC = self.parafoil.geometry.MAC
            c = self.parafoil.geometry.fc(0)
            d_cg, h_cg = self.cg_position(delta_S)

            kMy = CM_c4*MAC - Cx*h_cg - Cz*((d_cg - 1/4)*c)

            return np.abs(kMy)

        f = partial(moment_factor, delta_B, delta_S)
        alpha_min, alpha_max = np.deg2rad(-1.5), np.deg2rad(20)  # FIXME: magic
        r = minimize_scalar(f, bounds=(alpha_min, alpha_max), method='Bounded')

        return r.x

    def control_points(self, delta_s):
        """
        Compute the coordinates of the control points for force estimation.

        Parameters
        ----------
        delta_s : float [percentage?]
            The amount of speedbar

        Returns
        -------
        cps : ndarray of floats, shape (K,3) [meters]
            The control points in ParagliderWing coordinates
        """
        cps = self.parafoil.control_points.copy()  # In Parafoil coordinates
        d_cg, h_cg = self.cg_position(delta_s)
        cps[:, 0] += (d_cg - 1/4) * self.C0
        cps[:, 2] += h_cg
        return cps

    # FIXME: moved from foil. Verify and test.
    def surface_distributions(self):
        """The surface area distributions for computing inertial moments.

        The moments of inertia for the parafoil are the mass distribution of
        the air and wing material. That distribution is typically decomposed
        into the product of volumetric density and volume, but a simplification
        is to calculate the density per unit area.

        FIXME: this description is mediocre.

        Ref: "Paraglider Flight Dynamics", page 48 (56)

        Returns
        ------
        S : 3x3 matrix of float
            The surface distributions, such that `J = (p_w + p_air)*s`
        """

        # FIXME: this was moved from Parafoil when creating ParagliderWing.
        #        That move eliminated dcg/h0 from the ParafoilGeometry, so
        #        ParafoilGeometry.fx and ParafoilGeometry.fz no longer position
        #        the Parafoil relative to the cg. Calculating the moments of
        #        inertia requires that the ParagliderWing add those terms back.
        print("BROKEN! See the source FIXME")
        1/0

        # FIXME: technically, the speedbar would rotate the foil coordinates.
        #        The {fx, fz} should be corrected for that.
        # FIXME: linear distribution? A cosine is probably more suitable
        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.geometry.b/2, self.geometry.b/2, N)

        fx = self.fx(y)  # FIXME: add the `dcg` term after moving to Glider
        fz = self.fz(y)  # FIXME: add the h0 term after moving to Glider
        fc = self.fc(y)

        # FIXME: needs verification. What about weight shifting? Speed bar?
        Sx = trapz((y**2 + fz**2)*fc, dy)
        Sy = trapz((3*fx**2 - fx*fc + (7/32)*fc**2 + 6*fz**2)*fc, dy)
        Sz = trapz((3*fx**2 - fx*fc + (7/32)*fc**2 + 6*y**2)*fc, dy)
        Sxy = 0
        Sxz = trapz((2*fx - fc/2)*fz*fc, dy)
        Syz = 0

        S = np.array([
            [Sx, Sxy, Sxz],
            [Sxy, Sy, Syz],
            [Sxz, Syz, Sz]])

        return S

    # FIXME: moved from Parafoil. Verify and test.
    def J(self, rho=1.3, N=2000):
        """Compute the 3x3 moment of inertia matrix.

        Parameters
        ----------
        rho : float
            Volumetric air density of the atmosphere
        N : integer
            The number of points for integration across the span

        Returns
        -------
        J : 3x3 matrix of float
                [[Jxx Jxy Jxz]
            J =  [Jxy Jyy Jyz]
                 [Jxz Jyz Jzz]]
        """
        S = self.geometry.surface_distributions(N=N)
        wing_air_density = rho*self.density_factor
        surface_density = self.wing_density + wing_air_density
        return surface_density * S
