# import abc

from functools import partial

import numpy as np
# from numpy import sqrt, sin, cos, tan, arcsin, arctan, deg2rad
from numpy import sin, cos

from scipy.optimize import least_squares

from IPython import embed


def trapz(y, dx):
    # Trapezoidal integrator
    return np.sum(y[1:] + y[:-1]) / 2.0 * dx


class BrakeGeometry:
    """
    Implements the basic PFD braking design (PFD EQ:4.18, p75)

    FIXME: document
    """

    def __init__(self, b, delta_M, delta_f):
        self.b = b
        self.delta_M = delta_M
        self.delta_f = delta_f

    def delta(self, y, delta_Bl, delta_Br):
        # FIXME: verify and test
        left = delta_Bl*self.delta_M*(100/self.delta_f)**(-y/self.b - 1/2)
        right = delta_Br*self.delta_M*(100/self.delta_f)**(y/self.b - 1/2)
        return left + right


class Glider:
    def __init__(self, wing, d_cg, h_cg, S_cg, Cd_cg, kappa_a):
        """
        Parameters
        ----------
        wing : Wing
        d_cg : float [percentage]
            Distance of the cg from the central chord leading edge, as a
            percentage of the chord length, where 0 < d_cg < 1
        h_cg : float [meters]
            Perpendiular distance from the cg to the central chord
        S_cg : float [meters**2]
            Surface area of the cg
        Cd_cg : float [N/m**2]
            Drag coefficient of cg
        kappa_a : float [meters]
            The accelerator line length. This corresponds to the maximum change
            in the length of the lines to the leading edge.
        """
        self.wing = wing
        self.d_cg = d_cg
        self.h_cg = h_cg
        self.S_cg = S_cg
        self.Cd_cg = Cd_cg
        self.kappa_a = kappa_a

        C0 = wing.geometry.fc(0)
        self.LE = np.sqrt(h_cg**2 + (d_cg*C0)**2)
        self.TE = np.sqrt(h_cg**2 + ((1-d_cg)*C0)**2)

        # FIXME: pre-compute the `body->local` transformation matrices here?
        # FIXME: lots more to implement

    def cg_position(self, delta_a):
        """
        Compute {d_cg, h_cg} for a given speed bar position

        Parameters
        ----------
        delta_a : float or array of float
            The percent application of the speedbar, where `0 <= delta_1 <= 1`

        Returns
        -------
        d_cg : float or array of float [percent]
            The horizontal position of the cg relative to the central leading
            edge as a fraction of the central chord.
        h_cg : float or array of float [m]
            The vertical position of the cg relative to the central chord.
        """
        C0 = self.wing.geometry.fc(0)
        delta_LE = delta_a * self.kappa_a
        d_cg = (self.TE**2 - (self.LE-delta_LE)**2 - C0**2)/(-2*C0**2)
        h_cg = np.sqrt((self.LE-delta_LE)**2 - (d_cg*C0)**2)
        return d_cg, h_cg

    def wing_section_wind(self, y, state, control=None):
        # FIXME: Document
        # compute the local relative wind for wing sections
        U, V, W, P, Q, R = state

        # Compute the local section wind
        # PFD eqs 4.10-4.12, p73
        #
        # FIXME: rewrite this as a matrix equation?
        #  * Ref: 'Aircraft Control and Simulation', Eq:1.4-2, p31

        delta_a = 0  # FIXME: should be a parameter

        # FIXME: shouldn't this be {d_prime, h_prime}?
        #  * IIRC, those are the approximate position of the center of pressure
        #  * The CP is where the forces can be assumed to occur, and thus where
        #    the moment arms should be measured.
        C0 = self.wing.geometry.fc(0)
        d_cg, h_cg = self.cg_position(delta_a)

        x = self.geometry.fx(y) + (d_cg - 1/4)*C0  # FIXME?
        z = self.geometry.fz(y) + h_cg  # FIXME?
        uL = U + z*Q - y*R
        vL = V - z*P + x*R
        wL = W + y*P - x*Q

        return uL, vL, wL

    def equilibrium_parameters(self, delta_B=0):
        """Compute alpha_eq, d0, h0

        Parameters
        ----------
        delta_B : float
            The symmetric brake actuation, where `0 <= delta <= 1`

        Returns
        -------
        alpha_eq : float
            The equilibrium AOA for the given symmetric brakes actuation
        d0 : float
            The x-axis distance of the cg to the global AC
        h0 : float
            The z-axis distance of the cg to the global AC
        """

        wing = self.wing  # FIXME: cleanup

        # The integration points across the span
        N = 501
        dy = wing.geometry.b/(N - 1)  # Include the endpoint
        y = np.linspace(-wing.geometry.b/2, wing.geometry.b/2, N)

        Gammas = wing.geometry.Gamma(y)
        thetas = wing.geometry.ftheta(y)

        def calc_d0h0(wing, alpha_eq):
            """Calculate the global AC, {d0, h0}

            These points are deterministic given alpha_eq, but are used as part
            of the optimization routine to find alpha_eq.

            ref: PFD Eqs 5.44-5.45
            """

            CL = wing.CL(alpha_eq)
            CD = wing.CD(alpha_eq)

            Cli = wing.Cl(alpha_eq)
            Cdi = wing.Cd(alpha_eq)
            alpha_i = alpha_eq*cos(Gammas) + thetas

            # PFD Eq:5.44, p120
            fx = wing.geometry.fx(y)
            fz = wing.geometry.fz(y)
            fc = wing.geometry.fc(y)
            S = wing.geometry.S
            numerator = (Cli*cos(alpha_i) + Cdi*sin(alpha_i)) * fx * fc
            denominator = (CL*cos(alpha_eq) + CD*sin(alpha_eq)) * S
            d0 = trapz(numerator, dy) / denominator

            # PFD Eq:5.45, p120
            numerator = -((Cli*sin(alpha_i) - Cdi*cos(alpha_i)) *
                          fz * fc / cos(Gammas))
            denominator = (CL*sin(alpha_eq) - CD*cos(alpha_eq)) * S
            h0 = trapz(numerator, dy) / denominator

            # print("DEBUG> d0: {}, h0: {}".format(d0, h0))

            return d0, h0

        def calc_my(wing, alpha, d0=None, h0=None):
            """Optimization target for computing alpha_eq

            Parameters
            ----------
            wing : Wing
            alpha : float
                Current guess for alpha_eq

            Returns
            -------
            My : float
                The total moment about the y-axis. Should be zero for
                equilibrium.
            """
            print("DEBUG> calc_my: alpha:", alpha)

            # Update {d0, h0} to track the changing alpha
            if d0 is None:
                d0, h0 = calc_d0h0(wing, alpha)

            CL = wing.CL(alpha)
            CD = wing.CD(alpha)
            Cm = wing.Cm(alpha)
            Cz = CL*cos(alpha) + CD*sin(alpha)  # PFD eq 4.76
            Cx = CL*sin(alpha) - CD*cos(alpha)  # PFD eq 4.77
            My = Cz*d0 - Cx*h0 + Cm*wing.geometry.MAC  # PFD Eq 4.78/5.37
            return My

        # alphas = np.linspace(-1.99, 24, 50)
        # d0h0s = np.asarray([calc_d0h0(wing, np.deg2rad(a)) for a in alphas])
        # Mys = np.asarray([calc_my(wing, np.deg2rad(a)) for a in alphas])
        # print("d0h0s")
        # input("Continue?")
        # embed()

        # FIXME: Initialize alpha_eq to something reasonable
        f_alpha = partial(calc_my, wing)
        alpha_eq_prime = least_squares(f_alpha, np.deg2rad(8)).x[0]
        # alpha_eq_prime = least_squares(
        #     f_alpha, np.deg2rad(2),
        #   bounds=(np.deg2rad(1.75), np.deg2rad(2.3))).x[0]
        #   bounds=(0, np.deg2rad(15))).x[0]

        print("Finished finding alpha_eq_prime:", alpha_eq_prime)
        input("Continue?")
        embed()

    # FIXME: moved from Wing. Verify and test.
    def surface_distributions(self):
        """The surface area distributions for computing inertial moments.

        The moments of inertia for the wing are the mass distribution of the
        air and wing material. That distribution is typically decomposed into
        the product of volumetric density and volume, but a simplification is
        to calculate the density per unit area.

        FIXME: this description is mediocre.

        Ref: "Paraglider Flight Dynamics", page 48 (56)

        Returns
        ------
        S : 3x3 matrix of float
            The surface distributions, such that `J = (p_w + p_air)*s`
        """

        # FIXME: this was moved from Wing when creating Glider. That move
        #        eliminated dcg/h0 from the WingGeometry, so WingGeometry.fx
        #        and WingGeometry.fz no longer position the wing relative to
        #        the cg. Calculating the moments of inertia requires that the
        #        Glider add those terms back in.
        print("BROKEN! See the source FIXME")
        1/0

        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.geometry.b/2, self.geometry.b/2, N)

        fx = self.fx(y)  # FIXME: add the `dcg` term after moving to Glider
        fz = self.fz(y)  # FIXME: add the h0 term after moving to Glider
        fc = self.fc(y)

        # FIXME: needs verification
        # FIXME: this is a crude rectangle rule integration
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

    # FIXME: moved from Wing. Verify and test.
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
