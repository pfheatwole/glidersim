import abc

import numpy as np
from numpy import sqrt, sin, cos, tan, arcsin, arctan, deg2rad
from numpy.polynomial import Polynomial

from IPython import embed

from numba import njit


@njit(cache=True)
def trapz(y, dx):
    # Trapezoidal integrator
    return np.sum(y[1:] + y[:-1]) / 2.0 * dx


def integrate(f, a, b, N):
    if N % 2 == 0:
        raise ValueError("trapezoid integration requires odd N")
    x = np.linspace(a, b, N)
    dx = (b - a)/(N - 1)  # Include the endpoints
    y = f(x)
    return trapz(y, dx)


class Wing:
    def __init__(self, geometry, airfoil, wing_density=0.2):
        self.geometry = geometry
        self.airfoil = airfoil
        self.wing_density = wing_density  # FIXME: no idea in general

        # Initialize the global (3D) aerodynamic coefficients
        CL, CD, Cm0 = self._compute_global_coefficients()
        self.CL = CL
        self.CD = CD
        self.Cm = Cm0

        # Initialize the section (2D) aerodynamic coefficients
        # FIXME

    @property
    def density_factor(self):
        # FIXME: I don't understand this. Ref: PFD 48 (56)
        return self.geometry.MAC * self.airfoil.t*self.airfoil.chord/3

    def Cl(self, alpha):
        # TODO: test, verify, and adapt for `delta`
        # TODO: document
        i0 = self.CL.roots()[0]  # Adjusted global zero-lift angle
        return self.a_bar * (alpha - i0)

    def Cd(self, alpha):
        # TODO: test, verify, and adapt for `delta`
        # TODO: document
        i0 = self.CL.roots()[0]  # Adjusted global zero-lift angle
        D0 = self.CD(i0)
        return self.D2_bar*(self.a_bar * (alpha - i0))**2 + D0

    def Cm0(self, alpha):
        # TODO: test, verify, and adapt for `delta`
        # TODO: document
        return self.Cm0_bar    # FIXME: I'm buzzing, fix this crap

    def section_forces(self, y, uL, vL, wL, rho=1):
        """
        Compute section forces and moments acting on the wing.

        Parameters
        ----------
        y : float or array of float
            The section position on the wing span, where -b/2 < y < b/2
        uL : float or array of float
            Section-local relative wind aligned to the x-axis
        vL : float or array of float
            Section-local relative wind aligned to the y-axis
        wL : float or array of float
            Section-local relative wind aligned to the y-axis

        Returns
        -------
        dLi : float
            Lift per unit span
        dDi : float
            Drag per unit span
        dm0i : float
            Moment per unit span
        """

        # FIXME: update docstring. These are forces per unit span, not the
        #        forces themselves. This function is suitable for an
        #        integration routine; `section_forces` is a deceptive name.
        # FIXME: the "Returns" in the docstring is also wrong

        Gamma = self.geometry.Gamma(y)  # PFD eq 4.13, p74
        theta = self.geometry.ftheta(y)  # FIXME: should include braking

        ui = uL  # PFD Eq:4.14, p74
        wi = wL*cos(Gamma) - vL*sin(Gamma)  # PFD Eq:4.15, p74

        alpha_i = arctan(wi/ui) + theta  # PFD Eq:4.17, p74

        # PFD Eq:4.65-4.67
        # NOTE: this does not include the `dy` term as in those equations
        fc = self.geometry.fc(y)
        K1 = (rho/2)*(ui**2 + wi**2)
        K2 = 1/cos(Gamma)
        dLi = K1*self.airfoil.coefficients.Cl(alpha_i)*fc*K2
        dDi = K1*self.airfoil.coefficients.Cd(alpha_i)*fc*K2
        dm0i = K1*self.airfoil.coefficients.Cm0(alpha_i)*(fc**2)*K2

        # Translate the section forces and moments into body axes
        #  * PFD Eqs:4.23-4.27, p76
        F_par_x = dLi*sin(alpha_i - theta) - dDi*cos(alpha_i - theta)
        F_perp_x = dLi*cos(alpha_i - theta) + dDi*sin(alpha_i - theta)
        F_par_y = F_perp_x * sin(Gamma)
        F_par_z = F_perp_x * cos(Gamma)
        mi_par_y = dm0i*cos(Gamma)

        return F_par_x, F_par_y, F_par_z, mi_par_y

    def _pointwise_global_coefficients(self, alpha):
        """
        Compute point estimates of the global CL, CD, and Cm

        This procedure is from Paraglider Flight Dynamics, Sec:4.3.3

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack for the current point estimate

        Returns
        -------
        CL : float
        CD : float
        Cm0 : float
        """

        # Build a set of integration points
        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.geometry.b/2, self.geometry.b/2, N)

        # Compute local relative winds to match `alpha`
        uL, wL = np.cos(alpha), np.sin(alpha)

        # Compute the local forces
        F_par_x, F_par_y, F_par_z, mi_par_y = self.section_forces(y, uL, 0, wL)

        # Convert the body-oriented forces into relative wind coordinates
        # PFD Eq:4.29-4.30, p77
        Li_prime = F_par_z*cos(alpha) + F_par_x*sin(alpha)
        Di_prime = -F_par_x*cos(alpha) + F_par_z*sin(alpha)

        L = trapz(Li_prime, dy)
        D = trapz(Di_prime, dy)
        My = trapz(mi_par_y, dy)

        # Compute the global coefficients
        rho = 1  # FIXME: is it correct to normalize around rho=1?
        S = self.geometry.S
        CL = L/((rho/2) * (uL**2 + wL**2) * S)
        CD = D/((rho/2) * (uL**2 + wL**2) * S)
        Cm0 = My/((rho/2) * (uL**2 + wL**2) * S * self.geometry.MAC)

        return CL, CD, Cm0

    def _compute_global_coefficients(self):
        """
        Fit polynomials to the adjusted global aerodynamic coefficients.

        This procedure is from Paraglider Flight Dynamics, Sec:4.3.4

        FIXME: the alpha range should depend on the airfoil.
        FIXME: currently assumes linear CL, with no expectation of stalls!!
        FIXME: for non-constant-linear airfoils, do these fittings hold?
        FIXME: seems convoluted for what it accomplishes
        """
        alphas = np.deg2rad(np.linspace(-1.99, 25, 1000))
        CLs = np.empty_like(alphas)
        CDs = np.empty_like(alphas)
        Cm0s = np.empty_like(alphas)

        # First, compute the unadjusted global coefficients
        for n, alpha in enumerate(alphas):
            CL, CD, Cm = self._pointwise_global_coefficients(alpha)
            CLs[n], CDs[n], Cm0s[n] = CL, CD, Cm

        # Second, adjust the global coefficients to account for the 3D wing
        # FIXME: Verify!
        CL_prime = Polynomial.fit(alphas, CLs, 1)
        i0 = CL_prime.roots()[0]  # Unadjusted global zero-lift angle
        a_prime = CL_prime.deriv()(0)  # Preliminary lift-curve slope
        a = a_prime/(1 + a_prime/(np.pi * self.geometry.AR))  # Adjusted slope
        D0 = self.airfoil.coefficients.D0  # Profile drag only
        D2_prime = CDs/CLs**2 - D0
        D2 = D2_prime + 1/(np.pi * self.geometry.AR)  # Adjusted induced drag

        # FIXME: this derivation and fitting for D2 seems strange to me. He
        # assumes a fixed D0 for the airfoil; the CDs are basically D0, but
        # with a small variation due to the 3D wing shape. Thus, aren't `CDs`
        # essentially the alpha-dependent profile drag? Then why does he
        # subtract D0 to calculate D2_prime, then add back D0 to calculate CD,
        # instead of subtracting and adding CDs, the "true" profile drag?

        # FIXME: compare these results against the wings on PFD pg 97

        CL = Polynomial.fit(alphas, a*(alphas - i0), 1)
        CD = Polynomial.fit(alphas, D2 * (CL(alphas)**2) + D0, 2)
        Cm0 = Polynomial.fit(alphas, Cm0s, 2)

        return CL, CD, Cm0

    def set_local_coefficients(self, alpha_eq):
        """
        This procedure is from PFD Sec:4.3.5
        """
        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        ys = np.linspace(-self.geometry.b/2, self.geometry.b/2, N)

        Gamma = self.geometry.Gamma(ys)
        theta = self.geometry.ftheta(ys)

        alpha_i = alpha_eq*cos(Gamma) + theta  # PFD Eq:4.46, p82

        i0 = self.CL.roots()[0]  # FIXME: test
        tmp_i_local = alpha_i - self.airfoil.coefficients.i0
        tmp_i_global = alpha_i - i0

        NZL = (1/10**5)**(1 - tmp_i_local/np.abs(tmp_i_local))

        CL_eq = self.CL(alpha_eq)
        # CD_eq = self.CD(alpha_eq)
        Cm0_eq = self.Cm(alpha_eq)

        # Recompute some stuff from `global_coefficients`
        a = self.CL.deriv()(0)  # FIXME: is this SUPPOSED to be a_global?
        D0 = self.CD(i0)  # Global D0
        D2 = (self.CD(alpha_eq) - D0) / CL_eq**2  # PFD eq 4.39

        fc = self.geometry.fc(ys)
        # PFD Equations 4.50-4.52 p74 (p82)
        # Note: the `dy` is left off at this point, and must be added later
        KX1 = NZL*tmp_i_global*sin(alpha_i - theta)/cos(Gamma)*fc
        KX2 = -(tmp_i_global**2)*cos(alpha_i - theta)/cos(Gamma)*fc
        KX3 = -D0*fc*cos(alpha_i - theta)/cos(Gamma)*fc

        # PFD Equations 4.53-4.55 p75 (p83)
        KZ1 = NZL*tmp_i_global*cos(alpha_i - theta)*fc
        KZ2 = (tmp_i_global**2)*sin(alpha_i - theta)*fc
        KZ3 = D0*fc*sin(alpha_i - theta)*fc

        # PFD equations 4.58-4.59, p75 (p83)
        KL1 = KZ1*cos(alpha_eq) + KX1*sin(alpha_eq)
        KL2 = KZ2*cos(alpha_eq) + KX2*sin(alpha_eq)
        KL3 = KZ3*cos(alpha_eq) + KX3*sin(alpha_eq)
        KD1 = KZ1*sin(alpha_eq) - KX1*cos(alpha_eq)
        KD2 = KZ2*sin(alpha_eq) - KX2*cos(alpha_eq)
        KD3 = KZ3*sin(alpha_eq) - KX3*cos(alpha_eq)

        # PFD eq 4.61, p76 (p84)
        SL1 = trapz(KL1, dy)
        SL2 = trapz(KL2, dy)
        SL3 = trapz(KL3, dy)
        SD1 = trapz(KD1, dy)
        SD2 = trapz(KD2, dy)
        SD3 = trapz(KD3, dy)

        S = self.geometry.S
        a_bar = (S*(SD2/SL2*a*(alpha_eq - i0) - (D2*CL_eq**2 + D0)) -
                 (SD2/SL2*SL3-SD3)) / (SD2/SL2*SL1 - SD1)

        D2_bar = (S*a*(alpha_eq - i0) - a_bar*SL1 - SL3)/(a_bar**2*SL2)

        Cm0_bar = Cm0_eq  # FIXME: correct? ref: "median", PFD p79

        # Update the section coefficients for calculating local forces
        self.a_bar = a_bar
        self.D2_bar = D2_bar
        self.Cm0_bar = Cm0_bar

    def fE(self, y, xa=None, N=150):
        """Airfoil upper camber line on the 3D wing

        Parameters
        ----------
        y : float
            Position on the span, where `-b/2 < y < b/2`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord
        """

        if xa is None:
            xa = np.linspace(0, 1, N)

        fc = self.geometry.fc(y)  # Chord length at `y` on the span
        upper = fc*self.airfoil.geometry.fE(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        Gamma = self.geometry.Gamma(y)

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) - zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(Gamma)
        z = self.geometry.fz(y) - \
            ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(Gamma)

        return np.c_[x, _y, z]

    def fI(self, y, xa=None, N=150):
        """Airfoil lower camber line on the 3D wing

        Parameters
        ----------
        y : float
            Position on the span, where `-b/2 < y < b/2`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord
        """

        if xa is None:
            xa = np.linspace(0, 1, N)

        fc = self.geometry.fc(y)  # Chord length at `y` on the span
        upper = fc*self.airfoil.geometry.fI(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        Gamma = self.geometry.Gamma(y)

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) + zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(Gamma)
        z = self.geometry.fz(y) - \
            ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(Gamma)

        return np.c_[x, _y, z]


class WingGeometry(abc.ABC):
    @property
    @abc.abstractmethod
    def S(self):
        """Projected surface area"""

    @property
    @abc.abstractmethod
    def AR(self):
        """Aspect ratio"""

    @property
    @abc.abstractmethod
    def MAC(self):
        """Mean aerodynamic chord"""

    @abc.abstractmethod
    def fx(self, y):
        """The quarter chord projected onto the XY plane"""

    @abc.abstractmethod
    def fz(self, y):
        """The quarter chord projected onto the YZ plane"""

    @abc.abstractmethod
    def fc(self, y):
        """Chord length along the span"""

    @abc.abstractmethod
    def ftheta(self, y):
        """Spanwise airfoil chord angle relative to the central airfoil"""

    @property
    def S_flat(self):
        """The area of the flattened wing"""
        # ref: PFD 46 (54)
        # FIXME: untested
        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.geometry.b/2, self.geometry.b/2, N)
        return trapz(self.fc(y) * sqrt(self.dfzdy(y)**2 + 1), dy)

    @property
    def b_flat(self):
        """The span of the flattened wing"""
        # ref: PFD 47 (54)
        # FIXME: untested
        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.geometry.b/2, self.geometry.b/2, N)
        return trapz(sqrt(self.dfzdy(y)**2 + 1), dy)

    @property
    def AR_flat(self):
        """The aspect ratio of the flattened wing"""
        # ref: PFD 47 (54)
        # FIXME: untested
        return self.b_flat**2 / self.S_flat

    @property
    def flattening_ratio(self):
        """Percent reduction in area of the inflated wing vs the flat wing"""
        # ref: PFD 47 (54)
        # FIXME: untested
        return (1 - self.S/self.S_flat)*100


class EllipticalWing(WingGeometry):
    """Ref: Paraglider Flying Dynamics, page 43 (51)"""

    def __init__(self, b, c0, taper, dihedralMed, dihedralMax,
                 sweepMed, sweepMax, torsion=0):
        self.b = b
        self.c0 = c0
        self.taper = taper
        self.dihedralMed = deg2rad(dihedralMed)
        self.dihedralMax = deg2rad(dihedralMax)
        self.sweepMed = deg2rad(sweepMed)
        self.sweepMax = deg2rad(sweepMax)
        self.torsion = deg2rad(torsion)

    @property
    def S(self):
        # ref: PDF 46 (54)
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return self.c0 * self.b/2 * taper_factor

    @property
    def AR(self):
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return 2 * self.b / (self.c0*taper_factor)

    @property
    def MAC(self):
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return (2/3) * self.c0 * (2 + t**2) / taper_factor

    # @property
    def dihedral_smoothness(self):
        """A measure of the rate of change in curvature along the span"""
        # ref: PFD 47 (54)
        # FIXME: untested
        dMax, min_dMax = abs(self.dihedralMax), abs(2 * self.dihedralMed)
        ratio = (dMax - min_dMax)/(np.pi/2 - min_dMax)
        return (1 - ratio)*100

    # @property
    def sweep_smoothness(self):
        """A measure of the rate of change in sweep along the span"""
        # ref: PFD 47 (54)
        # FIXME: untested
        sMax, min_sMax = abs(self.sweepMax), abs(2 * self.sweepMed)
        ratio = (sMax - min_sMax)/(np.pi/2 - min_sMax)
        return (1 - ratio)*100

    def fx(self, y):
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)

        Ax = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bx = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        # Cx = -Bx + (self.dcg - 1/4)*self.c0
        Cx = -Bx  # Modified from the original definition in PFD

        return Bx * sqrt(1 - (y**2)/Ax**2) + Cx

    def fz(self, y):
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)

        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        # Cz = -Bz - self.h0
        Cz = -Bz  # Modified from the original definition in PFD

        return Bz * sqrt(1 - (y**2)/Az**2) + Cz

    def dfzdy(self, y):
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)
        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        return Bz * -y / (Az**2 * sqrt(1 - y**2/Az**2))

    def Gamma(self, y):
        return arctan(self.dfzdy(y))

    def fc(self, y):
        Ac = (self.b/2) / sqrt(1 - self.taper**2)
        Bc = self.c0
        return Bc * sqrt(1 - (y**2)/Ac**2)

    def ftheta(self, y, linear=False):
        if linear:
            return 2*self.torsion/self.b*np.abs(y)  # Linear
        else:  # Use an exponential distribution of geometric torsion
            k = self.torsion/(np.exp(self.b/2) - 1)
            return k*(np.exp(np.abs(y)) - 1)

    @staticmethod
    def MAC_to_c0(MAC, taper):
        """Compute the central chord length of a tapered elliptical wing"""
        # PFD Table:3-6, p54
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        c0 = (MAC / (2/3) / (2 + taper**2)) * (taper + tmp)
        return c0

    @staticmethod
    def AR_to_b(c0, AR, taper):
        """Compute the span of a tapered elliptical wing"""
        # PFD Table:3-6, p54
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        b = (AR / 2)*c0*(taper + tmp)
        return b
