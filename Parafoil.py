import abc

import numpy as np
from numpy import sin, cos, arctan
from numpy.polynomial import Polynomial

from util import trapz

from IPython import embed


class Parafoil:
    def __init__(self, geometry, airfoil, wing_density=0.2):
        self.geometry = geometry
        self.airfoil = airfoil
        self.wing_density = wing_density  # FIXME: no idea in general

    @property
    def density_factor(self):
        # FIXME: I don't understand this. Ref: PFD 48 (56)
        return self.geometry.MAC * self.airfoil.t*self.airfoil.chord/3

    def section_forces(self, y, uL, vL, wL, coefs, delta_B=0, rho=1):
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
        delta_B : float [percentage]
            The amount of symmetric brake
        rho : float
            Air density
        use_2d : bool
            Use coefficients from the the 2D airfoil instead of the 3D wing

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
        # delta_i = self.brake.delta(y, delta_B, delta_B)
        delta_i = np.zeros_like(alpha_i)

        Cl = coefs.Cl(y, alpha_i, delta_i)
        Cd = coefs.Cd(y, alpha_i, delta_i)
        Cm = coefs.Cm(y, alpha_i, delta_i)

        # PFD Eq:4.65-4.67
        # NOTE: this does not include the `dy` term as in those equations
        fc = self.geometry.fc(y)
        K1 = (rho/2)*(ui**2 + wi**2)
        K2 = 1/cos(Gamma)
        dLi = K1*Cl*fc*K2
        dDi = K1*Cd*fc*K2
        dm0i = K1*Cm*(fc**2)*K2

        # Translate the section forces and moments into body axes
        #  * PFD Eqs:4.23-4.27, p76
        F_par_x = dLi*sin(alpha_i - theta) - dDi*cos(alpha_i - theta)
        F_perp_x = dLi*cos(alpha_i - theta) + dDi*sin(alpha_i - theta)
        F_par_y = F_perp_x * sin(Gamma)
        F_par_z = F_perp_x * cos(Gamma)
        mi_par_y = dm0i*cos(Gamma)

        return F_par_x, F_par_y, F_par_z, mi_par_y


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


class CoefficientsEstimator(abc.ABC):
    def Cl(self, y, alpha, delta):
        """The lift coefficient for the parafoil section"""

    def Cd(self, y, alpha, delta):
        """The drag coefficient for the parafoil section"""

    def Cm(self, y, alpha, delta):
        """The pitching moment coefficient for the parafoil section"""


class Coefs2D(CoefficientsEstimator):
    """
    Returns the 2D airfoil coefficients with no adjustments.

    FIXME: Only works for parafoils with constant airfoils
    """

    def __init__(self, parafoil):
        self.parafoil = parafoil

    def Cl(self, y, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha
        if np.isscalar(delta):
            delta = np.ones_like(y) * delta
        return self.parafoil.airfoil.coefficients.Cl(alpha, delta)

    def Cd(self, y, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha
        if np.isscalar(delta):
            delta = np.ones_like(y) * delta
        return self.parafoil.airfoil.coefficients.Cd(alpha, delta)

    def Cm(self, y, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha
        if np.isscalar(delta):
            delta = np.ones_like(y) * delta
        return self.parafoil.airfoil.coefficients.Cm0(alpha, delta)


class CoefsPFD(CoefficientsEstimator):
    """
    The coefficient estimation method from Paraglider Flight Dynamics

    This converts the wing into a uniform straight wing, so it eliminates the
    dependency on the span position.
    """

    def __init__(self, parafoil):
        self.parafoil = parafoil

        # Initialize the global (3D) aerodynamic coefficients
        CL, CD, Cm = self._compute_global_coefficients()
        self.CL = CL
        self.CD = CD
        self.Cm = Cm

        # Initialize the section (2D) aerodynamic coefficients
        # FIXME


    def Cl(self, y, alpha, delta):
        # TODO: test, verify, and adapt for `delta`
        # TODO: document
        i0 = self.CL.roots()[0]  # Adjusted global zero-lift angle
        return self.a_bar * (alpha - i0)

    def Cd(self, y, alpha, delta):
        # TODO: test, verify, and adapt for `delta`
        # TODO: document
        i0 = self.CL.roots()[0]  # Adjusted global zero-lift angle
        D0 = self.CD(i0)
        return self.D2_bar*(self.a_bar * (alpha - i0))**2 + D0

    def Cm(self, y, alpha, delta):
        # TODO: test, verify, and adapt for `delta`
        # TODO: document
        return self.Cm_bar


    def _pointwise_global_coefficients(self, alpha, delta):
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
        Cm : float
        """

        # Build a set of integration points
        N = 501
        dy = self.parafoil.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.parafoil.geometry.b/2,
                        self.parafoil.geometry.b/2, N)

        # Compute local relative winds to match `alpha`
        uL, wL = np.cos(alpha), np.sin(alpha)

        # Compute the local forces
        coefs = Coefs2D(self.parafoil)
        F_par_x, F_par_y, F_par_z, mi_par_y = self.parafoil.section_forces(
            y, uL, 0, wL, coefs=coefs)

        # Convert the body-oriented forces into relative wind coordinates
        # PFD Eq:4.29-4.30, p77
        Li_prime = F_par_z*cos(alpha) + F_par_x*sin(alpha)
        Di_prime = -F_par_x*cos(alpha) + F_par_z*sin(alpha)

        L = trapz(Li_prime, dy)
        D = trapz(Di_prime, dy)
        My = trapz(mi_par_y, dy)

        # Compute the global coefficients
        rho = 1  # FIXME: is it correct to normalize around rho=1?
        S = self.parafoil.geometry.S
        CL = L/((rho/2) * (uL**2 + wL**2) * S)
        CD = D/((rho/2) * (uL**2 + wL**2) * S)
        Cm = My/((rho/2) * (uL**2 + wL**2) * S * self.parafoil.geometry.MAC)

        if np.isnan(CL):
            print("\nBLARGH! CL is np.nan\n")
            embed()
            input("continue?")

        return CL, CD, Cm

    def _compute_global_coefficients(self):
        """
        Fit polynomials to the adjusted global aerodynamic coefficients.

        This procedure is from Paraglider Flight Dynamics, Sec:4.3.4

        FIXME: the alpha range should depend on the airfoil.
        FIXME: currently assumes linear CL, with no expectation of stalls!!
        FIXME: for non-constant-linear airfoils, do these fittings hold?
        FIXME: seems convoluted for what it accomplishes
        """
        alphas = np.deg2rad(np.linspace(-1.99, 24, 1000))
        CLs = np.empty_like(alphas)
        CDs = np.empty_like(alphas)
        Cms = np.empty_like(alphas)

        # First, compute the unadjusted global coefficients
        for n, alpha in enumerate(alphas):
            CL, CD, Cm = self._pointwise_global_coefficients(alpha, delta=0)
            CLs[n], CDs[n], Cms[n] = CL, CD, Cm

        # Second, adjust the global coefficients to account for the 3D wing
        # FIXME: Verify!

        # FIXME: already losing data with a linear fit!!!
        # CL_prime = Polynomial.fit(alphas, CLs, 1)
        mask = (alphas > np.deg2rad(-3)) & (alphas < np.deg2rad(9))
        CL_prime = Polynomial.fit(alphas[mask], CLs[mask], 1)

        i0 = CL_prime.roots()[0]  # Unadjusted global zero-lift angle
        # i0 = alphas[np.argmin(np.abs(CLs))]

        a_prime = CL_prime.deriv()(0.05)  # Preliminary lift-curve slope
        a = a_prime/(1 + a_prime/(np.pi * self.parafoil.geometry.AR))  # Adjusted slope

        # FIXME: should not rely on the AirfoilCoefficients providing D0
        # D0 = self.airfoil.coefficients.D0  # Profile drag only
        D0 = self.parafoil.airfoil.coefficients.Cd(alphas)

        D2_prime = CDs/CLs**2 - D0
        D2 = D2_prime + 1/(np.pi * self.parafoil.geometry.AR)  # Adjusted induced drag

        # FIXME: this derivation and fitting for D2 seems strange to me. He
        # assumes a fixed D0 for the airfoil; the CDs are basically D0, but
        # with a small variation due to the 3D wing shape. Thus, aren't `CDs`
        # essentially the alpha-dependent profile drag? Then why does he
        # subtract D0 to calculate D2_prime, then add back D0 to calculate CD,
        # instead of subtracting and adding CDs, the "true" profile drag?

        # FIXME: compare these results against the wings on PFD pg 97

        CL = Polynomial.fit(alphas, a*(alphas - i0), 1)
        CD = Polynomial.fit(alphas, D2 * (CL(alphas)**2) + D0, 2)
        Cm = Polynomial.fit(alphas, Cms, 2)

        # ------------------------------
        # Alternative formulation
        i0x = alphas[np.argmin(np.abs(CLs))]
        x = np.sqrt((a_prime/a))*(alphas - i0x) + i0x
        y = np.sqrt((a/a_prime))*CLs
        CLx = Polynomial.fit(x, y, 5)

        # Have to extend the coverage since alpha (x) got skewed
        D0x = self.parafoil.airfoil.coefficients.Cd(alphas)
        e = 0.95  # Efficiency factor
        Di = CLx(x)**2 / (np.pi * e * self.parafoil.geometry.AR)
        CDx = Polynomial.fit(alphas, Di + D0x, 5)

        # ------------------------------

        embed()

        return CL, CD, Cm

    def _pointwise_local_coefficients(self, alpha, delta_B):
        """
        This is my replacement for PFD Sec:4.3.5
        """
        N = 501
        y = np.linspace(-self.parafoil.geometry.b/2,
                        self.parafoil.geometry.b/2, 501)
        dy = self.parafoil.geometry.b/(N - 1)

        Gamma = self.parafoil.geometry.Gamma(y)
        theta = self.parafoil.geometry.ftheta(y)
        c = self.parafoil.geometry.fc(y)

        uL, wL = np.cos(alpha), np.sin(alpha)
        ui = uL
        wi = wL*cos(Gamma)
        alpha_i = np.arctan(wi/ui) + theta

        i0 = self.CL.roots()[0]
        D0 = self.CD(i0)

        numerator = self.CL(alpha)*self.geometry.S
        denominator = trapz((alpha_i - i0)*c/cos(Gamma), dy)
        a_bar = numerator/denominator

        numerator = self.CD(alpha)*self.geometry.S - trapz(D0*c/cos(Gamma), dy)
        Cl = a_bar*(alpha_i - i0)
        denominator = trapz(Cl**2 * c / cos(Gamma), dy)
        D2_bar = numerator/denominator

        self.a_bar = a_bar
        self.D2_bar = D2_bar
        return a_bar, D2_bar

    def _pointwise_local_coefficients_PFD(self, alpha_eq):
        """
        This procedure is from PFD Sec:4.3.5
        """
        N = 501
        dy = self.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.parafoil.geometry.b/2,
                        self.parafoil.geometry.b/2, 501)

        Gamma = self.geometry.Gamma(y)
        theta = self.geometry.ftheta(y)

        alpha_i = alpha_eq*cos(Gamma) + theta  # PFD Eq:4.46, p82

        CL_eq = self.CL(alpha_eq)
        CD_eq = self.CD(alpha_eq)
        Cm0_eq = self.Cm(alpha_eq)

        # Recompute some stuff from `global_coefficients`
        a = self.CL.deriv()(0)  # FIXME: is this SUPPOSED to be a_global?
        i0 = self.CL.roots()[0]  # FIXME: test
        D0 = self.CD(i0)  # Global D0
        D2 = (self.CD(alpha_eq) - D0) / CL_eq**2  # PFD eq 4.39

        tmp_i_local = alpha_i - self.parafoil.airfoil.coefficients.i0
        tmp_i_global = alpha_i - i0
        NZL = (1/10**5)**(1 - tmp_i_local/np.abs(tmp_i_local))

        fc = self.parafoil.geometry.fc(y)
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

        # Version 1: From PFD; susceptible to divison by zero (via SL2)
        #  * Specifically: when torsion=0 and alpha=0 (and thus sin(alpha) = 0)
        # a_bar = (S*(SD2/SL2*a*(alpha_eq - i0) - (D2*CL_eq**2 + D0)) -
        #          (SD2/SL2*SL3-SD3)) / (SD2/SL2*SL1 - SD1)
        # D2_bar = (S*a*(alpha_eq - i0) - a_bar*SL1 - SL3)/(a_bar**2*SL2)

        # This version is less susceptible for division-by-zero?
        a_bar = (S*CL_eq - ((S*CD_eq - SD3)/SD2)*SL2 - SL3)/(SL1 - (SD1/SD2)*SL2)
        D2_bar = (S*CD_eq - a_bar*SD1 - SD3)/(a_bar**2 * SD2)

        Cm0_bar = Cm0_eq  # FIXME: correct? ref: "median", PFD p79

        # Update the section coefficients for calculating local forces
        self.a_bar = a_bar
        self.D2_bar = D2_bar
        self.Cm0_bar = Cm0_bar
        return a_bar, D2_bar, Cm0_bar


class CoefsMine(CoefficientsEstimator):
    pass
