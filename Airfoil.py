"""
Models for airfoil geometry and airfoil coefficients. Geometry models are for
graphical purposes, such as drawing the wing. Coefficient models are for
evaluating the section coefficients for lift, drag, and pitching moment.
"""

import abc

import numpy as np
from numpy import arctan


class Airfoil:
    def __init__(self, coefficients, geometry=None):

        if not isinstance(coefficients, AirfoilCoefficients):
            raise ValueError("geometry is not an AirfoilGeometry")

        if geometry is None:
            geometry = NACA4(2415)

        if not isinstance(geometry, AirfoilGeometry):
            raise ValueError("geometry is not an AirfoilGeometry")

        self.coefficients = coefficients
        self.geometry = geometry


class AirfoilCoefficients(abc.ABC):
    """
    Provides functions for the aerodynamic coefficients of a wing section.

    FIXME: needs a better description
    """

    @abc.abstractmethod
    def Cl(self, alpha, delta):
        """
        Lift coefficient of the airfoil at the given angle of attack

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [radians]
            The angle of deflection of the trailing edge due to braking
        """
        # FIXME: constrain the AoA, like `-i0 < alpha < alpha_max` ?

    @abc.abstractmethod
    def Cd(self, alpha, delta):
        """
        Form drag coefficient of the airfoil at the given angle of attack

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [radians]
            The angle of deflection of the trailing edge due to braking

        Notes
        -----
        This is pressure drag only (asymmetric pressure forces across the
        airfoil due to flow separation). Calculating the skin friction drag of
        a section requires surface material properties, which are a property
        of the wing, not the airfoil (the shape of a wing cross-section).
        """
        # FIXME: constrain the AoA, like `-i0 < alpha < alpha_max` ?

    @abc.abstractmethod
    def Cm0(self, alpha, delta):
        """
        Pitching coefficient of the airfoil at the given angle of attack

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [radians]
            The angle of deflection of the trailing edge due to braking
        """


class LinearCoefficients(AirfoilCoefficients):
    """
    An airfoil model that assumes a strictly linear lift coefficient, constant
    form drag, and constant pitching moment.

    In addition, the effect of brakes is to shift the coefficient curves to the
    left. Brake deflections do not change the shape of the curves themselves.
    """

    def __init__(self, a0, i0, D0, Cm0):
        self.a0 = a0  # [1/rad]
        self.i0 = np.deg2rad(i0)
        self.D0 = D0
        self._Cm0 = Cm0  # FIXME: seems clunky; change the naming?

    def Cl(self, alpha, delta=0):
        # FIXME: verify the usage of delta
        return self.a0 * (alpha + delta - self.i0)

    def Cd(self, alpha, delta=0):
        alpha = np.asarray(alpha)
        # FIXME: verify the usage of delta
        return np.ones_like(alpha) * self.D0

    def Cm0(self, alpha, delta=0):
        return self.Cm0


# ---------------------------------------------------------------------------


class AirfoilGeometry(abc.ABC):
    """
    These are used for drawing the 3D wing, and have no effect on performance.
    """
    @property
    @abc.abstractmethod
    def t(self):
        """Maximum airfoil thickness as a percentage of chord length"""
        # ref PFD 48 (46)

    @abc.abstractmethod
    def yc(self, x):
        """Compute the y-coordinate of the mean camber line

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 < x < chord`
        """

    @abc.abstractmethod
    def yt(self, x):
        """Airfoil thickness, perpendicular to the camber line

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 < x < chord`
        """

    @abc.abstractmethod
    def fE(self, x):
        """Upper camber line corresponding to the point `x` on the chord

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 < x < chord`
        """

    @abc.abstractmethod
    def fI(self, x):
        """Lower camber line corresponding to the point `x` on the chord

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 < x < chord`
        """


class NACA4(AirfoilGeometry):
    """Airfoil geometry using a NACA4 parameterization"""

    def __init__(self, code, chord=1):
        """
        Generate a NACA4 airfoil

        Parameters
        ----------
        code : integer
            The 4-digit NACA code
        chord : float
            The length of the chord
        """
        self.chord = chord
        self.code = code
        self.m = (code // 1000) / 100       # Maximum camber
        self.p = ((code // 100) % 10) / 10  # location of max camber
        self.tcr = (code % 100) / 100       # Thickness to chord ratio
        self.pc = self.p * self.chord

    @property
    def t(self):
        return (self.code % 100) / 100

    def yc(self, x):
        m = self.m
        c = self.chord
        p = self.p
        pc = self.pc

        x = np.asarray(x)

        if np.any(x < 0) or np.any(x > c):
            raise ValueError("x must be between 0 and the chord length")

        f = x <= pc  # Filter for the two cases, `x <= pc` and `x > pc`
        cl = np.empty_like(x)
        cl[f] = (m/p**2)*(2*p*(x[f]/c) - (x[f]/c)**2)
        cl[~f] = (m/(1-p)**2)*((1-2*p) + 2*p*(x[~f]/c) - (x[~f]/c)**2)
        return cl

    def yt(self, x):
        t = self.tcr

        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > self.chord):
            raise ValueError("x must be between 0 and the chord length")

        return 5*t*(.2969*np.sqrt(x) - .126*x - .3516*x**2 +
                    .2843*x**3 - .1015*x**4)

    def _theta(self, x):
        """Angle of the mean camber line

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 < x < chord`
        """
        m = self.m
        p = self.p
        c = self.chord
        pc = self.pc

        x = np.asarray(x)
        assert np.all(x >= 0) and np.all(x <= self.chord)

        f = x <= pc  # Filter for the two cases, `x <= pc` and `x > pc`
        dyc = np.empty_like(x)
        dyc[f] = (2*m/p**2)*(p - x[f]/c)
        dyc[~f] = (2*m/(1-p)**2)*(p - x[~f]/c)

        return arctan(dyc)

    def fE(self, x):
        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > self.chord):
            raise ValueError("x must be between 0 and the chord length")

        theta = self._theta(x)
        yt = self.yt(x)
        yc = self.yc(x)
        return np.c_[x - yt*np.sin(theta), yc + yt*np.cos(theta)]

    def fI(self, x):
        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > self.chord):
            raise ValueError("x must be between 0 and the chord length")

        theta = self._theta(x)
        yt = self.yt(x)
        yc = self.yc(x)
        return np.c_[x + yt*np.sin(theta), yc - yt*np.cos(theta)]
