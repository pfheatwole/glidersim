import abc

import numpy as np
from numpy import arcsin, arctan, deg2rad, sqrt, tan

from util import trapz


class ParafoilGeometry(abc.ABC):
    @property
    @abc.abstractmethod
    def b(self):
        """Span of the inflated wing"""

    @property
    @abc.abstractmethod
    def S(self):
        """Projected surface area of the inflated wing"""

    @property
    @abc.abstractmethod
    def AR(self):
        """Aspect ratio of the inflated wing"""

    @property
    @abc.abstractmethod
    def MAC(self):
        """Mean aerodynamic chord"""

    @abc.abstractmethod
    def fx(self, t):
        """Section quarter chord x coordinate"""

    @abc.abstractmethod
    def fy(self, t):
        """Section quarter chord y coordinate"""

    @abc.abstractmethod
    def fz(self, t):
        """Section quarter chord z coordinate"""

    @abc.abstractmethod
    def fc(self, t):
        """Section chord length"""

    @abc.abstractmethod
    def ftheta(self, t):
        """
        Section chord angle relative to the central airfoil (geometric torsion)
        """

    @abc.abstractmethod
    def Gamma(self, t):
        """Dihedral angle"""

    @property
    def S_flat(self):
        """Area of the flattened wing"""
        # ref: PFD p46 (54)
        # FIXME: untested
        raise NotImplementedError("FIXME: broken by reparametrization!")
        N = 501
        dy = self.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.b/2, self.b/2, N)
        return trapz(self.fc(y) * sqrt(self.dfzdy(y)**2 + 1), dy)

    @property
    def b_flat(self):
        """Span of the flattened wing"""
        # ref: PFD p47 (54)
        # FIXME: untested
        raise NotImplementedError("FIXME: broken by reparametrization!")
        N = 501
        dy = self.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.b/2, self.b/2, N)
        return trapz(sqrt(self.dfzdy(y)**2 + 1), dy)

    @property
    def AR_flat(self):
        """Aspect ratio of the flattened wing"""
        # ref: PFD p47 (54)
        # FIXME: untested
        return self.b_flat**2 / self.S_flat

    @property
    def flattening_ratio(self):
        """Percent reduction in area of the inflated wing vs the flat wing"""
        # ref: PFD p47 (54)
        # FIXME: untested
        return (1 - self.S/self.S_flat)*100


class Elliptical(ParafoilGeometry):
    """
    A parametric geometry that uses ellipses for the spanwise chord length,
    dihedral, and sweep.

    ref: PFD p43 (51)
    """

    def __init__(self, b, c0, taper, dihedralMed, dihedralMax,
                 sweepMed, sweepMax, torsion=0, linear_torsion=False):
        self._b = b
        self.c0 = c0
        self.taper = taper
        self.dihedralMed = deg2rad(dihedralMed)
        self.dihedralMax = deg2rad(dihedralMax)
        self.sweepMed = deg2rad(sweepMed)
        self.sweepMax = deg2rad(sweepMax)
        self.torsion = deg2rad(torsion)
        self.linear_torsion = linear_torsion

        # Ellipse coefficients for quarter-chord projected on the xy plane
        #  * Note: this is for the arcced wing, not the planform
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)
        self.Ax = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        self.Bx = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        self.Cx = -self.Bx - self.c0/4

        # Ellipse coefficients for quarter-chord projected on the yz plane
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)
        self.Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        self.Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        self.Cz = -self.Bz

    @property
    def b(self):
        return self._b

    @property
    def S(self):
        # ref: PFD Table:3-6, p46 (54)
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return self.c0 * self.b/2 * taper_factor

    @property
    def AR(self):
        # ref: PFD Table:3-6, p46 (54)
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return 2 * self.b / (self.c0*taper_factor)

    @property
    def MAC(self):
        # ref: PFD Table:3-6, p46 (54)
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return (2/3) * self.c0 * (2 + t**2) / taper_factor

    # @property
    def dihedral_smoothness(self):
        """A measure of the rate of change in curvature along the span"""
        # ref: PFD p47 (54)
        # FIXME: untested
        dMax, min_dMax = abs(self.dihedralMax), abs(2 * self.dihedralMed)
        ratio = (dMax - min_dMax)/(np.pi/2 - min_dMax)
        return (1 - ratio)*100

    # @property
    def sweep_smoothness(self):
        """A measure of the rate of change in sweep along the span"""
        # ref: PFD p47 (54)
        # FIXME: untested
        sMax, min_sMax = abs(self.sweepMax), abs(2 * self.sweepMed)
        ratio = (sMax - min_sMax)/(np.pi/2 - min_sMax)
        return (1 - ratio)*100

    def fx(self, t):
        # Map the -1..1 parameter onto a parameter for the truncated ellipse
        # FIXME: needs a serious design review
        tbar_min = np.arccos(self.b/(2*self.Ax))
        tbar = np.pi/2 - t*(np.pi/2 - tbar_min)
        return self.Bx * np.sin(tbar) + self.Cx

    def fy(self, t):
        # Map the -1..1 parameter onto a parameter for the truncated ellipse
        # FIXME: needs a serious design review
        tbar_min = np.arccos(self.b/(2*self.Az))
        tbar = np.pi/2 - t*(np.pi/2 - tbar_min)
        return self.Az * np.cos(tbar)

    def fz(self, t):
        # Map the -1..1 parameter onto a parameter for the truncated ellipse
        # FIXME: needs a serious design review
        tbar_min = np.arccos(self.b/(2*self.Az))
        tbar = np.pi/2 - t*(np.pi/2 - tbar_min)
        return self.Bz * np.sin(tbar) + self.Cz

    def dfxdy(self, t):
        # FIXME: untested
        tbar_min = np.arccos(self.b/(2*self.Ax))
        tbar = np.pi/2 - t*(np.pi/2 - tbar_min)
        # FIXME: needs the dtbar/dt factor?
        return -self.Bx/self.Ax/np.tan(tbar)

    def Lambda(self, t):
        """Sweep angle"""
        # FIXME: should this be part of the ParafoilGeometry interface?
        return arctan(self.dfxdy(t))

    def dfzdy(self, t):
        # FIXME: untested
        tbar_min = np.arccos(self.b/(2*self.Az))
        tbar = np.pi/2 - t*(np.pi/2 - tbar_min)
        # FIXME: needs the dtbar/dt factor?
        return -self.Bz/self.Az/np.tan(tbar)

    def Gamma(self, t):
        return arctan(self.dfzdy(t))

    def fc(self, t):
        # FIXME: untested
        Ac = (self.b/2) / sqrt(1 - self.taper**2)
        Bc = self.c0
        tbar_min = np.arccos(self.b/(2*Ac))
        tbar = np.pi/2 - t*(np.pi/2 - tbar_min)
        return Bc * np.sin(tbar)

    def ftheta(self, t):
        # if self.linear_torsion:
        #     return 2*self.torsion*np.abs(y)  # Linear
        # else:  # Use an exponential distribution of geometric torsion
        #     k = self.torsion/(np.exp(self.b/2) - 1)
        #     return k*(np.exp(np.abs(y)) - 1)
        print("DEBUG> ftheta: FIXME: implement for the t parametrization")
        return np.zeros_like(t)

    @staticmethod
    def MAC_to_c0(MAC, taper):
        """Central chord length of a tapered elliptical wing

        This geometry class is parametrized by the central chord, but the MAC
        is more commonly known. If the MAC and taper of a wing are known, then
        this function can be used to determine the equivalent central chord
        for that wing.
        """
        # ref: PFD Table:3-6, p46 (54)
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        c0 = (MAC / (2/3) / (2 + taper**2)) * (taper + tmp)
        return c0

    @staticmethod
    def AR_to_b(c0, AR, taper):
        """Compute the span of a tapered elliptical wing"""
        # ref: PFD Table:3-6, p46 (54)
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        b = (AR / 2)*c0*(taper + tmp)
        return b
