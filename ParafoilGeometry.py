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
    def fx(self, y):
        """Section quarter chord projected onto the XY plane"""

    @abc.abstractmethod
    def fz(self, y):
        """Section quarter chord projected onto the YZ plane"""

    @abc.abstractmethod
    def fc(self, y):
        """Section chord length"""

    @abc.abstractmethod
    def ftheta(self, y):
        """
        Section chord angle relative to the central airfoil (geometric torsion)
        """

    @abc.abstractmethod
    def Gamma(self, y):
        """Dihedral angle"""

    @property
    def S_flat(self):
        """Area of the flattened wing"""
        # ref: PFD p46 (54)
        # FIXME: untested
        N = 501
        dy = self.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.b/2, self.b/2, N)
        return trapz(self.fc(y) * sqrt(self.dfzdy(y)**2 + 1), dy)

    @property
    def b_flat(self):
        """Span of the flattened wing"""
        # ref: PFD p47 (54)
        # FIXME: untested
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

    def fx(self, y):
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)

        Ax = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bx = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        # Cx = -Bx + (self.dcg - 1/4)*self.c0
        # Cx = -Bx  # Modified from the original definition in PFD
        Cx = -Bx - self.c0/4  # Modified v2: set the central LE as the origin

        return Bx * sqrt(1 - (y**2)/Ax**2) + Cx

    def fz(self, y):
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)

        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        # Cz = -Bz - self.h0
        Cz = -Bz  # Modified from the original definition in PFD

        return Bz * sqrt(1 - (y**2)/Az**2) + Cz

    def dfxdy(self, y):
        # FIXME: untested
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)
        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        return Bz * -y / (Az**2 * sqrt(1 - y**2/Az**2))

    def Lambda(self, y):
        """Sweep angle"""
        # FIXME: should this be part of the ParafoilGeometry interface?
        return arctan(self.dfxdy(y))

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

    def ftheta(self, y):
        if self.linear_torsion:
            return 2*self.torsion/self.b*np.abs(y)  # Linear
        else:  # Use an exponential distribution of geometric torsion
            k = self.torsion/(np.exp(self.b/2) - 1)
            return k*(np.exp(np.abs(y)) - 1)

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
