import abc

import numpy as np
from numpy import arcsin, arctan, deg2rad, sqrt, tan

from util import trapz


class ParafoilGeometry(abc.ABC):
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


class Elliptical(ParafoilGeometry):
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
        """Quarter-chord x-coordinate (central leading edge as the origin)"""
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)

        Ax = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bx = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        # Cx = -Bx + (self.dcg - 1/4)*self.c0
        # Cx = -Bx  # Modified from the original definition in PFD
        Cx = -Bx - self.c0/4  # Modified v2: set the central LE as the origin

        return Bx * sqrt(1 - (y**2)/Ax**2) + Cx

    def fz(self, y):
        """Quarter-chord z-coordinate (central leading edge as the origin)"""
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
        return arctan(self.dfxdy(y))

    def dfzdy(self, y):
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)
        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        return Bz * -y / (Az**2 * sqrt(1 - y**2/Az**2))

    def Gamma(self, y):
        """Dihedral angle"""
        return arctan(self.dfzdy(y))

    def fc(self, y):
        """Chord length"""
        Ac = (self.b/2) / sqrt(1 - self.taper**2)
        Bc = self.c0
        return Bc * sqrt(1 - (y**2)/Ac**2)

    def ftheta(self, y, linear=False):
        """Geometric torsion"""
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