import numpy as np
from numpy import sqrt, sin, cos, tan, arctan, deg2rad


class Wing:
    def __init__(self, geometry, airfoil):
        self.geometry = geometry
        self.airfoil = airfoil

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
        upper = fc*self.airfoil.fE(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        delta = arctan(self.geometry.dfzdy(y))

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) - zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(delta)
        z = np.abs(
            -self.geometry.fz(y) +
            ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(delta)
            )

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
        upper = fc*self.airfoil.fI(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        delta = arctan(self.geometry.dfzdy(y))

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) + zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(delta)
        z = np.abs(
                -self.geometry.fz(y) +
                ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(delta)
                )
        return np.c_[x, _y, z]


class WingGeometry:
    def __init__(self, c0, h0, dcg):
        """
        Build a wing from a parameterized description

        Parameters
        ----------
        c0, h0, dcg: float
            Configure the basic 2D triangle: ref page 34 (42)

        """
        self.c0 = c0
        self.h0 = h0
        self.dcg = dcg

    def fx(self, y):
        """The quarter chord projected onto the XY plane"""
        raise NotImplementedError("WingGeometry is a base class")

    def fz(self, y):
        """The quarter chord projected onto the YZ plane"""
        raise NotImplementedError("WingGeometry is a base class")

    def fc(self, y):
        """Chord length along the span"""
        raise NotImplementedError("WingGeometry is a base class")

    def ft(self, y):
        """Spanwise airfoil chord angle relative to the central airfoil"""
        raise NotImplementedError("WingGeometry is a base class")


class EllipticalWing(WingGeometry):
    """Ref: Paraglider Flying Dynamics, page 43 (51)"""

    def __init__(self, dcg, c0, h0, dihedralMed, dihedralMax, b, taper,
                 sweepMed, sweepMax, torsion=0):
        self.dcg = dcg
        self.c0 = c0
        self.h0 = h0
        self.dihedralMed = deg2rad(dihedralMed)
        self.dihedralMax = deg2rad(dihedralMax)
        self.sweepMed = deg2rad(sweepMed)
        self.sweepMax = deg2rad(sweepMax)
        self.b = b
        self.torsion = deg2rad(torsion)
        self.taper = taper

    def fx(self, y):
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)

        Ax = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bx = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        Cx = -Bx + (self.dcg - 1/4)*self.c0

        return Bx * sqrt(1 - (y**2)/Ax**2) + Cx

    def fz(self, y):
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)

        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        Cz = -Bz - self.h0

        return Bz * sqrt(1 - (y**2)/Az**2) + Cz

    def dfzdy(self, y):
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)
        Az = (self.b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        Bz = (self.b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        return Bz * -y / (Az**2 * sqrt(1 - y**2/Az**2))

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
