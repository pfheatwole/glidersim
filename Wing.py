import abc

import numpy as np
from numpy import sqrt, sin, cos, tan, arcsin, arctan, deg2rad


class Wing:
    def __init__(self, geometry, airfoil, wing_density=0.2):
        self.geometry = geometry
        self.airfoil = airfoil
        self.wing_density = wing_density  # FIXME: no idea in general

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
        delta = self.geometry.delta(y)

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
        upper = fc*self.airfoil.geometry.fI(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        delta = self.geometry.delta(y)

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) + zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(delta)
        z = np.abs(
                -self.geometry.fz(y) +
                ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(delta)
                )
        return np.c_[x, _y, z]

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

    @property
    def density_factor(self):
        # FIXME: I don't understand this. Ref: PFD 48 (46)
        return self.geometry.MAC * self.airfoil.t*self.airfoil.chord/3


class WingGeometry(abc.ABC):
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

    def surface_distributions(self, N=2000):
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
        dy = 1/N
        y = np.linspace(-self.b/2, self.b/2 - dy, N) + dy/2
        fx = self.fx(y)
        fz = self.fz(y)
        fc = self.fc(y)

        # FIXME: needs verification
        # FIXME: this is a crude rectangle rule integration
        Sx = ((y**2 + fz**2)*fc).sum()*dy
        Sy = ((3*fx**2 - fx*fc + (7/32)*fc**2 + 6*fz**2)*fc).sum()*dy
        Sz = ((3*fx**2 - fx*fc + (7/32)*fc**2 + 6*y**2)*fc).sum()*dy
        Sxy = 0
        Sxz = ((2*fx - fc/2)*fz*fc).sum()*dy
        Syz = 0

        S = np.array([
            [Sx, Sxy, Sxz],
            [Sxy, Sy, Syz],
            [Sxz, Syz, Sz]])

        return S

    @property
    def S_flat(self):
        """The area of the flattened wing"""
        # ref: PFD 46 (54)
        # FIXME: untested
        N = 1000
        dy = 1/N
        ys = np.linspace(-self.b/2, self.b/2 - dy, N) + dy/2
        return (self.fc(ys) * sqrt(self.dfzdy(ys)**2 + 1)).sum() * dy

    @property
    def b_flat(self):
        """The span of the flattened wing"""
        # ref: PFD 47 (54)
        # FIXME: untested
        N = 1000
        dy = 1/N
        ys = np.linspace(-self.b/2, self.b/2 - dy, N) + dy/2
        return sqrt(self.dfzdy(ys)**2 + 1).sum() * dy

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

    def delta(self, y):
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
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        c0 = (MAC / (2/3) / (2 + taper**2)) * (taper + tmp)
        return c0

    @staticmethod
    def AR_to_b(c0, AR, taper):
        """Compute the span of a tapered elliptical wing"""
        # ref: PFD 46 (54)
        # FIXME: rename tmp
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        b = (AR / 2)*c0*(taper + tmp)
        return b
