"""
Models for airfoil geometry and airfoil coefficients. Geometry models are for
graphical purposes, such as drawing the wing. Coefficient models are for
evaluating the section coefficients for lift, drag, and pitching moment.
"""

import abc

import numpy as np
from numpy import arctan

import pandas as pd
from scipy.integrate import simps
from scipy.interpolate import LinearNDInterpolator
from numpy.polynomial import Polynomial


class Airfoil:
    def __init__(self, coefficients, geometry=None):

        if not isinstance(coefficients, AirfoilCoefficients):
            raise ValueError("geometry is not an AirfoilCoefficients")

        # FIXME: reasonable default for inertia calculations?
        if geometry is None:
            geometry = NACA4(2415)

        if not isinstance(geometry, AirfoilGeometry):
            raise ValueError("geometry is not an AirfoilGeometry")

        self.coefficients = coefficients
        self.geometry = geometry


class AirfoilCoefficients(abc.ABC):
    """
    Provides the aerodynamic coefficients of a wing section.

    FIXME: needs a better description
    """

    @abc.abstractmethod
    def Cl(self, alpha, delta):
        """
        The lift coefficient of the airfoil

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [unitless distance]
            The deflection distance of the trailing edge due to braking,
            measured as a fraction of the chord length.

        Returns
        -------
        Cl : float
        """

    @abc.abstractmethod
    def Cd(self, alpha, delta):
        """
        The drag coefficient of the airfoil

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [unitless distance]
            The deflection distance of the trailing edge due to braking,
            measured as a fraction of the chord length.

        Returns
        -------
        Cd : float
        """

    @abc.abstractmethod
    def Cm(self, alpha, delta):
        """
        The pitching coefficient of the airfoil

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [unitless distance]
            The deflection distance of the trailing edge due to braking,
            measured as a fraction of the chord length.

        Returns
        -------
        Cm : float
        """

    def Cl_alpha(self, alpha, delta):
        """
        The derivative of the lift coefficient versus angle of attack

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack
        delta : float [unitless distance]
            The deflection distance of the trailing edge due to braking,
            measured as a fraction of the chord length.

        Returns
        -------
        Cl_alpha : float
        """


class LinearCoefficients(AirfoilCoefficients):
    """
    An airfoil model that assumes a strictly linear lift coefficient, constant
    form drag, and constant pitching moment.

    The effect of brakes is to shift the coefficient curves to the left; brake
    deflections do not change the shape of the curves. This is equivalent to
    an airfoil with a fixed flap hinge located at the leading edge.

    FIXME: the name is misleading: should be "FixedCoefficients" or similar
    FIXME: constrain the AoA, like `-i0 < alpha < alpha_max` ?
    """

    def __init__(self, a0, i0, D0, Cm):
        self.a0 = a0  # [1/rad]
        self.i0 = np.deg2rad(i0)
        self.D0 = D0
        self._Cm = Cm  # FIXME: seems clunky; change the naming?

    def Cl(self, alpha, delta):
        # FIXME: verify the usage of delta
        delta_angle = arctan(delta)  # tan(delta_angle) = delta/chord
        return self.a0 * (alpha + delta_angle - self.i0)

    def Cd(self, alpha, delta):
        return np.full_like(alpha, self.D0, dtype=self.D0.dtype)

    def Cm(self, alpha, delta):
        return np.full_like(alpha, self._Cm, dtype=self._Cm.dtype)

    def Cl_alpha(self, alpha, delta):
        return self.a0


class GridCoefficients(AirfoilCoefficients):
    """
    Uses the airfoil coefficients from a CSV file.

    The CSV must contain the following columns
     * alpha
     * flap
     * CL
     * CD
     * Cm

    This assumes a fixed-hinge design. A trailing edge deflection is simply a
    rotation of some trailing section of the chord, rotated by the angle `flap`
    about the point `xhinge`. That is: `delta = (1 - xhinge)*flap`, where
    `0 < xhinge < 1`.

    Note: this assumes normalized chord lengths; that is, `c = 1`.
    """

    def __init__(self, filename, xhinge, convert_degrees=True):
        # FIXME: docstring
        if (xhinge < 0) or (xhinge > 1):
            raise ValueError("xhinge should be a fraction of the chord length")

        # FIXME: add a dictionary for column name overrides?

        data = pd.read_csv(filename)
        self.data = data
        self.xhinge = xhinge  # hinge position as a percentage of the chord

        if convert_degrees:
            data['alpha'] = np.deg2rad(data.alpha)
            data['flap'] = np.deg2rad(data.flap)

        # Pre-transform local flap angles into into chord-global delta angles
        data['delta'] = data['flap'] * (1 - self.xhinge)

        self._Cl = LinearNDInterpolator(data[['alpha', 'delta']], data.CL)
        self._Cd = LinearNDInterpolator(data[['alpha', 'delta']], data.CD)
        self._Cm = LinearNDInterpolator(data[['alpha', 'delta']], data.Cm)

        # Construct another grid for the derivative of Cl vs alpha
        # FIXME: needs a design review
        points = []
        for flap, group in data.groupby('flap'):
            # FIXME: formalize the data sanitation strategy
            if group.shape[0] < 10:
                print("DEBUG> too few points for flap={}. Skipping.".format(
                    flap))
                continue
            poly = Polynomial.fit(group['alpha'], group['CL'], 7)
            Cl_alphas = poly.deriv()(group['alpha'])[:, None]
            points.append(np.hstack((group[['alpha', 'delta']], Cl_alphas)))
        points = np.vstack(points)  # Columns: [alpha, delta, Cl_alpha]
        self._Cl_alpha = LinearNDInterpolator(points[:, :2], points[:, 2])

    def Cl(self, alpha, delta):
        return self._Cl(alpha, delta)

    def Cd(self, alpha, delta):
        return self._Cd(alpha, delta)

    def Cm(self, alpha, delta):
        return self._Cm(alpha, delta)

    def Cl_alpha(self, alpha, delta):
        return self._Cl_alpha(alpha, delta)


# ---------------------------------------------------------------------------


class AirfoilGeometry(abc.ABC):
    """
    This class provides the parametric curves that define the upper and lower
    surfaces of an airfoil. It also provides the magnitudes, centroids, and
    inertia tensors of the upper curve, lower curve, and planar area. These are
    useful for calculating the surface areas and internal volume of a parafoil,
    and their individual moments of inertia. The curves are also useful for
    drawing a 3D wing.

    FIXME: finish the docstring
    FIXME: explicitly state that the units depend on the inputs?

    Attributes
    ----------
    area : float
        The area of the airfoil
    area_centroid : ndarray of float, shape (2,)
        The centroid of the area as [centroid_x, centroid_z]
    area_inertia : ndarray of float, shape (3,3)
        The inertia tensor of the area
    lower_length : float
        The total length of the lower surface curve
    lower_centroid : ndarray of float, shape (2,)
        The centroid of the lower surface curve as [centroid_x, centroid_z]
    lower_inertia : ndarray of float, shape (3,3)
        The inertia tensor of the lower surface curve
    upper_length : float
        The total length of the lower surface curve
    upper_centroid : ndarray of float, shape (2,)
        The centroid of the lower surface curve as [centroid_x, centroid_z]
    upper_inertia : ndarray of float, shape (3,3)
        The inertia tensor of the upper surface curve
    """

    def __init__(self):
        self._compute_inertia_tensors()  # Adds new instance members

    def _compute_inertia_tensors(self, N=200):
        """
        Calculate the inertia tensors for the the planar area and curves by
        treating them as (flat) 3D objects.

        Parameters
        ----------
        N : integer
            The number of chordwise sample points. Used to create the vertical
            strips for calculating the area, and for creating line segments of
            the parametric curves for the upper and lower surfaces.

        Notes
        -----
        These use the standard airfoil coordinate axes: the origin is fixed at
        the leading edge, the chord lies on the positive x-axis, and the z-axis
        points upward. Also, a y-axis that satisfies the right hand rule is
        added, for the purpose of creating a well-defined inertia tensor.
        """

        x = np.linspace(0, 1, N)  # FIXME: okay to assume a normalized airfoil?

        upper = self.upper_curve(x)
        lower = self.lower_curve(x)
        Ux, Uz = upper[:, 0], upper[:, 1]
        Lx, Lz = lower[:, 0], lower[:, 1]

        # -------------------------------------------------------------------
        # 1. Area calculations

        self.area = simps(Uz, Ux) - simps(Lz, Lx)
        xbar = (simps(Ux*Uz, Ux) - simps(Lx*Lz, Lx)) / self.area
        zbar = (simps(Uz**2/2, Ux) + simps(Lz**2/2, Lx)) / self.area
        self.area_centroid = np.array([xbar, zbar])

        # Area moments of inertia about the origin
        # FIXME: verify, including for airfoils where some `Lz > 0`
        Ix_o = 1/3 * (simps(Uz**3, Ux) - simps(Lz**3, Lx))
        Iz_o = simps(Ux**2 * Uz, Ux) - simps(Lx**2 * Lz, Lx)
        Ixz_o = 1/2 * (simps(Ux*(Uz**2), Ux) - simps(Lx*(Lz**2), Lx))  # FIXME?

        # Use the parallel axis theorem to find the inertias about the centroid
        Ix = Ix_o - self.area*zbar**2
        Iz = Iz_o - self.area*xbar**2
        Ixz = Ixz_o - self.area*xbar*zbar
        Iy = Ix + Iz  # Perpendicular axis theorem

        # Area inertia tensor, treating the plane as a 3D object
        self.area_inertia = np.array(
            [[Ix, 0, -Ixz],
             [0, Iy, 0],
             [-Ixz, 0, Iz]])

        # -------------------------------------------------------------------
        # 2. Surface line calculations

        # Line segment lengths and midpoints
        norm_U = np.linalg.norm(np.diff(upper, axis=0), axis=1)
        norm_L = np.linalg.norm(np.diff(lower, axis=0), axis=1)
        mid_U = (upper[:-1] + upper[1:])/2  # Midpoints of the upper segments
        mid_L = (lower[:-1] + lower[1:])/2  # Midpoints of the lower segments

        # Surface line lengths
        self.upper_length = norm_U.sum()  # Total upper line length
        self.lower_length = norm_L.sum()  # Total lower line length
        UL, LL = self.upper_length, self.lower_length  # Convenient shorthand

        # Surface line centroids
        self.upper_centroid = np.einsum('ij,i->j', mid_U, norm_U) / UL
        self.lower_centroid = np.einsum('ij,i->j', mid_L, norm_L) / LL

        # Surface line moments of inertia about their centroids
        # FIXME: not proper line integrals: treats segments as point masses
        cmUx, cmUz = self.upper_centroid
        mid_Ux, mid_Uz = mid_U[:, 0], mid_U[:, 1]
        Ix_U = np.sum(mid_Uz**2 * norm_U) - UL*cmUz**2
        Iz_U = np.sum(mid_Ux**2 * norm_U) - UL*cmUx**2
        Ixz_U = np.sum(mid_Ux * mid_Uz * norm_U) - UL*cmUx*cmUz
        Iy_U = Ix_U + Iz_U

        cmLx, cmLz = self.lower_centroid
        mid_Lx, mid_Lz = mid_L[:, 0], mid_L[:, 1]
        Ix_L = np.sum(mid_Lz**2 * norm_L) - LL*cmLz**2
        Iz_L = np.sum(mid_Lx**2 * norm_L) - LL*cmLx**2
        Ixz_L = np.sum(mid_Lx * mid_Lz * norm_L) - LL*cmLx*cmLz
        Iy_L = Ix_L + Iz_L

        # Line inertia tensors, treating the lines as 3D objects
        self.upper_inertia = np.array(
            [[Ix_U, 0, -Ixz_U],
             [0, Iy_U, 0],
             [-Ixz_U, 0, Iz_U]])

        self.lower_inertia = np.array(
            [[Ix_L, 0, -Ixz_L],
             [0, Iy_L, 0],
             [-Ixz_L, 0, Iz_L]])

    @property
    @abc.abstractmethod
    def tcr(self):
        """Maximum airfoil thickness-to-chord ratio"""
        # FIXME: does this belong in the API? When is this useful?

    @abc.abstractmethod
    def camber_curve(self, x):
        """Mean camber line coordinates

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= chord`

        Returns
        -------
        FIXME: describe the <x,y> array
        """

    @abc.abstractmethod
    def thickness(self, x):
        """Airfoil thickness, perpendicular to the camber line

        FIXME: there are two versions of this idea:
         1. Perpendicular to the camber line ("American convention")
         2. Vertical ("British convention")
         * Allow the class to specify the convention?

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= chord`

        Returns
        -------
        FIXME: describe
        """

    @abc.abstractmethod
    def upper_curve(self, x):
        """Upper surface coordinates

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= chord`

        Returns
        -------
        FIXME: describe the <x,y> array
        """

    @abc.abstractmethod
    def lower_curve(self, x):
        """Lower surface coordinates

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= chord`

        Returns
        -------
        FIXME: describe the <x,y> array
        """


class NACA4(AirfoilGeometry):
    def __init__(self, code, chord=1):
        """
        Generate an airfoil using a NACA4 parameterization

        Parameters
        ----------
        code : integer or string
            The 4-digit NACA code. If the code is an integer less than 1000,
            leading zeros are implicitly added; for example, 12 becomes 0012.
        chord : float
            The length of the chord
        """
        if isinstance(code, str):
            code = int(code)  # Let the ValueError exception through
        if not isinstance(code, int):
            raise ValueError("The NACA4 code must be an integer")
        if code < 0 or code > 9999:  # Leading zeros are implicit
            raise ValueError("Invalid 4-digit NACA code: '{}'".format(code))

        self.chord = chord
        self.code = code
        self.m = (code // 1000) / 100       # Maximum camber
        self.p = ((code // 100) % 10) / 10  # location of max camber
        self._tcr = (code % 100) / 100      # Thickness to chord ratio
        self.pc = self.p * self.chord

        super().__init__()  # Add the centroids, inertias, etc

    @property
    def tcr(self):
        return self._tcr

    def camber_curve(self, x):
        m = self.m
        c = self.chord
        p = self.p
        pc = self.pc

        x = np.asarray(x)

        if np.any(x < 0) or np.any(x > c):
            raise ValueError("x must be between 0 and the chord length")

        f = (x <= pc)  # Filter for the two cases, `x <= pc` and `x > pc`
        cl = np.empty_like(x)
        cl[f] = (m/p**2)*(2*p*(x[f]/c) - (x[f]/c)**2)
        cl[~f] = (m/(1-p)**2)*((1-2*p) + 2*p*(x[~f]/c) - (x[~f]/c)**2)
        return np.c_[x, cl]

    def thickness(self, x):
        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > self.chord):
            raise ValueError("x must be between 0 and the chord length")

        return 5*self.tcr*(.2969*np.sqrt(x) - .126*x - .3516*x**2 +
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

    def upper_curve(self, x):
        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > self.chord):
            raise ValueError("x must be between 0 and the chord length")

        theta = self._theta(x)
        t = self.thickness(x)
        yc = self.camber_curve(x)[:, 1]
        return np.c_[x - t*np.sin(theta), yc + t*np.cos(theta)]

    def lower_curve(self, x):
        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > self.chord):
            raise ValueError("x must be between 0 and the chord length")

        theta = self._theta(x)
        t = self.thickness(x)
        yc = self.camber_curve(x)[:, 1]
        return np.c_[x + t*np.sin(theta), yc - t*np.cos(theta)]
