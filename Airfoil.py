"""
Models for airfoil geometry and airfoil coefficients. Geometry models are for
graphical purposes, such as drawing the wing. Coefficient models are for
evaluating the section coefficients for lift, drag, and pitching moment.
"""

from IPython import embed
import abc

import numpy as np
from numpy import arctan

import pandas as pd
from scipy.integrate import simps
from scipy.interpolate import CloughTocher2DInterpolator as Clough2D
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

        self._Cl = Clough2D(data[['alpha', 'delta']], data.CL)
        self._Cd = Clough2D(data[['alpha', 'delta']], data.CD)
        self._Cm = Clough2D(data[['alpha', 'delta']], data.Cm)

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
        self._Cl_alpha = Clough2D(points[:, :2], points[:, 2])

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
        A z-axis that satisfies the right hand rule is added for the purpose of
        creating a well-defined inertia tensor. It is important to note that
        this xyz coordinate system is different from the `frd` coordinate axes
        used by a Parafoil. It is used to maintain consistency with traditional
        airfoil definitions in literature.

        To convert from this airfoil coordinate system ("acs") to frd:

        >>> C = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        >>> centroid_frd = C @ [*centroid_acs, 0]  # Augment with z_acs=0
        >>> inertia_frd = C @ inertia_acs @ C
        """

        x = np.linspace(0, 1, N)

        upper = self.upper_curve(x).T
        lower = self.lower_curve(x).T
        Ux, Uy = upper[0], upper[1]
        Lx, Ly = lower[0], lower[1]

        # -------------------------------------------------------------------
        # 1. Area calculations

        self.area = simps(Uy, Ux) - simps(Ly, Lx)
        xbar = (simps(Ux*Uy, Ux) - simps(Lx*Ly, Lx)) / self.area
        ybar = (simps(Uy**2 / 2, Ux) + simps(Ly**2 / 2, Lx)) / self.area
        self.area_centroid = np.array([xbar, ybar])

        # Area moments of inertia about the origin
        # FIXME: verify, including for airfoils where some `Ly > 0`
        Ixx_o = 1/3 * (simps(Uy**3, Ux) - simps(Ly**3, Lx))
        Iyy_o = simps(Ux**2 * Uy, Ux) - simps(Lx**2 * Ly, Lx)
        Ixy_o = 1/2 * (simps(Ux * Uy**2, Ux) - simps(Lx * Ly**2, Lx))  # FIXME?

        # Use the parallel axis theorem to find the inertias about the centroid
        Ixx = Ixx_o - self.area * ybar**2
        Iyy = Iyy_o - self.area * xbar**2
        Ixy = Ixy_o - self.area*xbar*ybar
        Izz = Ixx + Iyy  # Perpendicular axis theorem

        # Area inertia tensor, treating the plane as a 3D object
        self.area_inertia = np.array(
            [[ Ixx, -Ixy,   0],
             [-Ixy,  Iyy,   0],
             [   0,    0, Izz]])

        # -------------------------------------------------------------------
        # 2. Surface line calculations

        # Line segment lengths and midpoints
        norm_U = np.linalg.norm(np.diff(upper), axis=0)
        norm_L = np.linalg.norm(np.diff(lower), axis=0)
        mid_U = (upper[:, :-1] + upper[:, 1:])/2  # Midpoints of `upper`
        mid_L = (lower[:, :-1] + lower[:, 1:])/2  # Midpoints of `lower`

        # Total surface line lengths
        self.upper_length = norm_U.sum()
        self.lower_length = norm_L.sum()
        UL, LL = self.upper_length, self.lower_length  # Convenient shorthand

        # Surface line centroids
        #
        self.upper_centroid = np.einsum('ij,j->i', mid_U, norm_U) / UL
        self.lower_centroid = np.einsum('ij,j->i', mid_L, norm_L) / LL

        # Surface line moments of inertia about their centroids
        # FIXME: not proper line integrals: treats segments as point masses
        cmUx, cmUy = self.upper_centroid
        mid_Ux, mid_Uy = mid_U[0], mid_U[1]
        Ixx_U = np.sum(mid_Uy**2 * norm_U) - UL*cmUy**2
        Iyy_U = np.sum(mid_Ux**2 * norm_U) - UL*cmUx**2
        Ixy_U = np.sum(mid_Ux*mid_Uy*norm_U) - UL*cmUx*cmUy
        Izz_U = Ixx_U + Iyy_U

        cmLx, cmLy = self.lower_centroid
        mid_Lx, mid_Ly = mid_L[0], mid_L[1]
        Ixx_L = np.sum(mid_Ly**2 * norm_L) - LL*cmLy**2
        Iyy_L = np.sum(mid_Lx**2 * norm_L) - LL*cmLx**2
        Ixy_L = np.sum(mid_Lx*mid_Ly*norm_L) - LL*cmLx*cmLy
        Izz_L = Ixx_L + Iyy_L

        # Line inertia tensors, treating the lines as 3D objects
        self.upper_inertia = np.array(
            [[ Ixx_U, -Ixy_U,     0],
             [-Ixy_U,  Iyy_U,     0],
             [     0,      0, Izz_U]])

        self.lower_inertia = np.array(
            [[ Ixx_L, -Ixy_L,     0],
             [-Ixy_L,  Iyy_L,     0],
             [     0,      0, Izz_L]])

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
            Position on the chord line, where `0 <= x <= 1`

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
            Position on the chord line, where `0 <= x <= 1`

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
            Position on the chord line, where `0 <= x <= 1`

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
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        FIXME: describe the <x,y> array
        """


class NACA4(AirfoilGeometry):
    def __init__(self, code):
        """
        Generate an airfoil using a NACA4 parameterization

        Parameters
        ----------
        code : integer or string
            The 4-digit NACA code. If the code is an integer less than 1000,
            leading zeros are implicitly added; for example, 12 becomes 0012.
        """
        if isinstance(code, str):
            code = int(code)  # Let the ValueError exception through
        if not isinstance(code, int):
            raise ValueError("The NACA4 code must be an integer")
        if code < 0 or code > 9999:  # Leading zeros are implicit
            raise ValueError("Invalid 4-digit NACA code: '{}'".format(code))

        self.code = code
        self.m = (code // 1000) / 100       # Maximum camber
        self.p = ((code // 100) % 10) / 10  # location of max camber
        self._tcr = (code % 100) / 100      # Thickness to chord ratio

        super().__init__()  # Add the centroids, inertias, etc

    @property
    def tcr(self):
        return self._tcr

    def camber_curve(self, x):
        m = self.m
        p = self.p
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        f = (x <= p)  # Filter for the two cases, `x <= p` and `x > p`
        cl = np.empty_like(x)
        cl[f] = (m/p**2)*(2*p*(x[f]) - (x[f])**2)
        cl[~f] = (m/(1-p)**2)*((1-2*p) + 2*p*(x[~f]) - (x[~f])**2)
        return np.array([x, cl]).T

    def thickness(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        return 5*self.tcr*(.2969*np.sqrt(x) - .126*x - .3516*x**2 +
                           .2843*x**3 - .1015*x**4)

    def _theta(self, x):
        """Angle of the mean camber line

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= 1`
        """
        m = self.m
        p = self.p

        x = np.asarray(x, dtype=float)
        assert np.all(x >= 0) and np.all(x <= 1)

        f = x <= p  # Filter for the two cases, `x <= p` and `x > p`
        dyc = np.empty_like(x)
        dyc[f] = (2*m/p**2)*(p - x[f])
        dyc[~f] = (2*m/(1-p)**2)*(p - x[~f])

        return arctan(dyc)

    def upper_curve(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        theta = self._theta(x)
        t = self.thickness(x)
        yc = self.camber_curve(x)[:, 1]
        return np.array([x - t*np.sin(theta), yc + t*np.cos(theta)]).T

    def lower_curve(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        theta = self._theta(x)
        t = self.thickness(x)
        yc = self.camber_curve(x)[:, 1]
        return np.array([x + t*np.sin(theta), yc - t*np.cos(theta)]).T
