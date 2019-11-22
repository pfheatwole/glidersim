"""
Models for airfoil geometries and airfoil coefficients.

Geometry models are for computing mass distributions (for moments of inertia),
points on reference lines (such as the quarter-chord position, frequently used
by lifting line methods), and graphical purposes.

Coefficient models are for evaluating the section coefficients for lift, drag,
and pitching moment.
"""

import abc

from IPython import embed

import numpy as np
from numpy import arctan, cos, cumsum, sin
from numpy.linalg import norm
from numpy.polynomial import Polynomial

import pandas as pd

from scipy.integrate import simps
from scipy.interpolate import CloughTocher2DInterpolator as Clough2D
from scipy.interpolate import PchipInterpolator
from scipy.optimize import newton


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
        Compute the lift coefficient of the airfoil.

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
        Compute the drag coefficient of the airfoil.

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
        Compute the pitching coefficient of the airfoil.

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
        Compute the derivative of the lift coefficient versus angle of attack.

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
     * delta
     * CL
     * CD
     * Cm
    """

    def __init__(self, filename, convert_degrees=True):
        # FIXME: docstring

        data = pd.read_csv(filename)
        self.data = data

        if convert_degrees:
            data["alpha"] = np.deg2rad(data.alpha)
            data["delta"] = np.deg2rad(data.delta)

        self._Cl = Clough2D(data[["alpha", "delta"]], data.CL)
        self._Cd = Clough2D(data[["alpha", "delta"]], data.CD)
        self._Cm = Clough2D(data[["alpha", "delta"]], data.Cm)

        # Construct another grid with smoothed derivatives of Cl vs alpha
        # FIXME: needs a design review
        alpha_min, delta_min = self._Cl.tri.min_bound
        alpha_max, delta_max = self._Cl.tri.max_bound
        points = []
        for delta in np.linspace(delta_min, delta_max, 25):
            alphas = np.linspace(alpha_min, alpha_max, 150)
            deltas = np.full_like(alphas, delta)
            CLs = self._Cl(alphas, deltas)
            notnan = ~np.isnan(CLs)  # Some curves are truncated at high alpha
            alphas, deltas, CLs = alphas[notnan], deltas[notnan], CLs[notnan]
            poly = Polynomial.fit(alphas, CLs, 7)
            Cl_alphas = poly.deriv()(alphas)
            deltas -= 1e-9   # FIXME: HACK! Keep delta=0 inside the convex hull
            points.append(np.array((alphas, deltas, Cl_alphas)).T)

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
    FIXME: outdated, these are no longer attributes of the abstract class

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

    def __init__(self, points, normalize=True):
        if normalize:
            points = self._derotate_and_normalize(points)
        self._curve, L = self._build_curve(points)
        self.upper_length = self._find_LE(self._curve, L)
        self.lower_length = L - self.upper_length

        self._compute_inertia_tensors()  # Adds new instance members

    def _build_curve(self, points):
        # The points must be sorted: upper TE first, then counter-clockwise
        d = np.r_[0, cumsum(norm(np.diff(points.T), axis=0))]
        L = d[-1]  # The total length of the curve
        return PchipInterpolator(d, points), L

    def _find_LE(self, curve, L):
        # FIXME: this has not been thoroughly tested
        deriv = curve.derivative()
        TE = (curve(0) + curve(L)) / 2

        def normed(v):
            return v / np.linalg.norm(v)

        def target(d):  # Optimization target: `tangent @ chord == 0`
            dydx = deriv(d)
            chord = curve(d) - TE  # The proposed chord line
            return np.dot(normed(dydx), normed(chord))
        d_LE = newton(target, L / 2)  # FIXME: use x0 = the longest candidate?
        return d_LE

    def _derotate_and_normalize(self, points):
        # FIXME: this isn't quite right! Try zooming in on the new origin.
        #        I think the problem is that `points` probably doesn't contain
        #        a point directly on the new origin.
        points = points.copy()
        curve, L = self._build_curve(points)
        LE = curve(self._find_LE(curve, L))
        TE = (curve(0) + curve(L)) / 2
        chord = LE - TE
        delta = np.pi - np.arctan2(chord[1], chord[0])
        points -= TE  # Temporarily shift the origin to the TE
        R = np.array([[cos(delta), -sin(delta)], [sin(delta), cos(delta)]])
        points = (R @ points.T).T  # Derotate
        points /= np.linalg.norm(chord)  # Normalize the lengths
        points += [1, 0]  # Restore the normalized trailing edge
        return points

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

        s = (1 - np.cos(np.linspace(0, np.pi, N))) / 2

        # FIXME: the upper and lower surfaces are probably not be defined by
        #        the LE! Don't split at `s = 0`, split at configurable points
        #        on the curve (which are likely determined by the air intakes).
        upper = self.surface_curve(s).T
        lower = self.surface_curve(-s).T
        Ux, Uy = upper[0], upper[1]
        Lx, Ly = lower[0], lower[1]

        # -------------------------------------------------------------------
        # 1. Area calculations

        self.area = simps(Uy, Ux) - simps(Ly, Lx)
        xbar = (simps(Ux * Uy, Ux) - simps(Lx * Ly, Lx)) / self.area
        ybar = (simps(Uy ** 2 / 2, Ux) + simps(Ly ** 2 / 2, Lx)) / self.area
        self.area_centroid = np.array([xbar, ybar])

        # Area moments of inertia about the origin
        # FIXME: verify, including for airfoils where some `Ly > 0`
        Ixx_o = 1 / 3 * (simps(Uy ** 3, Ux) - simps(Ly ** 3, Lx))
        Iyy_o = simps(Ux ** 2 * Uy, Ux) - simps(Lx ** 2 * Ly, Lx)
        Ixy_o = 1 / 2 * (simps(Ux * Uy ** 2, Ux) - simps(Lx * Ly ** 2, Lx))  # FIXME: verify?

        # Use the parallel axis theorem to find the inertias about the centroid
        Ixx = Ixx_o - self.area * ybar ** 2
        Iyy = Iyy_o - self.area * xbar ** 2
        Ixy = Ixy_o - self.area * xbar * ybar
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
        mid_U = (upper[:, :-1] + upper[:, 1:]) / 2  # Midpoints of `upper`
        mid_L = (lower[:, :-1] + lower[:, 1:]) / 2  # Midpoints of `lower`

        # Total surface line lengths
        UL, LL = norm_U.sum(), norm_L.sum()  # Convenient shorthand
        assert np.isclose(UL, self.upper_length)
        assert np.isclose(LL, self.lower_length)

        # Surface line centroids
        #
        self.upper_centroid = np.einsum("ij,j->i", mid_U, norm_U) / UL
        self.lower_centroid = np.einsum("ij,j->i", mid_L, norm_L) / LL

        # Surface line moments of inertia about their centroids
        # FIXME: not proper line integrals: treats segments as point masses
        cmUx, cmUy = self.upper_centroid
        mid_Ux, mid_Uy = mid_U[0], mid_U[1]
        Ixx_U = np.sum(mid_Uy ** 2 * norm_U) - UL * cmUy ** 2
        Iyy_U = np.sum(mid_Ux ** 2 * norm_U) - UL * cmUx ** 2
        Ixy_U = np.sum(mid_Ux * mid_Uy * norm_U) - UL * cmUx * cmUy
        Izz_U = Ixx_U + Iyy_U

        cmLx, cmLy = self.lower_centroid
        mid_Lx, mid_Ly = mid_L[0], mid_L[1]
        Ixx_L = np.sum(mid_Ly ** 2 * norm_L) - LL * cmLy ** 2
        Iyy_L = np.sum(mid_Lx ** 2 * norm_L) - LL * cmLx ** 2
        Ixy_L = np.sum(mid_Lx * mid_Ly * norm_L) - LL * cmLx * cmLy
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

    def _s2d(self, s):
        """Compute the reverse mapping: s->d.

        The two parametrizations for the the curve are:
         * s: the normalized distance, moving clockwise, starting at the lower
           trailing edge, from `-1 <= s <= 1`
         * d: the absolute distance, moving counterclockwise, starting at the
           upper trailing edge, from `0 <= d <= upper_length+lower_length`

        These two parametrizations are in opposite directions because it seems
        more natural to refer to the upper surface with a positive normalized
        distance (`s`), but the points on an airfoil are traditionally
        specified in a counterclockwise order (`d`).
        """
        s = np.asarray(s)
        if np.any(abs(s)) > 1:
            raise ValueError("`s` must be between -1..1")

        d = np.empty(s.shape)
        mask = (s >= 0)
        d[mask] = (1 - s[mask]) * self.upper_length
        d[~mask] = (-s[~mask] * self.lower_length) + self.upper_length

        return d

    def surface_curve(self, s):
        """Compute points on the surface curve.

        Parameters
        ----------
        s : float
            The curve parameter from -1..1, with -1 the lower surface trailing
            edge, 0 is the leading edge, and +1 is the upper surface trailing
            edge

        Returns
        -------
        FIXME: describe
        """
        d = self._s2d(s)
        return self._curve(d)

    def surface_curve_tangent(self, s):
        """Compute the tangent unit vector at points on the surface curve."""
        d = self._s2d(s)
        dxdy = -self._curve.derivative()(d).T  # w.r.t. `s`
        return (dxdy / np.linalg.norm(dxdy, axis=0)).T

    def surface_curve_normal(self, s):
        """Compute the normal unit vector at points on the surface curve."""
        d = self._s2d(s)
        dxdy = (self._curve.derivative()(d) * [1, -1]).T
        dxdy = (dxdy[::-1] / np.linalg.norm(dxdy, axis=0))  # w.r.t. `s`
        return dxdy.T

    def camber_curve(self, x):
        raise NotImplementedError

    def thickness(self, x):
        """Compute airfoil thickness perpendicular to a reference line.

        This measurement is perpendicular to either the camber line ('American'
        convention) or the chord ('British' convention). Refer to the specific
        AirfoilGeometry class documentation to determine which is being used.

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        FIXME: describe
        """
        raise NotImplementedError


class NACA4(AirfoilGeometry):
    def __init__(self, code, open_TE=True, convention="American"):
        """
        Generate an airfoil using a NACA4 parameterization.

        Parameters
        ----------
        code : integer or string
            The 4-digit NACA code. If the code is an integer less than 1000,
            leading zeros are implicitly added; for example, 12 becomes 0012.
        open_TE : bool, optional
            Generate airfoils with an open trailing edge. Default: True.
        convention : {'American', 'British'}, optional
            The convention to use for calculating the airfoil thickness. The
            default is 'American'.

            The American convention measures airfoil thickness perpendicular to
            the camber line. The British convention measures airfoil thickness
            perpendicular to the chord. Most texts use the American definition
            to define the NACA geometry, but beware that XFOIL uses the British
            convention.
        """
        if isinstance(code, str):
            code = int(code)  # Let the ValueError exception through
        if not isinstance(code, int):
            raise ValueError("The NACA4 code must be an integer")
        if code < 0 or code > 9999:  # Leading zeros are implicit
            raise ValueError("Invalid 4-digit NACA code: '{}'".format(code))

        self.code = code
        self.open_TE = open_TE
        self.convention = convention.lower()

        valid_conventions = {"american", "british"}
        if self.convention not in valid_conventions:
            raise ValueError("The convention must be 'American' or 'British'")

        # FIXME: these comments belong in the class docstring?
        self.m = (code // 1000) / 100  # Maximum camber
        self.p = ((code // 100) % 10) / 10  # Location of max camber
        self.tcr = (code % 100) / 100  # Thickness-to-chord ratio

        N = 5000
        x = (1 - np.cos(np.linspace(0, np.pi, N))) / 2
        xyu, xyl = self._yu(x), self._yl(x[1:])

        super().__init__(np.r_[xyu[::-1], xyl])

    def thickness(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        open_TE = [0.2969, 0.1260, 0.3516, 0.2843, 0.1015]
        closed_TE = [0.2969, 0.1260, 0.3516, 0.2843, 0.1036]
        a0, a1, a2, a3, a4 = open_TE if self.open_TE else closed_TE
        return (
            5
            * self.tcr
            * (a0 * np.sqrt(x) - a1 * x - a2 * x ** 2 + a3 * x ** 3 - a4 * x ** 4)
        )

    def _theta(self, x):
        """Compute the angle of the mean camber line.

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
        dyc[f] = (2 * m / p ** 2) * (p - x[f])
        dyc[~f] = (2 * m / (1 - p) ** 2) * (p - x[~f])

        return arctan(dyc)

    def _yc(self, x):
        """Compute positions on the mean camber line.

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        FIXME: describe the <x,y> array
        """
        # The camber curve
        m = self.m
        p = self.p
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        f = (x <= p)  # Filter for the two cases, `x <= p` and `x > p`
        cl = np.empty_like(x)
        cl[f] = (m / p ** 2) * (2 * p * (x[f]) - (x[f]) ** 2)
        cl[~f] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * (x[~f]) - (x[~f]) ** 2)
        return np.array([x, cl]).T

    def _yu(self, x):
        # The upper curve
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        t = self.thickness(x)
        yc = self._yc(x).T[1]
        if self.convention == "american":  # Standard NACA definition
            theta = self._theta(x)
            curve = np.array([x - t * np.sin(theta), yc + t * np.cos(theta)]).T
        elif self.convention == "british":  # XFOIL style
            curve = np.array([x, yc + t]).T
        else:
            raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve

    def _yl(self, x):
        # The lower curve
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        t = self.thickness(x)
        yc = self._yc(x).T[1]
        if self.convention == "american":  # Standard NACA definition
            theta = self._theta(x)
            curve = np.array([x + t * np.sin(theta), yc - t * np.cos(theta)]).T
        elif self.convention == "british":  # XFOIL style
            curve = np.array([x, yc - t]).T
        else:
            raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve


class NACA5(AirfoilGeometry):
    # FIXME: merge with `NACA4`? Lots of overlapping code
    def __init__(self, code, open_TE=True, convention="American"):
        """
        Generate an airfoil using a NACA5 parameterization.

        Parameters
        ----------
        code : integer or string
            The 5-digit NACA code. If the code is an integer less than 10000,
            leading zeros are implicitly added; for example, 12 becomes 00012.

            The NACA5 code can be expressed as LPSTT. Valid codes for this
            current implementation are restricted to ``L = 2`` and ``S = 0``.
        open_TE : bool, optional
            Generate airfoils with an open trailing edge. Default: True
        convention : {'American', 'British'}, optional
            The convention to use for calculating the airfoil thickness. The
            default is 'American'.

            The American convention measures airfoil thickness perpendicular to
            the camber line. The British convention measures airfoil thickness
            perpendicular to the chord. Most texts use the American definition
            to define the NACA geometry, but beware that XFOIL uses the British
            convention.
        """
        if isinstance(code, str):
            code = int(code)  # Let the ValueError exception through
        if not isinstance(code, int):
            raise ValueError("The NACA4 code must be an integer")
        if code < 0 or code > 99999:  # Leading zeros are implicit
            raise ValueError("Invalid 5-digit NACA code: '{}'".format(code))

        self.code = code
        self.open_TE = open_TE
        self.convention = convention.lower()

        valid_conventions = {"american", "british"}
        if self.convention not in valid_conventions:
            raise ValueError("The convention must be 'American' or 'British'")

        L = (code // 10000)  # Theoretical optimum lift coefficient
        P = (code // 1000) % 10  # Point of maximum camber on the chord
        S = (code // 100) % 10  # 0 or 1 for simple or reflexed camber line
        TT = code % 100

        if L != 2:
            raise ValueError(f"Invalid optimum lift factor: {L}")
        if S != 0:
            # FIXME: implement?
            raise ValueError("Reflex airfoils are not currently supported")

        # Choose `m` and `k1` based on the first three digits (FIXME: clarify)
        coefficient_options = {
            # code : (M, C)
            210: (0.0580, 361.4),
            220: (0.1260, 51.64),
            230: (0.2025, 15.957),
            240: (0.2900, 6.643),
            250: (0.3910, 3.230),
        }

        try:
            self.m, self.k1 = coefficient_options[code // 100]
        except KeyError:
            raise RuntimeError("Unhandled NACA5 code")

        self.p = 0.05 * P
        self.tcr = TT / 100

        N = 200
        x = (1 - np.cos(np.linspace(0, np.pi, N))) / 2
        xyu, xyl = self._yu(x), self._yl(x[1:])

        super().__init__(np.r_[xyu[::-1], xyl])

    def thickness(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        open_TE = [0.2969, 0.1260, 0.3516, 0.2843, 0.1015]
        closed_TE = [0.2969, 0.1260, 0.3516, 0.2843, 0.1036]
        a0, a1, a2, a3, a4 = open_TE if self.open_TE else closed_TE
        return (
            5
            * self.tcr
            * (a0 * np.sqrt(x) - a1 * x - a2 * x ** 2 + a3 * x ** 3 - a4 * x ** 4)
        )

    def _theta(self, x):
        """Compute the angle of the mean camber line.

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
        dyc[f] = (2 * m / p ** 2) * (p - x[f])
        dyc[~f] = (2 * m / (1 - p) ** 2) * (p - x[~f])

        return arctan(dyc)

    def _yc(self, x):
        """Compute positions on the mean camber line.

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        FIXME: describe the <x,y> array
        """
        # The camber curve
        m, k1 = self.m, self.k1
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        f = (x < m)  # Filter for the two cases, `x < m` and `x >= m`
        cl = np.empty_like(x)
        cl[f] = (k1 / 6) * (x[f] ** 3 - 3 * m * (x[f] ** 2) + (m ** 2) * (3 - m) * x[f])
        cl[~f] = (k1 * m ** 3 / 6) * (1 - x[~f])
        return np.array([x, cl]).T

    def _yu(self, x):
        # The upper curve
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        t = self.thickness(x)
        yc = self._yc(x).T[1]
        if self.convention == "american":  # Standard NACA definition
            theta = self._theta(x)
            curve = np.array([x - t * np.sin(theta), yc + t * np.cos(theta)]).T
        elif self.convention == "british":  # XFOIL style
            curve = np.array([x, yc + t]).T
        else:
            raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve

    def _yl(self, x):
        # The lower curve
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        t = self.thickness(x)
        yc = self._yc(x).T[1]
        if self.convention == "american":  # Standard NACA definition
            theta = self._theta(x)
            curve = np.array([x + t * np.sin(theta), yc - t * np.cos(theta)]).T
        elif self.convention == "british":  # XFOIL style
            curve = np.array([x, yc - t]).T
        else:
            raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve
