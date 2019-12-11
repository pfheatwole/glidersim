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
from numpy.polynomial import Polynomial

import pandas as pd

import scipy.optimize
from scipy.integrate import simps
from scipy.interpolate import CloughTocher2DInterpolator as Clough2D
from scipy.interpolate import PchipInterpolator


class Airfoil:
    """
    Dumb wrapper class to bundle AirfoilCoefficients with AirfoilGeometry.

    This class probably shouldn't exist, but was added during the design
    exploration phase.
    """

    def __init__(self, coefficients, geometry=None):
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

    # FIXME: make this an abstractmethod? Must all subclasses implement it?
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
    Classes that describe the shapes and mass properties of an airfoil.

    The most general description of an airfoil is a set of points that define
    the upper and lower surfaces. This class divides that set of points into
    top and bottom regions, separated by the leading edge, and provides access
    to those curves as a parametric function. It also provides the unitless
    magnitudes, centroids, and inertia matrices of the upper curve, lower
    curve, and planar area, which can be scaled by the physical units of the
    target application.

    The curves are also useful for drawing a 3D wing. The mass properties are
    useful for calculating the upper and lower surface areas, internal volume,
    and inertia matrix of a 3D wing.

    Unlike standard airfoil definitions, this class converts the set of
    counter-clockwise points to a parametric curve parametrized by a clockwise
    normalized position `-1 <= s <= 1`, where `s = 0` is the leading edge,
    `s = 1` is the tip of the upper surface, and `s = -1` is the tip of the
    lower surface. Midpoints of the upper and lower surface curves are given by
    `s = 0.5` and `s = -0.5`.

    Parameters
    ----------
    points : array of float, shape (N,2)
        A sequence of (x, y) points on the airfoil. The points must follow the
        XFOIL convention: the x-coordinate of the leading edge is less than the
        x-coordinate of the trailing edge, and the points are given in
        counter-clockwise order, starting with the tip of the upper surface.
    normalize : bool, default: True
        Ensures the leading edge is at (0, 0) and the trailing edge is at
        (1, 0). This ensures the chord is aligned with the x-axis, and is
        unit length.
    s_upper, s_lower : float, where `-1 < s_lower <= s_upper < 1`
        The normalized starting positions of the upper and lower surfaces.
        These are used for determining the inertial properties of the upper and
        lower surfaces by allowing the upper surface to wrap over the leading
        edge (`s_upper < 0`), and for the presence of an air intake. They do
        not change the underlying curve specified by the input points; users of
        this class are responsible for checking these positions when drawing
        the wing.

    """

    def __init__(self, points, *, normalize=True, s_upper=0, s_lower=0):
        def _target(d, curve, derivative, TE):
            # Optimization target for finding the leading edge. The position
            # `0 < d < L` is a leading edge proposal, where L is the total
            # length of the curve. The leading edge is the point where
            # the chord is perpendicular to the curve at the leading edge.
            dydx = derivative(d)  # Tangent line at the proposal
            chord = curve(d) - TE  # Chord line to the proposal
            dydx /= np.linalg.norm(dydx)
            chord /= np.linalg.norm(chord)
            return dydx @ chord

        self.s_upper = s_upper
        self.s_lower = s_lower
        self._raw_points = points.copy()  # FIXME: for debugging
        points = points[::-1]  # Clockwise order is more convenient for `s`

        # Find the chord, align it to the x-axis, and scale it to unit length
        if normalize:
            d = np.r_[0, cumsum(np.linalg.norm(np.diff(points.T), axis=0))]
            curve = PchipInterpolator(d, points)
            TE = (curve(0) + curve(d[-1])) / 2
            result = scipy.optimize.root_scalar(
                _target,
                args=(curve, curve.derivative(), TE),
                bracket=(d[-1] * 0.4, d[-1] * 0.6),
            )
            d_LE = result.root
            chord = curve(d_LE) - TE
            delta = np.arctan2(chord[1], -chord[0])  # rotated angle
            points -= TE  # Temporarily shift the origin to the TE
            R = np.array([[cos(delta), -sin(delta)], [sin(delta), cos(delta)]])
            points = (R @ points.T).T  # Derotate
            points /= np.linalg.norm(chord)  # Normalize the lengths
            points += [1, 0]  # Restore the normalized trailing edge

        self.points = points

        # Parametrize the curve with the normalized distance `s`. Assumes the
        # airfoil chord lies on the x-axis, in which case the leading edge is
        # simply where the curve crosses the x-axis.
        d = np.r_[0, cumsum(np.linalg.norm(np.diff(points.T), axis=0))]
        curve = PchipInterpolator(d, points)
        TE = (curve(0) + curve(d[-1])) / 2
        result = scipy.optimize.root_scalar(
            _target,
            args=(curve, curve.derivative(), TE),
            bracket=(d[-1] * 0.4, d[-1] * 0.6),
        )
        d_LE = result.root
        idx_u = np.arange(len(d))[d >= d_LE]
        idx_l = np.arange(len(d))[d < d_LE]
        du = d[idx_u]
        dl = d[idx_l]
        su = (du - du[0]) / (du[-1] - du[0])  # d >= d_LE -> `0 <= s <= 1`
        sl = dl / du[0] - 1  # d < d_LE -> `0 < s <= -1`
        self._curve = PchipInterpolator(np.r_[sl, su], points)

    def mass_properties(self, N=200):
        """
        Calculate the inertia matrices for the the planar area and curves.

        This procedure treats the 2D geometry as infinitely flat 3D objects,
        with the new `z` axis added according to the right-hand rule. See
        "Notes" for more details.

        Parameters
        ----------
        N : integer
            The number of chordwise sample points. Used to create the vertical
            strips for calculating the area, and for creating line segments of
            the parametric curves for the upper and lower surfaces.

        Returns
        -------
        dictionary
            upper_length : float
                The total length of the upper surface curve
            upper_centroid : array of float, shape (2,)
                The centroid of the upper surface curve as (x, y) in ACS
            upper_inertia : array of float, shape (3,3)
                The inertia matrix of the upper surface curve
            area : float
                The area of the airfoil
            area_centroid : array of float, shape (2,)
                The centroid of the area as (x, y) in ACS
            area_inertia : array of float, shape (3,3)
                The inertia matrix of the area
            lower_length : float
                The total length of the lower surface curve
            lower_centroid : array of float, shape (2,)
                The centroid of the lower surface curve as (x, y) in ACS
            lower_inertia : array of float, shape (3,3)
                The inertia matrix of the lower surface curve

            These are unitless quantities. The inertia matrices for each
            component are for rotations about that components' centroid.

        Notes
        -----
        In traditional airfoil definitions, the positive x-axis lies along the
        chord, directed from the leading edge to the trailing edge, and the
        positive y-axis points towards the upper surface.

        Here, a z-axis that satisfies the right hand rule is added for the
        purpose of creating a well-defined inertia matrix. Let this set of axes
        be called the "airfoil coordinate system" (ACS).

        Translating these ACS coordinates into the front-right-down (FRD)
        coordinate system requires reordering and reversing the direction of
        vector components. To convert ACS -> FRD: [x, y, z] -> [-x, -z, -y]

        In terms of code, to convert from ACS to FRD coordinates:

        >>> C = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        >>> centroid_frd = C @ [*centroid_acs, 0]  # Augment with z_acs=0
        >>> inertia_frd = C @ inertia_acs @ C
        """
        # -------------------------------------------------------------------
        # 1. Area calculations

        s = (1 - np.cos(np.linspace(0, np.pi, N))) / 2  # `0 <= s <= 1`
        top = self.surface_curve(s).T  # Top half (above s = 0)
        bottom = self.surface_curve(-s).T  # Bottom half (below s = 0)
        Tx, Ty = top[0], top[1]
        Bx, By = bottom[0], bottom[1]

        area = simps(Ty, Tx) - simps(By, Bx)
        xbar = (simps(Tx * Ty, Tx) - simps(Bx * By, Bx)) / area
        ybar = (simps(Ty ** 2 / 2, Tx) + simps(By ** 2 / 2, Bx)) / area
        area_centroid = np.array([xbar, ybar])

        # Area moments of inertia about the origin
        # FIXME: verify, especially `Ixy_o`. Check airfoils where some `By > 0`
        Ixx_o = 1 / 3 * (simps(Ty ** 3, Tx) - simps(By ** 3, Bx))
        Iyy_o = simps(Tx ** 2 * Ty, Tx) - simps(Bx ** 2 * By, Bx)
        Ixy_o = 1 / 2 * (simps(Tx * Ty ** 2, Tx) - simps(Bx * By ** 2, Bx))

        # Use the parallel axis theorem to find the inertias about the centroid
        Ixx = Ixx_o - area * ybar ** 2
        Iyy = Iyy_o - area * xbar ** 2
        Ixy = Ixy_o - area * xbar * ybar
        Izz = Ixx + Iyy  # Perpendicular axis theorem

        # Inertia matrix for the area about the origin
        area_inertia = np.array(
            [[ Ixx, -Ixy,   0],
             [-Ixy,  Iyy,   0],
             [   0,    0, Izz]])

        # -------------------------------------------------------------------
        # 2. Surface line calculations

        su = np.linspace(self.s_upper, 1, N)
        sl = np.linspace(self.s_lower, -1, N)
        upper = self.surface_curve(su).T
        lower = self.surface_curve(sl).T

        # Line segment lengths and midpoints
        norm_U = np.linalg.norm(np.diff(upper), axis=0)  # Segment lengths
        norm_L = np.linalg.norm(np.diff(lower), axis=0)
        mid_U = (upper[:, :-1] + upper[:, 1:]) / 2  # Segment midpoints
        mid_L = (lower[:, :-1] + lower[:, 1:]) / 2

        # Total line lengths and centroids
        upper_length = norm_U.sum()
        lower_length = norm_L.sum()
        upper_centroid = np.einsum("ij,j->i", mid_U, norm_U) / upper_length
        lower_centroid = np.einsum("ij,j->i", mid_L, norm_L) / lower_length

        # Surface line moments of inertia about their centroids
        # FIXME: not proper line integrals: treats segments as point masses
        cmUx, cmUy = upper_centroid
        mid_Ux, mid_Uy = mid_U[0], mid_U[1]
        Ixx_U = np.sum(mid_Uy ** 2 * norm_U) - upper_length * cmUy ** 2
        Iyy_U = np.sum(mid_Ux ** 2 * norm_U) - upper_length * cmUx ** 2
        Ixy_U = np.sum(mid_Ux * mid_Uy * norm_U) - upper_length * cmUx * cmUy
        Izz_U = Ixx_U + Iyy_U

        cmLx, cmLy = lower_centroid
        mid_Lx, mid_Ly = mid_L[0], mid_L[1]
        Ixx_L = np.sum(mid_Ly ** 2 * norm_L) - lower_length * cmLy ** 2
        Iyy_L = np.sum(mid_Lx ** 2 * norm_L) - lower_length * cmLx ** 2
        Ixy_L = np.sum(mid_Lx * mid_Ly * norm_L) - lower_length * cmLx * cmLy
        Izz_L = Ixx_L + Iyy_L

        # Inertia matrices for the lines about the origin
        upper_inertia = np.array(
            [[ Ixx_U, -Ixy_U,     0],
             [-Ixy_U,  Iyy_U,     0],
             [     0,      0, Izz_U]])

        lower_inertia = np.array(
            [[ Ixx_L, -Ixy_L,     0],
             [-Ixy_L,  Iyy_L,     0],
             [     0,      0, Izz_L]])

        properties = {
            'upper_length': upper_length,
            'upper_centroid': upper_centroid,
            'upper_inertia': upper_inertia,
            'area': area,
            'area_centroid': area_centroid,
            'area_inertia': area_inertia,
            'lower_length': lower_length,
            'lower_centroid': lower_centroid,
            'lower_inertia': lower_inertia,
        }

        return properties

    def surface_curve(self, s):
        """
        Compute points on the surface curve.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            The curve parameter from -1..1, with -1 the lower surface trailing
            edge, 0 is the leading edge, and +1 is the upper surface trailing
            edge

        Returns
        -------
        points : array of float, shape (N, 2)
            The (x, y) coordinates of points on the airfoil at `s`.
        """
        return self._curve(s)

    def surface_curve_tangent(self, s):
        """
        Compute the tangent unit vector at points on the surface curve.

        Parameters
        ----------
        s : array_like of float
            The surface curve parameter.

        Returns
        -------
        dxdy : array, shape (N, 2)
            The unit tangent lines at the specified points, oriented with
            increasing `s`, so the tangents trace from the lower surface to
            the upper surface.
        """
        dxdy = self._curve.derivative()(s).T
        dxdy /= np.linalg.norm(dxdy, axis=0)
        return dxdy.T

    def surface_curve_normal(self, s):
        """
        Compute the normal unit vector at points on the surface curve.

        Parameters
        ----------
        s : array_like of float
            The surface curve parameter.

        Returns
        -------
        dxdy : array, shape (N, 2)
            The unit normal vectors at the specified points, oriented with
            increasing `s`, so the normals point "out" of the airfoil.
        """
        dxdy = (self._curve.derivative()(s) * [1, -1]).T
        dxdy = (dxdy[::-1] / np.linalg.norm(dxdy, axis=0))
        return dxdy.T

    def camber_curve(self, x):
        raise NotImplementedError

    def thickness(self, x):
        """
        Compute airfoil thickness perpendicular to a reference line.

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


class NACA(AirfoilGeometry):
    def __init__(self, code, *, open_TE=True, convention="British", **kwargs):
        """
        Generate an airfoil using a NACA4 or NACA5 parameterization.

        Parameters
        ----------
        code : integer or string
            A 4- or 5-digit NACA code. If the code is an integer less than
            1000, leading zeros are implicitly added; for example, 12 becomes
            the NACA4 code 0012.

            Only a subset of NACA5 codes are supported. A NACA5 code can be
            expressed as LPSTT; valid codes for this implementation are
            restricted to ``L = 2``, ``1 <= P <= 5``, and ``S = 0``.
        open_TE : bool, optional
            Generate airfoils with an open trailing edge. Default: True.
        convention : {"American", "British"}, optional
            The convention to use for calculating the airfoil thickness. The
            default is 'British'.

            The American convention measures airfoil thickness perpendicular to
            the camber line. The British convention measures airfoil thickness
            perpendicular to the chord. Many texts use the American definition
            to define the NACA geometry, but the popular tool "XFOIL" uses the
            British convention.

            Beware that because the NACA equations use the thickness to define
            the upper and lower surface curves, this option changes the shape
            of the resulting airfoil.

        Any additional keyword parameters will be forwarded to the parent class
        initializer, `AirfoilGeometry.__init__`.
        """
        if not isinstance(code, int):
            try:
                code = int(code)
            except ValueError:
                raise ValueError(f"Invalid NACA code '{code}': must be an integer")

        if code < 0:
            raise ValueError(f"Invalid NACA code '{code}': must be positive")
        elif code > 99999:
            raise ValueError(f"Unsupported NACA code '{code}': more than 5 digits")

        convention = convention.lower()
        valid_conventions = {"american", "british"}
        if convention not in valid_conventions:
            raise ValueError("The convention must be 'American' or 'British'")

        self.code = code
        self.open_TE = open_TE
        self.convention = convention

        if code <= 9999:  # NACA4 code
            self.series = 4
            self.m = (code // 1000) / 100  # Maximum camber
            self.p = ((code // 100) % 10) / 10  # Location of max camber
            self.tcr = (code % 100) / 100  # Thickness-to-chord ratio

        elif code <= 99999:  # NACA5 code
            self.series = 5
            L = (code // 10000)  # Theoretical optimum lift coefficient
            P = (code // 1000) % 10  # Point of maximum camber on the chord
            S = (code // 100) % 10  # 0 or 1 for simple or reflexed camber line
            TT = code % 100

            if L != 2:
                raise ValueError(f"Invalid optimum lift factor: {L}")
            if P < 1 or P > 5:
                raise ValueError(f"Unsupported maximum camber factor: {P}")
            if S != 0:
                raise ValueError("Reflex airfoils are not currently supported")

            # Choose `m` and `k1` based on the first three digits
            coefficient_options = {
                # code : (M, C)
                210: (0.0580, 361.40),
                220: (0.1260, 51.640),
                230: (0.2025, 15.957),
                240: (0.2900, 6.643),
                250: (0.3910, 3.230),
            }

            self.m, self.k1 = coefficient_options[code // 100]
            self.p = 0.05 * P
            self.tcr = TT / 100

        N = 200
        x = (1 - np.cos(np.linspace(0, np.pi, N))) / 2
        xyu = self._xyu(x)[::-1]  # Move counter-clockwise
        xyl = self._xyl(x[1:])  # Skip `x = 0`

        super().__init__(np.r_[xyu, xyl], **kwargs)

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
        """
        Compute the angle of the mean camber line.

        Parameters
        ----------
        x : float
            Position on the chord line, where `0 <= x <= 1`
        """
        m = self.m
        p = self.p

        x = np.asarray(x, dtype=float)
        assert np.all(x >= 0) and np.all(x <= 1)

        f = x < p  # Filter for the two cases, `x < p` and `x >= p`
        dyc = np.empty_like(x)

        # The tests are necessary for when `m > 0` and `p = 0`
        if np.any(f):
            dyc[f] = (2 * m / p ** 2) * (p - x[f])
        if np.any(~f):
            dyc[~f] = (2 * m / (1 - p) ** 2) * (p - x[~f])

        return arctan(dyc)

    def _yc(self, x):
        """
        Compute the y-coordinate of points on the mean camber line.

        Parameters
        ----------
        x : array_like of float, shape (N,)
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        y : array_like of float, shape (N,)
            The y-coordinates of the mean camber line. These points lie
            directly above the chord line, regardless of the convention.
            (The convention only changes the definition of the surface curves.)
        """
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        if self.series == 4:
            m, p = self.m, self.p
            f = x < p  # Filter for the two cases, `x < p` and `x >= p`
            y = np.empty_like(x)

            # The tests are necessary for when `m > 0` and `p = 0`
            if np.any(f):
                y[f] = (m / p ** 2) * (2 * p * (x[f]) - (x[f]) ** 2)
            if np.any(~f):
                y[~f] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * (x[~f]) - (x[~f]) ** 2)

        elif self.series == 5:
            m, k1 = self.m, self.k1
            f = x < m  # Filter for the two cases, `x < m` and `x >= m`
            y = np.empty_like(x)
            y[f] = (k1 / 6) * (x[f] ** 3 - 3 * m * (x[f] ** 2) + (m ** 2) * (3 - m) * x[f])
            y[~f] = (k1 * m ** 3 / 6) * (1 - x[~f])

        else:
            raise RuntimeError(f"Invalid NACA series '{self.series}'")

        return y

    def _xyu(self, x):
        """
        Compute the x- and y-coordinates of points on the upper surface.

        Returns both `x` and `y` because the "American" convention computes the
        surface curve coordinates orthogonal to the camber curve instead of the
        chord, so the `x` coordinate for the surface curve will not be the same
        as the `x` coordinate for the chord (unless the airfoil is symmetric,
        in which case the camber curve lines directly on the chord). For the
        British convention, the input and output `x` will always be the same.

        Parameters
        ----------
        x : array_like of float, shape (N,)
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        xy : array_like of float, shape (N, 2)
            The x- and y-coordinatess of the points on the upper surface.
        """
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        t = self.thickness(x)

        if self.m == 0:  # Symmetric airfoil
            curve = np.array([x, t]).T
        else:  # Cambered airfoil
            yc = self._yc(x)
            if self.convention == "american":  # Standard NACA definition
                theta = self._theta(x)
                curve = np.array([x - t * np.sin(theta), yc + t * np.cos(theta)]).T
            elif self.convention == "british":  # XFOIL style
                curve = np.array([x, yc + t]).T
            else:
                raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve

    def _xyl(self, x):
        """
        Compute the x- and y-coordinates of points on the lower surface.

        Returns both `x` and `y` because the "American" convention computes the
        surface curve coordinates orthogonal to the camber curve instead of the
        chord, so the `x` coordinate for the surface curve will not be the same
        as the `x` coordinate for the chord (unless the airfoil is symmetric,
        in which case the camber curve lines directly on the chord). For the
        British convention, the input and output `x` will always be the same.

        Parameters
        ----------
        x : array_like of float, shape (N,)
            Position on the chord line, where `0 <= x <= 1`

        Returns
        -------
        xy : array_like of float, shape (N, 2)
            The x- and y-coordinatess of the points on the upper surface.
        """
        x = np.asarray(x, dtype=float)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        t = self.thickness(x)
        if self.m == 0:  # Symmetric airfoil
            curve = np.array([x, -t]).T
        else:  # Cambered airfoil
            yc = self._yc(x)
            if self.convention == "american":  # Standard NACA definition
                theta = self._theta(x)
                curve = np.array([x + t * np.sin(theta), yc - t * np.cos(theta)]).T
            elif self.convention == "british":  # XFOIL style
                curve = np.array([x, yc - t]).T
            else:
                raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve
