"""
Models for airfoil geometries and airfoil coefficients.

Geometry models are for computing mass distributions (for moments of inertia),
points on reference lines (such as the quarter-chord position, frequently used
by lifting line methods), and graphical purposes.

Coefficient models are for evaluating the section coefficients for lift, drag,
and pitching moment.
"""

import abc
import pathlib
import re

import numpy as np
from numpy.lib import recfunctions as rfn

import pandas as pd

import scipy.optimize
from scipy.integrate import simps
from scipy.interpolate import CloughTocher2DInterpolator as Clough2D
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import LinearNDInterpolator


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
        delta_angle = np.arctan(delta)  # tan(delta_angle) = delta/chord
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
            poly = np.polynomial.Polynomial.fit(alphas, CLs, 7)
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


class XFLR5Coefficients:
    """
    Loads a set of XFLR5 polars (.txt) from a directory.

    Requirements:

    1. All `.txt` files in the directory are valid XFLR5 polar files, generated
       using a single test configuration over a range of Reynolds numbers.

    2. Filenames must use `<...>_ReN.NNN_<...>` to encode the Reynolds number
       in units of millions (so `920,000` is encoded as `0.920`).

    3. All polars will be included.
    """
    def __init__(self, dirname):
        polars = self._load_xflr5_polar_set(dirname)
        points = rfn.structured_to_unstructured(polars[["alpha", "Re"]])
        self.Cl = LinearNDInterpolator(points, polars["Cl"])
        self.Cd = LinearNDInterpolator(points, polars["Cd"])
        self.Cm = LinearNDInterpolator(points, polars["Cm"])
        self.Cl_alpha = LinearNDInterpolator(points, polars["Cl_alpha"])

    @staticmethod
    def _load_xflr5_polar_set(dirname):
        d = pathlib.Path(dirname)

        if not (d.exists() and d.is_dir()):
            raise ValueError(f"'{dirname}' is not a valid directory")

        polar_files = d.glob("*.txt")

        # Standard XFLR5 polar column names (changes CL/CD to Cl/Cd)
        names = [
            "alpha",
            "Cl",
            "Cd",
            "CDp",
            "Cm",
            "Top Xtr",
            "Bot Xtr",
            "Cpmin",
            "Chinge",
            "XCp",
        ]

        # FIXME: handle cases with <= 1 polar files

        polars = []
        for polar_file in polar_files:
            Re = re.search("_Re(\d\.\d\d\d)_", polar_file.name).group(1)
            Re = float(Re) * 1e6
            data = np.genfromtxt(polar_file, skip_header=11, names=names)
            data['alpha'] = np.deg2rad(data['alpha'])

            # Smooth `CL` and compute a smoothed `CL_alpha` (improves convergence)
            poly = np.polynomial.Polynomial.fit(data['alpha'], data['Cl'], 10)
            data['Cl'] = poly(data['alpha'])
            Cl_alphas = poly.deriv()(data['alpha'])
            data = rfn.append_fields(data, "Cl_alpha", Cl_alphas)

            # Append the Reynolds number for the polar
            data = rfn.append_fields(data, "Re", np.full(data.shape[0], Re))

            polars.append((Re, data))

        polars = sorted(polars, key=lambda p: p[0])

        return np.concatenate([p[1] for p in polars])


# ---------------------------------------------------------------------------


def find_leading_edge(curve, definition):
    """
    Find the parametric coordinate defining the leading edge of an airfoil.

    Parameters
    ----------
    curve : PchipInterpolator
        An airfoil curve parametrized by absolute arc-length `d`.
    definition : {"smallest-radius", "chord", "vertical-face"}
        The criteria for selecting a point as the leading edge.

        smallest-radius: defines the leading edge as the point near the middle
        of the curve with the smallest curvature. This should be used when an
        airfoil will be characterized by a thickness distribution measured
        perpendicular to the camber line. It produces the same value whether
        the airfoil will be rotated or not.

        chord: defines the leading edge as the point that produces a chord that
        is perpendicular to the surface at that point. This should be used when
        an airfoil will be characterized by a thickness distribution measured
        vertically, and the airfoil will need to be derotated. If the airfoil
        will not be derotated, use the "x-axis" method.

        vertical-face: defines the leading edge as the point near the middle of
        the curve where the slope is vertical. This definition is suitable for
        airfoils that will be characterized by a vertical thickness
        distribution and will not be derotated. For airfoils that will be
        derotated, use the "chord" method.
    """
    def _target(d, curve, derivative, TE=None):
        """Optimization target for the "chord" and "x-axis" methods."""
        dydx = derivative(d)  # Tangent line at the proposal
        chord = curve(d) - TE if TE is not None else [1, 0]
        dydx /= np.linalg.norm(dydx)
        chord /= np.linalg.norm(chord)
        return dydx @ chord

    definitions = {"smallest-radius", "chord", "vertical-face"}
    if definition not in definitions:
        raise ValueError(f"`definition` must be one of `{definitions}`")

    # For all methods, assume the LE is between 40-60% of the total length
    bracket = [0.4 * curve.x[-1], 0.6 * curve.x[-1]]

    if definition == "smallest-radius":
        # The second derivative of a PchipInterpolator is very noisy, so it's a
        # bad idea to rely on optimization algorithms. A brute force sampling
        # method is simple and works well enough.
        d_resolution = 50000  # The number of points to sample
        ds = np.linspace(*bracket, d_resolution)
        kappa = np.sqrt(np.sum(curve.derivative().derivative()(ds) ** 2, axis=1))
        d_LE = ds[np.argmax(kappa)]  # The point of smallest curvature
    else:
        if definition == "chord":
            TE = (curve(curve.x[0]) + curve(curve.x[-1])) / 2
        else:  # definition == "vertical-face"
            TE = None
        result = scipy.optimize.root_scalar(
            _target,
            args=(curve, curve.derivative(), TE),
            bracket=bracket,
        )
        d_LE = result.root

    return d_LE


class AirfoilGeometry:
    """
    A class for describing wing section profiles.

    Provides the surface curve, mean camber curve, thickness, and dimensionless
    mass properties of the airfoil.

    Parameters
    ----------
    surface_curve : PchipInterpolator
        Profile xy-coordinates as a curve parametrized by the normalized
        arc-length `-1 <= sa <= 1`, where `sa = 0` is the leading edge, `sa =
        1` is the tip of the upper surface, and `sa = -1` is the tip of the
        lower surface. Midpoints by length of the upper and lower surface
        curves are given by `sa = 0.5` and `sa = -0.5`.
    camber_curve : PchipInterpolator
        Mean camber curve xy-coordinates as a function of normalized arc-length
        `0 <= pc <= 1`, where `pc = 0` is the leading edge, `pc = 1` is
        the trailing edge, and `pc = 0.5` is the midpoint by length.
    convention : {"perpendicular", "vertical"}
        Whether the airfoil thickness is measured perpendicular to the mean
        camber line or vertically (perpendicular to the chord).
    theta : float [radians]
        The angle between the chord and the x-axis that was removed during
        derotation. Useful for converting `alpha` of the derotated airfoil into
        `alpha` of the reference foil.
    scale : float [unitless]
        The rescaling factor due to normalization. Useful for adjusting an
        existing set of coefficients for the original foil to account for the
        normalization.
    """

    def __init__(
        self, surface_curve, camber_curve, thickness, convention, theta=0, scale=1,
    ):
        self._surface_curve = surface_curve
        self._camber_curve = camber_curve
        self._thickness = thickness
        self.convention = convention
        self.theta = theta
        self.scale = scale

    @classmethod
    def from_points(
        cls, points, convention, center=True, derotate=True, normalize=True,
    ):
        """
        Construct an AirfoilGeometry from a set of airfoil xy-coordinates.

        By default, the input coordinates will be centered, normalized, and
        derotated so the leading edge is at the origin and the chord lies on
        the x-axis between 0 and 1.

        The input coordinates are treated as the "reference" airfoil. If the
        user provides coefficient data to the `Airfoil` class, then it will be
        assumed that those coefficients were computed for the reference
        airfoil. If the reference coordinates are rotated and/or normalized by
        this function, the `theta` and `scale` properties allow those reference
        coefficients to be used with the modified airfoil. For example, if the
        lift coefficient for the reference airfoil was given by `Cl(alpha)`,
        then the lift coefficient for the modified airfoil can be computed as
        `scale * Cl(alpha + theta)`. (Centering has no effect on the
        coefficients.)

        Parameters
        ----------
        points : array of float, shape (N,2)
            The xy-coordinates of the airfoil in counter-clockwise order.
        convention : {"perpendicular", "vertical"}
            Whether the airfoil thickness is measured perpendicular to the mean
            camber line or vertically (the y-axis distance).
        center : bool
            Translate the curve leading edge to the origin. Default: True
        derotate : bool
            Rotate the the chord parallel to the x-axis. Default: True
        normalize : bool
            Scale the curve so the chord is unit length. Default: True

        Returns
        -------
        AirfoilGeometry
        """
        conventions = {"perpendicular", "vertical"}
        if convention not in conventions:
            raise ValueError(f"`convention` must be one of `{conventions}`")

        points = points[np.r_[True, np.diff(points.T).any(axis=0)]]  # Deduplicate
        points = points[::-1]  # I find clockwise orientations more natural
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points.T), axis=0))]
        raw_curve = PchipInterpolator(d, points)
        if convention == "perpendicular":
            definition = "smallest-radius"
        else:
            definition = "chord" if derotate else "vertical-face"
        d_LE = find_leading_edge(raw_curve, definition)
        LE = raw_curve(d_LE)
        TE = (raw_curve(d[0]) + raw_curve(d[-1])) / 2
        chord = TE - LE

        if derotate:
            theta = -np.arctan2(chord[1], chord[0])
            st, ct = np.sin(theta), np.cos(theta)
            R = np.array([[ct, -st], [st, ct]])
            points = (R @ (points - LE).T).T + LE
        else:
            theta = 0

        if center:
            points -= LE

        if normalize:
            scale = 1 / np.linalg.norm(chord)
            points *= scale
        else:
            scale = 1

        # Parametrize by the normalized arc-length for each surface:
        #   d <= d_LE  ->  -1 <= sa <= 0  <-  lower surface
        #   d >= d_LE  ->   0 <= sa <= 1  <-  upper surface
        sa = np.empty(d.shape)
        f = d <= d_LE
        sa[f] = -1 + d[f] / d_LE
        sa[~f] = (d[~f] - d_LE) / (d[-1] - d_LE)
        surface = PchipInterpolator(sa, points, extrapolate=False)

        # Estimate the mean camber curve and thickness distribution as
        # functions of the normalized arc-length of the mean camber line.
        # FIXME: Crude, ignores convention
        N = 300
        sa = (1 - np.cos(np.linspace(0, np.pi, N))) / 2  # `0 <= sa <= 1`
        xyu = surface(sa)
        xyl = surface(-sa)
        xyc = (xyu + xyl) / 2
        t = np.linalg.norm(xyu - xyl, axis=1)
        pc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyc.T), axis=0))]
        pc /= pc[-1]
        camber = PchipInterpolator(pc, (xyu + xyl) / 2, extrapolate=False)
        thickness = PchipInterpolator(pc, t, extrapolate=False)

        return cls(surface, camber, thickness, convention, theta, scale)

    def mass_properties(self, sa_upper=0, sa_lower=0, N=200):
        """
        Calculate the inertial properties for the curves and planar area.

        These unitless magnitudes, centroids, and inertia matrices can be
        scaled by the physical units of the target application in order to
        calculate the upper and lower surface areas, internal volume, and
        inertia matrix of a 3D wing.

        This procedure treats the 2D geometry as perfectly flat 3D objects,
        with a new `z` axis added according to the right-hand rule. See
        "Notes" for more details.

        Parameters
        ----------
        sa_upper, sa_lower : float
            The starting coordinates of the upper and lower surfaces. Requires
            that `-1 <= sa_lower <= sa_upper, 1`.
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
        if sa_lower < -1:
            raise ValueError("Required: sa_lower >= -1")
        if sa_lower > sa_upper:
            raise ValueError("Required: sa_lower <= sa_upper")
        if sa_upper > 1:
            raise ValueError("Required: sa_upper <= 1")

        # -------------------------------------------------------------------
        # 1. Area calculations

        sa = (1 - np.cos(np.linspace(0, np.pi, N))) / 2  # `0 <= sa <= 1`
        top = self.surface_curve(sa).T  # Top half (above sa = 0)
        bottom = self.surface_curve(-sa).T  # Bottom half (below sa = 0)
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

        su = np.linspace(sa_upper, 1, N)
        sl = np.linspace(sa_lower, -1, N)
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

    def surface_curve(self, sa):
        """
        Compute points on the surface curve.

        Parameters
        ----------
        sa : array_like of float, shape (N,)
            The curve parameter from -1..1, with -1 the lower surface trailing
            edge, 0 is the leading edge, and +1 is the upper surface trailing
            edge

        Returns
        -------
        points : array of float, shape (N, 2)
            The (x, y) coordinates of points on the airfoil at `sa`.
        """
        return self._surface_curve(sa)

    def surface_curve_tangent(self, sa):
        """
        Compute the tangent unit vector at points on the surface curve.

        Parameters
        ----------
        sa : array_like of float
            The surface curve parameter.

        Returns
        -------
        dxdy : array, shape (N, 2)
            The unit tangent lines at the specified points, oriented with
            increasing `sa`, so the tangents trace from the lower surface to
            the upper surface.
        """
        dxdy = self._surface_curve.derivative()(sa).T
        dxdy /= np.linalg.norm(dxdy, axis=0)
        return dxdy.T

    def surface_curve_normal(self, sa):
        """
        Compute the normal unit vector at points on the surface curve.

        Parameters
        ----------
        sa : array_like of float
            The surface curve parameter.

        Returns
        -------
        dxdy : array, shape (N, 2)
            The unit normal vectors at the specified points, oriented with
            increasing `sa`, so the normals point "out" of the airfoil.
        """
        dxdy = (self._surface_curve.derivative()(sa) * [1, -1]).T
        dxdy = (dxdy[::-1] / np.linalg.norm(dxdy, axis=0))
        return dxdy.T

    def camber_curve(self, pc):
        """
        Compute points on the camber curve.

        Parameters
        ----------
        pc : array_like of float [percentage]
            Fractional position on the camber line, where `0 <= pc <= 1`

        Returns
        -------
        array of float, shape (N, 2)
            The xy coordinate pairs of the camber line.
        """
        return self._camber_curve(pc)

    def thickness(self, pc):
        """
        Compute airfoil thickness.

        Parameters
        ----------
        pc : array_like of float [percentage]
            Fractional position on the camber line, where `0 <= pc <= 1`

        Returns
        -------
        thickness : array_like of float
        """
        return self._thickness(pc)


class NACA(AirfoilGeometry):
    def __init__(self, code, *, open_TE=False, convention="perpendicular", N_points=300):
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
            Generate airfoils with an open trailing edge. Default: False.
        convention : {"perpendicular", "vertical"}, optional
            The convention to use for defining the airfoil thickness.
            Default: "perpendicular".

            The "perpendicular" convention (sometimes called the "American"
            convention) measures airfoil thickness perpendicular to the mean
            camber line. The "vertical" convention (sometimes called the
            "British" convention) measures airfoil thickness in vertical strips
            (the y-axis distance between points on the upper and lower
            surfaces).

            The "American" convention is used here since it was the original
            definition (see [0]_), but the "British" convention is available
            in case the output needs to match the popular tool "XFOIL".
        N_points : integer
            The number of sample points from each surface. Default: 300

        References
        ----------

        .. [0] Jacobs, Eastman N., Ward, Kenneth E., Pinkerton, Robert M. "The
           characteristics of 78 related airfoil sections from tests in the
           variable-density wind tunnel". NACA Technical Report 460. 1933.
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

        valid_conventions = {"perpendicular", "vertical"}
        if convention not in valid_conventions:
            raise ValueError("The convention must be 'perpendicular' or 'vertical'")

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

        x = (1 - np.cos(np.linspace(0, np.pi, N_points))) / 2
        xyu = self._xyu(x)
        xyl = self._xyl(x)
        xyc = np.c_[x, self._yc(x)]
        du = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyu.T), axis=0))]
        dl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyl.T), axis=0))]
        dc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyc.T), axis=0))]
        pc = dc / dc[-1]
        surface_xyz = np.r_[xyl[::-1], xyu[1:]]
        sa = np.r_[-dl[::-1] / dl[-1], du[1:] / du[-1]]
        surface_curve = PchipInterpolator(sa, surface_xyz, extrapolate=False)
        camber_curve = PchipInterpolator(pc, xyc, extrapolate=False)
        thickness = PchipInterpolator(pc, 2 * self._yt(x), extrapolate=False)
        super().__init__(surface_curve, camber_curve, thickness, convention)

    def _yt(self, x):
        """
        Compute the thickness of the airfoil.

        Whether the thickness is measured orthogonal to the camber curve or the
        chord depends on the convention. See the docstring for this class.
        """
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

        x = np.asarray(x)
        assert np.all(x >= 0) and np.all(x <= 1)

        if self.series == 4:
            dyc = np.full(x.shape, 2 * m * (p - x))  # Common factors
            f = x < p  # Filter for the two cases, `x < p` and `x >= p`
            if p > 0:
                dyc[f] /= p ** 2
            dyc[~f] /= (1 - p) ** 2

        elif self.series == 5:
            dyc = np.full(x.shape, self.k1 / 6)  # Common factors
            f = x < m  # Filter for the two cases, `x < m` and `x >= m`
            dyc[f] *= 3 * x[f] ** 2 - 6 * m * x[f] + m ** 2 * (3 - m)
            dyc[~f] *= -m ** 3

        else:
            raise RuntimeError(f"Invalid NACA series '{self.series}'")

        return np.arctan(dyc)

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
        x = np.asarray(x)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        y = np.empty(x.shape)
        if self.series == 4:
            m, p = self.m, self.p
            f = x < p  # Filter for the two cases, `x < p` and `x >= p`
            if p > 0:
                y[f] = (m / p ** 2) * (2 * p * (x[f]) - (x[f]) ** 2)
            y[~f] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * (x[~f]) - (x[~f]) ** 2)

        elif self.series == 5:
            m, k1 = self.m, self.k1
            f = x < m  # Filter for the two cases, `x < m` and `x >= m`
            y[f] = (k1 / 6) * (x[f] ** 3 - 3 * m * (x[f] ** 2) + (m ** 2) * (3 - m) * x[f])
            y[~f] = (k1 * m ** 3 / 6) * (1 - x[~f])

        else:
            raise RuntimeError(f"Invalid NACA series '{self.series}'")

        return y

    def _xyu(self, x):
        """
        Compute the x- and y-coordinates of points on the upper surface.

        Returns both `x` and `y` because the "perpendicular" convention
        computes the surface curve coordinates orthogonal to the camber curve
        instead of the chord, so the `x` coordinate for the surface curve will
        not be the same as the `x` coordinate that parametrizes the chord
        (unless the airfoil is symmetric, in which case the camber curve lines
        directly on the chord). For the "vertical" convention, the input and
        output `x` will always be the same.

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

        t = self._yt(x)

        if self.m == 0:  # Symmetric airfoil
            curve = np.array([x, t]).T
        else:  # Cambered airfoil
            yc = self._yc(x)
            if self.convention == "perpendicular":  # Standard NACA definition
                theta = self._theta(x)
                curve = np.array([x - t * np.sin(theta), yc + t * np.cos(theta)]).T
            elif self.convention == "vertical":  # XFOIL style
                curve = np.array([x, yc + t]).T
            else:
                raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve

    def _xyl(self, x):
        """
        Compute the x- and y-coordinates of points on the lower surface.

        Returns both `x` and `y` because the "perpendicular" convention
        computes the surface curve coordinates orthogonal to the camber curve
        instead of the chord, so the `x` coordinate for the surface curve will
        not be the same as the `x` coordinate that parametrizes the chord
        (unless the airfoil is symmetric, in which case the camber curve lines
        directly on the chord). For the "vertical" convention, the input and
        output `x` will always be the same.

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

        t = self._yt(x)
        if self.m == 0:  # Symmetric airfoil
            curve = np.array([x, -t]).T
        else:  # Cambered airfoil
            yc = self._yc(x)
            if self.convention == "perpendicular":  # Standard NACA definition
                theta = self._theta(x)
                curve = np.array([x + t * np.sin(theta), yc - t * np.cos(theta)]).T
            elif self.convention == "vertical":  # XFOIL style
                curve = np.array([x, yc - t]).T
            else:
                raise RuntimeError(f"Invalid convention '{self.convention}'")
        return curve
