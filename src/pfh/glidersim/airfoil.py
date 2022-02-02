"""
Models for the geometry and aerodynamic coefficients of 2D wing sections.

Geometry models are for computing mass distributions (for moments of
inertia), points on reference lines (such as the quarter-chord position,
frequently used by lifting line methods), and graphical purposes.

Coefficient models are for querying the section coefficients for lift,
drag, and pitching moment.
"""

from __future__ import annotations

import abc
import pathlib
import re
from itertools import product
from typing import Any, Callable, Protocol, TextIO, runtime_checkable

import numpy as np
import scipy.optimize
from numpy.lib import recfunctions as rfn
from scipy.interpolate import (
    LinearNDInterpolator,
    PchipInterpolator,
    RegularGridInterpolator,
)

from ._fast_interp import interp3d


__all__ = [
    "AirfoilCoefficientsInterpolator",
    "GridCoefficients",
    "XFLR5Coefficients",
    "AirfoilGeometry",
    "AirfoilGeometryInterpolator",
    "NACA",
]


def __dir__():
    return __all__


@runtime_checkable
class AirfoilCoefficientsInterpolator(Protocol):
    """
    Interface for classes that provide airfoil coefficients.

    Aerodynamic coefficients for a single airfoil are typicall calculated over
    a range of angle of attack and Reynolds number. For wings with control
    surfaces, the section profiles are variable; the control inputs choose the
    section profile from a set of airfoils by specifying their unique "airfoil
    index". The definition of airfoil index is up to the user, but common
    choices are deflection angle, normalized vertical deflection distance, etc.

    Coefficients interpolators calculate the aerodynamic coefficients over the
    range of airfoil index, angle of attack, and Reynolds number.
    """

    @abc.abstractmethod
    def Cl(self, ai, alpha, Re, clamp=False):
        """
        Compute the lift coefficient of the airfoil.

        Parameters
        ----------
        ai
            Airfoil index
        alpha : float [radians]
            The angle of attack
        Re : float [unitless]
            The Reynolds number
        clamp : bool
            Whether to clamp `alpha` to the highest non-nan value supported by
            the (ai, Re) pair.

        Returns
        -------
        Cl : float
        """

    @abc.abstractmethod
    def Cd(self, ai, alpha, Re, clamp=False):
        """
        Compute the drag coefficient of the airfoil.

        Parameters
        ----------
        ai
            Airfoil index
        alpha : float [radians]
            The angle of attack
        Re : float [unitless]
            The Reynolds number
        clamp : bool
            Whether to clamp `alpha` to the highest non-nan value supported by
            the (ai, Re) pair.

        Returns
        -------
        Cd : float
        """

    @abc.abstractmethod
    def Cm(self, ai, alpha, Re, clamp=False):
        """
        Compute the pitching coefficient of the airfoil.

        Parameters
        ----------
        ai
            Airfoil index
        alpha : float [radians]
            The angle of attack
        Re : float [unitless]
            The Reynolds number
        clamp : bool
            Whether to clamp `alpha` to the highest non-nan value supported by
            the (ai, Re) pair.

        Returns
        -------
        Cm : float
        """

    @abc.abstractmethod
    def Cl_alpha(self, ai, alpha, Re, clamp=False):
        """
        Compute the derivative of the lift coefficient versus angle of attack.

        Parameters
        ----------
        ai
            Airfoil index
        alpha : float [radians]
            The angle of attack
        Re : float [unitless]
            The Reynolds number
        clamp : bool
            Whether to return `0` if `alpha` exceeds the the highest non-nan
            value supported by the (ai, Re) pair.

        Returns
        -------
        Cl_alpha : float
        """


class GridCoefficients(AirfoilCoefficientsInterpolator):
    """
    Loads a set of polars from a CSV file.

    The CSV must contain a header with the following columns:
     * delta_d
     * alpha [degrees]
     * Re
     * Cl
     * Cd
     * Cm
     * Cl_alpha

    All values must lie on a grid over `delta`, `alpha`, and `Re`. The points
    must be on a grid, but the spacing in each dimension is not required to
    be uniform. If the grid has uniform spacing in each dimension, the
    GridCoefficients2 class is faster.

    FIXME: requires a valid `ai` column name
    """

    def __init__(
        self,
        file: str | pathlib.Path | TextIO,
        ai: str = "delta_d",
    ) -> None:
        names = np.loadtxt(file, max_rows=1, dtype=str, delimiter=",")
        data = np.genfromtxt(file, skip_header=1, names=list(names), delimiter=",")
        data.sort(order=[ai, "alpha", "Re"])

        # All points must be present (even if `nan`)
        self._ai = np.unique(data[ai])
        self._alpha = np.deg2rad(np.unique(data["alpha"]))
        self._alpha_step = self._alpha[1] - self._alpha[0]
        self._Re = np.unique(data["Re"])
        shape = (len(self._ai), len(self._alpha), len(self._Re))
        points = (self._ai, self._alpha, self._Re)

        self._Cl = RegularGridInterpolator(
            points, data["Cl"].reshape(shape), bounds_error=False
        )
        self._Cd = RegularGridInterpolator(
            points, data["Cd"].reshape(shape), bounds_error=False
        )
        self._Cm = RegularGridInterpolator(
            points, data["Cm"].reshape(shape), bounds_error=False
        )
        self._Cl_alpha = RegularGridInterpolator(
            points, data["Cl_alpha"].reshape(shape), bounds_error=False
        )

        # Trilinear interpolation uses the values at all 8 corners of the
        # bounding cube. If any values are nan then interpolation in that cube
        # will be nan. Clamping to the highest valid alpha requires finding the
        # cube containing the same ai and Re and the largest alpha.
        lai = len(self._ai)
        lRe = len(self._Re)
        non_nan = ~np.isnan(data["Cl"].reshape(shape))
        max_non_nan_indices = np.empty((lai, lRe), dtype=int)
        subgrid = np.empty((lai - 1, lRe - 1), dtype=int)
        for (D, R) in product(range(lai), range(lRe)):
            max_non_nan_indices[D, R] = np.max(np.nonzero(non_nan[D, :, R]))
        for (D, R) in product(range(lai - 1), range(lRe - 1)):
            subgrid[D, R] = np.min(max_non_nan_indices[D : D + 2, R : R + 2])
        self._max_alphas = self._alpha[subgrid]  # Upper-bound for each cube

    def _max_alpha(self, ai, Re):
        # These are not strictly correct since it clips ai and Re, but
        # querying the coefficients using an out-of-bounds ai or Re will
        # produce nan anyway.
        ai = ai.copy()
        Re = Re.copy()
        ai[ai < self._ai[0]] = self._ai[0]
        Re[Re < self._Re[0]] = self._Re[0]
        ix_d = np.argmax(ai[:, None] < self._ai, axis=-1) - 1
        ix_Re = np.argmax(Re[:, None] < self._Re, axis=-1) - 1
        return self._max_alphas[ix_d, ix_Re]

    def _query(self, f, clamp, ai, alpha, Re):
        ai, alpha, Re = np.broadcast_arrays(ai, alpha, Re / 1e6)

        # Set clamped sections to their maximum non-nan values by setting alpha
        # to be just inside the cube associated with the maximum valid alpha.
        if np.any(clamp):
            clamped = np.broadcast_to(clamp, alpha.shape)
            alpha = alpha.copy()
            max_alpha = self._max_alpha(ai[clamped], Re[clamped])
            alpha[clamped] = np.minimum(
                alpha[clamped],
                max_alpha - self._alpha_step * 0.0001,
            )
        return f((ai, alpha, Re))

    def Cl(self, ai, alpha, Re, clamp=False):
        return self._query(self._Cl, clamp, ai, alpha, Re)

    def Cd(self, ai, alpha, Re, clamp=False):
        return self._query(self._Cd, clamp, ai, alpha, Re)

    def Cm(self, ai, alpha, Re, clamp=False):
        return self._query(self._Cm, clamp, ai, alpha, Re)

    def Cl_alpha(self, ai, alpha, Re, clamp=False):
        ai, alpha, Re = np.broadcast_arrays(ai, alpha, Re / 1e6)
        out = self._Cl_alpha((ai, alpha, Re))
        if np.any(clamp):
            clamped = np.broadcast_to(clamp, out.shape)
            max_alpha = self._max_alpha(ai[clamped], Re[clamped])
            tmp = out[clamped]
            tmp[alpha[clamped] >= max_alpha] = 0
            out[clamped] = tmp
        return out


class GridCoefficients2(AirfoilCoefficientsInterpolator):
    """
    Loads a set of polars from a CSV file.

    The CSV must contain a header with the following columns:
     * delta_d
     * alpha [degrees]
     * Re
     * Cl
     * Cd
     * Cm
     * Cl_alpha

    All values must lie on a grid over `delta`, `alpha`, and `Re`. The points
    must be on a grid, and the spacing in each dimension must be uniform.  All
    values must lie on a grid over `delta`, `alpha`, and `Re`. The points must
    be on a grid, and the spacing in each dimension must be uniform. If the
    grid with non-uniform spacing in each dimension,
    use the GridCoefficients class.

    All values must lie on a grid over `delta`, `alpha`, and `Re`. The points
    must be on a grid, and the spacing in each dimension must be uniform. If
    the grid has non-uniform spacing, use the GridCoefficients class.

    FIXME: requires a valid `ai` column name
    """

    def __init__(
        self,
        file: str | pathlib.Path | TextIO,
        ai: str = "delta_d",
    ) -> None:
        names = np.loadtxt(file, max_rows=1, dtype=str, delimiter=",")
        data = np.genfromtxt(file, skip_header=1, names=list(names), delimiter=",")
        data.sort(order=[ai, "alpha", "Re"])

        # All points must be present (even if `nan`)
        self._ai = np.unique(data[ai])
        self._alpha = np.deg2rad(np.unique(data["alpha"]))
        self._alpha_step = self._alpha[1] - self._alpha[0]
        self._Re = np.unique(data["Re"])
        shape = (len(self._ai), len(self._alpha), len(self._Re))

        kwargs = {
            "a": (self._ai.min(), self._alpha.min(), self._Re.min()),
            "b": (self._ai.max(), self._alpha.max(), self._Re.max()),
            "h": (
                np.diff(self._ai)[0],
                np.diff(self._alpha)[0],
                np.diff(self._Re)[0],
            ),
            "k": 1,  # FIXME: why does it crash for k=3?
        }
        self._Cl = interp3d(f=data["Cl"].reshape(shape), **kwargs)
        self._Cd = interp3d(f=data["Cd"].reshape(shape), **kwargs)
        self._Cm = interp3d(f=data["Cm"].reshape(shape), **kwargs)
        self._Cl_alpha = interp3d(f=data["Cl_alpha"].reshape(shape), **kwargs)

        # Trilinear interpolation uses the values at all 8 corners of the
        # bounding cube. If any values are nan then interpolation in that cube
        # will be nan. Clamping to the highest valid alpha requires finding the
        # cube containing the same ai and Re and the largest alpha.
        lai = len(self._ai)
        lRe = len(self._Re)
        non_nan = ~np.isnan(data["Cl"].reshape(shape))
        max_non_nan_indices = np.empty((lai, lRe), dtype=int)
        subgrid = np.empty((lai - 1, lRe - 1), dtype=int)
        for (D, R) in product(range(lai), range(lRe)):
            max_non_nan_indices[D, R] = np.max(np.nonzero(non_nan[D, :, R]))
        for (D, R) in product(range(lai - 1), range(lRe - 1)):
            subgrid[D, R] = np.min(max_non_nan_indices[D : D + 2, R : R + 2])
        self._max_alphas = self._alpha[subgrid]  # Upper-bound for each cube

    def _max_alpha(self, ai, Re):
        # These are not strictly correct since it clips ai and Re, but
        # querying the coefficients using an out-of-bounds ai or Re will
        # produce nan anyway.
        ai = ai.copy()
        Re = Re.copy()
        ai[ai < self._ai[0]] = self._ai[0]
        Re[Re < self._Re[0]] = self._Re[0]
        ix_d = np.argmax(ai[:, None] < self._ai, axis=-1) - 1
        ix_Re = np.argmax(Re[:, None] < self._Re, axis=-1) - 1
        return self._max_alphas[ix_d, ix_Re]

    def _query(self, f, clamp, ai, alpha, Re):
        ai, alpha, Re = np.broadcast_arrays(ai, alpha, Re / 1e6)
        ai.flags.writeable = False  # Silence deprecation warnings
        alpha.flags.writeable = False
        Re.flags.writeable = False

        # Set clamped sections to their maximum non-nan values by setting alpha
        # to be just inside the cube associated with the maximum valid alpha.
        if np.any(clamp):
            clamped = np.broadcast_to(clamp, alpha.shape)
            alpha = alpha.copy()
            max_alpha = self._max_alpha(ai[clamped], Re[clamped])
            alpha[clamped] = np.minimum(
                alpha[clamped],
                max_alpha - self._alpha_step * 0.0001,
            )
        return f(ai, alpha, Re)

    def Cl(self, ai, alpha, Re, clamp=False):
        return self._query(self._Cl, clamp, ai, alpha, Re)

    def Cd(self, ai, alpha, Re, clamp=False):
        return self._query(self._Cd, clamp, ai, alpha, Re)

    def Cm(self, ai, alpha, Re, clamp=False):
        return self._query(self._Cm, clamp, ai, alpha, Re)

    def Cl_alpha(self, ai, alpha, Re, clamp=False):
        ai, alpha, Re = np.broadcast_arrays(ai, alpha, Re / 1e6)
        ai.flags.writeable = False  # Silence deprecation warnings
        alpha.flags.writeable = False
        Re.flags.writeable = False
        out = self._Cl_alpha(ai, alpha, Re)
        if np.any(clamp):
            clamped = np.broadcast_to(clamp, out.shape)
            max_alpha = self._max_alpha(ai[clamped], Re[clamped])
            tmp = out[clamped]
            tmp[alpha[clamped] >= max_alpha] = 0
            out[clamped] = tmp
        return out


class XFLR5Coefficients(AirfoilCoefficientsInterpolator):
    """
    Loads a set of XFLR5 polars (.txt) from a directory.

    Requirements:

    1. All `.txt` files in the directory are valid XFLR5 polar files, generated
       using a single test configuration over a range of Reynolds numbers.

    2. Filenames must use `<...>_ReN.NNN_<...>` to encode the Reynolds number
       in units of millions (so `920,000` is encoded as `0.920`).

    3. All polars will be included.

    FIXME: doesn't support clamping
    """

    def __init__(self, dirname: str, flapped: bool) -> None:
        self.flapped = flapped
        polars = self._load_xflr5_polar_set(dirname, flapped)
        self.polars = polars

        if flapped:
            columns = ["delta_d", "alpha", "Re"]
        else:
            columns = ["alpha", "Re"]

        points = rfn.structured_to_unstructured(polars[columns])
        self._Cl = LinearNDInterpolator(points, polars["Cl"])
        self._Cd = LinearNDInterpolator(points, polars["Cd"])
        self._Cm = LinearNDInterpolator(points, polars["Cm"])
        self._Cl_alpha = LinearNDInterpolator(points, polars["Cl_alpha"])

    @staticmethod
    def _load_xflr5_polar_set(dirname: str, flapped: bool):
        d = pathlib.Path(dirname)

        if not (d.exists() and d.is_dir()):
            raise ValueError(f"'{dirname}' is not a valid directory")

        polar_files = d.glob("*.txt")

        # Standard XFLR5 polar column names (renames CL/CD to Cl/Cd)
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

        # FIXME: handle cases with <= 1 polar files?

        polars = []
        for polar_file in polar_files:
            if match := re.search(r"_Re(\d+\.\d+)_", polar_file.name):
                Re = float(match.group(1))
            else:
                raise ValueError(f"Invalid filename {polar_file}; needs `Re`")
            data = np.genfromtxt(polar_file, skip_header=11, names=names)
            data["alpha"] = np.deg2rad(data["alpha"])

            # Smoothed `Cl` and `Cl_alpha` improve convergence
            poly = np.polynomial.Polynomial.fit(data["alpha"], data["Cl"], 10)
            data["Cl"] = poly(data["alpha"])
            Cl_alphas = poly.deriv()(data["alpha"])
            data = rfn.append_fields(data, "Cl_alpha", Cl_alphas)

            # Append the Reynolds number for the polar
            data = rfn.append_fields(data, "Re", np.full(data.shape[0], Re))

            if flapped:
                if match := re.search(r"_deltad(\d+\.\d+)_", polar_file.name):
                    delta_d = np.deg2rad(float(match.group(1)))
                else:
                    raise ValueError(
                        f"Invalid filename {polar_file}; needs `deltad`",
                    )

                data = rfn.append_fields(
                    data,
                    "delta_d",
                    np.full(data.shape[0], delta_d),
                )

            polars.append(data)

        return np.concatenate(polars)

    def Cl(self, ai, alpha, Re, clamped=None):
        Re = Re / 1e6
        if self.flapped:
            return self._Cl(ai, alpha, Re)
        else:
            return self._Cl(alpha, Re)

    def Cd(self, ai, alpha, Re, clamped=None):
        Re = Re / 1e6
        if self.flapped:
            return self._Cd(ai, alpha, Re)
        else:
            return self._Cd(alpha, Re)

    def Cm(self, ai, alpha, Re, clamped=None):
        Re = Re / 1e6
        if self.flapped:
            return self._Cm(ai, alpha, Re)
        else:
            return self._Cm(alpha, Re)

    def Cl_alpha(self, ai, alpha, Re, clamped=None):
        Re = Re / 1e6
        if self.flapped:
            return self._Cl_alpha(ai, alpha, Re)
        else:
            return self._Cl_alpha(alpha, Re)


# ---------------------------------------------------------------------------


def find_leading_edge(curve, definition: str):
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
    bracket = (0.4 * curve.x[-1], 0.6 * curve.x[-1])

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
    profile_curve : callable
        Profile xy-coordinates as a curve parametrized by the normalized
        arc-length `-1 <= r <= 1`, where `r = 0` is the leading edge, `r = 1`
        is the tip of the upper surface, and `r = -1` is the tip of the lower
        surface. Midpoints by length of the upper and lower surface curves are
        given by `r = 0.5` and `r = -0.5`.
    camber_curve : callable
        Mean camber curve xy-coordinates as a function of normalized arc-length
        `0 <= r <= 1`, where `r = 0` is the leading edge, `r = 1` is
        the trailing edge, and `r = 0.5` is the midpoint by length.
    thickness : callable
        Airfoil thickness as a function of `r`.
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
        self,
        profile_curve: Callable[[float], Any],
        camber_curve: Callable[[float], Any],
        thickness: Callable[[float], Any],
        convention: str,
        theta: float = 0,
        scale: float = 1,
    ) -> None:
        self._profile_curve = profile_curve
        self._camber_curve = camber_curve
        self._thickness = thickness
        self.convention = convention
        self.theta = theta
        self.scale = scale

    @classmethod
    def from_points(
        cls,
        points,
        convention: str,
        center: bool = False,
        derotate: bool = False,
        normalize: bool = False,
    ) -> AirfoilGeometry:
        """
        Construct an AirfoilGeometry from a set of airfoil xy-coordinates.

        By default, the input coordinates will be centered, normalized, and
        derotated so the leading edge is at the origin and the chord lies on
        the x-axis between 0 and 1.

        The input coordinates are treated as the "reference" airfoil. If the
        user provides coefficient data to `FoilSections`, then it will be
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
            Translate the curve leading edge to the origin. Default: False
        derotate : bool
            Rotate the the chord parallel to the x-axis. Default: False
        normalize : bool
            Scale the curve so the chord is unit length. Default: False

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
        #   d <= d_LE  ->  -1 <= r <= 0  <-  lower surface
        #   d >= d_LE  ->   0 <= r <= 1  <-  upper surface
        r = np.empty(d.shape)
        f = d <= d_LE
        r[f] = -1 + d[f] / d_LE
        r[~f] = (d[~f] - d_LE) / (d[-1] - d_LE)
        profile = PchipInterpolator(r, points, extrapolate=False)

        # Estimate the mean camber curve and thickness distribution as
        # functions of the normalized arc-length of the mean camber line.
        # FIXME: Crude, ignores convention
        N = 300
        r = (1 - np.cos(np.linspace(0, np.pi, N))) / 2  # `0 <= r <= 1`
        xyu = profile(r)
        xyl = profile(-r)
        xyc = (xyu + xyl) / 2
        t = np.linalg.norm(xyu - xyl, axis=1)
        r = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyc.T), axis=0))]
        r /= r[-1]
        camber = PchipInterpolator(r, (xyu + xyl) / 2, extrapolate=False)
        thickness = PchipInterpolator(r, t, extrapolate=False)

        return cls(profile, camber, thickness, convention, theta, scale)

    def profile_curve(self, r):
        """
        Compute points on the profile curve.

        Parameters
        ----------
        r : array_like of float, shape (N,)
            The curve parameter from -1..1, with -1 the lower surface trailing
            edge, 0 is the leading edge, and +1 is the upper surface trailing
            edge

        Returns
        -------
        points : array of float, shape (N, 2)
            The (x, y) coordinates of points on the airfoil at `r`.
        """
        return self._profile_curve(r)

    def profile_curve_tangent(self, r):
        """
        Compute the tangent unit vector at points on the profile curve.

        Parameters
        ----------
        r : array_like of float
            The profile curve parameter.

        Returns
        -------
        dxdy : array, shape (N, 2)
            The unit tangent lines at the specified points, oriented with
            increasing `r`, so the tangents trace from the lower surface to
            the upper surface.
        """
        dxdy = self._profile_curve.derivative()(r).T
        dxdy /= np.linalg.norm(dxdy, axis=0)
        return dxdy.T

    def profile_curve_normal(self, r):
        """
        Compute the normal unit vector at points on the profile curve.

        Parameters
        ----------
        r : array_like of float
            The profile curve parameter.

        Returns
        -------
        dxdy : array, shape (N, 2)
            The unit normal vectors at the specified points, oriented with
            increasing `r`, so the normals point "out" of the airfoil.
        """
        dxdy = (self._profile_curve.derivative()(r) * [1, -1]).T
        dxdy = dxdy[::-1] / np.linalg.norm(dxdy, axis=0)
        return dxdy.T

    def camber_curve(self, r):
        """
        Compute points on the camber curve.

        Parameters
        ----------
        r : array_like of float [percentage]
            Fractional position on the camber line, where `0 <= r <= 1`

        Returns
        -------
        array of float, shape (N, 2)
            The xy coordinate pairs of the camber line.
        """
        return self._camber_curve(r)

    def thickness(self, r):
        """
        Compute airfoil thickness.

        Parameters
        ----------
        r : array_like of float [percentage]
            Fractional position on the camber line, where `0 <= r <= 1`

        Returns
        -------
        thickness : array_like of float
        """
        return self._thickness(r)


class NACA(AirfoilGeometry):
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

    def __init__(
        self,
        code: int | str,
        *,
        open_TE: bool = False,
        convention: str = "perpendicular",
        N_points: int = 300,
    ) -> None:
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
            L = code // 10000  # Theoretical optimum lift coefficient
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
        surface_xyz = np.r_[xyl[::-1], xyu[1:]]
        r = np.r_[-dl[::-1] / dl[-1], du[1:] / du[-1]]
        profile_curve = PchipInterpolator(r, surface_xyz, extrapolate=False)
        r = dc / dc[-1]
        camber_curve = PchipInterpolator(r, xyc, extrapolate=False)
        thickness = PchipInterpolator(r, 2 * self._yt(x), extrapolate=False)
        super().__init__(profile_curve, camber_curve, thickness, convention)

    def _yt(self, x):
        """
        Compute the thickness of the airfoil.

        Whether the thickness is measured orthogonal to the camber curve or the
        chord depends on the convention. See the docstring for this class.
        """
        x = np.asfarray(x)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        open_TE = [0.2969, 0.1260, 0.3516, 0.2843, 0.1015]
        closed_TE = [0.2969, 0.1260, 0.3516, 0.2843, 0.1036]
        a0, a1, a2, a3, a4 = open_TE if self.open_TE else closed_TE
        return (
            5
            * self.tcr
            * (a0 * np.sqrt(x) - a1 * x - a2 * x**2 + a3 * x**3 - a4 * x**4)
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

        x = np.asfarray(x)
        assert np.all(x >= 0) and np.all(x <= 1)

        if self.series == 4:
            dyc = np.full(x.shape, 2 * m * (p - x))  # Common factors
            f = x < p  # Filter for the two cases, `x < p` and `x >= p`
            if p > 0:
                dyc[f] /= p**2
            dyc[~f] /= (1 - p) ** 2

        elif self.series == 5:
            dyc = np.full(x.shape, self.k1 / 6)  # Common factors
            f = x < m  # Filter for the two cases, `x < m` and `x >= m`
            dyc[f] *= 3 * x[f] ** 2 - 6 * m * x[f] + m**2 * (3 - m)
            dyc[~f] *= -(m**3)

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
        x = np.asfarray(x)
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be between 0 and 1")

        y = np.empty(x.shape)
        if self.series == 4:
            m, p = self.m, self.p
            f = x < p  # Filter for the two cases, `x < p` and `x >= p`
            if p > 0:
                y[f] = (m / p**2) * (2 * p * (x[f]) - (x[f]) ** 2)
            y[~f] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * (x[~f]) - (x[~f]) ** 2)

        elif self.series == 5:
            m, k1 = self.m, self.k1
            f = x < m  # Filter for the two cases, `x < m` and `x >= m`
            y[f] = (k1 / 6) * (
                x[f] ** 3 - 3 * m * (x[f] ** 2) + (m**2) * (3 - m) * x[f]
            )
            y[~f] = (k1 * m**3 / 6) * (1 - x[~f])

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
        x = np.asfarray(x)
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
        x = np.asfarray(x)
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


class AirfoilGeometryInterpolator:
    """Simple airfoil geometry interpolator."""

    def __init__(self, airfoils: dict[float, AirfoilGeometry]) -> None:
        ai = np.array(list(airfoils))
        ix = np.argsort(ai)
        self.ai = ai[ix]  # Airfoil indices, such as normalized `delta_d`
        self.airfoils = [airfoils[k] for k in ai]
        self._ai_min = ai.min()
        self._ai_max = ai.max()

    @property
    def index_bounds(self) -> tuple[float, float]:
        return (self._ai_min, self._ai_max)

    def _neighbors(self, ai):
        """Find the bounding indices and their distances."""
        i0 = np.empty(np.shape(ai), dtype=int)
        i1 = np.empty(np.shape(ai), dtype=int)
        p0 = np.empty(np.shape(ai))
        p1 = np.empty(np.shape(ai))
        exact_match = np.isclose(ai[..., None], self.ai)
        if np.any(exact_match):
            matches = np.nonzero(exact_match)
            i0[matches[:-1]] = self.ai[matches[-1]]
            i1[matches[:-1]] = 0
            p0[matches[:-1]] = 1
            p1[matches[:-1]] = 0
        others = np.nonzero(~np.any(exact_match, axis=-1))
        i0[others] = np.argmax(~(ai[others][..., None] >= self.ai), axis=-1) - 1
        i1[others] = np.argmax(ai[others][..., None] < self.ai, axis=-1)
        delta = self.ai[i1[others]] - self.ai[i0[others]]
        p0[others] = (self.ai[i1[others]] - ai[others]) / delta
        p1[others] = (ai[others] - self.ai[i0[others]]) / delta
        return (i0, i1, p0, p1)

    def _interpolate(self, func, ai, r):
        """Interpolate `func(r)` between two indexed airfoils."""
        if np.any(ai < self._ai_min) or np.any(ai > self._ai_max):
            raise ValueError(f"Airfoil index {ai} is out of bounds")

        # FIXME: hack to work with scalar `ai` and `r`. Indexing doesn't work
        # with scalars and arrays of dimension 0 so those get reshaped to 1D,
        # but that add an extra outter dimension into the output that should
        # be dropped.
        squeeze = np.ndim(ai) == 0 and np.ndim(r) == 0

        ai = np.atleast_1d(ai)
        r = np.atleast_1d(r)
        ai, r = np.broadcast_arrays(ai, r)
        if func in {"profile_curve", "camber_curve"}:
            out = np.zeros((*np.shape(r), 2))
        else:
            out = np.zeros(np.shape(r))

        # Function calls are expensive, so coalesce them across the groups.
        # FIXME: could group both f0 and f1 calls, but good enough for now.
        i0, i1, p0, p1 = self._neighbors(ai)
        for i in range(len(self.ai)):
            if func == "profile_curve":
                f0 = self.airfoils[i].profile_curve
                f1 = self.airfoils[i].profile_curve
            elif func == "camber_curve":
                f0 = self.airfoils[i].camber_curve
                f1 = self.airfoils[i].camber_curve
            elif func == "thickness":
                f0 = self.airfoils[i].thickness
                f1 = self.airfoils[i].thickness

            _i0 = np.nonzero(i0 == i)
            _i1 = np.nonzero(i1 == i)

            if func in {"profile_curve", "camber_curve"}:
                out[_i0] += p0[_i0][..., None] * f0(r[_i0])
                out[_i1] += p1[_i1][..., None] * f1(r[_i1])
            else:
                out[_i0] += p0[_i0] * f0(r[_i0])
                out[_i1] += p1[_i1] * f1(r[_i1])

        if squeeze:
            out = out[0]  # Drop the first dimension of size 1

        return out

    def profile_curve(self, ai, r):
        return self._interpolate("profile_curve", ai, r)

    def camber_curve(self, ai, r):
        return self._interpolate("camber_curve", ai, r)

    def thickness(self, ai, r):
        return self._interpolate("thickness", ai, r)
