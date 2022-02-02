"""Models that specify the scale, position, and orientation of foil sections."""

from __future__ import annotations

import warnings
from typing import Callable, cast

import numpy as np
import scipy
import scipy.interpolate


__all__ = [
    "EllipticalChord",
    "EllipticalArc",
    "PolynomialTorsion",
    "FlatYZ",
    "FoilLayout",
]


def __dir__():
    return __all__


class EllipticalChord:
    """
    Build an elliptical chord distribution as a function of the section index.

    Parameters
    ----------
    root : float [length]
        The length of the central chord
    tip : float [length]
        The length of the wing tips
    """

    def __init__(self, root: float, tip: float) -> None:
        self.A = 1 / np.sqrt(1 - (tip / root) ** 2)
        self.B = root

    def __call__(self, s):
        s = np.asfarray(s)
        c = self.B * np.sqrt(1 - (s / self.A) ** 2)
        return c


class EllipticalArc:
    """
    Elliptical arc as a function of the section index.

    In this context the name is confusing because "arc" has two meanings: one
    for the traditional "elliptical arc segment", and one for the "arc" of a
    foil geometry (which is being modeled by an "elliptical arc segment").

    Expects the section index to be defined as the linear distance along the
    `yz` curve (that is, `s = y_flat / (b_flat / 2)`).

    This model scales the curve to a total length of 2, making it suitable
    for use with `FoilLayout` which will scale the curve length to `b_flat`.

    Parameters
    ----------
    mean_anhedral : float [degrees]
        The average anhedral angle of the wing sections, measured as the angle
        between the xy-plane and the line from the central section to the wing
        tip projected onto the yz-plane.
    tip_anhedral : float [degrees], optional
        The anhedral angle of the right wing tip section, measured as the angle
        between the xy-plane and the section y-axis projected onto the
        yz-plane. The wing is symmetric, so the left wing tip anhedral is the
        negative of this value. This optional value must satisfy `2 *
        mean_anhedral <= tip_anhedral <= 90`. If no value is specified the
        default is `2 * mean_anhedral`, which results in a circular arc.
    """

    def __init__(
        self,
        mean_anhedral: float,
        tip_anhedral: float | None = None,
    ) -> None:
        if tip_anhedral is None:  # Assume circular
            tip_anhedral = 2 * mean_anhedral

        if not (0 <= mean_anhedral <= 45):
            raise ValueError("mean_anhedral must be between 0 and 45 [degrees]")
        if not (0 <= tip_anhedral <= 90):
            raise ValueError("tip_anhedral must be between 0 and 90 [degrees]")
        if tip_anhedral < 2 * mean_anhedral:
            raise ValueError("tip_anhedral must be >= 2 * mean_anhedral")

        # Very small angles produce divide-by-zero, just assume the user wants
        # a zero-angle and "do the right thing".
        if mean_anhedral < 0.001:
            warnings.warn("Very small mean_anhedral. Use a FlatYZ.")
            mean_anhedral = 0.001

        mean_anhedral = np.deg2rad(mean_anhedral)
        tip_anhedral = np.deg2rad(tip_anhedral)

        # Two cases: perfectly circular, or elliptical
        if np.isclose(2 * mean_anhedral, tip_anhedral):  # Circular
            A = B = 1
            t_min = np.pi / 2 - 2 * mean_anhedral
        else:  # Elliptical
            v1 = 1 - np.tan(mean_anhedral) / np.tan(tip_anhedral)
            v2 = 1 - 2 * np.tan(mean_anhedral) / np.tan(tip_anhedral)
            A = v1 / np.sqrt(v2)
            B = np.tan(mean_anhedral) * v1 / v2
            t_min = np.arccos(1 / A)

        # FIXME: hack to avoid sign-flip issues at `np.sin(pi)`, which makes
        #        the normalized dyds explode (sign flip +1 to -1)
        if t_min < 1e-10:
            t_min = 1e-10

        # The parameter `t` is inconvenient. Instead, the FoilLayout expects
        # the yz-curve to be a function of `s`, so map `s -> t`. The FoilLayout
        # also expects the total length to be `2` (so it can be easily
        # rescaled).  Compute the the linear distances and scale the curve to
        # length = 2
        N = 101  # Number of samples to build `_s2t` (FIXME: overkill!)
        t = np.linspace(np.pi / 2, t_min, N)
        points = np.stack([A * np.cos(t), B * np.sin(t)])
        d = np.linalg.norm(np.diff(points), axis=0).cumsum()  # Distances
        self.A = A / d[-1]
        self.B = B / d[-1]
        d /= d[-1]

        # Section indices from the linear distances along the length=2 curve
        s = np.hstack([-d[::-1], 0, d])
        t = np.linspace(np.pi + t_min, 2 * np.pi - t_min, 2 * N - 1)
        self._t = scipy.interpolate.PchipInterpolator(s, t)
        self._dtds = self._t.derivative(1)

    def __call__(self, s):
        t = self._t(s)
        x = self.A * np.cos(t)
        y = self.B * np.sin(t)
        return np.stack((x, y), axis=-1) + [0, self.B]

    def derivative(self, s):
        t = self._t(s)
        dydt = self.A * -np.sin(t)
        dzdt = self.B * np.cos(t)
        dtds = self._dtds(s)
        return np.stack((dydt, dzdt), axis=-1) * dtds[..., None]


class PolynomialTorsion:
    """
    A functor that encodes geometric torsion as a polynomial.

    The domain is `[-1, 1]`, and the range is symmetric about the origin.
    Inputs `0 <= abs(s) <= start` are zero; inputs `start < abs(s) <= 1` grow
    from `0` to `peak` at a rate controlled by `exponent`.

    For example, if `start = 0.5`, and `exponent = 2`, then inputs `s` between
    `[-0.5, 0.5]` are zero, and `0.5 < abs(s) <= 1` grow quadratically to the
    peak value `peak` at `abs(s) == 1`.

    Parameters
    ----------
    start: float
        Absolute value section index where the curve begins.

    exponent : float
        The growth rate. Controls the steepness of the curve.

    peak : float
        The peak value of the curve at `s = 1`.
    """

    def __init__(self, start: float, exponent: float, peak: float) -> None:
        self.start = start
        self.exponent = exponent
        self.peak = np.deg2rad(peak)

    def __call__(self, s):
        s = np.asfarray(s)
        values = np.zeros(s.shape)
        m = abs(s) > self.start  # Mask
        if np.any(m):
            p = (abs(s[m]) - self.start) / (1 - self.start)
            # p /= p[-1]  # `p` goes from 0 to 1
            values[m] = self.peak * p**self.exponent
        return values


class FlatYZ:
    """Helper class for completely flat wings (no dihedral anywhere)."""

    def __call__(self, s):
        """Define `(y, z) = (s, 0)` for all sections."""
        _0 = np.full(np.shape(s), 0.0)
        return np.stack((s, _0), axis=-1)

    def derivative(self, s):
        """Define `(dyds, dzds) = (1, 0)` for all sections."""
        return np.broadcast_to([1.0, 0.0], (*np.shape(s), 2)).copy()


class FoilLayout:
    """
    FIXME: docstring. Describe the geometry.

    All input values must be normalized by `b_flat = 2`. Output values can be
    scaled as needed to achieve a given `b`, `b_flat`, `S`, or `S_flat`.

    Conceptually, this specifies a design target: an idealized goal to produce
    with a physical foil. Rigid foils can create this surface exactly, but
    flexible wings, like parafoils, can only approximate this shape through
    the internal structure of cells.

    Parameters
    ----------
    r_x : float or callable
        A ratio from 0 to 1 that defines what location on each chord is located
        at the x-coordinate defined by `x`. This can be a constant or a
        function of the section index. For example, `r_x = 1` says that `x` is
        specifying the x-coordinate of the trailing edge.
    x : float or callable
        The x-coordinates of each section as a function of the section index.
        Each chord is shifted forward until the x-coordinate of its leading
        edge is at `c * r_x`.
    r_yz : float or callable
        A ratio from 0 to 1 that defines the chord position of the `yz` curve.
        This can be a constant or a function of the section index. For example,
        `r_yz = 0.25` says that the `yz` curve is specifying the yz-coordinates
        of the quarter-chord.
    yz : callable
        The yz-coordinates of each section as a function of the section index.
        This curve shapes the yz-plane view of the inflated wing. Must be a
        functor with a `derivative` method.
    c : float or callable
        The section chord lengths as a function of section index.
    theta : float or callable, optional
        Geometric torsion as a function of the section index. These angles
        specify a positive rotation about the local (section) y-axis. Values
        must be in radians. Default: `0` at all sections.
    center : bool, optional
        Whether to center the surface such that the leading edge of the central
        section defines the origin. Default: True

    Notes
    -----
    Normalizing everything by `b_flat = 2` simplifies the equations since
    section indices `s` are on the same scale as the spatial coordinates.
    (If you flatten `yz`, then `y` is just `s`.)
    """

    def __init__(
        self,
        r_x: float | Callable,
        x: float | Callable,
        r_yz: float | Callable,
        yz,  # Must be a functor and include a `derivative` method
        c: float | Callable,
        theta: float | Callable = 0,
        center: bool = True,
    ) -> None:
        if callable(r_x):
            self.r_x = r_x
        else:
            self.r_x = lambda s: np.full(np.shape(s), float(r_x))  # type: ignore [arg-type]

        if callable(x):
            self.x = x
        else:
            self.x = lambda s: np.full(np.shape(s), float(x))  # type: ignore [arg-type]

        if callable(r_yz):
            self.r_yz = r_yz
        else:
            self.r_yz = lambda s: np.full(np.shape(s), float(r_yz))  # type: ignore [arg-type]

        if callable(yz):
            self.yz = yz
        elif yz is None:
            self.yz = FlatYZ()
        else:
            raise ValueError("FIXME: need a good `yz` error message")

        # FIXME: validate `length(yz) == 2`?

        if callable(c):
            self.c = c
        else:
            self.c = lambda s: np.full(np.shape(s), float(c))  # type: ignore [arg-type]

        if callable(theta):
            self.theta = theta
        else:
            self.theta = lambda s: np.full(np.shape(s), float(theta))  # type: ignore [arg-type]

        # Set the origin to the central chord leading edge. A default value of
        # zeros must be set before calling `xyz` to find the real offset.
        self.LE0 = [0, 0, 0]
        if center:
            self.LE0 = self.xyz(0, 0)

        # TODO: this `b` calculation is a reasonable placeholder for average
        # foil shapes, but it assumes the arc is symmetric and that it is the
        # wing-tip section that defines the span.
        self.b: float = 2 * max(self.xyz(1, [0, 1]).T[1])
        self.b_flat: float = 2  # FIXME: poor design? I like making it explicit.

    @property
    def AR(self) -> float:
        """Compute the projected aspect ratio of the foil."""
        return self.b**2 / self.S

    @property
    def AR_flat(self) -> float:
        """Compute the flattened aspect ratio of the foil."""
        return self.b_flat**2 / self.S_flat

    @property
    def S(self) -> float:
        """
        Compute the projected area of the surface.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        s = np.linspace(-1, 1, 1000)
        LEx, LEy = self.xyz(s, 0)[:, :2].T
        TEx, TEy = self.xyz(s, 1)[:, :2].T

        # Three areas: curved front, trapezoidal mid, and curved rear
        LE_area = scipy.integrate.simps(LEx - LEx.min(), LEy)
        mid_area = (LEy.ptp() + TEy.ptp()) / 2 * (LEx.min() - TEx.max())
        TE_area = -scipy.integrate.simps(TEx - TEx.max(), TEy)
        return cast(float, LE_area + mid_area + TE_area)

    @property
    def S_flat(self) -> float:
        """
        Compute the projected area of the flattened surface.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        s = np.linspace(-1, 1, 1000)
        c = self.c(s)
        lengths = (self._section_pitch(s)[..., 0] * c[:, np.newaxis]).T[0]
        return cast(float, scipy.integrate.simps(lengths, s) * self.b_flat / 2)

    def _section_pitch(self, s):
        """
        Compute the section pitch DCM (rotations about the planform y-axis).
        This corresponds to "geometric torsion" in some texts.

        These angles are between the x-axis of a section and the x-axis of the
        central chord when the wing is flat.
        """
        thetas = self.theta(s)
        ct, st = np.cos(thetas), np.sin(thetas)
        _0, _1 = np.zeros(np.shape(s)), np.ones(np.shape(s))
        # fmt: off
        Theta = np.array(
            [[ ct, _0, st],  # noqa: E201
             [ _0, _1, _0],  # noqa: E201
             [-st, _0, ct]],
        )
        # fmt: on

        # Ensure the results are shaped like (*s.shape, 3, 3)
        Theta = np.moveaxis(Theta, [0, 1], [-2, -1])
        return Theta

    def _section_roll(self, s):
        """
        Compute the section roll DCM (rotations about the arc x-axis).
        This corresponds to "section dihedral" in some texts.

        This rotation refers to the angle between the y-axis of a section and
        the y-axis of the central chord of the arc.
        """
        derivatives = self.yz.derivative(s).T
        dyds, dzds = derivatives[0].T, derivatives[1].T
        K = np.sqrt(dyds**2 + dzds**2)  # L2-norm
        dyds /= K
        dzds /= K
        _0, _1 = np.zeros(np.shape(s)), np.ones(np.shape(s))
        # fmt: off
        Gamma = np.array(
            [[_1,   _0,    _0],  # noqa: E241
             [_0, dyds, -dzds],
             [_0, dzds, dyds]],
        )
        # fmt: on

        # Ensure the results are shaped like (*s.shape, 3, 3)
        Gamma = np.moveaxis(Gamma, [0, 1], [-2, -1])
        return Gamma

    def orientation(self, s, flatten: bool = False):
        """
        Compute section coordinate axes as rotation matrices.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index
        flatten : bool
            Whether to ignore dihedral. Default: False

        Returns
        -------
        array of float, shape (N,3)
            Rotation matrices encoding section orientation, where the columns
            are the section (local) x, y, and z coordinate axes.
        """
        C_c2s = self._section_pitch(s)
        if not flatten:
            C_c2s = self._section_roll(s) @ C_c2s
        return C_c2s

    def xyz(self, s, r, flatten: bool = False):
        """
        Compute the coordinates of points on section chords in canopy frd.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index
        r : float
            Position on the chords as a percentage, where `r = 0` is the
            leading edge, and `r = 1` is the trailing edge.
        flatten : boolean
            Whether to flatten the chord surface by disregarding dihedral
            (curvature in the yz-plane). This is useful for inflatable wings,
            such as parafoils. Default: False.
        """
        s = np.asfarray(s)
        r = np.asfarray(r)
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if r.min() < 0 or r.max() > 1:
            raise ValueError("Chord ratios must be between 0 and 1.")

        r_x = self.r_x(s)
        r_yz = self.r_yz(s)
        x = self.x(s)
        yz = self.yz(s)
        c = self.c(s)
        Theta = self._section_pitch(s)
        Gamma = self._section_roll(s)

        if flatten:
            # FIXME: using `s` for `y_flat` assumes the input values have been
            #        correctly normalized to `total_length(yz) == 2`
            r_RP2O = np.stack((x, s, np.zeros(s.shape)), axis=-1)
            xhat = Theta @ [1, 0, 0]
        else:
            r_RP2O = np.concatenate((x[..., np.newaxis], yz), axis=-1)
            xhat = Gamma @ Theta @ [1, 0, 0]

        R = np.stack([r_x, r_yz, r_yz], axis=-1)
        r_LE2RP = np.einsum("...i,...,...i->...i", R, c, xhat)
        r_P2LE = -(r * c)[..., np.newaxis] * xhat - self.LE0
        r_P2O = r_P2LE + r_LE2RP + r_RP2O

        return r_P2O
