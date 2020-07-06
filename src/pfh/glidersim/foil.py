"""FIXME: add docstring."""

import abc
import warnings

import numpy as np

import scipy.interpolate
import scipy.optimize
from scipy.spatial import Delaunay

from pfh.glidersim.util import cross3


class EllipticalArc:
    """
    An arc segment from an ellipse.

    Although this internal representation uses the standard `t` parameter for
    the parametric functions, some applications are more easily described with
    a different domain. For example, all the `FoilGeometry` curves are defined
    as functions of the section index `s`, which ranges from -1 (the left wing
    tip) to +1 (the right wing tip).

    Setting different domains for the input coordinates vs the internal
    coordinates lets users ignore the implementation and focus on the
    functional domain. This is somewhat analogous to the `domain` and `window`
    parameters in a numpy `Polynomial`, although in that case the conversion
    is for numerical improvements, whereas this is for user convenience.

    This class supports both "implicit" and "parametric" representations of an
    ellipse. The implicit version returns the vertical component as a function
    of the horizontal coordinate, and the parametric version returns both the
    horizontal and vertical coordinates as a function of the parametric
    parameter.

    The parametric curve in this class is always parametrized by `t`. By
    setting `t_domain` you can constrain the curve to an elliptical arc.

    For both the implicit and parametric forms, you can change the input domain
    by setting `p_domain`, and they will be scaled automatically to map onto the
    implicit (`x`) or parametric (`t`) parameter.
    """

    def __init__(
        self,
        AB_ratio,
        A=None,
        B=None,
        length=None,
        origin=None,
        p_domain=None,
        t_domain=None,
        kind="parametric",
    ):
        """
        Construct an ellipse segment.

        Parameters
        ----------
        AB_ratio : float
            The ratio of the major vs minor axes.
        A, B, length : float, optional
            Scaling constraints. Choose only one.
        origin : array of float, shape (2,), optional
            The (x, y) offset of the ellipse origin. (For example, shifting the
            curve up or down to produce a constant offset.)
        p_domain : array of float, shape (2,), optional
            The domain of the input parameter. This encodes "what values in the
            input domain maps to `t_domain`?". For example, if you wanted to a
            parametric function where -1 and +1 are the start and end of the
            curve then `p_domain = [1, -1]` would map `t_min` to `p = 1` and
            `t_max` to `p = -1`.
        t_domain : array of float, shape (2,), optional
            The domain of the internal parameter. Values from 0 to pi/2 are the
            first quadrant, pi/2 to pi are the second, etc.
        kind : {'implicit', 'parametric'}, default: 'parametric'
            Specifies whether the class return `y = f(x)` or `<x, y> = f(p)`.
            The implicit version returns coordinates of the second axis given
            coordinates of the first axis. The parametric version returns both
            x and y given a parametric coordinate.
        """
        if sum(arg is not None for arg in [A, B, length]) > 1:
            raise ValueError("Specify only one of width, length, or height")

        if not origin:
            origin = [0, 0]

        if not t_domain:
            t_domain = [0, np.pi]  # Default to a complete half-ellipse

        # Initial values (before constraints)
        self.A = 1
        self.B = 1 / AB_ratio
        self.origin = origin
        self.p_domain = p_domain
        self.t_domain = t_domain
        self.kind = kind

        # Apply constraints, if any
        if A:
            self.A, self.B = A, A / AB_ratio
        elif B:
            self.A, self.B = B * AB_ratio, B
        elif length:
            L = self.length  # The length before rescaling A and B
            K = length / L
            self.A *= K
            self.B *= K

    def __call__(self, p, kind=None):
        if kind is None:
            kind = self.kind

        p = np.asarray(p)
        t = self._transform(p, kind)

        if kind == "implicit":  # the `p` are `x` in the external domain
            vals = self.B * np.sin(t)  # Return `y = f(x)`
        else:  # the `p` are `t` in the external domain
            x = self.A * np.cos(t)
            y = self.B * np.sin(t)
            vals = np.stack((x, y), axis=-1) + self.origin  # return `<x, y> = f(t)`

        return vals

    @property
    def p_domain(self):
        if self._p_domain is not None:
            return self._p_domain
        else:
            return self.t_domain  # No transform

        # FIXME: if p_domain is None, then `p` might be `x` **or** `t`

    @p_domain.setter
    def p_domain(self, new_p_domain):
        if new_p_domain:
            new_p_domain = np.asarray(new_p_domain, dtype=float)
            if new_p_domain.shape != (2,):
                raise ValueError("`p_domain` must be an array_like of shape (2,)")
        self._p_domain = new_p_domain

    @property
    def t_domain(self):
        return self._t_domain

    @t_domain.setter
    def t_domain(self, new_t_domain):
        new_t_domain = np.asarray(new_t_domain, dtype=float)
        if new_t_domain.shape != (2,):
            raise ValueError("`t_domain` must be an array_like of shape (2,)")

        if (new_t_domain.max() > 2 * np.pi) or (new_t_domain.min() < 0):
            raise ValueError("`t_domain` values must be between 0 and 2pi")

        self._t_domain = new_t_domain

    def _transform(self, p, kind):
        """
        Map the external domain onto `t` using an affine transformation.

        Parameters
        ----------
        p : array_like of float
            Parametric coordinates being used by the external application.
            Values fall inside to [p_0, p_1].

        Returns
        -------
        t : array_like of float
            The value of the internal curve parameter `t` that maps to `p`.

        """

        if kind == "implicit":
            # The implicit `y = f(x)` can be converted to `y = f(t)` by first
            # changing the external domain to the internal `x` coordinates
            p0, p1 = self.p_domain
            x0, x1 = self.A * np.cos(self.t_domain)  # The internal domain of `x`
            x = (x1 - x0) / (p1 - p0) * p
            t = np.arccos(x / self.A)

        else:
            p0, p1 = self.p_domain
            t0, t1 = self.t_domain
            a = (t0 - t1) / (p0 - p1)
            b = t1 - a * p1
            t = a * p + b

        return t

    @property
    def length(self):
        p = np.linspace(*self.p_domain, 10000)
        xy = self.__call__(p, kind="parametric")
        return np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()

    def derivative(self, p, kind=None):
        """
        Compute the derivatives of the parametric components.

        For now, I'm never using `dy/dx` directly, so this always returns both
        derivatives separately to avoid divide-by-zero issues.
        """
        if kind is None and self.kind == "implicit":
            raise RuntimeError("Are you SURE you really want dy/dx?")

        p = np.asarray(p)
        t = self._transform(p, "parametric")
        dxdt = self.A * -np.sin(t)
        dydt = self.B * np.cos(t)
        vals = np.stack((dxdt, dydt), axis=-1)
        return vals


def elliptical_chord(root, tip):
    """
    Build an elliptical chord distribution as a function of the section index.

    Parameters
    ----------
    root : float [length]
        The length of the central chord
    tip : float [length]
        The length of the wing tips

    Returns
    -------
    EllipticalArc
        A function `chord_length(s)` where `-1 <= s <= 1`, (suitable for use
        with `ChordSurface`).
    """

    taper = tip / root
    A = 1 / np.sqrt(1 - taper ** 2)
    B = root
    t_min = np.arcsin(taper)

    return EllipticalArc(
        A / B, B=B, p_domain=[1, -1], t_domain=[t_min, np.pi - t_min], kind="implicit",
    )


def elliptical_arc(mean_anhedral, tip_anhedral=None):
    """
    Build an elliptical arc curve as a function of the section index.

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

    Returns
    -------
    EllipticalArc
        A parametric function `<y(s), z(s)>` where `-1 <= s <= 1`, with total
        arc length `2` (suitable for use with `ChordSurface`).
    """
    if tip_anhedral is None:  # Assume circular
        tip_anhedral = 2 * mean_anhedral

    if mean_anhedral < 0 or mean_anhedral > 45:
        raise ValueError("mean_anhedral must be between 0 and 45 [degrees]")
    if tip_anhedral < 0 or tip_anhedral > 90:
        raise ValueError("tip_anhedral must be between 0 and 90 [degrees]")
    if tip_anhedral < 2 * mean_anhedral:
        raise ValueError("tip_anhedral must be >= 2 * mean_anhedral")

    # Very small angles produce divide-by-zero, just just assume the user wants
    # a zero-angle and "do the right thing".
    if mean_anhedral < 0.1:
        warnings.warn("Very small mean_anhedral. Returning a FlatYZ.")
        return FlatYZ()

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

    # FIXME: hack to avoid sign-flip issues at `np.sin(pi)`, which makes the
    #        normalized dyds explode (sign flip +1 to -1)
    if t_min < 1e-10:
        t_min = 1e-10

    arc = EllipticalArc(
        A / B, length=2, p_domain=[-1, 1], t_domain=[np.pi + t_min, 2 * np.pi - t_min],
    )
    arc.origin = -arc(0)  # The middle of the curve is the origin
    return arc


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

    def __init__(self, start, exponent, peak):
        self.start = start
        self.exponent = exponent
        self.peak = np.deg2rad(peak)

    def __call__(self, s):
        # FIXME: design review? Not a fan of these asarray
        s = np.asarray(s)
        values = np.zeros(s.shape)
        m = abs(s) > self.start  # Mask
        if np.any(m):
            p = (abs(s[m]) - self.start) / (1 - self.start)
            # p /= p[-1]  # `p` goes from 0 to 1
            values[m] = self.peak * p ** self.exponent
        return values


class SimpleIntakes:
    """
    Defines the upper and lower surface coordinates as constant along the span.

    This version currently uses explicit `sa_upper` and `sa_lower` in airfoil
    coordinates, but other parametrizations might be the intake midpoint and
    width (where "width" might be in the airfoil `s`, or as a percentage of the
    chord) or `c_upper` and `c_lower` as points on the chord.

    Parameters
    ----------
    s_end: float
        Section index. Air intakes are present between +/- `s_end`.
    sa_upper, sa_lower : float
        The starting coordinates of the upper and lower surface of the
        parafoil, given in airfoil surface coordinates. These are used to
        define air intakes, and for determining the inertial properties of the
        upper and lower surfaces.

        The airfoil coordinates use `s = 0` for the leading edge, `s = 1` for
        trailing edge of the curve above the chord, and `s = -1` for the
        trailing edge of the curve below the chord, so these choices must
        follow `-1 <= sa_lower <= sa_upper <= 1`.
    """

    def __init__(self, s_end, sa_upper, sa_lower):
        # FIXME: support more types of definition:
        #  1. su/sl : explicit upper/lower cuts in airfoil coordinates
        #  2. midpoint (in airfoil coordinates) and width
        #  3. Upper and lower cuts as a fraction of the chord (the "Paraglider
        #     Design Manual" does it this way).
        self.s_end = s_end
        self.sa_upper = sa_upper
        self.sa_lower = sa_lower

    def __call__(self, s, sa, surface):
        """
        Convert parafoil surface coordinates into airfoil coordinates.

        Parameters
        ----------
        s : array_like of float
            Section index.
        sa : array_like of float
            Parafoil surface coordinate, where `0 <= sa <= 1`, with `0`
            being the leading edge, and `1` being the trailing edge.
        surface : {"upper", "lower"}
            Which surface.

        Returns
        -------
        array_like of float, shape (N,)
            The normalized (unscaled) airfoil coordinates.
        """
        s = np.asarray(s)
        sa = np.asarray(sa)

        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if sa.min() < 0 or sa.max() > 1:
            raise ValueError("Surface coordinates must be between 0 and 1.")
        if surface not in {"upper", "lower"}:
            raise ValueError("`surface` must be one of {'upper', 'lower'}")

        if surface == "upper":
            values = self.sa_upper + sa * (1 - self.sa_upper)
            values = np.broadcast_arrays(s, values)[1]
        else:
            # The lower surface extends forward on sections without intakes
            starts = np.where(np.abs(s) < self.s_end, self.sa_lower, self.sa_upper)
            values = starts + sa * (-1 - starts)

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


# ---------------------------------------------------------------------------

class ChordSurface:
    """
    FIXME: docstring. Describe the geometry.

    All input values must be normalized by `b_flat = 2`. Output values can be
    scaled as needed to achieve a given `b`, `b_flat`, `S`, or `S_flat`.

    Conceptually, this surface is a design target: an idealized goal to produce
    with a physical foil. Rigid foils can create this surface exactly, but
    flexible wings, like parafoils, can only approximate this shape through
    the internal structure of cells.

    Parameters
    ----------
    r_x : float or callable
        A ratio from 0 to 1 that defines what location on each chord is located
        at the x-coordinate defined by `x`. This can be a constant or a
        function of the section index. For example, `xy_r = 1` says that `x` is
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
        This curve shapes the yz-plane view of the inflated wing.
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
        r_x,
        x,
        r_yz,
        yz,
        c,
        theta=0,
        center=True,
    ):
        if callable(r_x):
            self.r_x = r_x
        else:
            self.r_x = lambda s: np.full(np.shape(s), float(r_x))

        if callable(x):
            self.x = x
        else:
            self.x = lambda s: np.full(np.shape(s), float(x))

        if callable(r_yz):
            self.r_yz = r_yz
        else:
            self.r_yz = lambda s: np.full(np.shape(s), float(r_yz))

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
            self.c = lambda s: np.full(np.shape(s), float(c))

        if callable(theta):
            self.theta = theta
        else:
            self.theta = lambda s: np.full(np.shape(s), float(theta))

        # Set the origin to the central chord leading edge. A default value of
        # zeros must be set before calling `xyz` to find the real offset.
        self.LE0 = [0, 0, 0]
        if center:
            self.LE0 = self.xyz(0, 0)

        # TODO: this `b` calculation is a reasonable placeholder for average
        # foil shapes, but it assumes the arc is symmetric and that it is the
        # wing-tip section that defines the span.
        self.b = 2 * max(self.xyz(1, [0, 1]).T[1])
        self.b_flat = 2  # FIXME: poor design? I like making it explicit.

    @property
    def AR(self):
        """Compute the projected aspect ratio of the foil."""
        return self.b ** 2 / self.S

    @property
    def AR_flat(self):
        """Compute the flattened aspect ratio of the foil."""
        return self.b_flat ** 2 / self.S_flat

    @property
    def S(self):
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
        return LE_area + mid_area + TE_area

    @property
    def S_flat(self):
        """
        Compute the projected area of the flattened surface.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        s = np.linspace(-1, 1, 1000)
        c = self.length(s)  # Projected section lengths before torsion
        lengths = (self._planform_torsion(s)[..., 0] * c[:, np.newaxis]).T[0]
        return scipy.integrate.simps(lengths, s) * self.b_flat / 2

    def _planform_torsion(self, s):
        """
        Compute the planform torsion (rotations about the planform y-axis).

        These angles are between the x-axis of a section and the x-axis of the
        central chord when the wing is flat.
        """
        thetas = self.theta(s)
        ct, st = np.cos(thetas), np.sin(thetas)
        _0, _1 = np.zeros(np.shape(s)), np.ones(np.shape(s))
        # fmt: off
        torsion = np.array(
            [[ ct, _0, st],  # noqa: E201
             [ _0, _1, _0],  # noqa: E201
             [-st, _0, ct]],
        )
        # fmt: on

        # Ensure the results are shaped like (*s.shape, 3, 3)
        torsion = np.moveaxis(torsion, [0, 1], [-2, -1])
        return torsion

    def _arc_dihedral(self, s):
        """
        Compute the arc dihedral (rotations about the arc x-axis).

        This rotation refers to the angle between the y-axis of a section and
        the y-axis of the central chord of the arc.
        """
        derivatives = self.yz.derivative(s).T
        dyds, dzds = derivatives[0].T, derivatives[1].T
        K = np.sqrt(dyds ** 2 + dzds ** 2)  # L2-norm
        dyds /= K
        dzds /= K
        _0, _1 = np.zeros(np.shape(s)), np.ones(np.shape(s))
        # fmt: off
        dihedral = np.array(
            [[_1,   _0,    _0],  # noqa: E241
             [_0, dyds, -dzds],
             [_0, dzds, dyds]],
        )
        # fmt: on

        # Ensure the results are shaped like (*s.shape, 3, 3)
        dihedral = np.moveaxis(dihedral, [0, 1], [-2, -1])
        return dihedral

    def orientation(self, s, flatten=False):
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
        R = self._planform_torsion(s)
        if not flatten:
            R = self._arc_dihedral(s) @ R
        return R

    def length(self, s):
        """
        Compute section chord lengths.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index

        Returns
        -------
        array_like of float, shape (N,)
            The length of the section chord.
        """
        return self.c(s)

    def xyz(self, s, pc, flatten=False):
        """
        Compute the `xyz` coordinates of points on section chords.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index
        pc : float
            Position on the chords as a percentage, where `pc = 0` is the
            leading edge, and `pc = 1` is the trailing edge.
        flatten : boolean
            Whether to flatten the chord surface by disregarding dihedral
            (curvature in the yz-plane). This is useful for inflatable wings,
            such as parafoils. Default: False.
        """
        s = np.asarray(s)
        pc = np.asarray(pc)
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if pc.min() < 0 or pc.max() > 1:
            raise ValueError("Chord ratios must be between 0 and 1.")

        # FIXME? Written this way for clarity, but some terms are unused.
        r_x = self.r_x(s)
        r_yz = self.r_yz(s)
        x = self.x(s)
        c = self.c(s)
        torsion = self._planform_torsion(s)
        dihedral = self._arc_dihedral(s)
        xhat_planform = torsion @ [1, 0, 0]
        xhat_wing = dihedral @ torsion @ [1, 0, 0]

        # Ugly, but supports all broadcastable shapes for `s` and `pc`
        if flatten:  # Disregard dihedral (curvature in the yz-plane)
            # FIXME: using `s` for `y` assumes the input values have been
            #        correctly normalized to `b_flat == 2`, which requires
            #        the `total_length(yz) == 2`
            LE = (np.stack((x, s, np.zeros(s.shape)), axis=-1)
                  + ((r_x * c)[..., np.newaxis] * xhat_planform))
            xyz = LE - (pc * c)[..., np.newaxis] * xhat_planform - self.LE0
        else:
            # First, assume the x(s) and yz(s) are the leading edge points
            LE = np.concatenate((x[..., np.newaxis], self.yz(s)), axis=-1)

            # Shift the leading edges forward along the section (local) x-axes
            # until the y- and z-coordinates of the r_yz(s) reference points
            # lie on the yz(s) curve.
            LE += (r_yz * c)[..., np.newaxis] * xhat_wing

            # Residual adjustment so the r_x(s) reference points lie on x(s).
            LE[..., 0] += (r_x - r_yz) * c * xhat_wing[..., 0]

            # Compute the desired points on the chords and center them
            xyz = LE - (pc * c)[..., np.newaxis] * xhat_wing - self.LE0

        return xyz


class FoilSections:
    """
    Provides the section profile geometry and coefficients.

    This simple implementation only takes a single airfoil; it does not support
    spanwise interpolation of section profiles.

    Parameters
    ----------
    airfoil : Airfoil
        The airfoil that defines all section profiles.
    intakes : function, optional
        A function that defines the upper and lower intake positions in
        airfoil surface coordinates as a function of the section index.
    """

    def __init__(
        self,
        airfoil,
        intakes=None,
    ):
        self.airfoil = airfoil
        self.intakes = intakes if intakes else self._no_intakes

    def _no_intakes(self, s, sa, surface):
        # For foils with no air intakes the canopy upper and lower surfaces map
        # directly to the airfoil upper and lower surfaces, which were defined
        # by the airfoil leading edge.
        if surface == "lower":
            sa = -sa
        return np.broadcast_arrays(s, sa)[1]

    def surface_xz(self, s, sa, surface):
        """
        Compute unscaled surface coordinates along section profiles.

        These are unscaled since the FoilSections only defines the normalized
        airfoil geometry and coefficients. The Foil scales, translates, and
        orients these with the chord data it gets from the ChordSurface.

        Parameters
        ----------
        s : array_like of float
            Section index.
        sa : array_like of float
            Surface or airfoil coordinates, depending on the value of `surface`.
        surface : {"upper", "lower", "airfoil"}
            How to interpret the coordinates in `sa`. If "upper" or "lower",
            then `sa` is treated as surface coordinates, which range from 0 to
            1, and specify points on the upper or lower surfaces, as defined by
            the intakes. If "airfoil", then `sa` is treated as raw airfoil
            coordinates, which must range from -1 to +1, and map from the
            lower surface trailing edge to the upper surface trailing edge.

        Returns
        -------
        array of float
            A set of points from the surface of the airfoil in foil frd. The
            shape is determined by standard numpy broadcasting of `s` and `sa`.
        """
        s = np.asarray(s)
        sa = np.asarray(sa)
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if surface not in {"upper", "lower", "airfoil"}:
            raise ValueError("`surface` must be one of {'upper', 'lower', 'airfoil'}")
        if surface == "airfoil" and (sa.min() < -1 or sa.max() > 1):
            raise ValueError("Airfoil coordinates must be between -1 and 1.")
        elif surface != "airfoil" and (sa.min() < 0 or sa.max() > 1):
            raise ValueError("Surface coordinates must be between 0 and 1.")

        if surface == "airfoil":
            sa = np.broadcast_arrays(s, sa)[1]
        else:
            sa = self.intakes(s, sa, surface)

        return self.airfoil.geometry.surface_curve(sa)  # Unscaled airfoil

    def Cl(self, s, delta_f, alpha, Re):
        """
        Compute the lift coefficient of the airfoil.

        Parameters
        ----------
        s : array_like of float
            Section index.
        delta_f : array_like of float [radians]
            Deflection angle of the trailing edge due to control inputs, as
            measured between the deflected edge and the undeflected chord.
        alpha : array_like of float [radians]
            Angle of attack
        Re : float [unitless]
            Reynolds number

        Returns
        -------
        Cl : float
        """
        return self.airfoil.coefficients.Cl(delta_f, alpha, Re)

    def Cl_alpha(self, s, delta_f, alpha, Re):
        """
        Compute the derivative of the lift coefficient versus angle of attack.

        Parameters
        ----------
        s : array_like of float
            Section index.
        delta_f : array_like of float [radians]
            Deflection angle of the trailing edge due to control inputs, as
            measured between the deflected edge and the undeflected chord.
        alpha : array_like of float [radians]
            Angle of attack
        Re : float [unitless]
            Reynolds number

        Returns
        -------
        Cl_alpha : float
        """
        return self.airfoil.coefficients.Cl_alpha(delta_f, alpha, Re)

    def Cd(self, s, delta_f, alpha, Re):
        """
        Compute the drag coefficient of the airfoil.

        Parameters
        ----------
        s : array_like of float
            Section index.
        delta_f : array_like of float [radians]
            Deflection angle of the trailing edge due to control inputs, as
            measured between the deflected edge and the undeflected chord.
        alpha : array_like of float [radians]
            Angle of attack
        Re : float [unitless]
            Reynolds number

        Returns
        -------
        Cd : float
        """
        Cd = self.airfoil.coefficients.Cd(delta_f, alpha, Re)

        # Additional drag from the air intakes
        #
        # Ref: `babinsky1999AerodynamicPerformanceParagliders`
        la = np.linalg.norm(  # Length of the air intake
            self.surface_xz(s, 0, 'upper')
            - self.surface_xz(s, 0, 'lower'),
            axis=-1,
        )
        Cd += 0.07 * la  # Drag due to air intakes

        # Additional drag from "surface characteristics"
        #
        # Ref: `ware1969WindtunnelInvestigationRamair`
        Cd += 0.004

        return Cd

    def Cm(self, s, delta_f, alpha, Re):
        """
        Compute the pitching coefficient of the airfoil.

        Parameters
        ----------
        delta_f : float [radians]
            The deflection angle of the trailing edge due to control inputs,
            as measured between the deflected edge and the undeflected chord.
        alpha : float [radians]
            The angle of attack
        Re : float [unitless]
            The Reynolds number

        Returns
        -------
        Cm : float
        """
        return self.airfoil.coefficients.Cm(delta_f, alpha, Re)

    def thickness(self, s, pc):
        """
        Compute section thickness.

        These are the normalized thicknesses, not the absolute values. The
        absolute thickness requires knowledge of the chord length.

        Parameters
        ----------
        s : array_like of float
            Section index.
        pc : array_like of float [percentage]
            Fractional position on the camber line, where `0 <= pc <= 1`

        Returns
        -------
        thickness : array_like of float
            The normalized section profile thicknesses.
        """
        return self.airfoil.geometry.thickness(pc)

    def _mass_properties(self):
        """Pass-through to the airfoil `mass_properties` function."""
        # FIXME: I hate this. Also, should be a function of `s`
        return self.airfoil.geometry._mass_properties()


class SimpleFoil:
    """
    A foil geometry that applies a constant airfoil along a chord surface.

    These are idealized foils that exactly match the scaled chord surface and
    profiles. Such an idealized shape is only possible for rigid foils that can
    enforce absolute geometry, unlike flexible foils, which can only attempt to
    create the target shape through internal structure.
    """

    def __init__(
        self,
        chords,
        sections,
        b=None,
        b_flat=None,
    ):
        """
        Add a docstring.

        Parameters
        ----------
        chords : ChordSurface
            FIXME: docstring
        sections : FoilSections
            The geometry and coefficients for the section profiles.
        b, b_flat : float
            The arched and flattened spans of the chords. Specify only one.
            These function as scaling factors for the ChordSurface.
        """
        self._chords = chords
        self.sections = sections

        if b is not None and b_flat is not None:
            raise ValueError("Specify only one of `b` or `b_flat`")

        # FIXME: support `S` and `S_flat` as scaling factors
        if b:
            self.b = b
        else:  # b_flat
            self.b_flat = b_flat

    @property
    def b(self):
        """The projected span of the foil."""
        return self._b

    @b.setter
    def b(self, new_b):
        self._b = new_b
        self._b_flat = new_b * self._chords.b_flat / self._chords.b

    @property
    def b_flat(self):
        """The projected span of the foil with section dihedral removed."""
        return self._b_flat

    @b_flat.setter
    def b_flat(self, new_b_flat):
        self._b_flat = new_b_flat
        self._b = new_b_flat * self._chords.b / self._chords.b_flat

    @property
    def AR(self):
        """The aspect ratio of the foil."""
        return self.b ** 2 / self.S

    @property
    def AR_flat(self):
        """The aspect ratio of the foil with section dihedral removed."""
        return self.b_flat ** 2 / self.S_flat

    @property
    def S(self):
        """
        The projected area of the surface.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        return self._chords.S * (self.b_flat / 2) ** 2

    @property
    def S_flat(self):
        """
        The projected area of the surface with section dihedral removed.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        return self._chords.S_flat * (self.b_flat / 2) ** 2

    def chord_length(self, s):
        """
        Compute chord lengths.

        Parameters
        ----------
        s : array_like of float
            Section index

        Returns
        -------
        array_like of float
            The section chord lengths.
        """
        return self._chords.length(s) * (self.b_flat / 2)

    def chord_xyz(self, s, pc, flatten=False):
        """
        Sample points on section chords in foil frd.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index
        pc : float
            Position on the chords as a percentage, where `pc = 0` is the
            leading edge, and `pc = 1` is the trailing edge.
        flatten : boolean
            Whether to flatten the chord surface by disregarding dihedral
            (curvature in the yz-plane). This is useful for inflatable wings,
            such as parafoils. Default: False.
        """
        return self._chords.xyz(s, pc, flatten=flatten) * (self.b_flat / 2)

    def section_orientation(self, s, flatten=False):
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
        return self._chords.orientation(s, flatten)

    def section_thickness(self, s, pc):
        """
        Compute section thicknesses at chordwise stations.

        Note that the thickness is determined by the airfoil convention, so
        this value may be measured perpendicular to either the chord line or
        to the camber line.

        Parameters
        ----------
        s : array_like of float
            Section index.
        pc : float
            Position on the chords as a percentage, where `pc = 0` is the
            leading edge, and `pc = 1` is the trailing edge.
        """
        # FIXME: does `pc` specify stations along the chord or the camber?
        return self.sections.thickness(s, pc) * self.chord_length(s)

    def surface_xyz(self, s, sa, surface, flatten=False):
        """
        Sample points on section surfaces in foil frd.

        Parameters
        ----------
        s : array_like of float
            Section index.
        sa : array_like of float
            Surface or airfoil coordinates, depending on the value of `surface`.
        surface : {"upper", "lower", "airfoil"}
            How to interpret the coordinates in `sa`. If "upper" or "lower",
            then `sa` is treated as surface coordinates, which range from 0 to
            1, and specify points on the upper or lower surfaces, as defined by
            the intakes. If "airfoil", then `sa` is treated as raw airfoil
            coordinates, which must range from -1 to +1, and map from the
            lower surface trailing edge to the upper surface trailing edge.
        flatten : boolean
            Whether to flatten the foil by disregarding dihedral (curvature in
            the yz-plane). This is useful for inflatable wings, such as
            parafoils. Default: False.

        Returns
        -------
        array of float
            A set of points from the surface of the airfoil in foil frd. The
            shape is determined by standard numpy broadcasting of `s` and `sa`.
        """
        s = np.asarray(s)
        sa = np.asarray(sa)
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")

        LE = self.chord_xyz(s, 0, flatten=flatten)
        c = self.chord_length(s)
        coords_a = self.sections.surface_xz(s, sa, surface)  # Unscaled airfoil
        coords = np.stack(
            (-coords_a[..., 0], np.zeros(coords_a.shape[:-1]), -coords_a[..., 1]),
            axis=-1,
        )
        orientations = self.section_orientation(s, flatten)
        surface = np.einsum("...ij,...j,...->...i", orientations, coords, c)
        return LE + surface

    def mass_properties(self, N=250):
        """
        Compute the quantities that control inertial behavior.

        (This method is deprecated by the new mesh-based method, and is only
        used for sanity checks.)

        The inertia matrices returned by this function are proportional to the
        values for a physical wing, and do not have standard units. They must
        be scaled by the wing materials and air density to get their physical
        values. See "Notes" for a thorough description.

        Returns
        -------
        dictionary
            upper_area: float [m^2]
                foil upper surface area
            upper_centroid: ndarray of float, shape (3,) [m]
                center of mass of the upper surface material in foil frd
            upper_inertia: ndarray of float, shape (3, 3) [m^4]
                The inertia matrix of the upper surface
            volume: float [m^3]
                internal volume of the inflated foil
            volume_centroid: ndarray of float, shape (3,) [m]
                centroid of the internal air mass in foil frd
            volume_inertia: ndarray of float, shape (3, 3) [m^5]
                The inertia matrix of the internal volume
            lower_area: float [m^2]
                foil lower surface area
            lower_centroid: ndarray of float, shape (3,) [m]
                center of mass of the lower surface material in foil frd
            lower_inertia: ndarray of float, shape (3, 3) [m^4]
                The inertia matrix of the upper surface

        Notes
        -----
        The foil is treated as a composite of three components: the upper
        surface, internal volume, and lower surface. Because this class only
        defines the geometry of the foil, not the physical properties, each
        component is treated as having unit densities, and the results are
        proportional to the values for a physical wing. To compute the values
        for a physical wing, the upper and lower surface inertia matrices must
        be scaled by the aerial densities [kg/m^2] of the upper and lower wing
        surface materials, and the volumetric inertia matrix must be scaled by
        the volumetric density [kg/m^3] of air.

        Keeping these components separate allows a user to simply multiply them
        by different wing material densities and air densities to compute the
        values for the physical wing.

        The calculation works by breaking the foil into N segments, where
        each segment is assumed to have a constant airfoil and chord length.
        The airfoil for each segment is extruded along the segment span using
        the perpendicular axis theorem, then oriented into body coordinates,
        and finally translated to the global centroid (of the surface or
        volume) using the parallel axis theorem.

        Limitations:

        * Assumes a constant airfoil over the entire foil.

        * Assumes the upper and lower surfaces of the foil are the same as the
          upper and lower surfaces of the airfoil. (Ignores air intakes.)

        * Places all the segment mass on the section bisecting the center of
          the segment instead of spreading the mass out along the segment span,
          so it underestimates `I_xx` and `I_zz` by a factor of `\int{y^2 dm}`.
          Doesn't make a big difference in practice, but still: it's wrong.

        * Requires the AirfoilGeometry to compute its own `mass_properties`,
          which is an extra layer of mess I'd like to eliminate.
        """
        s_nodes = np.cos(np.linspace(np.pi, 0, N + 1))
        s_mid_nodes = (s_nodes[1:] + s_nodes[:-1]) / 2  # Segment midpoints
        nodes = self.chord_xyz(s_nodes, 0.25)  # Segment endpoints
        section = self.sections._mass_properties()
        node_chords = self.chord_length(s_nodes)
        chords = (node_chords[1:] + node_chords[:-1]) / 2  # Mean average
        T = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])  # acs -> frd
        u = self.section_orientation(s_mid_nodes)
        u_inv = np.linalg.inv(u)

        # Segment centroids
        airfoil_centroids = np.array([
            [*section["upper_centroid"], 0],
            [*section["area_centroid"], 0],
            [*section["lower_centroid"], 0]])
        segment_origins = self.chord_xyz(s_mid_nodes, 0)
        segment_upper_cm, segment_volume_cm, segment_lower_cm = (
            np.einsum("K,Kij,jk,Gk->GKi", chords, u, T, airfoil_centroids)
            + segment_origins[None, ...]
        )

        # Scaling factors for converting 2D airfoils into 3D segments.
        # Approximates each segments' `chord * span` area as parallelograms.
        u_a = u[..., 0]  # The chordwise ("aerodynamic") unit vectors
        dl = nodes[1:] - nodes[:-1]
        segment_chord_area = np.linalg.norm(cross3(u_a, dl), axis=1)
        Kl = chords * segment_chord_area  # section curve length into segment area
        Ka = chords ** 2 * segment_chord_area  # section area into segment volume

        segment_upper_area = Kl * section["upper_length"]
        segment_volume = Ka * section["area"]
        segment_lower_area = Kl * section["lower_length"]

        # Total surface areas and the internal volume
        upper_area = segment_upper_area.sum()
        volume = segment_volume.sum()
        lower_area = segment_lower_area.sum()

        # The upper/volume/lower centroids for the entire foil
        upper_centroid = (segment_upper_area * segment_upper_cm.T).T.sum(axis=0) / upper_area
        volume_centroid = (segment_volume * segment_volume_cm.T).T.sum(axis=0) / volume
        lower_centroid = (segment_lower_area * segment_lower_cm.T).T.sum(axis=0) / lower_area

        # Segment inertia matrices in body frd coordinates
        Kl, Ka = Kl.reshape(-1, 1, 1), Ka.reshape(-1, 1, 1)
        segment_upper_J = u_inv @ T @ (Kl * section["upper_inertia"]) @ T @ u
        segment_volume_J = u_inv @ T @ (Ka * section["area_inertia"]) @ T @ u
        segment_lower_J = u_inv @ T @ (Kl * section["lower_inertia"]) @ T @ u

        # Parallel axis distances of each segment
        Ru = upper_centroid - segment_upper_cm
        Rv = volume_centroid - segment_volume_cm
        Rl = lower_centroid - segment_lower_cm

        # Segment distances to the group centroids
        R = np.array([Ru, Rv, Rl])
        D = (np.einsum("Rij,Rij->Ri", R, R)[..., None, None] * np.eye(3)
             - np.einsum("Rki,Rkj->Rkij", R, R))
        Du, Dv, Dl = D

        # And finally, apply the parallel axis theorem
        upper_J = (segment_upper_J + (segment_upper_area * Du.T).T).sum(axis=0)
        volume_J = (segment_volume_J + (segment_volume * Dv.T).T).sum(axis=0)
        lower_J = (segment_lower_J + (segment_lower_area * Dl.T).T).sum(axis=0)

        mass_properties = {
            "upper_area": upper_area,
            "upper_centroid": upper_centroid,
            "upper_inertia": upper_J,
            "volume": volume,
            "volume_centroid": volume_centroid,
            "volume_inertia": volume_J,
            "lower_area": lower_area,
            "lower_centroid": lower_centroid,
            "lower_inertia": lower_J
        }

        return mass_properties

    def mass_properties2(self, N_s=301, N_sa=301):
        """
        Compute the quantities that control inertial behavior.

        The inertia matrices returned by this function are proportional to the
        values for a physical wing, and do not have standard units. They must
        be scaled by the wing materials and air density to get their physical
        values. See "Notes" for a thorough description.

        Parameters
        ----------
        N_s : int
            The grid resolution over the section index.
        N_sa : int
            The grid resolution over the surface coordinates.

        Returns
        -------
        dictionary
            upper_area: float [m^2]
                foil upper surface area
            upper_centroid: ndarray of float, shape (3,) [m]
                center of mass of the upper surface material in foil frd
            upper_inertia: ndarray of float, shape (3, 3) [m^4]
                The inertia matrix of the upper surface
            volume: float [m^3]
                internal volume of the inflated foil
            volume_centroid: ndarray of float, shape (3,) [m]
                centroid of the internal air mass in foil frd
            volume_inertia: ndarray of float, shape (3, 3) [m^5]
                The inertia matrix of the internal volume
            lower_area: float [m^2]
                foil lower surface area
            lower_centroid: ndarray of float, shape (3,) [m]
                center of mass of the lower surface material in foil frd
            lower_inertia: ndarray of float, shape (3, 3) [m^4]
                The inertia matrix of the upper surface

        Notes
        -----
        The foil is treated as a composite of three components: the upper
        surface, internal volume, and lower surface. Because this class only
        defines the geometry of the foil, not the physical properties, each
        component is treated as having unit densities, and the results are
        proportional to the values for a physical wing. To compute the values
        for a physical wing, the upper and lower surface inertia matrices must
        be scaled by the aerial densities [kg/m^2] of the upper and lower wing
        surface materials, and the volumetric inertia matrix must be scaled by
        the volumetric density [kg/m^3] of air.

        Keeping these components separate allows a user to simply multiply them
        by different wing material densities and air densities to compute the
        values for the physical wing.

        The volume calculation requires a closed surface mesh, so it ignores
        air intakes and assumes a closed trailing edge (which is fine for
        inflatable foils like a paraglider). The concept of using a summation
        of signed tetrahedron volumes is developed in [1]. This implementation
        of that idea is from [2], which first computes the inertia tensor of a
        "canonical tetrahedron" then applies an affine transformation to
        compute the inertia tensors of each individual tetrahedron.

        References
        ----------
        1. Efficient feature extraction for 2D/3D objects in mesh
           representation, Zhang and Chen, 2001(?)

        2. How to find the inertia tensor (or other mass properties) of a 3D
           solid body represented by a triangle mesh, Jonathan Blow, 2004.
           http://number-none.com/blow/inertia/index.html
        """
        # Note to self: the triangles are not symmetric about the xz-plane,
        # which produces non-zero terms that should have cancelled out. Would
        # need to reverse the left-right triangle directions over one semispan.
        tu, tl = self._mesh_triangles(N_s, N_sa)

        # Triangle and net surface areas
        u1, u2 = np.swapaxes(np.diff(tu, axis=1), 0, 1)
        l1, l2 = np.swapaxes(np.diff(tl, axis=1), 0, 1)
        au = np.linalg.norm(np.cross(u1, u2), axis=1) / 2
        al = np.linalg.norm(np.cross(l1, l2), axis=1) / 2
        Au = np.sum(au)
        Al = np.sum(al)

        # Triangle and net surface area centroids
        cu = np.einsum("Nij->Nj", tu) / 3
        cl = np.einsum("Nij->Nj", tl) / 3
        Cu = np.einsum("N,Ni->i", au, cu) / Au
        Cl = np.einsum("N,Ni->i", al, cl) / Al

        # Surface area inertia tensors
        cov_au = np.einsum("N,Ni,Nj->ij", au, cu, cu)
        cov_al = np.einsum("N,Ni,Nj->ij", al, cl, cl)
        Ju = np.trace(cov_au) * np.eye(3) - cov_au
        Jl = np.trace(cov_al) * np.eye(3) - cov_al

        # -------------------------------------------------------------------
        # Volumes

        # The volume calculation requires a closed mesh, so resample the
        # surface using the closed section profiles (ie, ignore air intakes).
        s = np.linspace(-1, 1, N_s)
        sa = 1 - np.cos(np.linspace(0, np.pi / 2, N_sa))
        sa = np.concatenate((-sa[:0:-1], sa))
        surface_vertices = self.surface_xyz(s[:, None], sa, "airfoil")

        # Using Delaunay is too slow as the number of vertices increases.
        S, SA = np.meshgrid(np.arange(N_s - 1), np.arange(2 * N_sa - 2), indexing='ij')
        triangle_indices = np.concatenate(
            (
                [[S, SA], [S + 1, SA + 1], [S + 1, SA]],
                [[S, SA], [S, SA + 1], [S + 1, SA + 1]],
            ),
            axis=-2,
        )
        ti = np.moveaxis(triangle_indices, (0, 1), (-2, -1)).reshape(-1, 3, 2)
        surface_indices = np.ravel_multi_index(ti.T, (N_s, 2 * N_sa - 1)).T
        surface_tris = surface_vertices.reshape(-1, 3)[surface_indices]

        # Add two meshes to close the wing tips so the volume is counted
        # correctly. Uses the 2D section profile to compute the triangulation.
        # If a wing tip has a closed trailing edge there will be coplanar
        # vertices, which `Delaunay` will discard automatically.
        left_vertices = self.surface_xyz(-1, sa, "airfoil")
        right_vertices = self.surface_xyz(1, sa, "airfoil")

        # Verson 1: use scipy.spatial.Delaunay. This version will automatically
        # discard coplanar points if the trailing edge is closed.
        #
        left_points = self.sections.surface_xz(-1, sa, 'airfoil')
        right_points = self.sections.surface_xz(1, sa, 'airfoil')
        left_tris = left_vertices[Delaunay(left_points).simplices]
        right_tris = right_vertices[Delaunay(right_points).simplices[:, ::-1]]

        # Version 2: build the list of simplices explicitly. This version just
        # assumes the trailing edge is closed (reasonable for inflatable foils)
        # and discards the vertex at the upper surface trailing edge. This is
        # fine even if the trailing edge is not truly closed as long as it's
        # _effectively_ closed and N_sa is reasonably large.
        #
        # ix = np.arange(N_sa - 2)
        # upper_simplices = np.stack((2 * N_sa - 3 - ix, 2 * N_sa - 4 - ix, ix + 1)).T
        # lower_simplices = np.stack((ix + 1, ix, 2 * N_sa - 3 - ix)).T
        # simplices = np.concatenate((upper_simplices, lower_simplices))
        # left_tris = left_vertices[simplices]
        # right_tris = right_vertices[simplices[:, ::-1]]

        tris = np.concatenate((left_tris, surface_tris, right_tris))

        # Tetrahedron signed volumes and net volume
        v = np.einsum("Ni,Ni->N", cross3(tris[..., 0], tris[..., 1]), tris[..., 2]) / 6
        V = np.sum(v)

        # Tetrahedron centroids and net centroid
        cv = np.einsum("Nij->Nj", tris) / 4
        Cv = np.einsum("N,Ni->i", v, cv) / V

        # Volume inertia tensors
        cov_canonical = np.full((3,3), 1 / 120) + np.eye(3) / 120
        cov_v = np.einsum(
            "N,Nji,jk,Nkl->il",
            np.linalg.det(tris),
            tris,
            cov_canonical,
            tris,
            optimize=True,
        )
        Jv = np.eye(3) * np.trace(cov_v) - cov_v

        mass_properties = {
            "upper_area": Au,
            "upper_centroid": Cu,
            "upper_inertia": Ju,
            "volume": V,
            "volume_centroid": Cv,
            "volume_inertia": Jv,
            "lower_area": Al,
            "lower_centroid": Cl,
            "lower_inertia": Jl,
        }

        return mass_properties

    def _mesh_vertex_lists(self, N_s=131, N_sa=151, filename=None):
        """
        Generate sets of triangle faces on the upper and lower surfaces.

        Each triangle mesh is described by a set of vertices and a set of
        "faces". The vertices are the surface coordiantes sampled on a
        rectilinear grid over the section indices and surface coordinates. The
        faces are the list of vertices that define the triangles.

        Parameters
        ----------
        N_s : int
            The grid resolution over the section index.
        N_sa : int
            The grid resolution over the surface coordinates.
        filename : string, optional
            Save the outputs in a numpy `.npz` file.

        Returns
        -------
        vertices_upper, vertices_lower : array of float, shape (N_s * N_sa, 3)
            Vertices on the upper and lower surfaces.
        simplices : array of int, shape (N_s * N_sa * 2, 3)
            Lists of vertex indices for each triangle. The same grid was used
            for both surfaces, so this array defines both meshes.

        See Also
        --------
        _mesh_triangles : Helper function that produces the meshes themselves
                          as two lists of vertex triplets (the triangles in
                          frd coordinates).

        Examples
        --------
        To export the mesh into Blender:

        1. Generate the mesh

           >>> foil._mesh_vertex_lists(filename='/path/to/mesh.npz')

        2. In the Blender (v2.8) Python console:

           import numpy
           data = numpy.load('/path/to/mesh.npz')
           # Blender doesn't support numpy arrays
           vu = data['vertices_upper'].tolist()
           vl = data['vertices_lower'].tolist()
           simplices = data['simplices'].tolist()
           mesh_upper = bpy.data.meshes.new("upper")
           mesh_lower = bpy.data.meshes.new("lower")
           object_upper = bpy.data.objects.new("upper", mesh_upper)
           object_lower = bpy.data.objects.new("lower", mesh_lower)
           bpy.context.scene.collection.objects.link(object_upper)
           bpy.context.scene.collection.objects.link(object_lower)
           mesh_upper.from_pydata(vu, [], simplices)
           mesh_lower.from_pydata(vl, [], simplices)
           mesh_upper.update(calc_edges=True)
           mesh_lower.update(calc_edges=True)
        """
        # Compute the vertices
        s = np.linspace(-1, 1, N_s)
        sa = 1 - np.cos(np.linspace(0, np.pi / 2, N_sa))

        # The lower surface goes right->left to ensure the normals point down,
        # which is important for computing the enclosed volume of air between
        # the two surfaces, and for 3D programs in general (which tend to
        # expect the normals to point "out" of the volume).
        vu = self.surface_xyz(s[:, np.newaxis], sa, 'upper').reshape(-1, 3)
        vl = self.surface_xyz(s[::-1, np.newaxis], sa, 'lower').reshape(-1, 3)

        # Compute the vertex lists for all of the faces (the triangles).
        S, SA = np.meshgrid(np.arange(N_s - 1), np.arange(N_sa - 1), indexing='ij')
        triangle_indices = np.concatenate(
            (
                [[S, SA], [S + 1, SA + 1], [S + 1, SA]],
                [[S, SA], [S, SA + 1], [S + 1, SA + 1]],
            ),
            axis=-2,
        )
        ti = np.moveaxis(triangle_indices, (0, 1), (-2, -1)).reshape(-1, 3, 2)
        simplices = np.ravel_multi_index(ti.T, (N_s, N_sa)).T

        if filename:
            np.savez_compressed(
                filename,
                vertices_upper=vu,
                vertices_lower=vl,
                simplices=simplices,  # Same list for both surfaces
            )

        return vu, vl, simplices

    def _mesh_triangles(self, N_s=131, N_sa=151, filename=None):
        """Generate triangle meshes over the upper and lower surfaces.

        Parameters
        ----------
        N_s : int
            The grid resolution over the section index.
        N_sa : int
            The grid resolution over the surface coordinates.
        filename : string, optional
            Save the outputs in a numpy `.npz` file.

        Returns
        -------
        tu, tl : array of float, shape ((N_s - 1) * (N_sa - 1) * 2, 3, 3)
            Lists of vertex triplets that define the triangles on the upper
            and lower surfaces.

            The shape warrants an explanation: the grid has `N_s * N_sa` points
            for `(N_s - 1) * (N_sa - 1)` rectangles. Each rectangle requires
            2 triangles, each triangle has 3 vertices, and each vertex has
            3 coordinates (in frd).

        See Also
        --------
        _mesh_vertex_lists : the sets of vertices and the indices that define
                             the triangles

        Examples
        --------
        To export the mesh into FreeCAD:

        1. Generate the mesh

           >>> foil._mesh_triangles('/path/to/triangles.npz')

        2. In the FreeCAD Python (v0.18) console:

           >>> # See: https://www.freecadweb.org/wiki/Mesh_Scripting
           >>> import Mesh
           >>> triangles = np.load('/path/to/triangles.npz')
           >>> # As of FreeCAD v0.18, `Mesh` doesn't support numpy arrays
           >>> mesh_upper = Mesh.Mesh(triangles['triangles_upper'].tolist())
           >>> mesh_lower = Mesh.Mesh(triangles['triangles_lower'].tolist())
           >>> Mesh.show(mesh_upper)
           >>> Mesh.show(mesh_lower)
        """
        vu, vl, fi = self._mesh_vertex_lists(N_s=N_s, N_sa=N_sa)
        triangles_upper = vu[fi]
        triangles_lower = vl[fi]

        if filename:
            np.savez_compressed(
                filename,
                triangles_upper=triangles_upper,
                triangles_lower=triangles_lower,
            )

        return triangles_upper, triangles_lower


# ---------------------------------------------------------------------------


class ForceEstimator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, delta_f, v_W2f, rho_air):
        """
        Estimate the forces and moments on a foil.

        Parameters
        ----------
        delta_f : array_like of float [radians]
            The deflection angle of each section. The shape must be able to
            broadcast to (K,), where `K` is the number of control points being
            used by the estimator.
        v_W2f : array_like of float [m/s]
            The velocity of the wind relative to the control points in foil frd
            coordinates. The shape must be able to broadcast to (K, 3), where
            `K` is the number of control points being used by the estimator.
        rho_air : float [kg/m^3]
            Air density
        """

    @property
    @abc.abstractmethod
    def control_points(self):
        """The reference points for calculating the section forces"""

    class ConvergenceError(RuntimeError):
        """The estimator failed to converge on a solution."""


class Phillips(ForceEstimator):
    """
    A non-linear numerical lifting-line method.

    Uses a set of spanwise bound vortices instead of a single, uniform lifting
    line. Unlike the Prandtl's classic lifting-line theory, this method allows
    for wing sweep and dihedral.

    Parameters
    ----------
    foil : FoilGeometry
        Defines the lifting-line and section coefficients.
    v_ref_mag : float [m/s]
        The reference solution airspeed
    alpha_ref : float [degrees]
        The reference solution angle of attack
    K : integer
        The number of bound vortex segments. Default: 51

    References
    ----------
    .. [1] Phillips and Snyder, "Modern Adaptation of Prandtl’s Classic
       Lifting-Line Theory", Journal of Aircraft, 2000

    .. [2] McLeanauth, "Understanding Aerodynamics - Arguing from the Real
       Physics", p382

    .. [3] Hunsaker and Snyder, "A lifting-line approach to estimating
       propeller/wing interactions", 2006

    Notes
    -----
    This implementation uses a single distribution for the entire span, which
    is suitable for parafoils, which is a continuous lifting surface, but for
    wings with left and right segments separated by some discontinuity at the
    root you should distribute the points across each semispan independently.
    See _[1] for a related discussion.

    This method does suffer an issue where induced velocity goes to infinity as
    the segment lengths tend toward zero (as the number of segments increases,
    or for a poorly chosen point distribution). See _[2], section 8.2.3.
    """

    def __init__(self, foil, v_ref_mag, alpha_ref=5, K=51):
        self.foil = foil
        self.K = K

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution
        self.s_nodes = np.linspace(-1, 1, self.K + 1)

        # Option 2: cosine distribution
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K + 1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.foil.chord_xyz(self.s_nodes, 0.25)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1]) / 2
        self.cps = self.foil.chord_xyz(self.s_cps, 0.25)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]
        self.r1 = np.linalg.norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = np.linalg.norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        # Note: Phillip's derivation uses back-left-up coordinates (not `frd`)
        u = -self.foil.section_orientation(self.s_cps).T
        self.u_a, self.u_s, self.u_n = u[0].T, u[1].T, u[2].T

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        node_chords = self.foil.chord_length(self.s_nodes)
        self.c_avg = (node_chords[1:] + node_chords[:-1]) / 2
        self.dA = self.c_avg * np.linalg.norm(cross3(self.u_a, self.dl), axis=1)

        # Precompute the `v` terms that do not depend on `u_inf`
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2  # Shorthand
        self.v_ij = np.zeros((self.K, self.K, 3))  # Extra terms when `i != j`
        for ij in [(i, j) for i in range(self.K) for j in range(self.K)]:
            if ij[0] == ij[1]:  # Skip singularities when `i == j`
                continue
            self.v_ij[ij] = ((r1[ij] + r2[ij]) * cross3(R1[ij], R2[ij])) / \
                (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + np.dot(R1[ij], R2[ij])))

        # Precompute a reference solution from a (hopefully easy) base case.
        # Sets an initial "solution" (which isn't actually a solution) just to
        # bootstrap the `__call__` method with an initial `Gamma` value.
        alpha_ref = np.deg2rad(alpha_ref)
        v_mag = np.broadcast_to(v_ref_mag, (self.K, 3))
        v_W2f_ref = -v_mag * np.array([np.cos(alpha_ref), 0, np.sin(alpha_ref)])
        self._reference_solution = {
            'delta_f': 0,
            'v_W2f': v_W2f_ref,
            'Gamma': np.sqrt(1 - self.s_cps ** 2),  # Naive ellipse
        }
        try:
            _, _, self._reference_solution = self.__call__(0, v_W2f_ref, 1.2)
        except ForceEstimator.ConvergenceError as e:
            raise RuntimeError("Phillips: failed to initialize base case")

    def _compute_Reynolds(self, v_W2f, rho_air):
        """Compute the Reynolds number at each control point."""

        # FIXME: verify that using the total airspeed (including spanwise flow)
        #        is okay. A few tests show minimal differences, so for now I'm
        #        not wasting time computing the normal and chordwise flows.
        u = np.linalg.norm(v_W2f, axis=-1)  # airspeed [m/s]
        mu = 1.81e-5  # Standard dynamic viscosity of air
        Re = rho_air * u * self.c_avg / mu
        # print("\nDEBUG> Re:", Re, "\n")
        return Re

    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def _induced_velocities(self, u_inf):
        # 2. Compute the "induced velocity" unit vectors
        #  * ref: Phillips, Eq:6
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2  # Shorthand
        v = self.v_ij.copy()
        v += (
            cross3(u_inf, R2)
            / (r2 * (r2 - np.einsum("k,ijk->ij", u_inf, R2)))[..., None]
        )
        v -= (
            cross3(u_inf, R1)
            / (r1 * (r1 - np.einsum("k,ijk->ij", u_inf, R1)))[..., None]
        )

        return v / (4 * np.pi)  # axes: (inducer, inducee, 3-vector)

    def _local_velocities(self, v_W2f, Gamma, v):
        # Compute the local fluid velocities
        #  * ref: Hunsaker Eq:5
        #  * ref: Phillips Eq:5 (nondimensional version)
        V = v_W2f + np.einsum("j,jik->ik", Gamma, v)

        # Compute the local angle of attack for each section
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_n = np.einsum("ik,ik->i", V, self.u_n)  # Normal-wise
        V_a = np.einsum("ik,ik->i", V, self.u_a)  # Chordwise
        alpha = np.arctan(V_n / V_a)

        return V, V_n, V_a, alpha

    def _f(self, Gamma, delta_f, v_W2f, v, Re):
        # Compute the residual error vector
        #  * ref: Hunsaker Eq:8
        #  * ref: Phillips Eq:14
        V, V_n, V_a, alpha = self._local_velocities(v_W2f, Gamma, v)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(np.einsum("ik,ik->i", W, W))
        Cl = self.foil.sections.Cl(self.s_cps, delta_f, alpha, Re)
        # return 2 * Gamma * W_norm - np.einsum("ik,ik,i,i->i", V, V, self.dA, Cl)
        return 2 * Gamma * W_norm - (V_n ** 2 + V_a ** 2) * self.dA * Cl

    def _J(self, Gamma, delta_f, v_W2f, v, Re, verify_J=False):
        # 7. Compute the Jacobian matrix, `J[ij] = d(f_i)/d(Gamma_j)`
        #  * ref: Hunsaker Eq:11
        V, V_n, V_a, alpha = self._local_velocities(v_W2f, Gamma, v)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(np.einsum("ik,ik->i", W, W))
        Cl = self.foil.sections.Cl(self.s_cps, delta_f, alpha, Re)
        Cl_alpha = self.foil.sections.Cl_alpha(self.s_cps, delta_f, alpha, Re)

        J = 2 * np.diag(W_norm)  # Additional terms for i==j
        J2 = 2 * np.einsum("i,ik,i,jik->ij", Gamma, W, 1 / W_norm, cross3(v, self.dl))
        J3 = (np.einsum("i,jik,ik->ij", V_a, v, self.u_n)
              - np.einsum("i,jik,ik->ij", V_n, v, self.u_a))
        J3 *= (
            (self.dA * Cl_alpha)[:, None]
            * np.einsum("ik,ik->i", V, V)
            / (V_n ** 2 + V_a ** 2)
        )
        J4 = 2 * np.einsum("i,i,ik,jik->ij", self.dA, Cl, V, v)
        J += J2 - J3 - J4

        # Compare the analytical gradient to the finite-difference version
        if verify_J:
            J_true = self._J_finite(Gamma, delta_f, v_W2f, v, Re)
            if not np.allclose(J, J_true):
                print("\n !!! The analytical Jacobian disagrees. Halting. !!!")
                embed()

        return J

    def _J_finite(self, Gamma, delta_f, v_W2f, v, Re):
        """Compute the Jacobian using a centered finite distance.

        Useful for checking the analytical gradient.

        Examples
        --------
        >>> J1 = self._J(Gamma, v_W2f, v, delta_f)
        >>> J2 = self._J_finite(Gamma, v_W2f, v, delta_f)
        >>> np.allclose(J1, J2)  # FIXME: tune the tolerances?
        True
        """
        # This uses the same method as `scipy.optimize.approx_fprime`, but that
        # function only works for scalar-valued functions.
        JT = np.empty((self.K, self.K))  # Jacobian transpose  (J_ji)
        eps = np.sqrt(np.finfo(float).eps)

        # Build the Jacobian column-wise (row-wise of the tranpose)
        Gp, Gm = Gamma.copy(), Gamma.copy()
        for k in range(self.K):
            Gp[k], Gm[k] = Gamma[k] + eps, Gamma[k] - eps
            fp = self._f(Gp, delta_f, v_W2f, v, Re)
            fm = self._f(Gm, delta_f, v_W2f, v, Re)
            JT[k] = (fp - fm) / (2 * eps)
            Gp[k], Gm[k] = Gamma[k], Gamma[k]

        return JT.T

    def _solve_circulation(self, delta_f, v_W2f, Re, Gamma0):
        """
        Solve for the spanwise circulation distribution.

        Parameters
        ----------
        delta_f : array of float, shape (K,) [radians]
            The deflection angle of each section.
        v_W2f : array of float, shape (K,) [m/s]
            Relative wind velocity at each control point.
        Re : array of float, shape (K,)
            Reynolds number at each segment
        Gamma0 : array of float, shape (K,)
            The initial proposal

        Returns
        -------
        Gamma : array of float, shape (K,)
            Circulation strengths of each segment.
        v : array, shape (K,K,3) [m/s]
            Induced velocities between each segment, indexed as (inducer,
            inducee).
        """
        v_mid = v_W2f[self.K // 2]
        u_inf = v_mid / np.linalg.norm(v_mid)  # FIXME: what if PQR != 0?
        v = self._induced_velocities(u_inf)
        args = (delta_f, v_W2f, v, Re)
        res = scipy.optimize.root(self._f, Gamma0, args, jac=self._J, tol=1e-4)

        if not res["success"]:
            raise ForceEstimator.ConvergenceError

        return res["x"], v

    def __call__(self, delta_f, v_W2f, rho_air, reference_solution=None, max_splits=10):
        # FIXME: this doesn't match the ForceEstimator.__call__ signature
        delta_f = np.broadcast_to(delta_f, (self.K))
        v_W2f = np.broadcast_to(v_W2f, (self.K, 3))
        Re = self._compute_Reynolds(v_W2f, rho_air)

        if reference_solution is None:
            reference_solution = self._reference_solution

        delta_f_ref = reference_solution['delta_f']
        v_W2f_ref = reference_solution['v_W2f']
        Gamma_ref = reference_solution['Gamma']

        # Try to solve for the target (`Gamma` as a function of `v_W2f` and
        # `delta_f`) directly using the `reference_solution`. If that fails,
        # pick a point between the target and the reference, solve for that
        # easier case, then use its solution as the new starting point for the
        # next target. Repeat for intermediate targets until either solving for
        # the original target, or exceeding `max_splits`.
        target_backlog = []  # Stack of pending targets
        num_splits = 0
        while True:
            try:
                Gamma, v = self._solve_circulation(delta_f, v_W2f, Re, Gamma_ref)
            except ForceEstimator.ConvergenceError:
                if num_splits == max_splits:
                    raise ForceEstimator.ConvergenceError("max splits reached")
                num_splits += 1
                target_backlog.append((delta_f, v_W2f))
                P = 0.5  # Ratio, a point between the reference and the target
                delta_f = (1 - P) * delta_f_ref + P * delta_f
                v_W2f = (1 - P) * v_W2f_ref + P * v_W2f
                continue

            delta_f_ref = delta_f
            v_W2f_ref = v_W2f
            Gamma_ref = Gamma

            if target_backlog:
                delta_f, v_W2f = target_backlog.pop()
            else:
                break

        V, V_n, V_a, alpha = self._local_velocities(v_W2f, Gamma, v)

        # Compute the inviscid forces using the 3D vortex lifting law
        #  * ref: Hunsaker Eq:1
        #  * ref: Phillips Eq:4
        dF_inviscid = Gamma * cross3(V, self.dl).T

        # Compute the viscous forces.
        #  * ref: Hunsaker Eq:17
        #
        # The equation in the paper uses the "characteristic chord", but I
        # believe that is a mistake; it produces *massive* drag. Here I use the
        # section area like they do in "MachUp_Py" (see where they compute
        # `f_parasite_mag` in `llmodel.py:LLModel:_compute_forces`).
        Cd = self.foil.sections.Cd(self.s_cps, delta_f, alpha, Re)
        V2 = np.einsum("ik,ik->i", V, V)
        u_drag = V.T / np.sqrt(V2)
        dF_viscous = 0.5 * V2 * self.dA * Cd * u_drag

        # The total forces applied at each control point
        dF = dF_inviscid + dF_viscous

        # Compute the section moments.
        #  * ref: Hunsaker Eq:19
        #  * ref: Phillips Eq:28
        #
        # These are strictly the section moments caused by airflow around the
        # section. It does not include moments about the aircraft reference
        # point (commonly the center of gravity); those extra moments must be
        # calculated by the wing.
        #  * ref: Hunsaker Eq:19
        #  * ref: Phillips Eq:28
        Cm = self.foil.sections.Cm(self.s_cps, delta_f, alpha, Re)
        dM = -0.5 * V2 * self.dA * self.c_avg * Cm * self.u_s.T

        solution = {
            'delta_f': delta_f_ref,
            'v_W2f': v_W2f_ref,
            'Gamma': Gamma_ref,
        }

        # print("\nFinished `Phillips.__call__`")
        # embed()

        dF *= rho_air
        dM *= rho_air

        return dF.T, dM.T, solution
