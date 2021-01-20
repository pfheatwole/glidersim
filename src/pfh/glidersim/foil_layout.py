"""FIXME: add docstring."""

import warnings

import numpy as np
import scipy


__all__ = [
    "EllipticalArc",
    "elliptical_chord",
    "elliptical_arc",
    "PolynomialTorsion",
    "FlatYZ",
    "SectionLayout",
]


def __dir__():
    return __all__


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

    This class supports both "explicit" and "parametric" representations of an
    ellipse. The explicit version returns the vertical component as a function
    of the horizontal coordinate, and the parametric version returns both the
    horizontal and vertical coordinates as a function of the parametric
    parameter.

    The parametric curve in this class is always parametrized by `t`. By
    setting `t_domain` you can constrain the curve to an elliptical arc.

    For both the explicit and parametric forms, you can change the input domain
    by setting `p_domain`, and they will be scaled automatically to map onto the
    explicit (`x`) or parametric (`t`) parameter.
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
        kind : {'explicit', 'parametric'}, default: 'parametric'
            Specifies whether the class return `y = f(x)` or `<x, y> = f(p)`.
            The explicit version returns coordinates of the second axis given
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

        if kind == "explicit":  # the `p` are `x` in the external domain
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

        if kind == "explicit":
            # The explicit `y = f(x)` can be converted to `y = f(t)` by first
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
        if kind is None and self.kind == "explicit":
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
        with `SectionLayout`).
    """

    taper = tip / root
    A = 1 / np.sqrt(1 - taper ** 2)
    B = root
    t_min = np.arcsin(taper)

    return EllipticalArc(
        A / B, B=B, p_domain=[1, -1], t_domain=[t_min, np.pi - t_min], kind="explicit",
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
        arc length `2` (suitable for use with `SectionLayout`).
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


class FlatYZ:
    """Helper class for completely flat wings (no dihedral anywhere)."""

    def __call__(self, s):
        """Define `(y, z) = (s, 0)` for all sections."""
        _0 = np.full(np.shape(s), 0.0)
        return np.stack((s, _0), axis=-1)

    def derivative(self, s):
        """Define `(dyds, dzds) = (1, 0)` for all sections."""
        return np.broadcast_to([1.0, 0.0], (*np.shape(s), 2)).copy()


class SectionLayout:
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
        c = self.c(s)
        lengths = (self._section_pitch(s)[..., 0] * c[:, np.newaxis]).T[0]
        return scipy.integrate.simps(lengths, s) * self.b_flat / 2

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
        K = np.sqrt(dyds ** 2 + dzds ** 2)  # L2-norm
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
        C_c2s = self._section_pitch(s)
        if not flatten:
            C_c2s = self._section_roll(s) @ C_c2s
        return C_c2s

    def xyz(self, s, r, flatten=False):
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
        s = np.asarray(s)
        r = np.asarray(r)
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if r.min() < 0 or r.max() > 1:
            raise ValueError("Chord ratios must be between 0 and 1.")

        # FIXME? Written this way for clarity, but some terms may be unused.
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
