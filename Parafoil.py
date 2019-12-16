"""FIXME: add docstring."""

import abc
import warnings

from IPython import embed

import numpy as np

import scipy.interpolate
import scipy.optimize

from util import cross3


class EllipticalArc:
    """
    An arc segment from an ellipse.

    Although this internal representation uses the standard `t` parameter for
    the parametric functions, some applications are more easily described with
    a different domain. For example, all the `ParafoilGeometry` curves are
    defined as functions of the section index `s`, which ranges from -1 (the
    left wing tip) to +1 (the right wing tip).

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
        t_domain=None,
        p_domain=None,
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
        origin : array of float, shape (2,)
            The (x, y) offset of the ellipse origin.
        t_domain : array of float, shape (N, 2), optional
            FIXME
        p_domain : array of float, shape (N, 2), optional
            FIXME: docstring.

            The domain of the input parameter. This encodes "what values in the
            input domain maps to `t_domain`?". For example, if you wanted to a
            parametric function where -1 is the start and +1 is the end, then
            `p_domain = [1, -1]` would map `t_min` to `p = 1` and `t_max` to
            `p = -1`.

        kind : {'implicit', 'parametric'}, default: 'parametric'
            Does the class return `y = f(x)` or `<x, y> = f(p)`? The implicit
            version returns coordinates of the second axis given coordinates
            of the first axis. The parametric version returns both x and y
            given a parametric coordinate.

            Notice that the curve is parametrized by `p`, which may differ from
            the internal parameter `t` if `p_domain` has been modified.
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
        self.t_domain = t_domain
        self.p_domain = p_domain
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
    """Build an elliptical chord distribution as a function of the section index."""
    # FIXME: for now, only support this set of parameters

    taper = tip / root

    # Config inputs: root, tip, max_taper_rate
    #
    # For an elliptical chord functio, dc/dy = 0 at the root and the maximum
    # rate of change of taper will always occur at the tip. This function will
    # assume the user wants the fastest rate possible at the tip, but let them
    # override that if they want.

    # This rate is implicitly "per unit span" since it's over the normalized
    # planform span coordinate.
    # mean_taper_rate = root - tip

    # From Benedetti: verify
    A = 1 / np.sqrt(1 - taper ** 2)
    B = root
    t_min = np.arcsin(taper)

    return EllipticalArc(
        A / B, B=B, t_domain=[t_min, np.pi - t_min], p_domain=[1, -1], kind="implicit",
    )


def elliptical_lobe(mean_anhedral, max_anhedral_rate=90):
    """
    Build an elliptical lobe curve as a function of the section index.

    FIXME: accept the alternative pair: {b/b_flat, max_anhedral_rate}, since
    you typically know b/b_flat from wing specs, and max_anhedral is pretty
    easy to approximate from pictures.
    """
    # For a paraglider, dihedral should be negative (anhedral)
    if max_anhedral_rate <= 2 * mean_anhedral:
        raise ValueError("max_anhedral_rate <= 2 * mean_anhedral")

    mean_anhedral = np.deg2rad(mean_anhedral)
    max_anhedral_rate = np.deg2rad(max_anhedral_rate)

    v1 = 1 - np.tan(mean_anhedral) / np.tan(max_anhedral_rate)
    v2 = 1 - 2 * np.tan(mean_anhedral) / np.tan(max_anhedral_rate)
    A = v1 / np.sqrt(v2)
    B = np.tan(mean_anhedral) * v1 / v2
    t_min = np.arccos(1 / A)
    lobe = EllipticalArc(
        A / B, length=2, t_domain=[np.pi + t_min, 2 * np.pi - t_min], p_domain=[-1, 1],
    )
    lobe.origin = -lobe(0)  # The middle of the curve is the origin
    return lobe


class PolynomialTorsion:
    """
    A functor that for producing polynomial curves at the wing tips.

    The curve is symmetric about the origin. Points between `[-start, start]`
    are zero; points between [start, 1] grow from `0` to `peak` a rate
    controlled by `exponent`.
    """

    def __init__(self, start, exponent, peak, symmetric=True):
        """
        Construct the curve.

        Parameters
        ----------
        start: float
            Where the exponential curve begins, where `0 <= start <= 1`. The
            region between +/- `start` is zero.

        exponent : float
            The exponential growth rate. Controls the steepness of the curve.
            For example, `2` produces a quadratic curve between `start` and 1.

        peak : float
            The peak value of the curve at `s = 1`.
        """
        if exponent < 1:
            print("\nWarning: exponent is less than 1?\n")

        self.start = start
        self.exponent = exponent
        self.peak = np.deg2rad(peak)

    def __call__(self, s):
        # FIXME: design review? Not a fan of these asarray
        s = np.asarray(s)
        values = np.zeros_like(s)
        m = abs(s) > self.start  # Mask
        if np.any(m):
            p = (abs(s[m]) - self.start) / (1 - self.start)
            # p /= p[-1]  # `p` goes from 0 to 1
            values[m] = self.peak * p ** self.exponent
        return values


class SimpleIntakes:
    """
    Defines the upper and lower surface airfoil coordinates.

    This version currently uses explicit `s_upper` and `s_lower` in airfoil
    coordinates, but other options might be the intake midpoint and width
    (where "width" might be in the airfoil `s`, or as a percentage of the
    chord) or `c_upper` and `c_lower` as points on the chord.
    """

    def __init__(self, s_end, s_upper, s_lower):
        """

        Parameters
        ----------
        s_end: float
            Section index. Air intakes are present between +/- `s_end`.
        s_upper, s_lower : float
            The starting coordinates of the upper and lower surface of the
            parafoil, given in airfoil surface coordinates. The airfoil uses
            s = 0 for the LE, s = 1 for the upper surface trailing edge, and
            s = -1 for the lower surface trailing edge, so these choices must
            follow `-1 <= s_lower <= s_upper <= 1`.


        FIXME: support more types of definition:

        1. su/sl : explicit upper/lower cuts in airfoil coordinates

        2. midpoint (in airfoil coordinates) and width

        3. Upper and lower cuts as a fraction of the chord (the "Paraglider
           Design Manual" does it this way).
        """
        self.s_end = s_end
        self.s_upper = s_upper
        self.s_lower = s_lower

    def upper(self, s, su):
        """
        Convert parafoil upper surface coordinates into airfoil coordinates.

        Parameters
        ----------
        s : array_like of float
            Section index. Unused for the upper surface in this simple model,
            since the upper surface always extends to the same airfoil
            coordinate. It is the lower surface that extends to close the
            intake.
        su : array_like of float
            Parafoil upper surface coordinate, where `0 <= su <= 1`, with `0`
            being the leading edge, and `1` being the trailing edge.

        Returns
        -------
        array_like of float, shape (N,)
            The airfoil surface coordinates from `self.s_upper` to 1.
        """
        values = self.s_upper + su * (1 - self.s_upper)
        return values

    def lower(self, s, sl):
        """
        Convert parafoil lower surface coordinates into airfoil coordinates.

        Parameters
        ----------
        s : array_like of float
            Section index. Unused for this simple model, since the upper
            surface always extends to the same airfoil coordinate, while its
            the lower surface that changes to close the intake.
        sl : array_like of float, shape (N,)
            Parafoil lower surface coordinate, where `0 <= sl <= 1`, with `0`
            being the leading edge, and `1` being the trailing edge.

        Returns
        -------
        array_like of float, shape (N,)
            The airfoil surface coordinates from `self.s_lower` to -1.
        """
        s = np.asarray(s)
        sl = np.asarray(sl)

        # The lower section extends forward over sections without intakes
        starts = np.where(np.abs(s) < self.s_end, self.s_lower, self.s_upper)
        values = starts + sl * (-1 - starts)
        return values


# ---------------------------------------------------------------------------


class ParafoilGeometry:
    """A parafoil geometry definition using a set of parametric functions."""

    def __init__(
        self,
        x,
        r_x,
        yz,
        r_yz,
        torsion,
        airfoil,
        chord_length,
        intakes=None,
        b_flat=None,
        b=None,
        N=500,
    ):
        """
        Add a docstring.

        Parameters
        ----------
        x : function
            The x-coordinates of each section as a function of the section
            index. Each chord is shifted forward until the x-coordinate of its
            leading edge is at `chord_length * r_x`. This curve shapes the
            top-down view of the flattened wing.
        r_x : float, or a function
            A ratio from 0 to 1 that defines what location on the chord is
            located at the x-coordinate defined by `x`. This can be a constant
            or a function of the normalized span. For example, `xy_r = 1` says
            that the `x` is specifying the x-coordinate of the trailing edge.
        yz : array_like of float, shape (N_YZ, 3)
        xy_r : float, or a function
            A ratio from 0 to 1 that defines the chord position of the `yz`
            curve. This can be a constant or a function of the normalized span.
            For example, `yz_r = 1` says that the `yz` curve is specifying the
            z-coordinate of the trailing edge.
        torsion : function
            Geometric torsion as a function of the normalized span. These
            angles specify a positive rotation about the `y-axis`, relative to
            the central section.
        airfoil : Airfoil
            The airfoil.
        chord_length : function
            The length of each section chord as a function of section index.
        intakes : class, optional
            A class that provides two functions, `upper` and `lower`, that
            define the upper and lower air intake positions in airfoil surface
            coordinates as a function of the section index.
        b, b_flat : float
            The projected or the flattened wing span. Specify only one.
        """
        if b_flat is None and b is None:
            raise ValueError("Specify one of `b` or `b_flat`")
        elif b_flat is not None and b is not None:
            raise ValueError("Specify only one of `b` or `b_flat`")

        self.x = x
        self.r_x = r_x
        self.yz = yz
        self.r_yz = r_yz
        self.torsion = torsion
        self.airfoil = airfoil
        self._chord_length = chord_length  # Normalized by `b_flat / 2`
        self.intakes = intakes

        # The lobe semi-span is defined by its maximum y-coordinate. Because
        # the section index is equivalent to the length of the curve to that
        # point, this coordinate also defines the ration between the flattened
        # span `b_flat` and the projected span `b`.
        #
        # TODO: assumes the lobe is symmetric
        # FIXME: this should be the *smallest* normalized span `y` that produces
        #        maximum lobe y-coordinate. Subtle, but would fail if the lobe
        #        has a vertical segment at the semi-span.
        res = scipy.optimize.minimize_scalar(
            lambda s: -yz(s)[0], bounds=(0, 1), method="bounded",
        )
        s_lobe_span = res.x
        self.span_ratio = yz(s_lobe_span)[0] / s_lobe_span  # The b/b_flat ratio

        # The property setters are now able to convert between `b` and `b_flat`
        if b_flat is not None:
            self.b_flat = b_flat
        else:
            self.b = b

        # Set the origin to the central chord leading edge. (FIXME: kludgy)
        self.LE0 = [0, 0, 0]  # Initial zero-adjustment
        self.LE0 = self.chord_xyz(0, 0) / (self.b_flat / 2)

    @property
    def AR(self):
        """Compute the aspect ratio of the inflated wing."""
        return self.b ** 2 / self.S

    @property
    def AR_flat(self):
        """Compute the aspect ratio of the flattened wing."""
        return self.b_flat ** 2 / self.S_flat

    @property
    def b(self):
        """The project span of the inflated parafoil."""
        return self._b

    @b.setter
    def b(self, new_b):
        self._b = new_b
        self.b_flat = new_b / self.span_ratio

    @property
    def b_flat(self):
        """The span of the flattened parafoil."""
        return self._b_flat

    @b_flat.setter
    def b_flat(self, new_b_flat):
        self._b_flat = new_b_flat
        self._b = new_b_flat * self.span_ratio

    @property
    def S(self):
        """
        Approximate the projected surface area of the inflated parafoil.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane. It is not the total surface
        area of the volume.
        """
        s = np.linspace(-1, 1, 1000)
        LEx, LEy = self.chord_xyz(s, 0)[:, :2].T
        TEx, TEy = self.chord_xyz(s, 1)[:, :2].T

        # Three areas: curved front, trapezoidal mid, and curved rear
        LE_area = scipy.integrate.simps(LEx - LEx.min(), LEy)
        mid_area = (LEy.ptp() + TEy.ptp()) / 2 * (LEx.min() - TEx.max())
        TE_area = -scipy.integrate.simps(TEx - TEx.max(), TEy)
        return LE_area + mid_area + TE_area

    @property
    def S_flat(self):
        """
        Approximate the projected surface area of the flattened parafoil.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane. It is not the total surface
        area of the volume.
        """
        warnings.warn("ParafoilGeometry.S_flat tends to overestimate slightly")

        # FIXME: Tends to overestimate since it calculates raw chord areas, not
        # those areas projected onto the xy-plane. This is approximately
        # correct since the planform is mostly flat, but it does mean it does
        # not account for torsion.

        s_nodes = np.linspace(-1, 1, 1000)
        nodes = self.chord_xyz(s_nodes, 0.5)
        midpoints = (s_nodes[1:] + s_nodes[:-1]) / 2
        u = self.section_orientation(midpoints)[..., 0]

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        dl = nodes[1:] - nodes[:-1]
        node_chords = self.chord_length(s_nodes)
        c_avg = (node_chords[1:] + node_chords[:-1]) / 2
        dA = c_avg * np.linalg.norm(cross3(u, dl), axis=1)
        return dA.sum()

    def _planform_torsion(self, s):
        """
        Compute the planform torsion (rotations about the planform y-axis).

        These angles are between the x-axis of a section and the x-axis of the
        central chord when the wing is flat.
        """
        thetas = self.torsion(s)
        ct, st = np.cos(thetas), np.sin(thetas)
        _0, _1 = np.zeros_like(s), np.ones_like(s)
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

    def _lobe_dihedral(self, s):
        """
        Compute the lobe dihedral (rotations about the lobe x-axis).

        This rotation refers to the angle between the y-axis of a section and
        the y-axis of the central chord of the lobe.
        """
        derivatives = self.yz.derivative(s).T
        dyds, dzds = derivatives[0].T, derivatives[1].T
        K = np.sqrt(dyds ** 2 + dzds ** 2)  # L2-norm
        dyds /= K
        dzds /= K
        _0, _1 = np.zeros_like(s), np.ones_like(s)
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

    def section_orientation(self, s):
        """Compute the section coordinate axis unit vectors."""
        torsion = self._planform_torsion(s)
        dihedral = self._lobe_dihedral(s)
        return dihedral @ torsion

    def chord_length(self, s):
        return self._chord_length(s) * (self.b_flat / 2)

    def chord_xyz(self, s, chord_ratio):
        """
        Compute the coordinates of points on section chords.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index
        chord_ratio : float
            Position on the chords, where `chord_ratio = 0` is the leading
            edge, and `chord_ratio = 1` is the trailing edge.
        """
        assert chord_ratio >= 0 and chord_ratio <= 1

        s = np.asarray(s)
        torsion = self._planform_torsion(s)
        dihedral = self._lobe_dihedral(s)
        xhat_planform = torsion @ [1, 0, 0]
        xhat_wing = dihedral @ torsion @ [1, 0, 0]
        c = self._chord_length(s)[..., np.newaxis]  # Proportional chords!
        LE = (np.concatenate((np.zeros_like(s)[..., np.newaxis], self.yz(s)), axis=-1)
              + (self.r_x - self.r_yz) * c * xhat_planform
              + self.r_yz * c * xhat_wing)
        xyz = LE - chord_ratio * c * xhat_wing - self.LE0
        xyz *= self.b_flat / 2
        return xyz

    def upper_surface(self, s, su):
        """Sample points on the upper surface curve of a parafoil section.

        Parameters
        ----------
        s : array_like of float
            Section index.
        su : array_like of float
            Upper surface coordinates, where `0 <= su <= 1`.

        Returns
        -------
        array of float
            A set of points from the upper surface of the airfoil in FRD. The
            shape is determined by standard numpy broadcasting of `s` and `su`.
        """
        s = np.asarray(s)
        su = np.asarray(su)
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between 0 and 1.")
        if su.min() < 0 or su.max() > 1:
            raise ValueError("The airfoil coordinates must be between 0 and 1.")

        if self.intakes:
            sa = self.intakes.upper(s, su)
        else:
            sa = su  # No intakes, the upper surface starts at the leading edge

        c = self.chord_length(s)
        upper_a = self.airfoil.geometry.surface_curve(sa)  # Unscaled airfoil
        upper = np.stack((-upper_a[..., 0], np.zeros_like(sa), -upper_a[..., 1]), axis=-1)
        orientations = self.section_orientation(s)
        surface = np.einsum("...ij,...j,...->...i", orientations, upper, c)
        return surface + self.chord_xyz(s, 0)

    def lower_surface(self, s, sl):
        """Sample points on the lower surface curve of a parafoil section.

        Parameters
        ----------
        s : array_like of float
            Section index.
        sl : array_like of float
            Lower surface coordinates, where `0 <= sl <= 1`.

        Returns
        -------
        array of float, shape (..., 3)
            A set of points from the upper surface of the airfoil in FRD. The
            shape is determined by standard numpy broadcasting of `s` and `sl`.
        """
        s = np.asarray(s)
        sl = np.asarray(sl)
        if sl.min() < 0 or sl.max() > 1:
            raise ValueError("The airfoil coordinates must be between 0 and 1.")

        if self.intakes:
            sa = self.intakes.lower(s, sl)
        else:
            sa = -sl  # No intakes, the lower surface starts at the leading edge

        c = self.chord_length(s)
        lower_a = self.airfoil.geometry.surface_curve(sa)  # Unscaled airfoil
        lower = np.stack((-lower_a[..., 0], np.zeros_like(sa), -lower_a[..., 1]), axis=-1)
        orientations = self.section_orientation(s)
        surface = np.einsum("...ij,...j,...->...i", orientations, lower, c)
        return surface + self.chord_xyz(s, 0)

    def mass_properties(self, N=250):
        """
        Compute the quantities that control inertial behavior.

        The inertia matrices returned by this function are proportional to the
        values for a physical wing, and do not have standard units. They must
        be scaled by the wing materials and air density to get their physical
        values. See "Notes" for a thorough description.

        Returns
        -------
        dictionary
            upper_area: float [m^2]
                parafoil upper surface area
            upper_centroid: ndarray of float, shape (3,) [m]
                center of mass of the upper surface material in parafoil FRD
            upper_inertia: ndarray of float, shape (3, 3) [m^4]
                The inertia matrix of the upper surface
            volume: float [m^3]
                internal volume of the inflated parafoil
            volume_centroid: ndarray of float, shape (3,) [m]
                centroid of the internal air mass in parafoil FRD
            volume_inertia: ndarray of float, shape (3, 3) [m^5]
                The inertia matrix of the internal volume
            lower_area: float [m^2]
                parafoil lower surface area
            lower_centroid: ndarray of float, shape (3,) [m]
                center of mass of the lower surface material in parafoil FRD
            lower_inertia: ndarray of float, shape (3, 3) [m^4]
                The inertia matrix of the upper surface

        Notes
        -----
        The parafoil is treated as a composite of three components: the upper
        surface, internal volume, and lower surface. Because this class only
        defines the geometry of the parafoil, not the physical properties, each
        component is treated as having unit densities, and the results are
        proportional to the values for a physical wing. To compute the values
        for a physical wing, the upper and lower surface inertia matrices must
        be scaled by the aerial densities [kg/m^2] of the upper and lower wing
        surface materials, and the volumetric inertia matrix must be scaled by
        the volumetric density [kg/m^3] of air.

        Keeping these components separate allows a user to simply multiply them
        by different wing material densities and air densities to compute the
        values for the physical wing.

        The calculation works by breaking the parafoil into N segments, where
        each segment is assumed to have a constant airfoil and chord length.
        The airfoil for each segment is extruded along the segment span using
        the perpendicular axis theorem, then oriented into body coordinates,
        and finally translated to the global centroid (of the surface or
        volume) using the parallel axis theorem.

        """
        s_nodes = np.cos(np.linspace(np.pi, 0, N + 1))
        s_mid_nodes = (s_nodes[1:] + s_nodes[:-1]) / 2  # Segment midpoints
        nodes = self.chord_xyz(s_nodes, 0.25)  # Segment endpoints
        section = self.airfoil.geometry.mass_properties()
        node_chords = self.chord_length(s_nodes)
        chords = (node_chords[1:] + node_chords[:-1]) / 2  # Dumb average
        T = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])  # ACS -> FRD
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
            + segment_origins[None, ...])

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

        # The upper/volume/lower centroids for the entire parafoil
        upper_centroid = (segment_upper_area * segment_upper_cm.T).T.sum(axis=0) / upper_area
        volume_centroid = (segment_volume * segment_volume_cm.T).T.sum(axis=0) / volume
        lower_centroid = (segment_lower_area * segment_lower_cm.T).T.sum(axis=0) / lower_area

        # Segment inertia matrices in body FRD coordinates
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
            "lower_inertia": lower_J}

        return mass_properties


# ---------------------------------------------------------------------------


class ForceEstimator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, V_rel, delta):
        """
        Estimate the forces and moments on a Parafoil.

        Parameters
        ----------
        V_rel : array_like of float [m/s]
            The velocity of the control points relative to the wind in FRD
            coordinates. The shape must be able to broadcast to (K, 3), where
            `K` is the number of control points being used by the estimator.
        delta : array_like of float [m/s]
            The deflection angle of each section. The shape must be able to
            broadcast to (K,), where `K` is the number of control points being
            used by the estimator.
        """

    @property
    @abc.abstractmethod
    def control_points(self):
        """The reference points for calculating the section forces"""


class Phillips(ForceEstimator):
    """
    A non-linear numerical lifting-line method.

    Uses a set of spanwise bound vortices instead of a single, uniform lifting
    line. Unlike the Prandtl's classic lifting-line theory, this method allows
    for wing sweep and dihedral.

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

    def __init__(self, parafoil):
        self.parafoil = parafoil

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution
        self.K = 91  # The number of bound vortex segments
        self.s_nodes = np.linspace(-1, 1, self.K + 1)

        # Option 2: cosine distribution
        # self.K = 21  # The number of bound vortex segments
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K + 1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.chord_xyz(self.s_nodes, 0.25)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1]) / 2
        self.cps = self.parafoil.chord_xyz(self.s_cps, 0.25)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]
        self.r1 = np.linalg.norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = np.linalg.norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        # Note: Phillip's derivation uses back-left-up coordinates (not `frd`)
        u = -self.parafoil.section_orientation(self.s_cps).T
        self.u_a, self.u_s, self.u_n = u[0].T, u[1].T, u[2].T

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        node_chords = self.parafoil.chord_length(self.s_nodes)
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

    @property
    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def _induced_velocities(self, u_inf):
        # 2. Compute the "induced velocity" unit vectors
        #  * ref: Phillips, Eq:6
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2  # Shorthand
        v = self.v_ij.copy()
        v += cross3(u_inf, R2) / (r2 * (r2 - np.einsum("k,ijk->ij", u_inf, R2)))[..., None]
        v -= cross3(u_inf, R1) / (r1 * (r1 - np.einsum("k,ijk->ij", u_inf, R1)))[..., None]

        return v / (4 * np.pi)

    def _proposal1(self, V_w2cp, delta):
        # Generate an elliptical Gamma distribution that uses the wing root
        # velocity as a scaling factor

        # FIXME: this version is dumb in the sense that the alpha to the
        # *section* is much less (see the alternative below), but also in the
        # sense that I'm only using alpha at the root (CL_2d[mid]) anyway?
        alpha_2d = np.arctan(V_w2cp[:, 2] / V_w2cp[:, 0])

        # V_n = np.einsum("ik,ik->i", V_w2cp, self.u_n)  # Normal-wise
        # V_a = np.einsum("ik,ik->i", V_w2cp, self.u_a)  # Chordwise
        # alpha_2d = np.arctan(V_n / V_a)

        CL_2d = self.parafoil.airfoil.coefficients.Cl(alpha_2d, delta)
        mid = self.K // 2
        Gamma0 = np.linalg.norm(V_w2cp[mid]) * self.dA[mid] * CL_2d[mid]

        # Assumes the wing is symmetric and that `2 * cp_y[-1]` equals the
        # projected span. Using `parafoil.b` directly is dangerous: if it
        # wasn't calculated using the quarter-chord position at `s = 1` then
        # you might get negative values for the square root.
        # FIXME: add sanity checks.
        cp_y = self.cps[:, 1]
        Gamma = Gamma0 * np.sqrt(1 - (cp_y / cp_y[-1]) ** 2) + 0.01

        return Gamma

    def _local_velocities(self, V_w2cp, Gamma, v):
        # Compute the local fluid velocities
        #  * ref: Hunsaker-Snyder Eq:5
        #  * ref: Phillips Eq:5 (nondimensional version)
        V = V_w2cp + np.einsum("j,jik->ik", Gamma, v)

        # Compute the local angle of attack for each section
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_n = np.einsum("ik,ik->i", V, self.u_n)  # Normal-wise
        V_a = np.einsum("ik,ik->i", V, self.u_a)  # Chordwise
        alpha = np.arctan(V_n / V_a)

        return V, V_n, V_a, alpha

    def _f(self, Gamma, V_w2cp, v, delta):
        # Compute the residual error vector
        #  * ref: Phillips Eq:14
        #  * ref: Hunsaker-Snyder Eq:8
        V, V_n, V_a, alpha = self._local_velocities(V_w2cp, Gamma, v)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(np.einsum("ik,ik->i", W, W))
        Cl = self.parafoil.airfoil.coefficients.Cl(alpha, delta)

        # FIXME: verify: `V**2` or `(V_n**2 + V_a**2)` or `V_w2cp**2`
        f = 2 * Gamma * W_norm - (V_n ** 2 + V_a ** 2) * self.dA * Cl

        return f

    def _J(self, Gamma, V_w2cp, v, delta, verify_J=False):
        # 7. Compute the Jacobian matrix, `J[ij] = d(f_i)/d(Gamma_j)`
        #  * ref: Hunsaker-Snyder Eq:11
        V, V_n, V_a, alpha = self._local_velocities(V_w2cp, Gamma, v)
        V_na = (V_n[:, None] * self.u_n) + (V_a[:, None] * self.u_a)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(np.einsum("ik,ik->i", W, W))
        Cl = self.parafoil.airfoil.coefficients.Cl(alpha, delta)
        Cl_alpha = self.parafoil.airfoil.coefficients.Cl_alpha(alpha, delta)

        # Use precomputed optimal einsum paths
        opt2 = ["einsum_path", (0, 2), (0, 2), (0, 1)]
        opt3 = ["einsum_path", (0, 2), (0, 1)]
        opt4 = ["einsum_path", (0, 1), (1, 2), (0, 1)]

        J = 2 * np.diag(W_norm)  # Additional terms for i==j
        J2 = 2 * np.einsum("i,ik,i,jik->ij", Gamma, W, 1 / W_norm,
                        cross3(v, self.dl), optimize=opt2)
        J3 = (np.einsum("i,jik,ik->ij", V_a, v, self.u_n, optimize=opt3)
              - np.einsum("i,jik,ik->ij", V_n, v, self.u_a, optimize=opt3))
        J3 *= (self.dA * Cl_alpha)[:, None]
        J4 = 2 * np.einsum("i,i,jik,ik->ij", self.dA, Cl, v, V_na, optimize=opt4)
        J += J2 - J3 - J4

        # Compare the analytical gradient to the finite-difference version
        if verify_J:
            J_true = self._J_finite(Gamma, V_w2cp, v, delta)
            mask = ~np.isnan(J_true) | ~np.isnan(J)
            if not np.allclose(J[mask], J_true[mask]):
                print("\n !!! The analytical Jacobian is wrong. Halting. !!!")
                embed()

        return J

    def _J_finite(self, Gamma, V_w2cp, v, delta):
        """Compute the Jacobian using a centered finite distance.

        Useful for checking the analytical gradient.

        Examples
        --------
        >>> J1 = self._J(Gamma, V_w2cp, v, delta)
        >>> J2 = self._J_finite(Gamma, V_w2cp, v, delta)
        >>> np.allclose(J1, J2)  # FIXME: tune the tolerances?
        True
        """
        JT = np.empty((self.K, self.K))  # Jacobian transpose  (J_ji)
        eps = np.sqrt(np.finfo(float).eps)  # ref: `approx_prime` docstring

        # Build the Jacobian column-wise (row-wise of the tranpose)
        Gp, Gm = Gamma.copy(), Gamma.copy()
        for k in range(self.K):
            Gp[k], Gm[k] = Gamma[k] + eps, Gamma[k] - eps
            fp = self._f(Gp, V_w2cp, v, delta)
            fm = self._f(Gm, V_w2cp, v, delta)
            JT[k] = (fp - fm) / (2 * eps)
            Gp[k], Gm[k] = Gamma[k], Gamma[k]

        return JT.T

    def _fixed_point_circulation(self, Gamma, V_w2cp, v, delta,
                                 maxiter=500, xtol=1e-8):
        """Improve a proposal via fixed-point iterations.

        Warning: This method needs a lot of work and validation!

        Parameters
        ----------
        Gamma : array of float, shape (K,)
            The proposal circulation for the given {V_w2cp, delta}
        <etc>
        """
        # FIXME: needs validation; do not trust the results!
        #        Assumes the fixed points are attractive; are they?
        # FIXME: I should use some kind of "learning rate" parameter instead
        #        of a fixed 5% every iteration sort of thing; the current
        #        implementation can get stuck oscillating.
        G = Gamma
        Gammas = [G]  # FIXME: For debugging
        success = False
        for k in range(maxiter):
            print(".", end="")
            G_old = G

            # Equate the section lift predicted by both the lift coefficient
            # and the Kutta-Joukowski theorem, then solve for Gamma
            # FIXME: Should `G` use `V*V` or `(V_n**2 + V_a**2)`?
            V, V_n, V_a, alpha = self._local_velocities(V_w2cp, G, v)
            W_norm = np.linalg.norm(np.cross(V, self.dl), axis=1)
            Cl = self.parafoil.airfoil.coefficients.Cl(alpha, delta)
            G = (.5 * (V_n ** 2 + V_a ** 2) * self.dA * Cl) / W_norm

            # --------------------------------------------------------------
            # Smoothing/damping options to improve convergence

            # Option 0: Damped raw transitions, no smoothing
            # G = G_old + 0.05 * (G - G_old)

            # Option 1: Smooth the delta_Gamma
            # p = Polynomial.fit(self.s_cps, G - G_old, 9)
            # p = UnivariateSpline(self.s_cps, G - G_old, s=0.001)
            # G = Gamma + 0.05*p(self.s_cps)

            # Option 2: Smooth the final Gamma
            p = scipy.interpolate.UnivariateSpline(self.s_cps, G_old + 0.5 * (G - G_old), s=0.001)
            G = p(self.s_cps)

            # Option 3: Smooth Gamma, and force Gamma to zero at the tips
            # ss = np.r_[-1, self.s_cps, 1]
            # gg = np.r_[0, G_old + 0.05*(G - G_old), 0]
            # ww = np.r_[100, np.ones(self.K), 100]
            # p = UnivariateSpline(ss, gg, s=0.0001, w=ww)
            # G = p(self.s_cps)

            Gammas.append(G)

            if np.any(np.isnan(G)):
                break
            if np.all(np.abs((G - G_old) / G_old) < xtol):
                success = True
                break
        print()

        # For debugging
        # residuals = [self._f(G, V_w2cp, v, delta) for G in Gammas]
        # RMSE = [np.sqrt(sum(r**2) / self.K) for r in residuals]

        # print("Finished fixed_point iterations")
        # embed()

        return {"x": G, "success": success}

    def naive_root(self, Gamma0, args, rtol=1e-5, atol=1e-8):
        """
        Solve for Gamma using a naive Newton-Raphson iterative solution.

        Evaluates the Jacobian at every update, so it can be rather slow. It's
        value is in its simplicity, and the fact that it more closely matches
        the original solution used by Phillips.

        Warning: this method is crude, and not well tested. It was a testing
        hack, and will likely be removed.

        Parameters
        ----------
        Gamma0 : ndarray of float, shape (K,)
            The initial circulation distribution
        args : tuple
            The arguments to `_f` and `_J`
        rtol : float (optional)
            The minimum relative decrease before declaring convergence. When
            the largest relative reduction in the components of R is smaller
            than rtol, the method declares success and halts.
        atol : float (optional)
            The minimum absolute decrease before declaring convergence. When
            the largest reduction magnitude in the components of R is smaller
            than rtol, the method declares success and halts.

        Returns
        -------
        Gamma : ndarray of float, shape (K,)
            The circulation distribution that minimizes the magnitude of `_f`
        """
        Gamma = Gamma0.copy()

        R_max = 1e99
        Omega = 1
        success = False
        while Omega > 0.05:
            R = self._f(Gamma, *args)
            if np.any(np.isnan(R)):
                # FIXME: this fails if R has `nan` on the first run
                # FIXME: this will repeatedly fail if delta_Gamma has nan
                Gamma -= Omega * delta_Gamma  # Revert the previous adjustment
                Omega /= 2  # Backoff the step size
                Gamma += Omega * delta_Gamma  # Apply the reduced step
                continue
            # Check the convergence criteria
            #  1. Strictly decreasing R
            #  2. Absolute tolerance of the largest decrease
            #  3. Relative tolerance of the largest decrease
            R_max_new = np.max(np.abs(R))
            if (R_max_new >= R_max) or \
               R_max_new < atol or \
               (R_max - R_max_new) / R_max < rtol:
                success = True
                break
            R_max = R_max_new  # R_max_new is less than the previous R_max
            J = self._J(Gamma, *args)
            delta_Gamma = np.linalg.solve(J, -R)
            Gamma += Omega * delta_Gamma
            Omega += (1 - Omega) / 2  # Restore Omega towards unity

        return {"x": Gamma, "success": success}

    def _solve_circulation(self, V_w2cp, delta, Gamma0=None):
        if Gamma0 is None:  # Assume a simple elliptical distribution
            Gamma0 = self._proposal1(V_w2cp, delta)

        # FIXME: is using the freestream velocity at the central chord okay?
        V_mid = V_w2cp[self.K // 2]
        u_inf = V_mid / np.linalg.norm(V_mid)
        v = self._induced_velocities(u_inf)  # axes = (inducer, inducee)

        # Common arguments for the root-finding functions
        args = (V_w2cp, v, delta)

        # First, try a fast, gradient-based method. This will fail when wing
        # sections enter the stall region (where Cl_alpha goes to zero).
        # res = self.naive_root(Gamma0, args, rtol=1e-4)
        res = scipy.optimize.root(self._f, Gamma0, args, jac=self._J, tol=1e-4)

        # If the gradient method failed, try fixed-point iterations
        # if not res["success"]:
        #     print("The gradient method failed, using fixed-point iteration")
        #     res = self._fixed_point_circulation(Gamma0, *args, **options)

        if res["success"]:
            Gamma = res["x"]
        else:
            print("Phillips> failed to solve for Gamma")
            Gamma = np.full_like(Gamma0, np.nan)
            embed()

        # print("Phillips> finished _solve_circulation\n")
        # embed()

        return Gamma, v

    def __call__(self, V_rel, delta, initial_Gamma=None):
        V_rel = np.broadcast_to(V_rel, (self.K, 3))

        V_w2cp = -V_rel
        Gamma, v = self._solve_circulation(V_w2cp, delta, initial_Gamma)
        V, V_n, V_a, alpha = self._local_velocities(V_w2cp, Gamma, v)
        dF_inviscid = Gamma * cross3(V, self.dl).T

        # Nominal airfoil drag plus some extra hacks from PFD p63 (71)
        #  0. Nominal airfoil drag
        #  1. Additional drag from the air intakes
        #  2. Additional drag from "surface characteristics"
        # FIXME: these extra terms have not been verified. The air intake
        #        term in particular, which is for ram-air parachutes.
        # FIXME: these extra terms should be in the AirfoilCoefficients
        Cd = self.parafoil.airfoil.coefficients.Cd(alpha, delta)
        # Cd += 0.07 * self.parafoil.airfoil.geometry.thickness(0.03)
        Cd += 0.004

        V2 = (V ** 2).sum(axis=1)
        u_drag = V.T / np.sqrt(V2)
        dF_viscous = 1 / 2 * V2 * self.dA * Cd * u_drag  # FIXME: `dA` or `c_avg`? Hunsaker says the "characteristic chord"

        dF = dF_inviscid + dF_viscous

        # Compute the local pitching moments applied to each section
        #  * ref: Hunsaker-Snyder Eq:19
        #  * ref: Phillips Eq:28
        # FIXME: This is a hack! Should use integral(c**2), not `dA * c_avg`
        Cm = self.parafoil.airfoil.coefficients.Cm(alpha, delta)
        dM = -1 / 2 * V2 * self.dA * self.c_avg * Cm * self.u_s.T

        return dF.T, dM.T, Gamma


if __name__ == "__main__":
    import Airfoil
    import plots

    # Build an example wing: a Niviuk Hook 3, size 23

    # True technical specs
    chord_min, chord_max, chord_mean = 0.52, 2.58, 2.06
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat / AR_flat
    S, b, AR = 19.55, 8.84, 4.00

    # Build an approximate version. Anything not directly from the technical
    # specs is a guess to make the projected values match up.
    c_root = chord_max / (b_flat / 2)  # Proportional values
    c_tip = chord_min / (b_flat / 2)
    parafoil = ParafoilGeometry(
        x=0,
        r_x=0.75,
        yz=elliptical_lobe(mean_anhedral=32, max_anhedral_rate=75),
        r_yz=1.00,
        torsion=PolynomialTorsion(start=0.8, peak=4, exponent=2),
        airfoil=Airfoil.Airfoil(coefficients=None, geometry=Airfoil.NACA(23015)),
        chord_length=elliptical_chord(root=c_root, tip=c_tip),
        intakes=SimpleIntakes(0.85, -0.04, -0.09),
        b_flat=b_flat,  # Option 1: Determine the scale using the planform
        # b=b,  # Option 2: Determine the scale using the lobe
    )

    # FIXME: add `S`, `b`, etc to the new ParafoilGeometry, then verify this
    #        approximation.

    plots.plot_parafoil_geo(parafoil, N_sections=131)

    embed()
