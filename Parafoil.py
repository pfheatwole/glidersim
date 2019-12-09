"""FIXME: add docstring."""

import abc

from IPython import embed

import numpy as np
from numpy import arcsin, arctan, cos, deg2rad, sin, tan
from numpy import cross, dot, einsum, linspace, sqrt
from numpy.linalg import norm

import scipy.optimize
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline  # FIXME: use a Polynomial?

from util import cross3


class ParafoilGeometry:
    """FIXME: add docstring."""

    def __init__(self, planform, lobe, airfoil):
        self.planform = planform
        self.lobe = lobe
        self.airfoil = airfoil
        self.span_factor = planform.b_flat / lobe.span_ratio  # FIXME: API?

    @property
    def b(self):
        """Compute the span of the inflated wing."""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated span? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        return 2 * self.span_factor * self.lobe.fy(1)

    @property
    def S(self):
        """Compute the projected surface area of the inflated wing."""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated wing? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        s = linspace(-1, 1, 5000)
        return simps(self.planform.fc(s), self.span_factor * self.lobe.fy(s))

    @property
    def AR(self):
        """Compute the aspect ratio of the inflated wing."""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated wing? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        return self.b ** 2 / self.S

    def chord_xyz(self, s, chord_ratio, lobe_args={}):
        """Compute the coordinates of points on section chords.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Normalized span position, where `-1 <= s <= 1`.
        chord_ratio : float
            Position on the chord, where `chord_ratio = 0` is the leading edge,
            and `chord_ratio = 1` is the trailing edge.
        """
        if np.any(chord_ratio < 0) or np.any(chord_ratio > 1):
            raise ValueError("chord_ratio must be between 0 and 1")

        # Get the xyz-coordinates of the quarter-chords
        x = self.planform.fx(s)
        y = self.span_factor * self.lobe.fy(s, **lobe_args)
        z = self.span_factor * self.lobe.fz(s, **lobe_args)
        xyz = np.array([x, y, z])

        # Shift the points along their local chords to `chord_ratio * c`
        u = self.section_orientation(s, lobe_args)
        c = self.planform.fc(s)
        xyz += (0.25 - chord_ratio) * c * u.T[0]
        return xyz.T

    def section_orientation(self, s, lobe_args={}):
        """Compute section orientation unit vectors.

        The parafoil body <x, y, z> axes follow the "front, right, down"
        convention. To find the orientation of each section, copies of the
        parafoil body basis vectors are translated to each point on the span,
        transformed by the planform geometric torsion at that point (rotating
        them about their y-axis), then transformed again by the lobe dihedral
        at that point (rotating them about their x-axis).

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Normalized span position, where `-1 <= s <= 1`.

        Returns
        -------
        ndarray of float, shape (N, 3, 3)
            Column unit vectors for the orientation of each section.
        """
        torsion = self.planform.orientation(s)
        dihedral = self.lobe.orientation(s, **lobe_args)
        return dihedral @ torsion

    def upper_surface(self, s, N=50):
        """Sample points on the upper surface curve of a parafoil section.

        Parameters
        ----------
        s : float
            Normalized span position, where `-1 <= s <= 1`
        N : integer, optional
            The number of sample points along the chord. Default: 50

        Returns
        -------
        ndarray of float, shape (N, 3)
            A set of points from the upper surface of the airfoil in FRD.

        """
        # FIXME: support `s` broadcasting?
        if not np.isscalar(s) or np.abs(s) > 1:
            raise ValueError("`s` must be a scalar between -1..1")
        if not isinstance(N, int) or N < 1:
            raise ValueError("`N` must be a positive integer")

        sa = linspace(self.airfoil.geometry.s_upper, 1, N)
        upper = self.airfoil.geometry.surface_curve(sa).T  # Unscaled airfoil
        upper = np.array([-upper[0], np.zeros(N), -upper[1]])
        surface = self.section_orientation(s) @ upper * self.planform.fc(s)
        return surface.T + self.chord_xyz(s, 0)

    def lower_surface(self, s, N=50):
        """Sample points on the lower surface curve of a parafoil section.

        Parameters
        ----------
        s : float
            Normalized span position, where `-1 <= s <= 1`
        N : integer, optional
            The number of sample points along the chord. Default: 50

        Returns
        -------
        ndarray of float, shape (N, 3)
            A set of points from the lower surface of the airfoil in FRD.

        """
        # FIXME: support `s` broadcasting?
        if not np.isscalar(s) or np.abs(s) > 1:
            raise ValueError("`s` must be a scalar between -1..1")
        if not isinstance(N, int) or N < 1:
            raise ValueError("`N` must be a positive integer")

        sa = linspace(self.airfoil.geometry.s_lower, -1, N)
        lower = self.airfoil.geometry.surface_curve(sa).T  # Unscaled airfoil
        lower = np.array([-lower[0], np.zeros(N), -lower[1]])
        surface = self.section_orientation(s) @ lower * self.planform.fc(s)
        return surface.T + self.chord_xyz(s, 0)

    def mass_properties(self, lobe_args={}, N=250):
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
        s_nodes = cos(linspace(np.pi, 0, N + 1))
        s_mid_nodes = (s_nodes[1:] + s_nodes[:-1]) / 2  # Segment midpoints
        nodes = self.chord_xyz(s_nodes, 0.25, lobe_args)  # Segment endpoints
        section = self.airfoil.geometry.mass_properties()
        node_chords = self.planform.fc(s_nodes)
        chords = (node_chords[1:] + node_chords[:-1]) / 2  # Dumb average
        T = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])  # ACS -> FRD
        u = self.section_orientation(s_mid_nodes, lobe_args)
        u_inv = np.linalg.inv(u)

        # Segment centroids
        airfoil_centroids = np.array([
            [*section["upper_centroid"], 0],
            [*section["area_centroid"], 0],
            [*section["lower_centroid"], 0]])
        segment_origins = self.chord_xyz(s_mid_nodes, 0, lobe_args)
        segment_upper_cm, segment_volume_cm, segment_lower_cm = (
            einsum("K,Kij,jk,Gk->GKi", chords, u, T, airfoil_centroids)
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
        D = (einsum("Rij,Rij->Ri", R, R)[..., None, None] * np.eye(3)
             - einsum("Rki,Rkj->Rkij", R, R))
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


class ParafoilPlanform(abc.ABC):
    """
    Define the planform geometry of a flattened (non-inflated) parafoil.

    This contradicts the common definition of "planform", which usually refers
    to the projected area of a 3D wing, not a wing that has been flattened to
    remove dihedral (curvature in the yz-plane). This mild abuse of terminology
    is reasonable because the projected area of an inflated parafoil is not
    particularly useful, and this redefinition avoids prefixing "flattened" to
    the geometries.
    """

    @property
    def b(self):
        """Compute the span of the flattened wing."""
        return 2 * self.fy(1)

    @property
    def MAC(self):
        """Compute the mean aerodynamic chord.

        See Also
        --------
        MAC : standard aerodynamic chord
        """
        # Subclasses can override this brute method with an analytical solution
        s = linspace(0, 1, 5000)
        return (2 / self.S) * simps(self.fc(s) ** 2, self.fy(s))

    @property
    def SMC(self):
        """Compute the standard mean chord.

        See Also
        --------
        MAC : mean aerodynamic chord
        """
        # Subclasses can override this brute method with an analytical solution
        s = linspace(-1, 1, 5000)
        return np.mean(self.fc(s))

    @property
    def S(self):
        """Compute the surface area of the flattened wing."""
        return self.SMC * self.b

    @property
    def AR(self):
        """Compute the aspect ratio of the flattened wing."""
        return self.b / self.SMC

    @abc.abstractmethod
    def fx(self, s):
        """Compute the x-coordinate of the section quarter-chord."""

    @abc.abstractmethod
    def fy(self, s):
        """Compute the y-coordinate of the section quarter-chord."""

    @abc.abstractmethod
    def fc(self, s):
        """Compute the chord length of a section."""

    @abc.abstractmethod
    def ftheta(self, s):
        """Compute the geometric torsion of a section.

        That is, the section chord pitch angle relative to the central chord.
        """

    @abc.abstractmethod
    def orientation(self, s):
        """Compute the orientation unit vectors of a section.

        Parameters
        ----------
        s : float, or array_like of float, shape (N,)
            Normalized span position, where `-1 <= s <= 1`

        Returns
        -------
        torsion : ndarray of float, shape (3,3) or (N,3,3)
            The orientation matrices at each section. The columns of each
            matrix are the transformed <x,y,z> unit vectors.
        """


class EllipticalPlanform(ParafoilPlanform):
    """
    A planform that uses ellipses for the sweep and chord lengths.

    ref: PFD p43 (51)
    """

    def __init__(self, b_flat, c0, taper, sweepMed, sweepMax,
                 torsion_exponent=5, torsion_max=0):
        self.b_flat = b_flat
        self.c0 = c0
        self.taper = taper
        self.sweepMed = deg2rad(sweepMed)
        self.sweepMax = deg2rad(sweepMax)
        self.torsion_exponent = torsion_exponent
        self.torsion_max = np.deg2rad(torsion_max)

        if torsion_exponent < 1:
            raise ValueError("torsion_exponent must be >= 1")

        # Ellipse coefficients for quarter-chord projected on the xy plane
        tMed = tan(self.sweepMed)
        tMax = tan(self.sweepMax)
        self.Ax = (b_flat / 2) * (1 - tMed / tMax) / sqrt(1 - 2 * tMed / tMax)
        self.Bx = (b_flat / 2) * tMed * (1 - tMed / tMax) / (1 - 2 * tMed / tMax)
        self.Cx = -self.Bx - self.c0 / 4

        # Ellipse coefficients for the chord lengths
        self.Ac = (b_flat / 2) / sqrt(1 - self.taper ** 2)
        self.Bc = self.c0

    @property
    def MAC(self):
        # ref: PFD Table:3-6, p46 (54)
        tmp = arcsin(sqrt(1 - self.taper ** 2)) / sqrt(1 - self.taper ** 2)
        return (2 / 3 * self.c0 * (2 + self.taper ** 2)) / (self.taper + tmp)

    @property
    def SMC(self):
        # ref: PFD Table:3-6, p46 (54)
        L = 1 - self.taper ** 2
        return self.c0 / 2 * (self.taper + np.arcsin(np.sqrt(L)) / np.sqrt(L))

    @property
    def sweep_smoothness(self):
        """Compute the rate of change in sweep along the span."""
        # ref: PFD p47 (54)
        # FIXME: untested
        # FIXME: misnamed? Taken directly from PFD
        # FIXME: how does this differ from `Lambda`?
        sMax, min_sMax = abs(self.sweepMax), abs(2 * self.sweepMed)
        ratio = (sMax - min_sMax) / (np.pi / 2 - min_sMax)
        return (1 - ratio) * 100

    def fx(self, s):
        y = self.b_flat / 2 * s
        return self.Bx * np.sqrt(1 - y ** 2 / self.Ax ** 2) + self.Cx

    def fy(self, s):
        return self.b_flat / 2 * s

    def fc(self, s):
        y = self.b_flat / 2 * s
        return self.Bc * np.sqrt(1 - y ** 2 / self.Ac ** 2)

    def ftheta(self, s):
        return self.torsion_max * np.abs(s)**self.torsion_exponent

    def orientation(self, s):
        theta = self.ftheta(s)
        ct, st = cos(theta), sin(theta)
        _0, _1 = np.zeros_like(s), np.ones_like(s)
        torsion = np.array([
            [ct,  _0, st],
            [_0,  _1, _0],
            [-st, _0, ct]])

        # Rearrange into a (K,3,3) if necessary
        if torsion.ndim == 3:  # `s` was an array_like
            torsion = np.moveaxis(torsion, [0, 1, 2], [1, 2, 0])
        return torsion

    def _dfxdy(self, s):
        y = self.b_flat / 2 * s
        return -y * (self.Bx / self.Ax) / np.sqrt(self.Ax ** 2 - y ** 2)

    def Lambda(self, s):
        """Compute the sweep angle rate of change of a section."""
        # FIXME: should this be part of the ParafoilGeometry interface?
        return arctan(self._dfxdy(s))

    @staticmethod
    def SMC_to_c0(SMC, taper):
        """
        Compute the central chord of a tapered elliptical wing from the SMC.

        This elliptical geometry is parametrized by the central chord, but the
        SMC is more commonly known."""
        L = 1 - taper ** 2
        return SMC / (1 / 2 * (taper + np.arcsin(np.sqrt(L)) / np.sqrt(L)))

    @staticmethod
    def MAC_to_c0(MAC, taper):
        """
        Compute the central chord of a tapered elliptical wing from the MAC.

        If the MAC and taper of a wing are known, then this function can be
        used to determine the equivalent central chord for that wing.
        """
        # PFD Table:3-6, p54
        tmp = arcsin(sqrt(1 - taper ** 2)) / sqrt(1 - taper ** 2)
        c0 = (MAC / (2 / 3) / (2 + taper ** 2)) * (taper + tmp)
        return c0


# ---------------------------------------------------------------------------


class ParafoilLobe:
    """
    FIXME: document.

    In particular, note that this is a proportional geometry: the span of the
    lobes are defined as `b=1` to simplify the conversion between the flat and
    projected spans.
    """

    @property
    def span_ratio(self):
        """Compute the ratio of the planform span to the projected span."""
        # FIXME: should this be cached?
        # FIXME: is this always the nominal value, or does it deform?
        N = 500
        s = linspace(-1, 1, N)
        points = np.vstack([self.fy(s), self.fz(s)])
        L = np.sum(np.linalg.norm(points[:, :-1] - points[:, 1:], axis=0))
        return L  # The ellipse line length = b_flat/b_projected

    @abc.abstractmethod
    def fy(self, s):
        """FIXME: docstring."""

    @abc.abstractmethod
    def fz(self, s):
        """FIXME: docstring."""

    @abc.abstractmethod
    def orientation(self, s):
        """Compute the orientation unit vectors of a section.

        Parameters
        ----------
        s : float, or array_like of float, shape (N,)
            Normalized span position, where `-1 <= s <= 1`

        Returns
        -------
        dihedral : ndarray of float, shape (3,3) or (N,3,3)
            The orientation matrices at each section. The columns of each
            matrix are the transformed <x,y,z> unit vectors.
        """

    @abc.abstractmethod
    def Gamma(self, s):
        """Dihedral angle"""


class EllipticalLobe(ParafoilLobe):
    """
    A parafoil lobe that uses an ellipse for the arc dihedral.
    """

    def __init__(self, dihedralMed, dihedralMax):
        self.dihedralMed = deg2rad(dihedralMed)
        self.dihedralMax = deg2rad(dihedralMax)

        # Ellipse coefficients for quarter-chord projected on the yz plane
        # This ellipse will be proportional to the true ellipse by a scaling
        # factor:  true_ellipse = (b_flat/L) * this_ellipse
        #
        # FIXME: needs clearer documentation, and "span_ratio" is not defined
        #        correctly
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)
        b = 1  # Explicitly highlight that this class assumes a unit span
        self.Az = (b / 2) * (1 - tMed / tMax) / sqrt(1 - 2 * tMed / tMax)
        self.Bz = (b / 2) * tMed * (1 - tMed / tMax) / (1 - 2 * tMed / tMax)
        self.Cz = -self.Bz

        # The span is parametrized by the normalized span position `s`, but the
        # ellipse is parametrized by the angle `t`, so a change of variables
        # transformation is needed. There isn't a closed form solution for the
        # arc length of an ellipse (which is essentially `s`), so pre-compute
        # the mapping and fit it with a spline.
        t_min_z = np.arccos(b / (2 * self.Az))  # t_min <= t <= (np.pi-t_min)
        t = linspace(np.pi - t_min_z, t_min_z, 500)
        p = np.vstack((self.Az * cos(t), self.Bz * sin(t) + self.Cz))
        s = np.r_[0, np.cumsum(np.linalg.norm(p[:, 1:] - p[:, :-1], axis=0))]
        s = (s - s[-1] / 2) / (s[-1] / 2)  # Normalized span coordinates for `t`
        self.s2t = UnivariateSpline(s, t, k=5)

    def fy(self, s):
        t = self.s2t(s)
        return self.Az * cos(t)

    def fz(self, s):
        t = self.s2t(s)
        return self.Bz * sin(t) + self.Cz

    def orientation(self, s):
        t = self.s2t(s)
        dydt = -self.Az * sin(t)
        dzdt = self.Bz * cos(t)

        # Normalize the derivatives into unit vectors, and negate to orient
        # them with increasing `s` instead of increasing `t`
        K = np.sqrt(dydt ** 2 + dzdt ** 2)  # Faster version of 1d L2-norm
        dydt, dzdt = -dydt / K, -dzdt / K

        _0, _1 = np.zeros_like(s), np.ones_like(s)
        dihedral = np.array([
            [_1,   _0,    _0],
            [_0, dydt, -dzdt],
            [_0, dzdt,  dydt]])

        # Rearrange into a (K,3,3) if necessary
        if dihedral.ndim == 3:  # `s` was an array_like
            dihedral = np.moveaxis(dihedral, [0, 1, 2], [1, 2, 0])
        return dihedral

    def _dfzdy(self, s):
        t = self.s2t(s)
        return -self.Bz / self.Az / np.tan(t)

    def Gamma(self, s):
        return arctan(self._dfzdy(s))

    @property
    def dihedral_smoothness(self):
        """Measure the curvature rate of change along the span."""
        # ref: PFD p47 (54)
        # FIXME: untested
        # FIXME: unused? What's the purpose?
        dMax, min_dMax = abs(self.dihedralMax), abs(2 * self.dihedralMed)
        ratio = (dMax - min_dMax) / (np.pi / 2 - min_dMax)
        return (1 - ratio) * 100


class DeformingLobe:
    """
    Deforms a lobe it by rotating a central section.

    The wing is split into three sections by two points equidistant from the
    central chord. The central section rotates about the central chord based on
    the vertical displacement of the right riser, `deltaZR`.

    This is intended as a crude approximation of how the parafoil deforms when
    a pilot is applying weight shift.
    """

    def __init__(self, lobe, central_width):
        """

        Parameters
        ----------
        lobe : ParafoilLobe
            The nominal (non-deformed) lobe
        central_width : float
            The width of the central section that will rotate.


            FIXME: should this automatically scale the units for the user?
            Lobes don't know the planform span, and so are proportional scaling
            values. This seems frustrating for the user: they'd probably like
            to just say "the central section is 1m wide", not "the central
            section is 0.0772 units". The lobe already uses a scaling factor,
            but I don't think that has enough information to scale this
            unitless central_width into meters. More to the point, I don't see
            how you can do this without knowing the planform span.
             * What's the formula for planform span -> `central_width` here?
             * Should central_width be a percentage of the lobe "span"?
        """
        self.lobe = lobe
        self.w = central_width
        self.p = central_width / 2  # Two points at +/- central_width/2

    def fy(self, s, deltaZR):
        # deltaY = np.sqrt(self.w**2 - deltaZR**2) - 2*w
        # FIXME: implement
        return self.lobe.fy(s)

    def fz(self, s, deltaZR):
        # FIXME: implement
        return self.lobe.fz(s)

    def Gamma(self, s, deltaZR):
        # FIXME: implement
        return self.lobe.Gamma(s)

# ---------------------------------------------------------------------------

# Note sure where this piece goes: it's a utility function for computing the
# span of an EllipticalPlanform+EllipticalLobe, given the c0+AR+taper. It
# made sense when there was a single EllipticalGeometry, but now the planform
# and lobe are separate.
#
# This seems like a more general issue of design helper functions. You specify
# a planform+lobe, and different parameters and it tells you the others.
#
#   @staticmethod
#   def AR_to_b(c0, AR, taper):
#       """Compute the span of a tapered elliptical wing"""
#       # ref: PFD Table:3-6, p46 (54)
#       tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
#       b = (AR / 2)*c0*(taper + tmp)
#       return b


# ---------------------------------------------------------------------------


class ForceEstimator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, V_rel, delta):
        """Estimate the forces and moments on a Parafoil"""

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

    def __init__(self, parafoil, lobe_args={}):
        self.parafoil = parafoil
        self.lobe_args = lobe_args

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution
        self.K = 191  # The number of bound vortex segments
        self.s_nodes = linspace(-1, 1, self.K + 1)

        # Option 2: cosine distribution
        # self.K = 21  # The number of bound vortex segments
        # self.s_nodes = cos(linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.chord_xyz(self.s_nodes, 0.25, lobe_args)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1]) / 2
        self.cps = self.parafoil.chord_xyz(self.s_cps, 0.25, lobe_args)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        # Note: Phillip's derivation uses back-left-up coordinates (not `frd`)
        u = -self.parafoil.section_orientation(self.s_cps, lobe_args).T
        self.u_a, self.u_s, self.u_n = u[0].T, u[1].T, u[2].T

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        self.c_avg = (node_chords[1:] + node_chords[:-1]) / 2
        self.dA = self.c_avg * norm(cross3(self.u_a, self.dl), axis=1)

        # Precompute the `v` terms that do not depend on `u_inf`
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2  # Shorthand
        self.v_ij = np.zeros((self.K, self.K, 3))  # Extra terms when `i != j`
        for ij in [(i, j) for i in range(self.K) for j in range(self.K)]:
            if ij[0] == ij[1]:  # Skip singularities when `i == j`
                continue
            self.v_ij[ij] = ((r1[ij] + r2[ij]) * cross3(R1[ij], R2[ij])) / \
                (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + dot(R1[ij], R2[ij])))

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
        v += cross3(u_inf, R2) / (r2 * (r2 - einsum("k,ijk->ij", u_inf, R2)))[..., None]
        v -= cross3(u_inf, R1) / (r1 * (r1 - einsum("k,ijk->ij", u_inf, R1)))[..., None]

        return v / (4 * np.pi)

    def _proposal1(self, V_w2cp, delta):
        # Generate an elliptical Gamma distribution that uses the wing root
        # velocity as a scaling factor
        cp_y = self.cps[:, 1]
        alpha_2d = np.arctan(V_w2cp[:, 2] / V_w2cp[:, 0])
        CL_2d = self.parafoil.airfoil.coefficients.Cl(alpha_2d, delta)
        mid = self.K // 2
        Gamma0 = np.linalg.norm(V_w2cp[mid]) * self.dA[mid] * CL_2d[mid]
        Gamma = Gamma0 * np.sqrt(1 - ((2 * cp_y) / self.parafoil.b) ** 2)

        return Gamma

    def _local_velocities(self, V_w2cp, Gamma, v):
        # Compute the local fluid velocities
        #  * ref: Hunsaker-Snyder Eq:5
        #  * ref: Phillips Eq:5 (nondimensional version)
        V = V_w2cp + einsum("j,jik->ik", Gamma, v)

        # Compute the local angle of attack for each section
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_n = einsum("ik,ik->i", V, self.u_n)  # Normal-wise
        V_a = einsum("ik,ik->i", V, self.u_a)  # Chordwise
        alpha = arctan(V_n / V_a)

        return V, V_n, V_a, alpha

    def _f(self, Gamma, V_w2cp, v, delta):
        # Compute the residual error vector
        #  * ref: Phillips Eq:14
        #  * ref: Hunsaker-Snyder Eq:8
        V, V_n, V_a, alpha = self._local_velocities(V_w2cp, Gamma, v)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(einsum("ik,ik->i", W, W))
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
        W_norm = np.sqrt(einsum("ik,ik->i", W, W))
        Cl = self.parafoil.airfoil.coefficients.Cl(alpha, delta)
        Cl_alpha = self.parafoil.airfoil.coefficients.Cl_alpha(alpha, delta)

        # Use precomputed optimal einsum paths
        opt2 = ["einsum_path", (0, 2), (0, 2), (0, 1)]
        opt3 = ["einsum_path", (0, 2), (0, 1)]
        opt4 = ["einsum_path", (0, 1), (1, 2), (0, 1)]

        J = 2 * np.diag(W_norm)  # Additional terms for i==j
        J2 = 2 * einsum("i,ik,i,jik->ij", Gamma, W, 1 / W_norm,
                        cross3(v, self.dl), optimize=opt2)
        J3 = (einsum("i,jik,ik->ij", V_a, v, self.u_n, optimize=opt3)
              - einsum("i,jik,ik->ij", V_n, v, self.u_a, optimize=opt3))
        J3 *= (self.dA * Cl_alpha)[:, None]
        J4 = 2 * einsum("i,i,jik,ik->ij", self.dA, Cl, v, V_na, optimize=opt4)
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
            W_norm = norm(cross(V, self.dl), axis=1)
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
            p = UnivariateSpline(self.s_cps, G_old + 0.5 * (G - G_old), s=0.001)
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
            # print("Phillips> failed to solve for Gamma")
            Gamma = np.full_like(Gamma0, np.nan)

        # print("Phillips> finished _solve_circulation\n")
        # embed()

        return Gamma, v

    def __call__(self, V_rel, delta, Gamma=None):
        assert np.shape(V_rel) == (self.K, 3)

        V_w2cp = -V_rel
        Gamma, v = self._solve_circulation(V_w2cp, delta, Gamma)
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
        Cd += 0.07 * self.parafoil.airfoil.geometry.thickness(0.03)
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
