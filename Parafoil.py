import abc

import numpy as np
from numpy import sin, cos, arctan2, dot, cross, einsum
from numpy import arcsin, arctan, deg2rad, sqrt, tan
from numpy.linalg import norm
from numba import njit
from scipy.interpolate import UnivariateSpline  # FIXME: use a Polynomial?
from scipy.integrate import simps

from IPython import embed

from util import cross3

import matplotlib.pyplot as plt


class ParafoilGeometry:
    def __init__(self, planform, lobe, airfoil):
        self.planform = planform
        self.lobe = lobe
        self.airfoil = airfoil
        self.span_factor = planform.b_flat / lobe.span_ratio  # FIXME: API?

    @property
    def b(self):
        """Span of the inflated wing"""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated span? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        return 2 * self.fy(1)

    @property
    def S(self):
        """Projected surface area of the inflated wing"""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated wing? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        s = np.linspace(-1, 1, 5000)
        return simps(self.planform.fc(s), self.fy(s))

    @property
    def AR(self):
        """Aspect ratio of the inflated wing"""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated wing? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        return self.b**2 / self.S

    @property
    def flattening_ratio(self):
        """Percent reduction in area of the inflated wing vs the flat wing"""
        # ref: PFD p47 (54)
        return (1 - self.S/self.planform.S)*100

    def fx(self, s):
        """Section quarter-chord x coordinate"""
        # If the wing curvature defined by the lobe is strictly limited to the
        # yz plane, then the x-coordinate of the quarter chord is the same for
        # the 3D wing as for the planform, regardless of the lobe geometry.
        return self.planform.fx(s)

    def fy(self, s, lobe_args={}):
        """Section quarter-chord y coordinate"""
        return self.span_factor * self.lobe.fy(s, **lobe_args)

    def fz(self, s, lobe_args={}):
        """Section quarter-chord z coordinate"""
        return self.span_factor * self.lobe.fz(s, **lobe_args)

    def c0(self, s, lobe_args={}):
        """Section leading-edge coordinates

        Useful for points given in the local section coordinate system, such
        as for surface curves or the airfoil centroid.
        """
        u = self.section_orientation(s, lobe_args).T
        c = self.planform.fc(s)
        return self.c4(s, lobe_args) + (c/4 * u[0]).T

    def c4(self, s, lobe_args={}):
        """Section quarter-chord coordinates

        Useful as the point of application for section forces (if you assume
        the aerodynamic center of that section lies on the quarter chord).
        """
        x = self.fx(s)
        y = self.fy(s, lobe_args)
        z = self.fz(s, lobe_args)
        return np.array([x, y, z]).T

    def section_orientation(self, s, lobe_args={}):
        """Section orientation unit vectors

        Axes are defined as chordwise, section orthogonal, and vertical. This
        corresponds to the <x, y, z> unit vectors first being transformed by
        the planform geometric torsion, then by the lobe dihedral.
        """
        # FIXME: finish documenting
        # FIXME: this assumes the planform chords are all in the xz plane, and
        #        thus the only transformation is the torsion. Is that correct?
        torsion = self.planform.orientation(s)
        dihedral = self.lobe.orientation(s, **lobe_args)
        return dihedral @ torsion  # (N,3,3) ndarray, column unit vectors

    def upper_surface(self, s, N=50):
        """Airfoil upper surface curve on the 3D parafoil

        Parameters
        ----------
        s : float
            Normalized span position, where `-1 <= s <= 1`
        N : integer, optional
            The number of sample points along the chord. Default: 50

        Returns
        -------
        FIXME
        """
        # FIXME: support `s` broadcasting?
        if not np.isscalar(s) or np.abs(s) > 1:
            raise ValueError("`s` must be a scalar between -1..1")
        if not isinstance(N, int) or N < 1:
            raise ValueError("`N` must be a positive integer")

        xa = np.linspace(0, 1, N)
        upper = self.airfoil.geometry.upper_curve(xa).T  # Unscaled airfoil
        upper = np.array([-upper[0], np.zeros(N), -upper[1]])
        surface = self.section_orientation(s) @ upper * self.planform.fc(s)
        return surface.T + self.c0(s)

    def lower_surface(self, s, N=50):
        """Airfoil upper surface curve on the 3D parafoil

        Parameters
        ----------
        s : float
            Normalized span position, where `-1 <= s <= 1`
        N : integer, optional
            The number of sample points along the chord. Default: 50

        Returns
        -------
        FIXME
        """
        # FIXME: support `s` broadcasting?
        if not np.isscalar(s) or np.abs(s) > 1:
            raise ValueError("`s` must be a scalar between -1..1")
        if not isinstance(N, int) or N < 1:
            raise ValueError("`N` must be a positive integer")

        xa = np.linspace(0, 1, N)
        lower = self.airfoil.geometry.lower_curve(xa).T  # Unscaled airfoil
        lower = np.array([-lower[0], np.zeros(N), -lower[1]])
        surface = self.section_orientation(s) @ lower * self.planform.fc(s)
        return surface.T + self.c0(s)

    def mass_properties(self, lobe_args={}, N=250):
        """
        FIXME: docstring

        Breaks the wing into N sections, and assumes a constant airfoil and
        chord for each section. The airfoil for each section is extruded along
        the span using the perpendicular axis theorem, then oriented into body
        coordinates, and translated to the global centroid using the parallel
        axis theorem.

        The parafoil is treated as three separate components, so the sections
        are composites of three groups: the upper surface, the internal volume,
        and the lower surface.
        """
        s_nodes = np.cos(np.linspace(np.pi, 0, N+1))
        s_mid_nodes = (s_nodes[1:] + s_nodes[:-1])/2  # Section midpoints
        nodes = self.c4(s_nodes, lobe_args)  # Section endpoints
        geo = self.airfoil.geometry
        node_chords = self.planform.fc(s_nodes)
        chords = (node_chords[1:] + node_chords[:-1])/2  # Just a dumb average
        T = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])  # ACS -> FRD
        u = self.section_orientation(s_mid_nodes, lobe_args)
        u_inv = np.linalg.inv(u)

        # Section centroids
        airfoil_centroids = np.array([
            [*geo.upper_centroid, 0],
            [*geo.area_centroid, 0],
            [*geo.lower_centroid, 0]])
        section_origins = self.c0(s_mid_nodes)
        section_upper_cm, section_volume_cm, section_lower_cm = \
            (einsum('K,Kij,jk,Gk->GKi', chords, u, T, airfoil_centroids) +
             section_origins[None, ...])

        # Scaling factors for converting 2d airfoils into 3d sections
        # Approximates each section chord area as parallelograms
        u_a = u.T[0].T  # The chordwise ("aerodynamic") unit vectors
        dl = nodes[1:] - nodes[:-1]
        section_chord_area = np.linalg.norm(np.cross(u_a, dl), axis=1)
        Kl = chords * section_chord_area  # surface lines into surface area
        Ka = chords**2 * section_chord_area  # airfoil area into section volume

        section_upper_area = Kl * geo.upper_length
        section_volume = Ka * geo.area
        section_lower_area = Kl * geo.lower_length

        # Surface areas (upper and lower) and the internal volume
        upper_area = section_upper_area.sum()
        volume = section_volume.sum()
        lower_area = section_lower_area.sum()

        # The upper/volume/lower centroids
        upper_centroid = (section_upper_area * section_upper_cm.T).T.sum(axis=0) / upper_area
        volume_centroid = (section_volume * section_volume_cm.T).T.sum(axis=0) / volume
        lower_centroid = (section_lower_area * section_lower_cm.T).T.sum(axis=0) / lower_area

        # Section inertia tensors in body FRD coordinates
        Kl, Ka = Kl.reshape(-1, 1, 1), Ka.reshape(-1, 1, 1)
        section_upper_J = u_inv @ T @ (Kl * geo.upper_inertia) @ T @ u
        section_volume_J = u_inv @ T @ (Ka * geo.area_inertia) @ T @ u
        section_lower_J = u_inv @ T @ (Kl * geo.lower_inertia) @ T @ u

        # Parallel axis distances of each section
        Ru = upper_centroid - section_upper_cm
        Rv = volume_centroid - section_volume_cm
        Rl = lower_centroid - section_lower_cm

        # Section distances to the group centroids
        R = np.array([Ru, Rv, Rl])
        D = (np.einsum('Rij,Rij->Ri', R, R)[..., None, None] * np.eye(3) -
             np.einsum('Rki,Rkj->Rkij', R, R))
        Du, Dv, Dl = D

        # And finally, apply the parallel axis theorem
        upper_J = (section_upper_J + (section_upper_area * Du.T).T).sum(axis=0)
        volume_J = (section_volume_J + (section_volume * Dv.T).T).sum(axis=0)
        lower_J = (section_lower_J + (section_lower_area * Dl.T).T).sum(axis=0)

        mass_properties = {
            'upper_area': upper_area,
            'upper_centroid': upper_centroid,
            'upper_inertia': upper_J,
            'volume': volume,
            'volume_centroid': volume_centroid,
            'volume_inertia': volume_J,
            'lower_area': lower_area,
            'lower_centroid': lower_centroid,
            'lower_inertia': lower_J}

        return mass_properties

# ---------------------------------------------------------------------------


class ParafoilPlanform(abc.ABC):
    """
    Define the planform geometry of a flattened (non-inflated) parafoil.

    Note: this contradicts the common definition of "planform" as the projected
    area of a 3D wing, not a flattened wing. This mild abuse of terminology is
    reasonable because the projected area of an inflated parafoil is not
    particularly useful, and this redefinition avoids prefixing "flattened" to
    the geometries.
    """

    @property
    def b(self):
        """Flattened wing span"""
        return 2*self.fy(1)

    @property
    def MAC(self):
        """Mean aerodynamic chord"""
        # Subclasses can redefine this brute method with an analytical solution
        s = np.linspace(0, 1, 5000)
        return (2 / self.S) * simps(self.fc(s)**2, self.fy(s))

    @property
    def SMC(self):
        """Standard mean chord"""
        # Subclasses can redefine this brute method with an analytical solution
        s = np.linspace(-1, 1, 5000)
        return np.mean(self.fc(s))

    @property
    def S(self):
        """Flattened wing surface area"""
        return self.SMC * self.b

    @property
    def AR(self):
        """Flattened wing aspect ratio"""
        return self.b / self.SMC

    @abc.abstractmethod
    def fx(self, s):
        """Section quarter chord x coordinate"""

    @abc.abstractmethod
    def fy(self, s):
        """Section quarter chord y coordinate"""

    @abc.abstractmethod
    def fc(self, s):
        """Section chord length"""

    @abc.abstractmethod
    def ftheta(self, s):
        """Section geometric torsion

        That is, the section chord pitch angle relative to the central chord.
        """

    @abc.abstractmethod
    def orientation(self, s):
        """Section orientation unit vectors

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
        self.Ax = (b_flat/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        self.Bx = (b_flat/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        self.Cx = -self.Bx - self.c0/4

        # Ellipse coefficients for the chord lengths
        self.Ac = (b_flat/2) / sqrt(1 - self.taper**2)
        self.Bc = self.c0

    @property
    def MAC(self):
        # ref: PFD Table:3-6, p46 (54)
        tmp = arcsin(sqrt(1 - self.taper**2))/sqrt(1 - self.taper**2)
        return (2/3 * self.c0 * (2 + self.taper**2)) / (self.taper + tmp)

    @property
    def SMC(self):
        # ref: PFD Table:3-6, p46 (54)
        L = 1 - self.taper**2
        return self.c0 * 1/2 * (self.taper + np.arcsin(np.sqrt(L))/np.sqrt(L))

    @property
    def sweep_smoothness(self):
        """A measure of the rate of change in sweep along the span"""
        # ref: PFD p47 (54)
        # FIXME: untested
        sMax, min_sMax = abs(self.sweepMax), abs(2 * self.sweepMed)
        ratio = (sMax - min_sMax)/(np.pi/2 - min_sMax)
        return (1 - ratio)*100

    def fx(self, s):
        y = self.b_flat/2 * s
        return self.Bx * np.sqrt(1 - y**2/self.Ax**2) + self.Cx

    def fy(self, s):
        return self.b_flat/2 * s

    def fc(self, s):
        y = self.b_flat/2 * s
        return self.Bc * np.sqrt(1 - y**2/self.Ac**2)

    def ftheta(self, s):
        """Geometric torsion angle"""
        return self.torsion_max * np.abs(s)**self.torsion_exponent

    def orientation(self, s):
        theta = self.ftheta(s)
        ct, st = np.cos(theta), np.sin(theta)
        _0, _1 = np.zeros_like(s), np.ones_like(s)  # FIXME: broadcasting hack
        torsion = np.array([
            [ct,  _0, st],
            [_0,  _1, _0],
            [-st, _0, ct]])

        # Rearrange into a (K,3,3) if necessary
        if torsion.ndim == 3:  # `s` was an array_like
            torsion = np.moveaxis(torsion, [0, 1, 2], [1, 2, 0])
        return torsion

    def _dfxdy(self, s):
        y = self.b_flat/2 * s
        return -y * (self.Bx/self.Ax) / np.sqrt(self.Ax**2 - y**2)

    def Lambda(self, s):
        """Sweep angle"""
        # FIXME: should this be part of the ParafoilGeometry interface?
        return arctan(self._dfxdy(s))

    @staticmethod
    def SMC_to_c0(SMC, taper):
        """
        Compute the central chord of a tapered elliptical wing from the SMC.

        This elliptical geometry is parametrized by the central chord, but the
        SMC is more commonly known."""
        L = 1 - taper**2
        return SMC / (1/2 * (taper + np.arcsin(np.sqrt(L))/np.sqrt(L)))

    @staticmethod
    def MAC_to_c0(MAC, taper):
        """
        Compute the central chord of a tapered elliptical wing from the MAC.

        If the MAC and taper of a wing are known, then this function can be
        used to determine the equivalent central chord for that wing.
        """
        # PFD Table:3-6, p54
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        c0 = (MAC / (2/3) / (2 + taper**2)) * (taper + tmp)
        return c0


# ---------------------------------------------------------------------------


class ParafoilLobe:
    """
    FIXME: document

    In particular, note that this is a proportional geometry: the span of the
    lobes are defined as `b=1` to simplify the conversion between the flat and
    projected spans.
    """
    @property
    def span_ratio(self):
        """Ratio of the planform span to the projected span"""
        # FIXME: should this be cached?
        # FIXME: is this always the nominal value, or does it deform?
        N = 500
        s = np.linspace(-1, 1, N)
        points = np.vstack([self.fy(s), self.fz(s)])
        L = np.sum(np.linalg.norm(points[:, :-1] - points[:, 1:], axis=0))
        return L  # The ellipse line length = b_flat/b_projected

    @abc.abstractmethod
    def fy(self, s):
        """FIXME: docstring"""

    @abc.abstractmethod
    def fz(self, s):
        """FIXME: docstring"""

    @abc.abstractmethod
    def orientation(self, s):
        """Section orientation unit vectors

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
        # FIXME: needs clearer documentation, and 'span_ratio' is not defined
        #        correctly
        tMed = tan(self.dihedralMed)
        tMax = tan(self.dihedralMax)
        b = 1  # Explicitly highlight that this class assumes a unit span
        self.Az = (b/2) * (1 - tMed/tMax) / sqrt(1 - 2*tMed/tMax)
        self.Bz = (b/2) * tMed * (1-tMed/tMax)/(1 - 2*tMed/tMax)
        self.Cz = -self.Bz

        # The span is parametrized by the normalized span position `s`, but the
        # ellipse is parametrized by the angle `t`, so a change of variables
        # transformation is needed. There isn't a closed form solution for the
        # arc length of an ellipse (which is essentially `s`), so pre-compute
        # the mapping and fit it with a spline.
        t_min_z = np.arccos(b/(2*self.Az))  # t_min <= t <= (np.pi-t_min)
        t = np.linspace(np.pi - t_min_z, t_min_z, 500)
        p = np.vstack((self.Az*np.cos(t), self.Bz*np.sin(t) + self.Cz))
        s = np.r_[0, np.cumsum(np.linalg.norm(p[:, 1:] - p[:, :-1], axis=0))]
        s = (s - s[-1]/2) / (s[-1]/2)  # Normalized span coordinates for `t`
        self.s2t = UnivariateSpline(s, t, k=5)

    def fy(self, s):
        t = self.s2t(s)
        return self.Az * np.cos(t)

    def fz(self, s):
        t = self.s2t(s)
        return self.Bz * np.sin(t) + self.Cz

    def orientation(self, s):
        t = self.s2t(s)
        dydt = -self.Az * np.sin(t)
        dzdt = self.Bz * np.cos(t)

        # Normalize the derivatives into unit vectors, and negate to orient
        # them with increasing `s` instead of increasing `t`
        K = np.sqrt(dydt**2 + dzdt**2)  # Faster version of 1d L2-norm
        dydt, dzdt = -dydt/K, -dzdt/K

        _0, _1 = np.zeros_like(s), np.ones_like(s)  # FIXME: broadcasting hack
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
        return -self.Bz/self.Az/np.tan(t)

    def Gamma(self, s):
        return arctan(self._dfzdy(s))

    @property
    def dihedral_smoothness(self):
        """A measure of the rate of change in curvature along the span"""
        # ref: PFD p47 (54)
        # FIXME: untested
        dMax, min_dMax = abs(self.dihedralMax), abs(2 * self.dihedralMed)
        ratio = (dMax - min_dMax)/(np.pi/2 - min_dMax)
        return (1 - ratio)*100


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
    A numerical lifting-line method that uses a set of spanwise bound vortices
    instead of a single, uniform lifting line. Unlike the Prandtl's classic
    lifting-line theory, this method allows for wing sweep and dihedral.

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
    is suitable for parafoil,s but for wings with left and right segments
    separated by some discontinuity at the root you should distribute the
    points across each semispan independently. See _[1].

    This method does suffer an issue where induced velocity goes to infinity as
    the segment lengths tend toward zero (as the number of segments increases,
    or for a poorly chosen point distribution). See _[2], section 8.2.3.
    """

    def __init__(self, parafoil, lobe_args={}):
        self.parafoil = parafoil
        self.lobe_args = lobe_args

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution
        # self.K = 31  # The number of bound vortex segments
        # self.s_nodes = np.linspace(-1, 1, self.K+1)

        # Option 2: cosine distribution
        self.K = 15  # The number of bound vortex segments
        self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.c4(self.s_nodes)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1])/2
        self.cps = self.parafoil.c4(self.s_cps)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        u = self.parafoil.section_orientation(self.s_cps, lobe_args).T
        self.u_a, self.u_s, self.u_n = u[0].T, u[1].T, u[2].T

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        self.c_avg = (node_chords[1:] + node_chords[:-1])/2
        self.dA = self.c_avg * norm(cross(self.u_a, self.dl), axis=1)

        # --------------------------------------------------------------------
        # For debugging purposes: plot the quarter chord line, and segments
        plotit = False
        # plotit = True
        if plotit:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection='3d')
            ax.view_init(azim=-130, elev=25)

            # Plot the actual quarter chord
            # y = np.linspace(-b/2, b/2, 51)
            # t = np.linspace(-1, 1, 51)
            # ax.plot(self.parafoil.geometry.fx(t),
            #         self.parafoil.geometry.fy(t),
            #         -self.parafoil.geometry.fz(t), 'g--', lw=0.8)

            # Plot the segments and their nodes
            # ax.plot(self.nodes[:, 0], self.nodes[:, 1], -self.nodes[:, 2], marker='.')

            # Plot the dl segments
            segments = self.dl + self.nodes[:-1]  # Add their starting points
            ax.plot(segments[:, 0], segments[:, 1], -segments[:, 2], marker='.')

            # Plot the cps
            ax.scatter(self.cps[:, 0], self.cps[:, 1], -self.cps[:, 2], marker='x')

            set_axes_equal(ax)
            plt.show()
        self.f = None  # FIXME: design review Numba helper functions

    @property
    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def ORIG_induced_velocities(self, u_inf):
        #  * ref: Phillips, Eq:6
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2
        v = np.empty_like(R1)

        indices = [(i, j) for i in range(self.K) for j in range(self.K)]
        for ij in indices:
            v[ij] = cross(u_inf, R2[ij]) / \
                (r2[ij] * (r2[ij] - dot(u_inf, R2[ij])))

            v[ij] = v[ij] - cross(u_inf, R1[ij]) / \
                (r1[ij] * (r1[ij] - dot(u_inf, R1[ij])))

            if ij[0] == ij[1]:
                continue  # Skip singularities when `i == j`

            v[ij] = v[ij] + ((r1[ij] + r2[ij]) * cross(R1[ij], R2[ij])) / \
                (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + dot(R1[ij], R2[ij])))

        return v/(4*np.pi)

    def _induced_velocities(self, u_inf):
        #  * ref: Phillips, Eq:6
        # This version uses a Numba helper function
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2
        K = self.K

        if self.f is None:
            def f(u_inf):
                v = np.empty_like(R1)

                indices = [(i, j) for i in range(K) for j in range(K)]
                for ij in indices:
                    v[ij] = cross3(u_inf, R2[ij]) / \
                        (r2[ij] * (r2[ij] - dot(u_inf, R2[ij])))

                    v[ij] = v[ij] - cross3(u_inf, R1[ij]) / \
                        (r1[ij] * (r1[ij] - dot(u_inf, R1[ij])))

                    if ij[0] == ij[1]:
                        continue  # Skip singularities when `i == j`

                    v[ij] = v[ij] + ((r1[ij] + r2[ij]) * cross3(R1[ij], R2[ij])) / \
                        (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + dot(R1[ij], R2[ij])))

                return v/(4*np.pi)

            self.f = njit(f)

        return self.f(u_inf)

    def _vortex_strengths(self, V_rel, delta, max_runs=None):
        """
        FIXME: finish the docstring

        Parameters
        ----------
        V_rel : array of float, shape (K,3) [meters/second]
            Fluid velocity vectors for each section, in body coordinates. This
            is equal to the relative wind "far" from each wing section, which
            is absent of circulation effects.
        delta : array of float, shape (K,) [radians]
            The angle of trailing edge deflection

        Returns
        -------
        Gamma : array of float, shape (K,) [units?]
        V : array of float, shape (K,) [meters/second]

        """

        # FIXME: this implementation fails when wing sections go beyond the
        #        stall condition. In that case, use under-relaxed Picard
        #        iterations.  ref: Hunsaker and Snyder, 2006, pg 5
        # FIXME: find a better initial proposal
        # FIXME: return the induced AoA? Could be interesting

        assert np.shape(V_rel) == (self.K, 3)
        V_mid = V_rel[self.K // 2]

        # FIXME: is using the freestream velocity at the central chord okay?
        u_inf = V_mid / np.linalg.norm(V_mid)

        # 2. Compute the "induced velocity" unit vectors
        v = self._induced_velocities(u_inf)  # axes = (inducer, inducee)
        vT = np.swapaxes(v, 0, 1)  # Useful for broadcasting cross products

        # --------------------------------------------------------------------
        # 3. Propose an initial distribution for Gamma
        # FIXME: this is full of hacks and really needs a thorough review
        b = self.parafoil.b
        cp_y = self.cps[:, 1]

        # Option 1: elliptical using the mid-point velocity only
        Gamma0 = np.linalg.norm(V_mid)
        Gamma = Gamma0 * np.sqrt(1 - ((2*cp_y)/b)**2)

        # Option 2: elliptical using each section velocity
        # Gamma = np.linalg.norm(V_rel, axis=1) * np.sqrt(1 - ((2*cp_y)/b)**2)

        # Next, it helps to scale the basic ellipse based on the section Cl
        CL_2d = self.parafoil.airfoil.coefficients.Cl(
                np.arctan2(V_rel[:, 2], V_rel[:, 0]), delta)
        Gamma *= CL_2d

        # As alpha increases, downwash increases and thus Gamma increases,
        # so the Gamma tends to be higher than predicted by 2D flows
        # FIXME: these are total hacks! They get "close enough", no more.
        alpha_mid = np.arctan2(V_mid[2], V_mid[0])
        if alpha_mid > np.deg2rad(5):
            Gamma += 0.01*(V_mid.sum())  # Avoid extremely negative AoA at the tips
            Gamma *= 1 + (alpha_mid - np.deg2rad(5))/np.deg2rad(20)

        # --------------------------------------------------------------------
        # Iterate the Gamma proposal until convergence

        # Save intermediate values for debugging purposes
        Vs = [V_rel]
        Gammas = [Gamma]
        delta_Gammas = []
        fs = []
        Js = []
        alphas = []
        Cls = []
        Cl_alphas = []

        Omega = 0.5  # FIXME: tuning

        if max_runs is None:
            # max_runs = 5 + int(np.ceil(3*M))
            max_runs = 50  # FIXME: tuning

        # FIXME: add oscillation detection to reduce Omega if necessary
        n_runs = 0
        while n_runs < max_runs:
            # print("run:", n_runs)
            # 4. Compute the local fluid velocities
            #  * ref: Hunsaker-Snyder Eq:5
            #  * ref: Phillips Eq:5 (nondimensional version)
            V = V_rel + einsum('i,ijk->jk', Gamma, v)

            # 5. Compute the section local angle of attack
            #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
            V_a = einsum('ki,ki->k', V, self.u_a)  # Chordwise
            V_n = einsum('ki,ki->k', V, self.u_n)  # Normal-wise
            alpha = arctan2(V_n, V_a)

            min_alpha = min(alpha)
            if np.rad2deg(min_alpha) < -11:
                print("Encountered a very small alpha: {}".format(min_alpha))
                embed()

            # plt.plot(cp_y, np.rad2deg(alpha))
            # plt.ylabel('local section alpha')
            # plt.show()

            # For testing purposes: the global section alpha and induced AoA
            # V_chordwise_2d = einsum('ki,ki->i', V_rel, self.u_a)
            # V_normal_2d = einsum('ki,ki->i', V_rel, self.u_n)
            # alpha_2d = arctan2(V_normal_2d, V_chordwise_2d)
            # alpha_induced = alpha_2d - alpha

            # print("Stopping to investigate the alphas")
            # embed()
            # input('continue?')

            Cl = self.parafoil.airfoil.coefficients.Cl(alpha, delta)

            if np.any(np.isnan(Cl)):
                print("Cl has nan's")
                embed()
                return
                # FIXME: raise a RuntimeWarning?

            # 6. Compute the residual error
            #  * ref: Phillips Eq:15, or Hunsaker-Snyder Eq:8
            W = cross(V, self.dl)
            W_norm = norm(W, axis=1)
            f = 2 * Gamma * W_norm - (V*V).sum(axis=1) * self.dA * Cl

            # 7. Compute the gradient
            #  * ref: Hunsaker-Snyder Eq:11
            Cl_alpha = self.parafoil.airfoil.coefficients.Cl_alpha(alpha, delta)

            # plt.plot(cp_y, Cl_alpha)
            # plt.ylabel('local section Cl_alpha')
            # plt.show()

            # print("Check the Cl_alpha")
            # embed()
            # input('continue?')

            # J is a Jordan matrix, where `J[ij] = d(F_i)/d(Gamma_j)`
            J1 = 2 * np.diag(W_norm)  # terms for i==j
            J2 = 2 * einsum('ik,ijk->ij', W, cross(vT, self.dl))
            J2 = J2 * (Gamma / W_norm)[:, None]
            J3 = (einsum('i,jik,ik->ij', V_a, v, self.u_n) -
                  einsum('i,jik,ik->ij', V_n, v, self.u_a))
            J3 = J3 * ((V*V).sum(axis=1)*self.dA*Cl_alpha)[:, None]
            J3 = J3 / (V_a**2 + V_n**2)[:, None]
            J4 = 2*self.dA*Cl*einsum('ik,jik->ij', V, v)
            J = J1 + J2 - J3 - J4

            # Compute the Gamma update term
            delta_Gamma = np.linalg.solve(J, -f)

            # Use the residual error and gradient to update the Gamma proposal
            Gamma = Gamma + Omega*delta_Gamma

            Vs.append(V)
            alphas.append(alpha)
            delta_Gammas.append(delta_Gamma)
            Gammas.append(Gamma)
            fs.append(f)
            Js.append(J)
            Cls.append(Cl)
            Cl_alphas.append(Cl_alpha)

            # print("finished run", n_runs)
            # embed()
            # 1/0

            # FIXME: ad-hoc workaround to avoid massively negative AoA
            # print("DEBUG> Omega:", Omega)
            Omega += (1 - Omega)/2

            n_runs += 1

            if abs(max(delta_Gamma/Gamma)) < 0.005:
                print(f"Phillips: early termination after {n_runs} runs")
                break

            if n_runs == max_runs:
                print("Phillips: Failed to converge")
                embed()
                break

        # embed()

        # if n_runs < 10:
        #     thinning = 1
        # elif n_runs < 26:
        #     thinning = 2
        # else:
        #     thinning = 5
        # thinning = 1
        # Gammas = Gammas[::thinning]

        # for n, G in enumerate(Gammas):
        #     plt.plot(cp_y, G, marker='.', label=n*thinning)
        # plt.ylabel('Gamma')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return Gamma, V, alpha

    def __call__(self, V_rel, delta):
        Gamma, V, alpha = self._vortex_strengths(V_rel, delta)
        dF_inviscid = Gamma * cross(self.dl, V).T

        # Nominal airfoil drag plus some extra hacks from PFD p63 (71)
        #  0. Nominal airfoil drag
        #  1. Additional drag from the air intakes
        #  2. Additional drag from "surface characteristics"
        # FIXME: these terms have not been verified!!
        Cd = self.parafoil.airfoil.coefficients.Cd(alpha, delta)
        Cd += 0.07 * self.parafoil.airfoil.geometry.thickness(0.03)
        Cd += 0.004

        V2 = (V**2).sum(axis=1)
        u_drag = -V.T/np.sqrt(V2)  # V = V_cp2w = -V_w2cp
        dF_viscous = 1/2 * V2 * Cd * self.dA * u_drag

        dF = dF_inviscid.T + dF_viscous.T

        Cm = self.parafoil.airfoil.coefficients.Cm(alpha, delta)
        Mi = 1/2 * V2 * Cm * self.dA * self.c_avg
        dM = (Mi * self.u_s.T).T  # Pitching moments are about section y-axes

        return dF, dM


class Phillips2D(ForceEstimator):
    """
    This estimator is based on `Phillips` but it uses the 2D section lift
    coefficients directly instead of calculating the bound vorticity. This is
    equivalent to neglecting the induced velocities from other segments.

    See the documentation for `Phillips` for more information.
    """

    def __init__(self, parafoil, lobe_args={}):
        self.parafoil = parafoil
        self.lobe_args = lobe_args

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution
        # self.K = 31  # The number of bound vortex segments
        # self.s_nodes = np.linspace(-1, 1, self.K+1)

        # Option 2: cosine distribution
        self.K = 15  # The number of bound vortex segments
        self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.c4(self.s_nodes)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1])/2
        self.cps = self.parafoil.c4(self.s_cps)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        u = self.parafoil.section_orientation(self.s_cps, lobe_args).T
        self.u_a, self.u_s, self.u_n = u[0].T, u[1].T, u[2].T

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        self.c_avg = (node_chords[1:] + node_chords[:-1])/2
        self.dA = self.c_avg * norm(cross(self.u_a, self.dl), axis=1)

    @property
    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def __call__(self, V_rel, delta):
        assert np.shape(V_rel) == (self.K, 3)

        # FIXME: add pitching moment calculations
        # FIXME: this seems way too complicated; compare to `Phillips`

        # Compute the section local angle of attack
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_a = einsum('ki,ki->k', V_rel, self.u_a)  # Chordwise
        V_n = einsum('ki,ki->k', V_rel, self.u_n)  # Normal-wise
        alpha = arctan2(V_n, V_a)

        CL = self.parafoil.airfoil.coefficients.Cl(alpha, delta)
        CD = self.parafoil.airfoil.coefficients.Cd(alpha, delta)

        dL_hat = cross(self.dl, V_rel)
        dL_hat = dL_hat.T / norm(dL_hat, axis=1)  # Lift unit vectors
        dL = 1/2 * np.sum(V_rel**2, axis=1) * self.dA * CL * dL_hat

        dD_hat = -V_rel.T / norm(V_rel, axis=1)  # Drag unit vectors
        dD = 1/2 * np.sum(V_rel**2, axis=1) * self.dA * CD * dD_hat

        dF = dL.T + dD.T
        dM = 0

        return dF, dM
