import abc

import numpy as np
from numpy import arcsin, arctan, deg2rad, sqrt, tan
from scipy.interpolate import UnivariateSpline  # FIXME: use a Polynomial?


class ParafoilGeometry:
    def __init__(self, planform, lobe, sections):
        self.planform = planform
        self.lobe = lobe
        self.sections = sections  # Provides the airfoils for each section
        self.span_factor = planform.b_flat / lobe.span_ratio  # FIXME: API?

    @property
    def b(self):
        """Span of the inflated wing"""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated span? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        return self.span_factor * 2 * self.lobe.fy(1)

    @property
    def S(self):
        """Projected surface area of the inflated wing"""
        # FIXME: property, or function? This needs `lobe_args`? Or should this
        #        be the nominal value for the inflated wing? Even if it was the
        #        nominal value, might I still need a `lobe_args`?
        raise NotImplementedError("FIXME: implement")

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
        # FIXME: untested
        return (1 - self.S/self.planform.S)*100

    @property
    def MAC(self):
        """Mean aerodynamic chord"""
        # FIXME: if I make this a property, I'd never be able to pass it a
        #        `planform_args` dictionary. (Unless I define it as the nominal
        #        value of the MAC. I think I'm overthinking this.
        raise NotImplementedError("FIXME: implement")

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

    def c4(self, s, lobe_args={}):
        """Section quarter-chord coordinates"""
        x = self.fx(s)
        y = self.fy(s, lobe_args)
        z = self.fz(s, lobe_args)
        return np.c_[x, y, z]  # FIXME: switch to an array of column vectors?

    def section(self, s):
        # FIXME: needs a design review
        # Does it makes sense that a "section" is a normalized airfoil?
        """Section airfoil"""
        return self.sections(s)

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

    def inertias(self, s, lobe_args={}):
        """
        The three inertia tensors and centroids of the parafoil components:
         1. Upper surface
         2. Volume
         3. Lower surface

        The surfaces should be scaled by the areal densities of the wing
        materials, the volume should be scaled by the volumetric air density.
        """
        # FIXME: document
        raise NotImplementedError("FIXME: implement")


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
        return 2*self.fy(1)

    @property
    def S(self):
        raise NotImplementedError("FIXME: implement")

    @property
    def AR(self):
        raise NotImplementedError("FIXME: implement")

    @property
    def MAC(self):
        raise NotImplementedError("FIXME: implement")

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

        # The span is parametrized by the normalized span position `s`, but the
        # ellipses are parametrized by angles `t`, so change of variable
        # transformations are needed. There isn't a closed form solution for
        # the arc length of an ellipse (which is essentially `s`), so
        # pre-compute the mapping and fit it with a spline.
        t_min_x = np.arccos(b_flat/(2*self.Ax))  # (np.pi-t_min) <= t <= t_min
        t = np.linspace(np.pi - t_min_x, t_min_x, 500)
        p = np.vstack((self.Ax*np.cos(t), self.Bx*np.sin(t) + self.Cx)).T
        s = np.r_[0, np.cumsum(np.linalg.norm(p[1:] - p[:-1], axis=1))]
        s = (s - s[-1]/2) / (s[-1]/2)  # Normalized span coordinates for `t`
        self.s2tx = UnivariateSpline(s, t, k=5)

        t_min_c = np.arccos(b_flat/(2*self.Ac))  # (np.pi-t_min) <= t <= t_min
        t = np.linspace(np.pi - t_min_c, t_min_c, 500)
        p = np.vstack((self.Ac*np.cos(t), self.Bc*np.sin(t))).T
        s = np.r_[0, np.cumsum(np.linalg.norm(p[1:] - p[:-1], axis=1))]
        s = (s - s[-1]/2) / (s[-1]/2)  # Normalized span coordinates for `t`
        self.s2tc = UnivariateSpline(s, t, k=5)

    @property
    def S(self):
        # This is the flat planform area, right?
        # ref: PFD Table:3-6, p46 (54)
        raise NotImplementedError("FIXME: implement")
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return self.c0 * self.b/2 * taper_factor

    @property
    def AR(self):
        """Aspect ratio of the flattened wing"""
        # ref: PFD Table:3-6, p46 (54)
        raise NotImplementedError("FIXME: implement")
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return 2 * self.b / (self.c0*taper_factor)

    @property
    def MAC(self):
        # ref: PFD Table:3-6, p46 (54)
        raise NotImplementedError("FIXME: implement")
        t = self.taper
        taper_factor = t + arcsin(sqrt(1 - t**2))/sqrt(1 - t**2)
        return (2/3) * self.c0 * (2 + t**2) / taper_factor

    @property
    def sweep_smoothness(self):
        """A measure of the rate of change in sweep along the span"""
        # ref: PFD p47 (54)
        # FIXME: untested
        sMax, min_sMax = abs(self.sweepMax), abs(2 * self.sweepMed)
        ratio = (sMax - min_sMax)/(np.pi/2 - min_sMax)
        return (1 - ratio)*100

    def fx(self, s):
        t = self.s2tx(s)
        return self.Bx * np.sin(t) + self.Cx

    def fy(self, s):
        t = self.s2tx(s)
        return self.Ax * np.cos(t)

    def fc(self, s):
        t = self.s2tc(s)
        return self.Bc * np.sin(t)

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

        # Rearrange the axes to allow for section-wise matrix multiplication
        if torsion.ndim == 3:  # `s` was an array_like
            torsion = np.moveaxis(torsion, [0, 1, 2], [1, 2, 0])
        return torsion

    def _dfxdy(self, s):
        # FIXME: untested
        # FIXME: needs a ds/dt factor?
        return -self.Bx/self.Ax/np.tan(self.s2tx(s))

    def Lambda(self, s):
        """Sweep angle"""
        # FIXME: rewrite in terms of dx/ds and dy/ds?
        # FIXME: should this be part of the ParafoilGeometry interface?
        return arctan(self._dfxdy(s))

    @staticmethod
    def MAC_to_c0(MAC, taper):
        """Central chord length of a tapered elliptical wing

        This geometry class is parametrized by the central chord, but the MAC
        is more commonly known. If the MAC and taper of a wing are known, then
        this function can be used to determine the equivalent central chord
        for that wing.
        """
        # ref: PFD Table:3-6, p46 (54)
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
        points = np.c_[self.fy(s), self.fz(s)]
        L = np.sum(np.linalg.norm(points[:-1] - points[1:], axis=1))
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
        t_min_z = np.arccos(b/(2*self.Az))  # (np.pi-t_min) <= t <= t_min
        t = np.linspace(np.pi - t_min_z, t_min_z, 500)
        p = np.vstack((self.Az*np.cos(t), self.Bz*np.sin(t) + self.Cz)).T
        s = np.r_[0, np.cumsum(np.linalg.norm(p[1:] - p[:-1], axis=1))]
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

        # Rearrange the axes to allow for section-wise matrix multiplication
        if dihedral.ndim == 3:  # `s` was an array_like
            dihedral = np.moveaxis(dihedral, [0, 1, 2], [1, 2, 0])
        return dihedral

    def _dfzdy(self, s):
        # FIXME: untested
        # FIXME: needs a ds/dt factor?
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
