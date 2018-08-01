import abc

import numpy as np
from numpy import sin, cos, arctan2, dot, cross, einsum
from numpy import arcsin, arctan, deg2rad, sqrt, tan
from numpy.linalg import norm
from numba import njit
from scipy.interpolate import UnivariateSpline  # FIXME: use a Polynomial?

from IPython import embed

from util import cross3


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

    def upper_surface(self, s, xa=None, N=50):
        """Airfoil upper surface curve on the 3D parafoil

        Parameters
        ----------
        s : float
            Normalized span position, where `-1 <= s <= 1`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord

        Returns
        -------
        FIXME
        """
        # FIXME: support `s` broadcasting?
        if not np.isscalar(s):
            raise ValueError("`s` must be a scalar between -1..1")

        if xa is None:
            xa = np.linspace(0, 1, N)  # FIXME: assume normalized airfoils?

        fc = self.planform.fc(s)
        upper = fc*self.sections.upper_curve(s, xa)  # Scaled airfoil
        upper = np.c_[-upper[:, 0], np.zeros(N), -upper[:, 1]]  # Row vectors
        surface = (self.section_orientation(s) @ upper.T) + self.c4(s).T
        surface[0, :] = surface[0, :] + fc/4
        return surface.T  # Return as row vectors

    def lower_surface(self, s, xa=None, N=50):
        """Airfoil upper surface curve on the 3D parafoil

        Parameters
        ----------
        s : float
            Normalized span position, where `-1 <= s <= 1`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord

        Returns
        -------
        FIXME
        """
        # FIXME: support `s` broadcasting?
        if not np.isscalar(s):
            raise ValueError("`s` must be a scalar between -1..1")

        if xa is None:
            xa = np.linspace(0, 1, N)  # FIXME: assume normalized airfoils?

        fc = self.planform.fc(s)
        lower = fc*self.sections.lower_curve(s, xa)  # Scaled airfoil
        lower = np.c_[-lower[:, 0], np.zeros(N), -lower[:, 1]]  # Row vectors
        surface = (self.section_orientation(s) @ lower.T) + self.c4(s).T
        surface[0, :] = surface[0, :] + fc/4
        return surface.T  # Return as row vectors


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


# ----------------------------------------------------------------------------


class ParafoilSections(abc.ABC):
    """Defines the spanwise variation of the Parafoil sections"""

    # FIXME: bad naming? An instance of this class isn't a Parafoil section.
    #        Plus, these docstrings are highly redundant with Airfoil's.
    #        Should this API *access* the airfoils, or *return* an Airfoil?

    @abc.abstractmethod
    def Cl_alpha(self, s, alpha, delta):
        """The derivative of the lift coefficient vs alpha for the section"""

    @abc.abstractmethod
    def Cl(self, s, alpha, delta):
        """The lift coefficient for the section"""

    @abc.abstractmethod
    def Cd(self, s, alpha, delta):
        """The drag coefficient for the section"""

    @abc.abstractmethod
    def Cm(self, s, alpha, delta):
        """The pitching moment coefficient for the section"""

    @abc.abstractmethod
    def upper_curve(self, s, xa):
        """The upper airfoil curve for the section"""

    @abc.abstractmethod
    def lower_curve(self, s, xa):
        """The lower airfoil curve for the section"""


class ConstantCoefficients(ParafoilSections):
    """
    Uses the same airfoil for all wing sections, no spanwise variation.
    """

    def __init__(self, airfoil):
        self.airfoil = airfoil

    def Cl_alpha(self, s, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cl_alpha(alpha, delta)

    def Cl(self, s, alpha, delta):
        # FIXME: make AirfoilCoefficients responsible for broadcasting `alpha`?
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cl(alpha, delta)

    def Cd(self, s, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cd(alpha, delta)

    def Cm(self, s, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cm(alpha, delta)

    def upper_curve(self, s, xa):
        return self.airfoil.geometry.upper_curve(xa)

    def lower_curve(self, s, xa):
        return self.airfoil.geometry.lower_curve(xa)


# ----------------------------------------------------------------------------


class ForceEstimator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, V_rel, delta, rho=1):
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
    This implementation uses a single, linear point distribution in terms of
    the normalized span coordinate `s`. Using a single distribution that covers
    the entire span is suitable for parafoils, but for wings with left and
    right segments separated by some discontinuity at the root you should
    distribute the points across each semispan independently. Also, this method
    assumes a linear distribution in `s` provides reasonable point spacing, but
    depending on the wing curvature a different distribution, such as a cosine,
    may be more applicable. See _[1].

    This method does suffer an issue where induced velocity goes to infinity as
    the segment lengths tend toward zero (as the number of segments increases,
    or for a poorly chosen point distribution). See _[2], section 8.2.3.
    """

    def __init__(self, parafoil, lobe_args={}):
        self.parafoil = parafoil
        self.lobe_args = lobe_args

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution; less likely to induce large velocties
        self.K = 31  # The number of bound vortex segments
        self.s_nodes = np.linspace(-1, 1, self.K+1)

        # Option 2: cosine distribution; fast, very sensitive to segment length
        # self.K = 13  # The number of bound vortex segments
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.c4(self.s_nodes)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1])/2
        self.cps = self.parafoil.c4(self.s_cps)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]  # node N is at at axis0=N-1
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        # FIXME: is there a better way to select the column vectors?
        u = self.parafoil.section_orientation(self.s_cps, lobe_args)
        self.u_a, self.u_s, self.u_n = u[:, :, 0], u[:, :, 1], u[:, :, 2]

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]  # `R1 + R2`?
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        c_avg = (node_chords[1:] + node_chords[:-1])/2
        chord_vectors = c_avg[:, None] * self.u_a  # FIXME: verify
        self.dA = norm(cross(chord_vectors, self.dl), axis=1)
        # FIXME: faster+correct to post_multiply the scalars `c_avg`?

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
        V_rel : array of float, shape (K,) [meters/second]
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

        # FIXME: is using the freestream velocity at the central chord okay?
        u_inf = V_rel[self.K // 2]
        u_inf = u_inf / norm(u_inf)

        # 2. Compute the "induced velocity" unit vectors
        v = self._induced_velocities(u_inf)  # axes = (inducer, inducee)
        vT = np.swapaxes(v, 0, 1)  # Useful for broadcasting cross products

        # 3. Propose an initial distribution for Gamma
        #  * For now, use an elliptical Gamma
        b = self.parafoil.b
        cp_y = self.cps[:, 1]
        Gamma0 = 5

        # Alternative initial proposal
        # avg_brake = (delta_Bl + delta_Br)/2
        # CL_2d = self.coefs.CL(np.arctan2(u_inf[2], u_inf[0]), avg_brake)
        # S = self.parafoil.S
        # Gamma0 = 2*norm(V_rel[self.K//2])*S*CL_2d/(np.pi*b)  # c0 circulation

        Gamma = Gamma0 * np.sqrt(1 - ((2*cp_y)/b)**2)

        # Save intermediate values for debugging purposes
        Vs = [V_rel]
        Gammas = [Gamma]
        delta_Gammas = []
        fs = []
        Js = []
        alphas = []
        Cl_alphas = []

        # FIXME: very ad-hoc way to prevent large negative AoA at the wing tips
        # FIXME: why must Omega be so small? `Cl_alpha` sensitivity?
        # M = max(delta_Bl, delta_Br)  # Assumes the delta_B are 0..1
        # base_Omega, min_Omega = 0.2, 0.05
        # Omega = base_Omega - (base_Omega - min_Omega)*np.sqrt(M)
        Omega = 0.1

        if max_runs is None:
            # max_runs = 5 + int(np.ceil(3*M))
            max_runs = 30

        # FIXME: don't use a fixed number of runs
        # FIXME: how much faster is `opt_einsum` versus the scipy version?
        # FIXME: if `coefs2d.Cl` was Numba compatible, what about this loop?
        n_runs = 0
        while n_runs < max_runs:
            # print("run:", n_runs)
            # 4. Compute the local fluid velocities
            #  * ref: Hunsaker-Snyder Eq:5
            #  * ref: Phillips Eq:5 (nondimensional version)
            V = V_rel + einsum('i,ijk->jk', Gamma, v)

            # 5. Compute the section local angle of attack
            #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
            V_a = einsum('ik,ik->i', V, self.u_a)  # Chordwise
            V_n = einsum('ik,ik->i', V, self.u_n)  # Normal-wise
            alpha = arctan2(V_n, V_a)

            min_alpha = min(alpha)
            if np.rad2deg(min_alpha) < -11:
                print("Encountered a very small alpha: {}".format(min_alpha))
                embed()

            # plt.plot(cp_y, np.rad2deg(alpha))
            # plt.ylabel('local section alpha')
            # plt.show()

            # For testing purposes: the global section alpha and induced AoA
            # V_chordwise_2d = einsum('ij,ij->i', V_rel, self.u_a)
            # V_normal_2d = einsum('ij,ij->i', V_rel, self.u_n)
            # alpha_2d = arctan2(V_normal_2d, V_chordwise_2d)
            # alpha_induced = alpha_2d - alpha

            # print("Stopping to investigate the alphas")
            # embed()
            # input('continue?')

            Cl = self.parafoil.sections.Cl(self.s_cps, alpha, delta)

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
            Cl_alpha = self.parafoil.sections.Cl_alpha(self.s_cps, alpha, delta)

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
            Cl_alphas.append(Cl_alpha)

            # print("finished run", n_runs)
            # embed()
            # 1/0

            # FIXME: ad-hoc workaround to avoid massively negative AoA
            Omega += (1 - Omega)/4

            n_runs += 1

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

        return Gamma, V

    def __call__(self, V_rel, delta, rho=1):
        # FIXME: depenency on rho?
        # FIXME: include viscous effects as well; ref: the Phillips paper
        Gamma, V = self._vortex_strengths(V_rel, delta)
        dF = Gamma[:, None] * cross(self.dl, V)
        dM = None
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

        # Option 1: linear distribution; less likely to induce large velocties
        self.K = 31  # The number of bound vortex segments
        self.s_nodes = np.linspace(-1, 1, self.K+1)

        # Option 2: cosine distribution; fast, very sensitive to segment length
        # self.K = 13  # The number of bound vortex segments
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.c4(self.s_nodes)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1])/2
        self.cps = self.parafoil.c4(self.s_cps)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]  # node N is at at axis0=N-1
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        u = self.parafoil.section_orientation(self.s_cps, lobe_args)
        self.u_a, self.u_s, self.u_n = u[:, 0], u[:, 1], u[:, 2]

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]  # `R1 + R2`?
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        c_avg = (node_chords[1:] + node_chords[:-1])/2
        chord_vectors = c_avg[:, None] * self.u_a  # FIXME: verify
        self.dA = norm(cross(chord_vectors, self.dl), axis=1)
        # FIXME: faster+correct to post_multiply the scalars `c_avg`?

    @property
    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def __call__(self, V_rel, delta, rho=1):
        # FIXME: dependency on rho?
        assert np.shape(V_rel) == (self.K, 3)

        # Compute the section local angle of attack
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_a = einsum('ik,ik->i', V_rel, self.u_a)  # Chordwise
        V_n = einsum('ik,ik->i', V_rel, self.u_n)  # Normal-wise
        alpha = arctan2(V_n, V_a)

        CL = self.parafoil.sections.Cl(self.s_cps, alpha, delta)
        CD = self.parafoil.sections.Cd(self.s_cps, alpha, delta)

        dL_hat = cross(self.dl, V_rel)
        dL_hat = dL_hat / norm(dL_hat, axis=1)[:, None]  # Lift unit vectors
        dL = (1/2 * np.sum(V_rel**2, axis=1) * self.dA * CL)[:, None] * dL_hat

        dD_hat = -(V_rel / norm(V_rel, axis=1)[:, None])  # Drag unit vectors
        dD = (1/2 * np.sum(V_rel**2, axis=1) * self.dA * CD)[:, None] * dD_hat

        dF = dL + dD
        dM = 0

        return dF, dM
