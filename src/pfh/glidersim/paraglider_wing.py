"""
Models of paraglider wings.

FIXME: explain "wing = canopy + suspension lines"
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import numpy as np
import scipy.optimize
import scipy.spatial

from pfh.glidersim import foil_aerodynamics
from pfh.glidersim.util import cross3, crossmat


if TYPE_CHECKING:
    from pfh.glidersim.foil import SimpleFoil


__all__ = [
    "LineGeometry",
    "SimpleLineGeometry",
    "ParagliderWing",
]


def __dir__():
    return __all__


@runtime_checkable
class LineGeometry(Protocol):
    """Interface for classes that define a LineGeometry model."""

    @abc.abstractmethod
    def r_RM2LE(self, delta_a=0):
        """
        Compute the position of the riser midpoint `RM` in body frd.

        Parameters
        ----------
        delta_a : array_like of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application. Default: 0

        Returns
        -------
        r_RM2LE : array of float, shape (N,3) [unitless]
            The riser midpoint `RM` with respect to the canopy origin.
        """

    @abc.abstractmethod
    def r_CP2LE(self):
        """
        Compute the control points for the line geometry dynamics.

        Returns
        -------
        ndarray of float, shape (K,3) [m]
            Control points relative to the central leading edge `LE`.
            Coordinates are in canopy frd, and `K` is the number of points
            being used to distribute the surface area of the lines.
        """

    @abc.abstractmethod
    def delta_d(self, s, delta_bl, delta_br):
        """
        Compute the trailing edge deflection distance due to brake inputs.

        Parameters
        ----------
        s : array_like of float
            Section index, where `-1 <= s <= 1`
        delta_bl : array_like of float [percentage]
            Left brake application as a fraction of maximum braking
        delta_br : array_like of float [percentage]
            Right brake application as a fraction of maximum braking

        Returns
        -------
        delta_d : float [m]
            The deflection distance of the trailing edge.
        """

    @abc.abstractmethod
    def aerodynamics(self, v_W2b, rho_air: float):
        """
        Calculate the aerodynamic forces and moments at each control point.

        Parameters
        ----------
        v_W2b : array of float, shape (K,3) [m/s]
            The wind velocity at each of the K control points.
        rho_air : float [kg/m^3]
            Air density

        Returns
        -------
        dF, dM : array of float, shape (K,3) [N, N m]
            Aerodynamic forces and moments for each control point.
        """


class SimpleLineGeometry(LineGeometry):
    """
    FIXME: add docstring.

    Parameters
    ----------
    kappa_x : float [m]
        x-coordinate distance from `RM` to the canopy origin.
    kappa_z : float [m]
        z-coordinate distance from `RM` to the canopy origin.
    kappa_A, kappa_C : float [m]
        Distance of the A and C canopy connection points along the central
        chord. The accelerator adjusts the length of the A lines, while the C
        lines remain fixed length, effectively causing a rotation of the canopy
        about the point `kappa_C`.
    kappa_a : float [m]
        Accelerator line length. This is the maximum change in the length of
        the A lines.
    kappa_b : float [m]
        Brake line length. Equal to the maximum vertical deflection of the
        trailing edge, which occurs at `(s_delta_start + s_delta_stop) / 2`.
        This is the deflection distance supported by the model, not the true
        physical length of the line. The aerodynamics model can only support a
        limited range of edge deflection, and this value is the deflection
        associated with "100% brake input".
    s_delta_start0, s_delta_start1 : float
        Section indices where brake deflections begin, transitioning from
        `start0` when `delta_b = 0` to `start1` when `delta_b = 1`.
        FIXME: needs a proper docstring. For example, these are for the right
        brake, but the left is symmetric.
    s_delta_stop0, s_delta_stop1 : float
        Section indices where brake deflections end, transitioning from `stop0`
        when `delta_b = 0` to `stop1` when `delta_b = 1`.
        FIXME: needs a proper docstring. For example, these are for the right
        brake, but the left is symmetric.
    total_line_length : float [m]
        Total length of the lines from the risers to the canopy.
    average_line_diameter : float [m]
        Average diameter of the connecting lines.
    r_L2LE : array of float, shape (K,3) [m]
        Averaged location(s) of the connecting line surface area(s). If
        multiple positions are given, the total line length will be divided
        between them evenly.
    Cd_lines : float
        Drag coefficient of the lines.

    Notes
    -----
    FIXME: describe the design, and reference the sections in my thesis.

    **Accelerator**

    FIXME: describe

    **Brakes**

    The wing root typically experiences very little (if any) brake deflection,
    so this model allows for zero deflection until some section `s_delta_start`
    from the wing root. Similarly, brake deflections typically stop a little
    before reaching the wing tip.

    This models the deflection distribution with a quartic function. It assumes
    the deflection distance is symmetric about some peak at the middle of
    `s_delta_start` and `s_delta_stop`.

    Regarding the the derivation: a normal quartic goes like :math:`Ap^4 + Bp^3
    + Cp^2 + Dp + E = 0`. Assuming symmetry about `p = 0.5`, the six terms can
    be solved using:

        delta_d(0) = 0
        d(delta_d)/dp | s_0 = 0
        delta_d(1) = 0
        d(delta_d)/dp | s_1 = 0
        delta_d(0.5) = delta_d_peak

    I don't love this design, but I was hoping to find time to prototype a
    proper line geometry that computes the true angles throughout the bridle,
    and having the LineGeometry compute distances instead of angles would
    make that easier.
    """

    def __init__(
        self,
        kappa_x: float,
        kappa_z: float,
        kappa_A: float,
        kappa_C: float,
        kappa_a: float,
        kappa_b: float,
        total_line_length: float,
        average_line_diameter: float,
        r_L2LE,
        Cd_lines: float,
        s_delta_start0: float,
        s_delta_start1: float,
        s_delta_stop0: float,
        s_delta_stop1: float,
    ):
        self.kappa_A = kappa_A
        self.kappa_C = kappa_C
        self.kappa_a = kappa_a

        # Default lengths of the A and C lines (when `delta_a = 0`)
        self.A = np.sqrt(kappa_z**2 + (kappa_x - kappa_A) ** 2)
        self.C = np.sqrt(kappa_z**2 + (kappa_C - kappa_x) ** 2)

        # `L` is an array of points where line drag is applied
        r_L2LE = np.atleast_2d(r_L2LE)
        if r_L2LE.ndim != 2 or r_L2LE.shape[-1] != 3:
            raise ValueError("`r_L2LE` is not a (K,3) array")
        self._r_L2LE = r_L2LE
        self._S_lines = total_line_length * average_line_diameter / r_L2LE.shape[0]
        self._Cd_lines = Cd_lines

        # FIXME: add sanity checks
        self.s_delta_start0 = s_delta_start0
        self.s_delta_start1 = s_delta_start1
        self.s_delta_stop0 = s_delta_stop0
        self.s_delta_stop1 = s_delta_stop1
        self.kappa_b = kappa_b

        # The non-zero coefficients for a 4th-order polynomial such that the
        # value and slope are both zero at `p = 0` and `p = 1`, symmetric about
        # a peak of `1` at `p = 0.5`. Evaluate to (16, -32, 16), but I like how
        # the equations document the solutions given the constraints.
        p = 0.5
        self._K1 = 1 / (p**4 - 2 * p**3 + p**2)
        self._K2 = -2 * self._K1
        self._K3 = self._K1

    def r_RM2LE(self, delta_a=0):
        # The accelerator shortens the A lines, while C remains fixed
        delta_a = np.asfarray(delta_a)
        RM_x = (
            (self.A - delta_a * self.kappa_a) ** 2
            - self.C**2
            - self.kappa_A**2
            + self.kappa_C**2
        ) / (2 * (self.kappa_C - self.kappa_A))
        RM_y = np.zeros_like(delta_a)
        RM_z = np.sqrt(self.C**2 - (self.kappa_C - RM_x) ** 2)
        r_RM2LE = np.array([-RM_x, RM_y, RM_z]).T
        return r_RM2LE

    def r_CP2LE(self):
        r_L2LE = self._r_L2LE.view()
        r_L2LE.flags.writeable = False
        return r_L2LE

    def delta_d(self, s, delta_bl, delta_br):
        def _interp(A, B, d):
            # Interpolate from A to B as function of 0 <= d <= 1
            return A + (B - A) * d

        def q(s, s_start, s_stop):
            # Map `s` into the quartic domain `p` to compute deflections
            p = (s - s_start) / (s_stop - s_start)
            q = self._K1 * p**4 + self._K2 * p**3 + self._K3 * p**2
            q = np.array(q)  # Allow indexing in case `s` is a scalar
            q[(p < 0) | (p > 1)] = 0  # Zero outside `start <= s <= stop`
            return q

        s_start_l = _interp(self.s_delta_start0, self.s_delta_start1, delta_bl)
        s_start_r = _interp(self.s_delta_start0, self.s_delta_start1, delta_br)
        s_stop_l = _interp(self.s_delta_stop0, self.s_delta_stop1, delta_bl)
        s_stop_r = _interp(self.s_delta_stop0, self.s_delta_stop1, delta_br)

        # The start/stop indices are defined using the right side (s > 0), but
        # the line geometry is symmetric so the section indices can be negated
        # to compute contributions from the left brake.
        ql = q(-s, s_start_l, s_stop_l)
        qr = q(s, s_start_r, s_stop_r)
        return (delta_bl * ql + delta_br * qr) * self.kappa_b

    def aerodynamics(self, v_W2b, rho_air: float):
        v_W2b = np.asfarray(v_W2b)
        assert v_W2b.shape == self._r_L2LE.shape
        dF = np.zeros(np.shape(v_W2b))
        v2 = (v_W2b**2).sum(axis=-1)
        mask = ~np.isclose(v2, 0.0)
        if np.any(mask):
            v2m = v2[mask][..., np.newaxis]
            u_drag = v_W2b[mask] / np.sqrt(v2m)  # Drag force unit vectors
            dF[mask] = (
                0.5
                * rho_air
                * v2m
                * self._S_lines  # Line area per control point
                * self._Cd_lines
                * u_drag
            )
        dM = np.zeros(dF.shape)
        return dF, dM

    def maximize_kappa_b(
        self,
        delta_d_max: float,
        chord_length: Callable,
        margin: float = 1e-6,
    ) -> None:
        """Maximze kappa_b such that delta_d never exceeds delta_d_max.

        Useful to ensure the deflections don't exceed the maximum delta_d
        supported by the airfoil coefficients.

        Parameters
        ----------
        delta_d_max : float
            Maximum deflection distance the brakes should produce at any
            section, normalized to unit chord length.
        chord_length : function
            Canopy chord length as a function of section index `-1 <= s <= 1`.
        """
        # FIXME: I don't love this design. It requires the user to know to pass
        # an arbitrary `kappa_b` to the constructor, assumes the normalized
        # delta_d is a convex function, assumes the maximum normalized
        # deflection occurs with full brakes, etc.
        self.kappa_b = 1
        res = scipy.optimize.minimize_scalar(
            lambda s: -self.delta_d(s, 1, 1) / chord_length(s),
            bounds=(0, 1),
            method="bounded",
        )
        delta_d = self.delta_d(res.x, 1, 1) / chord_length(res.x)
        self.kappa_b = delta_d_max / delta_d - margin


class ParagliderWing:
    """
    FIXME: add class docstring.

    The system is referred to as the "body" since that is conventional in
    aeronautics literature. Vectors are in "body frd", a coordinate system
    inherited from the canopy: the xyz axes are oriented front-right-down with
    the origin at the central leading edge `LE`.

    Parameters
    ----------
    lines : LineGeometry
        Lines that position the riser and produce trailing edge deflections.
    canopy : SimpleFoil
        The geometric shape of the lifting surface.
    rho_upper, rho_lower : float [kg/m^2]
        Surface area densities of the upper and lower canopy surfaces.
    rho_ribs : float [kg/m^2]
        Surface area density of the internal vertical ribs.
    N_cells : integer, optional
        The number of canopy cells. This is only used for estimating the mass
        of the internal ribs. Proper support for ribs would require a new foil
        geometry with native support for cells, ribs, profile distortions, etc.
    """

    def __init__(
        self,
        lines: LineGeometry,
        canopy: SimpleFoil,
        rho_upper: float = 0,
        rho_lower: float = 0,
        rho_ribs: float = 0,
        N_cells: int = 1,
    ):
        self.lines = lines
        self.canopy = canopy
        self.rho_upper = rho_upper
        self.rho_lower = rho_lower
        self.rho_ribs = rho_ribs
        self.N_cells = N_cells

        self._compute_real_mass_properties()
        self._compute_apparent_mass_properties()

    def _compute_real_mass_properties(self):
        # Compute the canopy mass properties in canopy coordinates
        cmp = self.canopy.mass_properties(N_s=101, N_r=101)

        # Hack: the Ixy/Iyz terms are non-zero due to numerical issues. The
        # meshes should be symmetric about the xz-plane, but for now I'll just
        # assume symmetry and set the values to zero. (You can increase the
        # grid resolution but that makes `mass_properties2` slow for no real
        # gain. If I start using asymmetric geometries then I'll change this.)
        print("Applying manual symmetry corrections to the canopy inertia...")
        for k in ("upper", "volume", "lower"):
            cmp[k + "_centroid"][1] = 0
            cmp[k + "_inertia"][[0, 1, 1, 2], [1, 0, 2, 1]] = 0

        # The `SimpleFoil` has no concept of internal ribs, but I'd like to at
        # least account for the rib mass. Compute the inertia of vertical ribs
        # (including wing tips)
        #
        # FIXME: this is a kludge, but ribs need design review anyway
        s_ribs = np.linspace(-1, 1, self.N_cells + 1)
        N_r = 151  # Number of points around each profile
        r = 1 - np.cos(np.linspace(0, np.pi / 2, N_r))
        r = np.concatenate((-r[:0:-1], r))
        rib_vertices = self.canopy.surface_xyz(s_ribs[:, None], 0, r, "airfoil")
        rib_points = self.canopy.sections.surface_xz(s_ribs[:, None], 0, r, "airfoil")
        rib_tris = []
        for n in range(len(rib_vertices)):
            rib_simplices = scipy.spatial.Delaunay(rib_points[n]).simplices
            rib_tris.append(rib_vertices[n][rib_simplices])
        rib_tris = np.asarray(rib_tris)
        rib_sides = np.diff(rib_tris, axis=2)
        rib1 = rib_sides[..., 0, :]
        rib2 = rib_sides[..., 1, :]
        rib_areas_n = np.linalg.norm(np.cross(rib1, rib2), axis=2) / 2
        rib_areas = np.sum(rib_areas_n, axis=1)  # For debugging
        rib_area = rib_areas_n.sum()
        rib_centroids_n = np.einsum("NKij->NKj", rib_tris) / 3
        r_RIB2LE = np.einsum("NK,NKi->i", rib_areas_n, rib_centroids_n) / rib_area
        cov_ribs = np.einsum(
            "NK,NKi,NKj->ij",
            rib_areas_n,
            rib_centroids_n - r_RIB2LE,
            rib_centroids_n - r_RIB2LE,
        )
        J_rib2RIB = np.trace(cov_ribs) * np.eye(3) - cov_ribs
        cmp.update(
            {
                "rib_area": rib_area,
                "rib_centroid": r_RIB2LE,
                "rib_inertia": J_rib2RIB,
            }
        )

        m_upper = cmp["upper_area"] * self.rho_upper
        m_lower = cmp["lower_area"] * self.rho_lower
        m_rib = cmp["rib_area"] * self.rho_ribs
        J_u2U = cmp["upper_inertia"] * self.rho_upper
        J_l2L = cmp["lower_inertia"] * self.rho_lower
        J_rib2RIB = cmp["rib_inertia"] * self.rho_ribs
        m_s = m_upper + m_lower + m_rib  # Solid mass
        r_S2LE = (  # Solid mass centroid
            m_upper * cmp["upper_centroid"]
            + m_lower * cmp["lower_centroid"]
            + m_rib * cmp["rib_centroid"]
        ) / m_s
        r_S2U = r_S2LE - cmp["upper_centroid"]
        r_S2L = r_S2LE - cmp["lower_centroid"]
        r_S2RIB = r_S2LE - cmp["rib_centroid"]
        D_u = (r_S2U @ r_S2U) * np.eye(3) - np.outer(r_S2U, r_S2U)
        D_l = (r_S2L @ r_S2L) * np.eye(3) - np.outer(r_S2L, r_S2L)
        D_rib = (r_S2RIB @ r_S2RIB) * np.eye(3) - np.outer(r_S2RIB, r_S2RIB)
        J_u2S = J_u2U + m_upper * D_u
        J_l2S = J_l2L + m_lower * D_l
        J_rib2S = J_rib2RIB + m_rib * D_rib
        J_s2S = J_u2S + J_l2S + J_rib2S
        self._real_mass_properties = {
            "m_s": m_s,
            "r_S2LE": r_S2LE,  # In canopy coordinates
            "J_s2S": J_s2S,
            "v": cmp["volume"],
            "r_V2LE": cmp["volume_centroid"],
            "J_v2V": cmp["volume_inertia"],
        }

    def _compute_apparent_mass_properties(self):
        """
        Compute an approximate apparent mass matrix for the canopy volume.

        This follows the development in "Apparent Mass of Parafoils with
        Spanwise Camber" (Barrows, 2002), which means it assumes the canopy has
        a circular arch, has two planes of symmetry (xz and yz), and uniform
        thickness. In reality, paragliders have non-circular arch, non-uniform
        thickness, taper, and torsion. The assumptions in this paper, then, are
        clearly very strong, but it's still a good starting point for average
        paraglider wings.

        This implementation tries to follow the equations in the paper as
        closely as possible, with some exceptions:

          * pitch and roll centers: `p` and `r` are now `PC` and `RC`
          * circular confluence point: `c` is now `C`
          * arc semi-angle: `Theta` is now `theta`
          * radius: `R` is now `r`
          * origin: `o` is now `R` (a reference point in the xz-plane)

        Dynamics models that want to incorporate these effects can use
        equations 16, 24, 61, and 64 (making sure to remove the steady-state
        term from Eq:64, as noted in the paper).
        """
        # For testing: values for the flat and arched wings in Barrows. Verify
        # the results in Table:1 and Table:2.
        # AR = 3
        # b = 3
        # c = 1
        # r = 2.5
        # t = 0.15
        # S = b * c
        # theta = np.deg2rad(35)
        # hstar = eta / 4

        S = self.canopy.S
        b = self.canopy.b
        AR = self.canopy.AR
        c = self.canopy.S_flat / self.canopy.b_flat  # Standard mean chord

        # Barrows assumes uniform thickness, so I'm using an average of the
        # thickest region. Also note that although Barrows refers to `t` as a
        # thickness "ratio", it is the absolute, not relative, thickness.
        #
        # FIXME: There should be a balance between the thickness exposed to the
        #        forward and pitching moments versus the thickness exposed to
        #        the lateral, rolling, and yawing motions. Perhaps tailor
        #        different thicknesses for the different dimensions?
        t = np.mean(
            self.canopy.section_thickness(
                s=np.linspace(0.0, 0.5, 25),  # Central 50% of the canopy
                ai=0,
                r=np.linspace(0.1, 0.5, 25),  # Thickest parts of each airfoil
            ),
        )

        # Assuming the arch is circular, find its radius and arc angle using
        # the quarter-chords of the central section and the wing tip. See
        # Barrows Figure:5 for a diagram.
        r_tip2center = (
            self.canopy.surface_xyz(1, 0, 0.25, surface="chord")
            - self.canopy.surface_xyz(0, 0, 0.25, surface="chord")  # fmt: skip
        )
        dz = (r_tip2center[1] ** 2 - r_tip2center[2] ** 2) / (2 * r_tip2center[2])
        r = dz + r_tip2center[2]  # Arch radius
        theta = np.arctan2(r_tip2center[1], dz)  # Symmetric arch semi-angle
        h = r_tip2center[2]  # Height from the central section to the tip
        hstar = h / b

        # Three-dimensional correction factors
        k_A = 0.85
        k_B = 1.00

        # Flat wing values, Barrows Eqs:34-39
        mf11 = k_A * np.pi * t**2 * b / 4
        mf22 = k_B * np.pi * t**2 * c / 4
        mf33 = AR / (1 + AR) * np.pi * c**2 * b / 4
        If11 = 0.055 * AR / (1 + AR) * b * S**2
        If22 = 0.0308 * AR / (1 + AR) * c**3 * S
        If33 = 0.055 * b**3 * t**2

        # Compute the pitch and roll centers, treating the wing as a circular
        # arch with fore-aft (yz) and lateral (xz) planes of symmetry. The roll
        # center, pitch center, and the "confluence point" all lie on the
        # z-axis of the idealized circular arch.
        z_PC2C = -r * np.sin(theta) / theta  # Barrows Eq:44
        z_RC2C = z_PC2C * mf22 / (mf22 + If11 / r**2)  # Barrows Eq:50
        z_PC2RC = z_PC2C - z_RC2C

        # Arched wing values, Barrows Eqs:51-55
        m11 = k_A * (1 + 8 / 3 * hstar**2) * np.pi * t**2 * b / 4
        m22 = (r**2 * mf22 + If11) / z_PC2C**2
        m33 = mf33
        I11 = (
            z_PC2RC**2 / z_PC2C**2 * r**2 * mf22
            + z_RC2C**2 / z_PC2C**2 * If11
        )
        I22 = If22
        I33 = 0.055 * (1 + 8 * hstar**2) * b**3 * t**2

        # These values are constants for the geometry and choice of coordinate
        # system, and are used to compute the apparent inertia matrix about
        # `R`, a reference point that must lie in the xz-plane of symmetry. See
        # `ParagliderWing.mass_properties` for the calculation.
        r_C2LE = np.array([-0.5 * self.canopy.chord_length(0), 0, r])
        r_RC2C = np.array([0, 0, z_RC2C])
        self._apparent_mass_properties = {
            "r_RC2LE": r_RC2C + r_C2LE,
            "r_PC2RC": np.array([0, 0, z_PC2RC]),
            "M": np.diag([m11, m22, m33]),  # Barrows Eq:1
            "I": np.diag([I11, I22, I33]),  # Barrows Eq:17
        }

    def aerodynamics(
        self,
        delta_a: float,
        delta_bl: float,
        delta_br: float,
        v_W2b,
        rho_air: float,
        reference_solution: dict | None = None,
    ):
        """
        FIXME: add docstring.

        Parameters
        ----------
        delta_a : float [percentage]
            Fraction of accelerator, from 0 to 1
        delta_bl : float [percentage]
            Fraction of left brake, from 0 to 1
        delta_br : float [percentage]
            Fraction of right brake, from 0 to 1
        v_W2b : array of float, shape (K,3) [m/s]
            The wind vector at each control point in body frd
        rho_air : float [kg/m^3]
            The ambient air density
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        dF, dM : array of float, shape (K,3) [N, N m]
            Aerodynamic forces and moments for each control point.
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        # FIXME: (1) duplicates `self.r_CP2LE`
        #        (2) only used to get the shapes of the control point arrays
        foil_cps = self.canopy.r_CP2LE()
        line_cps = self.lines.r_CP2LE()

        # FIXME: uses "magic" indexing established in `self.r_CP2LE()`
        K_foil = foil_cps.shape[0]
        K_lines = line_cps.shape[0]

        # Support automatic broadcasting if v_W2b.shape == (3,)
        v_W2b = np.broadcast_to(v_W2b, (K_foil + K_lines, 3))
        v_W2b_foil = v_W2b[:-K_lines]
        v_W2b_lines = v_W2b[-K_lines:]

        s_cps = self.canopy.aerodynamics.s_cps  # FIXME: leaky
        delta_d = self.lines.delta_d(s_cps, delta_bl, delta_br)
        ai = delta_d / self.canopy.chord_length(s_cps)
        dF_foil, dM_foil, solution = self.canopy.aerodynamics(
            ai, v_W2b_foil, rho_air, reference_solution=reference_solution
        )

        dF_lines, dM_lines = self.lines.aerodynamics(v_W2b_lines, rho_air)
        dF = np.vstack((dF_foil, dF_lines))
        dM = np.vstack((dM_foil, dM_lines))

        return dF, dM, solution

    def equilibrium_alpha(
        self,
        delta_a: float,
        delta_b: float,
        v_mag: float,
        rho_air: float = 1.225,
        alpha_0: float = 9,
        alpha_1: float = 6,
        reference_solution: dict | None = None,
    ):
        """
        Compute the angle of attack with zero aerodynamic pitching moment.

        The final wing will have extra moments from the harness and weight of
        the wing, but this value is often a good estimate.

        Parameters
        ----------
        delta_a : float [percentage], optional
            Fraction of accelerator, from 0 to 1
        delta_b : float [percentage]
            Fraction of symmetric brake, from 0 to 1
        v_mag : float [m/s]
            Airspeed
        rho_air : float [kg/m^3], optional
            Air density
        alpha_0 : float [deg], optional
            First guess for the equilibrium alpha search.
        alpha_1 : float [deg], optional
            Second guess for the equilibrium alpha search.
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        float [rad]
            The angle of attack where the section pitching moments sum to zero.
        """
        r_CP2RM = self.r_CP2LE(delta_a) - self.r_RM2LE(delta_a)

        def target(alpha):
            v_W2b = -v_mag * np.array([np.cos(alpha), 0, np.sin(alpha)])
            dF_wing, dM_wing, _ = self.aerodynamics(
                delta_a, delta_b, delta_b, v_W2b, rho_air, reference_solution
            )
            M = dM_wing.sum(axis=0) + cross3(r_CP2RM, dF_wing).sum(axis=0)
            return M[1]  # Wing pitching moment

        x0, x1 = np.deg2rad([alpha_0, alpha_1])
        res = scipy.optimize.root_scalar(target, x0=x0, x1=x1)
        if not res.converged:
            raise foil_aerodynamics.ConvergenceError
        return res.root

    def r_CP2LE(self, delta_a=0):
        """
        Compute the FoilGeometry control points in frd.

        FIXME: descibe/define "control points"

        Parameters
        ----------
        delta_a : array_like of float, shape (K,) [percentage] (optional)
            Fraction of maximum accelerator application

        Returns
        -------
        ndarray of floats, shape (K,3) [m]
            The control points in frd coordinates
        """
        foil_cps = self.canopy.r_CP2LE()
        line_cps = self.lines.r_CP2LE()
        return np.vstack((foil_cps, line_cps))

    def r_RM2LE(self, delta_a=0):
        """
        Compute the position of the riser midpoint `RM` in frd coordinates.

        Parameters
        ----------
        delta_a : array_like of float, shape (K,) [percentage] (optional)
            Fraction of maximum accelerator application. Default: 0

        Returns
        -------
        r_RM2LE : array of float, shape (K,3) [m]
            The riser midpoint `RM` with respect to the canopy origin.
        """
        return self.lines.r_RM2LE(delta_a)

    def mass_properties(self, rho_air: float, r_R2LE):
        """
        Compute the mass properties of the materials and enclosed volume of air.

        Parameters
        ----------
        rho_air : float [kg/m^3]
            Air density
        r_R2LE : array of float, shape (3,) [m]
            Reference point `R` with respect to the canopy origin in body frd.

        Returns
        -------
        dictionary
            m_s : float [kg]
                The solid mass of the wing
            r_S2LE : array of float, shape (3,) [m]
                Vector from the canopy origin to the solid mass centroid
            r_S2R : array of float, shape (3,) [m]
                Vector from the reference point to the solid mass centroid
            J_s2S : array of float, shape (3,3) [kg m^2]
                The moment of inertia matrix of the solid mass about its cm
            v : float [m^3]
                The enclosed volume
            r_V2R : array of float, shape (3,) [m]
                Vector from the reference point to the volume centroid V
            J_v2V : array of float, shape (3,3) [m^2]
                The moment of inertia matrix of the volume about its centroid
            m_air : float [kg m^3]
                The enclosed air mass.
        """
        r_LE2R = -np.asfarray(r_R2LE)
        mp = self._real_mass_properties.copy()

        mp["m_air"] = mp["v"] * rho_air
        mp["m_b"] = mp["m_s"] + mp["m_air"]
        r_S2R = mp["r_S2LE"] + r_LE2R
        r_V2R = mp["r_V2LE"] + r_LE2R
        mp["r_S2R"] = r_S2R
        mp["r_V2R"] = r_V2R
        mp["r_B2R"] = (mp["m_s"] * r_S2R + mp["m_air"] * r_V2R) / mp["m_b"]
        D_s = (r_S2R @ r_S2R) * np.eye(3) - np.outer(r_S2R, r_S2R)
        D_v = (r_V2R @ r_V2R) * np.eye(3) - np.outer(r_V2R, r_V2R)
        mp["J_b2R"] = (
            mp["J_s2S"]
            + mp["m_s"] * D_s
            + mp["J_v2V"] * rho_air
            + mp["m_air"] * D_v  # fmt: skip
        )

        return mp

    def apparent_mass_properties(self, rho_air: float, r_R2LE, v_R2e, omega_b2e):
        """
        Compute the apparent mass matrix and momentum of the canopy.

        Parameters
        ----------
        rho_air : float [kg/m^3]
            Air density
        r_R2LE : array of float, shape (3,) [m]
            Reference point `R` with respect to the canopy origin in body frd.
        v_R2e : array of float, shape (3,) [m/s]
            The velocity of the `R` with respect to the inertial frame.
        omega_b2e : array of float, shape (3,) [rad/s]
            Angular velocity of the body, in body frd coordinates.

        Returns
        -------
        dictionary
            r_PC2RC : array of float, shape (3,) [m]
                Vector to the pitch center from the roll center
            r_RC2R : array of float, shape (3,) [m]
                Vector to the roll center from the riser connection point
            M_a : array of float, shape (3,3)
                The apparent mass matrix
            A_a2R : array of float, shape (6,6)
                The apparent inertia matrix of the volume with respect to `R`
            p_a2e : array of float, shape (3,)
                The apparent linear momentum with respect to the inertial frame
            h_a2R : array of float, shape (3,3)
                The angular momentum of the apparent mass with respect to `R`
        """
        # FIXME: log a warning if `R` does not lie in the # xz-plane
        r_LE2R = -np.asfarray(r_R2LE)
        ai = self._apparent_mass_properties  # Dictionary of precomputed values
        r_RC2R = ai["r_RC2LE"] + r_LE2R
        r_PC2RC = ai["r_PC2RC"]
        S_PC2RC = crossmat(r_PC2RC)
        S_RC2R = crossmat(r_RC2R)
        S2 = np.diag([0, 1, 0])  # "Selection matrix", Barrows Eq:15

        # fmt: off

        # Apparent inertia
        M_a = ai["M"] * rho_air  # Apparent mass matrix
        Q = S2 @ S_PC2RC @ M_a @ S_RC2R
        J_a2R = (  # Apparent moment of inertia matrix about `R`; Barrows Eq:25
            ai["I"] * rho_air
            - S_RC2R @ M_a @ S_RC2R
            - S_PC2RC @ M_a @ S_PC2RC @ S2
            - Q
            - Q.T
        )
        MC = -M_a @ (S_RC2R + S_PC2RC @ S2)
        A_a2R = np.block([[M_a, MC], [MC.T, J_a2R]])  # Barrows Eq:27
        p_a2e = M_a @ (  # Apparent linear momentum (Barrows Eq:16)
            v_R2e
            - cross3(r_RC2R, omega_b2e)
            - crossmat(r_PC2RC) @ S2 @ omega_b2e
        )
        h_a2R = (  # Apparent angular momentum (Barrows Eq:24)
            (S2 @ S_PC2RC + S_RC2R) @ M_a @ v_R2e
            + J_a2R @ omega_b2e
        )

        # fmt: on

        return {
            "r_PC2RC": r_PC2RC,
            "r_RC2R": r_RC2R,
            "M_a": M_a,
            "A_a2R": A_a2R,
            "p_a2e": p_a2e,
            "h_a2R": h_a2R,
        }

    def resultant_force(
        self,
        delta_a,
        delta_bl,
        delta_br,
        v_W2b,
        rho_air,
        g,
        r_R2LE,
        mp=None,
        reference_solution=None,
    ):
        """
        Calculate the net force and moment due to wind and gravity.

        Parameters
        ----------
        delta_a : float [percentage]
            Fraction of accelerator, from 0 to 1
        delta_bl : float [percentage]
            Fraction of left brake, from 0 to 1
        delta_br : float [percentage]
            Fraction of right brake, from 0 to 1
        v_W2b : array of float, shape (K,3) [m/s]
            The wind vector at each control point in body frd
        rho_air : float [kg/m^3]
            The ambient air density
        g : array of float, shape (3,) [m/s^s]
            The gravity vector in body frd
        r_R2LE : array of float, shape (3,) [m]
            Reference point `R` with respect to the canopy origin in body frd.
        mp : dictionary, optional
            The mass properties of the body associated with the given air
            density and reference point. Used to avoid recomputation.
        reference_solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        f_b, g_B2R : array of float, shape (3,) [N, N m]
           The net force and moment on the body with respect to `R`.
        """
        if mp is None:
            mp = self.mass_properties(rho_air, r_R2LE)

        r_CP2R = self.r_CP2LE(delta_a) - r_R2LE
        if v_W2b.shape not in [(3,), r_CP2R.shape]:
            raise ValueError(f"v_W2h must be a (3,) or a {r_CP2R.shape}")
        v_W2b = np.broadcast_to(v_W2b, r_CP2R.shape)
        df_aero, dg_aero, ref = self.aerodynamics(
            delta_a=delta_a,
            delta_bl=delta_bl,
            delta_br=delta_br,
            v_W2b=v_W2b,
            rho_air=rho_air,
            reference_solution=reference_solution,
        )
        f_aero = df_aero.sum(axis=0)
        f_weight = mp["m_s"] * g
        f_b = f_aero + f_weight
        g_b2R = dg_aero.sum(axis=0)
        g_b2R += cross3(r_CP2R, df_aero).sum(axis=0)
        g_b2R += cross3(mp["r_S2R"], f_weight)
        return f_b, g_b2R, ref
