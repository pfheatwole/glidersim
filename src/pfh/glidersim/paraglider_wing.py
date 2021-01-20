"""FIXME: add module docstring."""

import numpy as np
from scipy.optimize import root_scalar

from pfh.glidersim import foil
from pfh.glidersim.util import cross3, crossmat


__all__ = [
    "SimpleLineGeometry",
    "ParagliderWing",
]


def __dir__():
    return __all__


class SimpleLineGeometry:
    """
    FIXME: document the design.

    In particular, highlight that everything here is normalized by the length
    of the central chord.

    Parameters
    ----------
    kappa_x : float [percentage]
        The absolute x-coordinate distance from `RM` to the canopy origin,
        normalized by the length of the central chord.
    kappa_z : float [m]
        The absolute z-coordinate distance from `RM` to the canopy origin,
        normalized by the length of the central chord.
    kappa_A, kappa_C : float [percentage]
        The position of the A and C canopy connection points, normalized by the
        length of the central chord. The accelerator adjusts the length of the
        A lines, while the C lines remain fixed length, effectively causing a
        rotation of the canopy about the point `kappa_C`.
    kappa_a : float [m], optional
        The accelerator line length normalized by the length of the central
        chord. This is the maximum change in the length of the A lines.
    total_line_length : float [m]
        The total length of the lines from the risers to the canopy, normalized
        by the length of the central chord.
    average_line_diameter : float [m^2]
        The average diameter of the connecting lines
    r_L2LE : array of float, shape (K,3) [m]
        The averaged location(s) of the connecting line surface area(s),
        normalized by the length of the central chord. If multiple positions
        are given, the total line length will be divided between them evenly.
    Cd_lines : float
        The drag coefficient of the lines
    s_delta_start : float
        The section index where brake deflections begin.
    s_delta_max : float
        The section index where the brake deflection reaches its maximum. To
        prevent negative deflections at the wing tips, `s_delta_max` has a minimum
        value; see :py:meth:`s_delta_max_min`
    delta_max : float [radians]
        The maximum deflection angle, which occurs at `s_delta_max`

    Notes
    -----

    FIXME: describe the design, and maybe reference the sections in my thesis.

    Accelerator
    ^^^^^^^^^^^

    FIXME: describe

    Brakes
    ^^^^^^

    The brake deflection angles are approximated using a cubic function.

    The wing root typically experiences very little (if any) brake deflection,
    so this parametrization allows for zero deflection until a y-axis distance
    `s_delta_start` from the wing root.

    Also, the maximum delta does not typically occur at the wing tip, but at a
    point closer to the 3/4 semispan. This parametrization uses `s_delta_max` to
    control where that maximum deflection occurs.

    Regarding the the derivation: a normal cubic goes like :math:`ay^3 + by^2 +
    cy + d = 0`, but constraining the function to be zero at the origin forces
    :math:`d = 0`, and constraining the first derivative to be zero when
    evaluated at the origin forces :math:`c = 0`. This leaves only the cubic
    and quadratic terms.  The cubic distribution can then be defined using just
    two constraints: :math:`f(s_{peak}) = delta_{max}` and
    :math:`(df/dy)|s_{peak} = 0`.
    """

    def __init__(
        self,
        kappa_x,
        kappa_z,
        kappa_A,
        kappa_C,
        kappa_a,
        total_line_length,
        average_line_diameter,
        r_L2LE,
        Cd_lines,
        s_delta_start,
        s_delta_max,
        delta_max,
    ):
        self.kappa_A = kappa_A
        self.kappa_C = kappa_C
        self.kappa_a = kappa_a  # FIXME: strange notation. Why `kappa`?

        # Default lengths of the A and C lines (when `delta_a = 0`)
        self.A = np.sqrt(kappa_z ** 2 + (kappa_x - kappa_A) ** 2)
        self.C = np.sqrt(kappa_z ** 2 + (kappa_C - kappa_x) ** 2)

        # `L` is an array of points where line drag is applied
        r_L2LE = np.atleast_2d(r_L2LE)
        if r_L2LE.ndim != 2 or r_L2LE.shape[-1] != 3:
            raise ValueError("`r_L2LE` is not a (K,3) array")
        self._r_L2LE = r_L2LE
        self._S_lines = total_line_length * average_line_diameter / r_L2LE.shape[0]
        self._Cd_lines = Cd_lines

        # Brakes
        self.s_delta_start = s_delta_start
        self.s_delta_max = s_delta_max
        if s_delta_max < self.minimum_s_delta_max(s_delta_start):
            raise ValueError(
                "s_delta_max is too small: delta_f is negative at the wing tips"
            )
        self.K1 = -2 * delta_max / ((s_delta_max - s_delta_start) ** 3)
        self.K2 = 3 * delta_max / (s_delta_max - s_delta_start) ** 2

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
            The riser midpoint `RM` with respect to the canopy origin. As with
            all other values in this class, these values are assumed to have
            been normalized by the length of the central chord of the wing.
        """
        # The accelerator shortens the A lines, while C remains fixed
        delta_a = np.asarray(delta_a)
        RM_x = (
            (self.A - delta_a * self.kappa_a) ** 2
            - self.C ** 2
            - self.kappa_A ** 2
            + self.kappa_C ** 2
        ) / (2 * (self.kappa_C - self.kappa_A))
        RM_y = np.zeros_like(delta_a)
        RM_z = np.sqrt(self.C ** 2 - (self.kappa_C - RM_x) ** 2)
        r_RM2LE = np.array([-RM_x, RM_y, RM_z]).T
        return r_RM2LE

    def control_points(self):
        """
        Compute the control points for the line geometry dynamics.

        Returns
        -------
        r_CP2LE : float, shape (K,3) [m]
            Control points relative to the central leading edge `LE`.
            Coordinates are in canopy frd, and `K` is the number of points
            being used to distribute the surface area of the lines.
        """
        return self._r_L2LE

    def delta_f(self, s, delta_bl, delta_br):
        """
        Compute the trailing edge deflection due to the application of brakes.

        Parameters
        ----------
        s : float, or array_like of float, shape (N,)
            Normalized span position, where `-1 <= s <= 1`
        delta_bl : float [percentage]
            Left brake application as a fraction of maximum braking
        delta_br : float [percentage]
            Right brake application as a fraction of maximum braking

        Returns
        -------
        delta_f : float [radians]
            The deflection angle of the trailing edge, measured between the
            undeflected chord and the line connecting the leading edge to the
            deflected trailing edge.
        """
        fraction = np.where(s < 0, delta_bl, delta_br)
        s = np.abs(s)  # left and right side are symmetric
        fraction = fraction * (s > self.s_delta_start)
        p = s - self.s_delta_start  # The cubic uses a shifted origin, `s_delta_start`
        return fraction * (self.K1 * p ** 3 + self.K2 * p ** 2)

    def aerodynamics(self, v_W2b, rho_air):
        K_lines = self._r_L2LE.shape[0]

        # Simplistic model for line drag using `K_lines` isotropic points
        V = v_W2b[-K_lines:]  # FIXME: uses "magic" indexing
        V2 = (V ** 2).sum(axis=1)
        u_drag = V.T / np.sqrt(V2)
        dF_lines = (
            0.5
            * rho_air
            * V2
            * self._S_lines  # Line area per control point
            * self._Cd_lines
            * u_drag
        ).T
        dM_lines = np.zeros((K_lines, 3))

        return dF_lines, dM_lines  # Normalized by the length of the central chord

    @staticmethod
    def minimum_s_delta_max(s_delta_start):
        """The minimum value of `s_delta_max` to avoid negative deflections."""
        return (2 / 3) * (1 - s_delta_start) + s_delta_start


class ParagliderWing:
    """
    FIXME: add class docstring.

    Parameters
    ----------
    lines : LineGeometry
        Lines that position the riser and produce trailing edge deflections.
    canopy : foil.FoilGeometry
        The geometric shape of the lifting surface.
    rho_upper, rho_lower : float [kg m^-2]
        Surface area densities of the upper and lower canopy surfaces.
    foil_aerodynamics : foil.FoilAerodynamics
        Estimator for the aerodynamic forces and moments on the canopy.

    Notes
    -----
    The ParagliderWing coordinate axes are parallel to the canopy axes, but the
    origin is translated from `LE`, the central leading edge of the canopy, to
    `RM`, the midpoint between the two riser connections.
    """

    def __init__(
        self,
        lines,
        canopy,
        N_cells=1,
        rho_upper=0,
        rho_lower=0,
        rho_ribs=0,
        foil_aerodynamics=None,
    ):
        self.lines = lines
        self.canopy = canopy
        self.N_cells = N_cells
        self.rho_upper = rho_upper
        self.rho_lower = rho_lower
        self.rho_ribs = rho_ribs
        self.foil_aerodynamics = foil_aerodynamics
        self.c_0 = canopy.chord_length(0)  # Scales the line geometry

        # Compute the canopy mass properties in canopy coordinates
        # cmp = self.canopy.mass_properties(N=5000)  # Assumes `delta_a = 0`
        cmp = self.canopy.mass_properties2(N_s=101, N_r=101, N_cells=N_cells)

        # Hack: the Ixy/Iyz terms are non-zero due to numerical issues. The
        # meshes should be symmetric about the xz-plane, but for now I'll just
        # assume symmetry and set the values to zero. (You can increase the
        # grid resolution but that makes `mass_properties2` slow for no real
        # gain. If I start using asymmetric geometries then I'll change this.)
        print("Applying manual symmetry corrections to the canopy inertia...")
        for k in ("upper", "volume", "lower"):
            cmp[k + "_centroid"][1] = 0
            cmp[k + "_inertia"][[0, 1, 1, 2], [1, 0, 2, 1]] = 0

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
        self._mass_properties = {
            "m_s": m_s,
            "r_S2LE": r_S2LE,  # In canopy coordinates
            "J_s2S": J_s2S,
            "v": cmp["volume"],
            "r_V2LE": cmp["volume_centroid"],
            "J_v2V": cmp["volume_inertia"],
        }

        self._compute_apparent_masses()

    def _compute_apparent_masses(self):
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
          * origin: `o` is now `RM` (the riser connection point)

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

        # Values for the Hook 3 23
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
                np.linspace(0.0, 0.5, 25),  # Central 50% of the canopy
                np.linspace(0.1, 0.5, 25),  # Thickest parts of each airfoil
            ),
        )

        # Assuming the arch is circular, find its radius and arc angle using
        # the quarter-chords of the central section and the wing tip. See
        # Barrows Figure:5 for a diagram.
        r_tip2center = (self.canopy.surface_xyz(1, 0.25, surface="chord")
                        - self.canopy.surface_xyz(0, 0.25, surface="chord"))
        dz = (r_tip2center[1]**2 - r_tip2center[2]**2) / (2 * r_tip2center[2])
        r = dz + r_tip2center[2]  # Arch radius
        theta = np.arctan2(r_tip2center[1], dz)  # Symmetric arch semi-angle
        h = r_tip2center[2]  # Height from the central section to the tip
        hstar = h / b

        # Three-dimensional correction factors
        k_A = 0.85
        k_B = 1.00

        # Flat wing values, Barrows Eqs:34-39
        mf11 = k_A * np.pi * t ** 2 * b / 4
        mf22 = k_B * np.pi * t ** 2 * c / 4
        mf33 = AR / (1 + AR) * np.pi * c**2 * b / 4
        If11 = 0.055 * AR / (1 + AR) * b * S ** 2
        If22 = 0.0308 * AR / (1 + AR) * c ** 3 * S
        If33 = 0.055 * b ** 3 * t ** 2

        # Compute the pitch and roll centers, treating the wing as a circular
        # arch with fore-and-aft (yz) and lateral (xz) planes of symmetry.
        # The roll center, pitch center, and the "confluence point" all lie on
        # the z-axis of the idealized circular arch. The rest of the derivation
        # requires that the "origin" (the riser midpoint `RM`, in this case)
        # lies in the xz-plane of symmetry.
        z_PC2C = -r * np.sin(theta) / theta  # Barrows Eq:44
        z_RC2C = z_PC2C * mf22 / (mf22 + If11 / r ** 2)  # Barrows Eq:50
        z_PC2RC = z_PC2C - z_RC2C

        # Arched wing values, Barrows Eqs:51-55
        m11 = k_A * (1 + 8 / 3 * hstar ** 2) * np.pi * t**2 * b / 4
        m22 = (r ** 2 * mf22 + If11) / z_PC2C ** 2
        m33 = mf33
        I11 = (
            z_PC2RC ** 2 / z_PC2C ** 2 * r ** 2 * mf22
            + z_RC2C ** 2 / z_PC2C ** 2 * If11
        )
        I22 = If22
        I33 = 0.055 * (1 + 8 * hstar ** 2) * b ** 3 * t ** 2

        # Save the precomputed values for use in `mass_properties`. The vectors
        # are defined with respect to `C` and the canopy origin (the central
        # leading edge), but the final apparent inertia matrix is about `RM`,
        # which depends on `delta_a`.
        r_C2LE = np.array([-0.5 * self.c_0, 0, r])
        r_RC2C = np.array([0, 0, z_RC2C])
        self._apparent_inertia = {
            "r_RC2LE": r_RC2C + r_C2LE,
            "r_PC2RC": np.array([0, 0, z_PC2RC]),
            "M": np.diag([m11, m22, m33]),  # Barrows Eq:1
            "I": np.diag([I11, I22, I33]),  # Barrows Eq:17
        }

    def aerodynamics(
        self, delta_a, delta_bl, delta_br, v_W2b, rho_air, reference_solution=None,
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
        # FIXME: (1) duplicates `self.control_points`
        #        (2) only used to get the shapes of the control point arrays
        foil_cps = self.foil_aerodynamics.control_points()
        line_cps = self.c_0 * self.lines.control_points()

        # FIXME: uses "magic" indexing established in `self.control_points()`
        K_foil = foil_cps.shape[0]
        K_lines = line_cps.shape[0]

        # Support automatic broadcasting if v_W2b.shape == (3,)
        v_W2b = np.broadcast_to(v_W2b, (K_foil + K_lines, 3))
        v_W2b_foil = v_W2b[:-K_lines]
        v_W2b_lines = v_W2b[-K_lines:]

        delta_f = self.lines.delta_f(
            self.foil_aerodynamics.s_cps, delta_bl, delta_br,
        )  # FIXME: leaky, don't grab `s_cps` directly
        dF_foil, dM_foil, solution = self.foil_aerodynamics(
            delta_f, v_W2b_foil, rho_air, reference_solution,
        )

        dF_lines, dM_lines = self.lines.aerodynamics(v_W2b_lines, rho_air)
        dF_lines *= self.c_0
        dM_lines *= self.c_0

        dF = np.vstack((dF_foil, dF_lines))
        dM = np.vstack((dM_foil, dM_lines))

        return dF, dM, solution

    def equilibrium_alpha(
        self,
        delta_a,
        delta_b,
        v_mag,
        rho_air=1.225,
        alpha_0=9,
        alpha_1=6,
        reference_solution=None,
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
        alpha_0 : float [rad], optional
            First guess for the equilibrium alpha search.
        alpha_1 : float [rad], optional
            First guess for the equilibrium alpha search.
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`

        Returns
        -------
        float [rad]
            The angle of attack where the section pitching moments sum to zero.
        """
        r_CP2RM = self.control_points(delta_a) - self.r_RM2LE(delta_a)

        def target(alpha):
            v_W2b = -v_mag * np.array([np.cos(alpha), 0, np.sin(alpha)])
            dF_wing, dM_wing, _ = self.aerodynamics(
                delta_a, delta_b, delta_b, v_W2b, rho_air, reference_solution,
            )
            M = dM_wing.sum(axis=0) + cross3(r_CP2RM, dF_wing).sum(axis=0)
            return M[1]  # Wing pitching moment
        x0, x1 = np.deg2rad([alpha_0, alpha_1])
        res = root_scalar(target, x0=x0, x1=x1)
        if not res.converged:
            raise foil.FoilAerodynamics.ConvergenceError
        return res.root

    def control_points(self, delta_a=0):
        """
        Compute the FoilGeometry control points in frd.

        FIXME: descibe/define "control points"

        Parameters
        ----------
        delta_a : float or array of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application

        Returns
        -------
        r_CP2LE : array of floats, shape (K,3) [meters]
            The control points in frd coordinates
        """
        r_LE2RM = self.c_0 * self.lines.r_RM2LE(delta_a)
        foil_cps = self.foil_aerodynamics.control_points()
        line_cps = self.lines.control_points() * self.c_0
        return np.vstack((foil_cps, line_cps))

    def r_RM2LE(self, delta_a=0):
        """
        Compute the position of the riser midpoint `RM` in frd coordinates.

        Parameters
        ----------
        delta_a : array_like of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application. Default: 0

        Returns
        -------
        r_RM2LE : array of float, shape (N,3) [meters]
            The riser midpoint `RM` with respect to the canopy origin.
        """
        return self.lines.r_RM2LE(delta_a) * self.c_0

    def mass_properties(self, rho_air, delta_a=0):
        """
        Compute the inertial properties of the wing about `RM`.

        Includes terms for the solid mass, the enclosed air, and the apparent
        mass (which appears due to the inertial acceleration of the air).

        FIXME: make the reference point a parameter?

        Parameters
        ----------
        rho_air : float [kg/m^3]
            Air density
        delta_a : float [percentage], optional
            Fraction of accelerator application, from 0 to 1

        Returns
        -------
        dictionary
            m_s : float [kg]
                The solid mass of the wing
            r_S2LE : array of float, shape (3,) [m]
                Vector from the canopy origin to the solid mass centroid
            r_S2RM : array of float, shape (3,) [m]
                Vector from the reference point to the solid mass centroid
            J_s2S : array of float, shape (3,3) [kg m^2]
                The moment of inertia matrix of the solid mass about its cm
            v : float [m^3]
                The enclosed volume
            r_V2RM : array of float, shape (3,) [m]
                Vector from the reference point to the volume centroid V
            J_v2V : array of float, shape (3,3) [m^2]
                The moment of inertia matrix of the volume about its centroid
            m_air : float [kg m^3]
                The enclosed air mass.
            r_PC2RC : array of float, shape (3,) [m]
                Vector to the pitch center from the roll center
            r_RC2RM : array of float, shape (3,) [m]
                Vector to the roll center from the riser connection point
            A_a2RM : array of float, shape (6,6)
                The apparent inertia matrix of the volume about `RM`
        """
        r_LE2RM = -self.r_RM2LE(delta_a)
        mp = self._mass_properties.copy()
        mp["r_S2RM"] = mp["r_S2LE"] + r_LE2RM
        mp["r_V2RM"] = mp["r_V2LE"] + r_LE2RM
        mp["m_air"] = mp["v"] * rho_air

        # Apparent moment of inertia matrix about `RM` (Barrows Eq:25)
        ai = self._apparent_inertia  # Dictionary of precomputed values
        S2 = np.diag([0, 1, 0])  # "Selection matrix", Barrows Eq:15
        r_RC2RM = r_LE2RM + ai["r_RC2LE"]
        S_PC2RC = crossmat(ai["r_PC2RC"])
        S_RC2RM = crossmat(r_RC2RM)
        Q = S2 @ S_PC2RC @ ai["M"] @ S_RC2RM
        J_a2RM = (  # Barrows Eq:25
            ai["I"]
            - S_RC2RM @ ai["M"] @ S_RC2RM
            - S_PC2RC @ ai["M"] @ S_PC2RC @ S2
            - Q
            - Q.T
        )
        MC = -ai["M"] @ (S_RC2RM + S_PC2RC @ S2)
        A_a2RM = np.block([[ai["M"], MC], [MC.T, J_a2RM]])  # Barrows Eq:27

        # The vectors to the roll and pitch centers are required to compute the
        # apparent inertias. See Barrows Eq:16 and Eq:24.
        mp["r_RC2RM"] = r_RC2RM
        mp["r_PC2RC"] = ai["r_PC2RC"]
        mp["A_a2RM"] = A_a2RM * rho_air

        return mp
