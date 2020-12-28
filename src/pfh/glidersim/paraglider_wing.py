"""FIXME: add module docstring."""

import numpy as np

from scipy.optimize import root_scalar

from pfh.glidersim import foil
from pfh.glidersim.util import cross3, crossmat


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
    force_estimator : foil.ForceEstimator
        Estimator for the aerodynamic forces and moments on the canopy.

    Notes
    -----
    The ParagliderWing coordinate axes are parallel to the canopy axes, but the
    origin is translated from `LE`, the central leading edge of the canopy, to
    `R`, the midpoint between the two riser connections.
    """

    def __init__(
        self, lines, canopy, rho_upper, rho_lower, force_estimator,
    ):
        self.lines = lines
        self.canopy = canopy
        self.rho_upper = rho_upper
        self.rho_lower = rho_lower
        self.force_estimator = force_estimator
        self.c_0 = canopy.chord_length(0)  # Scales the line geometry

        # Compute the canopy mass properties in canopy coordinates
        # cmp = self.canopy.mass_properties(N=5000)  # Assumes `delta_a = 0`
        cmp = self.canopy.mass_properties2(101, 101)

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
        J_upper = cmp["upper_inertia"] * self.rho_upper
        J_lower = cmp["lower_inertia"] * self.rho_lower
        m_solid = m_upper + m_lower
        cm_solid = (
            m_upper * cmp["upper_centroid"] + m_lower * cmp["lower_centroid"]
        ) / m_solid
        Ru = cm_solid - cmp["upper_centroid"]
        Rl = cm_solid - cmp["lower_centroid"]
        Du = (Ru @ Ru) * np.eye(3) - np.outer(Ru, Ru)
        Dl = (Rl @ Rl) * np.eye(3) - np.outer(Rl, Rl)
        J_solid = J_upper + m_upper * Du + J_lower + m_lower * Dl
        self._mass_properties = {
            "m_solid": m_solid,
            "cm_solid": cm_solid,  # In canopy coordinates
            "J_solid": J_solid,
            "m_air": cmp["volume"],  # Normalized by unit air density
            "cm_air": cmp["volume_centroid"],  # In canopy coordinates
            "J_air": cmp["volume_inertia"],  # Normalized by unit air density
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
          * origin: `o` is now `R` (the riser connection point)

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
        # requires that the origin `R` lies in the xz-plane of symmetry.
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
        # leading edge), but the final apparent inertia matrix is about `R`,
        # which depends on `delta_a`.
        r_C2LE = np.array([-0.5 * self.c_0, 0, r])
        r_RC2C = np.array([0, 0, z_RC2C])
        self._apparent_inertia = {
            "r_RC2LE": r_RC2C + r_C2LE,
            "r_PC2RC": np.array([0, 0, z_PC2RC]),
            "M": np.diag([m11, m22, m33]),  # Barrows Eq:1
            "I": np.diag([I11, I22, I33]),  # Barrows Eq:17
        }

    def forces_and_moments(
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
            Aerodynamic forces and moments for each section.
        solution : dictionary, optional
            FIXME: docstring. See `Phillips.__call__`
        """
        # FIXME: (1) duplicates `self.control_points`
        #        (2) only used to get the shapes of the control point arrays
        foil_cps = self.force_estimator.control_points()
        line_cps = self.c_0 * self.lines.control_points()

        # FIXME: uses "magic" indexing established in `self.control_points()`
        K_foil = foil_cps.shape[0]
        K_lines = line_cps.shape[0]

        # Support automatic broadcasting if v_W2b.shape == (3,)
        v_W2b = np.broadcast_to(v_W2b, (K_foil + K_lines, 3))
        v_W2b_foil = v_W2b[:-K_lines]
        v_W2b_lines = v_W2b[-K_lines:]

        delta_f = self.lines.delta_f(
            self.force_estimator.s_cps, delta_bl, delta_br,
        )  # FIXME: leaky, don't grab `s_cps` directly
        dF_foil, dM_foil, solution = self.force_estimator(
            delta_f, v_W2b_foil, rho_air, reference_solution,
        )

        dF_lines, dM_lines = self.lines.forces_and_moments(v_W2b_lines, rho_air)
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
        rho_air,
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
        rho_air : float [kg/m^3]
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
        r_CP2R = self.control_points(delta_a) - self.r_R2LE(delta_a)

        def target(alpha):
            v_W2b = -v_mag * np.array([np.cos(alpha), 0, np.sin(alpha)])
            dF_wing, dM_wing, _ = self.forces_and_moments(
                delta_a, delta_b, delta_b, v_W2b, rho_air, reference_solution,
            )
            M = dM_wing.sum(axis=0) + cross3(r_CP2R, dF_wing).sum(axis=0)
            return M[1]  # Wing pitching moment
        x0, x1 = np.deg2rad([alpha_0, alpha_1])
        res = root_scalar(target, x0=x0, x1=x1)
        if not res.converged:
            raise foil.ForceEstimator.ConvergenceError
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
        r_LE2R = self.c_0 * self.lines.r_R2LE(delta_a)
        foil_cps = self.force_estimator.control_points()
        line_cps = self.lines.control_points() * self.c_0
        return np.vstack((foil_cps, line_cps))

    def r_R2LE(self, delta_a=0):
        """
        Compute the position of the riser midpoint `R` in frd coordinates.

        Parameters
        ----------
        delta_a : array_like of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application. Default: 0

        Returns
        -------
        r_R2LE : array of float, shape (N,3) [meters]
            The riser midpoint `R` with respect to the canopy origin.
        """
        return self.lines.r_R2LE(delta_a) * self.c_0

    def mass_properties(self, rho_air, delta_a=0):
        """
        Compute the inertial properties of the wing about `R`.

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
            m_solid : float [kg]
                The solid mass of the wing
            cm_solid : array of float, shape (3,) [m]
                The solid mass centroid
            J_solid : array of float, shape (3,3) [kg m^2]
                The moment of inertia matrix of the solid mass about its cm
            m_air : float [kg m^3]
                The enclosed air mass.
            cm_air : array of float, shape (3,) [m]
                The air mass centroid
            J_air : array of float, shape (3,3) [m^2]
                The moment of inertia matrix of the enclosed air mass about its cm
            r_PC2RC : array of float, shape (3,) [m]
                Vector to the pitch center from the roll center
            r_RC2R : array of float, shape (3,) [m]
                Vector to the roll center from the riser connection point
            A_a2R : array of float, shape (6,6)
                The apparent inertia matrix of the volume about `R`
        """
        r_LE2R = -self.r_R2LE(delta_a)
        mp = self._mass_properties.copy()
        mp["cm_solid"] = r_LE2R + mp["cm_solid"]
        mp["cm_air"] = r_LE2R + mp["cm_air"]
        mp["m_air"] = mp["m_air"] * rho_air
        mp["J_air"] = mp["J_air"] * rho_air

        # Apparent moment of inertia matrix about `R` (Barrows Eq:25)
        ai = self._apparent_inertia  # Dictionary of precomputed values
        S2 = np.diag([0, 1, 0])  # "Selection matrix", Barrows Eq:15
        r_RC2R = r_LE2R + ai["r_RC2LE"]
        S_PC2RC = crossmat(ai["r_PC2RC"])
        S_RC2R = crossmat(r_RC2R)
        Q = S2 @ S_PC2RC @ ai["M"] @ S_RC2R
        J_a2R = (  # Barrows Eq:25
            ai["I"]
            - S_RC2R @ ai["M"] @ S_RC2R
            - S_PC2RC @ ai["M"] @ S_PC2RC @ S2
            - Q
            - Q.T
        )
        MC = -ai["M"] @ (S_RC2R + S_PC2RC @ S2)
        A_a2R = np.block([[ai["M"], MC], [MC.T, J_a2R]])  # Barrows Eq:27

        # The vectors to the roll and pitch centers are required to compute the
        # apparent inertias. See Barrows Eq:16 and Eq:24.
        mp["r_RC2R"] = r_RC2R
        mp["r_PC2RC"] = ai["r_PC2RC"]
        mp["A_a2R"] = A_a2R * rho_air

        return mp
