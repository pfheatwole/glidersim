"""FIXME: add docstring."""

import numpy as np


class SimpleLineGeometry:
    """
    FIXME: document the design.

    In particular, highlight that everything here is normalized by the length
    of the central chord.

    Parameters
    ----------
    kappa_x : float [percentage]
        The absolute x-coordinate distance from `R` to the canopy origin,
        normalized by the length of the central chord.
    kappa_z : float [m]
        The absolute z-coordinate distance from `R` to the canopy origin,
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
    line_drag_positions : array of float, shape (K,3) [m]
        The mean location(s) of the connecting line surface area(s), normalized
        by the length of the central chord.  If multiple positions are given,
        the total line length will be divided between them evenly.
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
        line_drag_positions,
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
        r_L2R = np.atleast_2d(line_drag_positions)
        if r_L2R.ndim != 2 or r_L2R.shape[-1] != 3:
            raise ValueError("`line_drag_positions` is not a (K,3) array")
        self._r_L2LE = r_L2R - self.canopy_origin(0)
        self._S_lines = total_line_length * average_line_diameter / r_L2R.shape[0]
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

    def canopy_origin(self, delta_a=0):
        """
        Compute the origin of the FoilGeometry coordinate system in frd.

        Parameters
        ----------
        delta_a : array_like of float, shape (N,) [percentage] (optional)
            Fraction of maximum accelerator application. Default: 0

        Returns
        -------
        r_LE2R : array of float, shape (N,3) [meters]
            The canopy origin in wing coordinates.
        """
        # The accelerator shortens the A lines, while C remains fixed
        delta_a = np.asarray(delta_a)
        R_x = (
            (self.A - delta_a * self.kappa_a) ** 2
            - self.C ** 2
            - self.kappa_A ** 2
            + self.kappa_C ** 2
        ) / (2 * (self.kappa_C - self.kappa_A))
        R_y = np.zeros_like(delta_a)
        R_z = np.sqrt(self.C ** 2 - (self.kappa_C - R_x) ** 2)
        r_R2LE = np.array([-R_x, R_y, R_z]).T
        r_LE2R = -r_R2LE
        return r_LE2R  # Normalized by the length of the central chord

    def control_points(self, delta_a=0):
        return self._r_L2LE + self.canopy_origin(delta_a)

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

    def forces_and_moments(self, v_W2b, rho_air):
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
