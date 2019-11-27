import abc

import numpy as np


class Harness(abc.ABC):
    """
    FIXME: docstring
    """
    @abc.abstractmethod
    def control_points(self):
        """FIXME: docstring"""

    @abc.abstractmethod
    def forces_and_moments(self, xyz, v_cp2w):
        """
        Calculate the aerodynamic forces at the control points.

        The xyz and v_cp2w mirror that of the ParagliderWing. The idea is
        that this API will support future models with non-isotropic forces
        and non-point mass distributions.
        """


class Spherical(Harness):
    """
    Models the harness as a point mass inside a sphere.

    The mass is concentrated at a single point, and the spherical assumption
    implies isotropic drag, characterized by a single drag coefficient.

    FIXME: finish documentation

    Notes
    -----
    Typical drag coefficient for pilot + harness:
     * Conventional:    0.8
     * Performance:     0.4

    Typical cross-sectional area for pilot + harness:
     * <80kg:           0.5
     * 80kg to 100kg:   0.6
     * >100kg:          0.7

    ref: PFD p85 (93)
    """

    def __init__(self, mass, z_riser, S, CD):
        # FIXME: the pilot is the thing that weight shifts; does the Paraglider
        #        manage that mass? Seems the right decision...

        """

        Parameters
        ----------
        mass : float [kg]
            The mass of the harness, without a pilot
        z_riser : float [m]
            The distance from the risers to the harness center of mass, given
            in a local frd coordinate system with the origin at the risers.
        S : float [m^2]
            The projected area of the sphere (ie, the area of a circle)
        CD : float [FIXME: units?]
            The isotropic drag coefficient
        """
        self.mass = mass
        self.z_riser = z_riser
        self.S = S
        self.CD = CD

    def control_points(self):
        return np.array([0, 0, self.z_riser])

    def forces_and_moments(self, v_cp2w):
        V2 = (v_cp2w**2).sum()
        u_drag = -v_cp2w/np.sqrt(V2)  # Drag force unit vector
        dF = (1/2 * V2 * self.S * self.CD) * u_drag
        dM = np.zeros(3)
        return dF, dM