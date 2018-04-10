import abc

from scipy.optimize import minimize_scalar  # For `Exponential`
import numpy as np


class BrakeGeometry(abc.ABC):
    def __call__(self, y, delta_Bl, delta_Br):
        """
        Returns the trailing edge deflection due to the application of brakes.

        Parameters
        ----------
        y : float [meters]
            The wing section position on the span, -b/2 <= y <= b/2
        delta_Bl : float [percentage]
            Left brake application as a fraction of maximum braking
        delta_Br : float [percentage]
            Right brake application as a fraction of maximum braking

        Returns
        -------
        delta : float [meters]
            The trailing edge deflection as a fraction of the chord.
        """


class PFD(BrakeGeometry):
    """
    Implements the basic PFD braking design (PFD EQ:4.18, p75)

    FIXME: document
    """

    def __init__(self, b, delta_M, delta_f):
        self.b = b
        self.delta_M = delta_M
        self.delta_f = delta_f

    # def delta(self, y, delta_Bl, delta_Br):
    def __call__(self, y, delta_Bl, delta_Br):
        # FIXME: verify and test
        left = delta_Bl*self.delta_M*(100/self.delta_f)**(-y/self.b - 1/2)
        right = delta_Br*self.delta_M*(100/self.delta_f)**(y/self.b - 1/2)
        return left + right


class Exponential(BrakeGeometry):
    """
    Implements a brake deflection distribution according to:

        `delta = delta_max * (exp(y * tau1) - exp(-y * tau2)`

    The maximum deflection, `delta_max`, occurs at the wingtips (y = ±b/2),
    with an exponential decay to zero at the central chord.

    The two parameters, tau1 and tau2, allow the curve to intersect any desired
    deflection percentage at the midpoint of both semiwings (y = ±b/4).

    Note: this definition is a little confusing because `delta_max` is defined
    as [radians], but because a normalized chord length of 1 is assumed,
    the magnitude of the calculated delta is is equal to the magnitude of the
    deflection. That is, `deflection = radius*delta`, with `radius = 1`, thus
    `delta` in radians is directly usable as the `deflection` in meters.

    I did this because specifying `delta_max` in radians seemed more intuitive.
    """

    def __init__(self, b, p_b4, delta_max):
        """
        Parameters
        ----------
        b : float [meters]
            The total span of the wing
        p_b4 : float [percentage]
            The fraction of delta_max at the quarter-span (`y = b/4`)
        delta_max : float [radians]
            The maximum angle of deflection from the leading edge to the
            deflected trailing edge.
        """
        if (p_b4 < 0) or (p_b4 > 1):
            raise ValueError("p_b4 should be a number between 0 and 1")

        self.b = b
        self.p_b4 = p_b4
        self.delta_max = delta_max

        def _tau1(tau2):
            # Calculate tau1 as a function of tau2
            return (2/self.b)*np.log(1 + np.exp((-self.b/2)*tau2))

        def _target(tau2):
            # Optimization target to numerically solve for tau2
            tau1 = _tau1(tau2)
            residual = np.exp(b/4 * tau1) - np.exp((-b/4)*tau2) - self.p_b4
            return np.abs(residual)

        self.tau2 = minimize_scalar(_target).x  # FIXME: closed-form solution?
        self.tau1 = _tau1(self.tau2)

    def __call__(self, y, delta_Bl, delta_Br):
        magnitude = self.delta_max * np.choose(y < 0, [delta_Bl, delta_Br])
        abs_y = np.abs(y)  # Assume a symmetric brake design on both semiwings
        fraction = np.exp(abs_y*self.tau1) - np.exp(-abs_y*self.tau2)
        return fraction*magnitude  # The total deflection
