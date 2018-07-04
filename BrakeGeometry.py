import abc

from scipy.optimize import minimize_scalar  # For `Exponential`
import numpy as np


class BrakeGeometry(abc.ABC):
    def __call__(self, y, delta_Bl, delta_Br):
        """
        Computes the trailing edge deflection due to the application of brakes.

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
            The trailing edge deflection as an absolute distance in meters.
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


class Quadratic(BrakeGeometry):
    """
    FIXME: docstring  (reference the "Exponential" class)

        `delta = k * y**2`

    Constraining the function and its first deritative to be zero at the
    origin means that the linear term and offset are both zero.

    Maximum deflection occurs at the wingtips (y = ±b/2).
    """

    def __init__(self, b, delta_tip):
        """
        Parameters
        ----------
        b : float [meters]
            The total span of the wing
        delta_tip : float [meters]
            The maximum deflection at the wing tip
        """
        self.delta_tip = delta_tip
        self.k = 1 / (b/2)**2

    def __call__(self, y, delta_Bl, delta_Br):
        """
        Parameters
        ----------
        y : float [meters]
            The position along the span
        delta_Bl : float [percentage]
            The percentage of left brake application
        delta_Br : float [percentage]
            The percentage of right brake application
        """
        # This is an abuse of `np.choose` (so the choices are reversed)
        fraction = self.delta_tip * np.choose(y < 0, [delta_Br, delta_Bl])
        delta = fraction * self.k * (y**2)
        return delta


class Cubic(BrakeGeometry):
    """
    FIXME: docstring  (reference the "Exponential" class)

        `delta = K1 * y**3 + K2 * y**2`

    A normal cubic goes like `ay**3 + by**2 + cy + d = 0`, but constraining the
    function to be zero at the origin forces d=0, and constraining the first
    derivative to be zero when evaluated at the origin forces c=0. Thus, you're
    left with just the cubic and quadratic terms.

    If monotonicity is enforced, then maximum deflection occurs at the wingtips
    (y = ±b/2). If monotonicity is not enforced, then the deflection between
    the root and the tip can exceed the total deflection at the tip.

    Warning: if you set `delta_tip` based on the maximum delta available from
    your airfoil model, but do not enforce monotonicity, then you can generate
    deflections that violate your airfoil model.
    """

    def __init__(self, b, p, delta_max, check_monotonic=True):
        # FIXME: if I don't enforce monotonicity, then `delta_max` is misnamed
        p = p * (b/2)  # convert the ratio to the actual point
        self.K2 = delta_max * (1 - (b/2)**3/2) / ((b/2)**2 - (p**2)*((b/2)**3))
        self.K1 = delta_max/2 - self.K2 * (p**2)
        self.delta_max = delta_max

        if check_monotonic:
            # From the first derivative, evaluated at `y = b/2`
            is_monotonic = self.K1 >= -(2*self.K2)/(3 * b/2)
            if not is_monotonic:
                print("BrakeGeometric is not monotonic")

    def __call__(self, y, delta_Bl, delta_Br):
        # This is an abuse of `np.choose` (so the choices are reversed)
        fraction = np.choose(y < 0, [delta_Br, delta_Bl])
        abs_y = np.abs(y)  # left and right side are symmetric
        delta = fraction * (self.K1 * abs_y**3 + self.K2 * y**2)
        return delta


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


    # FIXME: this brake design is awful. Nuke it.


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
        # This is an abuse of `np.choose` (so the choices are reversed)
        magnitude = self.delta_max * np.choose(y < 0, [delta_Br, delta_Bl])
        abs_y = np.abs(y)  # Assume a symmetric brake design on both semiwings
        fraction = np.exp(abs_y*self.tau1) - np.exp(-abs_y*self.tau2)
        return fraction*magnitude  # The total deflection
