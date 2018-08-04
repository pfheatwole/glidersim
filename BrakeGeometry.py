import abc

import numpy as np


class BrakeGeometry(abc.ABC):
    def __call__(self, s, delta_Bl, delta_Br):
        """
        Computes the trailing edge deflection due to the application of brakes.

        Note: this is a tentative API. See the "Notes".

        Parameters
        ----------
        s : float, or array_like of float, shape (N,)
            Normalized span position, where `-1 <= s <= 1`
        delta_Bl : float [percentage]
            Left brake application as a fraction of maximum braking
        delta_Br : float [percentage]
            Right brake application as a fraction of maximum braking

        Returns
        -------
        delta : float [radians]
            The angle of the deflected chord to the nominal chord

        Notes
        -----
        This parametrization is peculiar in that it returns a deflection angle
        without knowing the spanwise chord. This is not how a true brake line
        geometry would work (which deals with distances, and only indirectly
        with the deflection angles), but it is simple to work with and does
        not depend on the chord distribution of the parafoil it is paired with.

        The downside is that plotting the deflection angle distribution is not
        indicative of the deflection distance distribution. Because the wing is
        tapered towards the wing tips, the deflection distance would need to
        decrease for a given deflection angle.

        Because of this peculiar simplification, this API should not be
        considered final. It may be useful to switch to a brake geometry that
        returns distances instead of angles.
        """


class Cubic(BrakeGeometry):
    """
    Implements a brake deflection distribution using a cubic function.

    The wing root typically experiences very little (if any) brake deflection,
    so this parametrization allows for zero deflection until a y-axis distance
    `p_start` from the wing root.

    Also, the maximum delta does not typically occur at the wing tip, but at a
    point closer to the 3/4 semispan. This parametrization uses `p_peak` to
    control where that maximum deflection occurs.

    This is not intended to be a high fidelity model of a real brake geometry,
    just a reasonable placeholder.

    Notes
    -----
    A normal cubic goes like `ay**3 + by**2 + cy + d = 0`, but constraining the
    function to be zero at the origin forces d=0, and constraining the first
    derivative to be zero when evaluated at the origin forces c=0. Thus, you're
    left with just the cubic and quadratic terms. The second two constraints
    are `f(p_peak) = delta_max` and `(df/dy)|p_peak = 0`.

    """

    def __init__(self, p_start, p_peak, delta_max):
        """
        Parameters
        ----------
        p_start : float
            The start of the brake deflection as a fraction of the semispan.
        p_peak : float
            The point of maximum delta as a fraction of the semispan. To
            prevent negative deflections at the wing tips, `p_peak` has a
            minimum value; see `p_peak_min`
        delta_max : float [radians]
            The maximum deflection angle, which occurs at p_peak
        """
        self.p_start = p_start
        self.p_peak = p_peak
        if p_peak < self.p_peak_min(p_start):
            raise ValueError(
                "p_peak is too small: delta is negative at the wing tips")
        self.K1 = -2*delta_max/((p_peak - p_start)**3)
        self.K2 = 3*delta_max/(p_peak - p_start)**2

    def __call__(self, s, delta_Bl, delta_Br):
        fraction = np.choose(s < 0, [delta_Br, delta_Bl])
        s = np.abs(s)  # left and right side are symmetric
        fraction = fraction * (s > self.p_start)  # Boolean hack
        p = s - self.p_start  # The cubic uses a shifted origin, `p_start`
        return fraction * (self.K1 * p**3 + self.K2 * p**2)

    @staticmethod
    def p_peak_min(p_start):
        """The minimum value of `p_peak` to avoid negative deflections."""
        return (2/3)*(1 - p_start) + p_start

    # def find_delta_max(p_start, p_peak, parafoil, max_flap=50, xhinge=0.8):
    #     """ Helper for finding `delta_max` *distance* (not currently used)
    #
    #     If the brake geo returns distances, then just because the maximum
    #     deflection at p_peak doesn't violate the airfoil's max delta *angle*
    #     doesn't mean that (due to wing taper) the brake geo deflection
    #     distances won't violate the max airfoil delta angle somewhere else.
    #
    #     This is a crude helper to determine the largest p_peak deflection
    #     distance (`delta_max`) that parametrizes the curve such that it won't
    #     violate the max delta angle *anywhere* on the wing.
    #
    #     I switched to returning delta angles directly, so this is not
    #     currently being used, but it might prove useful someday.
    #     """
    #     max_flap = np.deg2rad(max_flap)
    #     s = np.linspace(-1, 1, 1000)
    #     deflection_radius = (1-xhinge) * parafoil.planform.fc(s)
    #
    #     brakes = Cubic(parafoil.b, p_start, p_peak, delta_max=1)
    #     deflection_magnitude = brakes(parafoil.fy(s), 1, 1)
    #
    #     # Give a 1% margin of error
    #     return 0.99 * max_flap / max(deflection_magnitude/deflection_radius)
