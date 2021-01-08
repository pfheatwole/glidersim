"""A collection of useful functions when working with glider simulations."""

import numpy as np
from scipy.interpolate import interp1d


__all__ = [
    "linear_control",
    "CircularThermal",
    "HorizontalShear",
    "LateralGust",
]


def __dir__():
    return __all__


def linear_control(pairs):
    """
    Helper function to build linear interpolators for control inputs.

    The input is a sequence of tuples encoding  `(duration, value)`. An initial
    value can be set with a leading `(0, initial_value)` tuple. To "hold" a
    value, use `None` to repeat the previous value.

    For example, to ramp from 0 to 0.5 over the initial 15 seconds, then
    transition to 0.75 over a period of 2 seconds, hold for 10 seconds, then
    decrease to 0 over 10 seconds:

        pairs = [(15, 0.5), (2, 0.75), (10, None), (10, 0)]

    Parameters
    ----------
    pairs : list of 2-tuples of float
        Each tuple is (duration, value).
    """
    durations = np.array([t[0] for t in pairs])
    values = [t[1] for t in pairs]
    assert all(durations >= 0)
    for n, v in enumerate(values):  # Use `None` for "hold previous value"
        values[n] = v if v is not None else values[n - 1]
    times = np.cumsum(durations)
    c = interp1d(times, values, fill_value=(values[0], values[-1]), bounds_error=False)
    return c


# ---------------------------------------------------------------------------
# Simple wind "models"


class CircularThermal:
    """
    Functor to create circular thermals at specific <x,y> coordinates.

    Parameters
    ----------
    px, py : float [m]
        The x and y coordinates of the thermal center
    mag : float [m/s]
        The magnitude of the thermal center
    radius95 : float [m]
        The distance from the center where the magnitude has dropped to 5%
    t_start : float [sec], optional
        The time the output magnitudes switches from zero to `mag.`
    """

    def __init__(self, px, py, mag, radius5, t_start=0):
        self.c = np.array([px, py])
        self.mag = mag
        self.R = -(radius5 ** 2) / np.log(0.05)
        self.t_start = t_start

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        d2 = ((self.c - r[..., :2]) ** 2).sum(axis=1)
        wind = np.zeros(r.shape)
        if t > self.t_start:
            wind[..., 2] = self.mag * np.exp(-d2 / self.R)
        return wind


class HorizontalShear:
    """
    Functor to create increasing vertical wind when traveling north.

    Transitions from 0 to `mag` as a sigmoid function. The transition is
    stretch using `smooth`.

    Parameters
    ----------
    x_start : float [m]
        Northerly position to begin the sigmoid transition.
    mag : float [m/s]
        The peak vertical windspeed.
    smooth : float
        Scaling factor to stretch the transition. FIXME: explain (I forget!)
    t_start : float [sec], optional
        The time the output magnitudes switches from zero to `mag.`
    """

    def __init__(self, t_start, x_start, mag, smooth):
        self.t_start = t_start
        self.x_start = x_start
        self.mag = mag
        self.smooth = smooth

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        d = r[..., 0] - self.x_start
        wind = np.zeros(r.shape)
        if t > self.t_start:
            wind[..., 2] = (  # Sigmoid
                self.mag * np.exp(d / self.smooth) / (np.exp(d / self.smooth) + 1)
            )
        return wind


class LateralGust:
    """
    Functor to create a global east-to-west gust with linear ramps up and down.

    Parameters
    ----------
    t_start : float [sec]
        Time to start the ramp up.
    t_ramp : float [sec]
        Time duration for the linear ramps up/down to/from peak magnitude.
    t_duration : float [sec]
        Time to hold the maximum magnitude gust.
    mag : float [m/s]
        The peak gust magnitude.
    """

    def __init__(self, t_start, t_ramp, t_duration, mag):
        t0 = 0
        t1 = t_start  # Start the ramp-up
        t2 = t1 + t_ramp  # Start the hold
        t3 = t2 + t_duration  # Start the ramp-down
        t4 = t3 + t_ramp  # Finish the ramp down
        times = [t0, t1, t2, t3, t4]
        values = [0, 0, mag, mag, 0]
        self._func = interp1d(times, values, bounds_error=False, fill_value=0)

    def __call__(self, t, r):
        wind = np.zeros(r.shape)
        wind[..., 1] = self._func(t)
        return wind
