"""Utility functions for generating glider simulations."""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
from scipy.interpolate import interp1d

import pfh.glidersim as gsim


__all__ = [
    "compute_euler_derivatives",
    "linear_control",
    "CircularThermal",
    "HorizontalShear",
    "LateralGust",
]


def __dir__():
    return __all__


def compute_euler_derivatives(Theta, omega):
    """
    Compute the derivatives of a sequence of Euler angles over time.

    Parameters
    ----------
    Theta : ndarray of float, shape (T,3)
        Euler angles ([roll, pitch, yaw] or [phi, theta, gamma])
    omega : ndarray of float, shape (T,3)
        Angular velocities
    """
    Theta = np.asarray(Theta)
    omega = np.asarray(omega)

    if Theta.ndim != 2:
        raise ValueError("`Theta` must be an ndarray with shape (T,3)")
    if omega.ndim != 2:
        raise ValueError("`omega` must be an ndarray with shape (T,3)")
    if Theta.shape[0] != omega.shape[0]:
        raise ValueError("`Theta` and `omega` must be the same length")

    # Euler derivatives (Stevens Eq:1.4-4)
    K = len(Theta)
    _0, _1 = np.zeros(K), np.ones(K)
    sp, st, sg = np.sin(Theta.T)
    cp, ct, cg = np.cos(Theta.T)
    tp, tt, tg = np.tan(Theta.T)
    M = np.array([[_1, sp * tt, cp * tt], [_0, cp, -sp], [_0, sp / ct, cp / ct]])
    M = np.moveaxis(M, -1, 0)
    Theta_dot = np.einsum("kij,kj->ki", M, omega)
    return Theta_dot


def sample_paraglider_positions(
    model,
    states,
    times,
    samplerate: float = 0.5,
    include_times: bool = False,
):
    """
    Sample reference points on the risers, wing, and payload for plotting.

    All positions are with respect to the global flat-earth origin `O`. The
    state sequence is downsampled (decimated) to approximately `samplerate` to
    avoid excessive resolution when plotting; assumes the states were generated
    at a fixed timestep.

    This function is is not pretty; it was designed specifically for use with
    `plots.plot_3d_simulation_path` and depends heavily on the internal design
    of ParagliderModel6a/ParagliderModel9a, but this is crunch-time.

    Parameters
    ----------
    model : pfh.glidersim.simulator.ParagliderModel6a or pfh.glidersim.simulator.ParagliderModel9a
        The paraglider dynamics model that produced the states.
    states : array of model.state_dtype, shape (K,)
        The state values at each timestep.
    times : array of float, shape (K,) [sec]
        The timestamps.
    samplerate : float [sec], optional
        The desired samplerate. The points were already generated at some fixed
        timestep, so this method discards samples to achieve a new rate as
        close to `samplerate` as possible. If the `samplerate` is not an
        integer multiple of the original timestep then it is rounded to the
        nearest integer multiple.
    include_times : bool, optional
        Whether to include the `times` associated with the (potentially
        resampled) output.

    Returns
    -------
    dict
        times : array of float, shape (M,) [sec]
            The timestamps of the (potentially resampled) positions.
        r_RM2O : array of float, shape (M,3) [m]
            Position of the riser midpoint.
        r_LE2O : array of float, shape (M,3) [m]
            Position of the wing's central leading edge.
        r_P2O : array of float, shape (M,3) [m]
            Position of the payload's default center of mass. This ignores
            displacements due to weight shift because it is easier to visualize
            the orientation of the payload using a fixed reference point.
    """

    q_e2b = states["q_b2e"] * [-1, 1, 1, 1]  # Applies C_ned/frd
    if "q_p2e" in states.dtype.names:  # 9 DoF model
        q_e2p = states["q_p2e"] * [-1, 1, 1, 1]
    else:  # 6 DoF model
        q_e2p = q_e2b

    # Compute the vector from the RM straight up to the central chord
    r_C02RM = -model.paraglider.wing.r_RM2LE(model.delta_a(times))
    r_C02RM[..., 0] = 0  # Zero the x-offsets

    r_RM2O = states["r_RM2O"]
    r_C02O = states["r_RM2O"] + gsim.orientation.quaternion_rotate(q_e2b, r_C02RM)

    # FIXME: assumes the payload has only one control point (r_P2RM^p)
    r_P2RM = model.paraglider.payload.r_CP2RM(delta_w=0)
    r_P2O = states["r_RM2O"] + gsim.orientation.quaternion_rotate(q_e2p, r_P2RM)

    dt = times[1] - times[0]  # Assume a constant simulation timestep
    if samplerate is not None and samplerate > dt:
        samples = np.arange(0, len(times), round(samplerate / dt))
    else:
        samples = np.arange(0, len(times))

    points = {
        "r_RM2O": r_RM2O[samples],
        "r_C02O": r_C02O[samples],
        "r_P2O": r_P2O[samples],
    }
    if include_times:
        points["times"] = times[samples]

    return points


def linear_control(pairs: list[tuple[float, float | None]]) -> Callable:
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
    if len(pairs) == 1 and pairs[0][0] == 0:  # eg, `pairs = [(0, 1)]`
        pairs.append((1, None))  # The initial conditions never change
    durations = np.array([t[0] for t in pairs])
    values = [t[1] for t in pairs]
    assert all(durations >= 0)
    for n, v in enumerate(values):  # Use `None` for "hold previous value"
        values[n] = v if v is not None else values[n - 1]
    times = np.cumsum(durations)
    c = interp1d(
        times,
        values,
        fill_value=(values[0], values[-1]),
        bounds_error=False,
    )
    return cast(Callable, c)


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
    radius5 : float [m]
        The distance from the center where the magnitude has dropped to 5%
    t_enable : float [sec], optional
        The time the output magnitudes switches from zero to `mag.`
    """

    def __init__(
        self,
        px: float,
        py: float,
        mag: float,
        radius5: float,
        t_enable: float = 0,
    ) -> None:
        self.c = np.array([px, py])
        self.mag = mag
        self.R = -(radius5**2) / np.log(0.05)
        self.t_enable = t_enable

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        t = np.asarray(t)
        r = np.asarray(r)
        d2 = ((self.c - r[..., :2]) ** 2).sum(axis=-1)
        wind = np.zeros((*t.shape, *r.shape))
        wind[..., 2] = self.mag * np.exp(-d2 / self.R)
        wind[t < self.t_enable] = [0, 0, 0]
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
    t_enable : float [sec], optional
        The time the output magnitudes switches from zero to `mag.`
    """

    def __init__(
        self,
        x_start: float,
        mag: float,
        smooth: float,
        t_enable: float = 0,
    ) -> None:
        self.t_enable = t_enable
        self.x_start = x_start
        self.mag = mag
        self.smooth = smooth

    def __call__(self, t, r):
        # `t` is time, `r` is 3D position in ned coordinates
        d = r[..., 0] - self.x_start
        wind = np.zeros(r.shape)
        if t > self.t_enable:
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

    def __init__(
        self,
        t_start: float,
        t_ramp: float,
        t_duration: float,
        mag: float,
    ) -> None:
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
