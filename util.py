from numba import njit
import numpy as np


@njit(cache=True)
def trapz(y, dx):
    # Trapezoidal integrator
    return np.sum(y[1:] + y[:-1]) / 2.0 * dx


def integrate(f, a, b, N):
    if N % 2 == 0:
        raise ValueError("trapezoid integration requires odd N")
    x = np.linspace(a, b, N)
    dx = (b - a)/(N - 1)  # Include the endpoints
    y = f(x)
    return trapz(y, dx)
