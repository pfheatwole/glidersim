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


@njit
def cross3(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.empty(3)
    a1, a2, a3 = np.asarray(vec1, dtype=np.double)
    b1, b2, b3 = np.asarray(vec2, dtype=np.double)
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result
