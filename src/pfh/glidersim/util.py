from numba import float64, guvectorize, njit
import numpy as np


@njit(cache=True)
def trapz(y, dx):
    # Trapezoidal integrator with regularly spaced sample points
    return np.sum(y[1:] + y[:-1]) / 2.0 * dx


def integrate(f, a, b, N):
    if N % 2 == 0:
        raise ValueError("trapezoid integration requires odd N")
        # Wait, what? Why?
    x = np.linspace(a, b, N)
    dx = (b - a)/(N - 1)  # Include the endpoints
    y = f(x)
    return trapz(y, dx)


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)',
             nopython=True, cache=True)
def cross3(a, b, result):
    """Calculate the cross product of 3d vectors.

    This vectorized version supports automatic broadcasting. The only
    requirement is that the last dimension of both `a` and `b` are 3.

    TODO: this can be eliminated after Numba Github issue #2978 is closed.

    Parameters
    ----------
    a : array_like
        Vector components of the left-hand side
    b : array_like
        Vector components of the right-hand side

    Returns
    -------
    c : ndarray
        Vector cross product(s)
    """
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("All inputs must be 3-vectors")
    a1, a2, a3 = a
    b1, b2, b3 = b
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1


@njit(cache=True)
def _cross3(a, b):
    """Calculate the cross product of two 3d vectors.

    Note that the inputs must be of type ndarray, not array_like.

    This jitted function is for use inside `guvectorize`d functions. At the
    moment, numba does not allow a `guvectorize`d function to call another
    vectorized function; they can only call jitted functions.

    TODO: this can be eliminated after Numba Github issue #2089 is closed.

    Parameters
    ----------
    a : ndarray, shape (3,)
        Vector components of the left-hand side
    b : ndarray, shape (3,)
        Vector components of the right-hand side

    Returns
    -------
    c : ndarray
        Vector cross product(s)
    """
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("All inputs must be 3-vectors")
    a1, a2, a3 = a
    b1, b2, b3 = b
    result = np.empty(3)
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result