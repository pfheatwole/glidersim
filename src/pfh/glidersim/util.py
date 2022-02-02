"""Mathematical utility functions."""

import numpy as np
from numba import float64, guvectorize, njit


__all__ = [
    "trapz",
    "cross3",
    "_cross3",
    "crossmat",
]


def __dir__():
    return __all__


@guvectorize(
    [(float64[:], float64[:], float64[:])],
    "(n),(n)->(n)",
    nopython=True,
    cache=True,
)
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


def crossmat(v):
    """
    Build the cross-product matrix for a vector.

    The cross-product matrix of a vector `v` is a skew-symmetric matrix which
    can be left-mulitiplied by another vector to compute the cross-product
    operation. That is, `crossmat(v) @ u == cross(v, u)`.

    These are useful when building systems of equations since it turns a
    function call into a simple matrix multiplication.

    Parameters
    ----------
    v : array of float, shape (3,) or (3,1)
        The 3-vector for the left-hand term of a cross-product operation.

    Returns
    -------
    array of float, shape (3,3)
        The skew-symmetric matrix that encodes the cross-product opeation.
    """
    assert v.shape in ((3,), (3, 1))
    vx, vy, vz = v.flatten()
    # fmt: off
    sv = [[  0, -vz,  vy],  # noqa: 201, 241
          [ vz,   0, -vx],  # noqa: 201, 241
          [-vy,  vx,   0]]  # noqa: 201, 241
    # fmt: on
    return np.asfarray(sv)
