"""Utility functions for manipulating orientation encodings."""

import numpy as np
from numba import float64, guvectorize

from pfh.glidersim.util import _cross3, crossmat


__all__ = [
    "dcm_to_euler",
    "euler_to_dcm",
    "euler_to_quaternion",
    "quaternion_to_dcm",
    "quaternion_to_euler",
    "quaternion_product",
    "quaternion_rotate",
]


def __dir__():
    return __all__


def euler_to_dcm(euler, intrinsic: bool = True):
    """
    Convert a set of yaw-pitch-roll Tait-Bryan angles to a DCM.

    Intrinsic rotation: z-y'-x''
    Extrinsic rotation: z-y-x

    Parameters
    ----------
    euler : array of float, shape (...,3) [radians]
        The (roll, pitch, yaw) angles of a yaw-pitch-roll sequence.
    intrinsic : bool, optional
        Whether the sequence encodes an intrinsic or extrinsic rotation.

        FIXME: explain how extrinsic rotations are useful for computing vectors
        in wind axes: `C_w2b = euler_to_dcm([0, -alpha, beta])`

    Returns
    -------
    ndarray of float, shape (...,3,3)
        The direction cosine matrix that would apply the given rotation
    """
    if np.shape(euler)[-1] != 3:
        raise ValueError("The last dimension of `euler` must be 3")
    se = np.sin(euler)
    ce = np.cos(euler)
    sp, st, sg = se[..., 0], se[..., 1], se[..., 2]
    cp, ct, cg = ce[..., 0], ce[..., 1], ce[..., 2]

    if intrinsic:  # Encode a z-y'-x'' sequence
        dcm = [
            [ct * cg, ct * sg, -st],
            [-cp * sg + sp * st * cg, cp * cg + sp * st * sg, sp * ct],
            [sp * sg + cp * st * cg, -sp * cg + cp * st * sg, cp * ct],
        ]
    else:  # Encode a z-y-x sequence
        dcm = [
            [cg * ct, sg * cp + cg * st * sp, sg * sp - cg * st * cp],
            [-sg * ct, cg * cp - sg * st * sp, cg * sp + sg * st * cp],
            [st, -ct * sp, ct * cp],
        ]
    return np.moveaxis(dcm, [0, 1], [-2, -1])


def dcm_to_euler(dcm, intrinsic: bool = True):
    """
    Convert a DCM to a set of yaw-pitch-roll Tait-Bryan angles.

    Intrinsic rotation: z-y'-x''
    Extrinsic rotation: z-y-x

    Parameters
    ----------
    dcm : ndarray of float, shape (...,3,3)
        A direction cosine matrix.
    intrinsic : bool, optional
        Whether the sequence encodes an intrinsic or extrinsic rotation.

    Returns
    -------
    array of float, shape (...,3) [radians]
        The (roll, pitch, yaw) angles of a yaw-pitch-roll sequence.
    """
    if np.shape(dcm)[-2:] != (3, 3):
        raise ValueError("The last two dimensions of `dcm` must be (3, 3)")
    if intrinsic:
        # ref: Stevens Eq:1.3-11, pg12 (26)
        phi = np.arctan2(dcm[1, 2], dcm[2, 2])
        theta = -np.arcsin(dcm[0, 2])
        gamma = np.arctan2(dcm[0, 1], dcm[0, 0])
    else:
        theta = np.arcsin(dcm[2, 0])
        phi = np.arccos(dcm[2, 2] / np.cos(theta))
        gamma = np.arccos(dcm[0, 0] / np.cos(theta))
    return np.stack([phi, theta, gamma], axis=-1)


def euler_to_quaternion(euler):
    """
    Convert a set of intrinsic yaw-pitch-roll Tait-Bryan angles to a quaternion.

    Parameters
    ----------
    euler : array of float, shape (3,) [radians]
        The (roll, pitch, yaw) angles of a yaw-pitch-roll sequence.

    Returns
    -------
    q : array of float, shape (4,)
        The quaternion that encodes the given rotation
    """
    euler = np.asfarray(euler)
    sp, st, sg = np.sin(euler / 2)
    cp, ct, cg = np.cos(euler / 2)

    # ref: Stevens, Equation on pg 52 (66)
    # fmt: off
    q = np.asfarray([cp * ct * cg + sp * st * sg,
                     sp * ct * cg - cp * st * sg,
                     cp * st * cg + sp * ct * sg,
                     cp * ct * sg - sp * st * cg])
    # fmt: on
    return q


def quaternion_to_dcm(q):
    """Convert a quaternion to a DCM."""
    assert q.shape in ((4,), (4, 1))
    q = q.reshape(4, 1)
    qw, qv = q[0], q[1:]

    # ref: Stevens, Eq:1.8-16, pg 53 (67)
    dcm = 2 * qv @ qv.T + (qw**2 - qv.T @ qv) * np.eye(3) - 2 * qw * crossmat(qv)

    return np.asfarray(dcm)


def quaternion_to_euler(q):
    """
    Convert a quaternion to a set of intrinsic yaw-pitch-roll Tait-Bryan angles.

    Parameters
    ----------
    q : array_like of float, shape (K,4)
        The components of the quaternion(s)

    Returns
    -------
    v : array_like of float, shape (K,3)
        The [roll, pitch, yaw] angles (phi, theta, gamma) of a Tait-Bryan
        yaw-pitch-roll sequence.
    """
    # assert np.isclose(np.linalg.norm(q), 1)
    q = np.asfarray(q)
    w, x, y, z = q.T

    # ref: Merwe, Eq:B.5:7, p363 (382)
    # FIXME: These assume a unit quaternion?
    # FIXME: verify: these assume the quaternion is `q_local/global`? (Merwe-style?)
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    theta = np.arcsin(-2 * (x * z - w * y))
    gamma = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([phi, theta, gamma]).T


@guvectorize(
    [(float64[:], float64[:], float64[:])],
    "(n),(m)->(m)",
    nopython=True,
    cache=True,
)
def quaternion_rotate(q, u, v):
    """Rotate a 3-vector using a quaternion.

    Treats the vector as a pure quaternion, and returns only the vector part.
    Supports automatic broadcasting of `q` and `u`; the only requirement is
    that the last dimension of `q` is 4 and the last dimension of `u` is 3.

    Parameters
    ----------
    q : array_like, shape (4,)
        The components of the quaternion(s)
    u : array_like, shape (3,)
        The components of the vector(s) to be rotated

    Returns
    -------
    v : ndarray
    """
    if q.shape != (4,):
        raise ValueError("q must be an array_like with q.shape[-1] == 4")
    if u.shape != (3,):
        raise ValueError("u must be an array_like of u.shape[-1] == 3")
    qw, qv = q[0], q[1:]
    # ref: Stevens, Eq:1.8-8, p49
    v[:] = 2 * qv * (qv @ u) + (qw**2 - qv @ qv) * u - 2 * qw * _cross3(qv, u)


@guvectorize(
    [(float64[:], float64[:], float64[:])],
    "(n),(m)->(m)",
    nopython=True,
    cache=True,
)
def quaternion_product(p, q, v):
    r"""
    Multiply two quaternions.

    Parameters
    ----------
    p, q : array_like of float, shape (...,4)
        The quaternions to multiply. Inputs may be ndarrays compatible under
        normal numpy broadcasting rules.

    Returns
    -------
    ndarray of float, shape (...,4)
        The quaternion product, :math:`v = p \ast q`.
    """
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    v[0] = pw * qw - pv @ qv
    v[1:] = pw * qv + qw * pv + np.cross(pv, qv)


def main():

    # The quaternion and euler should produce the same rotation
    while True:
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi / 2, np.pi)
        gamma = np.random.uniform(-np.pi, np.pi)
        euler = [phi, theta, gamma]
        q = euler_to_quaternion(euler)

        # Cross-check the two DCM generation methods
        dcm_e = euler_to_dcm(euler)
        dcm_q = quaternion_to_dcm(q)
        if not np.allclose(dcm_e, dcm_q):
            print("\nThe DCM generated by the quaternion doesn't match\n")
            breakpoint()

        # Transform a random vector using both methods
        u = np.random.random(3)  # A random 3-vector
        ve = dcm_e @ u
        vq = quaternion_rotate(q, u)

        print("\n")
        print(f"euler: {np.rad2deg(euler)}")
        print(f"u:     {u}")
        print("------------------------------------------")
        print(f"ve:    {ve}")
        print(f"vq:    {vq[1:]}")

        if not np.isclose(vq[0], 0):
            print("\nNon-zero scalar part in the quaternion representation")
            breakpoint()

        if not np.allclose(ve, vq[1:]):
            print("\nWrong!")
            breakpoint()


if __name__ == "__main__":
    main()
