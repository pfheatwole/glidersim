"""A scratchpad for orientation-related work."""

from numba import float64, guvectorize
import numpy as np

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


def euler_to_dcm(euler):
    """
    Convert a set of yaw-pitch-role Tait-Bryan angles to a DCM.

    Parameters
    ----------
    euler : array of float, shape (3,) [radians]
        The [phi, theta, gamma] of a yaw-pitch-roll sequence.

    Returns
    -------
    dcm : ndarray of float, shape (3,3)
        The direction cosine matrix that would apply the given rotation
    """
    sp, st, sg = np.sin(euler)
    cp, ct, cg = np.cos(euler)
    dcm = [[ct * cg,                                 ct * sg,     -st],
           [-cp * sg + sp * st * cg,  cp * cg + sp * st * sg, sp * ct],
           [sp * sg + cp * st * cg,  -sp * cg + cp * st * sg, cp * ct]]

    return np.asarray(dcm)


def dcm_to_euler(dcm):
    """
    Convert a DCM to a set of yaw-pitch-role Tait-Bryan angles.

    Parameters
    ----------
    dcm : ndarray of float, shape (3,3)
        A direction cosine matrix.

    Returns
    -------
    euler : array of float, shape (3,) [radians]
        The [phi, theta, gamma] of a yaw-pitch-roll sequence.
    """
    # ref: Stevens Eq:1.3-11, pg12 (26)
    phi = np.arctan2(dcm[1, 2], dcm[2, 2])
    theta = -np.arcsin(dcm[0, 2])
    gamma = np.arctan2(dcm[0, 1], dcm[0, 0])
    return np.array([phi, theta, gamma])


def euler_to_quaternion(euler):
    """
    Convert a set of yaw-pitch-role Tait-Bryan angles to a quaternion.

    Parameters
    ----------
    euler : array of float, shape (3,) [radians]
        The [phi, theta, gamma] of a yaw-pitch-roll sequence.

    Returns
    -------
    q : array of float, shape (4,)
        The quaternion that encodes the given rotation
    """
    euler = np.asarray(euler)
    sp, st, sg = np.sin(euler / 2)
    cp, ct, cg = np.cos(euler / 2)

    # ref: Stevens, Equation on pg 52 (66)
    q = np.asarray([cp * ct * cg + sp * st * sg,
                    sp * ct * cg - cp * st * sg,
                    cp * st * cg + sp * ct * sg,
                    cp * ct * sg - sp * st * cg])
    return q


def quaternion_to_dcm(q):
    """Convert a quaternion to a DCM."""
    assert q.shape in ((4,), (4, 1))
    q = q.reshape(4, 1)
    qw, qv = q[0], q[1:]

    # ref: Stevens, Eq:1.8-16, pg 53 (67)
    dcm = 2 * qv @ qv.T + (qw ** 2 - qv.T @ qv) * np.eye(3) - 2 * qw * crossmat(qv)

    return np.asarray(dcm)


def quaternion_to_euler(q):
    """Convert a quaternion to a set of yaw-pitch-role Tait-Bryan angles.

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
    q = np.asarray(q)
    w, x, y, z = q.T

    # ref: Merwe, Eq:B.5:7, p363 (382)
    # FIXME: These assume a unit quaternion?
    # FIXME: verify: these assume the quaternion is `q_local/global`? (Merwe-style?)
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    theta = np.arcsin(-2 * (x * z - w * y))
    gamma = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([phi, theta, gamma]).T


@guvectorize([(float64[:], float64[:], float64[:])],
             '(n),(m)->(m)', nopython=True, cache=True)
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
    v[:] = 2 * qv * (qv @ u) + (qw ** 2 - qv @ qv) * u - 2 * qw * _cross3(qv, u)


@guvectorize([(float64[:], float64[:], float64[:])],
             '(n),(m)->(m)', nopython=True, cache=True)
def quaternion_product(p, q, v):
    """Multiply two quaternions."""
    # FIXME: document and test
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
