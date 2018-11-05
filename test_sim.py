import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
from IPython import embed

import Airfoil
import Parafoil
import BrakeGeometry
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider


def build_elliptical_parafoil(b_flat, taper, dMed, sMed, airfoil,
                              SMC=None, MAC=None,
                              dMax=None, sMax=None,
                              torsion_max=0, torsion_exponent=6):

    if SMC is None and MAC is None:
        raise ValueError("One of the SMC or MAC are required")

    if dMed > 0 or (dMax is not None and dMax > 0):
        raise ValueError("dihedral must be negative")

    if sMed < 0 or (sMax is not None and sMax < 0):
        raise ValueError("sweep must be positive")  # FIXME: why?

    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print(f"Using minimum max dihedral ({dMax})")

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print(f"Using minimum max sweep ({sMax})")

    if SMC is not None:
        c0 = Parafoil.EllipticalPlanform.SMC_to_c0(SMC, taper)
    else:
        c0 = Parafoil.EllipticalPlanform.MAC_to_c0(MAC, taper)

    planform = Parafoil.EllipticalPlanform(
        b_flat, c0, taper, sMed, sMax, torsion_exponent, torsion_max)
    lobe = Parafoil.EllipticalLobe(dMed, dMax)

    return Parafoil.ParafoilGeometry(planform, lobe, airfoil)


def build_glider():
    print("\nAirfoil: NACA 24018, curving flap")
    airfoil_geo = Airfoil.NACA5(24018, convention='british')
    airfoil_coefs = Airfoil.GridCoefficients('polars/exp_curving_24018.csv')
    delta_max = np.deg2rad(13.25)  # raw delta_max = 13.38
    airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)

    # Hook3 specs:
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40  # noqa: F841
    SMC_flat = b_flat/AR_flat
    S, b, AR = 19.55, 8.84, 4.00  # noqa: F841

    parafoil = build_elliptical_parafoil(   # Hook 3 (ish)
        b_flat=b_flat, SMC=SMC_flat, taper=0.35, dMed=-32, dMax=-75,
        sMed=13.5, sMax=40, torsion_max=0, airfoil=airfoil)

    p_start, p_peak = 0, 0.75
    brakes = BrakeGeometry.Cubic(p_start, p_peak, delta_max)

    wing = ParagliderWing(parafoil, Parafoil.Phillips, brakes,
                          d_riser=0.49, z_riser=6.8,
                          pA=0.08, pC=0.80,
                          kappa_s=0.15)

    harness = Harness.Spherical(mass=75, z_riser=0.5, S=0.55, CD=0.8)

    glider = Paraglider(wing, harness)

    return glider

# ---------------------------------------------------------------------------


state_dtype = [
    ('q', 'f4', (4,)),
    ('p', 'f4', (3,)),
    ('v', 'f4', (3,)),
    ('omega', 'f4', (3,))]


def apply_quaternion_rotation(q, u):
    # Encodes `v = q^-1 * u * q`, where `*` is the quaternion product,
    # and `u` has been converted to a quaternion, `u = [0, u]`
    #
    # ref: Stevens, Eg:1.8-8, pg 49 (63)
    q = np.asarray(q)
    u = np.asarray(u)
    assert q.shape in ((4,), (4, 1))
    assert u.shape in ((3,), (3, 1))
    q = q.ravel()
    u = u.ravel()

    assert np.isclose(np.linalg.norm(q), 1)

    qw, qv = q[0], q[1:]
    v = np.r_[0, 2*qv*(qv@u) + (qw**2 - qv@qv)*u - 2*qw*np.cross(qv, u)]

    assert np.isclose(v[0], 0)

    return v[1:]  # The `v` quaternion is vector only


def quaternion_product(p, q):
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    embed()
    1/0
    pq = np.array([
        pw*qw - pv@qv,
        pw*qv + qw*pv + np.cross(pv, qv)])
    return pq


def quaternion_to_euler(q):
    # assert np.isclose(np.linalg.norm(q), 1)
    w, x, y, z = q.T

    # FIXME: These assume a unit quaternion!
    phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    theta = np.arcsin(2*(w*y - x*z))
    gamma = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([phi, theta, gamma]).T


def simulate(glider, state0, v_w2e, num_steps=20, dt=0.1, rho_air=1.2):
    """

    Returns
    -------
    path : ndarray (num_steps, N)
        N : the number of state variables
    """

    path = np.empty(num_steps+1, dtype=state_dtype)
    path[0] = state0
    Gamma = None

    J_wing = glider.wing.inertia(rho_air=rho_air, N=5000)

    Fs, Ms, Gammas = [], [], []
    gs = []
    for k in range(num_steps):
        print(f"Step: {k} (t = {k*dt:.2f})")
        cur, next = path[k], path[k+1]  # Views onto the states

        if np.any(np.isnan(cur['q'])):
            print("Things exploded")
            path = path[:k]  # Only keep the good stuff
            break

        g = apply_quaternion_rotation(cur['q'], [0, 0, 1])
        gs.append(g)
        # g = [0, 0, 0]  # Disable the gravity force

        F, M, Gamma = glider.forces_and_moments(
                cur['v'], cur['omega'], g, rho=1.2, Gamma=Gamma)
        acc = F/75  # FIXME: crude, wrong, magic number (75kg)
        alpha = np.linalg.inv(J_wing) @ M
        # print("angular acceleration in deg/s**2:", np.rad2deg(alpha))
        # alpha = 0

        # print("\nReview the F, M, acc, and alpha")
        # embed()

        Fs.append(F)
        Ms.append(M)
        Gammas.append(Gamma)


        # FIXME: the glider should provide the accelerations, not just forces!
        #        I need a new function that computes this, I think?
        # a, alpha, Gamma = glider.accelerations(
        #         cur['v'], cur['omega'], g, rho=rho_air, Gamma=Gamma)


        # State update

        P, Q, R = cur['omega']
        Phi = dt * np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        v = np.linalg.norm(Phi[0])  # Merwe, Eq:B.19, pg 366 (385)
        s = 0.5 * v  # Merwe, pg 368 (387)

        # Modified result from Merwe, Appendix B4
        q_upd = np.eye(4)*np.cos(s) + .5 * Phi * np.sinc(s)

        if not np.isclose(np.linalg.det(q_upd), 1):
            print("The quaternion update matrix is not orthogonal")
            # embed()
            # print("\nQuitting.\n")
            # 1/0

        # FIXME: brute force normalizing the quaternion
        q_next = q_upd @ cur['q']
        q_next = q_next / np.linalg.norm(q_next)

        next['q'] = q_next
        next['p'] = cur['p'] + cur['v'] * dt
        next['v'] = cur['v'] + acc * dt
        next['omega'] = cur['omega'] + alpha * dt

    embed()

    return path  # The complete trajectory


def main():
    glider = build_glider()

    # Build some data
    alpha, beta = np.deg2rad(8), np.deg2rad(0)
    UVW = 10 * np.asarray(
        [cos(alpha)*cos(beta), sin(beta), sin(alpha)*cos(beta)])
    PQR = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]

    # F, M, Gamma = glider.forces_and_moments(UVW, PQR, [0, 0, 0], rho=1)
    # J_wing = glider.wing.inertia(rho_air=1.2, N=5000)
    # alpha_rad = np.linalg.inv(J_wing) @ M
    # print("angular acceleration in deg/s**2:", np.rad2deg(alpha_rad))

    # Choose an initial state
    state0 = np.empty(1, dtype=state_dtype)
    state0['q'] = [1, 0, 0, 0]  # The identity quaternion
    state0['p'] = [0, 0, 0]
    # state0['v'] = [1, 0, 0]
    state0['v'] = UVW
    # state0['omega'] = [0, 0, 0]
    state0['omega'] = PQR

    v_w2e = [0, 0, 0]
    path = simulate(glider, state0, v_w2e, num_steps=50000, dt=0.001)

    embed()


if __name__ == "__main__":
    main()
