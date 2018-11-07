import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
from numpy.linalg import norm
from IPython import embed
import pandas as pd
from scipy.interpolate import UnivariateSpline

import Airfoil
import Parafoil
import BrakeGeometry
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider
import quaternion


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


class FlaplessAirfoilCoefficients(Airfoil.AirfoilCoefficients):
    """
    Uses the airfoil coefficients from a CSV file.
    The CSV must contain the following columns: [alpha, delta, CL, CD, Cm]

    This is similar to `Airfoil.GridCoefficients`, but it assumes that delta
    is always zero. This is convenient, since no assuptions need to be made
    for the non-existent flaps on the wind tunnel model.
    """

    def __init__(self, filename, convert_degrees=True):
        data = pd.read_csv(filename)
        self.data = data

        if convert_degrees:
            data['alpha'] = np.deg2rad(data.alpha)

        self._Cl = UnivariateSpline(data[['alpha']], data.CL, s=0.001)
        self._Cd = UnivariateSpline(data[['alpha']], data.CD, s=0.0001)
        self._Cm = UnivariateSpline(data[['alpha']], data.Cm, s=0.0001)
        self._Cl_alpha = self._Cl.derivative()

    def _clean(self, alpha, val):
        # The UnivariateSpline doesn't fill `nan` outside the boundaries
        min_alpha, max_alpha = np.deg2rad(-9.9), np.deg2rad(24.9)
        mask = (alpha < min_alpha) | (alpha > max_alpha)
        val[mask] = np.nan
        return val

    def Cl(self, alpha, delta):
        return self._clean(alpha, self._Cl(alpha))

    def Cd(self, alpha, delta):
        return self._clean(alpha, self._Cd(alpha))

    def Cm(self, alpha, delta):
        return self._clean(alpha, self._Cm(alpha))

    def Cl_alpha(self, alpha, delta):
        return self._clean(alpha, self._Cl_alpha(alpha))


def build_glider():
    # print("\nAirfoil: NACA 24018, curving flap")
    # airfoil_geo = Airfoil.NACA5(24018, convention='british')
    # airfoil_coefs = Airfoil.GridCoefficients('polars/exp_curving_24018.csv')
    # delta_max = np.deg2rad(13.25)  # raw delta_max = 13.38

    print("\nAirfoil: NACA 23015, flapless (no support for delta)")
    airfoil_geo = Airfoil.NACA5(24018, convention='british')
    airfoil_coefs = FlaplessAirfoilCoefficients(
        'polars/NACA 23015_T1_Re0.920_M0.03_N7.0_XtrTop 5%_XtrBot 5%.csv')
    delta_max = np.deg2rad(0)  # Flapless coefficients don't support delta

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
                          d_riser=0.50, z_riser=6.8,
                          pA=0.08, pC=0.80,
                          kappa_s=0.15)

    # The 6 DoF glider model holds the harness orientation fixed, so if z_riser
    # is non-zero it will introduce an unnatural torque.
    harness = Harness.Spherical(mass=75, z_riser=0.0, S=0.55, CD=0.8)

    glider = Paraglider(wing, harness)

    return glider

# ---------------------------------------------------------------------------


state_dtype = [
    ('q', 'f4', (4,)),
    ('p', 'f4', (3,)),
    ('v', 'f4', (3,)),
    ('omega', 'f4', (3,))]


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

    # ref: Merwe, Eq:B.5:7, p363 (382)
    # FIXME: These assume a unit quaternion?
    # FIXME: verify: these assume the quaternion is `q_local/global`? (Merwe-style?)
    phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    theta = np.arcsin(-2*(x*z - w*y))
    gamma = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([phi, theta, gamma]).T


def simulate(glider, state0, v_w2e, num_steps=200, dt=0.1, rho_air=1.2):
    """

    Returns
    -------
    path : ndarray (num_steps, N)
        N : the number of state variables
    """

    # Preallocate storage for the sequence of states
    path = np.empty(num_steps+1, dtype=state_dtype)

    J = glider.wing.inertia(rho_air=rho_air, N=5000)
    J_inv = np.linalg.inv(J)

    path[0] = state0
    Gamma = None
    Fs, Ms, Gammas, gs = [], [], [], []  # For debugging purposes
    alphas, a_frds, a_neds = [], [], []

    t_start = time.time()
    for k in range(num_steps):
        if k == 0:
            msg = ""
        elif k % 50 == 0:
            avg_rate = k / (time.time() - t_start)
            rem = (num_steps - k) / avg_rate
            msg = f"ETA: {int(rem // 60)}m{int(rem % 60)}s"
        print(f"\rStep: {k} (t = {k*dt:.2f}). {msg}", end="")

        cur, next = path[k], path[k+1]  # Views onto the states
        if np.any(np.isnan(cur['q'])):
            print("Things exploded")
            path = path[:k]  # Discard unpopulated states
            break

        q_inv = cur['q'] * [1, -1, -1, -1]

        g = quaternion.apply_quaternion_rotation(cur['q'], [0, 0, 1])
        # g = [0, 0, 0]  # Disable the gravity force
        gs.append(g)

        # FIXME: compute the relative wind from v_w2e

        v_frd = quaternion.apply_quaternion_rotation(cur['q'], cur['v'])
        if not np.isclose(norm(cur['v']), norm(v_frd)):
            print("The velocity norms don't match")
            embed()

        F, M, Gamma = glider.forces_and_moments(
            v_frd, cur['omega'],
            g, rho=rho_air,
            Gamma=Gamma)

        # Translational acceleration
        a_frd = F/glider.harness.mass  # FIXME: crude, incomplete

        # FIXME: Paraglider should return accelerations directly, not forces

        # State update
        P, Q, R = cur['omega']
        Phi = dt * np.array([  # ref: Stevens, Eq:1.8-15, p51 (65)
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        Phi = -Phi  # FIXME: Merwe is weird (ie, negative); does this work? I think this is related to q_LG vs q_GL.
        nu = np.linalg.norm(Phi[0])  # Merwe, Eq:B.19, p366 (385)
        _alpha = 0.5  # This is *NOT* an angular acceleration
        s = _alpha * nu  # Merwe, p368 (387)

        # Modified result from Merwe, Appendix B4
        q_upd = np.eye(4)*np.cos(s) - _alpha * Phi * np.sinc(s)

        if not np.isclose(np.linalg.det(q_upd), 1):
            print("The quaternion update matrix is not orthogonal")
            # embed()
            # print("\nQuitting.\n")
            # 1/0

        # FIXME: brute force normalizing the quaternion
        q_next = q_upd @ cur['q']
        q_next = q_next / np.linalg.norm(q_next)

        a_ned = quaternion.apply_quaternion_rotation(q_inv, a_frd)
        if not np.isclose(norm(a_ned), norm(a_frd)):
            print("The acceleration norms don't match")
            embed()

        # Angular acceleration
        #  * ref: Stevens, Eq:1.7-5, p36 (50)
        alpha = J_inv @ (M - np.cross(cur['omega'], J @ cur['omega']))

        next['q'] = q_next
        next['p'] = cur['p'] + cur['v'] * dt
        next['v'] = cur['v'] + a_ned * dt
        next['omega'] = cur['omega'] + alpha * dt

        Fs.append(F)
        Ms.append(M)
        Gammas.append(Gamma)
        alphas.append(alpha)
        a_frds.append(a_frd)
        a_neds.append(a_ned)

    print()

    # Debugging stuff
    Fs = np.asarray(Fs)
    Ms = np.asarray(Ms)
    Gammas = np.asarray(Gammas)
    gs = np.asarray(gs)
    alphas = np.asarray(alphas)
    a_frds = np.asarray(a_frds)
    a_neds = np.asarray(a_neds)
    delta_PE = 9.8 * 75 * -path['p'].T[2]  # PE = mgh
    KE_trans = 1/2 * 75 * norm(path['v'], axis=1)**2  # translational KE
    KE_rot = 1/2 * np.einsum('ij,kj->k', J, path['omega']**2)
    delta_E = delta_PE + (KE_trans - KE_trans[0]) + KE_rot  # Should be strictly decreasing

    embed()

    return path  # The complete trajectory


def main():
    glider = build_glider()
    alpha_eq, Theta_eq, V_eq = 0.13964, 0.024184, 10.41  # For deltas=0

    # Build some data
    alpha, beta = np.deg2rad(8.5), np.deg2rad(0)
    UVW = 10.5 * np.asarray(
        [cos(alpha)*cos(beta), sin(beta), sin(alpha)*cos(beta)])
    PQR = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]  # omega [rad/sec]
    Phi = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]  # attitude [rad]

    # q = np.array([1, 0, 0, 0])  # The identity quaternion (zero attitude)
    q = quaternion.euler_to_quaternion(Phi)  # Encodes C_frd/ned
    q_inv = q * [1, -1, -1, -1]              # Encodes C_ned/frd

    # Define the initial state
    state0 = np.empty(1, dtype=state_dtype)
    state0['q'] = q
    state0['p'] = [0, 0, 0]
    state0['v'] = quaternion.apply_quaternion_rotation(q_inv, UVW)
    state0['omega'] = PQR

    v_w2e = [0, 0, 0]
    path = simulate(glider, state0, v_w2e, num_steps=2500, dt=0.01)

    # embed()


if __name__ == "__main__":
    main()
