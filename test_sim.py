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


def compute_dynamics(glider, state, v_w2e, extra):
    J, J_inv = extra['J'], extra['J_inv']
    rho_air = extra['rho_air']
    Gamma = extra['Gamma']

    g = quaternion.apply_quaternion_rotation(state['q'], [0, 0, 1])
    # g = [0, 0, 0]  # Disable the gravity force

    # FIXME: compute the relative wind from v_w2e

    v_frd = quaternion.apply_quaternion_rotation(state['q'], state['v'])
    if not np.isclose(norm(state['v']), norm(v_frd)):
        print("The velocity norms don't match")
        embed()

    # FIXME: Paraglider should return accelerations directly, not forces
    F, M, Gamma = glider.forces_and_moments(v_frd, state['omega'],
                                            g, rho=rho_air, Gamma=Gamma)

    # FIXME: what if Phillips fails? How do I abort gracefully?

    # Translational acceleration
    a_frd = F/glider.harness.mass  # FIXME: crude, incomplete

    # Angular acceleration
    #  * ref: Stevens, Eq:1.7-5, p36 (50)
    alpha = J_inv @ (M - np.cross(state['omega'], J @ state['omega']))

    q_inv = state['q'] * [1, -1, -1, -1]
    a_ned = quaternion.apply_quaternion_rotation(q_inv, a_frd)
    if not np.isclose(norm(a_ned), norm(a_frd)):
        print("The acceleration norms don't match")
        embed()

    x_dot = {'v': state['v'], 'a': a_ned, 'alpha': alpha, 'omega': state['omega']}
    extra = {'F': F, 'M': M, 'Gamma': Gamma}

    return x_dot, extra


def step(x, x_dot, dt):
    # State update
    P, Q, R = x_dot['omega']
    Phi = dt * np.array([  # ref: Stevens, Eq:1.8-15, p51 (65)
        [0, -P, -Q, -R],
        [P,  0,  R, -Q],
        [Q, -R,  0,  P],
        [R,  Q, -P,  0]])
    Phi = -Phi  # FIXME: Merwe uses the negative? Related to q_LG vs q_GL?
    nu = np.linalg.norm(Phi[0])  # Merwe, Eq:B.19, p366 (385)
    alpha = 0.5  # This is *NOT* an angular acceleration
    s = alpha * nu  # Merwe, p368 (387)

    # Merwe, Eq:5.31 (derived in Appendix B.4)
    #  * `np.sinc` uses the normalized sinc, so divide by pi
    q_upd = np.eye(4)*np.cos(s) - alpha * Phi * np.sinc(s/np.pi)

    if not np.isclose(np.linalg.det(q_upd), 1):
        print("The quaternion update matrix is not orthogonal")

    # FIXME: brute force normalizing the quaternion
    q_next = q_upd @ x['q']
    q_next = q_next / np.linalg.norm(q_next)

    x_next = np.empty((), dtype=state_dtype)
    x_next['q'] = q_next
    x_next['p'] = x['p'] + x_dot['v'] * dt  # FIXME: correct? no `1/2 a t^2`?
    x_next['v'] = x['v'] + x_dot['a'] * dt
    x_next['omega'] = x['omega'] + x_dot['alpha'] * dt

    return x_next


def simulate(glider, state0, v_w2e, num_steps=200, dt=0.1, rho_air=1.2):
    """

    Returns
    -------
    path : ndarray (num_steps, N)
        N : the number of state variables
    """

    path = np.empty(num_steps+1, dtype=state_dtype)  # The sequence of states

    J = glider.wing.inertia(rho_air=rho_air, N=5000)
    J_inv = np.linalg.inv(J)

    path[0] = state0
    Gamma = None
    extra = {'J': J, 'J_inv': J_inv, 'rho_air': rho_air, 'Gamma': Gamma}

    t_start = time.time()
    for k in range(num_steps):
        # Version 1: Euler
        # xdot, e = compute_dynamics(glider, path[k], v_w2e, extra)

        # Version 2: RK4
        k1, _ = compute_dynamics(glider, path[k], v_w2e, extra)
        x2 = step(path[k], k1, dt/2)
        k2, _ = compute_dynamics(glider, x2, v_w2e, extra)
        x3 = step(path[k], k2, dt/2)
        k3, _ = compute_dynamics(glider, x3, v_w2e, extra)
        x4 = step(path[k], k3, dt)
        k4, e = compute_dynamics(glider, x4, v_w2e, extra)
        xdot = {k: (k1[k] + 2*k2[k] + 2*k3[k] + k4[k])/6 for k in k1}

        # FIXME: needs a graceful failure mode

        next_state = step(path[k], xdot, dt)
        path[k+1] = next_state
        extra['Gamma'] = e['Gamma']  # Reuse solutions to improve convergence

        if k % 50 == 0:
            avg_rate = (k+1) / (time.time() - t_start)
            rem = (num_steps - k) / avg_rate
            msg = f"ETA: {int(rem // 60)}m{int(rem % 60):02d}s"
        print(f"\rStep: {k} (t = {k*dt:.2f}). {msg}", end="")
    print()

    # For verification purposes. `delta_E` should be strictly decreasing
    delta_PE = 9.8 * 75 * -path['p'].T[2]  # PE = mgh
    KE_trans = 1/2 * 75 * norm(path['v'], axis=1)**2  # translational KE
    KE_rot = 1/2 * np.einsum('ij,kj->k', J, path['omega']**2)
    delta_E = delta_PE + (KE_trans - KE_trans[0]) + KE_rot

    embed()

    return path


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
    state0 = np.empty((), dtype=state_dtype)
    state0['q'] = q
    state0['p'] = [0, 0, 0]
    state0['v'] = quaternion.apply_quaternion_rotation(q_inv, UVW)
    state0['omega'] = PQR

    v_w2e = [0, 0, 0]
    # path = simulate(glider, state0, v_w2e, num_steps=250, dt=0.1)
    path = simulate(glider, state0, v_w2e, num_steps=50, dt=.25)

    # embed()


if __name__ == "__main__":
    main()
