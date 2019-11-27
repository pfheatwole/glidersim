import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
from numpy.linalg import norm
from IPython import embed
import pandas as pd
from scipy.interpolate import UnivariateSpline
import scipy.integrate

import Airfoil
import Parafoil
import BrakeGeometry
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider
import quaternion

from util import cross3


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
    print("\nAirfoil: NACA 24018, curving flap")
    airfoil_geo = Airfoil.NACA(24018, convention='british')
    airfoil_coefs = Airfoil.GridCoefficients('polars/exp_curving_24018.csv')
    delta_max = np.deg2rad(13.25)  # raw delta_max = 13.38

    # print("\nAirfoil: NACA 23015, flapless (no support for delta)")
    # airfoil_geo = Airfoil.NACA(24018, convention='british')
    # airfoil_coefs = FlaplessAirfoilCoefficients(
    #     'polars/NACA 23015_T1_Re0.920_M0.03_N7.0_XtrTop 5%_XtrBot 5%.csv')
    # delta_max = np.deg2rad(0)  # Flapless coefficients don't support delta

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


class GliderSim:
    state_dtype = [
        ('q', float, (4,)),
        ('p', float, (3,)),
        ('v', float, (3,)),
        ('omega', float, (3,))]

    def __init__(self, glider, rho_air):
        self.glider = glider

        # FIXME: assumes J is constant. This is okay if rho_air is constant,
        #        and delta_s is zero (no weight shift deformations)
        self.J = glider.wing.inertia(rho_air=rho_air, N=5000)
        self.J_inv = np.linalg.inv(self.J)

        if np.isscalar(rho_air):
            self.rho_air = rho_air
        else:
            raise ValueError("Non-scalar rho_air is not yet supported")

    def dynamics(self, t, y, params):
        """The state dynamics for the model

        Matches the `f(t, y, *params)` signature for scipy.integrate.ode

        Parameters
        ----------
        t : float [s]
            Time
        y : ndarray of float, shape (N,)
            The array of N state components
        params : dictionary
            Any extra non-state parameters for computing the dynamics. Be aware
            that 'Gamma' an in-out parameter: solutions for the current Gamma
            distribution are passed forward (output) to be used as the proposal
            to the next time step.

        Returns
        -------
        x_dot : ndarray of float, shape (N,)
            The array of N state component derivatives
        """
        x = y.view(self.state_dtype)[0]  # The integrator uses a flat array

        # Determine the environmental conditions
        # rho_air = self.rho_air(t, x['p'])
        rho_air = self.rho_air  # FIXME: support air density functions

        delta_s, delta_Br, delta_Bl = 0, 0, 0  # FIXME: time-varying input

        q_inv = x['q'] * [1, -1, -1, -1]  # for frd->ned
        # cps_frd = self.glider.control_points(delta_s)  # In body coordinates
        # cps = x['p'] + quaternion.apply_quaternion_rotation(x['q'], cps_frd)
        # v_w2e = self.wind(t, cps)  # Lookup the wind at each `ned` coordinate
        v_w2e = 0  # FIXME: implement wind lookups

        g = 9.8 * quaternion.apply_quaternion_rotation(x['q'], [0, 0, 1])
        # g = [0, 0, 0]  # Disable the gravity force

        # FIXME: Paraglider should return accelerations directly, not forces.
        #        The Glider might want to utilize appparent inertia, etc.
        v_frd = quaternion.apply_quaternion_rotation(x['q'], x['v'])
        F, M, Gamma = self.glider.forces_and_moments(
            v_frd, x['omega'], g, rho=rho_air,
            delta_s=delta_s, delta_Bl=delta_Bl, delta_Br=delta_Br,
            Gamma=params['Gamma'])

        # FIXME: what if Phillips fails? How do I abort gracefully?

        # Translational acceleration of the cm
        a_frd = F/self.glider.harness.mass  # FIXME: crude, incomplete
        a_ned = quaternion.apply_quaternion_rotation(q_inv, a_frd)

        # Angular acceleration of the body relative to the ned frame
        #  * ref: Stevens, Eq:1.7-5, p36 (50)
        alpha = self.J_inv @ (M - cross3(x['omega'], self.J @ x['omega']))

        # Quatnernion derivative
        #  * ref: Stevens, Eq:1.8-15, p51 (65)
        P, Q, R = x['omega']
        Omega = np.array([
            [0, -P, -Q, -R],
            [P,  0,  R, -Q],
            [Q, -R,  0,  P],
            [R,  Q, -P,  0]])
        q_dot = 0.5 * Omega @ x['q']

        x_dot = np.empty(1, self.state_dtype)
        x_dot['q'] = q_dot
        x_dot['p'] = x['v']
        x_dot['v'] = a_ned
        x_dot['omega'] = alpha

        # Save the Gamma distribution for use with the next time update
        params['Gamma'] = Gamma  # FIXME: needs a design review

        return x_dot.view(float)  # The integrator expects a flat array


# ---------------------------------------------------------------------------


def simulate(model, state0, T=10, T0=0, dt=0.5, first_step=0.25, max_step=0.5):
    """

    Parameters
    ----------
    model
        The model that provides `dynamics`
    state0
        The initial state
    T : float [seconds]
        The total simulation time
    T0 : float [seconds]
        The start time of the simulation. Useful for models with time varying
        behavior (eg, wind fields).
    dt : float [seconds]
        The simulation step size. This determines the time separation of each
        point in the state trajectory, but the RK4 integrator is free to use
        a different step size internally.

    Returns
    -------
    times : ndarray, shape (K+1, N)
    path : ndarray, shape (K+1, N)
        K : the number of discrete time values
        N : the number of state variables
    """

    num_steps = int(np.ceil(T / dt)) + 1  # Include the initial state
    times = np.zeros(num_steps)  # The simulation times
    path = np.empty(num_steps, dtype=model.state_dtype)
    path[0] = state0

    solver = scipy.integrate.ode(model.dynamics)
    solver.set_integrator('dopri5', rtol=1e-3, first_step=0.25, max_step=0.5)
    solver.set_initial_value(state0.view(float))
    solver.set_f_params({'Gamma': None})  # Is changed by `model.dynamics`

    t_start = time.time()
    msg = ''
    k = 1  # Number of completed states (including the initial state)
    print("\nRunning the simulation.")
    try:
        while solver.successful() and k < num_steps:
            if k % 25 == 0:  # Update every 25 iterations
                avg_rate = (k-1) / (time.time() - t_start)  # k=0 was free
                rem = (num_steps - k) / avg_rate  # Time remaining in seconds
                msg = f"ETA: {int(rem // 60)}m{int(rem % 60):02d}s"
            print(f"\rStep: {k} (t = {k*dt:.2f}). {msg}", end="")

            # WARNING: `solver.integrate` returns a *reference* to `_y`
            #          Modifying `state` modifies `solver.y` directly.
            # FIXME: Is that valid for all `integrate` methods (eg, Adams)?
            #        Using `solver.set_initial_value` would reset the
            #        integrator, but that's not what we want here.
            state = solver.integrate(solver.t + dt).view(model.state_dtype)
            state['q'] /= np.sqrt((state['q']**2).sum())  # Normalize `q`
            path[k] = state  # Makes a copy of `solver._y`
            times[k] = solver.t
            k += 1
    except RuntimeError:  # The model blew up
        # FIXME: refine this idea
        print("\n--- Simulation failed. Terminating. ---")
    except KeyboardInterrupt:
        print("\n--- Simulation interrupted. ---")
    finally:
        if k < num_steps:  # Truncate if the simulation did not complete
            times = times[:k]
            path = path[:k]

    print(f"\nTotal simulation time: {time.time() - t_start}\n")

    # For debugging
    eulers = np.rad2deg(quaternion.quaternion_to_euler(path['q']))
    cps = model.glider.control_points(0)  # Control points in frd
    cp0 = cps[95]  # The central control point in frd
    p_cp0 = path['p'] + quaternion.apply_quaternion_rotation(path['q'], cp0)
    v_cp0 = path['v'] + cross3(path['omega'], cp0)

    embed()

    return times, path


def main():
    glider = build_glider()
    model = GliderSim(glider, 1.2)

    alpha_eq, Theta_eq, V_eq = glider.equilibrium_glide(0, 0, 1.2)

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
    state0 = np.empty(1, dtype=GliderSim.state_dtype)
    state0['q'] = q
    state0['p'] = [0, 0, 0]
    state0['v'] = quaternion.apply_quaternion_rotation(q_inv, UVW)
    state0['omega'] = PQR

    print("Preparing the simulation.")
    print("Initial state:", state0)

    # times05, path05 = simulate(model, state0_raw, dt=0.05, T=120)
    # times10, path10 = simulate(model, state0_raw, dt=0.10, T=120)
    times25, path25 = simulate(model, state0, dt=0.25, T=120)
    # times50, path50 = simulate(model, state0_raw, dt=0.50, T=120)

    # For verification purposes
    # delta_PE = 9.8 * 75 * -path['p'].T[2]  # PE = mgh
    # KE_trans = 1/2 * 75 * norm(path['v'], axis=1)**2  # translational KE
    # KE_rot = 1/2 * np.einsum('ij,kj->k', J, path['omega']**2)
    # delta_E = delta_PE + (KE_trans - KE_trans[0]) + KE_rot

    # embed()


if __name__ == "__main__":
    main()