# import abc

from functools import partial

import numpy as np
from numpy import sqrt, sin, cos, tan, arcsin, arctan, deg2rad

from scipy.optimize import least_squares


def trapz(y, dx):
    # Trapezoidal integrator
    return np.sum(y[1:] + y[:-1]) / 2.0 * dx


class Glider:
    def __init__(self, wing, d_cg, h_cg, S_cg, Cd_cg):
        self.wing = wing
        self.d_cg = d_cg
        self.h_cg = h_cg
        self.S_cg = S_cg
        self.Cd_cg = Cd_cg

        # FIXME: pre-compute the `body->local` transformation matrices here?
        # FIXME: lots more to implement

    def wing_section_wind(self, y, state, control=None):
        # FIXME: Document
        # compute the local relative wind for wing sections
        U, V, W, P, Q, R = state

        # Compute the local section wind
        # PFD eqs 4.10-4.12, p73
        #
        # FIXME: rewrite this as a matrix equation?
        #  * Ref: 'Aircraft Control and Simulation', Eq:1.4-2, p31
        fx = self.geometry.fx(y)
        fz = self.geometry.fz(y)
        uL = U + fz*Q - y*R
        vL = V - fz*P + fx*R
        wL = W + y*P - fx*Q

        return uL, vL, wL

    def equilibrium_parameters(self, delta):
        """Compute alpha_eq, d0, h0

        Parameters
        ----------
        delta : float
            The symmetric brake actuation, where `0 <= delta <= 1`

        Returns
        -------
        alpha_eq : float
            The equilibrium AOA for the given symmetric brakes actuation
        d0 : float
            The x-axis distance of the cg to the global AC
        h0 : float
            The z-axis distance of the cg to the global AC
        """

        wing = self.wing  # FIXME: cleanup

        # The integration points across the span
        N = 501
        dy = wing.geometry.b/(N - 1)  # Include the endpoint
        y = np.linspace(-wing.geometry.b/2, wing.geometry.b/2, N)

        deltas = wing.geometry.delta(y)
        thetas = wing.geometry.ftheta(y)

        def calc_d0h0(wing, alpha_eq):
            """Calculate the global AC, {d0, h0}

            These points are deterministic given alpha_eq, but are used as part
            of the optimization routine to find alpha_eq.

            ref: PFD Eqs 5.44-5.45
            """

            CL = wing.CL(alpha_eq)
            CD = wing.CD(alpha_eq)

            Cli = wing.Cl(alpha_eq)
            Cdi = wing.Cd(alpha_eq)
            alpha_i = alpha_eq*cos(deltas) + thetas

            # PFD Eq:5.44, p120
            fx = wing.geometry.fx(y)
            fz = wing.geometry.fz(y)
            fc = wing.geometry.fc(y)
            S = wing.geometry.S
            numerator = (Cli*cos(alpha_i) + Cdi*sin(alpha_i)) * fx * fc
            denominator = (CL*cos(alpha_eq) + CD*sin(alpha_eq)) * S
            d0 = trapz(numerator, dy) / denominator

            # PFD Eq:5.45, p120
            numerator = -((Cli*sin(alpha_i) - Cdi*cos(alpha_i)) *
                          fz * fc / cos(deltas))
            denominator = (CL*sin(alpha_eq) - CD*cos(alpha_eq)) * S
            h0 = trapz(numerator, dy) / denominator

            # print("DEBUG> d0: {}, h0: {}".format(d0, h0))

            return d0, h0

        def calc_my(wing, alpha, d0=None, h0=None):
            """Optimization target for computing alpha_eq

            Parameters
            ----------
            wing : Wing
            alpha : float
                Current guess for alpha_eq

            Returns
            -------
            My : float
                The total moment about the y-axis. Should be zero for
                equilibrium.
            """
            print("DEBUG> calc_my: alpha:", alpha)

            # Update {d0, h0} to track the changing alpha
            if d0 is None:
                d0, h0 = calc_d0h0(wing, alpha)

            CL = wing.CL(alpha)
            CD = wing.CD(alpha)
            Cm = wing.Cm(alpha)
            Cz = CL*cos(alpha) + CD*sin(alpha)  # PFD eq 4.76
            Cx = CL*sin(alpha) - CD*cos(alpha)  # PFD eq 4.77
            My = Cz*d0 - Cx*h0 + Cm*wing.geometry.MAC  # PFD Eq 4.78/5.37
            return My

        # alphas = np.linspace(-1.99, 24, 50)
        # d0h0s = np.asarray([calc_d0h0(wing, np.deg2rad(a)) for a in alphas])
        # Mys = np.asarray([calc_my(wing, np.deg2rad(a)) for a in alphas])
        # print("d0h0s")
        # input("Continue?")
        # embed()

        # FIXME: Initialize alpha_eq to something reasonable
        f_alpha = partial(calc_my, wing)
        alpha_eq_prime = least_squares(f_alpha, np.deg2rad(8)).x[0]
        # alpha_eq_prime = least_squares(
        #     f_alpha, np.deg2rad(2),
        #   bounds=(np.deg2rad(1.75), np.deg2rad(2.3))).x[0]
        #   bounds=(0, np.deg2rad(15))).x[0]

        print("Finished finding alpha_eq_prime:", alpha_eq_prime)
        input("Continue?")
        embed()


