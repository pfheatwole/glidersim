"""FIXME: add docstring."""

import abc

import numpy as np
import scipy.optimize

from pfh.glidersim.util import cross3


__all__ = [
    "ForceEstimator",
    "Phillips",
]


def __dir__():
    return __all__


class ForceEstimator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, delta_f, v_W2f, rho_air):
        """
        Estimate the forces and moments on a foil.

        Parameters
        ----------
        delta_f : array_like of float [radians]
            The deflection angle of each section. The shape must be able to
            broadcast to (K,), where `K` is the number of control points being
            used by the estimator.
        v_W2f : array_like of float [m/s]
            The velocity of the wind relative to the control points in foil frd
            coordinates. The shape must be able to broadcast to (K, 3), where
            `K` is the number of control points being used by the estimator.
        rho_air : float [kg/m^3]
            Air density
        """

    @property
    @abc.abstractmethod
    def control_points(self):
        """
        Compute the control points for the section aerodynamics.

        Returns
        -------
        r_CP2LE : float, shape (K,3) [m]
            Control points relative to the central leading edge `LE`.
            Coordinates are in canopy frd, and `K` is the number of points
            being used by the estimation method.
        """

    class ConvergenceError(RuntimeError):
        """The estimator failed to converge on a solution."""


class Phillips(ForceEstimator):
    """
    A non-linear numerical lifting-line method.

    Uses a set of spanwise bound vortices instead of a single, uniform lifting
    line. Unlike the Prandtl's classic lifting-line theory, this method allows
    for wing sweep and dihedral.

    Parameters
    ----------
    foil : FoilGeometry
        Defines the lifting-line and section coefficients.
    v_ref_mag : float [m/s]
        The reference solution airspeed
    alpha_ref : float [degrees]
        The reference solution angle of attack
    K : integer
        The number of bound vortex segments. Default: 51

    References
    ----------
    .. [1] Phillips and Snyder, "Modern Adaptation of Prandtl’s Classic
       Lifting-Line Theory", Journal of Aircraft, 2000

    .. [2] McLeanauth, "Understanding Aerodynamics - Arguing from the Real
       Physics", p382

    .. [3] Hunsaker and Snyder, "A lifting-line approach to estimating
       propeller/wing interactions", 2006

    Notes
    -----
    This implementation uses a single distribution for the entire span, which
    is suitable for parafoils, which is a continuous lifting surface, but for
    wings with left and right segments separated by some discontinuity at the
    root you should distribute the points across each semispan independently.
    See _[1] for a related discussion.

    This method does suffer an issue where induced velocity goes to infinity as
    the segment lengths tend toward zero (as the number of segments increases,
    or for a poorly chosen point distribution). See _[2], section 8.2.3.
    """

    def __init__(self, foil, v_ref_mag, alpha_ref=5, K=51):
        self.foil = foil
        self.K = K

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution
        self.s_nodes = np.linspace(-1, 1, self.K + 1)

        # Option 2: cosine distribution
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K + 1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.foil.surface_xyz(self.s_nodes, 0.25, surface="chord")

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1]) / 2
        self.cps = self.foil.surface_xyz(self.s_cps, 0.25, surface="chord")

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]
        self.r1 = np.linalg.norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = np.linalg.norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        # Note: Phillip's derivation uses back-left-up coordinates (not `frd`)
        u = -self.foil.section_orientation(self.s_cps).T
        self.u_a, self.u_s, self.u_n = u[0].T, u[1].T, u[2].T

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        node_chords = self.foil.chord_length(self.s_nodes)
        self.c_avg = (node_chords[1:] + node_chords[:-1]) / 2
        self.dA = self.c_avg * np.linalg.norm(cross3(self.u_a, self.dl), axis=1)

        # Precompute the `v` terms that do not depend on `u_inf`
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2  # Shorthand
        self.v_ij = np.zeros((self.K, self.K, 3))  # Extra terms when `i != j`
        for ij in [(i, j) for i in range(self.K) for j in range(self.K)]:
            if ij[0] == ij[1]:  # Skip singularities when `i == j`
                continue
            self.v_ij[ij] = ((r1[ij] + r2[ij]) * cross3(R1[ij], R2[ij])) / \
                (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + np.dot(R1[ij], R2[ij])))

        # Precompute a reference solution from a (hopefully easy) base case.
        # Sets an initial "solution" (which isn't actually a solution) just to
        # bootstrap the `__call__` method with an initial `Gamma` value.
        alpha_ref = np.deg2rad(alpha_ref)
        v_mag = np.broadcast_to(v_ref_mag, (self.K, 3))
        v_W2f_ref = -v_mag * np.array([np.cos(alpha_ref), 0, np.sin(alpha_ref)])
        self._reference_solution = {
            'delta_f': 0,
            'v_W2f': v_W2f_ref,
            'Gamma': np.sqrt(1 - self.s_cps ** 2),  # Naive ellipse
        }
        try:
            _, _, self._reference_solution = self.__call__(0, v_W2f_ref, 1.2)
        except ForceEstimator.ConvergenceError as e:
            raise RuntimeError("Phillips: failed to initialize base case")

    def _compute_Reynolds(self, v_W2f, rho_air):
        """Compute the Reynolds number at each control point."""

        # FIXME: verify that using the total airspeed (including spanwise flow)
        #        is okay. A few tests show minimal differences, so for now I'm
        #        not wasting time computing the normal and chordwise flows.
        u = np.linalg.norm(v_W2f, axis=-1)  # airspeed [m/s]
        mu = 1.81e-5  # Standard dynamic viscosity of air
        Re = rho_air * u * self.c_avg / mu
        # print("\nDEBUG> Re:", Re, "\n")
        return Re

    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def _induced_velocities(self, u_inf):
        # 2. Compute the "induced velocity" unit vectors
        #  * ref: Phillips, Eq:6
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2  # Shorthand
        v = self.v_ij.copy()
        v += (
            cross3(u_inf, R2)
            / (r2 * (r2 - np.einsum("k,ijk->ij", u_inf, R2)))[..., None]
        )
        v -= (
            cross3(u_inf, R1)
            / (r1 * (r1 - np.einsum("k,ijk->ij", u_inf, R1)))[..., None]
        )

        return v / (4 * np.pi)  # axes: (inducer, inducee, 3-vector)

    def _local_velocities(self, v_W2f, Gamma, v):
        # Compute the local fluid velocities
        #  * ref: Hunsaker Eq:5
        #  * ref: Phillips Eq:5 (nondimensional version)
        V = v_W2f + np.einsum("j,jik->ik", Gamma, v)

        # Compute the local angle of attack for each section
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_n = np.einsum("ik,ik->i", V, self.u_n)  # Normal-wise
        V_a = np.einsum("ik,ik->i", V, self.u_a)  # Chordwise
        alpha = np.arctan2(V_n, V_a)

        return V, V_n, V_a, alpha

    def _f(self, Gamma, delta_f, v_W2f, v, Re):
        # Compute the residual error vector
        #  * ref: Hunsaker Eq:8
        #  * ref: Phillips Eq:14
        V, V_n, V_a, alpha = self._local_velocities(v_W2f, Gamma, v)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(np.einsum("ik,ik->i", W, W))
        Cl = self.foil.sections.Cl(self.s_cps, delta_f, alpha, Re)
        # return 2 * Gamma * W_norm - np.einsum("ik,ik,i,i->i", V, V, self.dA, Cl)
        return 2 * Gamma * W_norm - (V_n ** 2 + V_a ** 2) * self.dA * Cl

    def _J(self, Gamma, delta_f, v_W2f, v, Re, verify_J=False):
        # 7. Compute the Jacobian matrix, `J[ij] = d(f_i)/d(Gamma_j)`
        #  * ref: Hunsaker Eq:11
        V, V_n, V_a, alpha = self._local_velocities(v_W2f, Gamma, v)
        W = cross3(V, self.dl)
        W_norm = np.sqrt(np.einsum("ik,ik->i", W, W))
        Cl = self.foil.sections.Cl(self.s_cps, delta_f, alpha, Re)
        Cl_alpha = self.foil.sections.Cl_alpha(self.s_cps, delta_f, alpha, Re)

        J = 2 * np.diag(W_norm)  # Additional terms for i==j
        J2 = 2 * np.einsum("i,ik,i,jik->ij", Gamma, W, 1 / W_norm, cross3(v, self.dl))
        J3 = (np.einsum("i,jik,ik->ij", V_a, v, self.u_n)
              - np.einsum("i,jik,ik->ij", V_n, v, self.u_a))
        J3 *= (
            (self.dA * Cl_alpha)[:, None]
            * np.einsum("ik,ik->i", V, V)
            / (V_n ** 2 + V_a ** 2)
        )
        J4 = 2 * np.einsum("i,i,ik,jik->ij", self.dA, Cl, V, v)
        J += J2 - J3 - J4

        # Compare the analytical gradient to the finite-difference version
        if verify_J:
            J_true = self._J_finite(Gamma, delta_f, v_W2f, v, Re)
            if not np.allclose(J, J_true):
                print("\n !!! The analytical Jacobian disagrees. Halting. !!!")
                breakpoint()

        return J

    def _J_finite(self, Gamma, delta_f, v_W2f, v, Re):
        """Compute the Jacobian using a centered finite distance.

        Useful for checking the analytical gradient.

        Examples
        --------
        >>> J1 = self._J(Gamma, v_W2f, v, delta_f)
        >>> J2 = self._J_finite(Gamma, v_W2f, v, delta_f)
        >>> np.allclose(J1, J2)  # FIXME: tune the tolerances?
        True
        """
        # This uses the same method as `scipy.optimize.approx_fprime`, but that
        # function only works for scalar-valued functions.
        JT = np.empty((self.K, self.K))  # Jacobian transpose  (J_ji)
        eps = np.sqrt(np.finfo(float).eps)

        # Build the Jacobian column-wise (row-wise of the tranpose)
        Gp, Gm = Gamma.copy(), Gamma.copy()
        for k in range(self.K):
            Gp[k], Gm[k] = Gamma[k] + eps, Gamma[k] - eps
            fp = self._f(Gp, delta_f, v_W2f, v, Re)
            fm = self._f(Gm, delta_f, v_W2f, v, Re)
            JT[k] = (fp - fm) / (2 * eps)
            Gp[k], Gm[k] = Gamma[k], Gamma[k]

        return JT.T

    def _solve_circulation(self, delta_f, v_W2f, Re, Gamma0):
        """
        Solve for the spanwise circulation distribution.

        Parameters
        ----------
        delta_f : array of float, shape (K,) [radians]
            The deflection angle of each section.
        v_W2f : array of float, shape (K,) [m/s]
            Relative wind velocity at each control point.
        Re : array of float, shape (K,)
            Reynolds number at each segment
        Gamma0 : array of float, shape (K,)
            The initial proposal

        Returns
        -------
        Gamma : array of float, shape (K,)
            Circulation strengths of each segment.
        v : array, shape (K,K,3) [m/s]
            Induced velocities between each segment, indexed as (inducer,
            inducee).
        """
        v_mid = v_W2f[self.K // 2]
        u_inf = v_mid / np.linalg.norm(v_mid)  # FIXME: what if PQR != 0?
        v = self._induced_velocities(u_inf)
        args = (delta_f, v_W2f, v, Re)
        res = scipy.optimize.root(self._f, Gamma0, args, jac=self._J, tol=1e-4)

        if not res["success"]:
            raise ForceEstimator.ConvergenceError

        return res["x"], v

    def __call__(self, delta_f, v_W2f, rho_air, reference_solution=None, max_splits=10):
        # FIXME: this doesn't match the ForceEstimator.__call__ signature
        delta_f = np.broadcast_to(delta_f, (self.K))
        v_W2f = np.broadcast_to(v_W2f, (self.K, 3))
        Re = self._compute_Reynolds(v_W2f, rho_air)

        if reference_solution is None:
            reference_solution = self._reference_solution

        delta_f_ref = reference_solution['delta_f']
        v_W2f_ref = reference_solution['v_W2f']
        Gamma_ref = reference_solution['Gamma']

        # Try to solve for the target (`Gamma` as a function of `v_W2f` and
        # `delta_f`) directly using the `reference_solution`. If that fails,
        # pick a point between the target and the reference, solve for that
        # easier case, then use its solution as the new starting point for the
        # next target. Repeat for intermediate targets until either solving for
        # the original target, or exceeding `max_splits`.
        target_backlog = []  # Stack of pending targets
        num_splits = 0
        while True:
            try:
                Gamma, v = self._solve_circulation(delta_f, v_W2f, Re, Gamma_ref)
            except ForceEstimator.ConvergenceError:
                if num_splits == max_splits:
                    raise ForceEstimator.ConvergenceError("max splits reached")
                num_splits += 1
                target_backlog.append((delta_f, v_W2f))
                P = 0.5  # Ratio, a point between the reference and the target
                delta_f = (1 - P) * delta_f_ref + P * delta_f
                v_W2f = (1 - P) * v_W2f_ref + P * v_W2f
                continue

            delta_f_ref = delta_f
            v_W2f_ref = v_W2f
            Gamma_ref = Gamma

            if target_backlog:
                delta_f, v_W2f = target_backlog.pop()
            else:
                break

        V, V_n, V_a, alpha = self._local_velocities(v_W2f, Gamma, v)

        # Compute the inviscid forces using the 3D vortex lifting law
        #  * ref: Hunsaker Eq:1
        #  * ref: Phillips Eq:4
        dF_inviscid = Gamma * cross3(V, self.dl).T

        # Compute the viscous forces.
        #  * ref: Hunsaker Eq:17
        #
        # The equation in the paper uses the "characteristic chord", but I
        # believe that is a mistake; it produces *massive* drag. Here I use the
        # section area like they do in "MachUp_Py" (see where they compute
        # `f_parasite_mag` in `llmodel.py:LLModel:_compute_forces`).
        Cd = self.foil.sections.Cd(self.s_cps, delta_f, alpha, Re)
        V2 = np.einsum("ik,ik->i", V, V)
        u_drag = V.T / np.sqrt(V2)
        dF_viscous = 0.5 * V2 * self.dA * Cd * u_drag

        # The total forces applied at each control point
        dF = dF_inviscid + dF_viscous

        # Compute the section moments.
        #  * ref: Hunsaker Eq:19
        #  * ref: Phillips Eq:28
        #
        # These are strictly the section moments caused by airflow around the
        # section. It does not include moments about the aircraft reference
        # point (commonly the center of gravity); those extra moments must be
        # calculated by the wing.
        #  * ref: Hunsaker Eq:19
        #  * ref: Phillips Eq:28
        Cm = self.foil.sections.Cm(self.s_cps, delta_f, alpha, Re)
        dM = -0.5 * V2 * self.dA * self.c_avg * Cm * self.u_s.T

        solution = {
            'delta_f': delta_f_ref,
            'v_W2f': v_W2f_ref,
            'Gamma': Gamma_ref,
        }

        # print("\nFinished `Phillips.__call__`")
        # breakpoint()

        dF *= rho_air
        dM *= rho_air

        return dF.T, dM.T, solution
