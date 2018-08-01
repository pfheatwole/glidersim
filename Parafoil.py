import abc

import numpy as np
from numpy import sin, cos, arctan2, dot, cross, einsum
from numpy.linalg import norm

from numba import njit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

from util import cross3

from IPython import embed


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class Parafoil:
    def __init__(self, geometry, sections):
        self.geometry = geometry
        self.sections = sections  # Provides the airfoils for each section

    def upper_surface(self, t, xa=None, N=150):
        """Airfoil upper surface curve on the 3D parafoil

        Parameters
        ----------
        t : float
            Position on the parametric curve, where `-1 <= t <= 1`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord

        Returns
        -------
        FIXME
        """

        # FIXME: support `t` broadcasting?

        if xa is None:
            xa = np.linspace(0, 1, N)  # FIXME: assume normalized airfoils?

        fc = self.geometry.fc(t)
        upper = fc*self.airfoil.geometry.upper_curve(xa)  # Scaled airfoil
        # FIXME: this ^ should use a ParafoilSection interface
        xU, zU = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(t)
        Gamma = self.geometry.Gamma(t)

        x = self.geometry.fx(t) + (fc/4 - xU)*cos(theta) - zU*sin(theta)
        y = self.geometry.fy(t) + \
            ((fc/4 - xU)*sin(theta) + zU*cos(theta))*sin(Gamma)
        z = self.geometry.fz(t) - \
            ((fc/4 - xU)*sin(theta) + zU*cos(theta))*cos(Gamma)

        return np.c_[x, y, z]

    def lower_surface(self, t, xa=None, N=150):
        """Airfoil lower surface curve on the 3D parafoil

        Parameters
        ----------
        t : float
            Position on the parametric curve, where `-1 <= t <= 1`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord

        Returns
        -------
        FIXME
        """

        # FIXME: support `t` broadcasting?

        if xa is None:
            xa = np.linspace(0, 1, N)  # FIXME: assume normalized airfoils?

        fc = self.geometry.fc(t)
        lower = fc*self.airfoil.geometry.lower_curve(xa)  # Scaled airfoil
        # FIXME: this ^ should use a ParafoilSection interface
        xL, zL = lower[:, 0], lower[:, 1]

        theta = self.geometry.ftheta(t)
        Gamma = self.geometry.Gamma(t)

        x = self.geometry.fx(t) + (fc/4 - xL)*cos(theta) + zL*sin(theta)
        y = self.geometry.fy(t) + \
            ((fc/4 - xL)*sin(theta) + zL*cos(theta))*sin(Gamma)
        z = self.geometry.fz(t) - \
            ((fc/4 - xL)*sin(theta) + zL*cos(theta))*cos(Gamma)

        return np.c_[x, y, z]


# ----------------------------------------------------------------------------

class ParafoilSections(abc.ABC):
    """Defines the spanwise variation of the Parafoil sections"""

    # FIXME: bad naming? An instance of this class isn't a Parafoil section.

    @abc.abstractmethod
    def Cl_alpha(self, s, alpha, delta):
        """The derivative of the lift coefficient vs alpha for the section"""

    @abc.abstractmethod
    def Cl(self, s, alpha, delta):
        """The lift coefficient for the section"""

    @abc.abstractmethod
    def Cd(self, s, alpha, delta):
        """The drag coefficient for the section"""

    @abc.abstractmethod
    def Cm(self, s, alpha, delta):
        """The pitching moment coefficient for the section"""


class ConstantCoefficients(ParafoilSections):
    """
    Uses the same airfoil for all wing sections, no spanwise variation.
    """

    def __init__(self, airfoil):
        self.airfoil = airfoil

    def Cl_alpha(self, s, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cl_alpha(alpha, delta)

    def Cl(self, s, alpha, delta):
        # FIXME: make AirfoilCoefficients responsible for broadcasting `alpha`?
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cl(alpha, delta)

    def Cd(self, s, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cd(alpha, delta)

    def Cm(self, s, alpha, delta):
        if np.isscalar(alpha):
            alpha = np.ones_like(s) * alpha  # FIXME: replace with `full`
        return self.airfoil.coefficients.Cm(alpha, delta)


# ----------------------------------------------------------------------------

class ForceEstimator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, V_rel, delta, rho=1):
        """Estimate the forces and moments on a Parafoil"""

    @property
    @abc.abstractmethod
    def control_points(self):
        """The reference points for calculating the section forces"""


class Phillips(ForceEstimator):
    """
    A numerical lifting-line method that uses a set of spanwise bound vortices
    instead of a single, uniform lifting line. Unlike the Prandtl's classic
    lifting-line theory, this method allows for wing sweep and dihedral.

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
    This implementation uses a single, linear point distribution in terms of
    the normalized span coordinate `s`. Using a single distribution that covers
    the entire span is suitable for parafoils, but for wings with left and
    right segments separated by some discontinuity at the root you should
    distribute the points across each semispan independently. Also, this method
    assumes a linear distribution in `s` provides reasonable point spacing, but
    depending on the wing curvature a different distribution, such as a cosine,
    may be more applicable. See _[1].

    This method does suffer an issue where induced velocity goes to infinity as
    the segment lengths tend toward zero (as the number of segments increases,
    or for a poorly chosen point distribution). See _[2], section 8.2.3.
    """

    def __init__(self, parafoil, lobe_args={}):
        self.parafoil = parafoil
        self.lobe_args = lobe_args

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution; less likely to induce large velocties
        self.K = 31  # The number of bound vortex segments
        self.s_nodes = np.linspace(-1, 1, self.K+1)

        # Option 2: cosine distribution; fast, very sensitive to segment length
        # self.K = 13  # The number of bound vortex segments
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.c4(self.s_nodes)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1])/2
        self.cps = self.parafoil.c4(self.s_cps)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]  # node N is at at axis0=N-1
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        # FIXME: is there a better way to select the column vectors?
        u = self.parafoil.section_orientation(self.s_cps, lobe_args)
        self.u_a, self.u_s, self.u_n = u[:, :, 0], u[:, :, 1], u[:, :, 2]

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]  # `R1 + R2`?
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        c_avg = (node_chords[1:] + node_chords[:-1])/2
        chord_vectors = c_avg[:, None] * self.u_a  # FIXME: verify
        self.dA = norm(cross(chord_vectors, self.dl), axis=1)
        # FIXME: faster+correct to post_multiply the scalars `c_avg`?

        # --------------------------------------------------------------------
        # For debugging purposes: plot the quarter chord line, and segments
        plotit = False
        # plotit = True
        if plotit:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection='3d')
            ax.view_init(azim=-130, elev=25)

            # Plot the actual quarter chord
            # y = np.linspace(-b/2, b/2, 51)
            # t = np.linspace(-1, 1, 51)
            # ax.plot(self.parafoil.geometry.fx(t),
            #         self.parafoil.geometry.fy(t),
            #         -self.parafoil.geometry.fz(t), 'g--', lw=0.8)

            # Plot the segments and their nodes
            # ax.plot(self.nodes[:, 0], self.nodes[:, 1], -self.nodes[:, 2], marker='.')

            # Plot the dl segments
            segments = self.dl + self.nodes[:-1]  # Add their starting points
            ax.plot(segments[:, 0], segments[:, 1], -segments[:, 2], marker='.')

            # Plot the cps
            ax.scatter(self.cps[:, 0], self.cps[:, 1], -self.cps[:, 2], marker='x')

            set_axes_equal(ax)
            plt.show()
        self.f = None  # FIXME: design review Numba helper functions

    @property
    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def ORIG_induced_velocities(self, u_inf):
        #  * ref: Phillips, Eq:6
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2
        v = np.empty_like(R1)

        indices = [(i, j) for i in range(self.K) for j in range(self.K)]
        print()
        for ij in indices:
            v[ij] = cross(u_inf, R2[ij]) / \
                (r2[ij] * (r2[ij] - dot(u_inf, R2[ij])))

            v[ij] = v[ij] - cross(u_inf, R1[ij]) / \
                (r1[ij] * (r1[ij] - dot(u_inf, R1[ij])))

            if ij[0] == ij[1]:
                continue  # Skip singularities when `i == j`

            v[ij] = v[ij] + ((r1[ij] + r2[ij]) * cross(R1[ij], R2[ij])) / \
                (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + dot(R1[ij], R2[ij])))

        return v/(4*np.pi)

    def _induced_velocities(self, u_inf):
        #  * ref: Phillips, Eq:6
        # This version uses a Numba helper function
        R1, R2, r1, r2 = self.R1, self.R2, self.r1, self.r2
        K = self.K

        if self.f is None:
            def f(u_inf):
                v = np.empty_like(R1)

                indices = [(i, j) for i in range(K) for j in range(K)]
                print()
                for ij in indices:
                    v[ij] = cross3(u_inf, R2[ij]) / \
                        (r2[ij] * (r2[ij] - dot(u_inf, R2[ij])))

                    v[ij] = v[ij] - cross3(u_inf, R1[ij]) / \
                        (r1[ij] * (r1[ij] - dot(u_inf, R1[ij])))

                    if ij[0] == ij[1]:
                        continue  # Skip singularities when `i == j`

                    v[ij] = v[ij] + ((r1[ij] + r2[ij]) * cross3(R1[ij], R2[ij])) / \
                        (r1[ij] * r2[ij] * (r1[ij] * r2[ij] + dot(R1[ij], R2[ij])))

                return v/(4*np.pi)

            self.f = njit(f)

        return self.f(u_inf)

    def _vortex_strengths(self, V_rel, delta, max_runs=None):
        """
        FIXME: finish the docstring

        Parameters
        ----------
        V_rel : array of float, shape (K,) [meters/second]
            Fluid velocity vectors for each section, in body coordinates. This
            is equal to the relative wind "far" from each wing section, which
            is absent of circulation effects.
        delta : array of float, shape (K,) [radians]
            The angle of trailing edge deflection

        Returns
        -------
        Gamma : array of float, shape (K,) [units?]
        V : array of float, shape (K,) [meters/second]

        """

        # FIXME: this implementation fails when wing sections go beyond the
        #        stall condition. In that case, use under-relaxed Picard
        #        iterations.  ref: Hunsaker and Snyder, 2006, pg 5
        # FIXME: find a better initial proposal
        # FIXME: return the induced AoA? Could be interesting

        assert np.shape(V_rel) == (self.K, 3)

        # FIXME: is using the freestream velocity at the central chord okay?
        u_inf = V_rel[self.K // 2]
        u_inf = u_inf / norm(u_inf)

        # 2. Compute the "induced velocity" unit vectors
        v = self._induced_velocities(u_inf)  # axes = (inducer, inducee)
        vT = np.swapaxes(v, 0, 1)  # Useful for broadcasting cross products

        # 3. Propose an initial distribution for Gamma
        #  * For now, use an elliptical Gamma
        b = self.parafoil.b
        cp_y = self.cps[:, 1]
        Gamma0 = 5

        # Alternative initial proposal
        # avg_brake = (delta_Bl + delta_Br)/2
        # CL_2d = self.coefs.CL(np.arctan2(u_inf[2], u_inf[0]), avg_brake)
        # S = self.parafoil.S
        # Gamma0 = 2*norm(V_rel[self.K//2])*S*CL_2d/(np.pi*b)  # c0 circulation

        Gamma = Gamma0 * np.sqrt(1 - ((2*cp_y)/b)**2)

        # Save intermediate values for debugging purposes
        Vs = [V_rel]
        Gammas = [Gamma]
        delta_Gammas = []
        fs = []
        Js = []
        alphas = []
        Cl_alphas = []

        # FIXME: very ad-hoc way to prevent large negative AoA at the wing tips
        # FIXME: why must Omega be so small? `Cl_alpha` sensitivity?
        # M = max(delta_Bl, delta_Br)  # Assumes the delta_B are 0..1
        # base_Omega, min_Omega = 0.2, 0.05
        # Omega = base_Omega - (base_Omega - min_Omega)*np.sqrt(M)
        Omega = 0.1

        if max_runs is None:
            # max_runs = 5 + int(np.ceil(3*M))
            max_runs = 30

        # FIXME: don't use a fixed number of runs
        # FIXME: how much faster is `opt_einsum` versus the scipy version?
        # FIXME: if `coefs2d.Cl` was Numba compatible, what about this loop?
        n_runs = 0
        while n_runs < max_runs:
            # print("run:", n_runs)
            # 4. Compute the local fluid velocities
            #  * ref: Hunsaker-Snyder Eq:5
            #  * ref: Phillips Eq:5 (nondimensional version)
            V = V_rel + einsum('i,ijk->jk', Gamma, v)

            # 5. Compute the section local angle of attack
            #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
            V_a = einsum('ik,ik->i', V, self.u_a)  # Chordwise
            V_n = einsum('ik,ik->i', V, self.u_n)  # Normal-wise
            alpha = arctan2(V_n, V_a)

            min_alpha = min(alpha)
            if np.rad2deg(min_alpha) < -11:
                print("Encountered a very small alpha: {}".format(min_alpha))
                embed()

            # plt.plot(cp_y, np.rad2deg(alpha))
            # plt.ylabel('local section alpha')
            # plt.show()

            # For testing purposes: the global section alpha and induced AoA
            # V_chordwise_2d = einsum('ij,ij->i', V_rel, self.u_a)
            # V_normal_2d = einsum('ij,ij->i', V_rel, self.u_n)
            # alpha_2d = arctan2(V_normal_2d, V_chordwise_2d)
            # alpha_induced = alpha_2d - alpha

            # print("Stopping to investigate the alphas")
            # embed()
            # input('continue?')

            Cl = self.parafoil.sections.Cl(self.s_cps, alpha, delta)

            if np.any(np.isnan(Cl)):
                print("Cl has nan's")
                embed()
                return
                # FIXME: raise a RuntimeWarning?

            # 6. Compute the residual error
            #  * ref: Phillips Eq:15, or Hunsaker-Snyder Eq:8
            W = cross(V, self.dl)
            W_norm = norm(W, axis=1)
            f = 2 * Gamma * W_norm - (V*V).sum(axis=1) * self.dA * Cl

            # 7. Compute the gradient
            #  * ref: Hunsaker-Snyder Eq:11
            Cl_alpha = self.parafoil.sections.Cl_alpha(self.s_cps, alpha, delta)

            # plt.plot(cp_y, Cl_alpha)
            # plt.ylabel('local section Cl_alpha')
            # plt.show()

            # print("Check the Cl_alpha")
            # embed()
            # input('continue?')

            # J is a Jordan matrix, where `J[ij] = d(F_i)/d(Gamma_j)`
            J1 = 2 * np.diag(W_norm)  # terms for i==j
            J2 = 2 * einsum('ik,ijk->ij', W, cross(vT, self.dl))
            J2 = J2 * (Gamma / W_norm)[:, None]
            J3 = (einsum('i,jik,ik->ij', V_a, v, self.u_n) -
                  einsum('i,jik,ik->ij', V_n, v, self.u_a))
            J3 = J3 * ((V*V).sum(axis=1)*self.dA*Cl_alpha)[:, None]
            J3 = J3 / (V_a**2 + V_n**2)[:, None]
            J4 = 2*self.dA*Cl*einsum('ik,jik->ij', V, v)
            J = J1 + J2 - J3 - J4

            # Compute the Gamma update term
            delta_Gamma = np.linalg.solve(J, -f)

            # Use the residual error and gradient to update the Gamma proposal
            Gamma = Gamma + Omega*delta_Gamma

            Vs.append(V)
            alphas.append(alpha)
            delta_Gammas.append(delta_Gamma)
            Gammas.append(Gamma)
            fs.append(f)
            Js.append(J)
            Cl_alphas.append(Cl_alpha)

            # print("finished run", n_runs)
            # embed()
            # 1/0

            # FIXME: ad-hoc workaround to avoid massively negative AoA
            Omega += (1 - Omega)/4

            n_runs += 1

        # embed()

        # if n_runs < 10:
        #     thinning = 1
        # elif n_runs < 26:
        #     thinning = 2
        # else:
        #     thinning = 5
        # thinning = 1
        # Gammas = Gammas[::thinning]

        # for n, G in enumerate(Gammas):
        #     plt.plot(cp_y, G, marker='.', label=n*thinning)
        # plt.ylabel('Gamma')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return Gamma, V

    def __call__(self, V_rel, delta, rho=1):
        # FIXME: depenency on rho?
        # FIXME: include viscous effects as well; ref: the Phillips paper
        Gamma, V = self._vortex_strengths(V_rel, delta)
        dF = Gamma[:, None] * cross(self.dl, V)
        dM = None
        return dF, dM


class Phillips2D(ForceEstimator):
    """
    This estimator is based on `Phillips` but it uses the 2D section lift
    coefficients directly instead of calculating the bound vorticity. This is
    equivalent to neglecting the induced velocities from other segments.

    See the documentation for `Phillips` for more information.
    """

    def __init__(self, parafoil, lobe_args={}):
        self.parafoil = parafoil
        self.lobe_args = lobe_args

        # Define the spanwise and nodal and control points

        # Option 1: linear distribution; less likely to induce large velocties
        self.K = 31  # The number of bound vortex segments
        self.s_nodes = np.linspace(-1, 1, self.K+1)

        # Option 2: cosine distribution; fast, very sensitive to segment length
        # self.K = 13  # The number of bound vortex segments
        # self.s_nodes = np.cos(np.linspace(np.pi, 0, self.K+1))

        # Nodes are indexed from 0..K+1
        self.nodes = self.parafoil.c4(self.s_nodes)

        # Control points are indexed from 0..K
        self.s_cps = (self.s_nodes[1:] + self.s_nodes[:-1])/2
        self.cps = self.parafoil.c4(self.s_cps)

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]  # node N is at at axis0=N-1
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Wing section orientation unit vectors at each control point
        u = self.parafoil.section_orientation(self.s_cps, lobe_args)
        self.u_a, self.u_s, self.u_n = u[:, 0], u[:, 1], u[:, 2]

        # Define the differential areas as parallelograms by assuming a linear
        # chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]  # `R1 + R2`?
        node_chords = self.parafoil.planform.fc(self.s_nodes)
        c_avg = (node_chords[1:] + node_chords[:-1])/2
        chord_vectors = c_avg[:, None] * self.u_a  # FIXME: verify
        self.dA = norm(cross(chord_vectors, self.dl), axis=1)
        # FIXME: faster+correct to post_multiply the scalars `c_avg`?

    @property
    def control_points(self):
        cps = self.cps.view()  # FIXME: better than making a copy?
        cps.flags.writeable = False  # FIXME: make the base ndarray immutable?
        return cps

    def __call__(self, V_rel, delta, rho=1):
        # FIXME: dependency on rho?
        assert np.shape(V_rel) == (self.K, 3)

        # Compute the section local angle of attack
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_a = einsum('ik,ik->i', V_rel, self.u_a)  # Chordwise
        V_n = einsum('ik,ik->i', V_rel, self.u_n)  # Normal-wise
        alpha = arctan2(V_n, V_a)

        CL = self.parafoil.sections.Cl(self.s_cps, alpha, delta)
        CD = self.parafoil.sections.Cd(self.s_cps, alpha, delta)

        dL_hat = cross(self.dl, V_rel)
        dL_hat = dL_hat / norm(dL_hat, axis=1)[:, None]  # Lift unit vectors
        dL = (1/2 * np.sum(V_rel**2, axis=1) * self.dA * CL)[:, None] * dL_hat

        dD_hat = -(V_rel / norm(V_rel, axis=1)[:, None])  # Drag unit vectors
        dD = (1/2 * np.sum(V_rel**2, axis=1) * self.dA * CD)[:, None] * dD_hat

        dF = dL + dD
        dM = 0

        return dF, dM
