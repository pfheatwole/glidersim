import abc

import numpy as np
from numpy import sin, cos, arctan, arctan2, dot, cross, einsum
from numpy.polynomial import Polynomial
from numpy.linalg import norm

from numba import njit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

from util import trapz, cross3

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
    def __init__(self, geometry, airfoil, wing_density=0.2):
        self.geometry = geometry
        self.airfoil = airfoil
        self.wing_density = wing_density  # FIXME: no idea in general

    @property
    def density_factor(self):
        # FIXME: I don't understand this. Ref: PFD 48 (56)
        return self.geometry.MAC * self.airfoil.t*self.airfoil.chord/3

    def fE(self, y, xa=None, N=150):
        """Airfoil upper camber line on the 3D wing

        Parameters
        ----------
        y : float
            Position on the span, where `-b/2 < y < b/2`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord
        """

        # FIXME: rename: "extrudo" isn't English

        if xa is None:
            xa = np.linspace(0, 1, N)

        fc = self.geometry.fc(y)  # Chord length at `y` on the span
        upper = fc*self.airfoil.geometry.fE(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        Gamma = self.geometry.Gamma(y)

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) - zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(Gamma)
        z = self.geometry.fz(y) - \
            ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(Gamma)

        return np.c_[x, _y, z]

    def fI(self, y, xa=None, N=150):
        """Airfoil lower camber line on the 3D wing

        Parameters
        ----------
        y : float
            Position on the span, where `-b/2 < y < b/2`
        xa : float or array of float, optional
            Positions on the chord line, where all `0 < xa < chord`
        N : integer, optional
            If xa is `None`, sample `N` points along the chord
        """

        # FIXME: rename: "intrudo" isn't English

        if xa is None:
            xa = np.linspace(0, 1, N)

        fc = self.geometry.fc(y)  # Chord length at `y` on the span
        upper = fc*self.airfoil.geometry.fI(xa)  # Scaled airfoil
        xs, zs = upper[:, 0], upper[:, 1]

        theta = self.geometry.ftheta(y)
        Gamma = self.geometry.Gamma(y)

        x = self.geometry.fx(y) + (fc/4 - xs)*cos(theta) + zs*sin(theta)
        _y = y + ((fc/4 - xs)*sin(theta) + zs*cos(theta))*sin(Gamma)
        z = self.geometry.fz(y) - \
            ((fc/4 - xs)*sin(theta) + zs*cos(theta))*cos(Gamma)

        return np.c_[x, _y, z]


class CoefficientsEstimator(abc.ABC):
    @abc.abstractmethod
    def Cl(self, y, alpha, delta_Bl, delta_Br):
        """The lift coefficient for the parafoil section"""

    @abc.abstractmethod
    def Cd(self, y, alpha, delta_Bl, delta_Br):
        """The drag coefficient for the parafoil section"""

    @abc.abstractmethod
    def Cm(self, y, alpha, delta_Bl, delta_Br):
        """The pitching moment coefficient for the parafoil section"""

    def _pointwise_global_coefficients(self, alpha, delta_B):
        """
        Compute point estimates of the global CL, CD, and Cm

        This procedure is from Paraglider Flight Dynamics, Sec:4.3.3

        Parameters
        ----------
        alpha : float [radians]
            The angle of attack for the current point estimate
        delta_B : float [percentage]
            The amount of symmetric brakes, where `0 <= delta_b <= 1`

        Returns
        -------
        CL : float
            The global lift coefficient
        CD : float
            The global drag coefficient
        CM_c4 : float
            The global pitching moment coefficient about quarter chord of the
            central chord.
        """

        # Build a set of integration points
        N = 501
        dy = self.parafoil.geometry.b/(N - 1)  # Include the endpoints
        y = np.linspace(-self.parafoil.geometry.b/2,
                        self.parafoil.geometry.b/2, N)

        # Compute local relative winds to match `alpha`
        uL, wL = np.cos(alpha), np.sin(alpha)

        # Compute the local forces
        dF, dM = self.section_forces(y, uL, 0, wL, delta_B, delta_B)
        dFx, dFz, dMy = dF[:, 0], dF[:, 2], dM[:, 1]  # Convenience views

        # Convert the body-oriented forces into relative wind coordinates
        # PFD Eq:4.29-4.30, p77
        Li_prime = dFx*sin(alpha) - dFz*cos(alpha)
        Di_prime = -dFx*cos(alpha) - dFz*sin(alpha)
        L = trapz(Li_prime, dy)
        D = trapz(Di_prime, dy)
        My = trapz(dMy, dy)

        # Relocating the dFx and dFz to the central quarter chord introduces
        # new pitching moments. These must be subtracted from the overall
        # pitching moment for the global force+moment pair to be equivalent.
        c4 = self.parafoil.geometry.fx(0)
        x = self.parafoil.geometry.fx(y)
        z = self.parafoil.geometry.fz(y)
        mFx = trapz(dFx*(0-z), dy)  # The moment from relocating dFx to c4
        mFz = -trapz(dFz*(c4-x), dy)  # The moment from relocating dFz to c4
        My_c4 = My - mFx - mFz  # Apply counteracting moments for equivalence

        # Compute the global coefficients
        # Note: in this context, rho=1 and V_infinity=1 and have been dropped
        S = self.parafoil.geometry.S
        MAC = self.parafoil.geometry.MAC
        CL = L / (1/2 * S)
        CD = D / (1/2 * S)
        CM_c4 = My_c4 / (1/2 * S * MAC)

        return CL, CD, CM_c4

    # FIXME: this doesn't seem to belong in a "CoefficientsEstimator"!
    #
    #        Update: doubly so, since Phillip's method has it's own equations
    #        based on the estimated circulation
    def section_forces(self, y, uL, vL, wL, delta_Bl=0, delta_Br=0, rho=1):
        """
        Compute section forces and moments acting on the wing.

        Parameters
        ----------
        y : float or array of float
            The section position on the wing span, where -b/2 < y < b/2
        uL : float or array of float
            Section-local relative wind aligned to the Parafoil x-axis
        vL : float or array of float
            Section-local relative wind aligned to the Parafoil y-axis
        wL : float or array of float
            Section-local relative wind aligned to the Parafoil z-axis
        delta_Bl : float [percentage]
            The amount of left brake
        delta_Br : float [percentage]
            The amount of right brake
        rho : float
            Air density

        Returns
        -------
        dF : ndarray, shape (M, 3)
            The force differentials parallel to the X, Y, and Z axes
        dM : ndarray, shape (M, 3)
            The moment differentials about the X, Y, and Z axes
        """

        # FIXME: verify docstring. These are forces per unit span, not the

        Gamma = self.parafoil.geometry.Gamma(y)  # PFD eq 4.13, p74
        theta = self.parafoil.geometry.ftheta(y)

        # Compute the section (local) relative wind
        u = uL  # PFD Eq:4.14, p74
        w = wL*cos(Gamma) - vL*sin(Gamma)  # PFD Eq:4.15, p74
        alpha = arctan(w/u) + theta  # PFD Eq:4.17, p74

        Cl = self.Cl(y, alpha, delta_Bl, delta_Br)
        Cd = self.Cd(y, alpha, delta_Bl, delta_Br)
        Cm = self.Cm(y, alpha, delta_Bl, delta_Br)

        # PFD Eq:4.65-4.67 (with an implicit `dy` term)
        c = self.parafoil.geometry.fc(y)
        K1 = (rho/2)*(u**2 + w**2)
        K2 = 1/cos(Gamma)
        dL = K1*Cl*c*K2
        dD = K1*Cd*c*K2
        dm0 = K1*Cm*(c**2)*K2

        # Translate the section forces and moments into Parafoil body axes
        #  * PFD Eqs:4.23-4.28, pp76-77
        dF_perp_x = dL*cos(alpha - theta) + dD*sin(alpha - theta)
        dF_par_x = dL*sin(alpha - theta) - dD*cos(alpha - theta)
        dF_par_y = dF_perp_x * sin(Gamma)
        dF_par_z = -dF_perp_x * cos(Gamma)
        dM_par_x = np.zeros_like(y)
        dM_par_y = dm0*cos(Gamma)
        dM_par_z = -dm0*sin(Gamma)  # FIXME: verify

        dF = np.vstack([dF_par_x, dF_par_y, dF_par_z]).T
        dM = np.vstack([dM_par_x, dM_par_y, dM_par_z]).T

        return dF, dM


class Coefs2D(CoefficientsEstimator):
    """
    Returns the 2D airfoil coefficients with no adjustments.

    FIXME: Only works for parafoils with constant airfoils
    """

    def __init__(self, parafoil, brake_geo):
        self.parafoil = parafoil
        self.brake_geo = brake_geo

        # Compute 25 Polynomial approximations over the operating brake range
        self.delta_Bs = np.linspace(0, 1, 25)

        # Initialize the global aerodynamic coefficients
        CLs, CDs, CM_c4s = self._compute_global_coefficients()
        self.CLs = CLs
        self.CDs = CDs
        self.CM_c4s = CM_c4s

    # FIXME: reparameterize the coefficients to use the normalized deltas

    def Cl_alpha(self, y, alpha, delta_Bl, delta_Br):
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha
        delta_distance = self.brake_geo(y, delta_Bl, delta_Br)

        # Convert the deflection distances to deflection angles.
        #
        # From the arc length equation, `s = radius*theta`, where `s` is the
        # trailing edge deflection distance, `radius` is the chord length, and
        # `theta` is the deflection angle.
        c = self.parafoil.geometry.fc(y)
        delta_angle = delta_distance / c

        return self.parafoil.airfoil.coefficients.Cl_alpha(alpha, delta_angle)

    def Cl(self, y, alpha, delta_Bl, delta_Br):
        # FIXME: make AirfoilCoefficients responsible for broadcasting `alpha`?
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha

        c = self.parafoil.geometry.fc(y)
        delta = self.brake_geo(y, delta_Bl, delta_Br) / c

        return self.parafoil.airfoil.coefficients.Cl(alpha, delta)

    def Cd(self, y, alpha, delta_Bl, delta_Br):
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha

        c = self.parafoil.geometry.fc(y)
        delta = self.brake_geo(y, delta_Bl, delta_Br) / c

        return self.parafoil.airfoil.coefficients.Cd(alpha, delta)

    def Cm(self, y, alpha, delta_Bl, delta_Br):
        if np.isscalar(alpha):
            alpha = np.ones_like(y) * alpha

        c = self.parafoil.geometry.fc(y)
        delta = self.brake_geo(y, delta_Bl, delta_Br) / c

        return self.parafoil.airfoil.coefficients.Cm0(alpha, delta)

    # FIXME: doesn't do any interpolation between deltas
    def CL(self, alpha, delta_B):
        di = np.argmin(np.abs(self.delta_Bs - delta_B))
        return self.CLs[di](alpha)

    def CD(self, alpha, delta_B):
        di = np.argmin(np.abs(self.delta_Bs - delta_B))
        return self.CDs[di](alpha)

    def CM_c4(self, alpha, delta_B):
        di = np.argmin(np.abs(self.delta_Bs - delta_B))
        return self.CM_c4s[di](alpha)

    def _compute_global_coefficients(self):
        alphas = np.deg2rad(np.linspace(-1.99, 24, 100))
        _CLs, _CDs, _CM_c4s = [], [], []

        for n_d, delta_B in enumerate(self.delta_Bs):
            CLs = np.empty_like(alphas)
            CDs = np.empty_like(alphas)
            CM_c4s = np.empty_like(alphas)

            for n_a, alpha in enumerate(alphas):
                CL, CD, CM_c4 = self._pointwise_global_coefficients(
                        alpha, delta_B)
                CLs[n_a], CDs[n_a], CM_c4s[n_a] = CL, CD, CM_c4

            CL = Polynomial.fit(alphas, CLs, 5)
            CD = Polynomial.fit(alphas, CDs, 5)
            CM_c4 = Polynomial.fit(alphas, CM_c4s, 2)

            _CLs.append(CL)
            _CDs.append(CD)
            _CM_c4s.append(CM_c4)

        return _CLs, _CDs, _CM_c4s


class Phillips(CoefficientsEstimator):
    """
    Supposed to work with wings with sweep and dihedral, but much more complex.

    References
    ----------
    .. [1] Phillips and Snyder, "Modern Adaptation of Prandtl’s Classic
       Lifting- Line Theory", Journal of Aircraft, 2000

    .. [2] McLeanauth, "Understanding Aerodynamics - Arguing from the Real
       Physics", p382

    .. [3] Hunsaker and Snyder, "A lifting-line approach to estimating
       propeller/wing interactions", 2006

    Notes
    -----
    This method does suffer an issue where induced velocity goes to infinity as
    the segment length is decreased. See _[2], section 8.2.3.
    """

    def __init__(self, parafoil, brake_geo):
        self.parafoil = parafoil
        self.brake_geo = brake_geo
        self.coefs2d = Coefs2D(parafoil, brake_geo)

        # Define the spanwise and nodal and control points
        # NOTE: this is suitable for parafoils, but for wings made of left
        #       and right segments, you should distribute the points across
        #       each span independently. See _[1].
        # FIXME: Phillips indexes the nodal points from zero, and the control
        #        points from 1. Should I do the same?
        # FIXME: how many segments for reasonable accuracy?
        self.K = 51  # The number of bound vortex segments for the entire span
        k = np.arange(self.K+1)
        b = self.parafoil.geometry.b

        # Nodes are indexed from 0..K+1
        node_y = (-b/2) * np.cos(k * np.pi / self.K)
        node_x = self.parafoil.geometry.fx(node_y)
        node_z = self.parafoil.geometry.fz(node_y)
        self.nodes = np.c_[node_x, node_y, node_z]

        # Control points are indexed from 0..K
        cp_y = (-b/2) * (np.cos(np.pi/(2*self.K) + k[:-1]*np.pi/self.K))
        cp_x = self.parafoil.geometry.fx(cp_y)
        cp_z = self.parafoil.geometry.fz(cp_y)
        self.cps = np.c_[cp_x, cp_y, cp_z]

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]  # node N is at at axis0=N-1
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Define the orthogonal unit vectors for each control point
        # FIXME: these need verification; their orientation in particular
        # FIXME: also, check their magnitudes
        dihedral = self.parafoil.geometry.Gamma(cp_y)
        twist = self.parafoil.geometry.ftheta(cp_y)  # Angle of incidence
        sd, cd = sin(dihedral), cos(dihedral)
        st, ct = sin(twist), cos(twist)
        self.u_s = np.c_[np.zeros_like(cp_y), cd, sd]  # Spanwise
        self.u_a = np.c_[ct, st*sd, st*cd]  # Chordwise
        self.u_n = np.cross(self.u_a, self.u_s)  # Normal to the span and chord

        assert np.allclose(norm(self.u_s, axis=1), 1)
        assert np.allclose(norm(self.u_a, axis=1), 1)
        assert np.allclose(norm(self.u_n, axis=1), 1)

        # Define the differential areas. Uses a trapezoidal area by assuming a
        # linear chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        c_nodes = self.parafoil.geometry.fc(self.nodes[:, 1])
        self.c_avg = (c_nodes[1:] + c_nodes[:-1])/2
        self.dA = self.c_avg * norm(self.dl, axis=1)
        print("DEBUG> using the dl to compute dA")
        # FIXME: does the planform area use dl or dy?

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
            # ax.plot(self.parafoil.geometry.fx(y), y, -self.parafoil.geometry.fz(y), 'g--', lw=0.8)

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

    def Cl(self, y, alpha, delta_Bl, delta_Br):
        raise NotImplementedError

    def Cd(self, y, alpha, delta_Bl, delta_Br):
        raise NotImplementedError

    def Cm(self, y, alpha, delta_Bl, delta_Br):
        raise NotImplementedError

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

    def _vortex_strengths(self, V_rel, delta_Bl, delta_Br, max_runs=None):
        """
        FIXME: finish the docstring

        Parameters
        ----------
        V_rel : array of float, shape (K,) [meters/second]
            Fluid velocity vectors for each section, in body coordinates. This
            is equal to the relative wind "far" from each wing section, which
            is absent of circulation effects.
        delta_Bl : float [percentage]
            The amount of left brake
        delta_Br : float [percentage]
            The amount of right brake

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
        b = self.parafoil.geometry.b
        cp_y = self.cps[:, 1]
        Gamma0 = 5

        # Alternative initial proposal
        # avg_brake = (delta_Bl + delta_Br)/2
        # CL_2d = self.coefs2d.CL(np.arctan2(u_inf[2], u_inf[0]), avg_brake)
        # S = self.parafoil.geometry.S
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
        M = max(delta_Bl, delta_Br)  # Assumes the delta_B are 0..1
        base_Omega, min_Omega = 0.2, 0.05
        Omega = base_Omega - (base_Omega - min_Omega)*np.sqrt(M)
        if max_runs is None:
            # max_runs = 5 + int(np.ceil(3*M))
            max_runs = 10

        # FIXME: don't use a fixed number of runs
        # FIXME: how much faster is `opt_einsum` versus the scipy version?
        # FIXME: if `coefs2d.Cl` was Numba compatible, what about this loop?
        n_runs = 0
        while n_runs < max_runs:
            print("run:", n_runs)
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

            Cl = self.coefs2d.Cl(cp_y, alpha, delta_Bl, delta_Br)

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
            Cl_alpha = self.coefs2d.Cl_alpha(cp_y, alpha, delta_Bl, delta_Br)

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

    def forces_and_moments(self, V_rel, delta_Bl, delta_Br, rho=1):
        # FIXME: depenency on rho?
        # FIXME: include viscous effects as well; ref: the Phillips paper
        Gamma, V = self._vortex_strengths(V_rel, delta_Bl, delta_Br)
        dF = Gamma[:, None] * cross(self.dl, V)
        dM = None
        return dF, dM


class Phillips2D(CoefficientsEstimator):
    """
    This is a finite-element method, based on Phillips, but it uses the 2D
    section lift coefficients directly instead of calculating the bound
    vorticity. This is equivalent to neglecting the induced velocities from
    other segments.
    """

    def __init__(self, parafoil, brake_geo):
        self.parafoil = parafoil
        self.brake_geo = brake_geo
        self.coefs2d = Coefs2D(parafoil, brake_geo)

        # Define the spanwise and nodal and control points
        # NOTE: this is suitable for parafoils, but for wings made of left
        #       and right segments, you should distribute the points across
        #       each span independently. See _[1].
        # FIXME: Phillips indexes the nodal points from zero, and the control
        #        points from 1. Should I do the same?
        # FIXME: how many segments for reasonable accuracy?
        self.K = 51  # The number of bound vortex segments for the entire span
        k = np.arange(self.K+1)
        b = self.parafoil.geometry.b

        # Nodes are indexed from 0..K+1
        node_y = (-b/2) * np.cos(k * np.pi / self.K)
        node_x = self.parafoil.geometry.fx(node_y)
        node_z = self.parafoil.geometry.fz(node_y)
        self.nodes = np.c_[node_x, node_y, node_z]

        # Control points are indexed from 0..K
        cp_y = (-b/2) * (np.cos(np.pi/(2*self.K) + k[:-1]*np.pi/self.K))
        cp_x = self.parafoil.geometry.fx(cp_y)
        cp_z = self.parafoil.geometry.fz(cp_y)
        self.cps = np.c_[cp_x, cp_y, cp_z]

        # axis0 are nodes, axis1 are control points, axis2 are vectors or norms
        self.R1 = self.cps - self.nodes[:-1, None]
        self.R2 = self.cps - self.nodes[1:, None]  # node N is at at axis0=N-1
        self.r1 = norm(self.R1, axis=2)  # Magnitudes of R_{i1,j}
        self.r2 = norm(self.R2, axis=2)  # Magnitudes of R_{i2,j}

        # Define the orthogonal unit vectors for each control point
        # FIXME: these need verification; their orientation in particular
        # FIXME: also, check their magnitudes
        dihedral = self.parafoil.geometry.Gamma(cp_y)
        twist = self.parafoil.geometry.ftheta(cp_y)  # Angle of incidence
        sd, cd = sin(dihedral), cos(dihedral)
        st, ct = sin(twist), cos(twist)
        self.u_s = np.c_[np.zeros_like(cp_y), cd, sd]  # Spanwise
        self.u_a = np.c_[ct, st*sd, st*cd]  # Chordwise
        self.u_n = np.cross(self.u_a, self.u_s)  # Normal to the span and chord

        assert np.allclose(norm(self.u_s, axis=1), 1)
        assert np.allclose(norm(self.u_a, axis=1), 1)
        assert np.allclose(norm(self.u_n, axis=1), 1)

        # Define the differential areas. Uses a trapezoidal area by assuming a
        # linear chord variation between nodes.
        self.dl = self.nodes[1:] - self.nodes[:-1]
        c_nodes = self.parafoil.geometry.fc(self.nodes[:, 1])
        self.c_avg = (c_nodes[1:] + c_nodes[:-1])/2
        # self.dA = c_avg * np.diff(self.nodes[:, 1])  # ignores dihedral
        # self.dA = self.c_avg * np.diff(self.nodes[:, 1]) / cos(self.parafoil.geometry.Gamma(self.cps[:, 1]))
        self.dA = self.c_avg * norm(self.dl, axis=1)
        print("DEBUG> using the dl to compute dA")
        # FIXME: does the planform area use dl or dy?

    def Cl(self, y, alpha, delta_Bl, delta_Br):
        raise NotImplementedError

    def Cd(self, y, alpha, delta_Bl, delta_Br):
        raise NotImplementedError

    def Cm(self, y, alpha, delta_Bl, delta_Br):
        raise NotImplementedError

    def forces_and_moments(self, V_rel, delta_Bl, delta_Br, rho=1):
        # FIXME: dependency on rho?
        assert np.shape(V_rel) == (self.K, 3)

        cp_y = self.cps[:, 1]

        # Compute the section local angle of attack
        #  * ref: Phillips Eq:9 (dimensional) or Eq:12 (dimensionless)
        V_a = einsum('ik,ik->i', V_rel, self.u_a)  # Chordwise
        V_n = einsum('ik,ik->i', V_rel, self.u_n)  # Normal-wise
        alpha = arctan2(V_n, V_a)

        CL = self.coefs2d.Cl(cp_y, alpha, delta_Bl, delta_Br)
        CD = self.coefs2d.Cd(cp_y, alpha, delta_Bl, delta_Br)

        dL_hat = cross(self.dl, V_rel)
        dL_hat = dL_hat / norm(dL_hat, axis=1)[:, None]  # Lift unit vectors
        dL = (1/2 * np.sum(V_rel**2, axis=1) * self.dA * CL)[:, None] * dL_hat

        dD_hat = -(V_rel / norm(V_rel, axis=1)[:, None])  # Drag unit vectors
        dD = (1/2 * np.sum(V_rel**2, axis=1) * self.dA * CD)[:, None] * dD_hat

        dF = dL + dD
        dM = 0

        return dF, dM
