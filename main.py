import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`
import numpy as np
from numpy import sin, cos, sqrt, arcsin, arctan  # noqa: F401

from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

from IPython import embed  # noqa: F401

from Airfoil import Airfoil, LinearCoefficients, NACA4
from Wing import Wing, EllipticalWing
from plots import plot_airfoil_geo


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


def plot_wing(wing):
    """Make a plot of a 3D wing"""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-130, elev=25)

    b = wing.geometry.b
    for y in np.linspace(-b/2, b/2, 21):
        coords = wing.lower_surface(y, N=50)
        ax.plot(coords[:, 0], coords[:, 1], -coords[:, 2], c='r', zorder=.9,
                lw=0.8)
        coords = wing.upper_surface(y, N=50)
        ax.plot(coords[:, 0], coords[:, 1], -coords[:, 2], c='b', lw=0.8)

    y = np.linspace(-b/2, b/2, 51)
    ax.plot(wing.geometry.fx(y), y, -wing.geometry.fz(y), 'g--', lw=0.8)

    set_axes_equal(ax)
    plt.show()


def animated_wing_plotter():
    # Setup all the configurations I want to step through
    k = 30
    dmeds = np.linspace(-30, -15, k)[::-1]
    smeds = np.linspace(10, 25, k)

    # First, just sweep the dihedral forward and backwards
    t1 = np.r_[dmeds, dmeds[::-1]]
    t2 = np.r_[smeds, smeds[::-1]]

    seq1 = np.c_[t1, 10*np.ones_like(t1)]   # Sweep dihedralMed
    seq2 = np.c_[-15*np.ones_like(t2), t2]  # Sweep sweepMed
    seq3 = np.c_[t1, t2]                    # Sweep both
    seq = np.vstack((seq1, seq2, seq3))

    fig = plt.figure(figsize=(10, 10))
    ax = p3.Axes3D(fig)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-6, 4)
    ax.view_init(azim=-130, elev=25)

    N = 21  # How many airfoil slices (makes 2N lines, for top and bottom)
    lines = [ax.plot([0], [0], [0], 'r' if n < N else 'b', lw=0.8)[0]
             for n in range(2*N)]

    def update(frame):
        print("seq[{}]: {}".format(frame, seq[frame]))
        dMed, sMed = seq[frame]

        wing = build_elliptical(MAC=2.4, AR=3.9, taper=0.4,
                                dMed=dMed, sMed=sMed)

        ys = np.linspace(-wing.geometry.b/2, wing.geometry.b/2, N)

        # Update the bottom lines
        for n in range(N):
            coords = wing.lower_surface(ys[n], N=50)
            coords[:, 2] = -coords[:, 2]
            lines[n].set_data(coords[:, 0:2].T)
            lines[n].set_3d_properties(coords[:, 2])

        # Update the top lines
        for n in range(N):
            coords = wing.upper_surface(ys[n], N=50)
            coords[:, 2] = -coords[:, 2]
            lines[n+N].set_data(coords[:, 0:2].T)
            lines[n+N].set_3d_properties(coords[:, 2])

        return lines

    ani = animation.FuncAnimation(fig, update, frames=np.arange(len(seq)),
                                  interval=10)
    # ani.save('test.mp4', fps=25)
    plt.show()


def animate_wing_torsion():
    # Setup all the configurations I want to step through
    k = 85
    torsions = np.linspace(-20, 20, k)
    torsions = np.r_[torsions, torsions[::-1]]
    torsions = np.r_[torsions, torsions]

    fig = plt.figure(figsize=(10, 10))
    ax = p3.Axes3D(fig)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-6, 4)
    ax.view_init(azim=-145, elev=0)

    N = 21  # How many airfoil slices (makes 2N lines, for top and bottom)
    lines = [ax.plot([0], [0], [0], 'r' if n < N else 'b', lw=0.8)[0]
             for n in range(2*N)]

    def update(frame):

        wing = build_elliptical(MAC=2.2, AR=4.5, taper=0.4,
                                dMed=-20, sMed=10, torsion=torsions[frame])

        ys = np.linspace(-wing.geometry.b/2, wing.geometry.b/2, N)

        # Update the bottom lines
        for n in range(N):
            coords = wing.lower_surface(ys[n], N=50)
            coords[:, 2] = -coords[:, 2]
            lines[n].set_data(coords[:, 0:2].T)
            lines[n].set_3d_properties(coords[:, 2])

        # Update the top lines
        for n in range(N):
            coords = wing.upper_surface(ys[n], N=50)
            coords[:, 2] = -coords[:, 2]
            lines[n+N].set_data(coords[:, 0:2].T)
            lines[n+N].set_3d_properties(coords[:, 2])

        return lines

    ani = animation.FuncAnimation(fig, update, frames=np.arange(len(torsions)),
                                  interval=20)
    # ani.save('torsion.mp4', fps=50)
    plt.show()


def build_elliptical(MAC, AR, taper, dMed, sMed, dMax=None, sMax=None,
                     torsion=0, airfoil_geo=None):
    if dMax is None:
        print("Using minimum max dihedral")
        dMax = 2*dMed - 1  # ref page 48 (56)

    if sMax is None:
        print("Using minimum max sweep")
        sMax = (2*sMed) + 1  # ref page 48 (56)

    # Compute some missing data in reverse
    c0 = EllipticalWing.MAC_to_c0(MAC, taper)
    b = EllipticalWing.AR_to_b(c0, AR, taper)

    dcg = 0.25  # FIXME: unlisted
    h0 = 7  # FIXME: unlisted
    wing_geo = EllipticalWing(
        b, c0, taper, dMed, dMax, sMed, sMax, torsion=torsion)

    if airfoil_geo is None:
        airfoil_geo = NACA4(2415)

    coefs = LinearCoefficients(5.73, -2, 0.007, -0.05)  # a0, i0, D0, Cm0

    return Wing(wing_geo, Airfoil(coefs, airfoil_geo))


if __name__ == "__main__":
    # plot_airfoil_geo(NACA4(2412))
    # plot_airfoil_geo(NACA4(4412))
    # plot_airfoil_geo(NACA4(2415))

    # animated_wing_plotter()
    animate_wing_torsion()

    print("\n\n-----\nTrying to produce the 'standard wing' from page 89 (97)")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=10)
    plot_wing(wing)

    print("\n\nPausing\n\n")
    embed()

    # Demonstrate how dMax controls "smoothness"
    print("With dMax=-70")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, dMax=-55, sMed=10)
    plot_wing(wing)
    print("With dMax=-85")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, dMax=-85, sMed=10)
    plot_wing(wing)

    # Demonstrate how sMax controls ???
    print("sMed=10, sMax=50")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=10, sMax=50)
    plot_wing(wing)
    print("sMed=10, sMax=85")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=10, sMax=85)
    plot_wing(wing)
    print("sMed=30, sMax=85")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=30, sMax=85)
    plot_wing(wing)

    # Torsion
    print("With 15 degree torsion")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=10, torsion=15)
    plot_wing(wing)
    print("With -15 degree torsion")
    wing = build_elliptical(
        MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=10, torsion=-15)
    plot_wing(wing)

    # More wings from page 89
    wing = build_elliptical(MAC=2.4, AR=3.9, taper=0.4, dMed=-35, sMed=10)
    plot_wing(wing)

    wing = build_elliptical(MAC=2.4, AR=3.9, taper=0.4, dMed=-20, sMed=25)
    plot_wing(wing)

    wing = build_elliptical(MAC=2.2, AR=4.9, taper=0.4, dMed=-20, sMed=10)
    plot_wing(wing)

    wing = build_elliptical(MAC=2.3, AR=4.0, taper=0.6, dMed=-20, sMed=10)
    plot_wing(wing)
