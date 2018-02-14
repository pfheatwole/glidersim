import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`
import numpy as np
from numpy import sin, cos, sqrt, arcsin, arctan  # noqa: F401

from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

from IPython import embed

from Airfoil import NACA4
from Wing import Wing, EllipticalWing


def plot_airfoil(foil):
    x = np.linspace(0, 1, 1500)
    upper = foil.fE(x)
    lower = foil.fI(x)

    fig, ax = plt.subplots()
    ax.plot(x, foil.yc(x), label='mean camber line')
    ax.plot(upper[:, 0], upper[:, 1], c='r', lw=0.75)
    ax.plot(lower[:, 0], lower[:, 1], c='b', lw=0.75)
    ax.scatter(0.25, foil.yc(0.25), c='k')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True)
    plt.show()


def sample_wing(wing):
    print("Collect a set of points along a 3D wing, suitable for plotting")
    return


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
    # ax.set_aspect('equal')
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-10, 10)

    b = wing.geometry.b
    for y in np.linspace(-b/2, b/2, 21):
        coords = wing.fI(y, N=50)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], c='r', zorder=.9)
        coords = wing.fE(y, N=50)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], c='b')


    y = np.linspace(-b/2, b/2, 51)
    ax.plot(wing.geometry.fx(y), y, -wing.geometry.fz(y), 'g--', lw=0.8)

    set_axes_equal(ax)
    plt.show()






def animated_wing_plotter():

    print("Trying out the animation module")

    # Setup all the configurations I want to step through
    # dmeds = np.arange(-30, -15)[::-1]
    # smeds = np.arange(10, 25)
    k = 50
    dmeds = np.linspace(-30, -15, k)[::-1]
    smeds = np.linspace(10, 25, k)

    print("dmeds.shape:", dmeds.shape)
    print("smeds.shape:", smeds.shape)

    # First, just sweep the dihedral forward and backwards
    t1 = np.r_[dmeds, dmeds[::-1]]
    seq1 = np.c_[t1, 10*np.ones_like(t1)]
    print("\nDEBUG> seq1:\n", seq1)

    # Second, sweep the sweep
    t2 = np.r_[smeds, smeds[::-1]]
    seq2 = np.c_[-15*np.ones_like(t2), t2]
    print("\nDEBUG> seq2:\n", seq2)

    # Third, sweep both simultaneously
    seq3 = np.c_[t1, t2]

    # Now, the final set of sequences
    seq = np.vstack((seq1, seq2, seq3))

    fig = plt.figure(figsize=(10, 10))
    # ax = fig.gca(projection='3d')
    ax = p3.Axes3D(fig)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(0, 11)
    ax.view_init(azim=-130, elev=25)

    N = 21  # How many slices (total number is 2N, for top and bottom)
    lines = [
        ax.plot([0], [0], [0], 'r' if n < N else 'b')[0] for n in range(2*N)
        ]

    def init():
        return lines

    def update(frame):
        S = 20.8
        MAC = 2.4
        AR = 3.9
        taper = 0.4
        tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
        c0 = (MAC / (2/3) / (2 + taper**2)) * (taper + tmp)
        b = (AR / 2)*c0*(taper + tmp)
        ys = np.linspace(-b/2, b/2, N)
        dcg = 0.25  # FIXME: unlisted
        h0 = 7  # FIXME: unlisted
        airfoil = NACA4(2415)

        print("seq[{}]: {}".format(frame, seq[frame]))
        dihedralMed = seq[frame][0]
        dihedralMax = 2*dihedralMed - 1

        sweepMed = seq[frame][1]
        sweepMax = (2*sweepMed) + 5  # ref page 48 (56)

        wing_geo = EllipticalWing(dcg, c0, h0, dihedralMed, dihedralMax, b,
                                  taper, sweepMed, sweepMax)
        wing = Wing(wing_geo, airfoil)

        # Update the bottom lines
        for n in range(N):
            coords = wing.fI(ys[n], N=50)
            lines[n].set_data(coords[:, 0:2].T)
            lines[n].set_3d_properties(coords[:, 2])

        for n in range(N):
            coords = wing.fE(ys[n], N=50)
            lines[n+N].set_data(coords[:, 0:2].T)
            lines[n+N].set_3d_properties(coords[:, 2])

        return lines

    ani = animation.FuncAnimation(fig, update, frames=np.arange(len(seq)))
    ani.save('test.mp4', fps=25)


if __name__ == "__main__":
    # plot_airfoil(NACA4(2412))
    # plot_airfoil(NACA4(4412))
    # plot_airfoil(NACA4(2415))

    print("\n\n-----\nTrying to produce the 'standard wing' from page 89 (97)")

    S = 20.8
    MAC = 2.4
    AR = 3.9
    taper = 0.4

    # ref page 48 (56), `dMax > 2*dMed`  (should be `<` since negative?)
    # dihedralMed, dihedralMax = -20, -45
    # dihedralMed, dihedralMax = -35, -55
    dihedralMed = -20
    # dihedralMed = -30
    # dihedralMed = -35
    dihedralMax = 2*dihedralMed - 1
    sweepMed = 10  # degrees
    # sweepMed = 15  # degrees
    # sweepMed = 25  # degrees
    sweepMax = (2*sweepMed) + 1  # ref page 48 (56)

    # Compute some missing data in revers
    tmp = arcsin(sqrt(1 - taper**2))/sqrt(1 - taper**2)
    c0 = (MAC / (2/3) / (2 + taper**2)) * (taper + tmp)
    # c0 *= .75
    print("c0:", c0)
    b = (AR / 2)*c0*(taper + tmp)
    print("b:", b)

    print("\nWIP!\n")
    # embed(display_banner=False)

    # def __init__(self, dcg, c0, h0, dihedralMed, dihedralMax, b, taper,
    #              sweepMed, sweepMax):

    dcg = 0.25  # FIXME: unlisted
    h0 = 7  # FIXME: unlisted
    wing_geo = EllipticalWing(dcg, c0, h0, dihedralMed, dihedralMax, b, taper,
                              sweepMed, sweepMax)

    airfoil = NACA4(2415)
    wing = Wing(wing_geo, airfoil)
    plot_wing(wing)

    print("\n\n\n\n\nHere goes...")
    animated_wing_plotter()
