import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

from IPython import embed


def plot_airfoil_geo(foil_geo):
    x = np.linspace(0, 1, 250)
    upper = foil_geo.upper_curve(x)
    lower = foil_geo.lower_curve(x)
    camberline = foil_geo.camber_curve(x)

    fig, ax = plt.subplots()
    ax.plot(camberline[:, 0], camberline[:, 1], label='mean camber line')
    ax.plot(upper[0], upper[1], c='r', lw=0.75)
    ax.plot(lower[0], lower[1], c='b', lw=0.75)
    ax.scatter(foil_geo.camber_curve(0.25)[0],
               foil_geo.camber_curve(0.25)[1], c='k')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True)
    plt.show()


def plot_airfoil_coef(airfoil, coef, N=100):
    """
    Parameters
    ----------
    airfoil : Airfoil
    coef : string
        The airfoil coefficient to plot: 'cl', 'cl_alpha', 'cd', or 'cm'
        (case insensitive)
    N : integer
        The number of sample points per dimension
    """

    alpha = np.deg2rad(np.linspace(-3, 22, N))
    delta = np.deg2rad(np.linspace(0, 50*(1-0.8), N))
    grid = np.meshgrid(alpha, delta)

    coef = coef.lower()
    if coef == 'cl':
        values = airfoil.coefficients.Cl(grid[0], grid[1])
    elif coef == 'cl_alpha':
        values = airfoil.coefficients.Cl_alpha(grid[0], grid[1])
    elif coef == 'cd':
        values = airfoil.coefficients.Cd(grid[0], grid[1])
    elif coef == 'cm':
        values = airfoil.coefficients.Cd(grid[0], grid[1])

    fig = plt.figure(figsize=(13, 10))
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid[0], grid[1], values)
    ax.set_xlabel('alpha [rad]')
    ax.set_ylabel('delta [rad]')
    ax.set_zlabel(coef)
    plt.show()


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


def plot_parafoil_geo(parafoil, N_sections=21, N_points=50):
    """Make a plot of a 3D wing"""

    fig = plt.figure(figsize=(16, 16))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-130, elev=25)

    for s in np.linspace(-1, 1, N_sections):
        coords = parafoil.lower_surface(s, N_points)
        ax.plot(coords[0], coords[1], -coords[2], c='r', zorder=.9,
                lw=0.8)
        coords = parafoil.upper_surface(s, N_points)
        ax.plot(coords[0], coords[1], -coords[2], c='b', lw=0.8)

    s = np.linspace(-1, 1, 51)
    c4 = parafoil.c4(s)
    ax.plot(c4[0], c4[1], -c4[2], 'g--', lw=0.8)

    set_axes_equal(ax)
    plt.show()
