import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

import numpy as np


def _set_axes_equal(ax):
    """
    Set equal scaling for 3D plot axes.

    This ensures that spheres appear as spheres, cubes as cubes, etc.  This is
    one possible solution to Matplotlib's ``ax.set_aspect('equal')`` and
    ``ax.axis('equal')`` not working for 3D.

    Must be called after the data has been plotted, since that establishes the
    baseline axes limits. This function then computes a bounding sphere over
    those axes, and scales each axis until they have equal scales.

    Original source: https://stackoverflow.com/a/31364297. Modified to restore
    inverted axes.

    Parameters
    ----------
    ax: matplotlib axis
        The axes to equalize.
    """
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
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Restore any inverted axes
    if x_limits[0] > x_limits[1]:
        ax.invert_xaxis()
    if y_limits[0] > y_limits[1]:
        ax.invert_yaxis()
    if z_limits[0] > z_limits[1]:
        ax.invert_zaxis()


def _clean_3d_axes(ax, ticks=False, spines=False, panes=False):
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if not spines:
        ax.w_xaxis.line.set_color((1, 1, 1, 0))
        ax.w_yaxis.line.set_color((1, 1, 1, 0))
        ax.w_zaxis.line.set_color((1, 1, 1, 0))
    if not panes:
        ax.w_xaxis.set_pane_color((1, 1, 1, 0))
        ax.w_yaxis.set_pane_color((1, 1, 1, 0))
        ax.w_zaxis.set_pane_color((1, 1, 1, 0))


def _create_3d_axes(figsize=(12, 12), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca(projection="3d")
    ax.view_init(azim=-120, elev=20)
    ax.invert_yaxis()
    ax.invert_zaxis()
    return fig, ax


def plot_airfoil_geo(foil_geo, N_points=200):
    sa = (1 - np.cos(np.linspace(0, np.pi, N_points))) / 2
    upper = foil_geo.surface_curve(sa).T
    lower = foil_geo.surface_curve(-sa).T
    fig, ax = plt.subplots()
    ax.plot(upper[0], upper[1], c="b", lw=0.75)
    ax.plot(lower[0], lower[1], c="r", lw=0.75)

    pc = (1 - np.cos(np.linspace(0, np.pi, N_points))) / 2
    cc = foil_geo.camber_curve(pc).T
    ax.plot(
        cc[0],
        cc[1],
        label="mean camber line",
        color="k",
        linestyle="--",
        linewidth=0.75,
    )

    ax.set_aspect("equal")
    ax.margins(x=0.1, y=0.40)
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_airfoil_coef(airfoil, coef, N=100):
    """
    Parameters
    ----------
    airfoil : Airfoil
        The airfoil to plot.
    coef : {'cl', 'cl_alpha', 'cd', 'cm'}
        The airfoil coefficient to plot. Case-insensitive.
    N : integer
        The number of sample points per dimension
    """

    alpha = np.deg2rad(np.linspace(-10, 25, N))
    delta = np.deg2rad(np.linspace(0, 15, N))
    grid = np.meshgrid(alpha, delta)

    coef = coef.lower()
    if coef == "cl":
        values = airfoil.coefficients.Cl(grid[0], grid[1])
    elif coef == "cl_alpha":
        values = airfoil.coefficients.Cl_alpha(grid[0], grid[1])
    elif coef == "cd":
        values = airfoil.coefficients.Cd(grid[0], grid[1])
    elif coef == "cm":
        values = airfoil.coefficients.Cm(grid[0], grid[1])
    else:
        raise ValueError("`coef` must be one of {cl, cl_alpha, cd, cm}")

    fig = plt.figure(figsize=(17, 15))
    ax = fig.gca(projection="3d")
    ax.plot_surface(np.rad2deg(grid[0]), np.rad2deg(grid[1]), values)

    try:  # Kludge: Try to plot the raw coefficient data from the DataFrame
        for delta, group in airfoil.coefficients.data.groupby("delta"):
            deltas = np.full_like(group["alpha"], delta)
            if coef == "cl":
                values = airfoil.coefficients.Cl(group["alpha"], deltas)
            elif coef == "cd":
                values = airfoil.coefficients.Cd(group["alpha"], deltas)
            elif coef == "cm":
                values = airfoil.coefficients.Cm(group["alpha"], deltas)
            else:  # FIXME: does the data ever provide `cl_alpha` directly?
                break
            ax.plot(np.rad2deg(group["alpha"]), np.rad2deg(deltas), values)
    except AttributeError:
        pass

    ax.set_xlabel("alpha [degrees]")
    ax.set_ylabel("delta [degrees]")
    ax.set_zlabel(coef)
    plt.show()


def plot_foil(foil, N_sections=21, N_points=50, flatten=False, ax=None):
    """Plot a FoilGeometry in 3D."""
    if ax is None:
        fig, ax = _create_3d_axes()
        independent_plot = True
    else:
        independent_plot = False

    sa = 1 - np.cos(np.linspace(np.pi / 2, 0, N_points))
    for s in np.linspace(-1, 1, N_sections):
        coords = foil.surface_points(s, sa, "lower", flatten=flatten).T
        ax.plot(coords[0], coords[1], coords[2], c="r", zorder=0.9, lw=0.25)
        coords = foil.surface_points(s, sa, "upper", flatten=flatten).T
        ax.plot(coords[0], coords[1], coords[2], c="b", lw=0.25)

    s = np.linspace(-1, 1, N_sections)
    LE = foil.chord_xyz(s, 0, flatten=flatten).T
    c4 = foil.chord_xyz(s, 0.25, flatten=flatten).T
    TE = foil.chord_xyz(s, 1, flatten=flatten).T
    ax.plot(LE[0], LE[1], LE[2], "k--", lw=0.8)
    ax.plot(c4[0], c4[1], c4[2], "g--", lw=0.8)
    ax.plot(TE[0], TE[1], TE[2], "k--", lw=0.8)

    _set_axes_equal(ax)
    ax.set_proj_type('ortho')  # FIXME: better for this application?

    # Plot projections of the quarter-chord
    xlim = ax.get_xlim3d()
    zlim = ax.get_zlim3d()

    # Outline and quarter-chord projection onto the xy-pane (`z` held fixed)
    z = max(zlim)
    z *= 1.035  # Fix the distortion due to small distance from the xy-pane
    vertices = np.vstack((LE[0:2].T, TE[0:2].T[::-1]))  # shape: (2 * N_sections, 2)
    poly = PolyCollection([vertices], facecolors=['k'], alpha=0.25)
    ax.add_collection3d(poly, zs=[z], zdir='z')
    ax.plot(c4[0], c4[1], z, "g--", lw=0.8)

    # `x` reference curve projection onto the xy-pane
    xyz = foil.chord_xyz(s, foil._chords.r_x(s))
    x, y = xyz[..., 0], xyz[..., 1]
    ax.plot(x, y, z, 'r--', lw=0.8, label="reference lines")

    # Quarter-chord projection onto the yz-pane (`x` held fixed)
    x = np.full(*c4[1].shape, min(xlim))
    x *= 1.035  # Fix distortion due to small distance from the yz-pane
    ax.plot(x, c4[1], c4[2], "g--", lw=0.8, label="quarter-chord")

    # `yz` reference curve projection onto the yz-pane
    xyz = foil.chord_xyz(s, foil._chords.r_yz(s))
    y, z = xyz[..., 1], xyz[..., 2]
    ax.plot(x, y, z, 'r--', lw=0.8)

    ax.legend()

    if independent_plot:
        fig.tight_layout()
        plt.show()
    else:
        return (*ax.lines, *ax.collections)


def plot_foil_topdown(foil, N_sections=21, N_points=50, flatten=False, ax=None):
    """Plot a 3D foil in topdown projection."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        independent_plot = True
    else:
        independent_plot = False

    for s in np.linspace(-1, 1, N_sections):
        LE_xy = foil.chord_xyz(s, 0, flatten=flatten)[:2]
        TE_xy = foil.chord_xyz(s, 1, flatten=flatten)[:2]
        coords = np.stack((LE_xy, TE_xy))
        ax.plot(coords.T[1], coords.T[0], linewidth=0.75, c='k')

    s = np.linspace(-1, 1, N_sections)
    LE = foil.chord_xyz(s, 0, flatten=flatten)
    TE = foil.chord_xyz(s, 1, flatten=flatten)
    ax.plot(LE.T[1], LE.T[0], linewidth=0.75, c='k')
    ax.plot(TE.T[1], TE.T[0], linewidth=0.75, c='k')

    if independent_plot:
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.show()
    else:
        return ax.lines


def plot_wing(wing, delta_Bl=0, delta_Br=0, N_sections=131, N_points=50, ax=None):
    """
    Plot a ParagliderWing using 3D cross-sections.

    Uses a dashed black line to approximately visualize brake deflections.
    Deflections are assumed to start at 80% of the section chord, and deflect
    the last 20% of the chord as a straight line to an angle `delta` downwards
    from the section chord.

    This isn't terribly accurate, but it's decently helpful for checking if
    a brake deflection distribution seems reasonable.
    """
    if ax is None:
        fig, ax = _create_3d_axes()
        independent_plot = True
    else:
        independent_plot = False

    plot_foil(wing.parafoil, N_sections=N_sections, N_points=N_points, ax=ax)

    # Add a dashed brake deflection line
    s = np.linspace(-1, 1, N_sections)
    delta = wing.brake_geo(s, delta_Bl, delta_Br)
    flap = delta / 0.2
    c = wing.parafoil.chord_length(s)
    orientations = wing.parafoil.section_orientation(s)
    p = (np.array([-0.8 * c, np.zeros_like(s), np.zeros_like(s)])
         + 0.2 * c * np.array([-np.cos(flap), np.zeros_like(s), np.sin(flap)]))
    p = np.einsum("Sij,Sj->Si", orientations, p.T) + wing.parafoil.chord_xyz(s, 0)
    ax.plot(p.T[0], p.T[1], p.T[2], "k--", lw=0.8)

    if independent_plot:
        _set_axes_equal(ax)
        ax.view_init(azim=0, elev=0)  # Rear view to see deflections
        fig.tight_layout()
        plt.show()
    else:
        return ax.lines
