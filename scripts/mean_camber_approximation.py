"""
Demonstrate using normalized surface curves to approximate mean camber lines.
It works fine as a rough approximation, but error accumulates depending on the
concavity of the mean line.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def plot_approximate_camber(xyu, xyl, xyc, ax):
    # Build surface curves that are parametrized by the normalized distance
    # along the curve (so `0` is the leading edge, `1` is the trailing edge)
    Lu = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyu.T), axis=0))]
    Ll = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyl.T), axis=0))]
    cu = PchipInterpolator(Lu / Lu[-1], xyu)
    cl = PchipInterpolator(Ll / Ll[-1], xyl)

    # Approximate the camber curve using the linear distance surface curves.
    s = np.linspace(0, 1, 250)
    ac = (cu(s) + cl(s)) / 2

    ax.plot(xyu.T[0], xyu.T[1], "b--", lw=0.75)
    ax.plot(xyl.T[0], xyl.T[1], "r--", lw=0.75)
    ax.plot(xyc.T[0], xyc.T[1], "k--", lw=0.75, label="true camber")
    ax.plot(ac.T[0], ac.T[1], "r--", lw=0.75, label="approximate camber")
    ax.legend()


def main():
    # Invent an airfoil with a sinusoidal camber line and quadratic thickness.
    #
    # Technically this "leading edge" makes it impossible for the chord or
    # mean camber lines to be perpendicular to the surface curve, and yet we
    # can define it. The "conventional" definitions of the curves make lots of
    # (typically) unstated assumptions like this. (This isn't a *good* foil,
    # and would have terrible stall behavior, but there's nothing stopping us
    # from building one.
    N = 250
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 2 * np.pi, N)  # Implicit: frequency = 1
    yc = 0.1 * np.sin(t)
    thickness = -0.5 * x**2 + 0.5 * x
    xyc = np.c_[x, yc]  # The xy-coordinates of the "true" camber curve

    fig, axes = plt.subplots(2)

    # Example 1: using the "vertical" convention
    xyu = np.c_[x, yc + thickness / 2]
    xyl = np.c_[x, yc - thickness / 2]
    plot_approximate_camber(xyu, xyl, xyc, axes[0])
    axes[0].text(0, -0.1, "Vertical convention")

    # Example 2: using the "perpendicular" convention
    dxdy = np.c_[np.ones(N), np.cos(t)].T
    dx, dy = dxdy / np.linalg.norm(dxdy, axis=0) * thickness / 2
    orthogonal_vector = np.c_[-dy, dx]
    xyu = xyc + orthogonal_vector
    xyl = xyc - orthogonal_vector
    plot_approximate_camber(xyu, xyl, xyc, axes[1])
    axes[1].text(0, -0.1, "Perpendicular convention")

    plt.show()


if __name__ == "__main__":
    main()
