from IPython import embed

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np

import pfh.glidersim as gsim


airfoil1 = gsim.airfoil.NACA(24018)
# airfoil2 = gsim.airfoil.NACA(22010)
# airfoil1 = gsim.airfoil.NACA(7111, convention='perpendicular')
airfoil2 = gsim.airfoil.NACA(7911, convention='perpendicular')

gsim.plots.plot_airfoil_geo(airfoil1)
gsim.plots.plot_airfoil_geo(airfoil2)

fig, axes = plt.subplots(2)
axes[0].set_xlim(-0.1, 1.1)
axes[0].set_ylim(-0.4, 0.4)
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(-0.4, 0.4)
# ax.set_aspect("equal")
# ax.margins(x=0.1, y=0.40)
ul1 = axes[0].plot([], [], c="b", lw=0.75)[0]
cl1 = axes[0].plot([], [], c="k", lw=0.75, linestyle="--")[0]
ll1 = axes[0].plot([], [], c="r", lw=0.75)[0]
ul2 = axes[1].plot([], [], c="k", lw=0.55)[0]
# cl2 = axes[1].plot([], [], c="k", lw=0.75, linestyle="--")[0]
ll2 = axes[1].plot([], [], c="k", lw=0.55)[0]
axes[0].grid(True)
axes[1].grid(True)
axes[0].set_aspect("equal")
axes[1].set_aspect("equal")
N_points = 300
N_steps = 60 * 3

pc = (1 - np.cos(np.linspace(0, np.pi, N_points))) / 2  # `0 <= x <= 1`
xyu1 = airfoil1.surface_curve(pc)
xyu2 = airfoil2.surface_curve(pc)
xyl1 = airfoil1.surface_curve(-pc)
xyl2 = airfoil2.surface_curve(-pc)
xyc1 = airfoil1.camber_curve(pc)
xyc2 = airfoil2.camber_curve(pc)
t1 = airfoil1.thickness(pc)
t2 = airfoil2.thickness(pc)


def setup():
    ul1.set_data([], [])
    cl1.set_data([], [])
    ll1.set_data([], [])
    ul2.set_data([], [])
    # cl2.set_data([], [])
    ll2.set_data([], [])
    return ul1, cl1, ll1, ul2, ll2


def update(frame):
    if frame <= N_steps:
        p = frame / N_steps  # Counting up
    else:
        p = 1 - (frame - N_steps) / N_steps  # Counting down
    print(f"\r{frame}", end="")


    # First method: interpolate the camber and thickness
    xyc = xyc1 + p * (xyc2 - xyc1)  # The interpolated camber curve
    cx, cy = xyc.T
    t = t1 + p * (t2 - t1)
    ul1.set_data(cx, cy + t / 2)
    cl1.set_data(cx, cy)
    ll1.set_data(cx, cy - t / 2)

    # Second method: interpolate points on the surfaces
    xyu = xyu1 + p * (xyu2 - xyu1)
    xyl = xyl1 + p * (xyl2 - xyl1)
    ul2.set_data(xyu.T[0], xyu.T[1])
    ll2.set_data(xyl.T[0], xyl.T[1])

    return (ul1, cl1, ll1, ul2, ll2)


# embed()

ani = animation.FuncAnimation(
    fig, update, frames=N_steps * 2, init_func=setup, blit=True, interval=20,
)

plt.show()
print()
# embed()
