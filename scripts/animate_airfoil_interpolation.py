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

fig, ax = plt.subplots()
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.4, 0.4)
# ax.set_aspect("equal")
# ax.margins(x=0.1, y=0.40)
ul = plt.plot([], [], c="b", lw=0.75)[0]
cl = plt.plot([], [], c="k", lw=0.75, linestyle="--")[0]
ll = plt.plot([], [], c="r", lw=0.75)[0]
ax.grid(True)
N_points = 300
N_steps = 60 * 3

pc = (1 - np.cos(np.linspace(0, np.pi, N_points))) / 2  # `0 <= x <= 1`
c1 = airfoil1.camber_curve(pc)
c2 = airfoil2.camber_curve(pc)
t1 = airfoil1.thickness(pc)
t2 = airfoil2.thickness(pc)


def setup():
    ul.set_data([], [])
    cl.set_data([], [])
    ll.set_data([], [])
    return ul, cl, ll


def update(frame):
    print(f"\r{frame}", end="")
    if frame <= N_steps:
        p = frame / N_steps  # Counting up
    else:
        p = 1 - (frame - N_steps) / N_steps  # Counting down

    cc = c1 + p * (c2 - c1)
    cx, cy = cc.T
    t = t1 + p * (t2 - t1)

    ul.set_data(cx, cy + t / 2)
    cl.set_data(cx, cy)
    ll.set_data(cx, cy - t / 2)

    ax.set_aspect("equal")

    return (ul, cl, ll)


ani = animation.FuncAnimation(
    fig, update, frames=N_steps * 2, init_func=setup, blit=True, interval=20,
)

plt.show()
print()
# embed()
