from collections import ChainMap

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

import numpy as np
from numpy import sin, cos, sqrt, arcsin, arctan  # noqa: F401

from IPython import embed  # noqa: F401

import pfh.glidersim as gsim
from pfh.glidersim.airfoil import Airfoil, NACA
from pfh.glidersim.foil import (
    elliptical_lobe,
    elliptical_chord,
    FlatYZ,
    PolynomialTorsion,
    SimpleIntakes,
)


def sweep(vstart, vstop, T, reverse=True, fps=60):
    """
    Generate a sequence of scalars.

    name : string
        The parameter to sweep across
    vstart, vstop : float
        The starting and stopping values of the sweep
    T : float [seconds]
        Total duration of the sweep. If `reverse = True`, then the forward and
        reverse sequences take `T / 2` seconds.
    fps : float [Hz]
        Frames per second
    """
    if reverse:
        T /= 2

    N = round(T * fps)

    if np.isclose(vstart, vstop):  # Hack to enable constant values
        values = np.full(N, vstart)
    else:
        values = np.linspace(vstart, vstop, N)  # Linear sweep

    for n in range(N):
        yield values[n]

    if reverse:
        for n in range(N):
            yield values[::-1][n]


def sweep_scalar(name, vstart, vstop, T, reverse=True, fps=60):
    """Generate a sequence of dictionaries like `{name: scalar}`."""
    for v in sweep(vstart, vstop, T, reverse, fps):
        yield {name: f"{v:<4.3g}"}


# ---------------------------------------------------------------------------


def sweep_elliptical_lobe1(vstart, vstop, T, reverse=True, fps=60):
    for am in sweep(vstart, vstop, T, reverse, fps):
        yield {"yz": f"elliptical_lobe(mean_anhedral={am:<4.3g})"}


def sweep_elliptical_lobe2(vstart, vstop, T, reverse=True, fps=60):
    for am in sweep(vstart, vstop, T, reverse, fps):
        yield {"yz": f"elliptical_lobe(mean_anhedral={am:<4.3g}, max_anhedral_rate=89)"}


def sweep_torsion1(s_start, s_stop, peak, exponent, T, fps):
    for start in sweep(s_start, s_stop, T, fps=fps):
        yield {
            "torsion": f"PolynomialTorsion(start={start:<4.3g}, peak={peak:<4.3g}, exponent={exponent:<4.3g})"
        }


def sweep_torsion2(start, peak_start, peak_stop, exponent, T, fps):
    for peak in sweep(peak_start, peak_stop, T, fps=fps):
        yield {
            "torsion": f"PolynomialTorsion(start={start:<4.3g}, peak={peak:<4.3g}, exponent={exponent:<4.3g})"
        }


# ---------------------------------------------------------------------------


def foil_generator(fps=60):
    base_config = {
        "airfoil": "Airfoil(None, NACA(24018))",
        "b_flat": "11",
        "chord_length": "elliptical_chord(root=0.5, tip=0.1)",
        "r_x": "0.75",
        "x": "0",
        "r_yz": "1.00",
        "yz": "elliptical_lobe(34, 75)",
        "torsion": "PolynomialTorsion(start=0.8, peak=4, exponent=2)",
        "intakes": "SimpleIntakes(s_end=0.85, s_upper=-0.04, s_lower=-0.09)",
    }

    # -----------------------------------------------------------------------

    T1 = 3
    seq1 = [
        sweep_scalar("r_x", 0, 1, T1, fps=fps),
    ]

    T2 = 3
    seq2 = [
        sweep_elliptical_lobe1(1e-99, 44, T2, fps=fps),
    ]

    T3 = 3
    seq3 = [
        sweep_elliptical_lobe2(1e-99, 44, T3, fps=fps),
    ]

    T4 = 3
    seq4 = [
        sweep_scalar("r_x", 1, 0, T4, fps=fps),
        sweep_elliptical_lobe1(1, 44, T4, fps=fps),
    ]

    T5 = 3
    seq5 = [
        sweep_elliptical_lobe1(1e-99, 1e-99, T5, fps=fps),
        sweep_torsion1(0, 0.8, 10, 2, T5, fps=fps),
    ]

    T6 = 3
    seq6 = [
        # sweep_elliptical_lobe1(1e-99, 1e-99, T6, fps=fps),
        [{"yz": "FlatYZ()"}] * T6 * fps,
        [{"r_x": "1"}] * T6 * fps,
        sweep_torsion2(0.25, 0, 25, 2, T6, fps=fps),
    ]

    # ----------------------------------------------------------------------

    T7 = 6

    def f7b(T, fps):
        for p in sweep(0, 1, T, fps=fps):
            yield {"r_x": f"lambda s: {p:<5.3g} * (1 - s**2)"}

    def f7c(T, fps):
        for p in sweep(0, 1, T, fps=fps):
            yield {"r_x": f"lambda s: {p:<5.3g} * s**2"}

    seq7a = [
        [{"yz": "FlatYZ()"}] * T7 * fps,
        sweep_scalar("r_x", 0, 1, T7, fps=fps),
    ]
    seq7b = [
        [{"yz": "FlatYZ()"}] * T7 * fps,
        f7b(T7, fps),
    ]
    seq7c = [
        [{"yz": "FlatYZ()"}] * T7 * fps,
        f7c(T7, fps),
    ]
    seq7d = [
        [{"yz": "FlatYZ()"}] * T7 * fps,
        f7b(T7, fps),
        sweep_torsion2(0, 0, 25, 2, T7, fps=fps),
    ]
    seq7e = [
        [{"yz": "FlatYZ()"}] * T7 * fps,
        f7c(T7, fps),
        sweep_torsion2(0, 0, 25, 2, T7, fps=fps),
    ]
    seq7f = [
        f7b(T7 * 2, fps),
        sweep_torsion2(0, 0, 45, 1, T7 * 2, fps=fps),
        sweep_elliptical_lobe1(1, 44, T7 * 2, fps=fps),
    ]

    # ----------------------------------------------------------------------

    def f8(T, fps):
        for p in sweep(25, -25, T, fps=fps):
            yield {"torsion": f"PolynomialTorsion(start=0, peak={p:<4.3g}, exponent=2)"}

    T8 = 6
    seq8 = [
        [{"yz": "FlatYZ()"}] * T8 * fps,
        sweep_scalar("r_x", 0, 1, T8, fps=fps),
        # [{"torsion": f"PolynomialTorsion(start=0, peak=25, exponent=2)"}] * T8 * fps,
        f8(T8, fps),
    ]

    # ----------------------------------------------------------------------

    sequences = [
        # [seq1, T1],
        # [seq2, T2],
        # [seq3, T3],
        # [seq4, T4],
        # [seq5, T5],
        # [seq6, T6],
        # [seq7a, T7],
        # [seq7b, T7],
        # [seq7c, T7],
        # [seq7d, T7],
        [seq7e, T7],
        [seq7f, T7 * 2],
        # [seq8, T8],
    ]

    # -----------------------------------------------------------------------

    n = 0  # Current frame
    N = sum(s[1] for s in sequences) * fps  # Total number of frames
    for seq, _T in sequences:  # Each sequence is a set of generators
        for config_set in zip(*seq):  # For each generated set of configurations
            modifications = dict(
                ChainMap(*config_set)
            )  # Combine the set into a single dict
            print(f"\rYielding frame: {n} / {N}", end="")
            # print("DEBUG> modifications:", modifications)
            config = base_config.copy()  # Baseline configuration strings
            config.update(modifications)  # Modified configuration strings
            params = {}  # The `exec`uted result
            for k, v in config.items():
                exec(f"params['{k}'] = {v}")
            yield config, gsim.foil.FoilGeometry(**params)
            n += 1
    print()


def update(config_and_foil, axes):
    """Update the plot."""
    # print(f"Entering `update`, frames={frames}")

    config, foil = config_and_foil

    # First, clear any old artists:
    while axes[0].texts:
        a = axes[0].texts[0]
        a.remove()
    while axes[1].lines:
        a = axes[1].lines[0]
        a.remove()
    while axes[1].collections:
        a = axes[1].collections[0]
        a.remove()

    code = """\
from pfh.glidersim.airfoil import Airfoil, NACA
from pfh.glidersim.foil import (
    elliptical_lobe,
    elliptical_chord,
    PolynomialTorsion,
    SimpleIntakes,
)

foil = pfh.glidersim.foil.FoilGeometry(
  """

    maxlen = max(len(k) for k in config.keys())  # For aligning the "="
    code += "\n  ".join(f"{k:>{maxlen}} = {v}" for k, v in config.items())
    code += "\n)\n\npfh.glidersim.plots.plot_parafoil_geo(foil)"
    code_text = axes[0].text(
        0,
        0.5,
        code,
        fontsize=8,
        family="monospace",
        transform=axes[0].transAxes,
        verticalalignment="center",
    )
    foil_artists = gsim.plots.plot_parafoil_geo(
        foil, N_sections=51, ax=axes[1], surface_lw=0.25,
    )

    return (code_text, *foil_artists)


if __name__ == "__main__":
    save = False
    # save = True

    dpi = 400 if save else 100
    fps = 60 if save else 10
    fig = plt.figure(figsize=(16, 9), dpi=dpi, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0, hspace=0)
    gs.tight_layout(fig, pad=1, h_pad=0, w_pad=0)

    # Setup the text region
    ax1 = fig.add_subplot(gs[0])
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.xaxis.set_ticks([])
    ax1.xaxis.set_ticklabels("")
    ax1.yaxis.set_ticks([])
    ax1.yaxis.set_ticklabels("")

    # Setup the 3D wing region
    ax2 = fig.add_subplot(gs[1], projection="3d")
    ax2.invert_yaxis()
    ax2.invert_zaxis()
    ax2.view_init(azim=-120, elev=20)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")

    axes = [ax1, ax2]
    fig.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=foil_generator(fps=fps),  # Produces (config, FoilGeometry)
        fargs=(axes,),
        repeat=False,
        interval=20,
        save_count=999999,
    )

    if save:
        # progress_callback = lambda i, n: print(f"\rSaving frame {i}", end="")  # noqa: E731
        progress_callback = None

        writer = animation.FFMpegWriter(
            fps=fps,
            # bitrate=10_000,
            extra_args=[
                "-vf",
                "scale='-1:min(1440,iw)'",  # Scale to 1440px horizontally
                "-sws_flags",
                "lanczos",
            ],
        )
        ani.save(
            "/home/peter/animated.mp4",
            progress_callback=progress_callback,
            dpi=dpi,
            writer=writer,
        )
    else:
        plt.show()

    print("\nDone.")

    # embed()
