from collections import ChainMap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

import pfh.glidersim as gsim
from pfh.glidersim.airfoil import NACA, AirfoilGeometryInterpolator
from pfh.glidersim.foil import SimpleFoil  # noqa: F401
from pfh.glidersim.foil_layout import EllipticalArc, EllipticalChord, FlatYZ, FoilLayout
from pfh.glidersim.foil_layout import PolynomialTorsion as PT  # noqa: F401
from pfh.glidersim.foil_sections import FoilSections


def sweep(vstart, vstop, T, fps, reverse=True):
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

    yield from values

    if reverse:
        yield from reversed(values)


def sweep_scalar(name, vstart, vstop, T, fps, reverse=True):
    """Generate a sequence of dictionaries like `{name: scalar}`."""
    for v in sweep(vstart, vstop, T, fps, reverse):
        yield {name: f"{v:<4.3g}"}


def repeat(config, T, fps):
    """Generator for copies of a config."""
    for _ in np.arange(round(T * fps)):
        yield config


# ---------------------------------------------------------------------------


def SEQS_sweep_chord_lengths(T, fps):
    """Sweep chord_length over a rectangular wing."""
    seq1 = [
        sweep_scalar("c", 0.3, 0.5, T / 2, fps, False),
    ]
    seq2 = [
        sweep_scalar("c", 0.5, 0.1, T, fps, False),
    ]
    seq3 = [
        sweep_scalar("c", 0.1, 0.3, T / 2, fps, False),
    ]
    return (seq1, T / 2), (seq2, T), (seq3, T / 2)


def SEQS_sweep_linear_chord_ratios(T, fps):
    def f1(vmin, vmax, tip, T, fps, reverse):
        for c0 in sweep(vmin, vmax, T, fps, reverse):
            m = c0 - tip
            yield {"c": f"lambda s: {c0:>4.3g} - {m:4.3g} * abs(s)"}

    def f2(vstart, vstop, c0, T, fps, reverse):
        for tip in sweep(vstart, vstop, T, fps, reverse):
            m = c0 - tip
            yield {"c": f"lambda s: {c0:>4.3g} - {m:4.3g} * abs(s)"}

    seq0 = [sweep_scalar("r_x", 0.5, 1, T / 2, fps, False)]
    seq1 = [
        repeat({"r_x": "1"}, T, fps),
        f1(0.3, 0.5, 0.3, T, fps, reverse=False),
    ]
    seq2 = [
        repeat({"r_x": "1"}, T, fps),
        f2(0.3, 0, 0.5, T, fps, reverse=False),
    ]

    return (seq0, T / 2), (seq1, T), (seq2, T)


def SEQS_sweep_elliptical_chord_ratios(T, fps):
    """Sweep the central chord while hold the tip constant."""

    def f(vmin, vmax, tip, T, fps, reverse):
        for c0 in sweep(vmin, vmax, T, fps, reverse):
            yield {"c": f"EllipticalChord({c0:4.3g}, {tip:4.3g})"}

    seq1 = [
        f(0.11, 0.7, 0.1, T, fps, False),
    ]
    seq2 = [
        f(0.7, 0.5, 0.1, T / 2, fps, False),
    ]

    return (seq1, T), (seq2, T / 2)


def SEQS_sweep_xrx_rectangle(T, fps):
    def f1(T, fps):
        for p in sweep(0, 0.5, T, fps, reverse=False):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - abs(s))"}

    def f2(T, fps):
        for p in sweep(1, 2, T, fps, reverse=False):
            yield {"x": f"lambda s: 0.5 * (1 - abs(s)**{p:#<.3f})"}

    def f3(T, fps):
        for p in sweep(0.5, 0, T, fps, reverse=False):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - s**2)"}

    seq1a = [
        repeat({"x": "0"}, T, fps),
        sweep_scalar("r_x", 0.5, 1, T, fps),
    ]
    seq1b = [
        repeat({"x": "0"}, T, fps),
        sweep_scalar("r_x", 0.5, 0, T, fps),
    ]

    seq2 = [
        repeat({"r_x": "0.5"}, T, fps),
        f1(T, fps),
    ]
    seq3 = [
        repeat({"r_x": "0.5"}, T, fps),
        f2(T, fps),
    ]
    seq4 = [
        repeat({"r_x": "0.5"}, T, fps),
        f3(T, fps),
    ]
    return (seq1a, T), (seq1b, T), (seq2, T), (seq3, T), (seq4, T)


def SEQS_sweep_xrx_triangle(T, fps):
    def f1(T, fps):
        for p in sweep(0, 0.5, T, fps, reverse=False):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - abs(s))"}

    def f2(T, fps):
        for p in sweep(1, 2, T, fps, reverse=False):
            yield {"x": f"lambda s: 0.5 * (1 - abs(s)**{p:<#.3f})"}

    def f3(T, fps):
        for p in sweep(0.5, 0, T, fps, reverse=False):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - s**2)"}

    base = {"c": "lambda s: 0.5 * (1 - abs(s))"}

    seq1a = [
        repeat({**base, "x": "0"}, T * 0.75, fps),
        sweep_scalar("r_x", 1, 0, T * 0.75, fps, False),
    ]
    seq1b = [
        repeat({**base, "x": "0"}, T * 0.75, fps),
        sweep_scalar("r_x", 0, 0.5, T * 0.75, fps, False),
    ]
    seq2 = [
        repeat({**base, "r_x": "0.5"}, T, fps),
        f1(T, fps),
    ]
    seq3 = [
        repeat({**base, "r_x": "0.5"}, T, fps),
        f2(T, fps),
    ]
    seq4 = [
        repeat({**base, "r_x": "0.5"}, T, fps),
        f3(T, fps),
    ]
    return (seq1a, T * 0.75), (seq1b, T * 0.75), (seq2, T), (seq3, T), (seq4, T)


def SEQS_sweep_xrx_elliptical(T, fps):
    def f1(T, fps):
        for p in sweep(0, 0.5, T, fps, reverse=False):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - abs(s))"}

    def f2(T, fps):
        for p in sweep(1, 2, T, fps, reverse=False):
            yield {"x": f"lambda s: 0.5 * (1 - abs(s)**{p:<#.3f})"}

    def f3(T, fps):
        for p in sweep(0.5, 0, T, fps, reverse=False):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - s**2)"}

    base = {"c": "EllipticalChord(root=0.5, tip=0.1)"}
    seq1a = [
        repeat({**base, "x": "0"}, T, fps),
        sweep_scalar("r_x", 0.5, 1, T, fps, True),
    ]
    seq1b = [
        repeat({**base, "x": "0"}, T, fps),
        sweep_scalar("r_x", 0.5, 0, T, fps, True),
    ]

    seq2 = [
        repeat(base, T, fps),
        f1(T, fps),
    ]
    seq3 = [
        repeat(base, T, fps),
        f2(T, fps),
    ]
    seq4 = [
        repeat(base, T, fps),
        f3(T, fps),
    ]
    return (seq1a, T), (seq1b, T), (seq2, T), (seq3, T), (seq4, T)


def SEQS_sweep_elliptical_anhedral(T, fps):
    def f1(vstart, vstop, T, fps, reverse=True):
        for ma in sweep(vstart, vstop, T, fps, reverse):
            yield {"yz": f"EllipticalArc({ma:<#5.2f})"}

    def f2(vstart, vstop, T, fps, reverse=True):
        ma = vstart
        for mar in sweep(vstart * 2 + 1, vstop, T, fps, reverse):
            yield {"yz": f"EllipticalArc({ma:<#5.2f}, {mar:<#5.2f})"}

    base = {"c": "EllipticalChord(root=0.5, tip=0.1)"}
    seq1a = [
        repeat(base, T, fps),
        f1(1, 44, T, fps, False),
    ]
    seq1b = [
        repeat(base, T / 2, fps),
        f1(44, 30, T / 2, fps, False),
    ]
    seq2 = [
        repeat(base, T, fps),
        f2(30, 85, T, fps, False),
    ]

    return (seq1a, T), (seq1b, T / 2), (seq2, T)


def SEQS_sweep_xrx_elliptical_arched(T, fps):
    def f(start, stop, T, fps, reverse=True):
        for p in sweep(start, stop, T, fps, reverse):
            yield {"x": f"lambda s: {p:>#.3f} * (1 - (s**2))"}

    base = {
        "c": "EllipticalChord(root=0.5, tip=0.1)",
        "yz": "EllipticalArc(30, 85)",
    }

    seq1a = [
        repeat({**base, "x": "0"}, T, fps),
        sweep_scalar("r_x", 0.5, 0, T, fps),
    ]
    seq1b = [
        repeat({**base, "x": "0"}, T, fps),
        sweep_scalar("r_x", 0.5, 1, T, fps),
    ]
    seq2a = [
        repeat(base, T, fps),
        f(0, -0.2, T, fps),
    ]
    seq2b = [
        repeat(base, T, fps),
        f(0, 0.2, T, fps),
    ]

    return (seq1a, T), (seq1b, T), (seq2a, T), (seq2b, T)


def SEQS_sweep_torsion(T, fps):
    def f1(start, peak_start, peak_stop, exponent, T, fps):
        for peak in sweep(peak_start, peak_stop, T, fps=fps, reverse=False):
            yield {
                "theta": f"PT(start={start}, peak={peak:<#6.3f}, exponent={exponent})"
            }

    def f2(s_start, s_stop, peak, exponent, T, fps):
        for start in sweep(s_start, s_stop, T, fps=fps):
            yield {
                "theta": f"PT(start={start:<#6.3f}, peak={peak}, exponent={exponent})"
            }

    seq1 = [f1(0, 0, 25, 2, T, fps)]
    seq2 = [f2(0, 0.8, 25, 2, T, fps)]
    seq3a = [
        repeat({"theta": "PT(start=0, peak=25.0, exponent=2)"}, T / 2, fps),
        sweep_scalar("r_x", 0.5, 1, T / 2, fps, True),
    ]
    seq3b = [
        repeat({"theta": "PT(start=0, peak=25.0, exponent=2)"}, T / 2, fps),
        sweep_scalar("r_x", 0.5, 0, T / 2, fps, True),
    ]
    seq4 = [f1(0, 25, 0, 2, T, fps)]

    return (seq1, T), (seq2, T), (seq3a, T / 2), (seq3b, T / 2), (seq4, T)


# ---------------------------------------------------------------------------


def setup_figure(dpi):
    fig = plt.figure(figsize=(16, 9), dpi=dpi, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0, hspace=0)
    # gs.tight_layout(fig, pad=100, h_pad=0, w_pad=0)  # FIXME: useless?

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

    return fig, axes


def foil_generator(base_config, sequences, fps=60):
    n = 1  # Current frame
    N = sum(round(s[1] * fps) for s in sequences)  # Total number of frames
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
            layout = FoilLayout(**params)
            profiles = AirfoilGeometryInterpolator({0: NACA(24018)})
            yield config, gsim.foil.SimpleFoil(
                layout=layout,
                sections=FoilSections(profiles=profiles),
                b_flat=10,
            )
            n += 1
    print()


def update(config_and_foil, axes):
    """Update the plots."""
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
import pfh.glidersim as gsim
from pfh.glidersim.airfoil import AirfoilGeometryInterpolator, NACA
from pfh.glidersim.foil import (
    SimpleFoil,
)
from pfh.glidersim.foil_layout import (
    EllipticalArc,
    EllipticalChord,
    FlatYZ,
    PolynomialTorsion as PT,
    FoilLayout,
)
from pfh.glidersim.foil_sections import FoilSections

layout = FoilLayout(
  """

    maxlen = max(len(k) for k in config.keys())  # For aligning the "="
    code += "\n  ".join(f"{k:>{maxlen}} = {v}" for k, v in config.items())
    code += """
)

profiles = AirfoilGeometryInterpolator({0: NACA(24018)})
foil = gsim.foil.SimpleFoil(
    sections=FoilSections(profiles),
    layout=layout,
    b_flat=10,
)

pfh.glidersim.extras.plots.plot_foil(foil)
"""
    code_text = axes[0].text(
        0,
        0.5,
        code,
        fontsize=8,
        family="monospace",
        transform=axes[0].transAxes,
        verticalalignment="center",
    )
    foil_artists = gsim.extras.plots.plot_foil(foil, 51, ax=axes[1])

    return (code_text, *foil_artists)


if __name__ == "__main__":
    outfile = "/home/peter/animated.mp4"
    outfile = ""  # Disable movie output (use a live plot)

    # Use high quality output for movies
    dpi = 200 if outfile else 150
    fps = 60 if outfile else 10

    # Each sequence modifies this baseline configuration
    base_config = {
        "r_x": "0.5",
        "x": "0",
        "r_yz": "1.00",
        "yz": "FlatYZ()",
        "c": "0.3",
        "theta": "0",
        "center": "False",
    }

    sequences = [
        *SEQS_sweep_chord_lengths(3, fps),
        *SEQS_sweep_torsion(3, fps),
        *SEQS_sweep_xrx_rectangle(3, fps),
        *SEQS_sweep_linear_chord_ratios(3, fps),
        *SEQS_sweep_xrx_triangle(3, fps),
        *SEQS_sweep_elliptical_chord_ratios(3, fps),
        *SEQS_sweep_xrx_elliptical(3, fps),
        *SEQS_sweep_elliptical_anhedral(3, fps),
        *SEQS_sweep_xrx_elliptical_arched(3, fps),
    ]

    frames = foil_generator(base_config, sequences, fps)

    fig, axes = setup_figure(dpi)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,  # Each frame is a (config, SimpleFoil)
        fargs=(axes,),
        repeat=False,
        interval=20,
        save_count=999999,
    )

    if outfile:
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
            outfile,
            dpi=dpi,
            writer=writer,
        )
    else:
        plt.show()
