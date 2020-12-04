from IPython import embed

import matplotlib.pyplot as plt

import numpy as np

import pfh.glidersim as gsim

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


class SimpleRibs:
    """
    So, in summary:

    You start with a continuous curve over `s`. You define a set of "ribs",
    which are just coordinates in `s`. You then sample the curve at the ribs to
    get their xyz coordinates and assume linear sections in between the ribs to
    get a new intermediate curve. (We still need to "compress" the sections.
    Each section gets distorted ("ovalized"), which squishes it along it's
    local spanwise axis.

    You can squish each section using whatever squishing method you've chosen.
    (In the above code, I've assumed a fixed percent compression for all
    sections.) Compress each section, keeping them end to end, then define a
    new parametric coordinate `sbar`. You now have a piecewise linear curve
    using those new coordinates. This part is easy since each endpoints remain
    the same (the ribs don't let the surfaces "slip" over each other, so
    they're always the same relative distance from the wing root).

    Now you need a way to map the original surface coordinates `s` onto those
    distorted panels. Use the non-linear transformation to "squish" the `s` in
    each panel onto their `sbar` equivalents.
    """
    def __init__(self, num_ribs, p=0.95):
        # FIXME: accept predefined `ribs` as an input
        self.num_ribs = num_ribs
        self.num_panels = num_ribs - 1
        self.ribs = np.linspace(-1, 1, num_ribs)
        self.p = np.full(self.num_panels, p)

        # Panel midpoints and ribs have the same coordinates in `s` and `sbar`
        self.s_mid = (self.ribs[1:] + self.ribs[:-1]) / 2  # Panel midpoints

        # FIXME: allow non-uniform panel widths. For now, assume uniformity.
        #        I'm using arrays in preparation for non-uniformity.
        self.L = self.ribs[1:] - self.ribs[:-1]  # Undeformed Panel widths
        self.Lbar = self.p * self.L
        self.smid = (self.ribs[1:] + self.ribs[:-1]) / 2
        self.alphas = np.full(self.num_panels, self._compute_alpha(p))
        self.R = self.L / self.alphas

        # Interesting note to self:
        # (ribs[n+1] - ribs[n]) / (s_deltas[-1] - s_deltas[0]) == 1 / p

        # Hack: linear interpolator for computing panel indices
        self._ic = interp1d(self.ribs, np.arange(self.num_ribs))


    def _compute_alpha(self, p):
        """
        Compute the angle domain

        Parameters
        ----------
        p : float
            The compression ratio such that `Lbar = p * L`

        Returns
        -------
        alpha
        """
        # p : compression ratio (eg, 0.95 means L shrunk by 5%)
        def target(R, p):
            # Optimization target for computing `alpha`
            return p - 2 * R * np.sin(1 / (2 * R))

        res = root_scalar(target, args=(p,), x0=1, x1=0.5)
        R = res.root
        # I *think* that for the purposes of computing alpha, I'm just assuming
        # `L=1`, so `R` in this case has nothing to do with a specific panel.
        alpha = 1 / R
        return alpha

    def panel_index(self, s):
        """
        Compute panel indices from section indices.
        """
        # Let the last rib section index "belong" to the last panel
        return self._ic(s).astype(int).clip(0, self.num_panels - 1)

    def s2sbar(self, s):
        """
        The deformed curve also uses section indices, but they are not the same
        indices as defined by the SimpleFoil. The SimpleFoil section indices
        `s` must be mapped onto the deformed section indices `sbar`.

        The ribs and mid-points of the panels stay the same, but the section
        indices between those places required a non-linear mapping that
        squishes points near the ribs towards the ribs.
        """
        ix = self.panel_index(s)
        s_delta = s - self.smid[ix]
        alpha_s = s_delta / self.R[ix]
        sbar_delta = self.R[ix] * np.sin(alpha_s) / self.p[ix]
        sbar = self.smid[ix] + sbar_delta
        return sbar

    def sbar2s(self, sbar):
        ix = self.panel_index(sbar)
        sbar_delta = sbar - self.smid[ix]
        alpha_s = np.arcsin((sbar_delta * self.p[ix]) / self.R[ix])
        s_delta = alpha_s * self.R[ix]
        s = self.smid[ix] + s_delta
        return s

    def h(self, s):
        # FIXME: needs `alpha_s`?
        ix = self.panel_index(s)
        # h = R * (np.cos(alphas) - np.cos(self.alpha[ix] / 2)) / (self.p[ix] * self.L[ix])


# sr = SimpleRibs(31)
sr = SimpleRibs(10, p=0.67)
s = np.linspace(sr.ribs[5], sr.ribs[6], 50)
sbar = sr.s2sbar(s)
plt.plot(
    np.c_[s, sr.s2sbar(s)].T,
    np.c_[np.ones(len(s)), np.zeros(len(s))].T,
    'k',
    lw=0.75
)
plt.show()
embed()
1 / 0

# --------------------------------------------------------------------------


def compute_alpha(p, L):
    # p : compression ratio (eg, 0.95 means L shrunk by 5%)
    # L : the starting length

    def target(R, p, L):
        return p * L - 2 * R * np.sin(L / (2 * R))

    res = root_scalar(target, args=(p, L), x0=1, x1=0.5)
    R = res.root
    alpha = L / R
    return alpha


# The deformed curve also uses section indices, but the undeformed section
# indices need to be mapped onto them. The ribs and mid-points of the panels
# stay the same, but the section indices between those places required a
# non-linear mapping that squishes points near the ribs towards the ribs.


N = 51
ribs = np.cos(np.linspace(np.pi, 0, N))
ic = interp1d(ribs, np.arange(N))  # "index curve"
s = np.linspace(-1, 1, 501)
indices = ic(s).astype(int)  # The panel indices for those `s`


for n in range(10, 12):
    p = 0.95
    L = ribs[n+1] - ribs[n]

    # FIXME: if you use the true `L` the numbers can get very small and produce
    #        numerical issues. I think you can just use `L = 1` to compute the
    #        `alpha_s` then scale after the fact.
    _L = 1
    alpha = compute_alpha(p, L=1)  # Compute `alpha` assuming `L = 1`
    R = L / alpha  # Now scale the radius using the true panel width
    smid = (ribs[n+1] + ribs[n]) / 2
    K = 51
    s = np.linspace(ribs[n], ribs[n+1], K)
    alphas = np.linspace(-alpha/2, alpha/2, K)

    # If you layed the coordinate out in `s`, this is how they would move in
    # that original `s` coordinate system.
    s_deltas = R * np.sin(alphas)

    # The `s` get "squished" (including the endpoints), so to have `sbar` cover
    # the same "length" (relative to the sbar-curve), we have to stretch them
    # back out. This is the "squishing factor" between the length of the
    # original and squished coordinates (still in `s`, not `sbar`).
    #
    # Confirmed: `length_ratio = 1 / p`
    length_ratio = (ribs[n+1] - ribs[n]) / (s_deltas[-1] - s_deltas[0])


    # FIXME: Wait! Aren't the length ratios just `p`?


    # The endpoints remain unchanged by definition. This doesn't mean the
    # sections aren't changing their `s` coordinates (their position along the
    # *original* parametric curve), but because after moving them we then
    # stretch the transformed points to normalize their span to [-1, +1].
    # These are those values in a *new coordinate system*, which here I'm
    # calling sbar.
    #
    # The midpoints remain unchanged since the warping is symmetric on their
    # left and right, so they also stay the same.
    sbar_deltas = s_deltas * length_ratio
    sbar = smid + sbar_deltas

    for k in range(K):
        plt.plot([s[k], sbar[k]], [1, 0], 'k', lw=0.5)

    # The changes in section heights due to ovalization
    #
    # FIXME: are the `h` are wrt to the uscaled (compressed) L? The unscaled
    #        `s` are the "physical" units, or sorts, so I think those are the
    #        ones that would maintain the surface area correctly.
    h = R * (np.cos(alphas) - np.cos(alpha / 2)) / (p * L)

plt.show()


# Now the point of contention: how to embiggen those airfoils. My current
# thought is that the heights are proportional to the section width, and
# should enlarge the wing thickness perpendicular to the chord. Easy way to do
# this is to compute the thickness above/below the camber line (the midpoint)
# and multiply all those thicknesses by `h`. I was originally thinking I'd just
# multiply the airfoil `y` coordinates, but that is conceptually flawed: what
# if the bottom surface of the airfoil lies on the chord? You wouldn't expand
# the bottom surface at all, which doesn't make sense. No, I think it should be
# relative to the thickness as measured by the camber line.
airfoil = gsim.airfoil.NACA(23015)
r = np.linspace(0, 1, 2000)
points_upper = airfoil.surface_curve(r)
points_lower = airfoil.surface_curve(-r)
points_camber = (points_upper + points_lower) / 2
thicknesses = points_upper.T[1] - points_lower.T[1]
# HP = h / magical_thickest_point  # Thickness change as a percentage
HP = 1.15
plt.plot(points_upper.T[0], points_upper.T[1], 'k', lw=0.75)
plt.plot(points_lower.T[0], points_lower.T[1], 'k', lw=0.75)
plt.plot(points_camber.T[0], points_camber.T[1] + HP * thicknesses/2, 'r--', lw=0.75)
plt.plot(points_camber.T[0], points_camber.T[1] - HP * thicknesses/2, 'r--', lw=0.75)
plt.gca().set_aspect("equal")
plt.show()

#
# This "thicken normal to the chord" probably only works if the airfoil has
# been normalized?
#
