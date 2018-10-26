"""
Recreates the paraglider analysis in "Wind Tunnel Investigation of a Rigid
Paraglider Reference Wing", H. Belloc, 2015


Some Questions:
    1. Should `V` be greater than or less than `V_rel`?
    2. Why am I still significantly overestimating CL?
        * Interesting fact: if I divide by `rho_air` then it matches closely...
    3. In Phillips, why are J2 and J4 so tiny? Is that correct?
    4. In Phillips, where did J4 come from in Hunsaker's derivation?
        * It wasn't in the original Phillips derivation
    5. How should I modify Phillips to handle non-uniform `u_inf`?
        * eg, during a turn
        * Need to modify the `v_ij` at least
    6. Why don't my `CM` curve like the Belloc paper?
        * For starters, his `Cm25%` are positive? How? The airfoil produces a
          negative Cm, as does my model
        * The shape is wrong too
        * Fig:6 shows it positve, decrease, then increase, then drop negative
          post-stall; mine just gradually slopes down until negative!
    7. How many segments are needed for an accurate Phillips method?
        * Using K=31 gives noisy results for beta=15

Some TODOs:
 * Review the force and moment calculations, given V+Gamma
 * Cleanup the Phillips code
 * Refactor the Phillips method
    * At the very least, put the "initial Gamma proposal" into a function?
 * Double check the wing geometry
    * Am I producing the correct linearly variation?
    * Am I braking anything by overwriting `planform.fc` and `planform.ftheta`?

 * I need to add Picard iterations to Phillips to deal with stalled sections
 * I need to fix how I calculate L and D when beta>0
    * Observe in my `CL vs CD`: CD decreases with beta? Nope, wrong
    * Also, in `CL vs alpha`, why does alpha_L0 increase with beta?

"""


import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from numpy import sin, cos
from IPython import embed
import pandas as pd

from scipy.interpolate import UnivariateSpline, PchipInterpolator
from numpy.polynomial import Polynomial

import Airfoil
import BrakeGeometry
import Parafoil
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider
import plots  # noqa: F401


class FlaplessAirfoilCoefficients(Airfoil.AirfoilCoefficients):
    """
    Uses the airfoil coefficients from a CSV file.
    The CSV must contain the following columns: [alpha, delta, CL, CD, Cm]

    This is similar to `Airfoil.GridCoefficients`, but it assumes that delta
    is always zero. This is convenient, since no assuptions need to be made
    for the non-existent flaps on the wind tunnel model.
    """

    def __init__(self, filename, convert_degrees=True):
        data = pd.read_csv(filename)
        self.data = data

        if convert_degrees:
            data['alpha'] = np.deg2rad(data.alpha)

        self._Cl = UnivariateSpline(data[['alpha']], data.CL, s=0.001)
        self._Cd = UnivariateSpline(data[['alpha']], data.CD, s=0.0001)
        self._Cm = UnivariateSpline(data[['alpha']], data.Cm, s=0.0001)
        self._Cl_alpha = self._Cl.derivative()

    def _clean(self, alpha, val):
        # The UnivariateSpline doesn't fill `nan` outside the boundaries
        min_alpha, max_alpha = np.deg2rad(-9.9), np.deg2rad(24.9)
        mask = (alpha < min_alpha) | (alpha > max_alpha)
        val[mask] = np.nan
        return val

    def Cl(self, alpha, delta):
        return self._clean(alpha, self._Cl(alpha))

    def Cd(self, alpha, delta):
        return self._clean(alpha, self._Cd(alpha))

    def Cm(self, alpha, delta):
        return self._clean(alpha, self._Cm(alpha))

    def Cl_alpha(self, alpha, delta):
        return self._clean(alpha, self._Cl_alpha(alpha))


# ---------------------------------------------------------------------------
# First, record the data from the paper

# Table 1: the Full-scale wing dimensions
h = 3 / 8  # Arch height (vertical deflection from wing root to tips) [m]
cc = 2.8 / 8  # The central chord [m]
b = 11.00 / 8  # The projected span [m]
S = 25.08 / (8**2)  # The projected area [m^2]
AR = 4.82  # The projected aspect ratio

b_flat = 13.64 / 8  # The flattened span [m]
S_flat = 28.56 / (8**2)  # The flattened area [m^2]
AR_flat = 6.52  # The flattened aspect ratio

# Convert the positive scalar `k` into a normal taper ratio
k = 1.05  # From Eq:2
taper = np.sqrt(1 - 1/k**2)  # Set `fc = c_i` and solve for lambda
# Alternatively, taper = .107/.350, using the 1/8 model coordinates directly


# From Table 2; these are the coordinates 0.6c line of the 1/8 size model
xyz = np.array([
    [0.000, -0.688,  0.000],
    [0.000, -0.664, -0.097],
    [0.000, -0.595, -0.188],
    [0.000, -0.486, -0.265],
    [0.000, -0.344, -0.325],
    [0.000, -0.178, -0.362],
    [0.000,  0.000, -0.375],
    [0.000,  0.178, -0.362],
    [0.000,  0.344, -0.325],
    [0.000,  0.486, -0.265],
    [0.000,  0.595, -0.188],
    [0.000,  0.664, -0.097],
    [0.000,  0.688,  0.000]
    ])

c = np.array([0.107, 0.137, 0.198, 0.259, 0.308, 0.339, 0.350,
              0.339, 0.308, 0.259, 0.198, 0.137, 0.107])

# The "x = 0" refers to the fact that the data assumes the 0.6c point of each
# section lines in the `x = 0` plane. To convert this data into c/4 coordinates
# we have to shift the x-coordinates of each section.
c0, c4 = xyz.copy(), xyz.copy()
c0[:, 0] += 0.6 * c  # The leading edge
c4[:, 0] += (0.6 - .25) * c  # The quarter-chord line


# FIXME: this is wrong! The 0.6c points do lie on the `x = 0` plane, but the
# outter sections are twisted, which means the c0 and c4 points need their
# y and z components adjusted along those twisted coord lines as well, not just
# their x-components


# fig, ax = plt.subplots()
# ax.plot(c0.T[1], c0.T[0])
# ax.plot(c0.T[1], c0.T[0] - c)
# ax.plot(c4.T[1], c4.T[0], 'g--')
# ax.set_aspect('equal')
# ax.grid(which='both')
# plt.show()


# What is the area of each panel, treating them as trapezoids?
#  * Note: this does not account for the 3degree washout at the tips
dA = (c[:-1] + c[1:])/2 * (xyz.T[1, 1:] - xyz.T[1, :-1])
print(f"Projected area> Expected: {S}, Actual: {dA.sum()}")
# I suspect the area is wrong because I'm not accounting for the twist

# What is the quarter-chord sweep?
v = c4[12] - c4[6]  # Vector from the central LE to the tip LE
sweepMed = -np.rad2deg(np.arctan2(v[0], v[1]))
# sweepMed -= 1.35
sweepMed -= .5
print(f"sweepMed: {sweepMed} [degrees]")

# You could do the same for sweepMax, by solving for fx(y) as defined in PFD,
# but that's a pain. Easier to find the root where the flat span matches the
# specs.


# ---------------------------------------------------------------------------
# Second, build an approximate version using a ParafoilGeometry

airfoil_geo = Airfoil.NACA5(23015, convention='british')
airfoil_coefs = FlaplessAirfoilCoefficients(
    'polars/NACA 23015_T1_Re0.920_M0.03_N7.0_XtrTop 5%_XtrBot 5%.csv')

airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)

# embed()
sweepMax = 2*sweepMed + 1e-20
# sweepMax += 11.75
print("DEBUG> sweepMax:", sweepMax)
planform = Parafoil.EllipticalPlanform(
    b_flat, cc, taper, sweepMed, sweepMax,
    torsion_exponent=1, torsion_max=0)


# Overwrite the default chord distribution; the Belloc model doesn't fit the
# elliptical parametrization easily (the taper doesn't develop correctly).
# Uses a smoothing spline to avoid sharp edges in the lifting line.
dl = c4[1:] - c4[:-1]
L = np.r_[0, np.cumsum(np.linalg.norm(dl, axis=1))]
s = 2*(L / L[-1]) - 1  # Normalized distance, from -1..1
# print("\n\n !!!! Avoiding the correct chord distribution !!!!\n")
print("Replacing the default elliptical chord distribution with the raw data")
planform.fc = PchipInterpolator(s, c)
# Or, just use his chord distribution directly?  Belloc, Eq:2, pg 2
# FIXME: this hotfix doesn't fix the EllipticalPlanform definition for SMC/MAC

# Similarly, overwrite `ftheta` to incorporate the linear torsion
twist = np.zeros(13)
twist[:2] = np.deg2rad(3)
twist[-2:] = np.deg2rad(3)
# print("\n !!!! Avoiding the correct twist !!!!\n")
print("Replacing the default `ftheta`")
planform.ftheta = PchipInterpolator(s, twist)

print("Planform:")
print(f" S: {planform.S}")
print(f" b: {planform.b}")
print(f" AR: {planform.AR}")

dMed = -np.rad2deg(np.arctan2(h, b/2))
dMax = -89.9
lobe = Parafoil.EllipticalLobe(dMed, dMax)
parafoil = Parafoil.ParafoilGeometry(planform, lobe, airfoil)

# Double check the final 0.6c line (should be a straight line)
# print("\nDEBUG> showing the final 0.6c line (should be straight)")
# s = np.linspace(-1, 1, 200)
# points = parafoil.c0(s)
# orientations = parafoil.section_orientation(s)
# points -= (0.6 * parafoil.planform.fc(s) * orientations.T[0]).T
# plt.plot(points.T[1], points.T[0])
# plt.xlabel('y')
# plt.ylabel('0.6c')
# plt.show()


print(f"Projected span>: Expected: {b:.3f}, Actual: {parafoil.b:.3f}")

print("\nFinished building the parafoil")

# plots.plot_parafoil_planform(parafoil, N_sections=250)
# plots.plot_parafoil_geo(parafoil, N_sections=250)


# Double check the chord lengths across the span
# s = np.linspace(-1, 1, 250)
# fig, ax = plt.subplots()
# ax.plot(parafoil.c4(s).T[1], parafoil.planform.fc(s), label='actual')
# ax.plot(c4.T[1], c, marker='.', label='expected')
# ax.legend()
# ax.set_xlabel('projected span position')
# ax.set_ylabel('chord length')
# ax.set_aspect('equal')
# plt.show()

# embed()
# 1/0


# ---------------------------------------------------------------------------
# Part 2: Define the wing

# I want the 60% of each chord at x=0
# ox = parafoil.planform.fc(0) * 0.6  # origin x at the central chord 60%
# s = np.linspace(-1, 1, 13)  # 12 segments, so 13 dividing sections
# print(parafoil.c0(s).T[0] - parafoil.planform.fc(s)*0.6 + ox)

brakes = BrakeGeometry.Cubic(0, 0.75, delta_max=0)

d_riser = 0.25  # For the 1/8 model, d_riser = 0.0875 / 0.350
z_riser = 1  # The 1/8 scale model has the cg 1m below the central chord
wing = ParagliderWing(parafoil, Parafoil.Phillips, brakes,
                      d_riser=d_riser, z_riser=z_riser,
                      pA=0.08, pC=0.80,  # unused
                      kappa_s=0.15)      # unused

# A `Harness` is required to instantiate the `Paraglider`, but should not
# produce any forces (so zero weight and drag).
harness = Harness.Spherical(mass=0, z_riser=0.0, S=0.0, CD=0.0)
glider = Paraglider(wing, harness)

print("\nFinished defining the brakes, wing, harness, and glider")
# embed()


# ---------------------------------------------------------------------------
# Part 3: Testing

# The paper says the wind tunnel is being used at 40m/s to produce a Reynold's
# number of 920,000. He neglects to mention the air density during the test,
# but if the dynamic viscosity of the air is standard, then we can compute the
# density of the air.
Re = 0.92e6
u = 40  # [m/s]
L = 0.350  # [m]  the central chord of the model
mu = 1.81e-5  # Standard dynamic viscosity of air
rho_air = Re * mu / (u * L)
print("rho_air:", rho_air)

# -----------------------------------
# One-off test cases
alpha, beta = np.deg2rad(7), np.deg2rad(0)
# alpha, beta = np.deg2rad(25), np.deg2rad(0)
# alpha, beta = np.deg2rad(24), np.deg2rad(5)
# alpha, beta = np.deg2rad(-5), np.deg2rad(10)
#
# UVW = np.asarray([cos(alpha)*cos(beta), sin(beta), sin(alpha)*cos(beta)])
# F, M, Gamma = glider.forces_and_moments(UVW, [0, 0, 0], [0, 0, 0], rho=1)
# embed()
# 1/0

# Full-range tests
Fs, Ms, Gammas = {}, {}, {}
alphas = np.deg2rad(np.linspace(-5, 25, 150))
betas = [0]
# betas = [5]
# betas = [0, 5]
# betas = [0, 5, 10]
# betas = [0, 5, 10, 15]
for beta_deg in betas:
    Fs[beta_deg], Ms[beta_deg], Gammas[beta_deg] = [], [], []

    Gamma = None
    for alpha in alphas:
        print(f"Test: alpha: {np.rad2deg(alpha):.2f}, beta: {beta_deg}")
        # The Paraglider computes the net moments about the "CG"

        beta = np.deg2rad(beta_deg)
        UVW = np.asarray([
            cos(alpha)*cos(beta), sin(beta), sin(alpha)*cos(beta)])
        PQR = [0, 0, 0]
        g = [0, 0, 0]
        F, M, Gamma = glider.forces_and_moments(UVW, PQR, g=g, rho=rho_air,
                                                Gamma=Gamma)
        Fs[beta_deg].append(F)
        Ms[beta_deg].append(M)
        Gammas[beta_deg].append(Gamma)
        if np.any(np.isnan(Gamma)):
            Gamma = None  # Don't propagate the errors!!


for beta in betas:
    Fs[beta] = np.asarray(Fs[beta])
    Ms[beta] = np.asarray(Ms[beta])
print()

# ---------------------------------------------------------------------------
# Compute the aerodynamic coefficients
#  * Uses the flattened wing area as the reference, as per the Belloc paper

# Hot-patching EllipticalPlanform broke `SMC`, `MAC`, and `S`
print("Workaround: Using the planform area from the force_estimator sections")
S = wing.force_estimator.dA.sum()

coefficients = {}
for beta in betas:
    print(f"\nResults for beta={beta} [degrees]")
    coefficients[beta] = {}

    CX, CY, CZ = Fs[beta].T / (.5 * rho_air * S)
    CN = -CZ
    CM = Ms[beta].T[1]/(.5 * rho_air * S * cc)  # Belloc uses the central chord

    # From Stevens, "Aircraft Control and Simulation", pg 90 (104)
    beta_rad = np.deg2rad(beta)
    CD = -cos(alphas)*cos(beta_rad)*CX - sin(beta_rad)*CY + sin(alphas)*cos(beta_rad)*CN
    CL = sin(alphas)*CX + cos(alphas)*CN

    # Compute the CL versus alpha slope using data from -5..5 degrees AoA
    nan_mask = np.isnan(alphas) | np.isnan(CL) | np.isnan(CD)
    alpha_mask = (alphas >= np.deg2rad(-5)) & (alphas <= np.deg2rad(5))
    mask = ~nan_mask & alpha_mask
    CLp = Polynomial.fit(alphas[mask], CL[mask], 1)
    CDp = Polynomial.fit(alphas[mask], CD[mask], 4)
    alpha_0 = CLp.roots()
    CD0 = CDp(alpha_0)
    print(f"CL slope: {CLp.deriv()(0)} [1/rad]")
    print(f"Zero-lift alpha: {np.rad2deg(alpha_0)} [degrees]")
    print("CD0:", CD0)

    # Check the effective aspect ratio
    # AR_effective = CL**2 / ((CD - CD0) * np.pi)  # Belloc Eq:6 (corrected)
    # print("Effective aspect ratios:\n", AR_effective)
    # plt.plot(np.rad2deg(alphas), AR_effective)
    # plt.hlines(parafoil.AR, np.rad2deg(alphas[0]), np.rad2deg(alphas[-1]),
    #            'r', 'dashed', linewidth=1)
    # plt.show()

    coefficients[beta]['CL'] = CL
    coefficients[beta]['CD'] = CD
    coefficients[beta]['CM'] = CM

    # embed()
    # 1/0


# ---------------------------------------------------------------------------
# Recreate Belloc figures 5..8

plotted_betas = {0, 5, 10, 15}  # The betas present in Belloc's plots
fig, ax = plt.subplots(2, 2)

for beta in sorted(plotted_betas.intersection(betas)):
    CL = coefficients[beta]['CL']
    CD = coefficients[beta]['CD']
    CM_G = coefficients[0]['CM']
    m = '.'
    m = None
    ax[0, 0].plot(np.rad2deg(alphas), CL, label=r'$\beta$={}°'.format(beta),
                  marker=m)
    ax[1, 0].plot(CD, CL, label=r'$\beta$={}°'.format(beta), marker=m)
    ax[1, 1].plot(CM_G, CL, label=r'$\beta$={}°'.format(beta), marker=m)

ax[0, 0].set_xlabel('alpha [degrees]')
ax[0, 0].set_ylabel('CL')
ax[0, 0].set_xlim(-10, 25)
ax[0, 0].set_ylim(-0.4, 1.0)
ax[0, 0].legend()
ax[0, 0].grid()

ax[1, 0].set_xlabel('CD')
ax[1, 0].set_ylabel('CL')
ax[1, 0].set_xlim(0, 0.2)
ax[1, 0].set_ylim(-0.4, 1.0)
ax[1, 0].legend()
ax[1, 0].grid()

CL = coefficients[0]['CL']
CM_G = coefficients[0]['CM']                   # The total moment coefficient
CM_CD = coefficients[0]['CD']*cos(alphas)/cc   # Eq: 8
CM_CL = -coefficients[0]['CL']*sin(alphas)/cc  # Eq: 9
CM_c4 = CM_G - CM_CL - CM_CD                   # Eq: 7
ax[0, 1].plot(CM_G, CL, label='CM_G', marker=m)
ax[0, 1].plot(CM_CL, CL, label='CM_CL', marker=m)
ax[0, 1].plot(CM_CD, CL, label='CM_CD', marker=m)
ax[0, 1].plot(CM_c4, CL, label='CM_25%', marker=m)
ax[0, 1].set_xlabel('CM')
ax[0, 1].set_ylabel('CL')
ax[0, 1].legend()
ax[0, 1].grid()
ax[0, 1].set_xlim(-0.5, 0.2)
ax[0, 1].set_ylim(-0.4, 1.0)

ax[1, 1].set_xlabel('CM_G')
ax[1, 1].set_ylabel('CL')
ax[1, 1].set_xlim(-0.5, 0.1)
ax[1, 1].set_ylim(-0.4, 1.0)
ax[1, 1].legend()
ax[1, 1].grid()


# Figures 9, 11, and 12
ix_a0 = np.argmin(np.abs(np.rad2deg(alphas) - 0))
ix_a5 = np.argmin(np.abs(np.rad2deg(alphas) - 5))
ix_a10 = np.argmin(np.abs(np.rad2deg(alphas) - 10))
ix_a15 = np.argmin(np.abs(np.rad2deg(alphas) - 15))

Cy_a0, Cy_a5, Cy_a10, Cy_a15 = [], [], [], []
Cl_a0, Cl_a5, Cl_a10, Cl_a15 = [], [], [], []
Cn_a0, Cn_a5, Cn_a10, Cn_a15 = [], [], [], []
for beta in betas:
    # Lateral force
    Cy_a0.append(Fs[beta].T[1][ix_a0] / (.5 * rho_air * S))
    Cy_a5.append(Fs[beta].T[1][ix_a5] / (.5 * rho_air * S))
    Cy_a10.append(Fs[beta].T[1][ix_a10] / (.5 * rho_air * S))
    Cy_a15.append(Fs[beta].T[1][ix_a15] / (.5 * rho_air * S))

    # Rolling moment coefficients
    Cl_a0.append(Ms[beta].T[0][ix_a0] / (.5 * rho_air * S * cc))
    Cl_a5.append(Ms[beta].T[0][ix_a5] / (.5 * rho_air * S * cc))
    Cl_a10.append(Ms[beta].T[0][ix_a10] / (.5 * rho_air * S * cc))
    Cl_a15.append(Ms[beta].T[0][ix_a15] / (.5 * rho_air * S * cc))

    # Yawing moment coeficients
    Cn_a0.append(Ms[beta].T[2][ix_a0] / (.5 * rho_air * S * cc))
    Cn_a5.append(Ms[beta].T[2][ix_a5] / (.5 * rho_air * S * cc))
    Cn_a10.append(Ms[beta].T[2][ix_a10] / (.5 * rho_air * S * cc))
    Cn_a15.append(Ms[beta].T[2][ix_a15] / (.5 * rho_air * S * cc))

fig9, ax9 = plt.subplots()
ax9.plot(betas, Cy_a0, label=r'$\alpha$=0°')
ax9.plot(betas, Cy_a5, label=r'$\alpha$=5°')
ax9.plot(betas, Cy_a10, label=r'$\alpha$=10°')
ax9.plot(betas, Cy_a15, label=r'$\alpha$=15°')
ax9.set_xlim(-20, 20)
ax9.set_ylim(-0.3, 0.3)
ax9.set_title("Figure 9: The effect of sideslip on the lateral force")
ax9.set_xlabel(r'$\beta$')
ax9.set_ylabel(r'$\mathrm{C_y}$')
ax9.legend()
ax9.grid()

fig11, ax11 = plt.subplots()
ax11.plot(betas, Cl_a0, label=r'$\alpha$=0°')
ax11.plot(betas, Cl_a5, label=r'$\alpha$=5°')
ax11.plot(betas, Cl_a10, label=r'$\alpha$=10°')
ax11.plot(betas, Cl_a15, label=r'$\alpha$=15°')
ax11.set_xlim(-20, 20)
ax9.set_ylim(-0.2, 0.2)
ax11.set_title("Figure 11: The effect of sideslip on the rolling moment")
ax11.set_xlabel(r'$\beta$')
ax11.set_ylabel(r'$\mathrm{Cl_G}$')
ax11.legend()
ax11.grid()

fig12, ax12 = plt.subplots()
ax12.plot(betas, Cn_a0, label=r'$\alpha$=0°')
ax12.plot(betas, Cn_a5, label=r'$\alpha$=5°')
ax12.plot(betas, Cn_a10, label=r'$\alpha$=10°')
ax12.plot(betas, Cn_a15, label=r'$\alpha$=15°')
ax12.set_xlim(-20, 20)
ax12.set_ylim(-0.1, 0.1)
ax12.set_title("Figure 12: The effect of sideslip on the yawing moment")
ax12.set_xlabel(r'$\beta$')
ax12.set_ylabel(r'$\mathrm{Cn_G}$')
ax12.legend()
ax12.grid()

plt.show()

embed()
