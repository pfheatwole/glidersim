"""
Recreates the paraglider analysis in "Wind Tunnel Investigation of a Rigid
Paraglider Reference Wing", H. Belloc, 2015

TODO:
* Why am I still significantly overestimating CL? Wrong airfoil data?
* The CM25% in the paper are positive? How? The section Cm are almost entirely
  negative until `alpha > 10`.
* Review my CM_G calculation; both it and CM25% look wrong (which make sense,
  since CM25% is computed directly from CM_G, while CM_CL and CM_CD look great.
* Review the force and moment calculations.
* Phillips needs "Picard iterations" to deal with stalled sections
* I need to fix how I calculate L and D when beta>0
   * Observe in my `CL vs CD`: CD decreases with beta? Nope, wrong
   * Also, in `CL vs alpha`, why does alpha_L0 increase with beta?
* Review the "effective AR" equation in Eq:6
"""


from IPython import embed

import matplotlib.pyplot as plt  # noqa: F401

import numpy as np

import pfh.glidersim as gsim

import scipy.interpolate


# ---------------------------------------------------------------------------
# Wing definition from the paper

# print("\n\nWARNING: Did you remove the extra air intake drag (~0.07)?\n\n")
# input("<press any key to continue>")

# Table 1: the Full-scale wing dimensions converted into 1/8 model
h = 3 / 8  # Arch height (vertical deflection from wing root to tips) [m]
cc = 2.8 / 8  # The central chord [m]
b = 11.00 / 8  # The projected span [m]
S = 25.08 / (8**2)  # The projected area [m^2]
AR = 4.82  # The projected aspect ratio

b_flat = 13.64 / 8  # The flattened span [m]
S_flat = 28.56 / (8**2)  # The flattened area [m^2]
AR_flat = 6.52  # The flattened aspect ratio

# Table 2: Coordinates along the 0.6c line for the 1/8 model in [m]
xyz = np.array(
    [
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
        [0.000,  0.688,  0.000],
    ],
)

c = np.array([0.107, 0.137, 0.198, 0.259, 0.308, 0.339, 0.350,
              0.339, 0.308, 0.259, 0.198, 0.137, 0.107])  # chords [m]

theta = np.deg2rad([3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3])  # torsion [deg]

# Compute the section indices
L_segments = np.linalg.norm(np.diff(xyz[..., 1:], axis=0), axis=1)
s_xyz = np.cumsum(np.r_[0, L_segments]) / L_segments.sum() * 2 - 1

# Coordinates and chords are in meters, and must be normalized
fx = scipy.interpolate.interp1d(s_xyz, xyz.T[0] / (b_flat / 2))
fy = scipy.interpolate.interp1d(s_xyz, xyz.T[1] / (b_flat / 2))
fz = scipy.interpolate.interp1d(s_xyz, (xyz.T[2] - xyz[6, 2]) / (b_flat / 2))
fc = scipy.interpolate.interp1d(s_xyz, c / (b_flat / 2))
ftheta = scipy.interpolate.interp1d(s_xyz, theta)


# ---------------------------------------------------------------------------
# Build the canopy and wing

airfoil_geo = gsim.airfoil.NACA(23015, convention='vertical')

polardir = "/home/peter/model/work/glidersim/scripts/polars/NACA23015_N7.0"
airfoil_coefs = gsim.airfoil.XFLR5Coefficients(polardir, flapped=False)

airfoil = gsim.airfoil.Airfoil(airfoil_coefs, airfoil_geo)


class InterpolatedLobe:
    """Interface to use a PchipInterpolator for the lobe."""

    def __init__(self, s, y, z):
        y = np.asarray(y)
        z = np.asarray(z)

        assert y.ndim == 1 and z.ndim == 1

        self._f = scipy.interpolate.PchipInterpolator(s, np.c_[y, z])
        self._fd = self._f.derivative()

    def __call__(self, s):
        return self._f(s)

    def derivative(self, s):
        return self._fd(s)


# FIXME: move the resampling logic into `InterpolatedLobe`, and make that an
#        official helper class in `foil.py`. It should also use more intelligent
#        resampling (only needs two extra samples on either side of each point)
s = np.linspace(-1, 1, 1000)  # Resample so the cubic-fit stays linear
lobe = InterpolatedLobe(s, fy(s), fz(s))

# Try overriding the sampled foil with smooth curves
# print("\n\nWARNING: Replacing the sampled curves with perfect functions\n")
# fc = gsim.foil.elliptical_chord(.350, .107)
# lobe = gsim.foil.elliptical_lobe(28.6, 85)

chord_surface = gsim.foil.ChordSurface(
    x=0,  # or `fx`
    r_x=0.6,
    yz=lobe,
    r_yz=0.6,
    chord_length=fc,
    torsion=ftheta,
)

canopy = gsim.foil.SimpleFoil(
    airfoil=airfoil,
    chords=chord_surface,
    b_flat=b_flat,
)

print("Finished defining the canopy. Checking fit...")
print(f"Projected area> Expected: {S}, Actual: {canopy.S}")
print(f"Flattened area> Expected: {S_flat}, Actual: {canopy.S_flat}")
print(f"Projected AR> Expected: {AR}, Actual: {canopy.AR}")
print(f"Flattened AR> Expected: {AR_flat}, Actual: {canopy.AR_flat}")

wing = gsim.paraglider_wing.ParagliderWing(
    canopy=canopy,
    force_estimator=gsim.foil.Phillips(canopy, 40),
    brake_geo=gsim.brake_geometry.Cubic(0, 0.75, delta_max=0),  # unused
    d_riser=0.25,  # For the 1/8 model, d_riser = 0.0875 / 0.350 = 25%
    z_riser=1,  # The 1/8 scale model has the cg 1m below the central chord
    pA=0.08,  # unused
    pC=0.80,  # unused
    kappa_a=0,  # unused
    rho_upper=0,  # Neglect gravitational forces
    rho_lower=0,
)

print("\nFinished defining the complete wing. Pausing for review.\n")
gsim.plots.plot_foil(canopy, N_sections=121)
embed()
# 1/0

# FIXME: add some comparisons of the different measurements. Call attention to
#        the discrepancy between the specified versus calculated projected
#        semispans `b`: they don't quite match since the paper used the 0.6c
#        reference curve while the `ChordSurface` used the true maximum
#        projected y-coordinates, which are greater due to the torsion. Not
#        sure why `b_flat` doesn't match; those should be exact?

# ---------------------------------------------------------------------------
# Testing

# The paper says the wind tunnel is being used at 40m/s to produce a Reynold's
# number of 920,000. He neglects to mention the air density during the test,
# but if the dynamic viscosity of the air is standard, then we can compute the
# density of the air.
Re = 0.92e6
v_mag = 40  # Wind tunnel airspeed [m/s]
L = 0.350  # central chord [m]
mu = 1.81e-5  # Standard dynamic viscosity of air
rho_air = Re * mu / (v_mag * L)
print("rho_air:", rho_air)

# Full-range tests
Fs, Ms, solutions = {}, {}, {}
alphas = {}
# alphas = np.deg2rad(np.linspace(-5, 20, 150))
# alphas = np.deg2rad(np.linspace(0, 10, 150))
# betas = [0]
# betas = [5]
# betas = [0, 5]
# betas = [0, 5, 10]
# betas = [0, 5, 10, 15]
betas = np.arange(16)

for kb, beta_deg in enumerate(betas):
    Fs[beta_deg], Ms[beta_deg], solutions[beta_deg] = [], [], []
    cp_wing = wing.control_points(0)  # Section control points

    alphas_up = np.deg2rad(np.linspace(2, 22, 75))
    alphas_down = np.deg2rad(np.linspace(2, -5, 30))[1:]

    # First, going down
    ref = None
    for ka, alpha in enumerate(alphas_down):
        print(f"\rTest: alpha: {np.rad2deg(alpha): 6.2f}, beta: {beta_deg}", end="")
        beta = np.deg2rad(beta_deg)
        v_W2b = np.asarray(
            [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
        )
        v_W2b *= -v_mag  # The Reynolds numbers are a function of the magnitude

        try:
            dF, dM, ref = wing.forces_and_moments(
                0, 0, v_W2b=v_W2b, rho_air=rho_air, reference_solution=ref,
            )
        except gsim.foil.ForceEstimator.ConvergenceError:
            ka -= 1  # FIXME: messing with the index!
            break
            # FIXME: continue, or break? Maybe try the solution from a previous
            #        `beta`? eg: ref = solutions[betas[kb - 1]][ka]

        F = dF.sum(axis=0)
        M = dM.sum(axis=0)  # Moment due to section `Cm`
        M += np.cross(cp_wing, dF).sum(axis=0)  # Add the moment due to forces

        Fs[beta_deg].append(F)
        Ms[beta_deg].append(M)
        solutions[beta_deg].append(ref)

    alphas_down = alphas_down[:ka+1]  # Truncate when convergence failed

    # Reverse the order
    Fs[beta_deg] = Fs[beta_deg][::-1]
    Ms[beta_deg] = Ms[beta_deg][::-1]
    solutions[beta_deg] = solutions[beta_deg][::-1]
    alphas_down = alphas_down[::-1]

    # Again, going up
    ref = None
    for ka, alpha in enumerate(alphas_up):
        print(f"\rTest: alpha: {np.rad2deg(alpha): 6.2f}, beta: {beta_deg}", end="")
        beta = np.deg2rad(beta_deg)
        v_W2b = np.asarray(
            [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
        )
        v_W2b *= -v_mag  # The Reynolds numbers are a function of the magnitude

        try:
            dF, dM, ref = wing.forces_and_moments(
                0, 0, v_W2b=v_W2b, rho_air=rho_air, reference_solution=ref,
            )
        except gsim.foil.ForceEstimator.ConvergenceError:
            ka -= 1  # FIXME: messing with the index!
            break
            # FIXME: continue, or break? Maybe try the solution from a previous
            #        `beta`? eg: ref = solutions[betas[kb - 1]][ka]

        F = dF.sum(axis=0)
        M = dM.sum(axis=0)  # Moment due to section `Cm`
        M += np.cross(cp_wing, dF).sum(axis=0)  # Add the moment due to forces

        Fs[beta_deg].append(F)
        Ms[beta_deg].append(M)
        solutions[beta_deg].append(ref)

    alphas_up = alphas_up[:ka+1]  # Truncate when convergence failed

    alphas[beta_deg] = np.r_[alphas_down, alphas_up]  # Stitch them together

    print()

for beta in betas:
    Fs[beta] = np.asarray(Fs[beta])
    Ms[beta] = np.asarray(Ms[beta])

print()
print("\nFinished tests, preparing to plot. Pausing for review.\n")
embed()

# ---------------------------------------------------------------------------
# Compute the aerodynamic coefficients
#
# Uses the flattened wing area as the reference, as per the Belloc paper.

S = canopy.S_flat

coefficients = {}
for beta in betas:
    print(f"\nResults for beta={beta} [degrees]")
    coefficients[beta] = {}

    CX, CY, CZ = Fs[beta].T / (0.5 * rho_air * v_mag**2 * S)
    CN = -CZ
    CM = Ms[beta].T[1] / (0.5 * rho_air * v_mag**2 * S * cc)  # The paper uses the central chord

    # From Stevens, "Aircraft Control and Simulation", pg 90 (104)
    beta_rad = np.deg2rad(beta)
    CD = (
        -np.cos(alphas[beta]) * np.cos(beta_rad) * CX
        - np.sin(beta_rad) * CY
        + np.sin(alphas[beta]) * np.cos(beta_rad) * CN
    )
    CL = np.sin(alphas[beta]) * CX + np.cos(alphas[beta]) * CN

    # Compute the CL versus alpha slope using data from -5..5 degrees AoA
    CLp = np.polynomial.Polynomial.fit(alphas[beta], CL, 1)
    CDp = np.polynomial.Polynomial.fit(alphas[beta], CD, 4)  # FIXME: 4?
    alpha_0 = CLp.roots()
    CD0 = CDp(alpha_0)
    print(f"CL slope: {CLp.deriv()(0)} [1/rad]")
    print(f"Zero-lift alpha: {np.rad2deg(alpha_0)} [degrees]")
    print("CD0:", CD0)

    # Check the effective aspect ratio
    # AR_effective = CL**2 / ((CD - CD0) * np.pi)  # Belloc Eq:6 (corrected)
    # print("Effective aspect ratios:\n", AR_effective)
    # plt.plot(np.rad2deg(alphas), AR_effective)
    # plt.hlines(canopy.AR, np.rad2deg(alphas[0]), np.rad2deg(alphas[-1]),
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
    CM_G = coefficients[beta]['CM']
    m = '.'
    m = None
    ax[0, 0].plot(np.rad2deg(alphas[beta]), CL, label=r'$\beta$={}°'.format(beta),
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

# Compute the pitching moment coefficients:
#
# CM_CD : due to the drag force applied to the wing
# CM_CL : due to the lift force applied to the wing
# CM_c4 : due to the wing shape
# CM_G: the total pitching moment = CM_CD + CM_CL + CM_c4
CL = coefficients[0]['CL']
CM_G = coefficients[0]['CM']
CM_CD = coefficients[0]['CD'] * np.cos(alphas[0]) / cc   # Eq: 8
CM_CL = -coefficients[0]['CL'] * np.sin(alphas[0]) / cc  # Eq: 9
CM_c4 = CM_G - CM_CL - CM_CD                       # Eq: 7

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
Cy_a0, Cy_a5, Cy_a10, Cy_a15 = [], [], [], []
Cl_a0, Cl_a5, Cl_a10, Cl_a15 = [], [], [], []
Cn_a0, Cn_a5, Cn_a10, Cn_a15 = [], [], [], []
for beta in betas:
    ix_a0 = np.argmin(np.abs(np.rad2deg(alphas[beta]) - 0))
    ix_a5 = np.argmin(np.abs(np.rad2deg(alphas[beta]) - 5))
    ix_a10 = np.argmin(np.abs(np.rad2deg(alphas[beta]) - 10))
    ix_a15 = np.argmin(np.abs(np.rad2deg(alphas[beta]) - 15))

    # Lateral force
    Cy_a0.append(Fs[beta].T[1][ix_a0] / (0.5 * rho_air * v_mag**2 * S))
    Cy_a5.append(Fs[beta].T[1][ix_a5] / (0.5 * rho_air * v_mag**2 * S))
    Cy_a10.append(Fs[beta].T[1][ix_a10] / (0.5 * rho_air * v_mag**2 * S))
    Cy_a15.append(Fs[beta].T[1][ix_a15] / (0.5 * rho_air * v_mag**2 * S))

    # Rolling moment coefficients
    Cl_a0.append(Ms[beta].T[0][ix_a0] / (0.5 * rho_air * v_mag**2 * S * cc))
    Cl_a5.append(Ms[beta].T[0][ix_a5] / (0.5 * rho_air * v_mag**2 * S * cc))
    Cl_a10.append(Ms[beta].T[0][ix_a10] / (0.5 * rho_air * v_mag**2 * S * cc))
    Cl_a15.append(Ms[beta].T[0][ix_a15] / (0.5 * rho_air * v_mag**2 * S * cc))

    # Yawing moment coeficients
    Cn_a0.append(Ms[beta].T[2][ix_a0] / (0.5 * rho_air * v_mag**2 * S * cc))
    Cn_a5.append(Ms[beta].T[2][ix_a5] / (0.5 * rho_air * v_mag**2 * S * cc))
    Cn_a10.append(Ms[beta].T[2][ix_a10] / (0.5 * rho_air * v_mag**2 * S * cc))
    Cn_a15.append(Ms[beta].T[2][ix_a15] / (0.5 * rho_air * v_mag**2 * S * cc))

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
