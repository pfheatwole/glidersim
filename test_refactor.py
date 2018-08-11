import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from IPython import embed

import Airfoil
import Parafoil

import BrakeGeometry
import Harness
from ParagliderWing import ParagliderWing
from Paraglider import Paraglider

from plots import plot_airfoil_geo, plot_parafoil_geo, plot_parafoil_planform


def LD_ratio(dF, alpha):
    Fx, Fz = dF[0], dF[2]
    L = Fx*np.sin(alpha) - Fz*np.cos(alpha)
    D = -Fx*np.cos(alpha) - Fz*np.sin(alpha)
    return L/D


def build_elliptical_parafoil(b_flat, taper, dMed, sMed, airfoil,
                              SMC=None, MAC=None,
                              dMax=None, sMax=None,
                              torsion_max=0, torsion_exponent=6):

    if SMC is None and MAC is None:
        raise ValueError("One of the SMC or MAC are required")

    if dMed > 0 or (dMax is not None and dMax > 0):
        raise ValueError("dihedral must be negative")

    if sMed < 0 or (sMax is not None and sMax < 0):
        raise ValueError("sweep must be positive")  # FIXME: why?

    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print("Using minimum max dihedral ({})".format(dMax))

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print("Using minimum max sweep ({})".format(sMax))

    if SMC is not None:
        c0 = Parafoil.EllipticalPlanform.SMC_to_c0(SMC, taper)
    else:
        c0 = Parafoil.EllipticalPlanform.MAC_to_c0(MAC, taper)

    planform = Parafoil.EllipticalPlanform(
        b_flat, c0, taper, sMed, sMax, torsion_exponent, torsion_max)
    lobe = Parafoil.EllipticalLobe(dMed, dMax)

    return Parafoil.ParafoilGeometry(planform, lobe, airfoil)


def plot_coefficients(coefs):
    CLs = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CLs.append(coefs.CL(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CLs):
        # ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
        ax.plot(alphas, np.full_like(alphas, deltas[n]), c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CL')
    plt.show()

    CDs = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CDs.append(coefs.CD(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CDs):
        # ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
        ax.plot(alphas, np.full_like(alphas, deltas[n]), c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CD')
    plt.show()

    CM_c4s = []
    alphas = np.deg2rad(np.linspace(-1.5, 30))
    deltas = np.linspace(0, 1, 25)
    for d in deltas:
        CM_c4s.append(coefs.CM_c4(alphas, d))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n, c in enumerate(CM_c4s):
        # ax.plot(alphas, np.ones_like(alphas)*deltas[n], c)
        ax.plot(alphas, np.full_like(alphas, deltas[n]), c)
    ax.set_xlabel('alpha')
    ax.set_ylabel('delta')
    ax.set_zlabel('CM')
    plt.show()


def plot_polar(glider):
    alpha_eq = []
    for d in np.linspace(0, 1, 21)[::-1]:
        alpha_eq.append(glider.wing.alpha_eq(d, 0))
    for d in np.linspace(0, 1, 21):
        alpha_eq.append(glider.wing.alpha_eq(0, d))
    alpha_eq = np.asarray(alpha_eq)

    w_brakes = []
    for d in np.linspace(0, 1, 21)[::-1]:
        w_brakes.append(glider.equilibrium_glide(d, 0))
    w_brakes = np.asarray(w_brakes)

    w_speedbar = []
    for d in np.linspace(0, 1, 21):
        w_speedbar.append(glider.equilibrium_glide(0, d))
    w_speedbar = np.asarray(w_speedbar)

    eq = np.vstack([w_brakes, w_speedbar])

    gamma_eq = alpha_eq - eq[:, 0]

    Vx = eq[:, 1] * np.cos(gamma_eq)
    Vz = -eq[:, 1] * np.sin(gamma_eq)

    plt.plot(Vx, Vz)
    plt.show()


def main():

    # -----------------------------------------------------------------------
    # Airfoil

    # print("\nAirfoil: GNULAB3, simple flap, hinge at 80%")
    # TODO: AirfoilGeometry importer for .DAT files (`gnulab3.dat`)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/gnulab3_polars.csv', 0.8)

    # print("\nAirfoil: NACA2412, simple flap, hinge at 80%")
    # airfoil_geo = Airfoil.NACA4(2412)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca2412_xhinge80_yhinge_50.csv', 0.8)

    print("\nAirfoil: NACA4412, simple flap, hinge at 80%")
    airfoil_geo = Airfoil.NACA4(4412)
    airfoil_coefs = Airfoil.GridCoefficients('polars/naca4412_xhinge80_yhinge_50.csv', 0.8)

    # print("\nAirfoil: NACA4415, simple flap, hinge at 80%")
    # airfoil_geo = Airfoil.NACA4(4415)
    # airfoil_coefs = Airfoil.GridCoefficients('polars/naca4415_xhinge80_yhinge_50.csv', 0.8)

    # plot_airfoil_geo(airfoil_geo)


    # -----------------------------------------------------------------------
    # Parafoil

    # Hook3 specs:
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat/AR_flat
    S, b, AR = 19.55, 8.84, 4.00

    airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)
    parafoil = build_elliptical_parafoil(
        b_flat=b_flat, SMC=SMC_flat, taper=0.35, dMed=-32, dMax=-75,
        sMed=13.5, torsion_max=0, airfoil=airfoil)

    print("planform flat span:", parafoil.planform.b)
    print("planform flat area:", parafoil.planform.S)
    print("planform flat AR:  ", parafoil.planform.AR)
    print("planform flat SMC: ", parafoil.planform.SMC)
    print("planform flat MAC: ", parafoil.planform.MAC)

    print("planform span:", parafoil.b)
    print("planform area:", parafoil.S)
    print("planform AR:  ", parafoil.AR)

    # print("Drawing the parafoil")
    # plot_parafoil_planform(parafoil)
    # plot_parafoil_geo(parafoil, N_sections=25)

    # -----------------------------------------------------------------------
    # Brake geometry

    p_start, p_peak = 0.05, 0.75
    delta_max = np.deg2rad(50)*0.99 * (1 - 0.8)   # FIXME: magic number!
    brakes = BrakeGeometry.Cubic(p_start, p_peak, delta_max)

    # -----------------------------------------------------------------------
    # Wing and glider (using two different force estimation methods)

    wing2d = ParagliderWing(parafoil, Parafoil.Phillips2D, brakes,
                            # d_riser=0.5, z_riser=7,
                            d_riser=0.35, z_riser=9,
                            kappa_s=0.4)
    wing3d = ParagliderWing(parafoil, Parafoil.Phillips, brakes,
                            # d_riser=0.5, z_riser=7,
                            d_riser=0.35, z_riser=7,
                            kappa_s=0.4)

    # ---------------------------------------------------------------------
    # Tests

    cp_y = wing2d.control_points(delta_s=0)[:, 1]
    K = len(cp_y)
    V_rel = np.asarray([[10.0, 0.0, 1.0]] * K)  # V_cp2w in frd
    # V_rel[:, 0] += np.linspace(0, 1, K)**2 * 2  # spinning!

    #          Bl     Br
    deltas = [0.00, 0.00]
    # deltas = [0.00, 0.25]
    # deltas = [0.00, 0.50]
    # deltas = [0.00, 0.75]
    # deltas = [0.00, 1.00]
    # deltas = [1.00, 1.00]

    print("Computing the forces and moments for the 2D and 3D wings")
    dF_2d, dM_2d = wing2d.forces_and_moments(V_rel, *deltas)
    dF_3d, dM_3d = wing3d.forces_and_moments(V_rel, *deltas)

    # print("Plotting the forces")
    # dF_2d, dF_3d = dF_2d.T, dF_3d.T
    # fig, ax = plt.subplots(3, sharex=True, figsize=(16, 10))
    # ax[0].plot(cp_y, dF_2d[0], label='2D')
    # ax[0].plot(cp_y, dF_3d[0], label='3D', marker='.')
    # ax[1].plot(cp_y, dF_2d[1], label='2D')
    # ax[1].plot(cp_y, dF_3d[1], label='3D', marker='.')
    # ax[2].plot(cp_y, dF_2d[2], label='2D')
    # ax[2].plot(cp_y, dF_3d[2], label='3D', marker='.')
    # ax[0].set_xlabel('spanwise position')
    # ax[0].set_ylabel('Fx')
    # ax[1].set_ylabel('Fy')
    # ax[2].set_ylabel('Fz')
    # ax[0].grid(True)
    # ax[1].grid(True)
    # ax[2].grid(True)
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # plt.show()
    # dF_2d, dF_3d = dF_2d.T, dF_3d.T

    harness = Harness.Spherical(mass=80, z_riser=0.5, S=0.55, CD=0.7)

    # Test the effect of the harness
    Vmid = V_rel[K//2]
    alpha = np.arctan2(Vmid[2], Vmid[0])
    dF_harness, dM_harness = harness.forces_and_moments(Vmid)
    print("LD ratio (w/o the harness):", LD_ratio(dF_3d.sum(axis=0), alpha))
    print("LD ratio (w/ the harness): ",
          LD_ratio(dF_3d.sum(axis=0) + dF_harness, alpha))

    # Dynamics
    J_wing = wing3d.inertia(N=5000)
    o = -wing3d.foil_origin()
    tau = (np.cross(wing3d.force_estimator.cps - o, dF_3d) + dM_3d).sum(axis=0)
    alpha_rad = np.linalg.inv(J_wing) @ tau
    print("angular acceleration in deg/s**2:", np.rad2deg(alpha_rad))
    # print("\n---Skipping the rest---\n")
    # embed()
    # return

    # ------------------------
    glider3d = Paraglider(wing3d, harness)
    UVW = np.asarray([10.0, 0.0, 1.0])  # V_cm2e in frd
    P = np.deg2rad(15)
    Q = np.deg2rad(-5)
    # R = 0
    R = np.deg2rad(15)  # yaw rate = 15 degrees/sec clockwise
    PQR = np.array([P, Q, R])
    xyz = glider3d.control_points(delta_s=0)
    dF, dM = glider3d.forces_and_moments(UVW, PQR, delta_Bl=0, delta_Br=0,
                                         xyz=xyz)
    # embed()

    print("\nGlider results:")
    print("UVW:", UVW.round(3))
    print("PQR:", PQR.round(3))
    print("dF: ", dF.round(3))
    print("dM: ", dM.round(3))


if __name__ == "__main__":
    main()
