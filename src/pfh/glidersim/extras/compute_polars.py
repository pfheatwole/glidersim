"""Utility functions for analyzing wing performance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pfh.glidersim import foil_aerodynamics, orientation


def compute_polar_data(glider, rho_air: float = 1.225, N: int = 21):
    """Compute the equilibrium conditions over the range of control inputs."""
    eqs_a = []
    delta_as = np.linspace(0, 1, N)
    equilibrium = {  # Initial guesses
        "alpha_b": np.deg2rad(9),
        "Theta_b2e": [0, np.deg2rad(3), 0],
        "v_RM2e": 10,  # Only need to guess the magnitude
        "reference_solution": None,
    }
    print("Calculating equilibrium states over the range of accelerator")
    for n, da in enumerate(delta_as):
        print(f"\r{da:.2f}", end="")
        try:
            equilibrium = glider.equilibrium_state(
                delta_a=da,
                delta_b=0,
                rho_air=rho_air,
                alpha_0=equilibrium["alpha_b"],
                theta_0=equilibrium["Theta_b2e"][1],
                v_0=np.linalg.norm(equilibrium["v_RM2e"]),
                reference_solution=equilibrium["reference_solution"],
            )
        except foil_aerodynamics.ConvergenceError:
            print("\nConvergence started failing. Aborting early.")
            delta_as = delta_as[:n]
            break
        eqs_a.append(equilibrium)
    print()

    eqs_b = []
    delta_bs = np.linspace(0, 1, N)
    equilibrium = {  # Initial guesses
        "alpha_b": np.deg2rad(9),
        "Theta_b2e": [0, np.deg2rad(3), 0],
        "v_RM2e": 10,  # Only need to guess the magnitude
        "reference_solution": None,
    }
    print("Calculating equilibrium states over the range of brake")
    for n, db in enumerate(delta_bs):
        print("\rdb: {:.2f}".format(db), end="")
        try:
            equilibrium = glider.equilibrium_state(
                delta_a=0,
                delta_b=db,
                alpha_0=equilibrium["alpha_b"],
                theta_0=equilibrium["Theta_b2e"][1],
                v_0=np.linalg.norm(equilibrium["v_RM2e"]),
                rho_air=rho_air,
                reference_solution=equilibrium["reference_solution"],
            )
        except foil_aerodynamics.ConvergenceError:
            print("\nConvergence started failing. Aborting early.")
            delta_bs = delta_bs[:n]
            break
        eqs_b.append(equilibrium)
    print()

    # Convert the airspeeds from body coordinates to Earth coordinates
    Theta_b2e_a = np.asarray([e["Theta_b2e"] for e in eqs_a])
    Theta_b2e_b = np.asarray([e["Theta_b2e"] for e in eqs_b])
    q_e2b_a = orientation.euler_to_quaternion(Theta_b2e_a.T).T * [1, -1, -1, -1]
    q_e2b_b = orientation.euler_to_quaternion(Theta_b2e_b.T).T * [1, -1, -1, -1]
    v_RM2e_a = [e["v_RM2e"] for e in eqs_a]
    v_RM2e_b = [e["v_RM2e"] for e in eqs_b]
    v_RM2e_a = orientation.quaternion_rotate(q_e2b_a, v_RM2e_a)
    v_RM2e_b = orientation.quaternion_rotate(q_e2b_b, v_RM2e_b)

    accelerating = {
        "delta": delta_as,
        "alpha_b": [e["alpha_b"] for e in eqs_a],
        "theta_b": [e["Theta_b2e"][1] for e in eqs_a],
        "v_RM2e": v_RM2e_a,
        "glide_ratio": [e["glide_ratio"] for e in eqs_a],
    }
    braking = {
        "delta": delta_bs,
        "alpha_b": [e["alpha_b"] for e in eqs_b],
        "theta_b": [e["Theta_b2e"][1] for e in eqs_b],
        "v_RM2e": v_RM2e_b,
        "glide_ratio": [e["glide_ratio"] for e in eqs_b],
    }

    return accelerating, braking


def plot_wing_coefficients(
    wing,
    delta_b: float = 0,
    v_mag: float = 10,
    rho_air: float = 1.225,
) -> None:
    alphas = np.deg2rad(np.linspace(-10, 25, 50))
    Fs, Ms = [], []
    reference_solution = None
    for k, alpha in enumerate(alphas):
        print(f"\ralpha: {np.rad2deg(alpha):6.2f}", end="")
        try:
            dF, dM, reference_solution, = wing.forces_and_moments(
                delta_bl=delta_b,
                delta_br=delta_b,
                v_W2b=-v_mag * np.array([np.cos(alpha), 0, np.sin(alpha)]),
                rho_air=rho_air,
                reference_solution=reference_solution,
            )
            Fs.append(dF.sum(axis=0))
            Ms.append(dM.sum(axis=0))
        except foil_aerodynamics.ConvergenceError:
            break
    alphas = alphas[:k]
    Fs = Fs[:k]
    Ms = Ms[:k]

    CLs, CDs, CMs = [], [], []
    for n, F in enumerate(Fs):
        L = F[0] * np.sin(alphas[n]) - F[2] * np.cos(alphas[n])
        D = -F[0] * np.cos(alphas[n]) - F[2] * np.sin(alphas[n])
        CL = L / (0.5 * rho_air * v_mag**2 * wing.canopy.S)
        CD = D / (0.5 * rho_air * v_mag**2 * wing.canopy.S)
        CM = Ms[n][1] / (
            0.5 * rho_air * v_mag**2 * wing.canopy.S * wing.canopy.chord_length(0)
        )
        CLs.append(CL)
        CDs.append(CD)
        CMs.append(CM)

    style = {"c": "k", "lw": 0.75, "ls": "-", "marker": "o", "markersize": "1.5"}
    fig, ax = plt.subplots(3, 2, figsize=(9, 8))
    ax[0, 0].plot(np.rad2deg(alphas), CLs, **style)
    ax[1, 0].plot(np.rad2deg(alphas), CDs, **style)
    ax[2, 0].plot(np.rad2deg(alphas), CMs, **style)
    plt.setp(ax[:, 0], xlabel="alpha [deg]")
    ax[0, 0].set_ylabel("Lift Coefficient")
    ax[1, 0].set_ylabel("Drag Coefficient")
    ax[2, 0].set_ylabel("Pitching Coefficient")

    ax[0, 1].plot(np.rad2deg(alphas), np.array(CLs) / np.array(CDs), **style)
    ax[0, 1].set_xlabel("alpha [deg]")
    ax[0, 1].set_ylabel("CL/CD")

    ax[1, 1].plot(CDs, CLs, **style)
    ax[1, 1].set_xlabel("CD")
    ax[1, 1].set_ylabel("CL")

    for ax in fig.axes:
        if len(ax.lines) > 0:
            ax.grid()

    fig.tight_layout()  # Prevent axes overlapping titles
    plt.show()
    breakpoint()
