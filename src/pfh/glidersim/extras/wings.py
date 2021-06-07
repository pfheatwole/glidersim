"""FIXME: add module docstring"""

import numpy as np

import pfh.glidersim as gsim
from pfh.glidersim.extras import plots


def build_hook3(num_control_points=31, verbose=True):
    """Build an approximate Niviuk Hook 3, size 23."""
    if verbose:
        print("Building an (approximate) Niviuk Hook 3 23\n")

    # -----------------------------------------------------------------------
    # Airfoil

    if verbose:
        print("Airfoil: braking_NACA24018_Xtr0.25\n")
    airfoils = gsim.extras.airfoils.load_datfile_set("braking_NACA24018_Xtr0.25")
    airfoils = {d: airfoils[d]['airfoil'] for d in airfoils}
    airfoil_geo = gsim.airfoil.AirfoilGeometryInterpolator(airfoils)
    airfoil_coefs = gsim.extras.airfoils.load_polar("braking_NACA24018_Xtr0.25")
    delta_d_max = 0.20273  # FIXME: magic number from the set of coefficients

    # -----------------------------------------------------------------------
    # Canopy

    # True technical specs
    chord_tip, chord_root, chord_mean = 0.52, 2.58, 2.06
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat / AR_flat
    S, b, AR = 19.55, 8.84, 4.00
    m_s = 4.7  # Solid mass [kg]

    c = gsim.foil_layout.EllipticalChord(
        root=chord_root / (b_flat / 2),
        tip=chord_tip / (b_flat / 2),
    )

    # Geometric torsion
    #
    # The distribution is uncertain. In Sec. 11.4, pg 17 of the manual ("Line
    # Plan") it appears to have a roughly square-root spanwise distribution
    # with a maximum value of 6 or so, but without better data I'm sticking to
    # a linear distribution with a smaller peak (easier for the aerodynamics).
    # theta = gsim.foil_layout.PolynomialTorsion(start=0.0, peak=6, exponent=0.5)
    theta = gsim.foil_layout.PolynomialTorsion(start=0.05, peak=4, exponent=1)

    # Using `tip_anhedral = 75` is probably more accurate, but it also
    # increases the chances of stalling the wing tips during hard turns.
    layout = gsim.foil_layout.FoilLayout(
        r_x=0.70,
        x=0,
        r_yz=1.00,
        # yz=gsim.foil_layout.EllipticalArc(mean_anhedral=33, tip_anhedral=67),
        yz=gsim.foil_layout.EllipticalArc(mean_anhedral=32, tip_anhedral=75),
        c=c,
        theta=theta,
    )

    sections = gsim.foil_sections.FoilSections(
        profiles=airfoil_geo,
        coefficients=airfoil_coefs,
        intakes=gsim.foil_sections.SimpleIntakes(0.85, -0.04, -0.09),  # FIXME: guess
    )

    canopy = gsim.foil.SimpleFoil(
        layout=layout,
        sections=sections,
        # b=b,  # Option 1: Scale the using the projected span
        b_flat=b_flat,  # Option 2: Scale the using the flattened span
        aerodynamics_method=gsim.foil_aerodynamics.Phillips,
        aerodynamics_config={
            "v_ref_mag": 10,
            "K": num_control_points,
            "s_clamp": 0.95,  # Mitigate fictitious stalls at wing tips
        },
    )

    # Most of these values are based on data, but `kappa_x` is simply the
    # choice that produces a nice polar (max and min speeds are reasonable) and
    # the best glide ratio occurs at zero control inputs (some wings produce
    # better glide ratios with small amounts of accelerator, but without
    # evidence this a reasonable assumption).
    riser_position_parameters = {
        "kappa_x": 0.50 * chord_root,
        "kappa_z": 6.8,  # "Technical specs", pg 2
        "kappa_A": 0.11 * chord_root,  # "Users manual", "Line plan", pg 17
        "kappa_C": 0.59 * chord_root,  # "Users manual", "Line plan", pg 17
        "kappa_a": 0.15,  # "Technical specs", pg 2
    }

    # Estimated from https://www.youtube.com/watch?v=D-OyGZbOmS0
    # The `0` parameters are easy to estimate directly from small brake inputs.
    # The "max brake" parameters are more difficult since they depend on the
    # value of `kappa_b` (the maximum deflection supported by the model). An
    # initial guess shows that the set of NACA 24018 coefficients is going to
    # limit `kappa_b` to roughly 45cm; using the video to observe deflections
    # at ~45cm (the risers are 47cm) produce `start1` and `stop1`.
    brake_parameters = {
        "kappa_b": None,  # Set later with `maximize_kappa_b`
        "s_delta_start0": 0.30,
        "s_delta_start1": 0.08,
        "s_delta_stop0": 0.70,
        "s_delta_stop1": 1.05,
    }

    # Crude guesses to account for the bulk of the line drag.
    line_drag_parameters = {
        "total_line_length": 218,  # "Technical specs", pg 2
        "average_line_diameter": 1e-3,  # Blind guess
        "r_L2LE": np.array([[-0.5 * chord_root, -1.75, 1.75],  # FIXME: review
                            [-0.5 * chord_root,  1.75, 1.75]]),
        "Cd_lines": 0.98,  # ref: Kulhánek, 2019; page 5
    }

    lines = gsim.paraglider_wing.SimpleLineGeometry(
        **riser_position_parameters,
        **brake_parameters,
        **line_drag_parameters,
    )
    lines.maximize_kappa_b(delta_d_max, canopy.chord_length)

    wing = gsim.paraglider_wing.ParagliderWing(
        lines=lines,
        canopy=canopy,
        rho_upper=39 / 1000,  # [kg/m^2]  Porcher 9017 E77A
        rho_lower=35 / 1000,  # [kg/m^2]  Dominico N20DMF
        rho_ribs=41 / 1000,   # [kg/m^2]  Porcher 9017 E29
        N_cells=52,
    )

    # -----------------------------------------------------------------------
    # Plots

    # print("Drawing the canopy")
    # plots.plot_foil(canopy, 131, surface="airfoil", flatten=False)
    # plots.plot_foil(canopy, 131, surface="chord", flatten=False)
    # plots.plot_foil(canopy, 131, surface="camber", flatten=False)
    # plots.plot_foil(canopy, 71, surface="airfoil", flatten=True)
    # plots.plot_foil(canopy, 71, surface="chord", flatten=True)
    # plots.plot_foil(canopy, 71, surface="camber", flatten=True)
    # plots.plot_foil_topdown(canopy, 51)
    # plots.plot_foil_topdown(canopy, 51, flatten=True)

    # Plot a braking wing
    # s = np.linspace(-1, 1, 131)  # Inflated plots
    # ai = wing.lines.delta_d(s, 0.5, 1) / wing.canopy.chord_length(s)
    # plots.plot_foil(canopy, s, ai=ai, surface="airfoil", flatten=False)

    # Compare to the Hook 3 manual, sec 11.4 "Line Plan", page 17
    # plots.plot_foil_topdown(canopy, 53)

    # -----------------------------------------------------------------------
    # Model diagnostics

    if verbose:
        print("Canopy geometry:           [Target]")
        print(f"  flattened span: {canopy.b_flat:>6.3f}   [{b_flat:>6.3f}]")
        print(f"  flattened area: {canopy.S_flat:>6.3f}   [{S_flat:>6.3f}]")
        print(f"  flattened AR:   {canopy.AR_flat:>6.3f}   [{AR_flat:>6.3f}]")
        # print(f"  planform flat SMC   {canopy.SMC:>6.3f}")
        # print(f"  planform flat MAC:  {canopy.MAC:>6.3f}")
        print(f"  projected span: {canopy.b:>6.3f}   [{b:>6.3f}]")
        print(f"  projected area: {canopy.S:>6.3f}   [{S:>6.3f}]")
        print(f"  projected AR:   {canopy.AR:>6.3f}   [{AR:>6.3f}]")
        print()

        # Reminder that I'm not accounting for mass from things like the lines,
        # internal vribs, internal horizontal straps, caribiners, etc.
        wmp = wing.mass_properties(rho_air=1.225)
        print("Wing inertia:              [Target]")
        print(f"  solid mass:     {wmp['m_s']:>6.3f}   [{m_s:>6.3f}]")
        print()

        print("Finished building the glider.\n")

    return wing
