Does my model demonstrate "control reversal" for small brake deflections?
 * aka, "roll steering" instead of "skid steering"
 * Tends to happen for flatter wings and/or as the angle of incidence becomes more negative (ie, the equilibrium `theta`, in my case)
    * It would be interesting to have a flat wing with the risers placed forward of the c4 (thus a very negative `theta_eq` to observe this behavior)
 * ref: "Apsects of control for a parafoil and payload system", Slegers and Costello, 2003

Would be really cool to reproduce the wing used in "Wind Tunnel Investigation of a Rigid Paraglider Reference Wing" (Belloc, 2015)
 * Includes full specs of the wing geometry?
 * **Uses a NACA 23015, because "it is representative" of the design tradeoffs for paragliders!!**
    * TODO: implement a NACA5!!
    * Oh snap, **the 23015 is so so different from a 4418...**, might fix my Hook 3 performance issues (the 23015 has a much lower L/D so I might should overestimating my glide ratios!)


If I'm using a UnivariateSpline for the airfoil coefs, I need to handle "out of bounds" better
 * Catch ValueError and return `nan`?



# General
 * Figure out why the polar curve look so terrible for small applications of brakes!!
 * Phillips should check for zero `Cl_alpha`
 * Design the "query control points, compute wind vectors, query dynamics" sequence and API
 * Review parameter naming conventions (like `kappa_S`, wtf is that?)
 * Fix the naming of the central chord `c0` versus the position of the leading edge `c0`

 * I might want to use Phillips2d for Gamma proposals, but first I need to add the viscous effects (Cd)
 * Include the weight of the wing when calculating the forces?

 * Review the wing performance under speedbar
   * Right now, I've capped the minimimum wing alpha_eq to avoid super gnarly results, but this is clearly **WRONG**
   * Test without the fixed bounds, and plot the polar curve with a large number of sample points
 * Still issues with the polar curves
   * My "Hook3-ish" min-sink is much too low; should be 1.1m/s (I should start by including the weight of the wing)
   * My "Hook3-ish" max speed is too low (should be 54kmh)
   * My "Hook3-ish" creates bad `alpha_eq` for small application of brakes; need to plot polar curves with a large number of points to detect this


 * Review the API for consistency
   * Do the wing+glider functions always parametrize like (<wing stuff>, <environment stuff>)? Can they?

## Low priority
 * Review function parameters for compatibility with either scalar or array arguments
 * Investigate Numba compatibility; where it's needed, and how to enable it
 * Investigate applying the PFD stability analyses in the context of my refactored design
    * for example, longitudinal static stability as a function of speed bar (and thus, {d_cg, h_cg})
 * The ParafoilSections don't broadcast `s` correctly
    * This is impeding my goal of broadcasting `upper/lower_surface`


# Airfoil
 * The AirfoilCoefficients should be responsible for all terms?
    * Right now, Phillips' method augments the raw coefs with terms for the airfoil opening, surface effects, etc
    * Not sure how to best handle that, since the air intakes seem to be more closely related to the 3d parafoil, moreso than the airfoil...
 * The AirfoilGeometry assumes that the LE is what defines the "upper" and "lower" surface
    * This assumption is almost surely incorrect in terms of parafoil construction. The heavier upper surface likely wraps beyond `d_LE` and down around the nose to the air intakes.
 * `AirfoilCoefficients` should support automatic broadcasting of `alpha` and `delta`
    * eg, suppose `alpha` is an array and `delta` is a scalar
 * `NACA4` doesn't work with symmetric airfoils (crashes!)

# Parafoil

## Parafoil Sections
(This is a long term goal.)

In theory, a designer may want a spanwise variation in the airfoil. This requires varying both the coefficients (for performance) and the geometry (for inertia calculations).

A `ParafoilSections` class should generate those Airfoils, and provide the Airfoil interface.
 * eg, you can do `sections(s).Cl(alpha, delta)` and it will return an array of the coefficients for each section in `s`
 * This is complicated for several reasons:
    1. How do you generate realistic coefficients?
    2. How do you generate realistic geometries?
    3. How does `sections` provide access to the Airfoil API? (it's a smart container, essentially)


## Coefficient Estimation
 * **Important**: I really need a THOROUGH review of the Phillips implementation
 * Phillips can have convergence issues; need a strategy to detect them
 * Phillips can't handle `Cl_alpha = 0` conditions; needs to detect this at a minimum, and better yet use "Picard iterations" to solve the system
 * Double check the drag correction terms for viscous effects
    * Should the section drag really include the local sideslip airspeed for calculating their drag?
    * Or should they "discard" the sideway velocity and calculate using only the chordwise+normal velocities?
    * Same goes for the direction of the drag vectors.
 * Why does Phillip's seem to be so sensitive to `sweepMax`? Needs testing
 * I could really use better Gamma proposals; they are super ugly right now
    * Is Phillips2d a good predictor? Maybe convert Phillip's velocities into <Gamma> and scale it?


# BrakeGeometry
 * Need a proper BrakeGeometry; the `Cubic` seems weird
    * Create a more realistic brake distribution based on line angles?
 * Nice to have: automatically compute an upper bound for `BrakeGeometry.delta_max` based on the maximum supported by the Airfoils


# ParagliderWing
 * Moment of inertia calculations
    1. Compute the area and volume moments for the Parafoil (a purely geometric entity)
    2. ParagliderWing applies the parallel axis theorem to the Parafoil moment of inertia
    3. Use wing material densities and air density to compute the final moment of inertia

 * Paraglider should be responsible for weight shifting
    * The wing doesn't care about the glider cm, only the changes to the riser positions!

## Wing mass moment
Technically, the mass of the wing materials add an extra moment. Unfortunately, this means that you can't calculate `alpha_eq` by itself anymore, since the moment created by the mass will depend on the orientation of the wing, not just the angle of attack. Thus, you have to solve for `alpha_eq` and `Theta_eq` simultaneously; you must find the pair such that there exists some `V_eq` that causes the net moments and forces to go to zero.

Thankfully, during normal equilibrium conditions the weight vector the wing doesn't have a large moment arm about the glider center of mass, so this contribution is probably (?) negligible.

# vim: set nospell:
