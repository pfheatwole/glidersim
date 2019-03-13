PRIORITY: review chreimChangesModernLiftingLine2018

 * Updated version of Phillips method?


General
=======

* Bundle the components (paraglider model, wind models, simulator) into
  a unified package for delivery with my thesis (#NEXT)

* Review the API for consistency

  * Do the wing+glider functions always parametrize like (<wing stuff>,
    <environment stuff>)? Can they?


Low priority
------------

* Review function parameters for compatibility with either scalar or array
  arguments

* Investigate Numba compatibility; where it's needed, and how to enable it

* Investigate applying the PFD stability analyses in the context of my
  refactored design (eg, longitudinal static stability as a function of speed
  bar, and thus as a function of {d_cg, h_cg})


Airfoil
=======

* The AirfoilCoefficients should be responsible for all terms? (#NEXT)

  * Right now, Phillips' method augments the raw coefs with terms for the
    airfoil opening, surface effects, etc

  * Not sure how to best handle that, since the air intakes seem to be more
    closely related to the 3d parafoil, moreso than the airfoil...

* The AirfoilGeometry assumes that the LE is what defines the "upper" and
  "lower" surface

  * This assumption is almost surely incorrect in terms of parafoil
    construction. The heavier upper surface likely wraps beyond `d_LE` and
    down around the nose to the air intakes.

* `AirfoilCoefficients` should support automatic broadcasting of `alpha` and
  `delta`

  * eg, suppose `alpha` is an array and `delta` is a scalar

* `NACA4` doesn't work with symmetric airfoils (crashes!)

* If I'm using a UnivariateSpline for the airfoil coefs, I need to handle "out
  of bounds" better. Catch ValueError and return `nan`?


Sample Airfoils
---------------

* Figure out why the polar curve look so terrible for small applications of
  brakes!!

**HIGH PRIORITY**: I really REALLY don't trust the XFOIL output. It is
extremely sensitive to tiny changes to the number of points, the point
distribution, and *super* sensitve to trailing edge gaps. Just creating
a nominal 23015 with the builtin generator then removing the tiny TE gap
causes the pitching moment in particular to change dramatically. For now I'm
focusing on the getting the wing calculations correct given the airfoil data,
but the sample airfoil data I'm using seems totally untrustworthy.  (#NEXT)


Parafoil
========

* Redesign the `ParafoilGeometry` functions fx/fy/fz/c0/c4 (#NEXT)

  * They're all basically doing the same thing. Like c0 and c4: you should
    have a general function that gives the coordinates anywhere along the
    chord for any given spanwise station, like
    `ParafoilGeometry.chord_position(s, pc)`, where `s` is -1..1 for the
    normalized spanwise coordinates, and `pc` is the fraction of the chord (so
    pc=0 for the leading edge, pc=0.25 for the quarter chord, and pc=1 for the
    trailing edge).

  * Should I do a similar treatment for the upper and lower surfaces?
    Shouldn't be difficult. For the airfoils I've defined 0..1 for the upper
    surface and 0..-1 for the lower surface, so the API would match very
    closely. Seems convenient since it'd let you draw arbitrary curves over
    the surface of the wing; might be useful for visualization purposes.

* Fix the naming of the central chord `c0` versus the position of the leading
  edge `c0`


Coefficient Estimation
----------------------

* Double check the drag correction terms for viscous effects

  * Should the section drag really include the local sideslip airspeed for
    calculating their drag?

  * Or should they "discard" the sideway velocity and calculate using only the
    chordwise+normal velocities?

  * Same goes for the direction of the drag vectors.


Phillips
^^^^^^^^

* Phillips should check for zero `Cl_alpha`

* Refactor Phillips outside `Parafoil.py` (#NEXT)

  * This is a general lifting-line method, isn't it?

* Phillips is unreliable post-stall:

  * The Jacobian explodes near `Cl_alpha = 0`

  * Phillips recommends using "Picard iterations" to solve the system

  * **WARNING**: I doubt the XFOIL data is suitable post stall

* Refactor the drag coefficient correction terms (skin friction, etc) outside
  Phillips (#NEXT)

  * This belongs with the aircraft model; Phillips shouldn't care

* Why does Phillip's seem to be so sensitive to `sweepMax`? Needs testing

* I could really use better Gamma proposals; they are super ugly right now

  * Is Phillips2d a good predictor? Maybe convert Phillip's velocities into
    <Gamma> and scale it?

* I compute the complete Jacobian, but MINPACK's documentation for `hybrj`
  says it should be the `Q` from a `QR` factorization?

* The Jacobian uses the smoothed `Cl_alpha`, which technically will not match
  the finite-difference of the raw `Cl`. Should I smooth the `Cl`, and
  replace that as well?

* Profile and optimize

  * `python -m cProfile -o belloc.prof belloc.py`, then `>>>
    p = pstats.Stats('belloc.prof');
    p.sort_stats('cumtime').print_stats(50)`

  * The `einsum` are not optimized by default; also, can precompute the
    optimal contraction "path" with `einsum_path`


ParafoilSections (Low priority)
-------------------------------

(This is a long term goal.)

In theory, a designer may want a spanwise variation in the airfoil. This
requires varying both the coefficients (for performance) and the geometry (for
inertia calculations).

A `ParafoilSections` class should generate those Airfoils, and provide the
Airfoil interface.

* eg, you can do `sections(s).Cl(alpha, delta)` and it will return an array of
  the coefficients for each section in `s`

* This is complicated for several reasons:

  1. How do you generate realistic coefficients?

  2. How do you generate realistic geometries?

  3. How does `sections` provide access to the Airfoil API? (it's a smart
     container, essentially)


BrakeGeometry
=============

* Need a proper BrakeGeometry; the `Cubic` seems weird

  * Create a more realistic brake distribution based on line angles?

* Nice to have: automatically compute an upper bound for
  `BrakeGeometry.delta_max` based on the maximum supported by the Airfoils

ParagliderWing
==============

 * `equilibrium_alpha` uses `minimize_scalar` as a root finder? (#NEXT)

 * Review parameter naming conventions (like `kappa_S`, wtf is that?)

 * Design the "query control points, compute wind vectors, query dynamics"
   sequence and API

 * Paraglider should be responsible for weight shifting?

    * The wing doesn't care about the glider cm, only the changes to the riser
      positions!


Wing inertia
------------

I'm using a naive isotropic model for wing inertia (the standard definition).
But, because the surrounding air mass is in motion, it adds an additional
damping effect, which combines with the true inertia. The effective inertia is
then the result of the **apparent mass**. There are several definitions, like
apparent mass, real mass, and solid mass; see "Apparent mass of parafoils with
spanwise camber" (Barrows; 2002) for more information.


Wing mass moment
----------------

Technically, the mass of the wing materials add an extra moment.
Unfortunately, this means that you can't calculate `alpha_eq` by itself
anymore, since the moment created by the mass will depend on the orientation
of the wing, not just the angle of attack. Thus, you have to solve for
`alpha_eq` and `Theta_eq` simultaneously; you must find the pair such that
there exists some `V_eq` that causes the net moments and forces to go to zero.

Thankfully, during normal equilibrium conditions the weight vector the wing
doesn't have a large moment arm about the glider center of mass, so this
contribution is (probably?) negligible.


Simulator
=========

* The simulator needs to understand that Phillips can fail, and
  degrade/terminate gracefully

* The simulator should handle premature termination (`Ctrl-C`)

* Review the GliderSim state definitions (a dictionary? a structured array?)


Scenario Design
---------------

* Design a set of flight scenarios (#NEXT)

  * Demonstrate wing behavior under different wind models and control inputs


Testing
=======

* Review the wing performance under speedbar

  * Right now, I've capped the minimimum wing alpha_eq to avoid super gnarly
    results, but this is clearly **WRONG**

  * Test without the fixed bounds, and plot the polar curve with a large
    number of sample points

* Still issues with the polar curves

  * My "Hook3-ish" min-sink is much too low; should be 1.1m/s (I should start
    by including the weight of the wing)

  * My "Hook3-ish" max speed is too low (should be 54kmh)

  * My "Hook3-ish" creates bad `alpha_eq` for small application of brakes;
    need to plot polar curves with a large number of points to detect this

* Does my model demonstrate "control reversal" for small brake deflections?

  * aka, "roll steering" instead of "skid steering"

  * Tends to happen for flatter wings and/or as the angle of incidence becomes
    more negative (ie, the equilibrium `theta`, in my case)

    * It would be interesting to have a flat wing with the risers placed
      forward of the c4 (thus a very negative `theta_eq` to observe this
      behavior)

  * ref: "Apsects of control for a parafoil and payload system", Slegers and
    Costello, 2003

* Finish reproducing "Wind Tunnel Investigation of a Rigid Paraglider
  Reference Wing" (Belloc, 2015)


# vim: set nospell:
