Replace ``cross3`` with the new numba version (introduced in version 0.46). Do
a basic performance comparison to verify, but it's probably worth it. I'd love
to just ``try`` to jit the functions, and fail gracefully if numba is not
installed. (Maybe define an ``njit`` wrapper that tries to import numba just
once, and defines a noop if that fails.)


Where did I get ``dMed`` and ``dMax`` for my Hook3 specs?



General
=======

* Bundle the components (paraglider model, wind models, simulator) into
  a unified package for delivery with my thesis

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
  edge `c0` (Should be a non-issue once I replace the `c0` function with
  a generalized "position on the chord" function.)


Geometry
--------


Generalize the chord position functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

I'd like to generalize the Parafoil model to return any position on the chord
for any spanwise station. Right now I have both ``ParafoilGeometry.c0`` and
``ParafoilGeometry.c4``, which is not only needlessly limiting (to specific
points on the chord), but more importantly they are confusing for several
reason:

* `c0` is basically `leading edge + c*0` whereas `c4` is really `leading edge
  + c/4`. (Multiplication versus division.)

* I use `c0` for the total length of the central chord of the planform, not
  a position.

  * Sidenote: **replace `planform.c0` with `planform.c_0`, for consistency**


What should the new signature for chord positions look like?

.. code::

   fun(s, d):
      s : float
         Planform spanwise position, where -1 <= s <= 1
      pc : float
         Chordwise position, where 0 <= pc < = 1

Is this consistent with my ParagliderWing terminology?

* eg, there I'm using `d` to indicate the chordwise position of the
  perpendicular line passing through the cg

  * Is `d` the best variable name for that parameter in the first place?

  * Seems like `f(s, pc)` is more intuitive: "spanwise, chordwise position".
    Could parametrize `pc_cg, z_cg` for chordwise+height

Also remember, the user may want this function for either the ParafoilGeometry
or the flat ParafoilPlanform. They both provide fx+fy

These changes should simplify the API by removing the ambiguous notation
(c0/c4), as well as making it easier to implement other coefficient estimation
methods that require chord points off the c/4 line (eg, the Pistolesi boundary
condition).


ParafoilSections (Low priority)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Coefficient Estimation
----------------------

* Design review how the coefficient estimator signals non-convergence (#NEXT)

  * Right now Phillips' just sets the Gamma to NaN

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

  * This is a general lifting-line method, not just for parafoils. Also,
    factoring it is the first step to generalizing for different estimation
    methods (Phillips, Hunsaker, Chreim, etc)

* Phillips is unreliable post-stall:

  * The Jacobian explodes near `Cl_alpha = 0`

  * Phillips recommends using "Picard iterations" to solve the system

  * **WARNING**: I doubt the XFOIL data is suitable post stall anyway

* Refactor the drag coefficient correction terms (skin friction, etc) outside
  Phillips (#NEXT)

  * This belongs with the parafoil model; Phillips shouldn't care. Maybe part
    of the tentative ParafoilSections design?

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

* Compare my Phillips implementation against some more straightforward wings,
  such as those in `chreimViscousEffectsAssessment2017`. Generating straight,
  untapered wings should be pretty straightforward using my geometry
  definitions.


BrakeGeometry
=============

* Need a proper BrakeGeometry; the `Cubic` seems weird

  * Create a more realistic brake distribution based on line angles?

* Nice to have: automatically compute an upper bound for
  `BrakeGeometry.delta_max` based on the maximum supported by the Airfoils


ParagliderWing
==============

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


Paraglider
==========

* Review the difference between:

  1. Assuming the harness is rigid (if it's not placed at the risers, it will
     introduce an unnatural pitching moment)

  2. Assuming the center of mass is at the origin

* The call signature for ``forces_and_moments`` has too many parameters! It's
  weird to pass in `xyz` since it's redundant with `delta_s`. Is that
  confusion-inducing redundancy worth saving the little bit of time to
  recompute those `xyz`?

* Should the glider really be returning the forces and moments? Seems like
  it'd be smart to return the accelerations (both translational and
  rotational). This also factors into how you compute the inertia: real mass
  versus apparent mass.


Simulator
=========

* The simulator needs to understand that Phillips can fail, and
  degrade/terminate gracefully. (Depends on how the ForceEstimators signal
  failures; that design is a WIP.)

* Design review support for early terminations (`Ctrl-C`) of fixed-length
  simulations (eg, "run for 120sec").

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

  * Why don't my results match as well as in
    `kulhanek2019IdentificationDegradationAerodynamic`? They use Phillips'
    method just like I do!


# vim: set nospell:
