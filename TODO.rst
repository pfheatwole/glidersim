* I claim that FoilGeometry is defined as having the central chord leading
  edge at `x = 0` and that the central chord lies in the xy-plane, **by
  definition**, but I never enforce that. I do shift the leading edge to the
  origin, but I don't derotate the global wing.

  I guess it'd be good enough to just require that `torsion(s=0) = 0`, but
  I guess I could also just compute `torsion(s=0)` and subtract that from all
  torsions, thus "centering" the twist in the same manner as the origin.


Packaging
=========

* Add a `README.rst`

* Fill out `setup.cfg` more thoroughly

  * **Add a license** (https://choosealicense.com/)

  * Add `author` and `author_email`? Required as a pair. Pity the email
    address becomes public.

  * Verify the dependencies

* Replace `setup.cfg` with `pyproject.toml`? (ie, use `poetry`?)


General
=======

* Define an `njit` wrapper that replaces `njit` with a noop

* Lots of missing/incomplete docstrings, and particularly for modules. (#NEXT)

* Verify function docstrings match the signatures

* How much do 'C' vs 'F' arrays affect dot product performance? Enough for
  Numba to warn me about it, at least. (see `test_sim.py` using `quaternion`)

* Should docstring types be "array of" or "ndarray of"? I lean towards
  "array", but would it be better to use the canonical name so sphinx can link
  to the numpy datatype documentation?

* Review the API for consistency

  * Do the wing+glider functions always parametrize like (<wing stuff>,
    <environment stuff>)? Can they?


Plots
-----

* I'd sure like it if the 3D plots could use a figsize that wasn't square (it
  wastes too much space). I think it's because `_set_axes_equal` uses
  a radius, and all axes must contain that sphere. **Can you keep the equal
  scaling property with different axes lengths?**


Low priority
------------

* Review function parameters for compatibility with either scalar or array
  arguments. (Broadcasting is useful together with `np.meshgrid`, etc.)

* Investigate applying the "Paraglider Flight Dynamics" (Bendetti, 2012)
  stability analyses in the context of my refactored design (eg, longitudinal
  static stability as a function of speed bar, and thus as a function of
  {d_cg, h_cg})

* Do a performance comparison between `cross3` and the `np.cross`
  implementation added to Numba `v0.46`. As of 2019-12-16, that function is
  roughly 60% slower on small arrays, and nearly 8x slower on `10000x1000x3`
  arrays.


Airfoil
=======

* Add some literature references. For NACA airfoils, there are:

  * Abbott, "Theory of Wing Sections, Sec. 6

  * https://www.hq.nasa.gov/office/aero/docs/rpt460/index.htm

  * The XFOIL source code?

* Should `AirfoilGeometry` be a regular base class? You give it a set of
  points, it provides default machinery from there? Functions like "thickness"
  seem like general enough concepts that a set of general equations should be
  sufficient; subclasses (like NACA) can override those general versions with
  custom versions if they want.

  More to the point, should `NACA` be a separate class? Or is it really
  just a generator of the `points` being passed to `AirfoilGeometry`?

* **HIGH PRIORITY**: Figure out why the polar curve look so terrible for small
  applications of brakes!!

  I really REALLY don't trust the XFOIL output (user error seems very likely).
  It is extremely sensitive to tiny changes to the number of points, the point
  distribution, and *super* sensitve to trailing edge gaps. Just creating
  a nominal 23015 with the builtin generator then removing the tiny TE gap
  causes the pitching moment in particular to change dramatically. For now I'm
  focusing on the getting the wing calculations correct given the airfoil
  data, but the sample airfoil data I'm using seems totally untrustworthy.
  (#NEXT)

* The NACA airfoils have the `convention` parameter, but the `AirfoilGeometry`
  superclass does not, yet the `AirfoilGeometry.thickness` docstring
  references the convention.

* Implement generalized `camber_curve` and `thickness` thickness functions in
  `AirfoilGeometry`. Their definitions depend on the `convention`: "American"
  defines "thickness is perpendicular to the camber line", British defines
  "thickness is perpendicular to the chord". (This is the same issue as in the
  definitions of the NACA equations.)

* Should `AirfoilGeometry` provide an `acs2frd` conversion method? Or include
  that as a boolean parameter to `AirfoilGeometry.mass_properties` or similar?

* Add a note somewhere about the "American" convention having stability issues
  with some codes (I forget now which! Check the NACA5 range.)

* NACA code `7199` really throws my "derotate and normalize" code for a loop.
  **Needs more testing.**


Low priority
------------

* `AirfoilCoefficients` should support automatic broadcasting of `alpha` and
  `delta`. (For example, suppose `alpha` is an array and `delta` is a scalar.)

* Why does `s` go clockwise? Why not just keep the counter-clockwise
  convention? After all, the z-axis of the parafoil is positive down anyway...

* AirfoilGeometry is for a single airfoil, but AirfoilCoefficients support
  `delta` for braking (ie, multiple airfoils). Among other things, this
  asymmetry means you can't compute the inertia matrices for braking wings
  (heck, you don't even have their geometry, right?)

* Should I provide `s2d` and `d2s` functions? Suppose a user wanted to step
  along the curve in equal steps; they'd need to convert those equally spaced
  `d` into `s`, which is weird since the upper and lower surfaces use
  different spacings for `s`...

* If I'm using a UnivariateSpline for the airfoil coefficients, I need to
  handle "out of bounds" better. Catch `ValueError` and return `nan`?


Parafoil
========

Geometry
--------

* The `ParafoilGeometry` docstrings are really hard to follow. The parameter
  names (`x`, `r_yz`, etc) are not very helpful.

* Should the `ParafoilGeometry.r_x` etc be private members (`_r_x`)?

* `ParafoilGeometry.mass_properties` does not pass `s_upper` and `s_lower` to
  `Airfoil.mass_properties`: the upper/lower surface inertias are likely
  overestimated/underestimated (a little bit).

* Fix the inertia calculations: right now it places all the segment mass on the
  airfoil bisecting the center of the segment. The code doesn't spread the mass
  out along the segment span, so it underestimates `I_xx` and `I_zz` by
  a factor of ``\int{y^2 dm}``. (Verify this.) Doesn't make a big difference in
  practice, but still: it's wrong.

* Add an example for exporting the triangle mesh to `vtkPolyData` (or whatever
  the correct data structure would be). Would make it easier to interface with
  OpenFOAM (you can import the mesh into Blender and export an STL, but I'm
  sure there are easier ways to go about it).


ParafoilSections
^^^^^^^^^^^^^^^^

Low priority, long term goal: a new class to encapsulate spanwise variation in
wing sections (airfoil geometry, airfoil coefficients, intakes, etc)

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

  * All users that call `Phillips.__call__` should be exception-aware

* Double check the drag correction terms for viscous effects

  * Should the section drag really include the local sideslip airspeed for
    calculating their drag?

  * Or should they "discard" the sideway velocity and calculate using only the
    chordwise+normal velocities?

  * Same goes for the direction of the drag vectors.

* Does Phillips' method detect significant differences in performance if the
  quarter-chord lies in a plane or not? The lobe makes it curve backwards at
  the tips, and I'm curious if that has performance considerations. You could
  theoretically define a function that "undoes" the curvature induced by the
  lobe.


Phillips
^^^^^^^^

* In `Phillips` I have a fixme about using the "characteristic chord", but
  right now I'm using the section area (`dA`). If I switch it to `c_avg`, the
  `CL vs CD` curve looks MUCH more like what's in the Belloc paper, but
  the other curves go to pot. **(#NEXT)**

* Refactor the drag coefficient correction terms (skin friction, etc) outside
  Phillips (#NEXT)

  * This belongs with the parafoil model; Phillips shouldn't care. Maybe part
    of the tentative ParafoilSections design?

* My Jacobian calculations seem to be broken again; at least, the
  finite-difference approximation disagrees with the analytical version. And
  the equations for the `J` terms don't match Hunsaker; why not?

* Phillips should check for zero `Cl_alpha`. What should it do if it does? Can
  it gracefully fail over to fixed-point iterations? Should it return a mask
  of which sections are experiencing stall conditions? Does it matter if XFOIL
  is unreliable post-stall anyway?

* Refactor Phillips outside `foil.py`?

* Why does Phillip's seem to be so sensitive to `sweepMax`? Needs testing

* Review the Gamma proposals.

* I compute the complete Jacobian, but MINPACK's documentation for `hybrj`
  says it should be the `Q` from a `QR` factorization? I can't say
  I understand this.

* The Jacobian uses the smoothed `Cl_alpha`, which technically will not match
  the finite-difference of the raw `Cl`. Should I smooth the `Cl` and replace
  that as well, so they match?

* Profile and optimize

  * For example, ``python -m cProfile -o belloc.prof belloc.py``, then ``>>>
    p = pstats.Stats('belloc.prof'); p.sort_stats('cumtime').print_stats(50)``

  * Do the matrices used in the `einsum` calls have the optimal in-memory
    layout? Consider the access patterns and verify they are contiguous in the
    correct dimensions (ie, `C` vs `F` contiguous; see ``ndarray.flags``)

* Phillips' could always use more testing against XFLR5 or similar. I don't
  have geometry export yet, but simple flat wings should be good for comparing
  my Phillips implementation against the VLM methods in XFLR5.


BrakeGeometry
=============

* Need a proper BrakeGeometry; the `Cubic` seems weird. (This will probably
  have to wait until I create a more realistic brake distribution based on
  line angles.)

* Nice to have: automatically compute an upper bound for
  `BrakeGeometry.delta_max` based on the maximum supported by the Airfoils.
  (Setting ``delta_max`` to a magic number is *awful*.)


ParagliderWing
==============

* Review parameter naming conventions (like `kappa_a`). Why "kappa"?

* `d_riser` and `z_riser` are different units, which is odd. Almost everything
  is proportional to `b_flat`, but `z_riser` is a concrete unit?

* ParagliderWing owns the force estimator for the Parafoil, but not for the
  harness...

* *Design* the "query control points, compute wind vectors, query dynamics"
  sequence and API

* Paraglider should be responsible for weight shifting?

  * The wing doesn't care about the glider cm, only the changes to the riser
    positions. However, **that would change if the lobe supports
    deformations** in response to weight shift.


Wing inertia
------------

I'm using a naive isotropic model for wing inertia (the standard definition),
but because the surrounding air mass is in motion it adds an additional
damping effect, which adds to the naive inertia. The *effective inertia* is
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


Documentation
=============

* I'm using `sphinx.ext.autosummary`, which uses `autodoc` under the hood.
  A set of Jinja2 templates from
  `<https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary>`_
  control the `autosummary` output. I'd kind of like it if each module would
  list its classes in the contents tree (left hand side of the `readthedocs`
  theme). I tried to achieve that by overriding the `module.rst` template to
  include the ``:toctree:`` directive to the ``.. autosummary::`` that's
  building up the classes in the module, but that makes sphinx angry since it
  generates duplicate stubs for those class definitions.


Testing
=======

* Still issues with the Hook 3 polar curves

  * Min-sink is much too low; should be 1.1m/s (I should start by including
    the weight of the wing)

  * Max speed is too low (should be 54kmh)

  * Is `alpha_eq` accurate when brakes are applied? It'd be fascinating if
    alpha and Theta do actually decrease; I'd have expected Theta to
    *increase*.

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
    method just like I do! I'm guessing my airfoil data is junk.
