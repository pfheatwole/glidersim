* I'm not crazy about the name `forces_and_moments` if they don't include
  weight. Should be `aerodynamic_forces_and_moments`, but that's really long.

  Or, **should the `ParagliderWing` and `Harness` be responsible for computing
  their own weight forces?**

* Note to self: a paraglider wing is specific type of 'parafoil' that includes
  the normal canopy + lines, but also a particular control input scheme. You
  can't use a general `Parafoil` class for all parafoils since the controls
  may differ. Same thing for paragliders: they are a particular
  parafoil/payload system, where the parafoil is a paraglider wing, and the
  payload is a paragliding harness.

* According to Nicolaides, "the Parafoil is both an airplane or glider and
  also a parachute or decelerator". Good point

* In `quaternion` I mention "yaw-pitch-role" Euler angles, but all parameters
  with independent inputs are `[phi, theta, gamma]` (roll-pitch-yaw). Is this
  usage consistent? I'm still not comfortable with the terminology, but
  I think it's okay? I suspect that the actual phi-theta-gamma values will
  depend on the order of their application, so the *values* phi-theta-gamma
  are the roll-pitch-yaw angles that would produce the desired rotation when
  applied in a yaw-pitch-role sequence. Probably just need a better docstring.


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

* I'm using `breakpoint()` a few places, which wasn't added until Python 3.7.
  Should I set that as a hard dependency?

* Define an `njit` wrapper that replaces `njit` with a noop

* Lots of missing/incomplete docstrings, and particularly for modules.

* Verify function docstrings match the signatures

* How much do 'C' vs 'F' arrays affect dot product performance? Enough for
  Numba to warn me about it, at least. (see `quaternion`)

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

Geometry
--------

* Write an `AirfoilGeometry` interpolator. Takes two geometries, and returns
  the interpolated surface points.

  **Does this make sense as a standalone thing?** It's so simple, it almost
  seems like overkill to make it it's own class. Might be preferable to have
  a single class that interpolates both the geometry and the coefficients?

* Implement **accurate** `camber_curve` and `thickness` estimators.

  If I'm going to scale airfoils by changing their thickness, then I need the
  correct camber and thickness functions. If I don't, then there will be weird
  disjoint surfaces at small thickness changes (since you'll move from the true
  surface to the version of that surface produced by estimates of its thickness
  and camber).

* Write a basic "trailing edge deflection" routine for airfoils. Doesn't have
  to be physically accurate for now, just need to establish the API.

* Add some literature references. For NACA airfoils, there are:

  * Abbott, "Theory of Wing Sections, Sec. 6

  * https://www.hq.nasa.gov/office/aero/docs/rpt460/index.htm

  * The XFOIL source code?

* Verify the polar curves, especially for curved airfoils.

  The airfoil data is still a bit of a mystery to me. I don't trust the XFOIL
  output (at least not my use of it). It is extremely sensitive to tiny
  changes in the number of points, the point distribution, and especially the
  trailing edge gaps (which look like they should produce negligible
  changes?). Just creating a nominal 23015 with the builtin generator then
  removing the tiny TE gap causes the pitching moment in particular to change
  dramatically.

* Should `AirfoilGeometry` provide an `acs2frd` conversion method? Or include
  that as a boolean parameter to `AirfoilGeometry.mass_properties` or similar?


Coefficients
------------

* It might be interesting if `GridCoefficients` supported CSV that lack `Re`.
  Wouldn't make for good analysis, but would be interesting for demonstrating
  the effect of ignoring Reynolds numbers.

* In `XFLR5Coefficients`, the `LinearNDInterpolator` should be able to use
  `scale=True` instead of the `Re = Re / 1e6` in the coefficients functions,
  but for some reason it doesn't work. Worth investigating?

* In `XFLR5Coefficients`, I could support XFOIL polars as well, but I'd need to
  read the columns differently. Easy way to read the headers is with `names
  = np.loadtxt(<filename>, skiprows=10, max_rows=1, dtype=str)`. I haven't
  tested it with XFOIL polars though, might be missing some nuance.


Low priority
------------

* Let NACA use it's actual explicit curve definitions. I'll have to compute `x`
  as a function of arc-lengths, but beyond that use the actual functions
  instead of relying on interpolated estimates. The annoying part will be
  calculating the `surface_curve_normal` and `surface_curve_tangent` functions.

* Rewrite `AirfoilGeometry.mass_properties` to handle airfoils that aren't
  simply `y_upper - y_lower` type surfaces. Not a high priority for now since
  I'm simple shapes with derotation. (Then again, I'm not sure this function
  will continue making sense later on (probably better ways compute the area
  and volume inertias, but beware this issue for now.)

* Rename airfoil's `surface` to `profile`? "Surface" suggests 2D.

* Consider Gaussian quadratures or other more efficient arc-length methods?

* `AirfoilCoefficients` should support automatic broadcasting of `alpha` and
  `delta`. (For example, suppose `alpha` is an array and `delta` is a scalar.)

* Why does `s` go clockwise? Why not just keep the counter-clockwise
  convention? I do like that there is a sort of right-hand rule that points in
  the +y direction though.

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

* Add Joukowski airfoil builders? Those are typically defined in terms of
  their surface coordinates, not mean camber and thickness curves. Neat
  airfoils though, conceptually. Very elegant.



Chord Surface
=============

* Is it bad for to use `r_x` and `r_yz` for the ratios when `r_A2B` are
  vectors? A bit of an overlap, but doesn't seem like a big conflict.

* Should `elliptical_lobe`: accept the alternative pair `{b/b_flat,
  max_anhedral}`? You often know b/b_flat from specs, and `max_anhedral` is
  easy to approximate from pictures.


Parafoil
========

Geometry
--------

* Review the air `intakes` design

  Should they be removed from `SimpleFoil`? If `surface_xyz` accepts the
  `surface` parameter, then you'll need *some* mapping between surface and
  airfoil coordinates.
  
  Also, reconsider the name "intakes": this concept doesn't *require* that
  `s_upper != s_lower`; maybe a user has other reasons to shifting the
  upper/lower surface boundary away from the leading edge. Might even be
  useful for **single surface designs**, that discard the lower portion of the
  majority of the section profiles.

* The `ChordSurface` requires the values to be proportional to `b_flat == 2`?
  **What if you don't know `b_flat`? Do you need to compute the total length
  of `yz` and re-normalize to that?** (I think I'm missing something here...
  As long as everything is proportional, who cares? I'll need to look for
  anywhere that uses `s` to stand in for `y`, but other than that, who cares?
  May want to introduce an scaling value as a convenience for the user
  though.)

* Define the fundamental `FoilGeometry` spec

  What are the essential needs of users like `SimpleFoil`, `Parafoil`, etc? At
  least: `section_orientation, chord_length, chord_xyz, surface_xyz`. Anything
  else? I think the least constraining view is "profiles as a function of
  section index positioned along some line". 



Lower priority
^^^^^^^^^^^^^^

* I claim that `FoilGeometry` is defined as having the central chord leading
  edge at `x = 0` and that the central chord lies in the xy-plane, **by
  definition**, but I never enforce that. I do shift the leading edge to the
  origin, but I don't derotate the global wing.

  I guess it'd be good enough to just require that `torsion(s=0) = 0`, but
  I guess I could also just compute `torsion(s=0)` and subtract that from all
  torsions, thus "centering" the twist in the same manner as the origin.
  
* Move `InterpolatedLobe` from `belloc.py` into `foil.py` and modify it to use
  intelligent resampling (near the given points, not just a blind resample).

* Review the API: accept any of `{b, b_flat, S, S_flat}` as scaling factors


Low Priority
^^^^^^^^^^^^

* Use a library like `https://github.com/orbingol/NURBS-Python` to export STL,
  NURBS, etc?

* Add an example for exporting the triangle mesh to `vtkPolyData` (or whatever
  the correct data structure would be). Would make it easier to interface with
  OpenFOAM (you can import the mesh into Blender and export an STL, but I'm
  sure there are easier ways to go about it, like `NURBS-Python`).

* Is *wetted area* same thing as total surface area? Also see *wetted aspect
  ratio*.

* Is the "mean aerodynamic chord" a useful concept for arched wings?

* Should the "projected surface area" methods take pitch angle as a parameter?

  I'm not sure what most paraglider wing manufacturers use for the projected
  area. My definitions requires that the central chord is parallel to the
  xy-plane, but I imagine some manufacturers would use the equilibrium angle
  of the wing. It's more in-line with what you'd use for classical aerodynamic
  analysis, and it's essential constant regardless of load.

  For my hook3 approximation, `Theta_eq = 3`. Rotating the foil before
  projecting changed `S` by `0.15%`, so it's not a big deal.


Inertia
^^^^^^^

* Should I rewrite the `mass_properties` to use the triangle mesh? It would
  make computing the surface areas more straightforward, but I'm not sure
  about the internal volumes. I suspect voxels may provide the solution, but
  I haven't researched it yet.

* `FoilGeometry.mass_properties` does not pass `sa_upper` and `sa_lower` to
  `Airfoil.mass_properties`: the upper/lower surface inertias are likely
  overestimated/underestimated (a little bit). (Using a mesh for the areas
  would fix this nicely.)

* Fix the inertia calculations: right now it places all the segment mass on the
  airfoil bisecting the center of the segment. The code doesn't spread the mass
  out along the segment span, so it underestimates `I_xx` and `I_zz` by
  a factor of ``\int{y^2 dm}``. (Verify this.) Doesn't make a big difference in
  practice, but still: it's wrong.


Meshes
^^^^^^

* Refactor the "mesh" functions to take the vertices as inputs.
  
  This would allow the user to generate a mesh over a subset of the foil, and
  (more importantly) allow me to generate a mesh over a single cell (which you
  can then use to compute the surface area.

* Rewrite the vertex generator functions to take `s` and `sa` as parameters.
  
  This would enable generating a mesh over the surfaces of individual cells
  (should work with inflated or deflated cells) and compute their surface area.
  (The surface area of a cell could be useful for estimating the inflated cell
  surfaces.)

* Write a function to compute the surface area of a mesh

  Not hard: `.5 * cross(AB, AC)` or some such, right?

  Would allow me to compute the `thickness_ratio` distribution (for the
  inflated cells) that would maintain a constant surface area.


ParafoilSections
^^^^^^^^^^^^^^^^

* Write a function that can return inflated profiles between two ribs.

  Use the logic from `ribs.py` and assume some `thickness_ratio`; don't worry
  about getting the areas correct for now. After that's working, estimate the
  `thickness_ratio` using meshes?

* Use the "mesh to cell surface area" function to compute the `thickness_ratio`
  that would maintain a constant surface area for the inflated and deflated
  cell surfaces.

  Verify: if the upper surfaces have the same area, do the lower surfaces also
  have the same area? Multiplying the thickness by a constant seems like it
  should be a linear function, so I *think* the lower and upper surfaces
  should both be correct, but it's worth checking.


* Write functions that compute points on the chords and surfaces of sections
  from inflated or deflated cells.

  Right now, you just flatten the chord surface and that the "flattened"
  position, but you don't just flatten an airfoil, you deflate it, which means
  the cells become wider. I think the `x` and `z` coordinates remain
  unchanged, but the `y` coordinates must increase (since the cell widths
  increase).

* Plot an inflated cell.

* Review options for adding section-wise adjustments to coefficients.

  Example: air intake drag.
  
  I'd prefer to keep adjustments independent of the foil geometry, but that
  doesn't mean the foil geometry can't *provide* the adjustments. You'll have
  to call `ParafoilSections` or whatever to get the coefficients; it can add
  the extra terms when it returns the values.

  My current thinking is that you'll specify ribs, and `InterpolatedAirfoil`
  for each rib (that provide the geometries+coefficients over the range of
  deltas), then a `SectionInterpolator` or something will interpolate the
  values of the two `InterpolatedAirfoils` at each rib. The
  `SectionInterpolator` will need to provide the coefficients for any given
  section index, so you can give it extra functions (also as functions of the
  section indices) that it can layer on top. For example, for air intakes, you
  could have a function that converts the intake size into extra drag.


Deformations
^^^^^^^^^^^^

* To warp the trailing edge, could you warp the mean camber line instead of
  the surfaces themselves, then constrain to maintain constant curve length?

* Starting with the `ChordSurface`, how hard would it be to warp the central
  sections to produce a "weight shift" effect?

* Is it a fools errand to support lifting-line methods in the presence of
  deformations? Cell billowing, weight shift, trailing edge braking: they all
  produce deformed profiles, adding many dimensions to the coefficients table.



Coefficient Estimation
----------------------

* Design review how the coefficient estimator signals non-convergence (#NEXT)

  * All users that call `Phillips.__call__` should be exception-aware

* Double check the drag correction terms for viscous effects

  * Should the section drag really include the local sideslip airspeed for
    calculating their drag? Or should they "discard" the sideways velocity and
    calculate using only the chordwise+normal velocities? [WAIT: doesn't it
    work out that the local velocity has no sideslip? Weird, but I think
    that's the case.] Same goes for the direction of the drag vectors.

* Does Phillips' method detect significant differences in performance if the
  quarter-chord lies in a plane or not? The lobe makes it curve backwards at
  the tips, and I'm curious if that has performance considerations. You could
  theoretically define a function that "undoes" the curvature induced by the
  lobe.


Phillips
^^^^^^^^

* Review the conditions for non-convergence. What are the primary causes, and
  can they be mitigated? Right now, convergence via iteration is uncommon:
  cases either succeed, or they don't.

  At a glance, if `beta = 0`, you don't really need an input reference
  solution; the base case works fine. The reference does improve convergence
  when you get to abnormal situations, like in `belloc` when `beta = 15`.

* **Review the iteration design**: should I be interpolating `Gamma`?

* What are the average number of iterations for convergence? It'd be nice to
  recognize "non-convergence" ASAP.

* Should `V` be greater or smaller than `V_rel`?

* Where did J4 come from in Hunsaker's derivation? It wasn't in Phillip's
  derivation.

* How should I handle a turning wing? (Non-uniform `u_inf`) Right now I just
  use the central `V_rel` for `u_inf` and assume it's the same everywhere.

* **Can I mitigate poor behavior near `Cl_alpha = 0`?** Consider pre-computing
  a function `stall_point(alpha, delta)` that checks where `Cl_alpha` goes to
  zero. The `delta` are fixed during iterations, but if proposals are pushing
  `alpha` beyond that stall point, bad things **will** be happening.

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

* Nice to have: automatically compute an upper bound for
  `BrakeGeometry.delta_max` based on the maximum supported by the Airfoils.
  (Setting ``delta_max`` to a magic number is *awful*.)

* Add support for proper line geometries.

  The `BrakeGeometry` are nothing more than quick-and-dirty hacks that produce
  deflection distributions that you're *assuming* can be produced by a line
  geometry. Checkout `altmann2015FluidStructureInteractionAnalysis` for
  a discussion on "identifying optimal line cascading"


ParagliderWing
==============

* Review parameter naming conventions (like `kappa_a`). Why "kappa"?

* `d_riser` and `z_riser` are different units, which is odd. Almost everything
  is proportional to `b_flat`, but `z_riser` is a concrete unit?

* *Design* the "query control points, compute wind vectors, query dynamics"
  sequence and API

* Paraglider should be responsible for weight shifting?

  * The wing doesn't care about the glider cm, only the changes to the riser
    positions. However, **that would change if the lobe supports
    deformations** in response to weight shift.

* Check if paragliders have aerodynamic centers. See "Aircraft Performance and
  Design" (Anderson; 1999), page 70 (89) for an equation that works **for
  airfoils**. The key requirement is that the foil has linear lift and moment
  curves, in which case the x-coordinate of the aerodynamic center is given by
  the slope of the pitching coefficient divided by the slope of the lift
  coefficient. But **is this accurate for an arched wing?** If so, what is the
  z-component?


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

* The call signature for ``Paraglider.accelerations`` has too many parameters!
  It's weird to pass in `r_CP2R` since it's redundant with `delta_a`. Is
  that confusion-inducing redundancy worth saving the little bit of time to
  recompute those `r_CP2R`?

* I don't like `v_W2b` etc. It's confusing that it's different for each
  control point. Conceptually, it's the local velocity of a parcel of air `W`,
  but the `W` is different for each control point. So it'd probably clean it
  up if I had some other symbol besides `W`; `Wcp` maybe?


Dynamics
--------

* If the center of mass moves (speedbar, relative harness pitch, etc) the
  angular velocity must change in order to conserve angular momentum. Same
  thing for changes to any inertia matrices; consider the angular momentum of
  all components and verify they are being maintained. (Non-rigid-body motion
  is a pain!)

  This may prove tricky. If you know the cm moved a particular way, you can
  compute the angular velocity that would satisfy conservation of angular
  momentum. **But, the `Paraglider` returns accelerations, not net changes
  in velocity; if the speedbar moved the cm over `dt`, who computes that net
  change in angular momentum?** Does rate change of controls need to be part
  of the state? How else do you determine the *change per time* of angular
  momentum in response to control inputs?

  First thing to do is probably to check how much the cm moves in response to
  speedbar and relative harness pitch. Hopefully the cm doesn't change too
  much. Or does conserving the angular momentums of the harness and parafoil
  independently successfully conserve angular momentum of the total system?
  **Is angular momentum of the system the sum of the components?**

  Reminder: Stevens Eq:1.7-3 gives the equation for angular momentum:
  `h_{cm/i}^{b}f = J^{bf} @ omega_{b/i}^{bf}`. So, if the wing had some
  rotation rate `omega0` and you go from 0 to 100% accelerator, `omega1
  = inv(J_delta1) @ J_delta0 @ omega0`

  Crazy: for the Hook3, a +5deg/s roll rate would turn into +5.77deg/s roll and
  +4.3deg/s yaw. That's a surprisingly big yaw effect.

  Also, consider where the energy from your legs dispersed into the system.
  It'll either have accelerated the wing, or lifted the payload mass (most
  likely a bit of both). Since the force is internal it won't accelerate the
  center of mass, but it will produce a change to the wing and payload
  position vectors; if you're tracking the velocity of the risers instead of
  the center of mass, you'd expect a new translational acceleration term as
  a function of the accelerator (eg, you'd expect `a_C/e` to have a -z
  contribution while the accelerator is being moved).

* Verify the roll-yaw coupling induced by the accelerator. If you hold `delta_a
  = 1` and `delta_br = 0.65` for a long time, the wing starts slipping left so
  fast the relative wind makes some sections stall. It'll probably just be
  a deficiency of my rigid-body model, but I should make note of it.


Simulator
=========

* The simulator needs to understand that Phillips can fail, and
  degrade/terminate gracefully. (Depends on how the `ForceEstimator` signal
  failures; that design is a WIP.)

* Design review support for early terminations (`Ctrl-C`) of fixed-length
  simulations (eg, "run for 120sec").

* Review the `GliderSim` state definitions (Dictionary? Structured array?)

* For the simulator, why integrate velocity in `ned` instead of `frd`?
  I forget the benefits.

* Verify the RK4 time steps and how I'm stepping the sim forward. Review `dt`,
  `first_step`, `max_step`, etc. Remember the simulation depends on the system
  dynamics (the vehicle) as well as the input dynamics (frequency content of
  the brake, speedbar, and wind values).


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
