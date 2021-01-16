* **Why don't `ParagliderWing` and `Harness` compute their own weight forces
  and moments?** If they don't include all the forces and moments, then the
  name `forces_and_moments` is misleading. I would probably need to pass the
  reference point for computing the moments, but so what? The `Paraglider`
  should know that. Would clean up the `Paraglider.forces_and_moments` quite
  a bit.


* If users load airfoils with `extras/airfoils/load_datfile`, how does that
  function return whether the airfoil uses `delta_f`, and if so what is its
  `delta_max`?

* Rename `delta_max` to `delta_f_max`, since `delta_f` is what
  `AirfoilCoefficients` uses for trailing edge deflections.


* Review `scripts/flat_wings.py`. Depends on pandas, hard coded paths to
  airfoil data, etc. Maybe just delete it? If it's going to stick around it
  should be more obvious that it's for checking `Phillips` against XFLR5.

* Convert `convert_xflr5_coefs_to_grid.py` into a proper CLI tool. Probably
  start by renaming it to `resample_xfoil_polars.py` or similar.

* It seems like a bad idea to use `Theta_p2b` to compute the payload restoring
  moment. It's fine for small displacements, but doesn't make sense for larger
  deviations.

* Rename the `control_points` functions `r_CP2LE`? (Just tonight I caught
  a bug because I used `r_LE2RM` instead of a vague function name.)

* Question: are the "rectangles" you get from sampling `s` and `sa`
  "quadrilaterals"?

* Aerodynamic centers exist for lifting bodies with linear lift coefficient
  and constant pitching moment? How useful is this concept for paragliders?
  (ie, over what range can you treat it as having an aerodynamic center, and
  what value would there be?)

* Note to self: different airfoils can have significantly different pitching
  coefficients (eg, the NACA 24018 vs the LS(1)-0417), which should produce
  significantly different equilibrium pitching angles. The arc of the wing
  will likely give those different wings noticeably different dynamics in the
  presence of a cross-wind, and **may have a significant impact on how the
  wing respond to encountering a thermal during a turn**.


Development
===========


Documentation
-------------

* Add a top-level introduction to the project, give the reader an overview of
  the project structure.

* Introduce each section, don't just link to the autosummaries

* Review sphinx domains (eg, Python) and the roles they define (eg, `:py:attr:`
  and `:py:class:`). Review the code for proper sphinx markup.

* Review all (sub)package, module, and class docstrings. They should have
  summaries, descriptions, parameters, attributes, etc.

* How should I document `simulator.ParagliderModel6a.state_dtype`?


* Should docstring types be "array of" or "ndarray of"? I lean towards
  "array", but would it be better to use the canonical name so sphinx can link
  to the numpy datatype documentation?

* Verify function docstrings match the signatures (`darglint` would be
  helpful, if only it worked)

* I must make sure to point out how I'm handling section dihedral angles.
  I made the conscious decision to allow step changes, even though it produces
  overlap at panel boundaries (as in my version of Belloc's reference wing).
  My assumption is that the small overlap is less important that getting the
  panel quarter-chord lines correct. You could try to account for airfoil
  thickness and round the dihedral angles at the panel boundaries, but if
  you're allowing continuously curving reference curves you'll have this issue
  anyway.

* I'm using `sphinx.ext.autosummary`, which uses `autodoc` under the hood.
  A set of Jinja2 templates from
  `<https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary>`_
  control the `autosummary` output. I'd kind of like it if each module would
  list its classes in the contents tree (left hand side of the `readthedocs`
  theme). I tried to achieve that by overriding the `module.rst` template to
  include the ``:toctree:`` directive to the ``.. autosummary::`` that's
  building up the classes in the module, but that makes sphinx angry since it
  generates duplicate stubs for those class definitions.


General
-------

* Replace `print` with `logging`

* I refer to "airfoil coordinates" in quite a few places. I'm not sure I like
  that term. It's more like the "parameter" of a parametric curve. When I read
  "coordinates" I think `xyz`.

* Vectorize `util.crossmat`?

* The control points need a redesign. I don't like stacking in them arrays
  since that requires "magic" indexing (remembering which rows belong to each
  component). I considered putting each component in a dictionary, but that
  starts to weigh on the users of the class to know what to do with each
  component. The paraglider classes shouldn't care what components are present
  in the paraglider wing (the foil and the lines). You could use an idiom like
  `moments = {key: np.cross(cps[key], forces[key]) for key in cp.keys}`, but
  sprinkling that all over seems kind of icky to me. I have a vague feeling
  a `ControlPoints` class might actually be warranted once the number of
  components gets higher, but for now I'll just make each class keep track of
  its own "magic" indices.

* Define an `njit` wrapper that replaces `njit` with a noop if Numba isn't
  installed

* How much do 'C' vs 'F' arrays affect dot product performance? Enough for
  Numba to warn me about it, at least. (see the error when defining
  `orientation.quaternion_rotate`)

* Review the API for consistency

  * Do the wing+glider functions always parametrize like (<wing stuff>,
    <environment stuff>)? Can they?


Low priority
------------

* Review function parameters for compatibility with `array_like` arguments.
  (Broadcasting is useful together with `np.meshgrid`, etc.)

* Do a performance comparison between `cross3` and the `np.cross`
  implementation added to Numba `v0.46`. As of 2019-12-16, that function is
  roughly 60% slower on small arrays, and nearly 8x slower on `10000x1000x3`
  arrays.


Packaging
---------

* Why do I need `python = "^3.6<3.9`? Why doesn't `python = "^3.6"` work? See
  https://github.com/python-poetry/poetry/issues/743#issuecomment-474304798
  I suspect a `poetry` bug; hopefully an update fixes this. Check it later.

* Complete `README.rst`

* Make `numba` a dev-only dependency by compiling the modules ahead of time.
  See https://numba.readthedocs.io/en/stable/user/pycc.html

* Make `matplotlib` an optional dependency. The goal is that it can work
  standalone in installations like Blender's built-in interpreter.


Plots
-----

* In `plots.plot_foil` I have a `surface` parameter. Should I use `airfoil` or
  `profile` for the profile surface? I'm using `airfoil` but in a way that
  contradicts its use in `surface_xyz` (`plot_foil(surface='airfoil')`
  actually plots the 'upper' and 'lower' surfaces).

* I'd sure like it if the 3D plots could use a `figsize` that wasn't square
  (it wastes too much space). I think it's because `_set_axes_equal` uses
  a radius, and all axes must contain that sphere. **Can you keep the equal
  scaling property with different axes lengths?**


Testing
-------

* What if the sensation of being "pushed out of a thermal" is a combination of
  effects: the wing yawing away and a *decrease in centripetal acceleration*?
  Maybe what's being interpreted as "being pushed out" is more a "lack of
  being pulled in"? All you know is that if feels like you're deviating from
  your desired course, that the radius of your turn is being increased.

  Oh, another interpretation: there is a reverse-pendulum after the initial
  reaction: first you roll right, yaw left (into the thermal on your right) as
  well as accelerating to your right, but then the wing snap quickly rolls
  left once you're past the thermal. A pilot might interpret this delayed
  roll-left motion as being pushed out?

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


Tooling
-------

* Try using `darglint` as a `flake8` plugin. As of 2021-01-01 this wasn't
  working well, needs review.


Airfoil
=======

* In `lingard1995RamairParachuteDesign` they suggest a NASA (NACA) LS(1)-0417
  airfoil. Good idea to compare it's basic performance to the NACA 23015. If
  I could create the airfoil data and use it for my Hook 3, even better. (At
  least review its performance characteristics: great L/D at low alpha, and
  dramatically smaller pitching moment across the range of alpha; interesting
  to consider how that'd change equilibrium conditions, etc.)

* What are "low-speed airfoils"? The `NACA LS(1)-0417` (aka the `GA(W)-1`) is
  considered low-speed, and is suggested in Lingard 1995 for ram-air
  parachutes. The UIUC low-speed airfoil data catalogs cover such airfoils,
  and they seem to use "low-speed" as synonymous with "low Reynolds number".
  I'm seeing ranges from 60,000 to 500,000, depending on the document. In that
  case, paragliders aren't particularly low-speed, but they're on the cusp,
  and the tapered wing tips certainly delve into that range. But isn't the
  "low Reynolds number" / "low-speed" assumption implying an assumption of
  laminar flow? That is, they might **only** provide superior performance
  **if** the flow is laminar? Seems like laminar flows are unlikely on
  a paraglider.


Geometry
--------

* If my airfoil coefficients are parametrized by `delta_f`, should the airfoil
  geometry be as well? I don't like either option: currently I have the
  `AirfoilCoefficients` handling the interpolation over `delta_f` since it's
  much easier to just dump all the coefficient data into a single `csv` file,
  but that implies the `AirfoilGeometry` should handle interpolating the
  geometry, which I think belongs in the `FoilSections`. The foil sections are
  there to eventually support airfoil interpolation, cell definitions, and the
  cell distortions, but maybe it'd make sense to let the `AirfoilGeometry`
  handle delta in the sense of "this is the idealized shape"? Related to this
  is "how do you compute the mass properties of a wing with brakes applied?"

* Write an `AirfoilGeometry` interpolator. Takes two geometries, and returns
  the interpolated surface points.

  **Does this make sense as a standalone thing?** It's so simple, it almost
  seems like overkill to make it it's own class. Might be preferable to have
  a single class that interpolates both the geometry and the coefficients?

* Implement **accurate** `camber_curve` and `thickness` estimators.

  This is mostly only an issue if I implement cell billowing (and thus ribs).
  If I'm going to scale airfoils by changing their thickness, then I need the
  correct camber and thickness functions. If I don't, then there will be weird
  disjoint surfaces at small thickness changes (since you'll move from the
  true surface to the version of that surface produced by estimates of its
  thickness and camber). See branch `WIP_airfoil_curves`.

* Write a basic "trailing edge deflection" routine for airfoils. Doesn't have
  to be physically accurate for now, just need to establish the API.

* Add some literature references. For NACA airfoils, there are:

  * Abbott, "Theory of Wing Sections, Sec. 6

  * https://www.hq.nasa.gov/office/aero/docs/rpt460/index.htm

  * The XFOIL source code?


Coefficients
------------

* Verify the polar curves, especially for flapped airfoils.

  The airfoil data is still a bit of a mystery to me. I don't trust the XFOIL
  output (at least not my use of it). It is extremely sensitive to tiny
  changes in the number of points, the point distribution, and especially the
  trailing edge gaps (which look like they should produce negligible
  changes?). Just creating a nominal 23015 with the builtin generator then
  removing the tiny TE gap causes the pitching moment in particular to change
  dramatically.

* Replace `AirfoilCoefficients` with `SectionCoefficients`. An airfoil is
  conceptually a fixed geometry entity, and doesn't change (no brake
  deflections). The section, however, is more general: a profile (which is
  a function of `delta_f`) and its aerodynamic coefficients (also a function
  of `delta_f`).

  If you really wanted to build a `SectionCoefficients` from individual
  airfoil polar files you could, but that should be the exception rather than
  the rule. Don't let that "atypical" use case complicate the API.

* It might be interesting if `GridCoefficients` automatically handled CSV
  files that lack `Re`. Maybe just print a warning that Reynolds values will
  be ignored. Wouldn't make for good analysis, but would be interesting for
  demonstrating the effect of ignoring Reynolds numbers.

* In `XFLR5Coefficients`, the `LinearNDInterpolator` should be able to use
  `scale=True` instead of the `Re = Re / 1e6` in the coefficients functions,
  but for some reason it doesn't work. Worth investigating?

* In `XFLR5Coefficients`, I could support XFOIL polars as well, but I'd need to
  read the columns differently. Easy way to read the headers is with `names
  = np.loadtxt(<filename>, skiprows=10, max_rows=1, dtype=str)`. I haven't
  tested it with XFOIL polars though, might be missing some nuance.


Low priority
------------

* Let `NACA` use its explicit curve definitions. I'll have to compute `x` as
  a function of arc-lengths, but beyond that use the actual functions instead
  of relying on interpolated estimates. The annoying part will be calculating
  the `profile_curve_normal` and `profile_curve_tangent` functions.

* Rewrite `AirfoilGeometry.mass_properties` to handle rotated airfoils
  (meaning you can't just integrate over `y_upper - y_lower`). Not a high
  priority for now since I'm simple shapes with derotation. Besides, I'm not
  sure this function will continue making sense later on (probably better ways
  compute the area and volume inertias of the wing (integrate the meshes for
  areas and voxels for the volume).

* Rename airfoil's `surface` to `profile`? "Surface" suggests 2D.

* Consider Gaussian quadratures or other more efficient arc-length methods?

* Why does `s` go clockwise? Why not just keep the counter-clockwise
  convention? I do like that there is a sort of right-hand rule that points in
  the +y direction though.

* Should I provide `s2d` and `d2s` functions? (Recall, `d` is the linear
  distance along the entire surface, `s` is the linear distance along each
  upper or lower surface) Suppose a user wanted to step along the curve in
  equal steps; they'd need to convert those equally spaced `d` into `s`, which
  is weird since the upper and lower surfaces use different spacings for `s`.

* Add Joukowski airfoil builders? Those are typically defined in terms of
  their surface coordinates, not mean camber and thickness curves. Neat
  airfoils though, conceptually. Very elegant.


SectionLayout
=============

* Review the calculation of the projected span `b` in `SectionLayout.__init__`.
  Should I use the furthest extent of the wing tips (typically happens at the
  leading edge if the wing has positive torsion and arc anhedral), or should
  I use `SectionLayout.b = xyz(1, r_yz(1))[1] - xyz(-1, r_yz(-1))[1]`?

* Should `SectionLayout` use the general form of the chord surface equation?
  Maybe have another class that presents the simplified parametrization I'm
  using for parafoil chord surfaces?

* Should I make the reference curves parametric functions? From a modelling
  perspective, it would be convenient if the reference curves were "owned" by
  the `LineGeometry`; it would allow things like making `yz` a function of
  `delta_a` (ie, let the `LineGeometry` own `yz`), approximate "piloting with
  the C's" control, etc. See branch `WIP_parametric_chords` for a mockup (and
  a discussion of the limitations).


Parametric functions
--------------------

* Add `taper` as an alternative parameter in `elliptical_chord`

* Should `elliptical_arc`: accept the alternative pair `{b/b_flat,
  max_anhedral}`? You often know b/b_flat from specs, and `max_anhedral` is
  relatively easy to approximate from pictures.

* I don't like requiring `yz` to be a functor that provides a `derivative`
  method. I originally did it to match the `scipy` interpolator API
  (`PchipInterpolator` in particular), but it's just awkward.

* Redefine the parameters in `foil.elliptical_arc`? This is a helper function
  that defines an angle distribution as an `EllipticalArc` parametrized by
  mean and maximum angles. This works for parafoil "arc" (not the same thing
  as the more general elliptical "arc") as well as sweep.

  And besides, I'm planning to use Euler angles (phi, theta, gamma) instead of
  the ambiguous "anhedral" angle anyway, so "tip_anhedral" is poorly named
  anyway.

  Oh, hang on: if I'm planning to use this for sweep, that'd only be a single
  function `x(s)`, so it'd be an "explicit" `EllipticalArc`. `x(s)` is
  probably more like the `elliptical_chord`, except the parameter represents
  `x` instead of `c`. Hrm. Well, probably still best to reparametrize
  `elliptical_arc` in terms of `mean_angle` and `tip_angle`.


FoilGeometry
============

* I refer to `FoilGeometry` in several places, but there's only one:
  `SimpleFoil`. There's no abstract base class anymore. Should there be? It'd
  be nice to be able to reference `FoilGeometry` and have it be a concrete
  thing in the code.

* Eliminate `Foil.chord_xyz` and add "chord" and "camber" to the `surface`
  parameter in `Foil.surface_xyz`. More recent versions of my paper discusses
  three surfaces (chords, camber lines, and section profiles); the code should
  mirror that.

  `Foil.chord_xyz` uses `pc` whereas the `surface_xyz` uses `sa`, but
  otherwise the signatures should be compatible. Actually, I'm considering
  using `r` for "position on the curve" to match `r_x` et al. So for the
  chord, camber line, upper surface, and lower surface you'd have `0 <= r <=
  1`, and for the combined profile you'd have `-1 <= r <= 1`.

* Refactor `mesh_vertex_lists` to work on any of the surfaces (`{upper, lower,
  airfoil, chord, camber}`)? Right now it just assumes you want both `upper`
  and `lower`.

* In `Foil.surface_xyz`, I use `airfoil` for the profile surfaces, but in my
  paper I'm referring to the airfoil as the unit-chord shape and "section
  profile" for the scaled shape. Should I rename `airfoil` -> `profile`?

* Should `S_flat`, `b`, etc really be class properties? Class properties don't
  support parameters, which means these break for parametric reference curves
  (eg, if arc anhedral is a function of `delta_a`). You could require users to
  specify "default parameters" for any extra parameters in the reference
  curves, but somehow that feels wrong.


FoilSections
============

* Document `FoilSections`; focus on how it uses section indices with no
  knowledge of spanwise coordinates (y-coordinates), it's xz coordinates have
  not been scaled by the chord length, etc.

  Heck, I need to document the entire stack: "a Foil is a combination of
  `SectionLayout` and `FoilSections`, both of which define units that are
  scaled by the span of the foil"


Profiles
--------

* `FoilSections.profiles` should be an airfoil interpolator. I should be able
  to load a set of datfiles and stick them in an airfoil interpolator that
  produces the right section profiles as a function of `s, delta_f`.

  Once this is done you could use the actual profiles then `plot_foil` could
  use the new `surface_xyz` to plot the actual braking surface.

* I need to review everywhere I talk about airfoil "thickness" and ensure I'm
  referring to "chordwise" or "camberwise" stations correctly. Some places
  I mention "chordwise" stations, but glancing at the code it actually looks
  like I'm computing `pc` as stations along the mean **camber** line.

* Who should be responsible for sanity checking the parameters for foil
  surface coordinates? For example, `FoilSections.surface_xz` could do it, or
  it could punt it downstream to the air intake functions (meaning each intake
  implementation should duplicate the sanity checking code).

* Reconsider the design/purpose of `surface_xz`. The name implies that the
  points are in foil frd (thus xyz, not just xy), but they're actually just
  normal airfoil xy-coordinates. I could make it transform to frd, but there's
  only one user of that: `SimpleFoil.surface_xyz`, which can do it itself
  easily enough.

  I was probably trying to maintain interface compatibility with
  `AirfoilGeometry`, but all the `FoilSections` functions require a section
  index anyway, so I'm not sure what I was going for.


Intakes
^^^^^^^

* Design review the air `intakes`. Possibly reconsider the name "intakes":
  this concept doesn't *require* that `s_upper != s_lower`; it simply means
  the upper/lower surface boundaries are different from the airfoil leading
  edge. Might even be useful for **single surface designs**, which discard the
  lower portion of the majority of the section profiles.

* Document the air intake functions (eg, `SimpleIntakes` and `_no_intakes`)


Coefficients
------------

* I'm not a fan of the duplicated docstrings in `FoilSections.Cl` and
  `AirfoilCoefficients.Cl`, etc, but if that API needs to include the section
  index I don't seen an obvious way around it.

* Review `kulhanek2019IdentificationDegradationAerodynamic` and compare his
  `C_d,f` to my "air intakes and skin friction drag" adjustments in
  `FoilSections.Cd`


Parafoil
========

* The name `SimpleFoil` is peculiar. Simple compared to what? (I think I was
  originally planning to create a `Parafoil` class which includes the cells
  and accounts for cell billowing).


Geometry
--------

* The `SectionLayout` requires the values to be proportional to `b_flat == 2`?
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


Inertia
^^^^^^^

* The new mesh-based `SimpleFoil.mass_properties2` uses triangles which are
  not symmetric outwards from the central section, so small numerical
  differences produce significantly non-zero Ixy/Iyz terms in the inertia
  tensors. Once I fix this I should also remove the manual symmetry
  corrections in `ParagliderWing.__init__`.

* Rename `Au` (upper area) to `au`? I've been trying to reserve uppercase for
  points/matrices, lowercase for scalars/vectors. (I think I did that because
  I used lowercase for individual triangles and uppercase for the sum.)

* Mark `AirfoilGeometry.mass_properties` and `SimpleFoil.mass_properties` as
  deprecated. Probably best to move it to a separate branch. Still useful for
  validation purposes, but they add way too much complexity to the overall
  codebase.

* Why doesn't the old `mass_properties` agree with the mesh-based method?

* Refactor the mesh sampling so I don't have to duplicate it in both
  `mass_properties` and `_mesh_vertex_lists`. Probably best to generalize
  `mesh_vertex_lists` to work on {"upper", "lower", "airfoil"} and add
  a different function that outputs the wing mesh to a file.


Cells
^^^^^

This is a catch-all group. Right now I'm using the idealized `SectionLayout`
directly, but real parafoils are comprised of cells, where the ribs provide
internal structure and attempt to produce the desired airfoil cross-sections,
but deformations (billowing, etc) cause deviations from that ideal shape.

Long term, I'd like to combine the idealized chord surface with a set of ribs
and produce the set of (approximately) deformed cells. There are many tasks
here:

* Replace explicit `Airfoil` references with (eg, `canopy.airfoil.geometry`)
  with a function that returns the profile as a function of section index.

* Define a set of rib types (vertical ribs, v-ribs, lateral bands, etc)

* Define a set of heuristics that approximate the inflated profiles for each
  cell (ie, profiles between the vertical ribs)

* Write functions that compute points on the chords and surfaces of sections
  from inflated or deflated cells. **There is a lot of sublety here.** There
  needs to be a mapping between the inflated and deflated section indices, so
  you can't just use the "flattened" values; the cell widths themselves
  change.

Some considerations:

* I'd like to at least try to maintain the surface areas during billowing; you
  can explicitly ignore the creases that will develop, but the total surface
  area shouldn't change THAT much. (Perhaps use the "mesh to cell surface
  area" function to compute the `thickness_ratio` that would maintain
  a constant surface area for the inflated and deflated cell surfaces?)

  Related thought: if the upper surfaces maintain the same area, do the lower
  surfaces also have the same area? Multiplying the thickness by a constant
  seems like it should be a linear function, so I *think* the lower and upper
  surfaces should both be correct, but it's worth checking.

* Try to anticipate some of the effects of billowing. For example, compare the
  performance of a normal `24018` to a 15% increased thickness `24018` using
  XFLR5 (which simply scales the airfoil by a constant factor). Make a list of
  anticipated deviations compared to the idealized `SectionLayout`. (decreased
  lift/drag ratio, etc)

* How a cell compresses during inflation depends on the shape of the parafoil
  (line loadings, etc). (ref: `altmann2019FluidStructureInteractionAnalysis`)


Deformations
^^^^^^^^^^^^

* To warp the trailing edge, could you warp the mean camber line instead of
  the surfaces themselves, then constrain to maintain constant curve length?

* Starting with the `SectionLayout`, how hard would it be to warp the central
  sections to produce a "weight shift" effect?

* Is it a fools errand to support lifting-line methods in the presence of
  deformations? Cell billowing, weight shift, trailing edge braking: they all
  produce deformed profiles, adding many dimensions to the coefficients table.


Meshes
^^^^^^

* I think my mesh functions are broken? The lower surface gave a bunch of "Bad
  face in mesh" errors that crashed Blender 2.82. See `notes-2020w19` for more
  details.

* Other issues:

  * The normals of my upper faces are backwards? (They point in, not out.)

  * When do you want triangles versus quadrilaterals? You can cut the number
    of edges and faces in half with "Edit -> Face -> Tris to Quads"

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


Lower priority
^^^^^^^^^^^^^^

* I claim that `FoilGeometry` is defined as having the central chord leading
  edge at `x = 0` and that the central chord lies in the xy-plane, **by
  definition**, but I never enforce that. I do shift the leading edge to the
  origin, but I don't derotate the global wing.

  I guess it'd be good enough to just require that `torsion(s=0) = 0`, but
  I guess I could also just compute `torsion(s=0)` and subtract that from all
  torsions, thus "centering" the twist in the same manner as the origin.

* Move `InterpolatedArc` from `belloc.py` into `foil.py` and modify it to use
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

  For my Hook3ish, `Theta_eq = 3`. Rotating the foil before projecting changed
  `S` by `0.15%`, so it's not a big deal.


Coefficient Estimation
----------------------

* **Add section-wise adjustments to coefficients.**

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

* Design review how the coefficient estimator signals non-convergence. (All
  users that call `Phillips.__call__` should be exception-aware.)

* Building a linear model for the paraglider dynamics requires the *stability
  derivatives* (derivatives of the coefficients with respect to `alpha` and
  `beta`). The direct approach is finite differencing, but for a "more
  economical approach", see "Flight Vehicle Aerodynamics" (Drela; 2014),
  Sec:6.5.7, "Stability and control derivative calculation".


Phillips
^^^^^^^^

* Add a `control_point_section_indices` or somesuch to `Phillips`. Should
  return a copy of `s_cps` so `ParagliderWing` will stop grabbing it directly.

* Review Phillips paper: he says not to use the spatial midpoints of the
  segments for the control points, and that "a significant improvement in
  accuracy for a given number of elements can be achieved", especially near
  the tips by placing the control points at the midpoints of the cosine
  distribution angle instead of the midpoints of the segments. Look into that?
  (Then again, I've been using a linear distribution in `s`, so I'm already
  deviating quite a lot from his recommendation anyway.)

* Review `github/usaero/MachUpX`, commit `93ae2a7`: "Overcame singularity in
  induced velocities by averaging the effective joint locations, thus forcing
  continuity in the vortex sheet." Useful? He may just be talking about
  discontinuities in the geometry, not the discontinuity at the wingtip.

* In `Phillips`, a comment says it's modeling the chord areas as
  parallelograms, but in general the leading and trailing edge lengths may be
  different. Is a parallelogram a reasonable shape? (Would happen in the
  presence of sweep and changing chord length; would also happen if I allowed
  section yaw, but my parametrization design avoids that.)

* By placing the boundary condition at `0.25c` instead of `0.75c` or similar,
  this method can produce infinite induced velocities as the number of
  sections increases. This is mostly a problem since it means `alpha` at the
  wing tips `alpha` can go to infinity, which produces `nan` for the lift
  coefficients. For an example that triggers this, change the arc anhedral for
  the Hook3ish from 33/67 degrees to 10/21 degrees and apply brakes; even
  though the flatter wing seems "easier" conceptually, the particularities of
  the geometry and lift curve causes failure for any reasonable number of
  segments.

* I'm using Hunsaker's derivation for `_f` and `_J`, but there is some
  uncertainty regarding his choice of wind vector (for the 3D vortex law) and
  airspeed (for section lift due to lift coefficient). Phillips uses "V_total"
  and "V_infinity", Hunsaker uses "V_total" and "V_total", and in
  "Weissinger's model of the nonlinear lifting-line method for aircraft
  design" (Owens; 1998) they appear to use "V_infinity" for both (he simply
  uses V_total for computing the induced angle of attack). These terms are all
  relatively close and don't make a huge difference, but it still bothers me.

  The bigger question is that **all of those seem wrong for a paraglider!!**
  Does the spanwise airspeed really contribute to section lift? Spanwise flow
  is significant at the wing tips of a parafoil; seems wrong for that to count
  towards section lift. I'd expect lift from the section lift coefficients to
  depend only on `V_n**2 + V_a**2`.

* The `_hybrj` solver retries a bazillion times when it encounters a `nan`.
  Can I use exceptions to abort early so I can use relaxation iterations
  instead of letting `hybrj` try to brute force bad solutions? What if `_f`
  threw an exception when it produces a `nan`, which is caught by Phillips to
  initiate a relaxation solution? (This probably depends on how scipy calls
  the Fortran code; not sure what happens to the Python exceptions.)

* If the target and reference are effectively the same, iteration will just
  waste time (since you'll keep pushing the same target onto the stack). There
  should be some kind of metric for deciding "the reference is too close to
  the target to be of much use, just abort"

* Review the conditions for non-convergence. What are the primary causes, and
  can they be mitigated? What are the average number of iterations for
  convergence? Right now, convergence via iteration is uncommon: cases either
  succeed, or they don't. It'd be nice to detect "non-convergence" ASAP.

* **Review the iteration design**: should I be interpolating `Gamma`?

* Verify the analytical Jacobian; right now the finite-difference
  approximation disagrees with the analytical version

* How should I handle a turning wing? (Non-uniform `u_inf`) Right now I just
  use the central `V_rel` for `u_inf` and assume it's the same everywhere.

* Using straight segments to approximate an curved wing will underestimate the
  upper surface and overestimate the lower surface. It'd be interesting to
  compute surface meshes for a range of `K` and (1) see how the error
  accumulates for both surfaces, and (2) consider how the upper and lower
  surfaces contribute to the airfoil coefficients. For example, if the
  dominant contributor to the section lift coefficient is the pressure over
  the upper surface of the airfoil, you'd expect an underestimate of the
  segment upper surface area to underestimate the segment lift coefficient,
  but I'm not sure what conclusions you could reliably produce from such
  a crude measure.

* Refactor Phillips outside `foil.py`?

* Why does Phillip's seem to be so sensitive to `sweepMax`? Needs testing

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


Harness
=======

* Should `delta_w` move the control point, or just the cm? Weight shift is
  mostly "inside" the payload volume.

* Redefine the `SphericalHarness` to use the radius, not the projected area.
  The projected area is not a common way to define a sphere; using the radius
  just just makes more sense.


LineGeometry
============

* The line parameters in `line_geometry` are super long. Should they be
  `kappa`-ized?

* Review the "4 riser speed system" in the "Paraglider design handbook":
  http://laboratoridenvol.com/paragliderdesign/risers.html. They use a 4-line
  setup instead of a 3-line (so the D lines are fixed), but otherwise his
  derivation closely matches my own.


ParagliderWing
==============

* Do speed bars on real wings decrease the length of all lines, or just those
  in the central sections? If they're unequal, you'd expect the arcs to
  flatten; do they?

* Review parameter naming conventions (like `kappa_a`). Why "kappa"?

* *Design* the "query control points, compute wind vectors, query dynamics"
  sequence and API

* Paraglider should be responsible for weight shifting?

  * The wing doesn't care about the glider cm, only the changes to the riser
    positions. However, **that would change if the arc supports deformations**
    in response to weight shift.

* Check if paragliders have aerodynamic centers. See "Aircraft Performance and
  Design" (Anderson; 1999), page 70 (89) for an equation that works **for
  airfoils**. The key requirement is that the foil has linear lift and moment
  curves, in which case the x-coordinate of the aerodynamic center is given by
  the slope of the pitching coefficient divided by the slope of the lift
  coefficient. But **is this accurate for an arched wing?** If so, what is the
  z-component?


Wing mass properties
--------------------

* My implementation of Barrows needs a design review. The thickness parameter
  `t` in particular. Barrows assumes a uniform thickness canopy, and I'm not
  sure how to best translate for a paraglider wing.

* `ParagliderWing.mass_properties` is ignoring the mass of the lines. Should
  `Paraglider` be responsible for including it in the center of mass
  calculations?

* `mass_properties` should take the reference point for the apparent mass as
  a parameter. It's only constraint should be that it lies in the xz-plane (to
  allow using Barrows to compute the apparent mass.) Using `R = RM` is fine
  for my primary models (6a and 9a), but models that use other reference
  points (like the wing center of mass) can't use apparent mass.

  Related: I don't like that the paraglider dynamics models have to implement
  the parallel axis theorem each time.


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

* I don't like integrating `omega_b2e` and `omega_p2e` separately. Seems like
  `Theta_p2b` (and by extension, the rest of the model dynamics) would
  accumulate error more slowly if it used `omega_p2b` (relative motion)
  instead of `omega_p2e`, but I could be wrong.

* Fix the "magic layout" for the control points in the paraglider models

* The call signature for ``Paraglider.accelerations`` needs review. I pass
  `delta_a` since that determines the control points and the wing inertia, but
  `r_CP2RM` is only there to avoid recomputing them. (I think.) Is that
  confusion-inducing redundancy worth saving the time to recompute the
  `r_CP2RM`?


Models
------

* How hard would it be to code up a linearized paraglider model? It'd be
  fascinating to see how the linear assumption performed, both in terms of
  accuracy and computation time.

* **The 9 DoF model performs very poorly with weight shift.** It looks like
  the spring-damper model isn't a good fit for a paraglider since the relative
  roll restoring force coefficient needs to be HUGE to eliminate relative roll
  (which is most noticeable during weight shift), but that introduces huge
  relative scale differences between the roll restoring force and the other
  components of the dynamics matrix, so solving becomes painfully slow.
  Probably a good idea to adapt Slegers' 8 DoF model to constrain relative
  roll to zero.

* If the center of mass moves (accelerator, weight shift, relative harness
  pitch, etc) the angular velocity must change in order to conserve angular
  momentum. Same thing for changes to any inertia matrices; consider the
  angular momentum of all components and verify they are being maintained.
  (Non-rigid-body motion is a pain!)

  This may prove tricky. If you know the cm moved a particular way, you can
  compute the angular velocity that would satisfy conservation of angular
  momentum. **But, the `Paraglider` returns accelerations, not net changes
  in velocity; if the speedbar moved the cm over `dt`, who computes that net
  change in angular momentum?** Does rate change of controls need to be part
  of the state? How else do you determine the *change per time* of angular
  momentum in response to control inputs?

  First thing to do is probably to check how much the cm moves in response to
  speedbar, weight shift, and relative harness pitch. Hopefully the cm doesn't
  change too much. Or does conserving the angular momentums of the harness and
  parafoil independently successfully conserve angular momentum of the total
  system? **Is angular momentum of the system the sum of the components?**

  Reminder: Stevens Eq:1.7-3 gives the equation for angular momentum:
  `h_{cm/i}^{b}f = J^{bf} @ omega_{b/i}^{bf}`. So, if the wing had some
  rotation rate `omega0` and you go from 0 to 100% accelerator, `omega1
  = inv(J_delta1) @ J_delta0 @ omega0`

  Crazy: for the Hook3ish, a +5deg/s roll rate would turn into +5.77deg/s roll
  and +4.3deg/s yaw. That's a surprisingly big yaw effect.

  Also, consider where the energy from your legs dispersed into the system.
  It'll either have accelerated the wing, or lifted the payload mass (most
  likely a bit of both). Since the force is internal it won't accelerate the
  center of mass, but it will produce a change to the wing and payload
  position vectors; if you're tracking the velocity of the risers instead of
  the center of mass, you'd expect a new translational acceleration term as
  a function of the accelerator (eg, you'd expect `a_R2e` to have a -z
  contribution while the accelerator is being moved).

* Investigate applying the "Paraglider Flight Dynamics" (Bendetti, 2012)
  stability analyses in the context of my refactored design (eg, longitudinal
  static stability as a function of speed bar)


Apparent Inertia
^^^^^^^^^^^^^^^^

* Is the way I'm removing the steady-state terms correct? Barrows mentions
  "simple theories, such as strip theory". Is my NLLT considered one of the
  family of strip theories, or he is referencing something more like what's
  described in "Basic Aerodynamics" (Flandro, McMahon, Roach; 2012), Sec:6.6
  "Aerodynamic strip theory"?

* Consider the apparent rolling inertia. In Barrows, Fig:6 shows the
  relationship of the apparent roll inertia versus the ratio of circular
  radius `R` to the span `b`. For my Hook 3, if `R = 4.84` and `b = 8.84`,
  then `R/b = 0.548`. They say that a ratio of 0.5 is "not realistic for
  a parafoil". Verify the results in Barrows are still valid for the Hook 3?

* Consider all the simplifications in using Barrows' method for estimating the
  apparent mass. Variable thickness, variable chord, elliptical (non-circular)
  arch, sweep, taper, torsion, etc. For example, the thickness at the wing
  tips is much thinner, so assuming uniform thickness is likely to
  overestimate the yaw apparent moment of inertia.

  Also, Barrows development of apparent inertia coefficients assumes the
  canopy has two planes of symmetry, which suggests the `x` principal axis of
  the volume is aligned with the central chord, but for normal parafoils the
  x-hat tends to be rotated pitch down (due to the non-uniform airfoil
  thickness). My current code assumes the two-planes of symmetry, and that the
  principal axes of the canopy are aligned with the body axes, but in reality
  the principal axes are rotated ~12deg pitch down. What affect does that
  have?

* I'm using Barrows equations for the *vehicle mass matrix*, which is
  equivalent to Eq:9 from (Thomasson; 2000). The limitation is that **in
  Barrows' formulation the relative accelerations mostly cancel**, so I'm not
  sure how well it works when entering/leaving lift/sink. The Thomasson (2000)
  paper goes on to develop a more general model in which the fluid medium may
  include **velocity gradients** and **accelerations**. Both of those seem
  relevant to the fine-resolution questions I'm asking of my paraglider
  dynamics (spanwise velocity gradients when you're partially in a thermal,
  for example).


Simulator
=========

* Ideally, the simulator would understand that Phillips can fail, and could
  degrade/terminate gracefully. (Depends on how the `ForceEstimator` signal
  failures; that design is a WIP.)

* Verify the RK4 time steps and how I'm stepping the sim forward. Review `dt`,
  `first_step`, `max_step`, etc. Remember the simulation depends on the system
  dynamics (the vehicle) as well as the input dynamics (frequency content of
  the brake, speedbar, and wind values).


Pre-built models
----------------

* Right now the only wing I've coded is a "Niviuk Hook 3 23". I need more
  wings (preferably at least one each from class A and C) for comparison and
  demonstration (both of how to use the library and of the difference in wing
  performance).

  I should probably also have some "suggested" paraglider models using those
  wings. Each wing has some info like weight limits; maybe that'd be good
  enough. For now just choose the parameters myself.

* For the prebuilt wings, should I have `hook3_23.canopy`, `hook3_23.wing`,
  `hook3_23.glider6a`, etc?

* For the prebuilt wings, they're made from specs. It'd be nice to standardize
  comparing the known ("expected") specs against the actual results from the
  coded version of that wing. (Right now my checks are in `build_hook3`.)


Scenarios
---------

* I'd love to demo the effect of assuming a fixed Reynolds number (eg,
  `1.5e6`) versus using their proper values. This is probably the most extreme
  during a turn. Maybe I could plot the range of values for fast straight
  flight versus a slow turn?

  Also, how does the performance of the wing change when ridge soaring into
  the wind with brakes compare to straight flight without brakes? The
  airspeed's of the different equilibriums are different, but by how much?
  Less than a factor of two, I think.

* Design a set of flight scenarios that demonstrate wing behavior under
  different wind models and control inputs.

  One thing I'd like to show is how different control+wind inputs can produce
  similar looking trajectories.

  Another thing that would be interesting is to show different scenarios where
  the controls are uncorrelated, positively correlated, or negatively
  correlated. This is interesting because it has a big impact on the proposal
  design for the control inputs (you can't just assume increasing right brake
  means decreasing left brake, for example); their *correlation depends on the
  maneuver*. Not sure if you could capture this behavior using standard
  kernels for a Gaussian process; it might need an extra parameter akin to
  a "maneuver" variable.

* Verify the roll-yaw coupling induced by the accelerator.
