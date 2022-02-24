* Rename `AirfoilCoefficients.Cl` -> `CL`? Airfoils are infinitely long
  **symmetric** wings, and their coefficients are always produced using `β = 0`
  so the roll and yaw coefficients are zero, meaning in the context of airfoils
  you don't need `Cl` for the roll coefficient.

  I think I went with `Cl` because Phillips uses `Cl` and `Cla` for the lift
  coefficient and lift coefficient slope.

* Convert `extras.sample_paraglider_positions` to resample points; it's current
  design is confusing. Just build a linear interpolator over the states and
  resample.



Development
===========


Documentation
-------------

* Review the README template from "write the docs":
  https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/

* Highlight that all models can be copied out of the tree to serve as
  a baseline to new models? For example, you can just copy and modify
  `extras.wings.niviuk_hook3` since it don't use relative imports.

* Make sure to point out how I'm handling section dihedral angles. I made the
  conscious decision to allow step changes, even though it produces overlap at
  panel boundaries (as in my version of Belloc's reference wing). My assumption
  is that the small overlap is less important that getting the panel
  quarter-chord lines correct. You could try to account for airfoil thickness
  and round the dihedral angles at the panel boundaries, but if you're allowing
  continuously curving reference curves you'll have this issue anyway.

* Document why I decomposed the `foil` like I did. For example, you can't have
  a pure `foil_geometry` because it needs the sections, but the sections are
  most easily described by combining the section profiles with their
  coefficients (so the geometry would "own" the section coefficient data).

  Why did I make `Foil` own `b_flat`? I remember that making the design curves
  in `foil_layout` normalized by `b_flat/2` allowed for simpler parametric
  forms, but why not put `b_flat` in `foil_layout`? It's weird.



Docstrings
^^^^^^^^^^

* Sphinx uses type hints to create cross-references in the source, but it
  doesn't do that automatically for the docstrings. For docstrings it requires
  explicit ReST markup like ``:py:class:`pfh.glidersim.module.class```.
  Assuming most consumers will view docstrings in the repl or their IDE, is
  this clutter worth it? In the HTML output they can still click the linked
  type in the function signature.

* Review source for docstring references to vectors like `\omega` and wrap them
  in `\vec{}`. Also check the paraglider system dynamics class docstrings for
  references like `a_B2e` or `omega_b2e` instead of their :math: equivalents.

* PEP8 recommends a docstring max line-length of 72, but that's a pain. I like
  what numpy does: keep docstring length at 75, which means if you wrap normal
  docstrings of top-level functions/classes by four spaces, then 75 puts you at
  the 79 total width.

* Review all (sub)package, module, and class docstrings. They should have
  summaries, descriptions, parameters, attributes, etc.

* How should I document `simulator.StateDynamics.state_dtype`?

* Verify function docstrings match the signatures (`darglint` would be
  helpful, if only it worked)

* Functions that accept array inputs should take `array_like` and return
  `ndarray`? Assume it's clear that a scalar input produces a scalar output?


Sphinx
^^^^^^

* `intersphinx` doesn't support equations yet (see sphinx #9483); once they do,
  add links to `eq:6dof_state_dynamics` in `StateDynamics6a`, etc

* In Sphinx there is already a "Module index" (via ":ref:`modindex`"). The
  "Library reference" section kind of overlaps with that?

* Review sphinx domains and the roles they define (eg, `:py:attr:` and
  `:py:class:`; think of `:class:` as being a role scoped inside the `:py`
  domain.) Not sure if I like fully-specified sphinx markup; it makes the
  docstrings a lot more messy (eg, instead of `LineGeometry` it's a concrete
  class like `:py:class:`pfh.glidersim.paraglider.ParagliderSystemDynamics6a`)

* Consider https://github.com/tox-dev/sphinx-autodoc-typehints

  It would be great to deduplicate type information in the signature and
  docstring, but it seems like formal `ndarray` type descriptions will always
  be a mess compared to English summaries.

* I'm using `sphinx.ext.autosummary`, which uses `autodoc` under the hood.
  A set of Jinja2 templates from [0] control the `autosummary` output. I'd
  kind of like it if each module would list its classes in the contents tree
  (left hand side of the `readthedocs` theme). I tried to achieve that by
  overriding the `module.rst` template to include the ``:toctree:`` directive
  to the ``.. autosummary::`` that's building up the classes in the module,
  but that makes sphinx angry since it generates duplicate stubs for those
  class definitions.

  [0] https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary


General
-------

* Review for standardized parameter order: `delta_*`, wind velocity, air
  density, `g`, etc

* Should I rename `mass_properties` to `inertial_properties`?

* Review default parameters; are they justified? And if so, are they
  consistent? For example, why is `rho_air` defaulted in some places but not
  others?

* Performance: why is importing `pfh.glidersim` so slow?

* Replace `print` with `logging` where suitable

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

* Review the API for consistency

  * Do the wing+glider functions always parametrize like (<wing stuff>,
    <environment stuff>)? Can they?

* According to PEP8, module-level dunders should come after `from __future__`
  imports but **before** normal imports. My `__all__` are technically wrong.

* I've been worried about how to let users write functions that support numpy
  broadcasting. You can do it manually, with array shape manipulations, or you
  can use `np.frompyfunc` and `np.vectorize`.

* Use a `.dev0` for in-development branches? See PEP440. Remove the `.dev0`
  when releasing. This helps avoid situations where you look at a commit and
  see a proper version number even though it's actually a development branch.

* I don't like that names like `v_W2h` and `v_W2b` (in the `resultant_force` of
  `ParagliderHarness` and `ParagliderinWing`) don't clearly communicate that
  they are ndarrays of vectors for each control point. Should I rename them
  `v_W2CP`?

* Do I ever use `ValueError` when `TypeError` would be more appropriate?


Refactor
^^^^^^^^

* Rename `LineGeometry` -> `SuspensionLines`. They're more than just the
  geometry; they provide mass properties and dynamics.

  Split `SuspensionLines` into `suspension_lines.py` (for consistency)

* Rename `SimpleFoil` -> `ParagliderFoil`?

  Reason being that it uses `FoilSections` which takes a `SimpleIntakes`.

  Should I make a `class ParafoilSections(FoilSections)`, leaving the
  `FoilSections` as a `Protocol`? It'd also let me isolate some of the stranger
  stuff like the coefficient offsets (like `Cd_intakes`).

  **More support for putting the `paraglider_*` modules into a package**

  If I'm refactoring stuff, what happens to `SimpleFoil`? And does it make
  sense to keep it scaled by `b_flat / 2`? Heck, does it make sense for `Foil`
  to be opinionated about the choice of section index? **Shouldn't that belong
  to the specific aircraft model choices?** Like, defining `s` using the length
  of the `yz`-curve makes sense of paragliders, but maybe not for other wings.
  It'd be nice if the top level `foil_` modules were unopinionated.

  What about `SimpleFoil.mass_properties`? It knows parafoil-specific details
  like upper vs lower surfaces. Should `mass_properties` be part of the
  `Foil(Protocol)`? Who uses it outside of the immediate consumer
  (`ParagliderWing` and `Paraglider`)? Same thing for the mesh generating
  functions; they return separate upper/lower meshes. I guess the general
  rule should be "don't try to generalize prematurely"; until somebody
  outside the model-specific code needs it, don't add it to the Protocol.

* Reorganize all the paraglider-specific bits into a subpackage? So I'd have
  `pfh.glidersim.paraglider` with `.wing`, `.lines`, `.harness`, etc? I'd
  like to move the paraglider-specific state dynamics out of the `simulator`
  module. Also, `paraglider.paraglider` is a bit weird; maybe
  `.system_dynamics` and `.state_dynamics`? Also,
  `ParagliderSystemDynamics6a` is kinda ridiculous.

  Might be helpful to consider what it'd look like if I added other aircraft
  like hang gliders, kites, or sailplanes.

* `extras/compute_polars.py`

  Rename the module? It computes polar curves and wing coefficients.

  Rename `plot_polar_curve` -> `polar_curve`; it doesn't plot anything.

  Refactor the plotting in `plot_wing_coefficients` into `plots.py`

  I should generalize the coefficients estimation in `belloc.py` and move it
  into `glidersim`; should take reference lengths/areas and return
  `{CX,CY,CZ,Cl,Cm,Cn,CXa,CYa,CZa,Cla,Cma,Cna}`. (Should it understand to start
  from `alpha = 0` and work outwards to improve convergence? Same for `beta`.)

  Would need to replace the `dict` with a numpy array with a structured dtype;
  instead of direct indexing by value, you'd need something like
  ``coeffs[:,betas==specific_beta]['CZa']``. Also, some of the inputs might fail
  to converge; I guess set their components to `nan`? (So instead of empty
  arrays, you'd have `CXa = np.nan` for those entries.

  .. code-block:: python

     def compute_coefficients(alphas, betas, r_CP2LE, c_ref, S_ref):
         # c_ref and S_ref are the reference length and area
         alphas = np.asarray(alpha)
         betas = np.asarray(betas)
         dtype = ("CX", "CXa", ...)
         coeffs = np.array(
             shape=np.broadcast_shapes(alpha, beta),
             np.nan,
             dtype=dtype,
         )
         for ka, alpha in enumerate(alphas):
            for kb, beta in enumerate(betas):
               dF, dM = wing.aerodynamics(...)
               F = dF.sum(...)
               M = dM.sum(...) + np.cross(r_CP2RM, ...)
               CX, CY, CZ = ...
               CXa, CYa, CZa = ...
            coefs[ka, kb] = (CX, CY, CZ, CXa, CYa, CZa)

   What about control inputs? For polar curves you accelerator and brakes. Use
   `kwargs`? Oh, and computing polar curves is NOT the same thing as
   coefficient curves; don't confuse the two. Coefficient curves are usually
   not associated with equilibrium states; polar curves are. Also, polar curves
   depend on both aerodynamics **and gravity**; very different.

   **Postpone**:: `belloc.py` uses `dict` indexed by `alpha` and `beta`



Static typing
-------------

* Remember to add a `py.typed` when ready. See PEP 561:
  https://www.python.org/dev/peps/pep-0561/#packaging-type-information

* A variety of functions take a callable as an argument. Add the callable
  parameters and return type, and document the signature in the docstring.

  Gets messy when using numpy types though. For example, `FoilLayout` takes
  a bunch of parameters that are `float | Callable`. You can type the callables
  with `Callable[[npt.ArrayLike], npt.ArrayLike]` (using the new `numpy.typing`
  module), but `mypy` still complains that I try to use `float(r_x)` even
  though that call is guarded. Needs review.

* Use `typing.Literal` for parameters like `surface: typing.Literal["upper",
  "lower", "camber", "airfoil"]`. Unlike the assertion-based checking, this
  alerts the programmer when they're writing the code instead of crashing at
  runtime (assuming they use `mypy`).


Numpy typing
^^^^^^^^^^^^

* Some functions have their return types marked `array_like`, but I think the
  numpy convention is to return "scalar or ndarray".

* When numpy 1.21 is released, consider using `numpy.typing.NDArray` for
  return values. It's a little ambiguous because if most functions return
  a scalar given scalar inputs, but I think this is consistent with how most
  numpy functions are typed. (related: numpy#19064)

* Type hint `array_like` inputs. Numpy 1.20 provides `npt.ArrayLike`, but it
  allows all scalar types; I need hinting like `ArrayLike[Any, bool]`.

  I think numpy v1.22 fixes this: "`ndarray`, `dtype`, and `number` are now
  runtime-subscriptable", allowing `np.ndarray[Any, np.dtype[np.float64]]`.
  Now I just need shape information; see numpy#16544

* Some parameters are `array_like of float`, but you can't set a type qualifier
  with `npt.ArrayLike`. Is that expected to change?

* I don't like duplicating the type information in the docstrings, but at the
  moment the formal types are much less friendly their informal docstring
  counterparts.

  Which types (formal signature or docstring) does Sphinx use?

  Don't think I should hold my breath for `numpydoc` support:
  https://github.com/numpy/numpydoc/issues/196

  Looks like Napolean handles it though:
  https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#type-annotations


Low priority
------------

* Review function parameters for compatibility with `array_like` arguments.
  (Broadcasting is useful together with `np.meshgrid`, etc.)


Packaging
---------

* Merge `setup.cfg` into `pyproject.toml` once `setuptools` supports PEP 621

* Create `.git_archival.txt` once `setuptools_scm` supports the new
  `git log --format=%(describe)` syntax in `git 2.35`

  https://github.com/pypa/setuptools_scm/issues/578#issue-913435885

  Adopt changes demonstrated in
  https://github.com/pypa/setuptools_scm/pull/580/files (add
  `.git_archival.txt`, update `.gitattributes` and `MANIFEST.in`, etc).
  Probably need to bump `setuptools>=` in `pyproject.toml`.

* Publish to Zenodo, add *concept DOI* to README and `CITATION.cff`, and add
  version DOI to thesis. After publishing the thesis, and its DOI to
  `glidersim` and add equation references (eg, `Heatwole Eq:2.7`)

* Make `matplotlib` an optional dependency? Put it in a `plotting` extras.
  Could lazy-load modules that import the library and present a user-friendly
  error if a user tries to use them without having `matplotlib` installed.


Plots
-----

* Could `plot_3d_simulation_path` be refactored to plot labeled ndarray of
  points that should be connected by lines? You could pass in `r_RM2O` as
  a `(K,)`, or `(r_RM2O, r_LE2O)` as a `(2,K)` and plot lines pairwise. Once
  you've got the general plotting function you could call it multiple times
  with the same `ax` and create custom plotters for scenarios like it's
  currently doing.

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

* Testing component models using realistic designs is a real chore because true
  values are difficult to calculate by hand. Instead, start with simplistic
  models for fixtures, like an `AirfoilCoefficientsInterpolator` that just
  returns `1`, a `FoilAerodynamics` that just returns `1`, a wing model that
  weighs 1kg with the cg 1m below the wing, etc.


Foil aerodynamics
^^^^^^^^^^^^^^^^^

* Method that always produce a force of 1 in the direction of `v_W2b`?

* Method that always produces a yaw restoring force proportional to the
  sideslip angle?


Foil
^^^^

* Test `mass_properties()` with simple shapes like spheres and cubes; should be
  easy to verify surface areas, volumes, centers of mass, and moments of
  inertia.


Paraglider wing
^^^^^^^^^^^^^^^

* `_compute_real_mass_properties`: difficult to test since it relies on
  `canopy.mass_properties`.


Orientation
^^^^^^^^^^^

* Move the tests embedded in `orientation.py` into `test_orientation.py`


Tooling
-------

* Update `MANIFEST.in` to prune `.flake8`, `.gitignore`,
  `.pre-commit-config.yaml`, `TODO.rst`, `requirements*.txt`, etc? Guess it
  doesn't REALLY matter, they're small. What about `docs`? Is the `sdist`
  strictly the content used to build the wheel? If so, it'd only need `src` and
  a few config files.

* Try using `darglint` as a `flake8` plugin. As of 2021-01-01 this wasn't
  working well, needs review.


Numba
^^^^^

* How much do 'C' vs 'F' arrays affect dot product performance? Enough for
  Numba to warn me about it, at least. (see the error when defining
  `orientation.quaternion_rotate`)

* Verify that setting `ai.flags.writeable = False` to silence Numpy warnings
  about `broadcast_arrays` is okay. I'm not sure who triggers the warning, but
  Numba doesn't seem to respect the `writeable` flag, so I need to verify that
  `interp3d` doesn't modify its arguments.

* Benchmark `cross3` versus `np.cross` in Numba `v0.46`. As of 2019-12-16, that
  function is roughly 60% slower on small arrays, and nearly 8x slower on
  `10000x1000x3` arrays.

* Make `numba` a dev-only dependency by compiling the modules ahead of time?
  See https://numba.readthedocs.io/en/stable/user/pycc.html Unfortunately, as
  of Numba 0.55 "ahead-of-time" compilation doesn't support numpy ufuncs.


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

* Different airfoils can have significantly different pitching coefficients
  (eg, the NACA 24018 vs the LS(1)-0417), which should produce significantly
  different equilibrium pitching angles. The arc of the wing will likely give
  those different wings noticeably different dynamics in the presence of
  a cross-wind, and **may have a significant impact on how the wing respond to
  encountering a thermal during a turn**.


Geometry
--------

* Implement **accurate** `camber_curve` and `thickness` estimators.

  This is mostly only an issue if I implement cell billowing (and thus ribs).
  If I'm going to scale airfoils by changing their thickness, then I need the
  correct camber and thickness functions. If I don't, then there will be weird
  disjoint surfaces at small thickness changes (since you'll move from the
  true surface to the version of that surface produced by estimates of its
  thickness and camber). See branch `WIP_airfoil_curves`.

* Add some literature references. For NACA airfoils, there are:

  * Abbott, "Theory of Wing Sections, Sec. 6

  * https://www.hq.nasa.gov/office/aero/docs/rpt460/index.htm

  * The XFOIL source code?


Coefficients
------------

* `GridCoefficients` and `GridCoefficients2` **require** the CSV to contain an
  "airfoil index" column; you can't use them to interpolate coefficients for
  a single airfoil geometry. They'll need special logic for calling the grid
  interpolators.

* Implement clamping in *XFLR5Coefficients*, or at least warn that it doesn't
  support clamping. The alternative is to add code to resample the coefficients
  onto a regular regular grid and return a `GridCoefficients` (didn't I have
  a script to do this already?), but that's a pain since the `GridCoefficients`
  expects the coefficients set includes a range of `ai`; does it work if `ai=0`
  always?

* Verify the polar curves, especially for flapped airfoils.

  The airfoil data is still a bit of a mystery to me. I don't trust the XFOIL
  output (at least not my use of it). It is extremely sensitive to tiny
  changes in the number of points, the point distribution, and especially the
  trailing edge gaps (which look like they should produce negligible
  changes?). Just creating a nominal 23015 with the builtin generator then
  removing the tiny TE gap causes the pitching moment in particular to change
  dramatically.

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

* Consider Gaussian quadratures or other more efficient arc-length methods?

* Why does `r` go clockwise? Why not just keep the counter-clockwise
  convention? I do like that there is a sort of right-hand rule that points in
  the +y direction though.

* Should I provide `r2d` and `d2r` functions? (Recall, `d` is the linear
  distance along the entire surface, `r` is the linear distance along each
  upper or lower surface) Suppose a user wanted to step along the curve in
  equal steps; they'd need to convert those equally spaced `d` into `r`, which
  is weird since the upper and lower surfaces use different spacings for `r`.

* Add Joukowski airfoil builders? Those are typically defined in terms of
  their surface coordinates, not mean camber and thickness curves. Neat
  airfoils though, conceptually. Very elegant.


FoilLayout
==========

* Define a `Protocol` for the `yz` parameter; it's VERY unclear that you need
  an object that defines `__call__` and `derivative`; very weird API.

* Review the calculation of the projected span `b` in `FoilLayout.__init__`.
  Should I use the furthest extent of the wing tips (typically happens at the
  leading edge if the wing has positive torsion and arc anhedral), or should
  I use `FoilLayout.b = xyz(1, r_yz(1))[1] - xyz(-1, r_yz(-1))[1]`?

* Should `FoilLayout` use the general form of the chord surface equation?
  Maybe have another class that presents the simplified parametrization I'm
  using for parafoil chord surfaces?

* Should I make the reference curves parametric functions? From a modelling
  perspective, it would be convenient if the reference curves were "owned" by
  the `LineGeometry`; it would allow things like making `yz` a function of
  `delta_a` (ie, let the `LineGeometry` own `yz`), approximate "piloting with
  the C's" control, etc. See branch `WIP_parametric_chords` for a mockup (and
  a discussion of the limitations).

* `FoilLayout` requires the values to be proportional to `b_flat == 2`? **What
  if you don't know `b_flat`? Do you need to compute the total length of `yz`
  and re-normalize to that?** (I think I'm missing something here... As long as
  everything is proportional, who cares? I'll need to look for anywhere that
  uses `s` to stand in for `y`, but other than that, who cares? May want to
  introduce an scaling value as a convenience for the user though.)


Parametric functions
--------------------

* Add `taper` as an alternative parameter in `EllipticalChord`

* Should `EllipticalArc`: accept the alternative pair `{b/b_flat,
  max_anhedral}`? You often know b/b_flat from specs, and `max_anhedral` is
  relatively easy to approximate from pictures.

* I don't like requiring `yz(s)` to be a functor that provides a `derivative`
  method. I originally did it to match the `scipy` interpolator API
  (`PchipInterpolator` in particular), but it's just awkward.

* Redefine the parameters in `EllipticalArc`? I've moved the paper away from
  "dihedral/anhedral" angles since they're ambiguous. Euler angles are more
  explicit, but it's not clear how to translate those into this usage.


FoilSections
============

* Rename `FoilSections` to `ParafoilSections`? They have intakes.

* The `FoilSections` is really a section interpolator. It's naming should make
  that clear, ala `AirfoilGeometryInterpolator`. It's the same concept, except
  it also adds `s` to beginning of the function signatures. While I'm at it,
  it should take a dictionary `{s: {"geometry": g, "coefficients": c}}`, and
  probably have a `symmetric : bool` property.

* Document `FoilSections`; focus on how it uses section indices with no
  knowledge of spanwise coordinates (y-coordinates), it's xz coordinates have
  not been scaled by the chord length, etc.

  Heck, I need to document the entire stack: "a Foil is a combination of
  `FoilLayout` and `FoilSections`, both of which define units that are
  scaled by the span of the foil"


Profiles
--------

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

* Design review the air `intakes`. Possibly reconsider the name "intakes": this
  concept doesn't require that `s_upper != s_lower`; it simply means the
  upper/lower surface boundaries are different from the airfoil leading edge.
  Might even be useful for *single surface* designs, which discard the lower
  portion of the majority of the section profiles.

* Document the air intake functions (eg, `SimpleIntakes` and `_no_intakes`)


Coefficients
------------

* Performance: precompute the "size" of the air intakes for `FoilSections.Cd`.
  It's surprisingly slow to recompute every call. Should probably be done as
  part of the larger work to factor out coefficients modifiers. At the same
  time I might want to consider the more general design issue of spanwise
  variation of coefficient adjustments.

* Review `kulhanek2019IdentificationDegradationAerodynamic` and compare his
  `C_d,f` to my "air intakes and skin friction drag" adjustments in
  `FoilSections.Cd`


FoilAerodynamics
================

* Rename `reference_solution`. It should be a generic `dict` that the
  aerodynamics method can stuff with whatever they need (like `solution`).

* Design review how the coefficient estimator signals non-convergence. (All
  users that call `Phillips.__call__` should be exception-aware.)

* Building a linear model for the paraglider dynamics requires the *stability
  derivatives* (derivatives of the coefficients with respect to `alpha` and
  `beta`). The direct approach is finite differencing, but for a "more
  economical approach", see "Flight Vehicle Aerodynamics" (Drela; 2014),
  Sec:6.5.7, "Stability and control derivative calculation". For an example of
  the defining equations for computing the linearized coefficients, check
  "Appendix A" of :cite:`slegers2017ComparisonParafoilDynamic`. For a paper
  with a set of numerical values, :cite:`toglia2010ModelingMotionAnalysis`.

* Aerodynamic centers exist for lifting bodies with linear lift coefficient
  and constant pitching moment? How useful is this concept for paragliders?
  (ie, over what range can you treat it as having an aerodynamic center, and
  what value would there be?)


Phillips
--------

* Verify `Phillips._J`. It doesn't match the finite-difference approximation in
  `Phillips._J_finite`. Should review `_J_finite` by comparing it to
  `scipy.optimize.approx_fprime` (which, sadly, is univariate only).

* How should I handle a turning wing? (Non-uniform `u_inf`) Right now I just
  use the central `V_rel` for `u_inf` and assume it's the same everywhere.

  This is a general issue with aerodynamic methods that rely on the
  *straight-wake assumption*. In general, vortex filaments do no have to be
  straight lines, but the math is much simpler if they do (for example, with
  Phillips they get to let two segments that share a node also share the
  trailing vortex filament from that node, and because they're in opposite
  directions the net shed vorticity is simply their sum). **The straight-wake
  assumption is invalid for a rotating wing.** Faster turn rates will produce
  larger error (but then most of these methods assume minimal spanwise flow, so
  it's already a crapshoot).

  Related: anytime you change speed and/or direction, you should shed an
  additional *starting vortex*, which (I think?) should require additional
  energy beyond what you'd expect from the steady-state aerodynamics.

  For insight into the magnitude of the error, consult the manual for AVL
  (`avl_dot.txt`). In the section "Unsteady Flow":

     AVL assumes quasi-steady flow, meaning that unsteady vorticity shedding is
     neglected.  More precisely, it assumes the limit of small reduced
     frequency, which means that any oscillatory motion (e.g. in pitch) must be
     slow enough so that the period of oscillation is much longer than the time
     it takes the flow to traverse an airfoil chord.  This is true for
     virtually any expected flight maneuver.  Also, the roll, pitch, and yaw
     rates used in the computations must be slow enough so that the resulting
     relative flow angles are small.  This can be judged by the dimensionless
     rotation rate parameters, which should fall within the following practical
     limits.

     -0.10 < pb/2V < 0.10
     -0.03 < qc/2V < 0.03
     -0.25 < rb/2V < 0.25

     These limits represent extremely violent aircraft motion, and are unlikely
     to exceeded in any typical flight situation, except possibly during
     low-airspeed aerobatic maneuvers.  In any case, if any of these parameters
     falls outside of these limits, the results should be interpreted with
     caution.

* Review what happens if `v_W2f` is all zeros

* Add a `control_point_section_indices()` or somesuch to `Phillips`. Should
  return a view of `s_cps` so `ParagliderWing` can stop grabbing it directly.

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
  approximation disagrees with the analytical version (which isn't unexpected,
  actually: it's computing `Cl_alpha` using finite differences of linearly
  interpolated values of `Cl`).

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

* Profile and optimize

  * For example, ``python -m cProfile -o belloc.prof belloc.py``, then ``>>>
    p = pstats.Stats('belloc.prof'); p.sort_stats('cumtime').print_stats(50)``

  * Do the matrices used in the `einsum` calls have the optimal in-memory
    layout? Consider the access patterns and verify they are contiguous in the
    correct dimensions (ie, `C` vs `F` contiguous; see ``ndarray.flags``)

* Phillips' could always use more testing against XFLR5 or similar. I don't
  have geometry export yet, but simple flat wings should be good for comparing
  my Phillips implementation against the VLM methods in XFLR5.


Foil
====

* HIGH: should there be a `FoilGeometry` class? Right now `SimpleFoil` combines
  the layout, sections, and aerodynamics, and you set aerodynamics to `None` if
  you don't care. A bit weird. I think I need a class `Foil(FoilGeometry,
  FoilAerodynamics)` or similar. I've NEVER liked this design where
  `SimpleFoil` passes `self` to initialize the `FoilAerodynamics`.

* HIGH: I refer to `FoilGeometry` in several places, but that class doesn't
  exist. There is only `SimpleFoil` in `foil.py`. Define a `Protocol`. What are
  the essential needs? `section_orientation, chord_length, surface_xyz`. More?
  I think the least constraining view is "profiles as a function of section
  index positioned along some line".

* HIGH: the name `SimpleFoil` is peculiar. Simple compared to what? (I think
  I was originally planning to create a `Parafoil` class which includes the
  cells and accounts for cell billowing).

* HIGH: In `Foil.surface_xyz`, I use `airfoil` for the profile surfaces, but in
  my paper I'm referring to the airfoil as the unit-chord shape and "section
  profile" for the scaled shape. Should I rename `airfoil` -> `profile`?


* Refactor `mesh_vertex_lists` to work on any of the surfaces (`{upper, lower,
  airfoil, chord, camber}`)? Right now it just assumes you want both `upper`
  and `lower`.

* The mesh functions don't support airfoil indices (they fix `ai = 0`)

* Should `S_flat`, `b`, etc really be class properties? Class properties don't
  support parameters, which means these break for parametric reference curves
  (eg, if arc anhedral is a function of `delta_a`). You could require users to
  specify "default parameters" for any extra parameters in the reference
  curves, but somehow that feels wrong.


Inertia
^^^^^^^

* The new mesh-based `SimpleFoil.mass_properties` uses triangles which are not
  symmetric outwards from the central section, so small numerical differences
  produce significantly non-zero Ixy/Iyz terms in the inertia tensors. Once
  I fix this I should also remove the manual symmetry corrections in
  `ParagliderWing.__init__`.

* Why doesn't the old `mass_properties` agree with the mesh-based method? See
  `scripts/validate_inertia.properties.py`

* Refactor the mesh sampling so I don't have to duplicate it in both
  `mass_properties` and `_mesh_vertex_lists`. Probably best to generalize
  `mesh_vertex_lists` to work on {"upper", "lower", "airfoil"} and add
  a different function that outputs the wing mesh to a file.


Cells
^^^^^

This is a catch-all group. Right now I'm using the idealized `FoilLayout`
directly, but real parafoils are comprised of cells, where the ribs provide
internal structure and attempt to produce the desired airfoil cross-sections,
but deformations (billowing, etc) cause deviations from that ideal shape.

Long term, I'd like to combine the idealized chord surface with a set of ribs
and produce the set of (approximately) deformed cells. There are many tasks
here:

* Replace explicit `AirfoilGeometry` references (eg,
  `canopy.airfoil.geometry`) with a function that returns the profile as
  a function of section index.

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
  anticipated deviations compared to the idealized `FoilLayout`. (decreased
  lift/drag ratio, etc)

* How a cell compresses during inflation depends on the shape of the parafoil
  (line loadings, etc). (ref: `altmann2019FluidStructureInteractionAnalysis`)


Deformations
^^^^^^^^^^^^

* To warp the trailing edge, could you warp the mean camber line instead of
  the surfaces themselves, then constrain to maintain constant curve length?

* Starting with the `FoilLayout`, how hard would it be to warp the central
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

* Is the "mean aerodynamic chord" a useful concept for arched wings?

* Should the "projected surface area" methods take pitch angle as a parameter?

  I'm not sure what most paraglider wing manufacturers use for the projected
  area. My definitions requires that the central chord is parallel to the
  xy-plane, but I imagine some manufacturers would use the equilibrium angle
  of the wing. It's more in-line with what you'd use for classical aerodynamic
  analysis, and it's essential constant regardless of load.

  For my Hook3ish, `Theta_eq ~= 3`. Rotating the foil before projecting
  changed `S` by `0.15%`, so it's not a big deal.


Harness
=======

* Should weight shift move the aerodynamic control point?

* Reparametrize `SphericalHarness` to use the radius instead of the projected
  area? Projected area is not a common way to define a sphere; using the
  radius just makes more sense. Then again, projected area is the common way
  for papers that suggest coefficients of drag for paraglider harnesses.


Line geometry
=============

* Should `LineGeometry` use `resultant_force` instead of `aerodynamics`?

* Add a proper line geometry. `SimpleLineGeometry` is a kludge that produces
  deflection distributions that you're *assuming* can be produced by a line
  geometry. Checkout `altmann2015FluidStructureInteractionAnalysis` for
  a discussion on "identifying optimal line cascading"

* The names of the line parameters in `SimpleLineGeometry` are super long.
  Should they be `kappa`-ized?

* Review the "4 riser speed system" in the "Paraglider design handbook":
  http://laboratoridenvol.com/paragliderdesign/risers.html. They use a 4-line
  setup instead of a 3-line (so the D lines are fixed), but otherwise his
  derivation closely matches my own.

* Assumes the total line length (for the line drag) is constant. Technically
  the lines get shorter when the accelerator is applied. Probably negligible.


ParagliderWing
==============

* Review `ParagliderWing.aerodynamics`; it's UGLY.

* Eliminate "magic indexing"

* Canopy parameters (`rho_upper`, `N_cells`, etc) should belong to the canopy,
  but first I need a foil with native support for internal ribs.

* Do speed bars on real wings decrease the length of all lines, or just those
  in the central sections? If they're unequal, you'd expect the arcs to
  flatten; do they?

* *Design* the "query control points, compute wind vectors, query dynamics"
  sequence and API. Pretty ad hoc right now.

* Check if paragliders have aerodynamic centers in any meaningful sense. See
  "Aircraft Performance and Design" (Anderson; 1999), page 70 (89) for an
  equation that works **for airfoils**. The key requirement is that the foil
  has linear lift and moment curves, in which case the x-coordinate of the
  aerodynamic center is given by the slope of the pitching coefficient divided
  by the slope of the lift coefficient. But **is this accurate for an arched
  wing?** If so, what is the z-component?


Wing mass properties
--------------------

* `ParagliderWing.mass_properties` no longer uses `delta_a`, which would
  technically mean the suspension line inertia is constant. Close enough, but
  technically not true. Then again, `ParagliderWing.mass_properties` ignores
  the mass of the lines anyway (it doesn't contribute weight or moment).

* My implementation of Barrows needs a design review. The thickness parameter
  `t` in particular. Barrows assumes a uniform thickness canopy, and I'm not
  sure how to best translate for a paraglider wing.

* `mass_properties` should take the reference point for the apparent mass as
  a parameter. It's only constraint should be that it lies in the xz-plane (to
  allow using Barrows to compute the apparent mass.) Using `R = RM` is fine
  for my primary models (6a and 9a), but models that use other reference
  points (like the wing center of mass) can't use apparent mass.

  Related: I don't like that the paraglider dynamics models have to implement
  the parallel axis theorem each time.


Wing mass moment
----------------

Technically, the weight of the wing materials add an extra moment.
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

* Eliminate "magic indexing"

* I don't like integrating `omega_b2e` and `omega_p2e` separately. Seems like
  `Theta_p2b` (and by extension, the rest of the model dynamics) would
  accumulate error more slowly if it used `omega_p2b` (relative motion)
  instead of `omega_p2e`, but I could be wrong.

* Review the `Paraglider.accelerations` API. In particular, the simulator
  needed `r_CP2RM` to lookup the wind vectors at each point, but no longer
  passes it them, so `Paraglider.accelerations` must recompute them from
  `delta_a` and `delta_w`. Not passing them costs a bit of compute time in
  exchange for a simpler API. Could add caching to `Paraglider.r_CP2RM` but
  I don't love the interface in the first place.


Models
------

* Why is `ParagliderSystemDynamics6a` faster with apparent mass **enabled**?

* It seems like a bad idea to use `Theta_p2b` to compute the payload restoring
  moment in the 9DoF models. The linear relationship is probably fine for
  small displacements, but would probably break down for larger deviations.

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

* The state dynamics models call `paraglider.control_points`, then
  `paraglider.accelerations` also calls `paraglider.control_points`, then
  `wing` and `harness` both also call their own `control_points`. The reason
  was to simplify the function signatures and avoid passing redundant
  information (since knowledge of the control points is contained in the
  control inputs), but YUCK.

* The simulator should use a generic `R` instead of `RM`. The system dynamics
  model are free to use whatever reference point is convenient and let the
  state dynamics model compute the dynamics wrt the `R`. Using `R` would make
  it easier to reuse the simulator `states` output in things like plots.

* Ideally, the simulator would understand that Phillips can fail, and could
  degrade/terminate gracefully. (Depends on how the `FoilAerodynamics` signals
  failures; that design is a WIP.)

* Verify the RK4 time steps and how I'm stepping the sim forward. Review `dt`,
  `first_step`, `max_step`, etc. Remember the simulation depends on the system
  dynamics (the vehicle) as well as the input dynamics (frequency content of
  the brake, speedbar, and wind values).

* Add calculated `alpha` and `v_mag` to `simulator.prettyprint_state`

* Documentation: highlight that state dynamics models expect deterministic
  input functions of the time `t`.


Pre-built models
----------------

* Right now the only wings I've coded are "Niviuk Hook 3". I need more wings
  (preferably at least one each from class A and C) for comparison and
  demonstration (both of how to use the library and of the difference in wing
  performance). I should probably also add complete glider models using those
  wings, adding suitable payloads to each wing.


Niviuk Hook 3
^^^^^^^^^^^^^

* `2013-01-23_hook3_23_en.pdf` says the "symmetric control travel" is `>60cm`.
  I've got `kappa_b = 0.43m`, so I'm hitting 60% or so brake input?


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


Scripts
=======

* Convert `convert_xflr5_coefs_to_grid.py` into a proper CLI tool. Probably
  start by renaming it to `resample_xfoil_polars.py` or similar.
