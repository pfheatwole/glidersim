Library Reference
=================

Wing modeling
-------------

The wing model is created from a continuum of wing sections.

.. autosummary::
   :toctree: _autosummary

   pfh.glidersim.airfoil
   pfh.glidersim.foil_sections
   pfh.glidersim.foil_layout
   pfh.glidersim.foil_aerodynamics
   pfh.glidersim.foil


Paraglider modeling
-------------------

A paraglider dynamics model is created by combining models for the canopy,
lines, and harness.

[[well, `wing = lines + canopy`, then `paraglider = wing + harness`]]

.. autosummary::
   :toctree: _autosummary

   pfh.glidersim.paraglider_wing
   pfh.glidersim.paraglider_harness
   pfh.glidersim.paraglider


Simulation
----------

A flight simulation is created by defining a set of state variables and
relating the state dynamics to the glider dynamics.

.. autosummary::
   :toctree: _autosummary

   pfh.glidersim.orientation
   pfh.glidersim.simulator


Extras
------

The ``extras`` sub-package provides extra resources such as predefined
component models and utility functions.

.. autosummary::
   :toctree: _autosummary

   pfh.glidersim.extras.airfoils
   pfh.glidersim.extras.compute_polars
   pfh.glidersim.extras.plots
   pfh.glidersim.extras.simulation
   pfh.glidersim.extras.wings
