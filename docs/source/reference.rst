Library Reference
=================

Foil modeling
-------------

The foil model is created from a continuum of foil sections.

.. autosummary::
   :toctree: _autosummary

   pfh.glidersim.airfoil
   pfh.glidersim.foil_sections
   pfh.glidersim.foil_layout
   pfh.glidersim.foil_aerodynamics
   pfh.glidersim.foil


Paraglider modeling
-------------------

A glider system dynamics model is created by combining component models for the
canopy, lines, and harness.

[[well, `wing = lines + canopy`, then `paraglider = wing + harness`]]

.. autosummary::
   :toctree: _autosummary

   pfh.glidersim.paraglider_wing
   pfh.glidersim.paraglider_harness
   pfh.glidersim.paraglider


Simulation
----------

A flight simulation is created by integrating a set of state derivatives over
time to create a state trajectory. A state dynamics model is responsible for
choosing a set of state variables and relating their derivatives to the system
dynamics.

.. FIXME: separate the StateDynamics from the simulator; this is confusing

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
