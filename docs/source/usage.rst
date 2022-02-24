Usage
=====

[[TODO: Walk through an example of how to use the library to design and use
a wing.]]


Developing models
-----------------

[[TODO: use the paraglider state dynamics to explain how to use the library by
working backwards from the state dynamics until you get to component models.]]


Hierarchy
^^^^^^^^^

State dynamics models are interfaces to system dynamics models, which in turn
are built from component models.

For example, the hierarchy to build up a state dynamics model for a paraglider:

* :class:`StateDynamics <pfh.glidersim.simulator.StateDynamics>`

  * :class:`SystemDynamics
    <pfh.glidersim.paraglider.ParagliderSystemDynamics6a>`

    * :class:`Wing <pfh.glidersim.paraglider_wing.ParagliderWing>`

      * :class:`Foil <pfh.glidersim.foil.SimpleFoil>`

        * :class:`FoilLayout <pfh.glidersim.foil_layout.FoilLayout>`

        * :class:`FoilSections <pfh.glidersim.foil_sections.FoilSections>`

          * :class:`Profiles
            <pfh.glidersim.airfoil.AirfoilGeometryInterpolator>`

          * :class:`Coefficients
            <pfh.glidersim.airfoil.AirfoilCoefficientsInterpolator>`

        * :class:`FoilAerodynamics
          <pfh.glidersim.foil_aerodynamics.FoilAerodynamics>`

      * :class:`LineGeometry <pfh.glidersim.paraglider_wing.LineGeometry>`

    * :class:`Harness <pfh.glidersim.paraglider_harness.ParagliderHarness>`


Running simulations
-------------------

[[TODO: elaborate]]

.. literalinclude:: ../../scripts/example_simulation.py
   :language: python

.. image:: images/example_simulation_light.svg
   :align: center
   :class: only-light

.. image:: images/example_simulation_dark.svg
   :align: center
   :class: only-dark

A paraglider simulation with a short right turn, marking the relative
positions of the wing and payload every 0.5 seconds.
