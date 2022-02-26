pfh.glidersim
=============

.. What is it?

.. Who is it for?

.. How is it used?

A Python library for estimating the flight dynamics of gliding aircraft.

To simplify its use in common engineering applications (such as control
modeling and statistical filtering), the glider dynamics are encoded as
differential equations (``ẋ = f(x, u)``).

* **Source code**: https://github.com/pfheatwole/glidersim/

* **License**: permissive (`MIT
  <https://github.com/pfheatwole/pfh.glidersim/blob/main/LICENSE.txt>`__)

* **Documentation**: https://pfheatwole.github.io/glidersim/

* **Questions**: for general questions, please use the `discussion forum
  <https://github.com/pfheatwole/glidersim/discussions>`__.

* **Support**: to report a problem, please use the `issue tracker
  <https://github.com/pfheatwole/glidersim/issues>`__.


Key Features
------------

.. Features are *what* it does, not *how* it works.

.. FIXME: how does it compare to existing simulators? What makes it special?

.. FIXME: rewrite; no clear structure to these lists


Modeling
^^^^^^^^

* Novel wing geometry model optimized for nonlinear designs

  .. This flexible geometry is what enables simple parametrizations

* Paraglider component models parametrized by basic technical data

* Paraglider system models that accept all three primary control inputs:
  brakes, accelerator, and weight shift (direct manipulation of the risers and
  individual lines is not supported)

* Stateless models that are easy understand, use, extend, and test


Aerodynamics
^^^^^^^^^^^^

* Nonlinear aerodynamics using a fast *nonlinear lifting-line theory* model

  .. NLLT is fast!

* Graceful accuracy degradation near stall (important for paragliders, which
  frequently operate at relatively high angles of attack)

* Supports non-uniform wind fields and non-uniform wind vectors along the
  aerodynamic surfaces, enabling simulations involving local wind sheer,
  thermal lift and sink, wing rotation, etc.

* Includes paraglider system dynamics models that account for *apparent mass*
  (nonlinear dynamics due to the motion of an object through a fluid that
  become more significant for low-density volumes such as parafoils)


Usability
^^^^^^^^^

* Flight dynamics are encoded as numerical state derivatives, providing
  a simple model interface

* Includes a rudimentary simulator for generating flight trajectories from
  predefined wing control and wind field inputs

* Open source with a permissive license, built using the Python scientific
  computing stack (NumPy, SciPy, and Numba). Usable from within embedded Python
  interpreters, such as the Python consoles in Blender and FreeCAD.

.. Who is the target audience?

   People interested in understanding wing behavior (static foil performance,
   dynamic wing response, etc), people needing so simulate flights (developing
   control systems, performing flight reconstruction, etc)


Non-features
------------

.. What are its non-goals?

Equally important is what it does not do:

* The library is not suitable for studying behaviors involving turbulence, wing
  deformations, rapid wing maneuvers, or post-stall behaviors.

* The library is not a structural simulator. The models are not adequate for
  designing or testing the structural integrity of a design; they assume
  rigid-body dynamics without estimating internal forces. (For example, it
  neglects suspension line tension, cell billowing, canopy wrinkling, etc.)

* The library is not focused on engineering control input sequences or wind
  field models. The bundled simulator is for executing simulations given some
  inputs, not for designing them.

* The library is not focused on execution time. Although speed is important,
  the design prioritizes readability, functionality, and flexibility.


Installation
------------

To install the latest release of the package into a Python virtual environment:

.. code-block:: bash

   $ pip install pfh.glidersim

Or, for the latest development version:

.. code-block:: bash

   $ pip install git+https://github.com/pfheatwole/glidersim


Documentation
-------------

* **Design overview**: for the background and architecture of the library,
  refer to the `design guide
  <https://pfheatwole.github.io/glidersim/design.html>`__.

* **Examples**: to use the library to build glider models or run flight
  simulations, refer to the `usage guide
  <https://pfheatwole.github.io/glidersim/usage.html>`__.

* **API**: to program using the library, refer to the `library reference
  <https://pfheatwole.github.io/glidersim/reference.html>`__.

* **Derivations**: for a deeper discussion of the modeling choices,
  mathematical derivations, and literature references, refer to my thesis:
  `Parametric Paraglider Modeling <https://pfheatwole.github.io/thesis/>`__.


Disclaimer
----------

.. State of the software

This software has been stable for my purposes, but should be considered "alpha"
quality. The design (including the API) needs more users to test it before it
could be considered stable. Also the estimates produced by the nonlinear
aerodynamics method needs more verification.
