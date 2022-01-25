pfh.glidersim
=============

.. What is it?

A Python library for estimating the flight dynamics of gliding aircraft from
simple parametric models.

* **Source code**: https://github.com/pfheatwole/glidersim/

* **License**: permissive (`MIT
  <https://github.com/pfheatwole/pfh.glidersim/tree/main/LICENSE.txt>`__)

* **Documentation**: https://pfheatwole.github.io/glidersim/

* **Questions**: for general questions, please use the `discussion forum
  <https://github.com/pfheatwole/glidersim/discussions>`__.

* **Support**: to report a problem, please use the `issue tracker
  <https://github.com/pfheatwole/glidersim/issues>`__.


Background
----------

.. What is it?

This library began as an implementation of my thesis: `Parametric Paraglider
Modeling <https://pfheatwole.github.io/thesis/>`__. Its motivation was to
enable statistical flight reconstruction of the wind fields present during
recorded paraglider flights. The task would require a flight dynamics model of
the glider that produced the flight record, so the objective of the paper was
to estimate the flight dynamics of commercial paraglider wings using only basic
technical data. Achieving that goal required decomposing a paraglider system
into a set of components that allowed for convenient model parametrizations.


.. How is it used?

.. Who is it for?

.. Who is its intended audience?

Although development focused on paraglider dynamics, the model decomposition
makes the library suitable for general gliding aircraft such as kites and hang
gliders. To encourage its use in common engineering applications (such as
control modeling and statistical filtering), the glider dynamics are encoded as
standard differential equations (``ẋ = f(x, u)``).


.. FIXME: how does it compare to existing simulators? What makes it special?



Key Features
------------

.. Features are *what* it does, not *how* it works.


.. FIXME: rewrite lists for parallel structure; easier to read


Modeling
^^^^^^^^

* Novel wing geometry model optimized for nonlinear designs

  .. This flexible geometry is what enables simple parametrizations

* Paraglider component models parametrized by basic technical data

* Paraglider system models that accept all three primary control inputs:
  brakes, accelerator, and weight shift. (Direct manipulation of the risers and
  individual lines is not supported.)

* Models are stateless, making them easier to understand, use, extend, and test


Aerodynamics
^^^^^^^^^^^^

* Nonlinear aerodynamics for the canopy using nonlinear lifting-line theory

* Demonstrates graceful accuracy degradation near stall (important for
  paragliders, which frequently operate at relatively high angles of attack)

* Supports non-uniform wind fields and non-uniform wind vectors along the
  aerodynamic surfaces, enabling simulations involving local wind sheer,
  thermal lift and sink, wing rotation, etc.

* Includes system dynamics models that account for *apparent mass* (nonlinear
  dynamics due to the motion of an object through a fluid that become more
  significant for low-density volumes such as parafoils)


Usability
^^^^^^^^^

* Includes a simple simulator for generating flight trajectories from
  [[custom]] wing control and wind field inputs.

* Open source with a permissive license, built using the Python scientific
  computing stack (NumPy, SciPy, and Numba). Useable from within embedded
  Python interpreters, such as the Python consoles in Blender and FreeCAD.

.. Who is the target audience?

   People interested in understanding wing behavior (static foil performance,
   dynamic wing response, etc), people needing so simulate flights (developing
   control systems, performing flight reconstruction, etc)


Non Features
------------

.. What are its non-goals?

Equally important is what it does not do:

* The library is not suitable for studying behaviors involving turbulence, wing
  deformations, rapid wing maneuvers, or post-stall behaviors.

* The library is not a structural simulator. The models are not adequate for
  designing or testing the structural integrity of a design; they assume
  rigid-body dynamics without estimating internal forces. (For example, it
  neglects suspension line tension, cell billowing, canopy wrinkling, etc.)

* The library is not focused on designing control input sequences or wind
  fields. Although it provides a simulator and helper functions for generating
  piecewise-linear control inputs, that is not its focus.


Installation
------------

To install the latest release of the package into a Python virtual environment:

.. code-block:: bash

   $ pip install pfh.glidersim

Or, for the latest development version:

.. code-block:: bash

   $ pip install git+https://github.com/pfheatwole/glidersim


Examples
--------

For examples of how to use the library to build glider models and to simulate
flights, refer to the `usage guide
<https://pfheatwole.github.io/glidersim/guide>`__ and `bundled scripts
<https://github.com/pfheatwole/glidersim/tree/main/scripts>`__ included with the
library source.

For example, to download a copy of the library, install it in a new virtual
environment, and execute a sample script:

.. code-block::

   $ git clone https://github.com/pfheatwole/glidersim.git
   $ cd glidersim
   $ python -m venv .venv
   $ source .venv/bin/activate
   $ pip install .
   $ cd scripts
   $ python sim_work.py


Documentation
-------------

Complete HTML documentation and API reference is available at
https://pfheatwole.github.io/glidersim/

For an overview of the library, refer to the `design guide
<https://pfheatwole.github.io/glidersim/design/>`__.

For an explanation of the component models, how to assemble them, and how to
estimate the resulting aerodynamics, refer to the `usage guide
<https://pfheatwole.github.io/glidersim/usage/>`__.

For a deeper discussion of the modeling choices, mathematical derivations, and
academic references, refer to my thesis: `Parametric Paraglider Modeling
<https://pfheatwole.github.io/thesis>`__.


Disclaimer
----------

.. State of the software

This software has been stable for my purposes, but should be considered "alpha"
quality. The design (including the API) needs more users to test it before it
could be considered stable. Also the estimates produced by the nonlinear
aerodynamics method needs more verification. (Current validation is only for
static scenarios with a single parafoil in a wind tunnel; refer to :ref:`case
study <thesis:foil_aerodynamics:case study>` for a discussion, and the
`associated script
<http://github.com/pfheatwole/thesis/source/figures/paragglider/belloc/belloc.py>`__
from my thesis for the test sources.)
