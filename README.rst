pfh.glidersim
=============

.. Summary: what is it?

A parafoil design tool and simulator written in python.

Written to support `my thesis <https://pfheatwole.github.io/thesis>`_.
I needed a paraglider dynamics model suitable for use in statistical state
estimation.


Purpose
-------

.. What problems is it intended to solve?

This project was motivated by several, specific tasks:

* To enable usefully-accurate approximations of existing paraglider wings with
  as little information and effort as possible

* To provide direct access to state derivatives for flight simulation

* To simulate specific flight scenarios (eg, off-center thermal interactions)

.. Who are the intended users?

People interested in understanding wing behavior (static foil performance,
dynamic wing response, etc), people needing so simulate flights (developing
control systems, performing flight reconstruction, etc)


Functionality
-------------

.. What tasks is it intended to perform?

[[Designing parafoil wings and paraglider wings, analyzing wing performance,
simulating flights, etc.]]


Features
--------

.. What tools does it provide to accomplish the desired tasks? What makes it
   special?

.. Flight simulators already exist; why another one? Paraglider models already
   exist: why another one? **What does this code bring to the table?**

* Versus existing flight simulators:

  * Allow creating wings from basic geometric specifications

    Existing flight sims (like FlightGear) assume you already have all the
    information about the wing; this tool enables **creating** the wing in the
    first place from basic geometric descriptions.

  * The purpose of a glider model is to define a dynamics function that
    follows a strict, minimalist, math-centric interface:

    :math:`\dot{x} = f(x, u)`

    Where :math:`x` is the model state, and :math:`u` are the model inputs.
    Maintaining the standard mathematical form of a differential equation
    makes it much easier to create tools using the dynamics. The primary
    motivation of this project was statistical state estimation; exposing the
    wing dynamics as a differential equation makes them directly usable for
    defining the transition function. [[**So the ultimate purpose of this
    library is to help build tools that need paraglider dynamics?**]]

    [[Keep the project focused; provide the dynamics as a simple math
    function, and let users build the tools on top.]]

  * Allow non-linear aerodynamics

    Useful for designing wing models: with a full non-linear model you can
    check if a linear model would suffice and create the linear model at some
    chosen operating point.

    Useful for detailed flight scenarios: what happens when a parafoil
    encounters asymmetric wind, such as an off-center thermal interaction?

* Doesn't rely on proprietary tools (like MATLAB)

* Minimal python dependencies. I'd like it to be accessible inside as many
  python environments as possible, such as those inside embedded interpreters
  (Blender, FreeCAD, etc).


Usage
-----

.. How is it used? (How do you interface with it?)


Installation
^^^^^^^^^^^^

[[FIXME]]


Examples
^^^^^^^^

[[FIXME: link to the scripts, and maybe the demos in my thesis?]]


Development
-----------


Design goals
^^^^^^^^^^^^

.. What are the guiding design principles?

* Avoid deep structure. Flat is preferred.

* Stateless models.

* Code that implements math should mirror the math.

  At least, as closely as reasonably possible. It is much easier to use and
  extend results from literature if their implementation matches the published
  work. It should be clear where equations come from.


Source
^^^^^^

The complete source code is available under a permissible MIT license:
https://github.com/pfheatwole/pfh.glidersim/


Support
^^^^^^^

* `Issue tracker <https://github.com/pfheatwole/pfh.glidersim/issues>`_

* FAQ?


License
^^^^^^^

This project is licensed under the permissible MIT license.
