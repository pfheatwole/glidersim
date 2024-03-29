.. The purpose of this document is explanation, not marketing. It should
   quickly build an understanding of how the source files are structured and
   how models are composed.


Design
======

This library is motivated by a need for flight dynamics models encoded as
simple systems of differential equations:

.. math::

   \dot{\vec{x}} = f(\vec{x}, \vec{u})

(where :math:`\vec{x}` is a vector of state variables and :math:`\vec{u}` is
a vector of system inputs).

[[Typical aircraft simulators expose a complicated model interface, making them
difficult to use and extend. Instead, deliberately targeting numerical
derivatives makes the flight dynamics models directly usable in common
applications such as control modeling and statistical filtering.]]


Background
----------

This library began as an implementation of my thesis: `Parametric Paraglider
Modeling <https://pfheatwole.github.io/thesis/>`__. Its motivation was to
enable statistical flight reconstruction of the wind fields present during
recorded paraglider flights. The task would require a flight dynamics model of
the glider that produced the flight record, so the objective of the paper was
to estimate the flight dynamics of commercial paraglider wings using only basic
technical data. Achieving that objective required decomposing the paraglider
system into components that enabled convenient model parametrizations. Although
development focused on paraglider dynamics, the model decomposition makes the
library suitable for general gliding aircraft such as kites and hang gliders.


Design overview
---------------

Creating a flight dynamics model using this library means implementing
a :py:class:`StateDynamics <pfh.glidersim.simulator.StateDynamics>` class that
defines a set of state variables and computes numerical state derivatives that
can be integrated over time to produce a flight trajectory.

Because gliding aircraft are complex systems, it is easier to estimate their
dynamics if they are decomposed into composite models, leading to a natural
hierarchy that builds up to the state dynamics to record the behavior of the
system.


Model hierarchy
---------------

The models in this library can be categorized into three groups:

1. Components

2. Systems

3. State variables


Components
^^^^^^^^^^

Component models define the control inputs of the individual components of the
glider system, their inertial properties, and the resultant forces that act on
them. The choice of components is whatever decomposition makes sense for
a specific system dynamics model.

.. FIXME: Examples are...


Systems
^^^^^^^

System models define the composite behavior produced by combining the component
models. They calculate whatever physical rates of change are necessary to
describe the system dynamics, such as translational and angular acceleration.

.. FIXME: Examples are...


State variables
^^^^^^^^^^^^^^^

`State variables <https://en.wikipedia.org/wiki/State_variable>`__ are numbers
that summarize the current "state" of a dynamical system, such as position and
velocity. The time derivatives of the state variables, called the *state
dynamics*, describe how the state of the system is reacting to the current
state and any system inputs. Flight simulation is performed by integrating the
state dynamics over time to produce a *state trajectory*: a record of the
dynamic behavior of the system over time. A state dynamics model is responsible
for choosing a set of state variables and relating their derivatives to the
derivatives generated by the system dynamics.

.. In this implementation they also "own" the inputs to the model, but that's
   not strictly necessary; it's just what I found convenient. The simulator is
   completely unaware of these choices beyond recording the numerical state
   values over time. The `StateDynamics` handle all inputs to the system model;
   inputs are internal to ("owned by") the `StateDynamics`.

At first glance the system dynamics and state dynamics can seem equivalent, but
differentiating the two adds significant freedom to system model design because
it separates the system dynamics from the representation of system state. For
example, the state dynamics may record orientation using Euler angles,
quaternions, or some other encoding; the system dynamics should not depend on
the choice of encoding. Similarly, the system dynamics should not depend on
position, nor should they depend on the choice of global coordinate system; the
system models are free to define all quantities in local, body-fixed coordinate
systems, and let the state dynamics choose representations that suit the
simulation applications.

Also, the state dynamics provide the interface between the system dynamics and
the simulator. Although the models included with this library are designed for
use with the (rudimentary) bundled simulator, alternative state dynamics
classes could — in theory — expose the same system models to more comprehensive
flight simulators such as FlightGear.

.. In other words, a state dynamics model is a wrapper: it provides an
   interface between the simulator and the underlying system dynamics. The bulk
   of the work is in defining the system dynamics. Most of the bits included
   with this library are for composing paraglider system models.

.. FIXME: Examples are...
