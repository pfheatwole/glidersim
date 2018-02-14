class Integrator:
    """A functor that implement a numerical integrator (eg, RK4)"""

    def __init__(self, equations, dt):
        """Configure the integrator using fixed time steps

        Parameters
        ----------
        equations : list of functions
            A list of `n` differential equations for for each of the`n` states
        dt : float
            the time step for the numerical integration
        """
        self.equations = equations
        self.dt = dt
        print("DEBUG> setting dt =", dt)
        print("Finished Integrator.__init")

    def __call__(self, state):
        """Calculate the transition dynamics to calculate the next state"""
        return [self.integrate(eqn, state) for eqn in self.equations]

    def integrate(self, eqn, state):
        raise NotImplementedError("Unavailable in the abstract integrator")


class Euler(Integrator):
    """Implements a first-order (Euler) forward integrator"""

    def integrate(self, eqn, state):
        """Use the Euler method to integrate the differential equation.

        Parameters
        ----------
        de : function
            The differential equation to be integrated
        state : array-like
            The current state values
        """
        next_state = state + eqn(state)*self.dt
        return next_state


class Wing:
    """Base class for different paragliding wings

    There are two types of wing: 2D and 3D

    Wings objects are responsible for:
     1. Current state
     2. System dynamics (the differential equations that control transitions)

    Wing objects are not responsible for:
     * Tracking past states (trajectories)
     * The numerical integration necessary for the transitions
    """

    def __init__(self, im):
        self.equations = None  # FIXME: the set of DEs for the integrator
        self._im = im(self.equations)

    def predict(self, state, u):
        """Calculate a future state using the system dymaics"""
        return self._im(state, u)

    def update(self, u):
        """Transition the current state using the prediction"""
        self.state = self.predict(self.state, u)


class Airfoil(Wing):
    """Implement a 2D wing as a simple airfoil + pilot mass system

    FIXME: use the gnulab3 to calculate the lift and drag forces
    FIXME: add pitch angle (phi) for rotations

    """

    def update():
        pass


if __name__ == "__main__":
    x = 1
    im = Euler([lambda x: 2*x], 0.1)
    for n in range(20):
        x = im(x)[0]
        print(n, ":", x)
