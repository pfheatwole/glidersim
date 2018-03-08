import abc

import numpy as np

class Glider:
    def __init__(self, wing, d_cg, h_cg, S_cg, Cd_cg):
        self.wing = wing
        self.d_cg = d_cg
        self.h_cg = h_cg
        self.S_cg = S_cg
        self.Cd_cg = Cd_cg

        # FIXME: pre-compute the `body->local` transformation matrices here?
        # FIXME: lots more to implement

    def blargh(self, y, state, control=None):
        # FIXME: Document
        # compute the local relative wind for wing sections
        U, V, W, P, Q, R = state

        # Compute the local section wind
        # PFD eqs 4.10-4.12, p73
        uL = U + self.geometry.fz(y)*Q - y*R
        vL = V - self.geometry.fz(y)*P + self.geometry.fx(y)*R
        wL = W + y*P - self.geometry.fx(y)*Q

        # PFD eqs 4.14-4.15, p74
        ui = U
        wi = W*cos(deltas)

        delta = self.geometry.delta(y)  # PFD eq 4.13, p74
        theta = self.geometry.ftheta(y)  # FIXME: should include braking
        alpha_i = arctan(wi/ui) + theta  # PFD eq 4.17, p74

        VT_i = ui**2 + wi**2
        Fx_i, Fy_i, Fz_i, my_i = self.section_forces(ys, VT_i, alpha_i)

        # The real purpose of this function seems to be calculating the
        # local relative wind. It doesn't depend on the control at all,
        # only on the state, airfoil geometry, and position on the span.
        #
        # So something more like `local_wind` seems appropriate
        #   local_wind(y, state) -> ui, wi
        #
        # But hang on: relative wind calculations need information about
        # the position of the wing to the glider. So this function really
        # belonds with the Glider...
        #
        # For the purposes of calculating the global coefficients, maybe its
        # better to just hand-roll {ui, wi}?
        #
        # The focus should probably be on simplifying everything in Wing to
        # be given the relative wind directly. For consistency, probably best
        # to parameterize everything in terms of {ui, wi}, I think.

        return Fx_i, Fy_i, Fz_i, my_i

