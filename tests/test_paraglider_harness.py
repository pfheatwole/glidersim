import numpy as np
import pytest

import pfh.glidersim as gsim


@pytest.fixture
def harness():
    return gsim.paraglider_harness.Spherical(
        mass=1,
        z_riser=1,
        S=2,
        CD=1,
        kappa_w=1,
    )


def test_1(harness):
    # No drag, no weight
    f, g = harness.resultant_force(
        delta_w=0,
        v_W2h=[0, 0, 0],
        rho_air=0,
        g=[0, 0, 0],  # No gravity
        r_R2RM=[0, 0, 0],  # R = RM
    )
    assert np.allclose(f, [0, 0, 0])
    assert np.allclose(g, [0, 0, 0])


def test_2(harness):
    # Drag straight backwards, no weight
    f, g = harness.resultant_force(
        delta_w=0,
        v_W2h=[-1, 0, 0],
        rho_air=1,
        g=[0, 0, 0],  # No gravity
        r_R2RM=[0, 0, 0],  # R = RM
    )
    assert np.allclose(f, [-1, 0, 0])
    assert np.allclose(g, [0, -1, 0])


def test_3(harness):
    f, g = harness.resultant_force(
        delta_w=0,
        v_W2h=[0, 0, 0],
        rho_air=1,
        g=[0, 0, 1],
        r_R2RM=[0, 0, 0],
    )
    assert np.allclose(f, [0, 0, 1])
    assert np.allclose(g, [0, 0, 0])


def test_4(harness):
    f, g = harness.resultant_force(
        delta_w=1,
        v_W2h=[0, 0, 0],
        rho_air=1,
        g=[0, 0, 1],
        r_R2RM=[0, 0, 0],
    )
    assert np.allclose(f, [0, 0, 1])
    assert np.allclose(g, [1, 0, 0])
