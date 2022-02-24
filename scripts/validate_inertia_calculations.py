"""
Alternative method for estimating the canopy inertial properties. Very ugly and
hacky, but still useful for checking the much improved mesh-based method.
"""

import numpy as np
from scipy.integrate import simps

import pfh.glidersim as gsim


# Yanked from from `AirfoilGeometry`
def airfoil_mass_properties(airfoil, r_upper=0, r_lower=0, N=200):
    """
    Calculate the inertial properties for the curves and planar area.

    These unitless magnitudes, centroids, and inertia matrices can be
    scaled by the physical units of the target application in order to
    calculate the upper and lower surface areas, internal volume, and
    inertia matrix of a 3D wing.

    This procedure treats the 2D geometry as perfectly flat 3D objects,
    with a new `z` axis added according to the right-hand rule. See
    "Notes" for more details.

    Parameters
    ----------
    r_upper, r_lower : float
        The starting coordinates of the upper and lower surfaces. Requires
        that `-1 <= r_lower <= r_upper, 1`.
    N : integer
        The number of chordwise sample points. Used to create the vertical
        strips for calculating the area, and for creating line segments of
        the parametric curves for the upper and lower surfaces.

    Returns
    -------
    dictionary
        upper_length : float
            The total length of the upper surface curve
        upper_centroid : array of float, shape (2,)
            The centroid of the upper surface curve as (x, y) in acs
        upper_inertia : array of float, shape (3,3)
            The inertia matrix of the upper surface curve
        area : float
            The area of the airfoil
        area_centroid : array of float, shape (2,)
            The centroid of the area as (x, y) in acs
        area_inertia : array of float, shape (3,3)
            The inertia matrix of the area
        lower_length : float
            The total length of the lower surface curve
        lower_centroid : array of float, shape (2,)
            The centroid of the lower surface curve as (x, y) in acs
        lower_inertia : array of float, shape (3,3)
            The inertia matrix of the lower surface curve

        These are unitless quantities. The inertia matrices for each
        component are for rotations about that components' centroid.

    Notes
    -----
    In traditional airfoil definitions, the positive x-axis lies along the
    chord, directed from the leading edge to the trailing edge, and the
    positive y-axis points towards the upper surface.

    Here, a z-axis that satisfies the right hand rule is added for the
    purpose of creating a well-defined inertia matrix. Let this set of axes
    be called the "airfoil coordinate system" (acs).

    Translating these acs coordinates into the front-right-down (frd)
    coordinate system requires reordering and reversing the direction of
    vector components. To convert acs -> frd: [x, y, z] -> [-x, -z, -y]

    In terms of code, to convert from acs to frd coordinates:

    >>> C = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    >>> centroid_frd = C @ [*centroid_acs, 0]  # Augment with z_acs=0
    >>> inertia_frd = C @ inertia_acs @ C
    """
    if r_lower < -1:
        raise ValueError("Required: r_lower >= -1")
    if r_lower > r_upper:
        raise ValueError("Required: r_lower <= r_upper")
    if r_upper > 1:
        raise ValueError("Required: r_upper <= 1")

    # -------------------------------------------------------------------
    # 1. Area calculations

    r = (1 - np.cos(np.linspace(0, np.pi, N))) / 2  # `0 <= r <= 1`
    top = airfoil.profile_curve(r).T  # Top half (above r = 0)
    bottom = airfoil.profile_curve(-r).T  # Bottom half (below r = 0)
    Tx, Ty = top[0], top[1]
    Bx, By = bottom[0], bottom[1]

    area = simps(Ty, Tx) - simps(By, Bx)
    xbar = (simps(Tx * Ty, Tx) - simps(Bx * By, Bx)) / area
    ybar = (simps(Ty**2 / 2, Tx) + simps(By**2 / 2, Bx)) / area
    area_centroid = np.array([xbar, ybar])

    # Area moments of inertia about the origin
    # FIXME: verify, especially `Ixy_o`. Check airfoils where some `By > 0`
    Ixx_o = 1 / 3 * (simps(Ty**3, Tx) - simps(By**3, Bx))
    Iyy_o = simps(Tx**2 * Ty, Tx) - simps(Bx**2 * By, Bx)
    Ixy_o = 1 / 2 * (simps(Tx * Ty**2, Tx) - simps(Bx * By**2, Bx))

    # Use the parallel axis theorem to find the inertias about the centroid
    Ixx = Ixx_o - area * ybar**2
    Iyy = Iyy_o - area * xbar**2
    Ixy = Ixy_o - area * xbar * ybar
    Izz = Ixx + Iyy  # Perpendicular axis theorem

    # Inertia matrix for the area about the origin
    # fmt: off
    area_inertia = np.array([
        [ Ixx, -Ixy,   0],  # noqa: E201, E241
        [-Ixy,  Iyy,   0],  # noqa: E201, E241
        [   0,    0, Izz],  # noqa: E201, E241
    ])
    # fmt: on

    # -------------------------------------------------------------------
    # 2. Surface line calculations

    su = np.linspace(r_upper, 1, N)
    sl = np.linspace(r_lower, -1, N)
    upper = airfoil.profile_curve(su).T
    lower = airfoil.profile_curve(sl).T

    # Line segment lengths and midpoints
    norm_U = np.linalg.norm(np.diff(upper), axis=0)  # Segment lengths
    norm_L = np.linalg.norm(np.diff(lower), axis=0)
    mid_U = (upper[:, :-1] + upper[:, 1:]) / 2  # Segment midpoints
    mid_L = (lower[:, :-1] + lower[:, 1:]) / 2

    # Total line lengths and centroids
    upper_length = norm_U.sum()
    lower_length = norm_L.sum()
    upper_centroid = np.einsum("ij,j->i", mid_U, norm_U) / upper_length
    lower_centroid = np.einsum("ij,j->i", mid_L, norm_L) / lower_length

    # Surface line moments of inertia about their centroids
    # FIXME: not proper line integrals: treats segments as point masses
    cmUx, cmUy = upper_centroid
    mid_Ux, mid_Uy = mid_U[0], mid_U[1]
    Ixx_U = np.sum(mid_Uy**2 * norm_U) - upper_length * cmUy**2
    Iyy_U = np.sum(mid_Ux**2 * norm_U) - upper_length * cmUx**2
    Ixy_U = np.sum(mid_Ux * mid_Uy * norm_U) - upper_length * cmUx * cmUy
    Izz_U = Ixx_U + Iyy_U

    cmLx, cmLy = lower_centroid
    mid_Lx, mid_Ly = mid_L[0], mid_L[1]
    Ixx_L = np.sum(mid_Ly**2 * norm_L) - lower_length * cmLy**2
    Iyy_L = np.sum(mid_Lx**2 * norm_L) - lower_length * cmLx**2
    Ixy_L = np.sum(mid_Lx * mid_Ly * norm_L) - lower_length * cmLx * cmLy
    Izz_L = Ixx_L + Iyy_L

    # Inertia matrices for the lines about the origin
    # fmt: off
    upper_inertia = np.array([
        [ Ixx_U, -Ixy_U,     0],  # noqa: E201, E241
        [-Ixy_U,  Iyy_U,     0],  # noqa: E201, E241
        [     0,      0, Izz_U],  # noqa: E201, E241
    ])
    lower_inertia = np.array([
        [ Ixx_L, -Ixy_L,     0],  # noqa: E201, E241
        [-Ixy_L,  Iyy_L,     0],  # noqa: E201, E241
        [     0,      0, Izz_L],  # noqa: E201, E241
    ])
    # fmt: on

    properties = {
        "upper_length": upper_length,
        "upper_centroid": upper_centroid,
        "upper_inertia": upper_inertia,
        "area": area,
        "area_centroid": area_centroid,
        "area_inertia": area_inertia,
        "lower_length": lower_length,
        "lower_centroid": lower_centroid,
        "lower_inertia": lower_inertia,
    }

    return properties


# Yanked from `SimpleFoil`
def canopy_mass_properties(canopy, amp, N=250):
    """
    Compute the quantities that control inertial behavior.

    (This method is deprecated by the new mesh-based method, and is only
    used for sanity checks.)

    The inertia matrices returned by this function are proportional to the
    values for a physical wing, and do not have standard units. They must
    be scaled by the wing materials and air density to get their physical
    values. See "Notes" for a thorough description.

    Returns
    -------
    dictionary
        upper_area: float [m^2]
            foil upper surface area
        upper_centroid: ndarray of float, shape (3,) [m]
            center of mass of the upper surface material in foil frd
        upper_inertia: ndarray of float, shape (3, 3) [m^4]
            The inertia matrix of the upper surface
        volume: float [m^3]
            internal volume of the inflated foil
        volume_centroid: ndarray of float, shape (3,) [m]
            centroid of the internal air mass in foil frd
        volume_inertia: ndarray of float, shape (3, 3) [m^5]
            The inertia matrix of the internal volume
        lower_area: float [m^2]
            foil lower surface area
        lower_centroid: ndarray of float, shape (3,) [m]
            center of mass of the lower surface material in foil frd
        lower_inertia: ndarray of float, shape (3, 3) [m^4]
            The inertia matrix of the upper surface

    Notes
    -----
    The foil is treated as a composite of three components: the upper
    surface, internal volume, and lower surface. Because this class only
    defines the geometry of the foil, not the physical properties, each
    component is treated as having unit densities, and the results are
    proportional to the values for a physical wing. To compute the values
    for a physical wing, the upper and lower surface inertia matrices must
    be scaled by the aerial densities [kg/m^2] of the upper and lower wing
    surface materials, and the volumetric inertia matrix must be scaled by
    the volumetric density [kg/m^3] of air.

    Keeping these components separate allows a user to simply multiply them
    by different wing material densities and air densities to compute the
    values for the physical wing.

    The calculation works by breaking the foil into N segments, where
    each segment is assumed to have a constant airfoil and chord length.
    The airfoil for each segment is extruded along the segment span using
    the perpendicular axis theorem, then oriented into body coordinates,
    and finally translated to the global centroid (of the surface or
    volume) using the parallel axis theorem.

    Limitations:

    * Assumes a constant airfoil over the entire foil.

    * Assumes the upper and lower surfaces of the foil are the same as the
      upper and lower surfaces of the airfoil. (Ignores air intakes.)

    * Places all the segment mass on the section bisecting the center of
      the segment instead of spreading the mass out along the segment span,
      so it underestimates `I_xx` and `I_zz` by a factor of `\int{y^2 dm}`.
      Doesn't make a big difference in practice, but still: it's wrong.

    * Requires the AirfoilGeometry to compute its own `mass_properties`,
      which is an extra layer of mess I'd like to eliminate.
    """
    s_nodes = np.cos(np.linspace(np.pi, 0, N + 1))
    s_mid_nodes = (s_nodes[1:] + s_nodes[:-1]) / 2  # Segment midpoints
    nodes = canopy.surface_xyz(s_nodes, 0, 0.25, surface="chord")  # Segment endpoints
    node_chords = canopy.chord_length(s_nodes)
    chords = (node_chords[1:] + node_chords[:-1]) / 2  # Mean average
    T = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])  # acs -> frd
    u = canopy.section_orientation(s_mid_nodes)
    u_inv = np.linalg.inv(u)

    # Segment centroids
    airfoil_centroids = np.array(
        [
            [*amp["upper_centroid"], 0],
            [*amp["area_centroid"], 0],
            [*amp["lower_centroid"], 0],
        ]
    )
    segment_origins = canopy.surface_xyz(s_mid_nodes, 0, 0, surface="chord")
    segment_upper_cm, segment_volume_cm, segment_lower_cm = (
        np.einsum("K,Kij,jk,Gk->GKi", chords, u, T, airfoil_centroids)
        + segment_origins[None, ...]
    )

    # Scaling factors for converting 2D airfoils into 3D segments.
    # Approximates each segments' `chord * span` area as parallelograms.
    u_a = u[..., 0]  # The chordwise ("aerodynamic") unit vectors
    dl = nodes[1:] - nodes[:-1]
    segment_chord_area = np.linalg.norm(np.cross(u_a, dl), axis=1)
    Kl = chords * segment_chord_area  # amp curve length into segment area
    Ka = chords**2 * segment_chord_area  # amp area into segment volume

    segment_upper_area = Kl * amp["upper_length"]
    segment_volume = Ka * amp["area"]
    segment_lower_area = Kl * amp["lower_length"]

    # Total surface areas and the internal volume
    upper_area = segment_upper_area.sum()
    volume = segment_volume.sum()
    lower_area = segment_lower_area.sum()

    # The upper/volume/lower centroids for the entire foil
    # fmt: off
    upper_centroid = (
        (segment_upper_area * segment_upper_cm.T).T.sum(axis=0)
        / upper_area
    )
    volume_centroid = (
        (segment_volume * segment_volume_cm.T).T.sum(axis=0)
        / volume
    )
    lower_centroid = (
        (segment_lower_area * segment_lower_cm.T).T.sum(axis=0)
        / lower_area
    )
    # fmt: on

    # Segment inertia matrices in body frd coordinates
    Kl, Ka = Kl.reshape(-1, 1, 1), Ka.reshape(-1, 1, 1)
    segment_upper_J = u_inv @ T @ (Kl * amp["upper_inertia"]) @ T @ u
    segment_volume_J = u_inv @ T @ (Ka * amp["area_inertia"]) @ T @ u
    segment_lower_J = u_inv @ T @ (Kl * amp["lower_inertia"]) @ T @ u

    # Parallel axis distances of each segment
    Ru = upper_centroid - segment_upper_cm
    Rv = volume_centroid - segment_volume_cm
    Rl = lower_centroid - segment_lower_cm

    # Segment distances to the group centroids
    R = np.array([Ru, Rv, Rl])
    D = (
        np.einsum("Rij,Rij->Ri", R, R)[..., None, None] * np.eye(3)
        - np.einsum("Rki,Rkj->Rkij", R, R)  # fmt: skip
    )
    Du, Dv, Dl = D

    # And finally, apply the parallel axis theorem
    upper_J = (segment_upper_J + (segment_upper_area * Du.T).T).sum(axis=0)
    volume_J = (segment_volume_J + (segment_volume * Dv.T).T).sum(axis=0)
    lower_J = (segment_lower_J + (segment_lower_area * Dl.T).T).sum(axis=0)

    mass_properties = {
        "upper_area": upper_area,
        "upper_centroid": upper_centroid,
        "upper_inertia": upper_J,
        "volume": volume,
        "volume_centroid": volume_centroid,
        "volume_inertia": volume_J,
        "lower_area": lower_area,
        "lower_centroid": lower_centroid,
        "lower_inertia": lower_J,
    }

    return mass_properties


if __name__ == "__main__":
    wing = gsim.extras.wings.niviuk_hook3(size=23)
    wing.canopy.sections.intakes = wing.canopy.sections._no_intakes
    using_mesh = wing.canopy.mass_properties(1001, 1001)
    amp = airfoil_mass_properties(gsim.airfoil.NACA(24018))
    using_slices = canopy_mass_properties(wing.canopy, amp, N=20000)
    for d in using_mesh:
        print(d)
        print("Mesh:\n", using_mesh[d])
        print("Slices:\n", using_slices[d])
        print()

    breakpoint()
