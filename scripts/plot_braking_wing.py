import numpy as np
from numpy import cumsum, diff, einsum, sin, cos, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; for `projection='3d'`

from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

import Airfoil
import BrakeGeometry
import Parafoil
import ParagliderWing


def build_deformed_curves(geo, theta, epsilon, Ku, Kl, d):
    """Returns a spline for the deformed lower surface

    Parameters
    ----------
    geo : AirfoilGeometry
    theta : float [radians]
        The tangent angle between the two upper circles, whose radii are `ru`
        and `Ru`
    epsilon : float [radians]
        The minimum angle between the second circle and the vertical axis at
        the new trailing edge (eg, epsilon=0 would end with a vertical segment)
    Ku : float
        Scaling factor for the second radius of the upper curve, so
        `Ru = Ku*ru`, where `ru` is the radius of the first upper circle
    Kl : float
        Scaling/smoothing factor for smoothing the lower curvature, so
        `Rl = Kl*ru`
    d : where to start deforming the upper curve


    The procedure:
     * Setup: build a spline that maps the distance `d` to position `x`
     * Given `d`, compute dy/dx at that point
     * Given dy/dx, compute `theta_d`: the tangent point on a circle
     * Fit a circle of size `rl` tangent to `d`
     * Fit a line tangent to that circle at an angle `epsilon`
     * Sample each section and build a spline over the entire deformed surface
    """
    epsilon = np.deg2rad(epsilon)
    theta = np.deg2rad(theta)

    N = 2000
    s = (1 - np.cos(np.linspace(0, np.pi, N)))[::-1] / 2  # From TE->LE

    # Map the surface distances `d` to their xy coordinates for each surface
    points_u = geo.surface_curve(s)
    points_l = geo.surface_curve(-s)
    distances_u = np.r_[0, cumsum(norm(diff(points_u.T), axis=0))]
    distances_l = np.r_[0, cumsum(norm(diff(points_l.T), axis=0))]
    d2xy_u = PchipInterpolator(distances_u, points_u)
    d2xy_l = PchipInterpolator(distances_l, points_l)
    # print("Total upper length:", distances_u[-1])
    # print("Total lower length:", distances_l[-1])

    if d == 0:  # Skip nonexistant deformations
        N = 125
        s = (1 - np.cos(np.linspace(0, np.pi, N)))[::-1] / 2  # From TE->LE
        points_u = geo.surface_curve(s)
        points_l = geo.surface_curve(-s)
        return points_u, points_l[:-1][::-1]  # Skip duplicates at `s == 0`

    # At what angle is a circle tangent to the upper surface curve at `d`?
    dxy_ds = geo.surface_curve_normal(1 - d)
    theta_d = np.arctan2(dxy_ds[1], dxy_ds[0])  # Pointing inwards, typically
    if theta_d < 0:
        # The tangent point should be on the upper half of the first circular
        # arc, so the angle must be pointing outwards
        theta_d += np.pi
    # print("Lower tangent angle:", np.rad2deg(theta_d))

    # Fit a circle that is tangent at that point
    #  * rr: the radius
    #  * cr: the center
    rr = (d * distances_u[-1]) / (theta_d + (Ku-1)*theta - Ku*epsilon)
    tangent_point = geo.surface_curve(1 - d)
    cr = tangent_point + [rr*np.sin(theta_d - pi/2), -rr*np.sin(theta_d)]

    # print("DEBUG> rr:", rr)

    # You end up with three sections:
    #  1. The undeformed curve from the start to the point `d`
    #  2. A circular arc `r`, tangent to the undeformed upper curve and to the
    #     second arc `R`
    #  3. A circular arc `R` from the circles' tangent point to the end of the
    #     total length `d`

    # Section 1
    s1 = (1 - d) * (1 - np.cos(np.linspace(0, np.pi/2, 100, endpoint=False)))
    section1 = geo.surface_curve(s1)

    # Section 2
    if theta_d > theta:  # Is there a small circle?
        t = np.linspace(theta_d, theta, 25, endpoint=False)
        section2 = np.array([cr[0] + rr*np.cos(t), cr[1] + rr*np.sin(t)]).T
    else:
        section2 = None
        print("\nDEBUG> theta_d <= theta?")

    # Section 3
    if theta > theta_d:
        theta = theta_d
    cR = cr - (Ku-1) * rr * np.array([cos(theta), sin(theta)])
    t = np.linspace(theta, epsilon, 25)
    section3 = cR + Ku * rr * np.array([cos(t), sin(t)]).T

    if section2 is not None:
        points_u = np.vstack((section1, section2, section3))
    else:
        points_u = np.vstack((section1, section3))

    # Reverse the order so `points_u` goes from TE->LE
    points_u = points_u[::-1]

    # Rebuild the d2xu curve for warped upper curve
    distances_u = np.r_[0, cumsum(norm(diff(points_u.T), axis=0))]
    d2xy_u = PchipInterpolator(distances_u, points_u)
    # print("Total upper length after warping:", distances_u[-1])

    # ----------------------------------------
    # Now, build a smoothly deformed lower surface curve

    # Step 1: find where the deformed upper curve crosses the lower curve
    def find_crossing(dudl):
        du, dl = dudl
        # return norm(d2xy_u(du) - d2xy_l(dl))
        return d2xy_u(du) - d2xy_l(dl)
    Du, Dl = fsolve(find_crossing, x0=np.array((d, d)))

    # Du/Dl are the crossing points for the upper/lower curves
    # print(f"DEBUG> crossings at {Du}/{Dl}")

    # Yay, now I know where they both cross. On to the smoothing!

    # So, what I'm going to do is move "forward" a smoothing factor or so
    # along the lower curve. Everything forward of that point stays as is.

    # The "smoothing factor" how far in front of Dl to smooth
    # s = rr / Kl / Ku
    sf = rr * Kl

    # Section 1
    s1 = -(1 - (Dl + sf)) * (1 - np.cos(np.linspace(0, np.pi/2, 100)))
    section1 = geo.surface_curve(s1)

    # Section 2
    # Use a pseudo-Bezier curve to smooth points from Dl+s to Dl (and Du..Du+s)
    points2 = []
    for t in np.linspace(0, 1, 50):
        xy_u = d2xy_u(Du * (1-t))
        xy_l = d2xy_l(Dl + sf*(1 - t))
        points2.append(xy_l + t*(xy_u - xy_l))
    section2 = np.asarray(points2)

    points_l = np.vstack((section1, section2))

    return points_u, points_l


def build_elliptical_parafoil(b_flat, taper, dMed, sMed, airfoil,
                              SMC=None, MAC=None,
                              dMax=None, sMax=None,
                              torsion_max=0, torsion_exponent=6):

    if SMC is None and MAC is None:
        raise ValueError("One of the SMC or MAC are required")

    if dMed > 0 or (dMax is not None and dMax > 0):
        raise ValueError("dihedral must be negative")

    if sMed < 0 or (sMax is not None and sMax < 0):
        raise ValueError("sweep must be positive")  # FIXME: why?

    if dMax is None:
        dMax = 2*dMed - 1  # ref page 48 (56)
        print(f"Using minimum max dihedral ({dMax})")

    if sMax is None:
        sMax = (2*sMed) + 1  # ref page 48 (56)
        print(f"Using minimum max sweep ({sMax})")

    if SMC is not None:
        c0 = Parafoil.EllipticalPlanform.SMC_to_c0(SMC, taper)
    else:
        c0 = Parafoil.EllipticalPlanform.MAC_to_c0(MAC, taper)

    planform = Parafoil.EllipticalPlanform(
        b_flat, c0, taper, sMed, sMax, torsion_exponent, torsion_max)
    lobe = Parafoil.EllipticalLobe(dMed, dMax)

    return Parafoil.ParafoilGeometry(planform, lobe, airfoil)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    print("\n\n--------------------------------------------------------\n")

    # Build the base airfoil
    print("\nAirfoil: NACA4418, curving flap")
    # airfoil_geo = Airfoil.NACA(4418, open_TE=False, convention='vertical')
    airfoil_geo = Airfoil.NACA(23015, open_TE=False, convention='vertical')
    airfoil_coefs = Airfoil.GridCoefficients('polars/NACA4418_theta30_epsilon10_Ku4_Kl0.5_ver3.csv')
    delta_max = np.deg2rad(10.8)  # FIXME: magic number

    # Build a mapping: delta->d
    theta = 30
    epsilon = 10
    Ku = 4
    Kl = 0.5
    tmp = []
    for d in np.linspace(0, 0.5, 9):
        points_u, points_l = build_deformed_curves(
            airfoil_geo, theta, epsilon, Ku, Kl, d=d)
        delta = np.arctan2(-points_l[-1, 1], points_l[-1, 0])
        if abs(delta) < 1e-4:  # Kludge: floating point fix (avoid -0.00)
            delta = 0
        print("d: {:.2f}> delta: {:.2f} degrees".format(d, np.rad2deg(delta)))
        tmp.append((d, delta))
    tmp = np.asarray(tmp).T
    delta2d = PchipInterpolator(tmp[1], tmp[0])

    airfoil = Airfoil.Airfoil(airfoil_coefs, airfoil_geo)

    # Build the parafoil
    S_flat, b_flat, AR_flat = 23, 11.15, 5.40
    SMC_flat = b_flat/AR_flat
    S, b, AR = 19.55, 8.84, 4.00
    parafoil = build_elliptical_parafoil(   # Hook 3 (ish)
        b_flat=b_flat, SMC=SMC_flat, taper=0.35, dMed=-32, dMax=-75,
        # sMed=13.5, sMax=40, torsion_max=0, airfoil=airfoil)
        sMed=11.5, sMax=40, torsion_max=0, airfoil=airfoil)

    # Build the wing
    p_start, p_peak = 0, 0.75
    # p_start, p_peak = 0.00, BrakeGeometry.Cubic.p_peak_min(0.00)
    brakes = BrakeGeometry.Cubic(p_start, p_peak, delta_max)

    wing = ParagliderWing.ParagliderWing(parafoil, Parafoil.Phillips, brakes,
                                         d_riser=0.37, z_riser=6.8,
                                         pA=0.08, pC=0.80,
                                         kappa_s=0.15)

    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 16))
    ax = fig.gca(projection='3d')
    # ax.view_init(azim=-130, elev=25)

    N_sections = 81
    delta_Bl, delta_Br = 0.0, 1
    for s in np.linspace(-1, 1, N_sections):
        # print()
        delta = brakes(s, delta_Bl, delta_Br)
        d = delta2d(delta)
        print(f"DEBUG> s: {s:.3f}, delta: {delta:.3f}, d: {d:.3f}")
        points_u, points_l = build_deformed_curves(
            airfoil_geo, theta, epsilon, Ku, Kl, d=d)
        orientation = wing.parafoil.chords.orientation(s)
        points_u = points_u.T
        points_l = points_l.T

        points_u = np.array([-points_u[0], np.zeros(points_u.shape[-1]), -points_u[1]])
        points_l = np.array([-points_l[0], np.zeros(points_l.shape[-1]), -points_l[1]])

        points_u = orientation @ points_u * parafoil.planform.fc(s)
        points_l = orientation @ points_l * parafoil.planform.fc(s)

        points_u = (points_u.T + parafoil.c0(s)).T
        points_l = (points_l.T + parafoil.c0(s)).T

        ax.plot(points_u[0], points_u[1], -points_u[2], c='b', lw=0.8)
        ax.plot(points_l[0], points_l[1], -points_l[2], c='r', zorder=.9, lw=0.8)

    set_axes_equal(ax)
    ax.invert_yaxis()
    plt.show()

    breakpoint()


if __name__ == "__main__":
    main()
