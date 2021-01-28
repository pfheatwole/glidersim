# Hack for loading a set of polars in `XFLR5Coefficients`, resampling them on
# a regular grid, and exporting them to a CSV suitable for use with an instance
# of `GridCoefficients`
#
# Important since the polars are easy to export with XFLR5, but slow to load
# and evaluate with `LinearNDInterpolator`.

import numpy as np

import pfh.glidersim as gsim


directory = "/home/peter/model/work/glidersim/scripts/polars/NACA23015_closed_N5.0/"
flapped = False  # Whether to look expect `delta` columns

print("Loading the polars...")
coefs = gsim.airfoil.XFLR5Coefficients(directory, flapped=flapped)
polars = coefs.polars

print("Sampling the points onto a grid...")
alphas = np.unique(polars["alpha"])
Re = np.unique(polars["Re"])
breakpoint()

if flapped:
    deltas = np.unique(polars["delta"])
    points_rad = np.array([[D, A, R * 1e6] for D in deltas for A in alphas for R in Re])
else:
    points_rad = np.array([[1234.5678, A, R * 1e6] for A in alphas for R in Re])

Cl = np.c_[coefs.Cl(*points_rad.T)]
Cd = np.c_[coefs.Cd(*points_rad.T)]
Cm = np.c_[coefs.Cm(*points_rad.T)]
Cl_alpha = np.c_[coefs.Cl_alpha(*points_rad.T)]

if flapped:
    points_deg = np.array([[D, A, R] for D in np.rad2deg(deltas) for A in np.rad2deg(alphas) for R in Re])
else:
    points_deg = np.array([[A, R] for A in np.rad2deg(alphas) for R in Re])

points = np.c_[points_deg, Cl, Cd, Cm, Cl_alpha]

print("Saving to file...")
with open("gridded.csv", "w") as f:
    if flapped:
        columns = "delta,alpha,Re,Cl,Cd,Cm,Cl_alpha\n"
    else:
        columns = "alpha,Re,Cl,Cd,Cm,Cl_alpha\n"
    f.writelines(columns)
    np.savetxt(f, points, fmt="%.6g", delimiter=",")
