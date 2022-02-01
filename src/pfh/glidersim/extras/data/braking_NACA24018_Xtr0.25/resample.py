# Hack to resample a set of XFLR5 (or XFOIL) airfoil data in `.txt` files into
# a regular grid (each dimension uses its own uniform spacing) to make it
# useable by faster linear interpolation methods, such as in `fast_interp.py`.

import itertools
import pathlib
import re

import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.interpolate import griddata


# Standard XFLR5 polar column names (renames CL/CD to Cl/Cd)
names = [
    "alpha",
    "Cl",
    "Cd",
    "CDp",
    "Cm",
    "Top Xtr",
    "Bot Xtr",
    "Cpmin",
    "Chinge",
    "XCp",
]

# Example filename:
#   NACA24018_theta30_Ku4.5_Kl0.5_d0.35_deltad0.20274_T1_Re4.000_M0.00_N5.0_XtrTop25%_XtrBot25%.txt
#
# Here `delta` refers to an outdated definition of the angle made between the
# trailing edge and the chord. I switched to the deflection distance `delta_d`
# but haven't changed the XFLR5 project file yet. Too much to do.
polars = []
for polar_file in pathlib.Path(".").glob("*.txt"):
    print(polar_file)
    data = np.genfromtxt(polar_file, skip_header=11, names=names)
    if match := re.search(r"_deltad(\d+\.\d+)_", polar_file.name):
        delta_d = float(match.group(1))
    else:
        raise ValueError(f"Invalid filename {polar_file}; needs `deltad`.")
    if match := re.search(r"_Re(\d\.\d\d\d)_", polar_file.name):
        Re = float(match.group(1))
    else:
        raise ValueError(f"Invalid filename {polar_file}; needs `Re`.")
    data = rfn.append_fields(data, "delta_d", np.full(data.shape[0], delta_d))
    data = rfn.append_fields(data, "Re", np.full(data.shape[0], Re))
    columns = ["delta_d", "alpha", "Re", "Cl", "Cm", "Cd"]
    data = data[columns]
    polars.append(data)

# Compute the ranges over the full dataset
data = np.concatenate(polars)
data.sort(order=["delta_d", "alpha", "Re"])
alphas = np.unique(data["alpha"])  # Angle of attack in degrees
delta_ds = np.unique(data["delta_d"])  # Normalized vertical deflection distance
Res = np.unique(data["Re"])  # Reynolds number

# Compute uniform spacings for each dimension  (FIXME: hacky strategies)
step_deltad = delta_ds.ptp() / np.ceil(delta_ds.max() / np.diff(delta_ds).min())
step_alpha = np.diff(alphas).min()
step_Re = np.diff(Res).min()  # FIXME: not guaranteed to work well
pdeltad = np.arange(delta_ds.min(), delta_ds.max() + step_deltad, step_deltad)
palpha = np.arange(alphas.min(), alphas.max() + step_alpha, step_alpha)
pRe = np.arange(Res.min(), Res.max() + step_Re, step_Re)
newshape = (len(pdeltad), len(palpha), len(pRe))

# Resample the entire grid with unstructured linear interpolation to fill holes
points = (data["delta_d"], data["alpha"], data["Re"])
DELTAD, ALPHA, RE = np.meshgrid(pdeltad, palpha, pRe, indexing="ij")
GRID = (DELTAD.ravel(), ALPHA.ravel(), RE.ravel())
print("Resampling Cl...")
Cl = griddata(points, data["Cl"], GRID)
print("Resampling Cd...")
Cd = griddata(points, data["Cd"], GRID)
print("Resampling Cm...")
Cm = griddata(points, data["Cm"], GRID)

# Linear interpolation of Cl isn't ideal for optimizing methods. Use polynomial
# regression to smooth Cl and sample a smooth Cl_alpha.
print("Smoothing Cl and Cl_alpha with a polynomial...")
Cl = Cl.reshape(newshape)
Cl_alpha = np.full(newshape, np.nan)
for (D, R) in itertools.product(range(len(pdeltad)), range(len(pRe))):
    valid = np.nonzero(~np.isnan(Cl[D, :, R]))[0]
    alphas = np.deg2rad(palpha[valid])  # Scale before taking the derivative!
    if len(valid) < 2:
        print(f"WARNING: less than two points for {pdeltad[D]=}, {pRe[R]}")
        continue
    poly = np.polynomial.Polynomial.fit(
        alphas,
        Cl[D, valid, R],
        10,
    )
    Cl[D, valid, R] = poly(alphas)
    Cl_alpha[D, valid, R] = poly.deriv()(alphas)

Cl = Cl.ravel()
Cl_alpha = Cl_alpha.ravel()

DATA = np.stack(
    (DELTAD.ravel(), ALPHA.ravel(), RE.ravel(), Cl, Cd, Cm, Cl_alpha),
    axis=-1,
)

with open("uniform_grid.csv", "w") as f:
    f.write("delta_d,alpha,Re,Cl,Cd,Cm,Cl_alpha\n")
    np.savetxt(f, DATA, fmt="%.6g", delimiter=",")
