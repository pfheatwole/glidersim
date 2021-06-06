"""FIXME: add module docstring"""

from pathlib import Path

import numpy as np

import pfh.glidersim as gsim


__all__ = [
    "load_polar",
    "load_datfile",
    "load_datfile_set",
]


def __dir__():
    return __all__


def load_polar(polarname):
    """
    Load a gridded section polar from a bundled directory.

    Parameters
    ----------
    polarname : string
        The name of a directory in `pfh/glidersim/extras/airfoils/`.

    Returns
    -------
    gsim.airfoil.AirfoilCoefficients
    """
    # polarfile = Path(__file__).parent / "data" / polarname / "nonuniform_grid.csv"
    # polar = gsim.airfoil.GridCoefficients(polarfile)
    polarfile = Path(__file__).parent / "data" / polarname / "uniform_grid.csv"
    polar = gsim.airfoil.GridCoefficients2(polarfile)
    return polar


def load_datfile(
    path, convention="perpendicular", center=False, derotate=False, normalize=False
):
    """
    Build an AirfoilGeometry from a `.dat` file using numpy.

    Parameters
    ----------
    path : string
        The filename.
    convention : {"perpendicular", "vertical"}, optional
        Whether the airfoil thickness is measured perpendicular to the mean
        camber line or vertically (the y-axis distance). Default: perpendicular
    center : bool, optional
        Translate the curve leading edge to the origin. Default: False
    derotate : bool, optional
        Rotate the the chord parallel to the x-axis. Default: False
    normalize : bool, optional
        Scale the curve so the chord is unit length. Default: False

    Returns
    -------
    gsim.airfoil.AirfoilGeometry
        The instantiated AirfoilGeometry object.
    """
    points = np.loadtxt(path, skiprows=1)
    airfoil = gsim.airfoil.AirfoilGeometry.from_points(
        points,
        convention=convention,
        center=center,
        derotate=derotate,
        normalize=normalize,
    )
    return airfoil


def load_datfile_set(
    directory,
    bundled=True,
    convention="perpendicular",
    center=False,
    derotate=False,
    normalize=False,
):
    """
    Load a set of airfoil `.dat` files using numpy.

    This is intended for use with a set of .dat files that represent a single
    airfoil geometry with deflected trailing edges. The vertical deflection
    distance must be the last part of the filename.

    For example, the directory might contain:

      naca23015_deltad0.00.dat
      naca23015_deltad0.10.dat
      naca23015_deltad0.20.dat

    Parameters
    ----------
    directory : string
        The directory containing a set of airfoil curves in the typical `.dat`
        format. The filenames must end with `_delta<###>`, where `<###>` is a
        floating point number
    bundled : bool
        Whether `directory` is the name of a bundled profile set.
    convention : {"perpendicular", "vertical"}, optional
        Whether the airfoil thickness is measured perpendicular to the mean
        camber line or vertically (the y-axis distance). Default: perpendicular
    center : bool, optional
        Translate the curve leading edge to the origin. Default: False
    derotate : bool, optional
        Rotate the the chord parallel to the x-axis. Default: False
    normalize : bool, optional
        Scale the curve so the chord is unit length. Default: False

    Returns
    -------
    dict
        A dictionary of {delta: gsim.airfoil.AirfoilGeometry} pairs, where
        `delta` is a `float`.
    """
    if bundled:
        datdir = Path(__file__).parent / "data" / directory
    else:
        datdir = Path(directory)
    airfoils = {}
    for datfile in sorted(datdir.glob("*.dat")):
        # Filenames are expected to follow `<basename>_delta<#######>.dat`.
        # The number of digits in `delta` is variable.
        datname = datfile.name
        delta_start = datname.find("_deltad")
        delta = float(datname[delta_start + 7 : -4])
        airfoil = load_datfile(
            datfile,
            convention=convention,
            center=center,
            derotate=derotate,
            normalize=normalize,
        )
        airfoils[delta] = {"name": datname, "airfoil": airfoil}

    return airfoils
