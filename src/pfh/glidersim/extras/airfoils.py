from pathlib import Path

import numpy as np

import pfh.glidersim as gsim

__all__ = [
    "load_airfoil",
    "load_airfoil_set",
]


def __dir__():
    return __all__


def load_airfoil():
    airfoil_geo = gsim.airfoil.NACA(24018, convention="vertical")
    polarfile = Path(__file__).parent / "airfoils/braking_NACA24018_Xtr0.25/gridded.csv"
    airfoil_coefs = gsim.airfoil.GridCoefficients(polarfile)
    # delta_max = np.deg2rad(13.37)  # FIXME: magic number
    airfoil = gsim.airfoil.Airfoil(airfoil_coefs, airfoil_geo)
    return airfoil


def load_airfoil_set(directory):
    """
    Load a set of airfoil `.dat` files using numpy. (Crude! Beware!)

    This is intended for use with a set of .dat files that represent a single
    airfoil geometry with deflected trailing edges. The deflection angle must
    be the last part of the filename

    Parameters
    ----------
    directory : string
        The directory containing a set of airfoil curves in the typical `.dat`
        format. The filenames must end with `_delta<###>`, where `<###>` is a
        floating point number

    Returns
    -------
    airfoils : dict
        A dictionary, keyed by the value of `delta`. Each value of `delta` is
        a dictionary for each datfile containing the values:

        name: string
            The airfoil name (minus the `_delta<###>.dat`)
        points: array of float, shape (K,2)
            The xy-coordinates of the airfoil curve.
        airfoil : gsim.airfoil.AirfoilGeometry
            The instantiated AirfoilGeometry object.
    """
    datdir = Path(directory)
    airfoils = {}
    for datfile in datdir.glob("*.dat"):
        # Filenames are expected to follow `<basename>_delta<#######>.dat`.
        # The number of digits in `delta` is variable.
        datname = datfile.name
        delta_start = datname.find("_delta")
        delta = float(datname[delta_start+6:-4])
        points = np.loadtxt(datfile, skiprows=1)
        airfoils[delta] = {"name": datname[:delta_start], "points": points}

    for key in sorted(airfoils.keys()):
        airfoil = gsim.airfoil.AirfoilGeometry.from_points(
            airfoils[key]["points"],
            convention="perpendicular",
            center=False,
            derotate=False,
            normalize=False,
        )
        airfoils[key]["airfoil"] = airfoil

    return airfoils
