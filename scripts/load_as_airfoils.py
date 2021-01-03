"""Demo how to load a set of airfoil `.dat` files using numpy."""

from pathlib import Path

import numpy as np

import pfh.glidersim as gsim


def load_delta_airfoil_set(directory):
    datdir = Path(directory)
    airfoils = {}
    for datfile in datdir.glob("*.dat"):
        # Filenames are expected to follow `<basename>_delta<#######>.dat`.
        # The number of digits in `delta` is variable.
        datname = datfile.name
        delta_start = datname.find('_delta')
        delta = float(datname[delta_start+6:-4])
        points = np.loadtxt(datfile, skiprows=1)
        airfoils[delta] = {"name": datname[:delta_start], "points": points}

    return airfoils


if __name__ == "__main__":
    directory = "/home/peter/model/work/curving_airfoils/exp_curving_24018"
    airfoils = load_delta_airfoil_set(directory)

    for key in sorted(airfoils.keys()):
        airfoil = gsim.airfoil.AirfoilGeometry.from_points(
            airfoils[key]['points'],
            convention="perpendicular",
            center=False,
            derotate=False,
            normalize=False,
        )
        airfoils[key]['airfoil'] = airfoil
        gsim.plots.plot_airfoil_geo(airfoil)

    breakpoint()
