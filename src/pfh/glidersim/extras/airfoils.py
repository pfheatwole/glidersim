from pathlib import Path

import pfh.glidersim as gsim

__all__ = [
    "load_airfoil",
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
