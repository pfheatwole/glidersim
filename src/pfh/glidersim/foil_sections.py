"""Models that provide foil section geometry and coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np


if TYPE_CHECKING:
    from pfh.glidersim.airfoil import (
        AirfoilCoefficientsInterpolator,
        AirfoilGeometryInterpolator,
    )


__all__ = [
    "SimpleIntakes",
    "FoilSections",
]


def __dir__():
    return __all__


class SimpleIntakes:
    """
    Defines the upper and lower surface coordinates as constant along the span.

    This version currently uses explicit `r_upper` and `r_lower` in airfoil
    coordinates, but other parametrizations might be the intake midpoint and
    width (where "width" might be in the airfoil `s`, or as a percentage of the
    chord) or `c_upper` and `c_lower` as points on the chord.

    Parameters
    ----------
    s_end: float
        Section index. Air intakes are present between +/- `s_end`.
    r_upper, r_lower : float
        The starting coordinates of the upper and lower surface of the
        parafoil, given in airfoil profile coordinates. These are used to
        define air intakes, and for determining the inertial properties of the
        upper and lower surfaces.

        The airfoil coordinates use `r = 0` for the leading edge, `r = 1` for
        trailing edge of the curve above the chord, and `r = -1` for the
        trailing edge of the curve below the chord, so these choices must
        follow `-1 <= r_lower <= r_upper <= 1`.
    """

    def __init__(self, s_end: float, r_upper: float, r_lower: float) -> None:
        # TODO: support more types of definition:
        #  1. su/sl : explicit upper/lower cuts in airfoil coordinates
        #  2. midpoint (in airfoil coordinates) and width
        #  3. Upper and lower cuts as a fraction of the chord (the "Paraglider
        #     Design Manual" does it this way).
        self.s_end = s_end
        self.r_upper = r_upper
        self.r_lower = r_lower

    def __call__(self, s, r, surface: str):
        """
        Convert parafoil surface coordinates into airfoil coordinates.

        Parameters
        ----------
        s : array_like of float
            Section index.
        r : array_like of float
            Parafoil surface coordinate, where `0 <= r <= 1`, with `0`
            being the leading edge, and `1` being the trailing edge.
        surface : {"upper", "lower"}
            Which surface.

        Returns
        -------
        array_like of float, shape (N,)
            The normalized (unscaled) airfoil coordinates.
        """
        s = np.asfarray(s)
        r = np.asfarray(r)

        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if r.min() < 0 or r.max() > 1:
            raise ValueError("Surface coordinates must be between 0 and 1.")
        if surface not in {"upper", "lower"}:
            raise ValueError("`surface` must be one of {'upper', 'lower'}")

        if surface == "upper":
            values = self.r_upper + r * (1 - self.r_upper)
            values = np.broadcast_arrays(s, values)[1]
        else:
            # The lower surface extends forward on sections without intakes
            starts = np.where(np.abs(s) < self.s_end, self.r_lower, self.r_upper)
            values = starts + r * (-1 - starts)

        return values


class FoilSections:
    """
    Provides the section profile geometry and coefficients.

    This simple implementation only takes a single airfoil; it does not support
    spanwise interpolation of section profiles.

    Parameters
    ----------
    profiles : AirfoilGeometryInterpolator
        The section profiles. This class currently assumes all sections have
        the same, fixed airfoil. In the future the section profiles will be
        functions of both `s` and `ai`, not just `ai`.
    coefficients : AirfoilCoefficientsInterpolator
        The section coefficients. This class currently assumes all sections
        have the section coefficients. In the future the coefficients will be
        functions of `s`.
    intakes : function, optional
        A function that defines the upper and lower intake positions in
        airfoil profile coordinates as a function of the section index.
    Cd_intakes : float, optional
        Additional drag coefficient due to air intake openings. See [1]_.
    Cd_surface : float, optional
        Additional drag coefficient due to surface characteristics. See [2]_.

    References
    ----------
    .. [1] Holger Babinsky, "The aerodynamic performance of paragliders", 1999.
           DOI: 10.1017/S0001924000027974

    .. [2] George M. Ware, "Wind-tunnel investigation of ram-air inflated all
           flexible wings of aspect ratios 1.0 to 3.0", 1969.
    """

    def __init__(
        self,
        profiles: AirfoilGeometryInterpolator,
        coefficients: AirfoilCoefficientsInterpolator = None,
        intakes: Callable | None = None,
        Cd_intakes: float = 0,
        Cd_surface: float = 0,
    ) -> None:
        self.profiles = profiles
        self.coefficients = coefficients
        self.intakes = intakes if intakes else self._no_intakes
        self.Cd_intakes = Cd_intakes
        self.Cd_surface = Cd_surface

    def _no_intakes(self, s, r, surface: str):
        # For foils with no air intakes the canopy upper and lower surfaces map
        # directly to the airfoil upper and lower surfaces, which were defined
        # by the airfoil leading edge.
        if surface == "lower":
            r = -r
        return np.broadcast_arrays(s, r)[1]

    def surface_xz(self, s, ai, r, surface: str):
        """
        Compute unscaled surface coordinates along section profiles.

        These are unscaled since the FoilSections only defines the normalized
        airfoil geometry and coefficients. The Foil scales, translates, and
        orients these with the chord data it gets from the FoilLayout.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        r : array_like of float
            Surface or airfoil coordinates, depending on the value of `surface`.
        surface : {"chord", "camber", "upper", "lower", "airfoil"}
            How to interpret the coordinates in `r`. If "upper" or "lower",
            then `r` is treated as surface coordinates, which range from 0 to
            1, and specify points on the upper or lower surfaces, as defined by
            the intakes. If "airfoil", then `r` is treated as raw airfoil
            profile coordinates, which must range from -1 to +1, and map from
            the lower surface trailing edge to the upper surface trailing edge.

        Returns
        -------
        array of float
            A set of points from the section surface in foil frd. The shape is
            determined by standard numpy broadcasting of `s`, `ai`, and `r`.
        """
        s, ai, r = np.broadcast_arrays(s, ai, r)
        valid_surfaces = {"chord", "camber", "upper", "lower", "airfoil"}
        if s.min() < -1 or s.max() > 1:
            raise ValueError("Section indices must be between -1 and 1.")
        if surface not in valid_surfaces:
            raise ValueError(f"`surface` must be one of {valid_surfaces}")
        if surface == "airfoil" and (r.min() < -1 or r.max() > 1):
            raise ValueError("Airfoil coordinates must be between -1 and 1.")
        elif surface != "airfoil" and (r.min() < 0 or r.max() > 1):
            raise ValueError("Surface coordinates must be between 0 and 1.")

        if surface in {"upper", "lower"}:
            r = self.intakes(s, r, surface)  # type: ignore [operator]
            r_P2LE = self.profiles.profile_curve(ai, r)
        else:
            if surface == "chord":
                r_P2LE = np.stack((r, np.zeros(r.shape)), -1)
            elif surface == "camber":
                r_P2LE = self.profiles.camber_curve(ai, r)
            elif surface == "airfoil":
                r_P2LE = self.profiles.profile_curve(ai, r)

        return r_P2LE

    def Cl(self, s, ai, alpha, Re, clamp=False):
        """
        Compute the lift coefficient of the airfoil.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        alpha : array_like of float [radians]
            Angle of attack
        Re : float [unitless]
            Reynolds number
        clamp : bool
            Whether to clamp `alpha` to the highest non-nan value supported by
            the (ai, Re) pair.

        Returns
        -------
        Cl : float
        """
        return self.coefficients.Cl(ai, alpha, Re, clamp)

    def Cl_alpha(self, s, ai, alpha, Re, clamp=False):
        """
        Compute the derivative of the lift coefficient versus angle of attack.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        alpha : array_like of float [radians]
            Angle of attack
        Re : float [unitless]
            Reynolds number
        clamp : bool
            Whether to return `0` if `alpha` exceeds the the highest non-nan
            value supported by the (ai, Re) pair.

        Returns
        -------
        Cl_alpha : float
        """
        return self.coefficients.Cl_alpha(ai, alpha, Re, clamp)

    def Cd(self, s, ai, alpha, Re, clamp=False):
        """
        Compute the drag coefficient of the airfoil.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        alpha : array_like of float [radians]
            Angle of attack
        Re : float [unitless]
            Reynolds number
        clamp : bool
            Whether to clamp `alpha` to the highest non-nan value supported by
            the (ai, Re) pair.

        Returns
        -------
        Cd : float
        """
        Cd = self.coefficients.Cd(ai, alpha, Re, clamp)

        # Additional drag from the air intakes
        #
        # Ref: `babinsky1999AerodynamicPerformanceParagliders`
        la = np.linalg.norm(  # Length of the air intake
            self.surface_xz(s, ai, 0, "upper") - self.surface_xz(s, ai, 0, "lower"),
            axis=-1,
        )
        Cd += self.Cd_intakes * la  # Drag due to air intakes

        # Additional drag from "surface characteristics"
        #
        # Ref: `ware1969WindtunnelInvestigationRamair`
        Cd += self.Cd_surface

        return Cd

    def Cm(self, s, ai, alpha, Re, clamp=False):
        """
        Compute the pitching coefficient of the airfoil.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        alpha : float [radians]
            The angle of attack
        Re : float [unitless]
            The Reynolds number
        clamp : bool
            Whether to clamp `alpha` to the highest non-nan value supported by
            the (ai, Re) pair.

        Returns
        -------
        Cm : float
        """
        return self.coefficients.Cm(ai, alpha, Re, clamp)

    def thickness(self, s, ai, r):
        """
        Compute section thickness.

        These are the normalized thicknesses, not the absolute values. The
        absolute thickness requires knowledge of the chord length.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        r : array_like of float [percentage]
            Fractional position on the camber line, where `0 <= r <= 1`

        Returns
        -------
        thickness : array_like of float
            The normalized section profile thicknesses.
        """
        return self.profiles.thickness(ai, r)
