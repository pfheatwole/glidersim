"""Models that combine a 3D foil geometry with an aerodynamics estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Delaunay

from pfh.glidersim.foil_aerodynamics import FoilAerodynamics

from .util import cross3


if TYPE_CHECKING:
    from pfh.glidersim.foil_layout import FoilLayout
    from pfh.glidersim.foil_sections import FoilSections


__all__ = [
    "SimpleFoil",
]


def __dir__():
    return __all__


class SimpleFoil:
    """
    A foil geometry that applies a constant airfoil along a chord surface.

    These are idealized foils that exactly match the scaled chord surface and
    profiles. Such an idealized shape is only possible for rigid foils that can
    enforce absolute geometry, unlike flexible foils, which can only attempt to
    create the target shape through internal structure.

    Parameters
    ----------
    layout : FoilLayout
        Provides the scale, position, and orientation of the section profiles.
    sections : FoilSections
        The geometry and coefficients for the section profiles.
    b, b_flat : float
        The arched and flattened spans of the chords. Specify only one. These
        function as scaling factors for the FoilLayout.
    aerodynamics_method : FoilAerodynamics, optional
        Estimator for the aerodynamic forces and moments.
    aerodynamics_config : dictionary, optional
        Keyword arguments for instantiating `aerodynamics_method`
    """

    def __init__(
        self,
        layout: FoilLayout,
        sections: FoilSections,
        b: float | None = None,
        b_flat: float | None = None,
        aerodynamics_method: type[FoilAerodynamics] | None = None,
        aerodynamics_config: dict = {},  # noqa: B006
    ) -> None:
        self._layout = layout
        self.sections = sections

        if b is not None and b_flat is not None:
            raise ValueError("Specify only one of `b` or `b_flat`")

        # TODO: support `S` and `S_flat` as scaling factors
        if b:
            self.b = b
        elif b_flat:
            self.b_flat = b_flat
        else:
            raise ValueError("Specify one of `b` or `b_flat`")

        if aerodynamics_method is None:
            self.aerodynamics = None
        elif issubclass(aerodynamics_method, FoilAerodynamics):
            self.aerodynamics = aerodynamics_method(self, **aerodynamics_config)
        else:
            raise ValueError(
                "`aerodynamics_method` must be a class that implements the "
                "FoilAerodynamics Protocol",
            )

    @property
    def b(self) -> float:
        """The projected span of the foil."""
        return self._b

    @b.setter
    def b(self, new_b: float) -> None:
        self._b = new_b
        self._b_flat = new_b * self._layout.b_flat / self._layout.b

    @property
    def b_flat(self) -> float:
        """The projected span of the foil with section dihedral removed."""
        return self._b_flat

    @b_flat.setter
    def b_flat(self, new_b_flat) -> None:
        self._b_flat = new_b_flat
        self._b = new_b_flat * self._layout.b / self._layout.b_flat

    @property
    def AR(self) -> float:
        """The aspect ratio of the foil."""
        return self.b**2 / self.S

    @property
    def AR_flat(self) -> float:
        """The aspect ratio of the foil with section dihedral removed."""
        return self.b_flat**2 / self.S_flat

    @property
    def S(self) -> float:
        """
        The projected area of the surface.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        return self._layout.S * (self.b_flat / 2) ** 2

    @property
    def S_flat(self) -> float:
        """
        The projected area of the surface with section dihedral removed.

        This is the conventional definition using the area traced out by the
        section chords projected onto the xy-plane.
        """
        return self._layout.S_flat * (self.b_flat / 2) ** 2

    def chord_length(self, s):
        """
        Compute section chord lengths.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index

        Returns
        -------
        array_like of float, shape (N,)
            The length of the section chord.
        """
        return self._layout.c(s) * (self.b_flat / 2)

    def section_orientation(self, s, flatten: bool = False):
        """
        Compute section coordinate axes as rotation matrices.

        Parameters
        ----------
        s : array_like of float, shape (N,)
            Section index
        flatten : bool, optional
            Whether to ignore dihedral. Default: False

        Returns
        -------
        array of float, shape (N,3)
            Rotation matrices encoding section orientation, where the columns
            are the section (local) x, y, and z coordinate axes.
        """
        return self._layout.orientation(s, flatten)

    def section_thickness(self, s, ai, r):
        """
        Compute section thicknesses.

        Note that the thickness is determined by the airfoil convention, so
        this value may be measured perpendicular to either the chord line or
        to the camber line. Be careful relying on these values.

        NOTE: the airfoil thickness "convention" can dramatically affect this
        value. For normalized and derotated airfoils with relatively mild
        camber the convention does not dramatically affect the thickness, but
        for airfoils with deflected trailing edges it can vary quite a lot.
        For now, the `ai` parameter is included to establish the API, but is
        constrained to zero-indexed profiles (which assumes the `ai=0` airfoil
        is normalized and derotated). Section thickness is in serious need of a
        design review.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai
            Airfoil index. Must be zero. (See NOTE above.)
        r : float
            Position along the chord or camber line as a percentage, where `r =
            0` is the leading edge, and `r = 1` is the trailing edge.
        """
        if np.any(ai != 0):
            raise ValueError("Airfoil indices must be zero. See docstring.")
        return self.sections.thickness(s, ai, r) * self.chord_length(s)

    def surface_xyz(self, s, ai, r, surface: str, flatten: bool = False):
        """
        Sample points on section surfaces in foil frd.

        Parameters
        ----------
        s : array_like of float
            Section index.
        ai : float
            Airfoil index.
        r : array_like of float
            Surface coordinate (normalized arc length). Meaning depends on the
            value of `surface`.
        surface : {"chord", "camber", "upper", "lower", "airfoil"}
            How to interpret the coordinates in `r`. If "upper" or "lower",
            then `r` is treated as surface coordinates, which range from 0 to
            1, and specify points on the upper or lower surfaces, as defined by
            the intakes. If "airfoil", then `r` is treated as raw airfoil
            coordinates, which must range from -1 to +1, and map from the
            lower surface trailing edge to the upper surface trailing edge.
        flatten : boolean
            Whether to flatten the foil by disregarding dihedral (curvature in
            the yz-plane). This is useful for inflatable wings, such as
            parafoils. Default: False.

        Returns
        -------
        array of float
            Coordinates on the section surface in foil frd. The shape is
            determined by standard numpy broadcasting of `s`, `ai`, and `r`.
        """
        s, ai, r = np.broadcast_arrays(s, ai, r)
        c = self.chord_length(s)
        r_P2LE_a = self.sections.surface_xz(s, ai, r, surface)  # Unscaled airfoil
        r_P2LE_s = np.stack(  # In section-local frd coordinates
            (-r_P2LE_a[..., 0], np.zeros(r_P2LE_a.shape[:-1]), -r_P2LE_a[..., 1]),
            axis=-1,
        )
        C_c2s = self.section_orientation(s, flatten)
        r_P2LE = np.einsum("...ij,...j,...->...i", C_c2s, r_P2LE_s, c)
        r_LE2O = self._layout.xyz(s, 0, flatten=flatten) * (self.b_flat / 2)
        return r_P2LE + r_LE2O

    def mass_properties(self, N_s: int = 301, N_r: int = 301):
        """
        Compute the quantities that control inertial behavior.

        The inertia matrices returned by this function are proportional to the
        values for a physical wing, and do not have standard units. They must
        be scaled by the wing materials and air density to get their physical
        values. See "Notes" for a thorough description.

        Parameters
        ----------
        N_s : int
            The grid resolution over the section index.
        N_r : int
            The grid resolution over the surface coordinates.

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

        The volume calculation requires a closed surface mesh, so it ignores
        air intakes and assumes a closed trailing edge (which is fine for
        inflatable foils like a paraglider). The concept of using a summation
        of signed tetrahedron volumes is developed in [1]. This implementation
        of that idea is from [2], which first computes the inertia tensor of a
        "canonical tetrahedron" then applies an affine transformation to
        compute the inertia tensors of each individual tetrahedron.

        References
        ----------
        1. Efficient feature extraction for 2D/3D objects in mesh
           representation, Zhang and Chen, 2001(?)

        2. How to find the inertia tensor (or other mass properties) of a 3D
           solid body represented by a triangle mesh, Jonathan Blow, 2004.
           http://number-none.com/blow/inertia/index.html
        """
        # Note to self: the triangles are not symmetric about the xz-plane,
        # which produces non-zero terms that should have cancelled out. Would
        # need to reverse the left-right triangle directions over one semispan.
        tu, tl = self._mesh_triangles(N_s, N_r)

        # Triangle and net surface areas
        u1, u2 = np.swapaxes(np.diff(tu, axis=1), 0, 1)
        l1, l2 = np.swapaxes(np.diff(tl, axis=1), 0, 1)
        au = np.linalg.norm(np.cross(u1, u2), axis=1) / 2
        al = np.linalg.norm(np.cross(l1, l2), axis=1) / 2
        Au = np.sum(au)
        Al = np.sum(al)

        # Triangle and net surface area centroids
        cu = np.einsum("Nij->Nj", tu) / 3
        cl = np.einsum("Nij->Nj", tl) / 3
        Cu = np.einsum("N,Ni->i", au, cu) / Au
        Cl = np.einsum("N,Ni->i", al, cl) / Al

        # Surface area inertia tensors
        cov_au = np.einsum("N,Ni,Nj->ij", au, cu - Cu, cu - Cu)
        cov_al = np.einsum("N,Ni,Nj->ij", al, cl - Cl, cl - Cl)
        J_u2U = np.trace(cov_au) * np.eye(3) - cov_au
        J_l2L = np.trace(cov_al) * np.eye(3) - cov_al

        # -------------------------------------------------------------------
        # Volumes

        # The volume calculation requires a closed mesh, so resample the
        # surface using the closed section profiles (ie, ignore air intakes).
        s = np.linspace(-1, 1, N_s)
        r = 1 - np.cos(np.linspace(0, np.pi / 2, N_r))
        r = np.concatenate((-r[:0:-1], r))
        surface_vertices = self.surface_xyz(s[:, None], 0, r, "airfoil")

        # Using Delaunay is too slow as the number of vertices increases.
        S, R = np.meshgrid(np.arange(N_s - 1), np.arange(2 * N_r - 2), indexing="ij")
        triangle_indices = np.concatenate(
            (
                [[S, R], [S + 1, R + 1], [S + 1, R]],
                [[S, R], [S, R + 1], [S + 1, R + 1]],
            ),
            axis=-2,
        )
        ti = np.moveaxis(triangle_indices, (0, 1), (-2, -1)).reshape(-1, 3, 2)
        surface_indices = np.ravel_multi_index(ti.T, (N_s, 2 * N_r - 1)).T
        surface_tris = surface_vertices.reshape(-1, 3)[surface_indices]

        # Add two meshes to close the wing tips so the volume is counted
        # correctly. Uses the 2D section profile to compute the triangulation.
        # If a wing tip has a closed trailing edge there will be coplanar
        # vertices, which `Delaunay` will discard automatically.
        left_vertices = self.surface_xyz(-1, 0, r, "airfoil")
        right_vertices = self.surface_xyz(1, 0, r, "airfoil")

        # Verson 1: use scipy.spatial.Delaunay. This version will automatically
        # discard coplanar points if the trailing edge is closed.
        #
        left_points = self.sections.surface_xz(-1, 0, r, "airfoil")
        right_points = self.sections.surface_xz(1, 0, r, "airfoil")
        left_tris = left_vertices[Delaunay(left_points).simplices]
        right_tris = right_vertices[Delaunay(right_points).simplices[:, ::-1]]

        # Version 2: build the list of simplices explicitly. This version just
        # assumes the trailing edge is closed (reasonable for inflatable foils)
        # and discards the vertex at the upper surface trailing edge. This is
        # fine even if the trailing edge is not truly closed as long as it's
        # _effectively_ closed and N_r is reasonably large.
        #
        # ix = np.arange(N_r - 2)
        # upper_simplices = np.stack((2 * N_r - 3 - ix, 2 * N_r - 4 - ix, ix + 1)).T
        # lower_simplices = np.stack((ix + 1, ix, 2 * N_r - 3 - ix)).T
        # simplices = np.concatenate((upper_simplices, lower_simplices))
        # left_tris = left_vertices[simplices]
        # right_tris = right_vertices[simplices[:, ::-1]]

        tris = np.concatenate((left_tris, surface_tris, right_tris))

        # Tetrahedron signed volumes and net volume
        v = np.einsum("Ni,Ni->N", cross3(tris[..., 0], tris[..., 1]), tris[..., 2]) / 6
        V = np.sum(v)

        # Tetrahedron centroids and net centroid
        cv = np.einsum("Nij->Nj", tris) / 4
        Cv = np.einsum("N,Ni->i", v, cv) / V

        # Volume inertia tensors
        cov_canonical = np.full((3, 3), 1 / 120) + np.eye(3) / 120
        cov_v = np.einsum(
            "N,Nji,jk,Nkl->il",  # A[n] = tris[n].T
            np.linalg.det(tris),
            tris,
            cov_canonical,
            tris,
            optimize=True,
        )
        J_v2LE = np.eye(3) * np.trace(cov_v) - cov_v
        J_v2V = J_v2LE - V * ((Cv @ Cv) * np.eye(3) - np.outer(Cv, Cv))

        mass_properties = {
            "upper_area": Au,
            "upper_centroid": Cu,
            "upper_inertia": J_u2U,
            "volume": V,
            "volume_centroid": Cv,
            "volume_inertia": J_v2V,
            "lower_area": Al,
            "lower_centroid": Cl,
            "lower_inertia": J_l2L,
        }

        return mass_properties

    def _mesh_vertex_lists(
        self,
        N_s: int = 131,
        N_r: int = 151,
        filename: str = None,
    ):
        """
        Generate sets of triangle faces on the upper and lower surfaces.

        Each triangle mesh is described by a set of vertices and a set of
        "faces". The vertices are the surface coordinates sampled on a
        rectilinear grid over the section indices and surface coordinates. The
        faces are the list of vertices that define the triangles.

        Parameters
        ----------
        N_s : int
            The grid resolution over the section index.
        N_r : int
            The grid resolution over the surface coordinates.
        filename : string, optional
            Save the outputs in a numpy `.npz` file.

        Returns
        -------
        vertices_upper, vertices_lower : array of float, shape (N_s * N_r, 3)
            Vertices on the upper and lower surfaces.
        simplices : array of int, shape (N_s * N_r * 2, 3)
            Lists of vertex indices for each triangle. The same grid was used
            for both surfaces, so this array defines both meshes.

        See Also
        --------
        _mesh_triangles : Helper function that produces the meshes themselves
                          as two lists of vertex triplets (the triangles in
                          frd coordinates).

        Examples
        --------
        To export the mesh into Blender:

        1. Generate the mesh

           >>> foil._mesh_vertex_lists(filename='/path/to/mesh.npz')

        2. In the Blender (v2.8) Python console:

           import numpy
           data = numpy.load('/path/to/mesh.npz')
           # Blender doesn't support numpy arrays
           vu = data['vertices_upper'].tolist()
           vl = data['vertices_lower'].tolist()
           simplices = data['simplices'].tolist()
           mesh_upper = bpy.data.meshes.new("upper")
           mesh_lower = bpy.data.meshes.new("lower")
           object_upper = bpy.data.objects.new("upper", mesh_upper)
           object_lower = bpy.data.objects.new("lower", mesh_lower)
           bpy.context.scene.collection.objects.link(object_upper)
           bpy.context.scene.collection.objects.link(object_lower)
           mesh_upper.from_pydata(vu, [], simplices)
           mesh_lower.from_pydata(vl, [], simplices)
           mesh_upper.update(calc_edges=True)
           mesh_lower.update(calc_edges=True)

        FIXME: doesn't support airfoil indexing (brake deflections, etc)
        """
        # Compute the vertices
        s = np.linspace(-1, 1, N_s)
        r = 1 - np.cos(np.linspace(0, np.pi / 2, N_r))

        # The lower surface goes right->left to ensure the normals point down,
        # which is important for computing the enclosed volume of air between
        # the two surfaces, and for 3D programs in general (which tend to
        # expect the normals to point "out" of the volume).
        vu = self.surface_xyz(s[:, np.newaxis], 0, r, "upper").reshape(-1, 3)
        vl = self.surface_xyz(s[::-1, np.newaxis], 0, r, "lower").reshape(-1, 3)

        # Compute the vertex lists for all of the faces (the triangles).
        S, R = np.meshgrid(np.arange(N_s - 1), np.arange(N_r - 1), indexing="ij")
        triangle_indices = np.concatenate(
            (
                [[S, R], [S + 1, R + 1], [S + 1, R]],
                [[S, R], [S, R + 1], [S + 1, R + 1]],
            ),
            axis=-2,
        )
        ti = np.moveaxis(triangle_indices, (0, 1), (-2, -1)).reshape(-1, 3, 2)
        simplices = np.ravel_multi_index(ti.T, (N_s, N_r)).T

        if filename:
            np.savez_compressed(
                filename,
                vertices_upper=vu,
                vertices_lower=vl,
                simplices=simplices,  # Same list for both surfaces
            )

        return vu, vl, simplices

    def _mesh_triangles(
        self,
        N_s: int = 131,
        N_r: int = 151,
        filename: str = None,
    ):
        """Generate triangle meshes over the upper and lower surfaces.

        Parameters
        ----------
        N_s : int
            The grid resolution over the section index.
        N_r : int
            The grid resolution over the surface coordinates.
        filename : string, optional
            Save the outputs in a numpy `.npz` file.

        Returns
        -------
        tu, tl : array of float, shape ((N_s - 1) * (N_r - 1) * 2, 3, 3)
            Lists of vertex triplets that define the triangles on the upper
            and lower surfaces.

            The shape warrants an explanation: the grid has `N_s * N_r` points
            for `(N_s - 1) * (N_r - 1)` rectangles. Each rectangle requires
            2 triangles, each triangle has 3 vertices, and each vertex has
            3 coordinates (in frd).

        See Also
        --------
        _mesh_vertex_lists : the sets of vertices and the indices that define
                             the triangles

        Examples
        --------
        To export the mesh into FreeCAD:

        1. Generate the mesh

           >>> foil._mesh_triangles('/path/to/triangles.npz')

        2. In the FreeCAD Python (v0.18) console:

           >>> # See: https://www.freecadweb.org/wiki/Mesh_Scripting
           >>> import Mesh
           >>> triangles = np.load('/path/to/triangles.npz')
           >>> # As of FreeCAD v0.18, `Mesh` doesn't support numpy arrays
           >>> mesh_upper = Mesh.Mesh(triangles['triangles_upper'].tolist())
           >>> mesh_lower = Mesh.Mesh(triangles['triangles_lower'].tolist())
           >>> Mesh.show(mesh_upper)
           >>> Mesh.show(mesh_lower)

        FIXME: doesn't support airfoil indexing (brake deflections, etc)
        """
        vu, vl, fi = self._mesh_vertex_lists(N_s=N_s, N_r=N_r)
        triangles_upper = vu[fi]
        triangles_lower = vl[fi]

        if filename:
            np.savez_compressed(
                filename,
                triangles_upper=triangles_upper,
                triangles_lower=triangles_lower,
            )

        return triangles_upper, triangles_lower

    def r_CP2LE(self):
        return self.aerodynamics.r_CP2LE()
