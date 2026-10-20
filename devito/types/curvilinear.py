"""
Curvilinear (boundary-fitted) structured grids.

A curvilinear grid is a logically Cartesian grid, the computational grid,
whose nodes are placed in physical space by a smooth map. The Dimensions of a
CurvilinearGrid index the computational grid, so loops, halos, domain
decomposition and SubDomains are those of a Cartesian Grid, while physical
derivatives are evaluated through the chain rule with the metric terms of the
map (see `Derivative`).
"""

from functools import cached_property

import numpy as np
import sympy

from devito.mpi import MPI
from devito.types.dense import Function
from devito.types.grid import Grid

__all__ = ['CurvilinearGrid', 'CurvilinearMap', 'TerrainFollowing']


# Names of the computational coordinates, one per grid axis
_COMPUTATIONAL_NAMES = {1: ('xi',), 2: ('xi', 'eta'), 3: ('xi', 'eta', 'zeta')}


def _centered_fd(values, axis, spacing, order, deriv_order=1):
    """
    Centered finite-difference derivative of `values` along `axis`.

    The result drops `order//2` points at each end of `axis`. Each tap is
    differenced against the center value, so that a constant array has an
    exactly zero derivative.
    """
    radius = order // 2
    indices = list(range(-radius, radius + 1))
    weights = sympy.finite_diff_weights(deriv_order, indices, 0)[deriv_order][-1]
    npoint = values.shape[axis]
    center = np.take(values, range(radius, npoint - radius), axis=axis)
    derivative = np.zeros_like(center)
    for i, w in zip(indices, weights, strict=True):
        if w == 0 or i == 0:
            continue
        shifted = np.take(values, range(radius + i, npoint - radius + i), axis=axis)
        derivative += float(w) * (shifted - center)
    return derivative / spacing**deriv_order


def _extend(vector, npad):
    """Extend a uniformly spaced 1D vector by `npad` points at each end."""
    spacing = vector[1] - vector[0] if vector.size > 1 else 1.
    left = vector[0] - spacing*np.arange(npad, 0, -1)
    right = vector[-1] + spacing*np.arange(1, npad + 1)
    return np.concatenate([left, vector, right])


def _odd_pad(values, npad):
    """
    Pad `values` by `npad` points at each end of every axis by odd reflection,
    `v(-i) = 2 v(0) - v(i)`, which continues the slope across the edge.
    """
    for axis in range(values.ndim):
        npoint = values.shape[axis]
        first = np.take(values, [0], axis=axis)
        last = np.take(values, [npoint - 1], axis=axis)
        left = np.clip(np.arange(npad, 0, -1), 0, npoint - 1)
        right = np.clip(npoint - 1 - np.arange(1, npad + 1), 0, npoint - 1)
        values = np.concatenate([2*first - np.take(values, left, axis=axis), values,
                                 2*last - np.take(values, right, axis=axis)], axis=axis)
    return values


def _crop(values, npad):
    """Remove `npad` points at each end of every axis of `values`."""
    if npad == 0:
        return values
    return values[tuple(slice(npad, -npad) for _ in range(values.ndim))]


class CurvilinearMap:

    """
    Abstract map from computational to physical coordinates.

    The computational coordinates of a node are those a Cartesian Grid of the
    same shape, extent and origin gives it, `o + i*h` along each axis; the map
    places the node at a physical position. A map must provide

    * `forward(grid, computational)`, the physical coordinates of computational
      points, which must be defined, smoothly, past the domain too, where it
      gives the halo nodes their positions;
    * `inverse(grid, points)`, the computational coordinates of physical
      points, needed to place SparseFunctions and to reflect points through the
      rows of the grid.

    The metric terms, their derivatives and the Jacobian default to finite
    differences of `forward` at the grid space order. A map with closed forms
    overrides `metrics`, `metric_derivatives` and `jacobian`, and `validate`
    to check its parameters against a grid.

    Metric terms are returned as a dict `{(i, j): value}`, with `i` a physical
    axis and `j` a computational axis. A value is an array, or a Python number
    for an entry that is constant by construction of the map (e.g. the zero
    entries of a vertical map), which a CurvilinearGrid keeps symbolic.
    """

    def forward(self, grid, computational):
        """
        Physical coordinates of computational points.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map is attached to.
        computational : tuple of ndarray
            Computational coordinates, one broadcastable array per axis.

        Returns
        -------
        tuple of ndarray
            Physical coordinates, one array per axis.
        """
        raise NotImplementedError

    def inverse(self, grid, points):
        """
        Computational coordinates of physical points.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map is attached to.
        points : ndarray
            Physical coordinates of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            Computational coordinates of shape `(npoint, ndim)`.
        """
        raise NotImplementedError(f"{type(self).__name__} provides no inverse map, "
                                  "required to place SparseFunctions")

    def validate(self, grid):
        """
        Raise an informative error if the map is incompatible with `grid`.
        Called upon creation of a CurvilinearGrid.
        """
        return

    def _jacobian_matrix(self, grid, vectors, npad):
        """
        `A[..., i, j] = d x_i / d xi_j` on the nodes spanned by `vectors`,
        obtained by finite differences of `forward`.
        """
        radius = grid.space_order // 2
        extended = [_extend(v, npad + radius) for v in vectors]
        mesh = np.meshgrid(*extended, indexing='ij')
        physical = [np.broadcast_to(c, mesh[0].shape) for c in self.forward(grid, mesh)]
        spacing = [v[1] - v[0] for v in vectors]
        ndim = len(vectors)
        shape = tuple(v.size + 2*npad for v in vectors)
        matrix = np.empty(shape + (ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                derivative = _centered_fd(physical[i], j, spacing[j], grid.space_order)
                # Crop the non-differentiated axes to the same extent
                crop = tuple(slice(None) if k == j else slice(radius, -radius)
                             for k in range(ndim))
                matrix[..., i, j] = derivative[crop]
        return matrix

    def _inverse_metrics(self, grid, vectors, npad):
        """`M[..., j, i] = d xi_j / d x_i`, with `npad` extra points per side."""
        return np.linalg.inv(self._jacobian_matrix(grid, vectors, npad))

    def metrics(self, grid, vectors):
        """
        The metric terms `d xi_j / d x_i` on the nodes spanned by `vectors`.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map is attached to.
        vectors : tuple of ndarray
            Computational coordinates of the nodes, one 1D array per axis.
        """
        inverse = self._inverse_metrics(grid, vectors, 0)
        ndim = len(vectors)
        return {(i, j): inverse[..., j, i] for i in range(ndim) for j in range(ndim)}

    def metric_derivatives(self, grid, vectors):
        """
        The second derivatives `d^2 xi_j / d x_i^2` on the nodes spanned by
        `vectors`, as `sum_k (d xi_k / d x_i) d/d xi_k (d xi_j / d x_i)`.
        """
        radius = grid.space_order // 2
        inverse = self._inverse_metrics(grid, vectors, radius)
        spacing = [v[1] - v[0] for v in vectors]
        ndim = len(vectors)
        derivatives = {}
        for i in range(ndim):
            for j in range(ndim):
                value = 0
                for k in range(ndim):
                    derivative = _centered_fd(inverse[..., j, i], k, spacing[k],
                                              grid.space_order)
                    crop = tuple(slice(None) if m == k else slice(radius, -radius)
                                 for m in range(ndim))
                    value = value + _crop(inverse[..., k, i], radius)*derivative[crop]
                derivatives[(i, j)] = value
        return derivatives

    def jacobian(self, grid, vectors):
        """
        The Jacobian determinant `det(d x / d xi)` on the nodes spanned by
        `vectors`, the ratio of a physical to a computational cell volume.
        """
        return np.linalg.det(self._jacobian_matrix(grid, vectors, 0))


class TerrainFollowing(CurvilinearMap):

    """
    A terrain-following map: the grid follows a surface, e.g. topography,
    which becomes the grid row at computational depth 0.

    It is the map to use when the top of the domain is not flat, so that a
    boundary condition such as a free surface sits exactly on a grid row
    instead of cutting through cells. The columns of the grid stay vertical:
    each column keeps its lateral position and its nodes are spread evenly
    between the surface and the bottom of the grid, which stays flat.

    With the vertical axis last and pointing down, as in a Cartesian Grid, a
    node of computational coordinates `(x, eta)` is placed at the depth

        z(x, eta) = s(x) + eta * (b - s(x)) / b

    where `s` is the depth of the surface and `b = o_z + l_z` the computational
    depth of the last row of the grid, `o_z` and `l_z` being the origin and
    extent of its vertical axis. The row at `eta = 0` is the surface, and the
    last row is the flat bottom at `z = b`. The relief of the surface fades out
    linearly with depth, and the grid is Cartesian if the surface is flat at
    `z = 0`. A grid whose vertical origin is negative has rows above the
    surface, placed by the same map, such as the mirror rows of a free surface
    (see `CurvilinearGrid`).

    The surface sets the cell sizes: cells shrink vertically where the surface
    is deep and stretch where it is high, so the smallest one, used for a
    stable time step (`CurvilinearGrid.min_spacing`), is under the deepest point
    of the surface. Steep slopes shear the cells and should be smoothed to a
    few grid spacings.

    Parameters
    ----------
    surface : array_like or callable
        Depth of the surface, positive down like the vertical axis of the grid.
        Topography given as an elevation, positive up, is passed as its
        negative, `surface=-elevation`. Either an array with one value per
        lateral grid node, or a callable of the lateral physical coordinates,
        `s(x)` in 2D and `s(x, y)` in 3D. An array is continued past the edges
        of the grid by odd reflection, interpolated linearly between nodes to
        place sparse points, and differentiated by finite differences for the
        metric terms; a callable is evaluated exactly.

    Examples
    --------
    A 50 m high hill over a 1 km line, the grid extending 1 km down:

    >>> import numpy as np
    >>> from devito import CurvilinearGrid, TerrainFollowing
    >>> x = np.linspace(0., 1000., 11)
    >>> elevation = 50.*np.exp(-((x - 500.)/200.)**2)
    >>> grid = CurvilinearGrid(shape=(11, 21), extent=(1000., 1000.),
    ...                        mapping=TerrainFollowing(surface=-elevation))
    >>> z = grid.coordinates[-1]
    >>> float(z.data[5, 0]), float(z.data[5, -1])
    (-50.0, 1000.0)
    """

    def __init__(self, surface):
        self._surface = surface if callable(surface) else np.asarray(surface,
                                                                     dtype=np.float64)

    @property
    def surface(self):
        """The surface, an array of lateral node values or a callable."""
        return self._surface

    @staticmethod
    def _bottom(grid):
        """Depth of the flat bottom of `grid`, its last row."""
        return np.float64(grid.origin[-1]) + np.float64(grid.extent[-1])

    def validate(self, grid):
        if grid.dim < 2:
            raise ValueError("A TerrainFollowing map requires a grid with at least "
                             "two dimensions")
        if self._bottom(grid) <= 0:
            raise ValueError("A TerrainFollowing map requires the last row of the "
                             "grid below the surface, at a positive computational "
                             "depth")
        if callable(self._surface):
            return
        if self._surface.shape != tuple(grid.shape[:-1]):
            raise ValueError(f"The surface has shape {self._surface.shape}, expected one "
                             f"value per lateral grid node {tuple(grid.shape[:-1])}")

    def surface_at(self, grid, lateral):
        """
        The surface at the lateral physical positions `lateral`, a tuple of
        broadcastable arrays, one per lateral axis.
        """
        if callable(self._surface):
            return np.asarray(self._surface(*lateral), dtype=np.float64)

        # Fractional node positions, snapped to the node when within roundoff
        npad = 2*grid.space_order + 2
        padded = _odd_pad(self._surface, npad)
        positions = []
        for axis, values in enumerate(lateral):
            spacing = np.float64(grid.spacing[axis])
            position = (np.asarray(values) - np.float64(grid.origin[axis])) / spacing
            nearest = np.rint(position)
            position = np.where(np.abs(position - nearest) < 1e-9, nearest, position)
            positions.append(np.clip(position + npad, 0, padded.shape[axis] - 1))
        positions = np.broadcast_arrays(*positions)

        # Multilinear interpolation of the padded surface
        base = [np.minimum(np.floor(p).astype(np.int64), n - 2)
                for p, n in zip(positions, padded.shape, strict=True)]
        fractions = [p - b for p, b in zip(positions, base, strict=True)]
        value = 0
        for corner in np.ndindex(*(2,)*len(base)):
            weight = 1
            index = []
            for c, b, f in zip(corner, base, fractions, strict=True):
                weight = weight * (f if c else 1 - f)
                index.append(b + c)
            value = value + weight*padded[tuple(index)]
        return value

    def _surface_derivatives(self, grid, lateral_vectors):
        """
        Surface, and its first and second derivatives along each lateral axis,
        on the lateral nodes spanned by `lateral_vectors`.
        """
        radius = grid.space_order // 2
        extended = [_extend(v, radius) for v in lateral_vectors]
        mesh = np.meshgrid(*extended, indexing='ij')
        surface = self.surface_at(grid, mesh)
        nlateral = len(lateral_vectors)
        first, second = [], []
        for axis in range(nlateral):
            crop = tuple(slice(None) if k == axis else slice(radius, -radius)
                         for k in range(nlateral))
            spacing = lateral_vectors[axis][1] - lateral_vectors[axis][0]
            first.append(_centered_fd(surface, axis, spacing, grid.space_order)[crop])
            second.append(_centered_fd(surface, axis, spacing, grid.space_order,
                                       deriv_order=2)[crop])
        return _crop(surface, radius), first, second

    def _column_terms(self, grid, vectors):
        """Shared terms of the closed-form metrics, broadcast to the nodes."""
        surface, first, second = self._surface_derivatives(grid, vectors[:-1])
        surface = surface[..., None]
        first = [f[..., None] for f in first]
        second = [f[..., None] for f in second]
        bottom = self._bottom(grid)
        height = bottom - surface
        # Distance from the bottom, in computational coordinates (non positive)
        eta = vectors[-1] - bottom
        return bottom, height, eta, first, second

    def forward(self, grid, computational):
        computational = [np.asarray(c, dtype=np.float64) for c in computational]
        surface = self.surface_at(grid, computational[:-1])
        bottom = self._bottom(grid)
        z = surface + computational[-1]*(bottom - surface)/bottom
        return tuple(computational[:-1]) + (z,)

    def inverse(self, grid, points):
        points = np.asarray(points, dtype=np.float64)
        lateral = tuple(points[:, i] for i in range(points.shape[1] - 1))
        surface = self.surface_at(grid, lateral)
        bottom = self._bottom(grid)
        eta = bottom*(points[:, -1] - surface)/(bottom - surface)
        return np.concatenate([points[:, :-1], eta[:, None]], axis=1)

    def metrics(self, grid, vectors):
        ndim = len(vectors)
        last = ndim - 1
        bottom, height, eta, first, _ = self._column_terms(grid, vectors)
        metrics = {(i, j): int(i == j) for i in range(ndim) for j in range(last)}
        for i in range(last):
            metrics[(i, last)] = first[i]*eta/height
        metrics[(last, last)] = bottom/height
        return metrics

    def metric_derivatives(self, grid, vectors):
        ndim = len(vectors)
        last = ndim - 1
        _, height, eta, first, second = self._column_terms(grid, vectors)
        derivatives = {(i, j): 0 for i in range(ndim) for j in range(ndim)}
        for i in range(last):
            derivatives[(i, last)] = eta*(second[i]/height + 2*first[i]**2/height**2)
        return derivatives

    def jacobian(self, grid, vectors):
        bottom, height, _, _, _ = self._column_terms(grid, vectors)
        return height/bottom


class CurvilinearGrid(Grid):

    """
    A structured grid whose nodes are placed in physical space by a map.

    A CurvilinearGrid keeps the logically Cartesian *computational* grid of a
    Grid, and places its nodes in *physical* space with a CurvilinearMap, so
    that a curved boundary, e.g. topography, is a grid line. It is used exactly
    as a Grid: swap `Grid` for `CurvilinearGrid` and keep the rest of the code.

    * Its Dimensions, `shape`, `spacing`, `extent` and `origin` describe the
      computational grid: loops, halos, domain decomposition and SubDomains are
      those of a Grid.
    * Derivatives of Functions defined on it, `u.dx`, `u.dx2`, `u.dxdy`,
      `u.laplace`, `grad`, `div`, nested derivatives such as `(b*u.dx).dx`,
      staggered (`x0`) and rotated (`method='RSFD'`) ones, are derivatives
      w.r.t. the physical coordinates, through the chain rule with the metric
      terms. The outer derivative of a nested one is taken in conservation
      form, so that nested operators such as `(b*u.dx).dx` are symmetric in the
      inner product weighted by the Jacobian, as they are on a Grid (see
      `Derivative`). `u.dx(computational=True)` is the derivative w.r.t. the
      computational coordinate. The transpose `.T` of a physical derivative is
      its exact discrete transpose.
    * The coordinates of SparseFunctions are physical. Interpolation and
      injection are those of a Grid, in the computational grid. As on a Grid,
      injection does not scale by the cell volume: a point source in physical
      space is injected divided by the Jacobian, `src.inject(u, src/J)` with
      `J = grid.jacobian`, read at each point the source is spread over. Its
      injection is then the transpose of interpolation in the inner product
      weighted by the Jacobian, `<P u, d> = sum(J * u * P.T (d/J))`.
    * A read at the mirror point of the written point through a row of the last
      Dimension, e.g. the odd mirror of a free surface on the row `s`,
      `Eq(u.forward.subs(z, s - k), -u.forward.subs(z, s + k))`, is the read at
      the reflection of the written point through the surface of the row,
      normal to it, rather than along the grid line, and an `Inc` writing at the
      mirror point of the point it reads is its transpose (see
      `devito.passes.equations.reflection`). Over a flat surface, it is the mirror of
      a Grid.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the computational domain in grid points.
    mapping : CurvilinearMap
        The map from computational to physical coordinates, e.g.
        TerrainFollowing.
    extent : tuple of floats, optional
        Extent of the computational domain, by default 1 in all dimensions.
    origin : tuple of floats, optional
        Computational coordinate of the origin, by default 0 in all dimensions.
    space_order : int, optional, default=8
        Space order of the coordinate and metric Functions: the order of the
        finite-difference metrics and of their interpolation to staggered
        points, and their halo, `space_order` points per side. The halo bounds
        the reach of physical derivatives: up to two nested derivatives of
        Functions of space order at most `space_order`, e.g. `(b*u.dx).dx` or
        `(b*u.dx.T).dx.T`, are supported, and a derivative of higher order
        raises an error.
    **kwargs
        The other parameters of Grid (`dimensions`, `spacing`, `time_dimension`,
        `dtype`, `comm`, `topology`).

    Attributes
    ----------
    coordinates : tuple of Function
        Physical coordinates of the nodes, one Function per Dimension.
    metrics : dict
        The metric terms `d xi_j / d x_i`, keyed `(physical Dimension,
        computational Dimension)`, Functions such as `eta_x`, or numbers for
        entries constant by construction of the map.
    metric_derivatives : dict
        The second derivatives `d^2 xi_j / d x_i^2`, keyed as `metrics`.
    jacobian : Function
        The Jacobian determinant `J = det(d x / d xi)`, the ratio of a physical
        to a computational cell volume.
    min_spacing : float
        The smallest physical distance between neighbouring nodes, e.g. to
        set a stable time step.

    Examples
    --------
    >>> import numpy as np
    >>> from devito import CurvilinearGrid, Function, TerrainFollowing
    >>> mapping = TerrainFollowing(surface=np.zeros(11))
    >>> grid = CurvilinearGrid(shape=(11, 11), extent=(500., 500.), mapping=mapping)
    >>> u = Function(name='u', grid=grid, space_order=4)
    >>> sorted(f.name for f in grid.metrics.values() if f.is_Function)
    ['eta_x', 'eta_y']

    Notes
    -----
    The coordinate and metric Functions are built from the map when first
    needed, on each rank for its own nodes under domain decomposition, halo
    included: a halo node takes the value of the map continued past the
    domain. `resample` maps Cartesian Functions onto the grid and back.
    """

    is_Curvilinear = True

    def __init__(self, shape, mapping, extent=None, origin=None, space_order=8,
                 **kwargs):
        super().__init__(shape, extent=extent, origin=origin, **kwargs)
        if not isinstance(mapping, CurvilinearMap):
            raise TypeError(f"The mapping must be a CurvilinearMap, not {type(mapping)}")
        if not isinstance(space_order, int) or space_order < 2 or space_order % 2:
            raise ValueError(f"The space_order must be an even integer >= 2, "
                             f"not {space_order}")
        self._mapping = mapping
        self._space_order = space_order
        mapping.validate(self)

    def __repr__(self):
        return 'CurvilinearGrid' + \
            f'[extent={self.extent}, shape={self.shape}, dimensions={self.dimensions}]'

    @property
    def mapping(self):
        """The map from computational to physical coordinates."""
        return self._mapping

    @property
    def space_order(self):
        """Space order of the coordinate and metric Functions."""
        return self._space_order

    @cached_property
    def _computational_names(self):
        """Names of the computational coordinates, one per Dimension."""
        return _COMPUTATIONAL_NAMES[self.dim]

    @cached_property
    def _local_vectors(self):
        """
        Computational coordinates (fp64) of this rank's nodes, halo of the
        metric Functions included, one 1D array per Dimension.
        """
        vectors = []
        for numbering, spacing, origin in zip(self.distributor.glb_numb, self.spacing,
                                              self.origin, strict=True):
            start = numbering[0] if len(numbering) else 0
            indices = np.arange(start - self.space_order,
                                start + len(numbering) + self.space_order)
            vectors.append(np.float64(origin) + np.float64(spacing)*indices)
        return tuple(vectors)

    def _make_function(self, name, values, staggered=None):
        """
        A Function on this grid holding `values` on the local nodes and halo,
        staggered by half a cell along the Dimensions `staggered`.
        """
        function = Function(name=name, grid=self, space_order=self.space_order,
                            interp_order=self.space_order, staggered=staggered,
                            dtype=self.dtype)
        if not self.distributor.loc_empty:
            data = function._data_with_inhalo
            data[:] = np.broadcast_to(values, data.shape)
        return function

    def _symbolic_terms(self, values, name, staggered=None):
        """Metric terms as Functions, keeping constant entries as numbers."""
        terms = {}
        for (i, j), v in values.items():
            physical, computational = self.dimensions[i], self.dimensions[j]
            if np.isscalar(v):
                terms[(physical, computational)] = sympy.sympify(v)
            else:
                terms[(physical, computational)] = self._make_function(
                    name(self._computational_names[j], physical.name), v, staggered
                )
        return terms

    def staggered_terms(self, staggered):
        """
        The metric terms and the Jacobian, as `metrics` and `jacobian`, at the
        points staggered by half a cell along the Dimensions `staggered`,
        computed from the map there, as Functions staggered like a Function
        defined there.
        """
        staggered = tuple(d for d in self.dimensions if d in staggered)
        if not staggered:
            return self.metrics, self.jacobian
        if staggered not in self._staggered_terms:
            self._staggered_terms[staggered] = self._make_staggered_terms(staggered)
        return self._staggered_terms[staggered]

    @cached_property
    def _staggered_terms(self):
        """The `staggered_terms` built so far, by staggering."""
        return {}

    def _make_staggered_terms(self, staggered):
        """`staggered_terms` along the Dimensions `staggered`, in grid order."""
        vectors = tuple(v + np.float64(h)/2 if d in staggered else v
                        for v, d, h in zip(self._local_vectors, self.dimensions,
                                           self.spacing, strict=True))
        suffix = ''.join(d.name for d in staggered)
        metrics = self._symbolic_terms(self.mapping.metrics(self, vectors),
                                       lambda c, p: f'{c}_{p}_{suffix}', staggered)
        jacobian = self._make_function(f'J_{suffix}',
                                       self.mapping.jacobian(self, vectors), staggered)
        return metrics, jacobian

    def terms_at(self, location):
        """
        The metric terms and the Jacobian, as `metrics` and `jacobian`, at
        `location`, a mapper from each Dimension to a node or a half point,
        e.g. `{x: x - h_x/2, y: y}`, computed from the map there.
        """
        offsets = {d: sympy.nsimplify((location.get(d, d) - d)/d.spacing)
                   for d in self.dimensions}
        staggered = tuple(d for d, o in offsets.items() if (2*o) % 2 == 1)
        metrics, jacobian = self.staggered_terms(staggered)
        # Whole cells from the points the Functions are defined at
        shifts = {d: d + (o - sympy.Rational(1, 2) if d in staggered else o)*d.spacing
                  for d, o in offsets.items()}
        shifts = {d: v for d, v in shifts.items() if v != d}
        metrics = {k: v.subs(shifts) for k, v in metrics.items()}
        return metrics, jacobian.subs(shifts)

    @cached_property
    def coordinates(self):
        """Physical coordinates of the nodes, one Function per Dimension."""
        mesh = np.meshgrid(*self._local_vectors, indexing='ij')
        physical = self.mapping.forward(self, mesh)
        return tuple(self._make_function(f'{d.name}_coord', c)
                     for d, c in zip(self.dimensions, physical, strict=True))

    @cached_property
    def metrics(self):
        """
        The metric terms `M[(x_i, d_j)] = d xi_j / d x_i`, keyed by the physical
        Dimension `x_i` and the computational Dimension `d_j`. An entry is a
        Function, named after the computational coordinate and the physical
        Dimension (e.g. `eta_x`), or a number for an entry that is constant by
        construction of the map.
        """
        values = self.mapping.metrics(self, self._local_vectors)
        return self._symbolic_terms(values, lambda c, p: f'{c}_{p}')

    @cached_property
    def metric_derivatives(self):
        """
        The second derivatives `d^2 xi_j / d x_i^2` of the computational
        coordinates, keyed as `metrics` (e.g. `eta_xx`).
        """
        values = self.mapping.metric_derivatives(self, self._local_vectors)
        return self._symbolic_terms(values, lambda c, p: f'{c}_{p}{p}')

    @cached_property
    def jacobian(self):
        """
        The Jacobian determinant `J = det(d x / d xi)`, the ratio of a physical
        to a computational cell volume, as a Function.
        """
        values = self.mapping.jacobian(self, self._local_vectors)
        return self._make_function('J', values)

    @cached_property
    def min_spacing(self):
        """
        Smallest physical distance between two neighbouring nodes of the domain,
        over all ranks. It plays the role of the grid spacing in a stability
        condition, e.g. a CFL time step `dt <= c * min_spacing / vmax`, since the
        map shrinks some cells below the computational spacing.
        """
        domain = tuple(v[self.space_order:-self.space_order] for v in self._local_vectors)
        if self.distributor.loc_empty:
            minimum = np.inf
        else:
            mesh = np.meshgrid(*domain, indexing='ij')
            physical = [np.broadcast_to(c, mesh[0].shape)
                        for c in self.mapping.forward(self, mesh)]
            minimum = np.inf
            for axis in range(self.dim):
                if mesh[0].shape[axis] < 2:
                    continue
                distance = np.sqrt(sum(np.diff(c, axis=axis)**2 for c in physical))
                minimum = min(minimum, distance.min())
        if self.distributor.is_parallel:
            minimum = self.comm.allreduce(minimum, op=MPI.MIN)
        return self.dtype(minimum)

    def to_computational(self, points):
        """
        Computational coordinates of physical points, through the inverse of the
        map. This is how SparseFunction coordinates, which are physical, are
        placed on the grid.

        Parameters
        ----------
        points : array_like
            Physical coordinates of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            Computational coordinates of shape `(npoint, ndim)`, in the frame of
            the `origin` and `spacing` of the grid.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        return self.mapping.inverse(self, points)

    def to_physical(self, points):
        """
        Physical coordinates of computational points, through the map, e.g. to
        place a source exactly on a grid node.

        Parameters
        ----------
        points : array_like
            Computational coordinates of shape `(npoint, ndim)`, in the frame of
            the `origin` and `spacing` of the grid.

        Returns
        -------
        ndarray
            Physical coordinates of shape `(npoint, ndim)`.

        Examples
        --------
        >>> import numpy as np
        >>> from devito import CurvilinearGrid, TerrainFollowing
        >>> grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.),
        ...                        mapping=TerrainFollowing(lambda x: -0.1*x))
        >>> grid.to_physical([[50., 0.]])
        array([[50., -5.]])
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        physical = self.mapping.forward(self, [points[:, i] for i in range(self.dim)])
        return np.stack(np.broadcast_arrays(*physical), axis=1)

    def _resampling_weights(self, cartesian_grid):
        """
        Multilinear interpolation weights from the nodes of `cartesian_grid` to
        this rank's nodes, as a list of `(cartesian node indices, weights)`
        pairs, one per corner of the interpolation cell. Positions are clamped
        to the Cartesian domain, and snapped to a node within roundoff, so that
        a node shared by both grids is copied exactly.
        """
        if cartesian_grid.dim != self.dim or cartesian_grid.is_Curvilinear:
            raise ValueError(f"Cannot resample from {cartesian_grid}, which is not a "
                             f"{self.dim}D Cartesian Grid")
        positions = []
        for coordinate, origin, spacing, size in zip(
            self.coordinates, cartesian_grid.origin, cartesian_grid.spacing,
            cartesian_grid.shape, strict=True
        ):
            values = np.asarray(coordinate.data_ro_domain._local, dtype=np.float64)
            position = (values - np.float64(origin))/np.float64(spacing)
            nearest = np.rint(position)
            position = np.where(np.abs(position - nearest) < 1e-6, nearest, position)
            positions.append(np.clip(position, 0, size - 1))

        base = [np.minimum(np.floor(p).astype(np.int64), max(size - 2, 0))
                for p, size in zip(positions, cartesian_grid.shape, strict=True)]
        fractions = [p - b for p, b in zip(positions, base, strict=True)]
        corners = []
        for corner in np.ndindex(*(2,)*self.dim):
            weight = 1
            index = []
            for c, b, f, size in zip(corner, base, fractions, cartesian_grid.shape,
                                     strict=True):
                weight = weight * (f if c else 1 - f)
                index.append(np.minimum(b + c, size - 1))
            corners.append((tuple(index), weight))
        return corners

    def _global_data(self, function):
        """The global domain values of a possibly distributed `function`."""
        values = np.zeros(function.shape_global, dtype=np.float64)
        if not function.grid.distributor.loc_empty:
            values[function.local_indices] = function.data_ro_domain._local
        if function.grid.distributor.is_parallel:
            values = function.grid.comm.allreduce(values, op=MPI.SUM)
        return values

    def resample(self, function, cartesian, transpose=False):
        """
        Resample a Function between a Cartesian Grid and this grid.

        With `transpose=False`, `function` is set to `cartesian` interpolated at
        the physical position of each node of this grid. With `transpose=True`,
        `cartesian` is set to the exact transpose of that interpolation applied
        to `function`. The interpolation is multilinear in the Cartesian grid,
        which reduces to a 1D interpolation along each column when the lateral
        nodes of both grids coincide, as for a TerrainFollowing map. Nodes above
        or below the Cartesian domain take the value of its nearest node.

        Parameters
        ----------
        function : Function
            A Function on this grid.
        cartesian : Function
            A Function on a Cartesian Grid of the same dimension, with any shape,
            extent, origin and domain decomposition.
        transpose : bool, optional, default=False
            Apply the transpose, from `function` to `cartesian`.

        Examples
        --------
        >>> import numpy as np
        >>> from devito import CurvilinearGrid, Function, Grid, TerrainFollowing
        >>> mapping = TerrainFollowing(surface=np.linspace(0., 20., 11))
        >>> grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.),
        ...                        mapping=mapping)
        >>> cartesian = Function(name='c', grid=Grid(shape=(11, 21),
        ...                                          extent=(100., 100.)))
        >>> cartesian.data[:] = 1.5
        >>> velocity = Function(name='v', grid=grid)
        >>> grid.resample(velocity, cartesian)
        >>> float(velocity.data.min())
        1.5
        """
        if function.grid.root is not self:
            raise ValueError(f"{function} is not defined on {self}")
        corners = self._resampling_weights(cartesian.grid.root)

        if not transpose:
            values = self._global_data(cartesian)
            resampled = sum(w*values[index] for index, w in corners)
            if not self.distributor.loc_empty:
                function.data_local[:] = resampled
            return

        values = np.asarray(function.data_ro_domain._local, dtype=np.float64)
        transposed = np.zeros(cartesian.shape_global, dtype=np.float64)
        for index, weight in corners:
            np.add.at(transposed, index, weight*values)
        if self.distributor.is_parallel:
            transposed = self.comm.allreduce(transposed, op=MPI.SUM)
        cartesian.data[:] = transposed

    def __getstate__(self):
        state = super().__getstate__()
        # The coordinate and metric Functions are rebuilt lazily from the map
        for i in ('coordinates', 'metrics', 'metric_derivatives', 'jacobian',
                  '_staggered_terms', '_local_vectors', 'min_spacing'):
            state.pop(i, None)
        return state
