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
from itertools import product

import numpy as np
import sympy

from devito.mpi import MPI
from devito.types.dense import Function
from devito.types.grid import Grid

__all__ = ['CurvilinearGrid', 'CurvilinearMap', 'TerrainFollowing']


# Names of the computational coordinates, one per grid axis
_COMPUTATIONAL_NAMES = {1: ('xi',), 2: ('xi', 'eta'), 3: ('xi', 'eta', 'zeta')}


def _window(values, axis, start, stop):
    """The view of `values` from `start` to `stop` along `axis`."""
    index = [slice(None)]*values.ndim
    index[axis] = slice(start, stop)
    return values[tuple(index)]


def _centered_fd(values, axis, spacing, order, deriv_order=1):
    """
    Centered finite-difference derivative of `values` along `axis`, of order
    `order`.

    The result drops `order//2` points at each end of `axis`. Each tap is
    differenced against the center value, so that a constant array has an
    exactly zero derivative.
    """
    radius = order // 2
    indices = list(range(-radius, radius + 1))
    weights = sympy.finite_diff_weights(deriv_order, indices, 0)[deriv_order][-1]
    npoint = values.shape[axis]
    center = _window(values, axis, radius, npoint - radius)
    derivative = np.zeros(center.shape, dtype=np.result_type(center, np.float64))
    for i, w in zip(indices, weights, strict=True):
        if w != 0 and i != 0:
            derivative += float(w)*(_window(values, axis, radius + i,
                                            npoint - radius + i) - center)
    return derivative / spacing**deriv_order


def _lagrange(positions, npoint, size):
    """
    The `npoint` nodes nearest to the fractional node positions `positions`, of
    an axis of `size` nodes, and the weights of the Lagrange polynomial through
    them at `positions`, as `(first node, [weight of each node])`. A position
    on a node has the weight one there.
    """
    positions = np.asarray(positions, dtype=np.float64)
    first = np.clip(np.floor(positions).astype(np.int64) - (npoint//2 - 1), 0,
                    size - npoint)
    offsets = positions - first
    weights = []
    for k in range(npoint):
        weight = np.ones_like(offsets)
        for j in range(npoint):
            if j != k:
                weight = weight*(offsets - j)/(k - j)
        weights.append(weight)
    return first, weights


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


def _cofactors(matrix):
    """
    The cofactors `C[i][j]` of a 1x1, 2x2 or 3x3 matrix of arrays `matrix[i][j]`,
    and its determinant.
    """
    n = len(matrix)
    if n == 1:
        cofactors = [[np.ones_like(matrix[0][0])]]
    elif n == 2:
        (a, b), (c, d) = matrix
        cofactors = [[d, -c], [-b, a]]
    else:
        m = matrix
        cofactors = [[m[(i + 1) % 3][(j + 1) % 3]*m[(i + 2) % 3][(j + 2) % 3] -
                      m[(i + 1) % 3][(j + 2) % 3]*m[(i + 2) % 3][(j + 1) % 3]
                      for j in range(3)] for i in range(3)]
    determinant = sum(matrix[0][j]*cofactors[0][j] for j in range(n))
    return cofactors, determinant


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
      points, needed to place SparseFunctions.

    The metric terms, their derivatives and the Jacobian default to finite
    differences of `forward` at the grid space order. A map with closed forms
    overrides `terms` and `jacobian`, and `validate`
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

    #: True if the map keeps the lateral coordinates, `x_i = xi_i` for every
    #: Dimension but the last, as a terrain-following map does
    keeps_lateral = False

    def validate(self, grid):
        """
        Raise an informative error if the map is incompatible with `grid`.
        Called upon creation of a CurvilinearGrid.
        """
        return

    def _jacobian_matrix(self, grid, vectors, npad):
        """
        `A[i][j] = d x_i / d xi_j` on the nodes spanned by `vectors`, extended by
        `npad` nodes past each end, as nested lists of arrays, obtained by
        finite differences of `forward`.
        """
        radius = grid.space_order // 2
        extended = [_extend(v, npad + radius) for v in vectors]
        shape = tuple(v.size for v in extended)
        physical = self.forward(grid, np.meshgrid(*extended, indexing='ij', sparse=True))
        if len(physical) != len(vectors):
            raise ValueError(f"{type(self).__name__}.forward returned {len(physical)} "
                             f"coordinates for a {len(vectors)}D grid")
        physical = [np.broadcast_to(np.asarray(c, dtype=np.float64), shape)
                    for c in physical]
        if not all(np.isfinite(c).all() for c in physical):
            raise ValueError(f"{type(self).__name__}.forward returned non-finite "
                             "coordinates")
        spacing = [v[1] - v[0] for v in vectors]
        ndim = len(vectors)
        matrix = []
        for i in range(ndim):
            row = []
            for j in range(ndim):
                derivative = _centered_fd(physical[i], j, spacing[j], grid.space_order)
                # Crop the non-differentiated axes to the same extent
                for k in range(ndim):
                    if k != j:
                        derivative = _window(derivative, k, radius, shape[k] - radius)
                row.append(derivative)
            matrix.append(row)
        return matrix

    def terms(self, grid, vectors, derivatives=True):
        """
        The metric terms `d xi_j / d x_i`, their second derivatives
        `d^2 xi_j / d x_i^2`, as `sum_k (d xi_k / d x_i) d/d xi_k (d xi_j / d x_i)`,
        and the Jacobian determinant `det(d x / d xi)`, the ratio of a physical to
        a computational cell volume, on the nodes spanned by `vectors`.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map is attached to.
        vectors : tuple of ndarray
            Computational coordinates of the nodes, one 1D array per axis.
        derivatives : bool, optional, default=True
            Compute the second derivatives, else return None for them.

        Returns
        -------
        tuple
            The metric terms and their derivatives, as dicts `{(i, j): value}`,
            and the Jacobian.
        """
        radius = grid.space_order // 2 if derivatives else 0
        cofactors, determinant = _cofactors(self._jacobian_matrix(grid, vectors,
                                                                  radius))
        ndim = len(vectors)
        # inverse[j][i] = d xi_j / d x_i = C[i][j] / det, in place of the cofactors
        inverse = [[None]*ndim for _ in range(ndim)]
        for i in range(ndim):
            for j in range(ndim):
                inverse[j][i] = np.divide(cofactors[i][j], determinant,
                                          out=np.asarray(cofactors[i][j]))
        del cofactors
        spacing = [v[1] - v[0] for v in vectors]
        metrics = {(i, j): _crop(inverse[j][i], radius) for i in range(ndim)
                   for j in range(ndim)}
        if not derivatives:
            return metrics, None, determinant
        derivatives = {}
        for i in range(ndim):
            for j in range(ndim):
                value = 0
                for k in range(ndim):
                    derivative = _centered_fd(inverse[j][i], k, spacing[k],
                                              grid.space_order)
                    for m in range(ndim):
                        if m != k:
                            derivative = _window(derivative, m, radius,
                                                 derivative.shape[m] - radius)
                    value = value + metrics[(i, k)]*derivative
                derivatives[(i, j)] = value
        return metrics, derivatives, _crop(determinant, radius)

    def jacobian(self, grid, vectors):
        """
        The Jacobian determinant `det(d x / d xi)` on the nodes spanned by
        `vectors` (see `terms`).
        """
        return _cofactors(self._jacobian_matrix(grid, vectors, 0))[1]


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
        of the grid by odd reflection, interpolated between nodes by Lagrange
        polynomials of the grid space order, e.g. at half points or to place
        sparse points, and differentiated by finite differences for the metric
        terms; a callable is evaluated exactly.

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

    keeps_lateral = True

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
        if not callable(self._surface) and \
                self._surface.shape != tuple(grid.shape[:-1]):
            raise ValueError(f"The surface has shape {self._surface.shape}, expected one "
                             f"value per lateral grid node {tuple(grid.shape[:-1])}")
        lateral = np.meshgrid(*[np.float64(o) + np.float64(h)*np.arange(n)
                                for o, h, n in zip(grid.origin[:-1], grid.spacing[:-1],
                                                   grid.shape[:-1], strict=True)],
                              indexing='ij')
        surface = self.surface_at(grid, lateral)
        if not np.all(np.isfinite(surface)) or np.max(surface) >= self._bottom(grid):
            raise ValueError("A TerrainFollowing map requires a finite surface above "
                             f"the bottom of the grid, at depth {self._bottom(grid)}")

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
        axes = []
        for axis, values in enumerate(lateral):
            spacing = np.float64(grid.spacing[axis])
            position = (np.asarray(values) - np.float64(grid.origin[axis])) / spacing
            nearest = np.rint(position)
            position = np.where(np.abs(position - nearest) < 1e-9, nearest, position)
            axes.append(_lagrange(np.clip(position + npad, 0, padded.shape[axis] - 1),
                                  grid.space_order, padded.shape[axis]))

        # Tensor-product Lagrange interpolation of the padded surface, of the
        # order of the grid
        value = 0
        for taps in product(range(grid.space_order), repeat=len(axes)):
            weight = 1
            index = []
            for k, (first, weights) in zip(taps, axes, strict=True):
                weight = weight*weights[k]
                index.append(first + k)
            value = value + weight*padded[tuple(np.broadcast_arrays(*index))]
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

    def terms(self, grid, vectors, derivatives=True):
        ndim = len(vectors)
        last = ndim - 1
        bottom, height, eta, first, second = self._column_terms(grid, vectors)
        metrics = {(i, j): int(i == j) for i in range(ndim) for j in range(last)}
        for i in range(last):
            metrics[(i, last)] = first[i]*eta/height
        metrics[(last, last)] = bottom/height
        if not derivatives:
            return metrics, None, height/bottom
        second_derivatives = {(i, j): 0 for i in range(ndim) for j in range(ndim)}
        for i in range(last):
            second_derivatives[(i, last)] = eta*(second[i]/height +
                                                 2*first[i]**2/height**2)
        return metrics, second_derivatives, height/bottom

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
      terms. `u.dx(computational=True)` is the derivative w.r.t. the
      computational coordinate. The transpose `.T` of a physical derivative is
      its exact discrete transpose, so a self-adjoint operator is written as one,
      e.g. `-(J*b*u.dx).dx.T/J` with `J = grid.jacobian`, symmetric in the inner
      product weighted by the Jacobian.
    * The coordinates of SparseFunctions are physical. Interpolation and
      injection are those of a Grid, in the computational grid. As on a Grid,
      injection does not scale by the cell volume: a point source in physical
      space is injected divided by the Jacobian, `src.inject(u, src/J)` with
      `J = grid.jacobian`, read at each point the source is spread over. Its
      injection is then the transpose of interpolation in the inner product
      weighted by the Jacobian, `<P u, d> = sum(J * u * P.T (d/J))`.
    * Indices are computational, as on a Grid: `u.subs(z, 2*s - z)` reads the
      row `2*s - z`. A free surface at the row `s` of a map whose surface is a
      grid row, e.g. the odd mirror `Eq(u.forward, -u.forward.subs(z, 2*s - z),
      subdomain=above)`, mirrors along the grid lines, through the surface,
      which is flat in computational space.

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
    domain. `devito.resample` maps Cartesian Functions onto the grid and back.
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
        self._validate_orientation()

    def __repr__(self):
        return 'CurvilinearGrid' + \
            f'[extent={self.extent}, shape={self.shape}, dimensions={self.dimensions}]'

    def _validate_orientation(self):
        """
        Raise an error unless the Jacobian of the map is finite and positive on
        the nodes of every rank, halo included: a map that is singular or turns
        cells inside out has no metric terms.
        """
        jacobian = np.asarray(self.mapping.jacobian(self, self._local_vectors))
        valid = bool(np.all(np.isfinite(jacobian)) and np.all(jacobian > 0))
        if self.distributor.is_parallel:
            valid = self.comm.allreduce(valid, op=MPI.LAND)
        if not valid:
            raise ValueError(f"{type(self.mapping).__name__} has a Jacobian that is not "
                             "finite and positive on the nodes of the grid: the map "
                             "is singular or turns cells inside out")

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
        metrics, _, jacobian = self._terms(staggered)
        return metrics, jacobian

    def terms_at(self, location):
        """
        The metric terms and the Jacobian, as `metrics` and `jacobian`, at
        `location`, a mapper from each Dimension to a node or a half point,
        e.g. `{x: x - h_x/2, y: y}`, computed from the map there.
        """
        staggered, shifts = self._staggering(location)
        metrics, jacobian = self.staggered_terms(staggered)
        return {k: v.subs(shifts) for k, v in metrics.items()}, jacobian.subs(shifts)

    def metric_derivatives_at(self, location):
        """
        The second derivatives of the computational coordinates, as
        `metric_derivatives`, at `location` (see `terms_at`), computed from the
        map there.
        """
        staggered, shifts = self._staggering(location)
        derivatives = self._terms(staggered, derivatives=True)[1]
        return {k: v.subs(shifts) for k, v in derivatives.items()}

    @cached_property
    def _terms_cache(self):
        """The `_terms` built so far, by staggering."""
        return {}

    def _terms(self, staggered, derivatives=False):
        """
        The metric terms, their second derivatives if `derivatives` (else None),
        and the Jacobian at the points staggered by half a cell along the
        Dimensions `staggered`, from one evaluation of the map there. The
        Functions built for a staggering are kept, so each is built once.
        """
        staggered = tuple(d for d in self.dimensions if d in staggered)
        cached = self._terms_cache.get(staggered)
        if cached is not None and (cached[1] is not None or not derivatives):
            return cached

        vectors = self._staggered_vectors(staggered)
        metrics, second, jacobian = self.mapping.terms(self, vectors,
                                                       derivatives=derivatives)
        suffix = ''.join(d.name for d in staggered)
        suffix = f'_{suffix}' if suffix else ''
        if cached is None:
            metrics = self._symbolic_terms(metrics, lambda c, p: f'{c}_{p}{suffix}',
                                           staggered)
            jacobian = self._make_function(f'J{suffix}', jacobian, staggered)
        else:
            metrics, _, jacobian = cached
        if second is not None:
            second = self._symbolic_terms(second, lambda c, p: f'{c}_{p}{p}{suffix}',
                                          staggered)
        self._terms_cache[staggered] = (metrics, second, jacobian)
        return self._terms_cache[staggered]

    def _staggered_vectors(self, staggered):
        """`_local_vectors` shifted by half a cell along the Dimensions `staggered`."""
        return tuple(v + np.float64(h)/2 if d in staggered else v
                     for v, d, h in zip(self._local_vectors, self.dimensions,
                                        self.spacing, strict=True))

    def _staggering(self, location):
        """
        The Dimensions along which `location`, a mapper from each Dimension to a
        node or a half point, is staggered, and the shift of whole cells from
        the points the Functions staggered there are defined at.
        """
        offsets = {}
        for d in self.dimensions:
            offset = sympy.nsimplify((location.get(d, d) - d)/d.spacing)
            if not offset.is_Rational or offset.q > 2:
                raise ValueError(f"The metric terms are defined at the nodes and the "
                                 f"half points, not at {location[d]} along {d}")
            offsets[d] = offset
        staggered = tuple(d for d, o in offsets.items() if o.q == 2)
        shifts = {d: d + (o - sympy.Rational(1, 2) if d in staggered else o)*d.spacing
                  for d, o in offsets.items()}
        return staggered, {d: v for d, v in shifts.items() if v != d}

    @cached_property
    def coordinates(self):
        """Physical coordinates of the nodes, one Function per Dimension."""
        mesh = np.meshgrid(*self._local_vectors, indexing='ij')
        physical = self.mapping.forward(self, mesh)
        return tuple(self._make_function(f'{d.name}_coord', c)
                     for d, c in zip(self.dimensions, physical, strict=True))

    @property
    def metrics(self):
        """
        The metric terms `M[(x_i, d_j)] = d xi_j / d x_i`, keyed by the physical
        Dimension `x_i` and the computational Dimension `d_j`. An entry is a
        Function, named after the computational coordinate and the physical
        Dimension (e.g. `eta_x`), or a number for an entry that is constant by
        construction of the map.
        """
        return self._terms(())[0]

    @property
    def metric_derivatives(self):
        """
        The second derivatives `d^2 xi_j / d x_i^2` of the computational
        coordinates, keyed as `metrics` (e.g. `eta_xx`).
        """
        return self._terms((), derivatives=True)[1]

    @property
    def jacobian(self):
        """
        The Jacobian determinant `J = det(d x / d xi)`, the ratio of a physical
        to a computational cell volume, as a Function.
        """
        return self._terms(())[2]

    @cached_property
    def min_spacing(self):
        """
        The smallest physical length of the cells of the domain, over all ranks,
        in the sense of a stability condition: it plays the role of the grid
        spacing in a CFL time step, e.g. `dt <= c * min_spacing / vmax`.

        At each node, it is `1/sqrt(l)`, with `l` the largest eigenvalue of the
        metric tensor scaled by the spacing, `G_jk = sum_i (d xi_j/d x_i)
        (d xi_k/d x_i) / (h_j h_k)`, the largest inverse squared length a
        stencil sees there. On a Cartesian grid it is the smallest spacing; it
        accounts for cells shrunk by the map, and for skewed ones, whose height
        across their longest edges is smaller than any of their edges.
        """
        if self.distributor.loc_empty:
            minimum = np.inf
        else:
            shape = self.shape_local
            spacing = [np.float64(h) for h in self.spacing]
            dimensions = self.dimensions
            # M[..., j, i] = d xi_j / d x_i / h_j, from the metric Functions
            matrix = np.empty(shape + (self.dim, self.dim))
            for (physical, computational), value in self.metrics.items():
                i, j = dimensions.index(physical), dimensions.index(computational)
                if value.is_Function:
                    value = np.asarray(value.data_local, dtype=np.float64)
                else:
                    value = float(value)
                matrix[..., j, i] = value/spacing[j]
            tensor = matrix @ np.swapaxes(matrix, -1, -2)
            minimum = 1/np.sqrt(np.linalg.eigvalsh(tensor)[..., -1].max())
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

    def __getstate__(self):
        state = super().__getstate__()
        # The coordinate and metric Functions are rebuilt lazily from the map
        for i in ('coordinates', '_terms_cache', '_local_vectors', 'min_spacing'):
            state.pop(i, None)
        return state
