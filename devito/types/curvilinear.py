"""
Curvilinear grids: structured grids whose nodes a map places in physical space.

A CurvilinearGrid is a Grid over computational coordinates `xi`, which a
CurvilinearMap sends to physical coordinates `x(xi)`. Loops, halos, domain
decomposition and SubDomains are those of the computational grid, while
derivatives are taken w.r.t. the physical coordinates through the metric terms
`d xi_j/d x_i` of the map (see `devito.finite_differences.curvilinear`).
"""
from collections import namedtuple
from functools import cached_property
from itertools import combinations_with_replacement, product

import numpy as np
import sympy

from devito.mpi import MPI
from devito.tools import generator
from devito.types.dense import Function
from devito.types.grid import Grid

__all__ = ['CurvilinearGrid', 'CurvilinearMap', 'MeshMap', 'TerrainFollowing']

# Names of the computational coordinates, by number of Dimensions
_NAMES = {1: ('xi',), 2: ('xi', 'eta'), 3: ('xi', 'eta', 'zeta')}

# The Newton iterations of `CurvilinearMap.inverse`: their number, the residual
# they stop at, relative to the extent of the grid, and the step of the finite
# differences of the map, relative to the spacing
_NEWTON_ITERATIONS = 50
_NEWTON_TOLERANCE = 1e-12
_NEWTON_STEP = 1e-6

# The spread, relative to its magnitude, under which a metric term is a constant
_CONSTANT_TOLERANCE = 1e-10

# The number of each CurvilinearGrid, which suffixes the names of its geometric
# Functions so that those of two grids differ
_grid_number = generator()

# The coarse search of `MeshMap.guess`: the mesh is searched every `1/16` of its
# smallest size, for blocks of 4096 points at a time to bound its memory
_COARSE_NODES = 16
_SEARCH_BLOCK = 4096

# The metric terms, the Jacobian and its inverse at the points of one
# staggering, each a Function or a number (see `CurvilinearGrid.metrics_at`)
_MetricTerms = namedtuple('MetricTerms',
                          ('metrics', 'jacobian', 'inverse_jacobian', 'normal_ratios'))


class CurvilinearMap:

    """
    A map from the computational coordinates of a grid to physical coordinates.

    A subclass implements `forward`, and may override `validate`, `guess` and
    `inverse`, e.g. with a closed form. The metric terms of a CurvilinearGrid
    are centered finite differences of `forward` at its nodes and half points.

    Parameters
    ----------
    order : int, optional, default=8
        The order of the finite differences of the map, and of the
        interpolation of a tabulated map. It is also the halo of the metric
        Functions of the grid, which the stencils of the derivatives read.
    """

    def __init__(self, order=8):
        if order < 2 or order % 2:
            raise ValueError(f"The order of a map is an even number from 2, not {order}")
        self._order = order

    @property
    def order(self):
        """The order of the finite differences and interpolation of the map."""
        return self._order

    def validate(self, grid):
        """
        Check that the map can place the nodes of a grid.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map places.

        Raises
        ------
        ValueError
            If it cannot.
        """

    def forward(self, grid, computational):
        """
        The physical coordinates of computational ones.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map places.
        computational : tuple of ndarray
            The computational coordinates, one broadcastable array per
            Dimension of `grid`.

        Returns
        -------
        tuple of ndarray
            The physical coordinates, one array per Dimension, broadcastable
            to the shape of the computational ones.
        """
        raise NotImplementedError

    def inverse(self, grid, points):
        """
        The computational coordinates of physical points, by Newton iterations
        from `guess`.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map places.
        points : ndarray
            The physical points, of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            The computational coordinates, of shape `(npoint, ndim)`.

        Raises
        ------
        ValueError
            If the iterations do not converge, e.g. for points the map does
            not reach.
        """
        points = np.asarray(points, dtype=np.float64)
        tolerance = _NEWTON_TOLERANCE*max(float(e) for e in grid.extent)
        steps = [_NEWTON_STEP*float(h) for h in grid.spacing]

        current = self.guess(grid, points)
        for _ in range(_NEWTON_ITERATIONS):
            residual = self.forward_points(grid, current) - points
            if np.abs(residual).max(initial=0) <= tolerance:
                return current

            # The Jacobian matrix of the map at each point, by centered differences
            columns = [(self.forward_points(grid, current + step*unit) -
                        self.forward_points(grid, current - step*unit))/(2*step)
                       for step, unit in zip(steps, np.eye(grid.dim), strict=True)]
            jacobian = np.stack(columns, axis=-1)
            try:
                step = np.linalg.solve(jacobian, residual[..., None])[..., 0]
            except np.linalg.LinAlgError:
                # The map is singular at some point, e.g. saturated
                break
            current = current - step

        unplaced = int(np.sum(~(np.abs(residual).max(axis=-1) <= tolerance)))
        raise ValueError(f"{type(self).__name__} cannot place {unplaced} of "
                         f"{len(points)} points: they may lie outside the map")

    def guess(self, grid, points):
        """
        The computational coordinates `inverse` starts from.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map places.
        points : ndarray
            The physical points, of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            The physical points themselves, unless a subclass knows better.
        """
        return np.array(points, dtype=np.float64)

    def forward_points(self, grid, points):
        """
        The physical coordinates of computational points (see `forward`).

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map places.
        points : ndarray
            The computational points, of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            The physical points, of shape `(npoint, ndim)`.
        """
        physical = self.forward(grid, tuple(points.T))
        return np.stack(np.broadcast_arrays(*physical), axis=-1)


class TerrainFollowing(CurvilinearMap):

    """
    A terrain-following map: the grid follows a surface, e.g. topography,
    which becomes the grid row at computational depth 0.

    The columns stay vertical, and their nodes are spread evenly between the
    surface and the flat bottom of the grid. With the vertical axis last and
    pointing down, a node of computational coordinates `(x, eta)` is at the
    depth

        z(x, eta) = s(x) + eta * (b - s(x)) / b

    with `s` the depth of the surface and `b` the computational depth of the
    last row. A negative vertical origin gives rows above the surface, e.g.
    for the mirror of a free surface.

    Parameters
    ----------
    surface : array_like or callable
        Depth of the surface, positive down like the vertical axis of the grid:
        an array with one value per lateral grid node, interpolated between
        them, or a callable of the lateral physical coordinates.
    order : int, optional, default=8
        The order of the finite differences and interpolation of the map (see
        `CurvilinearMap`).

    Examples
    --------
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

    def __init__(self, surface, order=8):
        super().__init__(order)
        if callable(surface):
            self._surface = surface
        else:
            self._surface = np.array(surface, dtype=np.float64)

    @property
    def surface(self):
        """The depth of the surface, an array of lateral node values or a callable."""
        return self._surface

    def validate(self, grid):
        if grid.dim < 2:
            raise ValueError("A TerrainFollowing map needs a lateral Dimension")
        if not callable(self._surface) and self._surface.shape != grid.shape[:-1]:
            raise ValueError(f"The surface has shape {self._surface.shape}, expected one "
                             f"value per lateral node, {grid.shape[:-1]}")
        if _bottom(grid) <= 0:
            raise ValueError("A TerrainFollowing map needs the last row of the grid at "
                             "a positive computational depth")

    def forward(self, grid, computational):
        *lateral, eta = [np.asarray(c, dtype=np.float64) for c in computational]
        surface = self._surface_at(grid, lateral)
        bottom = _bottom(grid)
        return (*lateral, surface + eta*(bottom - surface)/bottom)

    def inverse(self, grid, points):
        # The map is linear along each column, so it inverts in closed form
        points = np.asarray(points, dtype=np.float64)
        surface = self._surface_at(grid, list(points[:, :-1].T))
        bottom = _bottom(grid)
        eta = bottom*(points[:, -1] - surface)/(bottom - surface)
        return np.concatenate([points[:, :-1], eta[:, None]], axis=1)

    def _surface_at(self, grid, lateral):
        """The depth of the surface at the lateral physical coordinates `lateral`."""
        if callable(self._surface):
            shape = np.broadcast_shapes(*(np.shape(v) for v in lateral))
            return np.broadcast_to(np.asarray(self._surface(*lateral), dtype=np.float64),
                                   shape)

        # The surface is tabulated at the lateral nodes of the grid
        positions = [(v - float(o))/float(h)
                     for v, o, h in zip(lateral, grid.origin[:-1], grid.spacing[:-1],
                                        strict=True)]
        return _interpolate(self._surface, positions, self.order)


class MeshMap(CurvilinearMap):

    """
    The map given by the physical coordinates of the nodes of a mesh,
    interpolated between them and continued past its edges by odd reflection.

    The grid lines may bend in any direction, e.g. around an overhang of a
    seabed that a surface cannot describe.

    Parameters
    ----------
    coordinates : tuple of array_like
        The physical coordinates of the nodes, one array of the shape of the
        mesh per Dimension. They are read by slices only, so memory maps work.
    padding : tuple of int, optional
        The number of grid nodes before the first node of the mesh along each
        Dimension, e.g. for absorbing layers around it. Defaults to 0.
    order : int, optional, default=8
        The order of the finite differences and interpolation of the map (see
        `CurvilinearMap`).

    Examples
    --------
    >>> import numpy as np
    >>> from devito import CurvilinearGrid, MeshMap
    >>> xi, eta = np.meshgrid(np.linspace(0., 100., 11), np.linspace(0., 100., 11),
    ...                       indexing='ij')
    >>> mapping = MeshMap((xi + .1*eta, eta))
    >>> grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.), mapping=mapping)
    >>> float(grid.coordinates[0].data[0, -1])
    10.0
    """

    def __init__(self, coordinates, padding=None, order=8):
        super().__init__(order)
        self._coordinates = tuple(coordinates)
        shapes = {np.shape(c) for c in self._coordinates}
        if len(shapes) != 1:
            raise ValueError(f"The coordinates have the shapes {shapes}, not one")
        self._padding = tuple(padding or (0,)*len(self._coordinates))

    @property
    def coordinates(self):
        """The physical coordinates of the nodes of the mesh."""
        return self._coordinates

    @property
    def padding(self):
        """The number of grid nodes before the mesh along each Dimension."""
        return self._padding

    @property
    def shape(self):
        """The shape of the mesh."""
        return np.shape(self._coordinates[0])

    def validate(self, grid):
        if len(self._coordinates) != grid.dim:
            raise ValueError(f"The mesh has {len(self._coordinates)} coordinates, the "
                             f"grid {grid.dim} Dimensions")

    def forward(self, grid, computational):
        positions = self._positions(grid, computational)
        return tuple(_interpolate(c, positions, self.order) for c in self._coordinates)

    def guess(self, grid, points):
        """
        The computational coordinates of the node of a coarse mesh nearest to
        each point (see `CurvilinearMap.guess`).
        """
        # Every `stride`-th node, so that the search stays small on large meshes
        stride = max(1, min(self.shape)//_COARSE_NODES)
        coarse = tuple(slice(None, None, stride) for _ in self.shape)
        nodes = np.stack([np.asarray(c[coarse], dtype=np.float64).ravel()
                          for c in self._coordinates], axis=1)

        # The nearest coarse node, a block of points at a time to bound memory
        nearest = np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), _SEARCH_BLOCK):
            block = slice(start, start + _SEARCH_BLOCK)
            distances = ((points[block, None, :] - nodes[None])**2).sum(axis=-1)
            nearest[block] = np.argmin(distances, axis=1)

        indices = np.unravel_index(nearest, np.shape(self._coordinates[0][coarse]))
        return np.stack([float(o) + float(h)*(stride*i + p)
                         for i, o, h, p in zip(indices, grid.origin, grid.spacing,
                                               self._padding, strict=True)], axis=1)

    def _positions(self, grid, computational):
        """The fractional mesh indices of the `computational` coordinates."""
        return [(np.asarray(c, dtype=np.float64) - float(o))/float(h) - p
                for c, o, h, p in zip(computational, grid.origin, grid.spacing,
                                      self._padding, strict=True)]


def _bottom(grid):
    """The computational depth of the last row of `grid`."""
    return float(grid.origin[-1]) + float(grid.extent[-1])


def _interpolate(values, positions, order):
    """
    Lagrange interpolation of tabulated values at fractional indices.

    `values` is continued past its edges by odd reflection, `v(-i) = 2 v(0) -
    v(i)`, which keeps linear trends, and only the window of it the positions
    reach is read.

    Parameters
    ----------
    values : array_like
        The tabulated values, read by slices only.
    positions : list of ndarray
        The fractional indices along each axis of `values`, broadcastable to
        one another. If each varies along its own axis only, as on the nodes
        of a grid, the interpolation is separable and done axis by axis.
    order : int
        The number of Lagrange points along each axis.

    Returns
    -------
    ndarray
        The interpolated values, of the broadcast shape of `positions`.
    """
    positions = [np.asarray(p, dtype=np.float64) for p in positions]
    shape = np.shape(values)

    # The window of `values` the positions read, padded past the edges of
    # `values` by odd reflection as far as the positions reach
    starts = [max(0, int(np.floor(p.min(initial=0))) - order) for p in positions]
    stops = [min(n, int(np.ceil(p.max(initial=0))) + order + 1)
             for p, n in zip(positions, shape, strict=True)]
    window = np.asarray(values[tuple(slice(a, b) for a, b in zip(starts, stops,
                                                                 strict=True))],
                        dtype=np.float64)
    # The number of points past either edge of `values` the stencils read:
    # the furthest position past an edge, plus the width of a stencil
    reach = max([int(np.ceil(max(-p.min(initial=0), p.max(initial=0) - n + 1))) + order
                 for p, n in zip(positions, shape, strict=True)] + [0])
    window = np.pad(window, reach, mode='reflect', reflect_type='odd')
    positions = [p - start + reach for p, start in zip(positions, starts, strict=True)]

    # Separable if each position varies along its own axis only
    separable = all(size == 1 for k, p in enumerate(positions)
                    for axis, size in enumerate(np.shape(p)) if axis != k)
    if separable:
        # One axis at a time, each position along the axis being a single vector
        for axis, p in enumerate(positions):
            window = _interpolate_axis(window, axis, p.ravel(), order)
        return window.reshape(np.broadcast_shapes(*(np.shape(p) for p in positions)))

    # Scattered positions: the tensor product of the weights along each axis
    positions = np.broadcast_arrays(*positions)
    stencils = [_lagrange(p, order) for p in positions]
    out = 0
    for taps in product(range(order), repeat=len(positions)):
        weight = np.prod([w[k] for k, (_, w) in zip(taps, stencils, strict=True)],
                         axis=0)
        index = tuple(first + k for k, (first, _) in zip(taps, stencils, strict=True))
        out = out + weight*window[index]
    return out


def _interpolate_axis(values, axis, positions, order):
    """`values` interpolated along `axis` at the fractional indices `positions`."""
    first, weights = _lagrange(positions, order)
    moved = np.moveaxis(values, axis, 0)
    out = sum(w[(slice(None),) + (None,)*(moved.ndim - 1)]*moved[first + k]
              for k, w in enumerate(weights))
    return np.moveaxis(out, 0, axis)


def _lagrange(positions, order):
    """
    The Lagrange stencils of `order` points around the fractional indices
    `positions`: the index of their first point, and their weights, one array
    per point.
    """
    first = np.floor(positions).astype(np.int64) - (order - 1)//2
    offset = positions - first
    nodes = range(order)
    weights = [np.prod([(offset - m)/(k - m) for m in nodes if m != k], axis=0)
               for k in nodes]
    return first, weights


def _difference(values, axis, spacing, order):
    """
    The centered first derivative of `order` of `values` along `axis`, of
    `order//2` fewer points at each end.
    """
    radius = order//2
    offsets = range(-radius, radius + 1)
    weights = sympy.finite_diff_weights(1, offsets, 0)[1][-1]
    size = values.shape[axis]

    out = 0
    for offset, weight in zip(offsets, weights, strict=True):
        index = [slice(None)]*values.ndim
        index[axis] = slice(radius + offset, size - radius + offset)
        out = out + float(weight)*values[tuple(index)]
    return out/spacing


def _crop(values, radius, axes):
    """`values` without `radius` points at each end of `axes`."""
    return values[tuple(slice(radius, values.shape[a] - radius) if a in axes
                        else slice(None) for a in range(values.ndim))]


class CurvilinearGrid(Grid):

    """
    A structured grid whose nodes a CurvilinearMap places in physical space.

    Its Dimensions, shape, spacing, extent and origin are those of the
    logically Cartesian computational grid: loops, halos, domain decomposition
    and SubDomains are those of a Grid. Derivatives of Functions on it, e.g.
    `u.dx` or `u.laplace`, are taken w.r.t. the physical coordinates by the
    chain rule, and `u.dx(computational=True)` w.r.t. the computational one.
    The transpose `.T` of a derivative, and the injection of a SparseFunction,
    whose coordinates are physical, are transposes in the inner product
    weighted by the Jacobian `J = det(d x/d xi)`, `sum(J*u*v)`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the computational domain in grid points.
    mapping : CurvilinearMap
        The map from computational to physical coordinates.
    extent : tuple of float, optional
        Extent of the computational domain, 1 in all Dimensions by default.
    origin : tuple of float, optional
        Computational coordinates of the origin, 0 in all Dimensions by default.
    **kwargs
        The other parameters of Grid.

    Notes
    -----
    The geometric terms at the nodes or at the half points of a staggering,
    e.g. `metrics_at`, are computed the first time they are used, which
    reduces over the ranks under MPI to tell whether they are constants: every
    rank must use them, as every rank does in building the same Operator.

    Examples
    --------
    >>> import numpy as np
    >>> from devito import CurvilinearGrid, TerrainFollowing
    >>> x = np.linspace(0., 500., 11)
    >>> grid = CurvilinearGrid(shape=(11, 11), extent=(500., 500.),
    ...                        mapping=TerrainFollowing(-20.*np.sin(x/100.)))
    >>> x, y = grid.dimensions
    >>> sorted(str(m) for m in grid.metrics.values() if not m.is_Function)
    ['0', '1']
    >>> grid.metrics[(x, y)].name.startswith('eta_x_g')
    True
    """

    is_Curvilinear = True

    def __init__(self, shape, mapping, extent=None, origin=None, **kwargs):
        if not isinstance(mapping, CurvilinearMap):
            raise TypeError(f"The mapping must be a CurvilinearMap, not {type(mapping)}")
        super().__init__(shape, extent=extent, origin=origin, **kwargs)
        self._mapping = mapping
        mapping.validate(self)

        # The suffix of the names of its geometric Functions, unique to the grid
        self._suffix = f'_g{_grid_number()}'

        # The metric terms and their derivatives, computed when first used, by
        # the Dimensions along which their points are staggered
        self._metric_cache = {}
        self._curvature_cache = {}

    @property
    def mapping(self):
        """The map from computational to physical coordinates."""
        return self._mapping

    @cached_property
    def coordinates(self):
        """The physical coordinates of the nodes, one Function per Dimension."""
        return tuple(self._function(f'{d.name}_coord', c, ())
                     for d, c in zip(self.dimensions, self.physical_points(()),
                                     strict=True))

    def physical_points(self, staggered=()):
        """
        The physical coordinates of this rank's points of the domain.

        Parameters
        ----------
        staggered : tuple of Dimension, optional
            The Dimensions along which the points are staggered by half a
            cell, none for the nodes.

        Returns
        -------
        tuple of ndarray
            One array of the local shape of the grid per Dimension.
        """
        mesh = np.meshgrid(*self._vectors(staggered, 0), indexing='ij', sparse=True)
        return np.broadcast_arrays(*self._mapping.forward(self, mesh))

    @property
    def metrics(self):
        """
        The metric terms `d xi_j/d x_i` at the nodes, keyed by the pair of
        Dimensions of the physical coordinate `x_i` and of the computational
        one `xi_j` (see `metrics_at`).
        """
        return self._metric_terms(()).metrics

    @property
    def jacobian(self):
        """The Jacobian `J = det(d x/d xi)` at the nodes."""
        return self._metric_terms(()).jacobian

    @property
    def inverse_jacobian(self):
        """The inverse `1/J` of the Jacobian at the nodes (see `inverse_jacobian_at`)."""
        return self._metric_terms(()).inverse_jacobian

    def metrics_at(self, location):
        """
        The metric terms `d xi_j/d x_i` at a node or a half point.

        A Dimension of the grid indexes both a physical coordinate and a
        computational one, so the term `d xi_j/d x_i` is keyed by the pair of
        Dimensions `(x_i, x_j)`, e.g. `(x, y)` for `d eta/d x` in 2D.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point, e.g. `{x: x + h_x/2}`, the node along the Dimensions it
            does not mention.

        Returns
        -------
        dict
            Each term a Function indexed at `location`, or a number where it is
            the same everywhere.
        """
        staggered, shifts = self._staggering(location)
        metrics = self._metric_terms(staggered).metrics
        return {k: _shifted(v, shifts) for k, v in metrics.items()}

    def jacobian_at(self, location):
        """
        The Jacobian `J = det(d x/d xi)` at a node or a half point.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        Function or number
            The Jacobian indexed at `location`, or a number if it is the same
            everywhere.
        """
        staggered, shifts = self._staggering(location)
        return _shifted(self._metric_terms(staggered).jacobian, shifts)

    def inverse_jacobian_at(self, location):
        """
        The inverse `1/J` of the Jacobian at a node or a half point.

        Like every geometric Function, it is 0 in the halo past the edges of
        the grid, so that a stencil may read it there, where dividing by `J`
        would divide by 0.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        Function or number
            The inverse Jacobian indexed at `location`, or a number if it is
            the same everywhere.
        """
        staggered, shifts = self._staggering(location)
        return _shifted(self._metric_terms(staggered).inverse_jacobian, shifts)

    def normal_ratios_at(self, location):
        """
        The ratios `n_i/n_z` of the components of the normal of the rows of the
        grid, along which the last computational coordinate `xi_z` is constant,
        at a node or a half point: `n = grad xi_z`, and `n_i/n_z` the ratio of
        `d xi_z/d x_i` over `d xi_z/d z`, for `z` the last Dimension.

        Like every geometric Function, they are 0 in the halo past the edges of
        the grid, so that a stencil may read them there, where the ratio of the
        metric terms would be `0/0`.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        dict
            The ratio for each Dimension but the last, a Function indexed at
            `location` or a number where it is the same everywhere.
        """
        staggered, shifts = self._staggering(location)
        ratios = self._metric_terms(staggered).normal_ratios
        return {k: _shifted(v, shifts) for k, v in ratios.items()}

    def curvatures_at(self, location):
        """
        The second derivatives `d^2 xi_j/(d x_i d x_l)` of the computational
        coordinates at a node or a half point.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        dict
            The terms keyed by the Dimensions `(x_i, x_l, x_j)`, in both orders
            of `x_i` and `x_l`, each a Function indexed at `location` or a
            number where it is the same everywhere.
        """
        staggered, shifts = self._staggering(location)
        curvatures = self._curvature_terms(staggered)
        return {k: _shifted(v, shifts) for k, v in curvatures.items()}

    @cached_property
    def min_spacing(self):
        """
        The smallest physical length of the cells, in the sense of a stability
        condition: `1/sqrt(l)` for `l` the largest eigenvalue of the metric
        tensor scaled by the spacing, `sum_i (d xi_j/d x_i)(d xi_k/d x_i)/(h_j h_k)`.
        """
        # Every rank takes part in computing the metric terms, which reduces
        # over the ranks, before the ranks without nodes are left out
        metrics = self.metrics
        minimum = np.inf
        if not self.distributor.loc_empty:
            # The metric terms scaled by the spacing, one matrix per node
            matrix = np.empty(self.shape_local + (self.dim, self.dim))
            for (physical, computational), term in metrics.items():
                i = self.dimensions.index(physical)
                j = self.dimensions.index(computational)
                values = np.asarray(term.data, dtype=np.float64) if term.is_Function \
                    else float(term)
                matrix[..., j, i] = values/float(self.spacing[j])

            tensor = matrix @ np.swapaxes(matrix, -1, -2)
            minimum = 1/np.sqrt(np.linalg.eigvalsh(tensor)[..., -1].max())

        return self.dtype(self._reduce(minimum, MPI.MIN))

    def to_computational(self, points):
        """
        The computational coordinates of physical points.

        Parameters
        ----------
        points : array_like
            The physical points, of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            The computational coordinates, of shape `(npoint, ndim)`.

        Raises
        ------
        ValueError
            If the map cannot place some point (see `CurvilinearMap.inverse`).
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        return self._mapping.inverse(self, points)

    def to_physical(self, points):
        """
        The physical coordinates of computational points.

        Parameters
        ----------
        points : array_like
            The computational points, of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            The physical coordinates, of shape `(npoint, ndim)`.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        return self._mapping.forward_points(self, points)

    def _metric_terms(self, staggered):
        """
        The `_MetricTerms` at the points staggered by half a cell along the
        Dimensions `staggered`.
        """
        staggered = tuple(d for d in self.dimensions if d in staggered)
        if staggered in self._metric_cache:
            return self._metric_cache[staggered]

        # The Jacobian matrix `d x_i/d xi_j` of the map, and its inverse
        matrix = self._map_derivatives(staggered, 0)
        jacobian = np.linalg.det(matrix)
        self._check_jacobian(jacobian)
        inverse = np.linalg.inv(matrix)

        scale = self._reduce(np.abs(inverse).max(initial=0), MPI.MAX)
        metrics = {}
        for i, j in product(range(self.dim), repeat=2):
            name = f'{_NAMES[self.dim][j]}_{self.dimensions[i].name}'
            metrics[(self.dimensions[i], self.dimensions[j])] = \
                self._term(name, inverse[..., j, i], scale, staggered)
        # The normal of the rows, the gradient of the last computational
        # coordinate, relative to its last component: ratios, whose roundoff
        # is relative to 1
        z = self.dim - 1
        normal_ratios = {self.dimensions[i]: self._term(
            f'n_{self.dimensions[i].name}', inverse[..., z, i]/inverse[..., z, z], 1,
            staggered) for i in range(z)}
        terms = _MetricTerms(metrics, self._term('jacobian', jacobian, 0, staggered),
                             self._term('inverse_jacobian', 1/jacobian, 0, staggered),
                             normal_ratios)

        self._metric_cache[staggered] = terms
        return terms

    def _curvature_terms(self, staggered):
        """
        The second derivatives `d^2 xi_j/(d x_i d x_l)`, keyed `(x_i, x_l, d_j)`
        in both orders, at the points staggered along `staggered`.
        """
        staggered = tuple(d for d in self.dimensions if d in staggered)
        if staggered in self._curvature_cache:
            return self._curvature_cache[staggered]

        # The Jacobian matrix `A` of the map, `radius` nodes past the points so
        # that it can be differentiated once more
        radius = self._mapping.order//2
        axes = set(range(self.dim))
        matrix = self._map_derivatives(staggered, radius)
        inverse = np.linalg.inv(_crop(matrix, radius, axes))

        # d M/d xi_k = -M (d A/d xi_k) M for M = A^-1, so that
        # d^2 xi_j/(d x_i d x_l) = sum_k M_kl (d M/d xi_k)_ji
        hessian = 0
        for k in range(self.dim):
            derivative = _crop(_difference(matrix, k, float(self.spacing[k]),
                                           self._mapping.order), radius, axes - {k})
            change = -np.einsum('...ja,...ab,...bi->...ji', inverse, derivative, inverse)
            hessian = hessian + inverse[..., k, None, None, :]*change[..., None]

        # Roundoff in second derivatives of the map is that of the metric terms
        # over a cell
        scale = max(self._reduce(np.abs(inverse).max(initial=0), MPI.MAX) /
                    min(float(h) for h in self.spacing),
                    self._reduce(np.abs(hessian).max(initial=0), MPI.MAX))
        curvatures = {}
        for i, l in combinations_with_replacement(range(self.dim), 2):
            for j in range(self.dim):
                pair = ''.join(self.dimensions[a].name for a in (i, l))
                name = f'{_NAMES[self.dim][j]}_{pair}'
                term = self._term(name, hessian[..., j, i, l], scale, staggered)
                xi, xl, dj = self.dimensions[i], self.dimensions[l], self.dimensions[j]
                curvatures[(xi, xl, dj)] = curvatures[(xl, xi, dj)] = term

        self._curvature_cache[staggered] = curvatures
        return curvatures

    def _map_derivatives(self, staggered, pad):
        """
        The Jacobian matrix of the map, `d x_i/d xi_j` at index `[..., i, j]`,
        at this rank's points staggered along `staggered` and `pad` more points
        at each end of every Dimension, by centered finite differences.
        """
        radius = self._mapping.order//2
        mesh = np.meshgrid(*self._vectors(staggered, pad + radius), indexing='ij',
                           sparse=True)
        physical = np.broadcast_arrays(*self._mapping.forward(self, mesh))

        axes = set(range(self.dim))
        columns = []
        for coordinate in physical:
            row = [_crop(_difference(coordinate, j, float(h), self._mapping.order),
                         radius, axes - {j})
                   for j, h in enumerate(self.spacing)]
            columns.append(np.stack(row, axis=-1))
        return np.stack(columns, axis=-2)

    def _vectors(self, staggered, pad):
        """
        The computational coordinates of this rank's nodes, staggered by half a
        cell along `staggered`, with `pad` more nodes at each end, one vector
        per Dimension; empty vectors on a rank without nodes.
        """
        vectors = []
        for d, numbering, o, h in zip(self.dimensions, self.distributor.glb_numb,
                                      self.origin, self.spacing, strict=True):
            if self.distributor.loc_empty:
                vectors.append(np.empty(0))
                continue
            index = np.arange(numbering[0] - pad, numbering[-1] + pad + 1,
                              dtype=np.float64)
            shift = .5 if d in staggered else 0.
            vectors.append(float(o) + float(h)*(index + shift))
        return vectors

    def _check_jacobian(self, jacobian):
        """
        Raise on every rank if the Jacobian is not finite and positive on some
        rank's nodes, a singular map or one turning cells inside out.
        """
        valid = bool(np.all(np.isfinite(jacobian)) and np.all(jacobian > 0))
        if not self._reduce(valid, MPI.LAND):
            raise ValueError(f"{type(self._mapping).__name__} has a Jacobian that is not "
                             "finite and positive on the grid: the map is singular or "
                             "turns cells inside out")

    def _term(self, name, values, scale, staggered):
        """
        A geometric term holding `values` at this rank's points.

        It is an exact number if it is the same on every rank, which roundoff
        is measured against `scale`, the size of the terms of its kind, so
        that e.g. the zero metric terms of a flat map are exactly 0. Otherwise
        it is a Function.
        """
        low = self._reduce(values.min(initial=np.inf), MPI.MIN)
        high = self._reduce(values.max(initial=-np.inf), MPI.MAX)
        tolerance = _CONSTANT_TOLERANCE*max(scale, abs(low), abs(high))
        if high - low <= tolerance:
            value = (low + high)/2
            nearest = round(value)
            if abs(value - nearest) <= tolerance:
                return sympy.Integer(nearest)
            return sympy.Float(value)

        return self._function(f'{name}{_suffix(staggered)}', values, staggered)

    def _function(self, name, values, staggered):
        """
        A Function staggered along `staggered` holding `values` on this rank's
        domain, with the halo of the order of the map, its name suffixed by
        that of the grid.
        """
        f = Function(name=f'{name}{self._suffix}', grid=self,
                     space_order=self._mapping.order,
                     staggered=staggered or None, dtype=self.dtype)
        f.data_local[:] = values
        return f

    def _staggering(self, location):
        """
        The Dimensions along which `location`, a node or a half point, is
        staggered, and the shift of `location` from the points of the
        Functions staggered there.
        """
        staggered, shifts = [], {}
        for d in self.dimensions:
            offset = sympy.nsimplify((location.get(d, d) - d)/d.spacing)
            if offset.q not in (1, 2):
                raise ValueError(f"The metric terms of {self} are at the nodes and the "
                                 f"half points, not at {location[d]}")
            if offset.q == 2:
                staggered.append(d)
                offset = offset - sympy.Rational(1, 2)
            if offset:
                shifts[d] = d + offset*d.spacing
        return tuple(staggered), shifts

    def _reduce(self, value, op):
        """`value` reduced over the ranks by `op`."""
        if self.distributor.is_parallel:
            return self.comm.allreduce(value, op=op)
        return value

    def __getstate__(self):
        state = super().__getstate__()
        # The geometric Functions are rebuilt from the map when needed
        for name in ('coordinates', 'min_spacing'):
            state.pop(name, None)
        state['_metric_cache'] = {}
        state['_curvature_cache'] = {}
        return state


def _shifted(term, shifts):
    """`term`, a Function or a number, indexed at `shifts`."""
    return term.subs(shifts) if shifts and isinstance(term, Function) else term


def _suffix(staggered):
    """The suffix of the names of the geometric Functions staggered along `staggered`."""
    return ''.join(f'_{d.name}' for d in staggered)
