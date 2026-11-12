"""
Curvilinear grids: grids whose nodes are moved vertically by a map.

A CurvilinearGrid works like a Grid for loops, halos, domain decomposition and
SubDomains. Its derivatives are taken w.r.t. the physical coordinates, using the
metric terms of the map (see `devito.finite_differences.curvilinear`).
"""
from collections import namedtuple
from functools import cached_property
from itertools import combinations_with_replacement, product

import numpy as np
import sympy

from devito.mpi import MPI
from devito.tools import generator
from devito.types.dense import Function
from devito.types.dimension import DerivedDimension
from devito.types.grid import Grid

__all__ = ['CurvilinearGrid', 'CurvilinearMap', 'MeshMap', 'TerrainFollowing']

# Names of the computational coordinates, by number of Dimensions
_NAMES = {1: ('xi',), 2: ('xi', 'eta'), 3: ('xi', 'eta', 'zeta')}

# A geometric term whose relative spread is below this is a constant
_CONSTANT_TOLERANCE = 1e-10

# A counter that gives the geometric Functions of each grid unique names
_grid_number = generator()

# The geometric terms at the points of one staggering
_MetricTerms = namedtuple('MetricTerms',
                          ('metrics', 'jacobian', 'inverse_jacobian', 'normal_ratios'))


class CurvilinearMap:

    """
    A map from computational to physical coordinates.

    The map moves each node of a grid vertically, along the last Dimension. The
    lateral coordinates stay the same. A subclass implements `forward`, and may
    override `validate`.

    Parameters
    ----------
    order : int, optional, default=8
        The order of the finite differences and interpolation of the map, and
        the halo of the geometric Functions of the grid.
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
        Check that the map can place the nodes of `grid`.

        Raises
        ------
        ValueError
            If the map cannot place the nodes.
        """

    def forward(self, grid, computational):
        """
        Return the depth of computational points.

        Parameters
        ----------
        grid : CurvilinearGrid
            The grid the map places.
        computational : tuple of ndarray
            The computational coordinates, one array per Dimension of `grid`.
            The arrays must broadcast together.

        Returns
        -------
        ndarray
            The depth, which broadcasts to the shape of the computational
            coordinates.
        """
        raise NotImplementedError


class TerrainFollowing(CurvilinearMap):

    """
    A map whose top row follows a surface, e.g. topography.

    The surface is the row at computational depth 0. In each column, the nodes
    are spread evenly between the surface and the flat bottom of the grid:

        z(x, eta) = s(x) + eta * (b - s(x)) / b

    where `s` is the depth of the surface and `b` is the computational depth of
    the last row. A negative vertical origin adds rows above the surface.

    Parameters
    ----------
    surface : array_like or callable
        The depth of the surface, positive down: an array with one value per
        lateral grid node, or a function of the lateral coordinates.
    order : int, optional, default=8
        See `CurvilinearMap`.

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
        """The depth of the surface: an array of node values, or a callable."""
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
        return surface + eta*(bottom - surface)/bottom

    def _surface_at(self, grid, lateral):
        """Return the depth of the surface at the lateral coordinates `lateral`."""
        if callable(self._surface):
            shape = np.broadcast_shapes(*(np.shape(v) for v in lateral))
            return np.broadcast_to(np.asarray(self._surface(*lateral), dtype=np.float64),
                                   shape)

        positions = [(v - float(o))/float(h)
                     for v, o, h in zip(lateral, grid.origin[:-1], grid.spacing[:-1],
                                        strict=True)]
        return _interpolate(self._surface, positions, self.order)


class MeshMap(CurvilinearMap):

    """
    A map given by the depth of every node of a mesh.

    Depths between nodes are interpolated. Past the edges of the mesh, the depth
    is extended by odd reflection. The rows of the mesh may bend, e.g. to follow
    a seabed, as long as the depth increases down each column.

    Parameters
    ----------
    depth : array_like
        The depth of each node, positive down, with the shape of the mesh. It is
        only read in slices, so it can be a memory map.
    padding : tuple of int, optional
        The number of grid nodes before the mesh along each Dimension, e.g. for
        absorbing layers. Defaults to 0.
    order : int, optional, default=8
        See `CurvilinearMap`.

    Examples
    --------
    >>> import numpy as np
    >>> from devito import CurvilinearGrid, MeshMap
    >>> x, eta = np.meshgrid(np.linspace(0., 100., 11), np.linspace(0., 100., 11),
    ...                      indexing='ij')
    >>> grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.),
    ...                        mapping=MeshMap(eta + .1*x))
    >>> float(grid.coordinates[-1].data[-1, 0])
    10.0
    """

    def __init__(self, depth, padding=None, order=8):
        super().__init__(order)
        self._depth = depth
        self._padding = tuple(padding or (0,)*np.ndim(depth))

    @property
    def depth(self):
        """The depth of the nodes of the mesh."""
        return self._depth

    @property
    def padding(self):
        """The number of grid nodes before the mesh along each Dimension."""
        return self._padding

    @property
    def shape(self):
        """The shape of the mesh."""
        return np.shape(self._depth)

    def validate(self, grid):
        if len(self.shape) != grid.dim or len(self._padding) != grid.dim:
            raise ValueError(f"The mesh has {len(self.shape)} Dimensions and padding "
                             f"{self._padding}, the grid {grid.dim} Dimensions")
        if any(p + n > s for p, n, s in zip(self._padding, self.shape, grid.shape,
                                            strict=True)):
            raise ValueError(f"The mesh of shape {self.shape}, after the padding "
                             f"{self._padding}, does not fit the grid {grid.shape}")

    def forward(self, grid, computational):
        positions = [(np.asarray(c, dtype=np.float64) - float(o))/float(h) - p
                     for c, o, h, p in zip(computational, grid.origin, grid.spacing,
                                           self._padding, strict=True)]
        return _interpolate(self._depth, positions, self.order)


def _bottom(grid):
    """Return the computational depth of the last row of `grid`."""
    return float(grid.origin[-1]) + float(grid.extent[-1])


def _interpolate(values, positions, order):
    """
    Interpolate tabulated values at fractional indices, with Lagrange
    polynomials.

    Past the edges of `values`, the values are extended by odd reflection,
    `v(-i) = 2 v(0) - v(i)`, which keeps linear trends. Only the part of
    `values` that the stencils read is loaded.

    Parameters
    ----------
    values : array_like
        The tabulated values, read by slices only.
    positions : list of ndarray
        The fractional indices along each axis of `values`. They must broadcast
        together.
    order : int
        The number of Lagrange points along each axis.

    Returns
    -------
    ndarray
        The interpolated values, of the broadcast shape of `positions`.
    """
    positions = [np.asarray(p, dtype=np.float64) for p in positions]
    shape = np.shape(values)

    # Load only the part of `values` that the stencils read, and pad it by odd
    # reflection as far as they reach past its edges
    starts = [max(0, int(np.floor(p.min(initial=0))) - order) for p in positions]
    stops = [min(n, int(np.ceil(p.max(initial=0))) + order + 1)
             for p, n in zip(positions, shape, strict=True)]
    window = np.asarray(values[tuple(slice(a, b) for a, b in zip(starts, stops,
                                                                 strict=True))],
                        dtype=np.float64)
    # How far the stencils reach past the edges of `values`
    reach = max([int(np.ceil(max(-p.min(initial=0), p.max(initial=0) - n + 1))) + order
                 for p, n in zip(positions, shape, strict=True)] + [0])
    window = np.pad(window, reach, mode='reflect', reflect_type='odd')
    positions = [p - start + reach for p, start in zip(positions, starts, strict=True)]

    # The interpolation is separable if each position varies along its own axis only
    separable = all(size == 1 for k, p in enumerate(positions)
                    for axis, size in enumerate(np.shape(p)) if axis != k)
    if separable:
        # Interpolate one axis at a time
        for axis, p in enumerate(positions):
            window = _interpolate_axis(window, axis, p.ravel(), order)
        return window.reshape(np.broadcast_shapes(*(np.shape(p) for p in positions)))

    # Scattered positions: sum the tensor product of the weights along each axis
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
    """Interpolate `values` along `axis` at the fractional indices `positions`."""
    first, weights = _lagrange(positions, order)
    moved = np.moveaxis(values, axis, 0)
    out = sum(w[(slice(None),) + (None,)*(moved.ndim - 1)]*moved[first + k]
              for k, w in enumerate(weights))
    return np.moveaxis(out, 0, axis)


def _lagrange(positions, order):
    """
    Return the Lagrange stencils of `order` points around `positions`.

    The result is the index of the first point of each stencil, and the weights,
    one array per point.
    """
    first = np.floor(positions).astype(np.int64) - (order - 1)//2
    offset = positions - first
    nodes = range(order)
    weights = [np.prod([(offset - m)/(k - m) for m in nodes if m != k], axis=0)
               for k in nodes]
    return first, weights


def _difference(values, axis, spacing, order):
    """
    Return the centered first derivative of `values` along `axis`.

    The stencil has order `order`, so the result is `order//2` points shorter at
    each end.
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
    """Remove `radius` points at each end of the axes `axes` of `values`."""
    return values[tuple(slice(radius, values.shape[a] - radius) if a in axes
                        else slice(None) for a in range(values.ndim))]


class CurvilinearGrid(Grid):

    """
    A grid whose nodes are moved vertically by a CurvilinearMap.

    The shape, spacing, extent and origin are those of the computational grid.
    Loops, halos, domain decomposition and SubDomains work as on a Grid.

    Derivatives along the Dimensions, e.g. `u.dx` or `u.laplace`, are w.r.t. the
    physical coordinates. Derivatives along the `computational_dimensions`, e.g.
    `u.diff(xi)`, are w.r.t. the computational coordinates. Transposes, `.T` and
    the injection of a SparseFunction, are adjoints in the inner product
    `sum(J*u*v)`, where `J` is the Jacobian of the map.

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
    The geometric terms are computed on first use with collective MPI calls,
    so every rank must use them, as it does when building the same Operator.

    Examples
    --------
    >>> import numpy as np
    >>> from devito import CurvilinearGrid, TerrainFollowing
    >>> x = np.linspace(0., 500., 11)
    >>> grid = CurvilinearGrid(shape=(11, 11), extent=(500., 500.),
    ...                        mapping=TerrainFollowing(-20.*np.sin(x/100.)))
    >>> x, z = grid.dimensions
    >>> xi, eta = grid.computational_dimensions
    >>> sorted(str(m) for m in grid.metrics.values() if not m.is_Function)
    ['0', '1']
    >>> grid.metrics[(x, eta)].name.startswith('eta_x_g')
    True
    """

    is_Curvilinear = True

    def __init__(self, shape, mapping, extent=None, origin=None, **kwargs):
        if not isinstance(mapping, CurvilinearMap):
            raise TypeError(f"The mapping must be a CurvilinearMap, not {type(mapping)}")
        super().__init__(shape, extent=extent, origin=origin, **kwargs)
        if self.dim not in _NAMES:
            raise ValueError(f"A CurvilinearGrid has 1 to 3 Dimensions, not {self.dim}")
        self._mapping = mapping
        mapping.validate(self)

        self._computational_dimensions = tuple(
            DerivedDimension(name, d)
            for name, d in zip(_NAMES[self.dim], self.dimensions, strict=True)
        )

        # A suffix for the names of the geometric Functions of this grid
        self._suffix = f'_g{_grid_number()}'

        # The geometric terms of each staggering, computed on first use
        self._metric_cache = {}
        self._curvature_cache = {}

    @property
    def computational_dimensions(self):
        """
        The computational coordinates, one per Dimension, e.g. `xi` for `x`.

        A derivative along one of them is the derivative of a Grid.
        """
        return self._computational_dimensions

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
        Return the physical coordinates of the points of this rank.

        Parameters
        ----------
        staggered : tuple of Dimension, optional
            The Dimensions along which the points are half a cell off the nodes.
            Empty for the nodes.

        Returns
        -------
        tuple of ndarray
            One array per Dimension, with the local shape of the grid.
        """
        mesh = np.meshgrid(*self._vectors(staggered, 0), indexing='ij', sparse=True)
        return self._physical(mesh)

    @property
    def metrics(self):
        """
        The metric terms `d xi_j/d x_i` at the nodes.

        They are keyed by `(x_i, xi_j)`. For example, `(x, eta)` is `d eta/d x`.
        """
        return self._metric_terms(()).metrics

    @property
    def jacobian(self):
        """The Jacobian `J = det(d x/d xi)` at the nodes."""
        return self._metric_terms(()).jacobian

    @property
    def inverse_jacobian(self):
        """The inverse Jacobian `1/J` at the nodes."""
        return self._metric_terms(()).inverse_jacobian

    def metrics_at(self, location):
        """
        Return the metric terms `d xi_j/d x_i` at a node or a half point.

        They are keyed by `(x_i, xi_j)`, as in `metrics`.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point, e.g. `{x: x + h_x/2}`. Along the Dimensions it leaves
            out, the point is at the node.

        Returns
        -------
        dict
            Each term is a Function indexed at `location`, or a number if it is
            constant.
        """
        staggered, shifts = self._staggering(location)
        metrics = self._metric_terms(staggered).metrics
        return {k: _shifted(v, shifts) for k, v in metrics.items()}

    def jacobian_at(self, location):
        """
        Return the Jacobian `J = det(d x/d xi)` at a node or a half point.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        Function or number
            The Jacobian indexed at `location`, or a number if it is constant.
        """
        staggered, shifts = self._staggering(location)
        return _shifted(self._metric_terms(staggered).jacobian, shifts)

    def inverse_jacobian_at(self, location):
        """
        Return the inverse Jacobian `1/J` at a node or a half point.

        It is 0 in the halo, so a stencil can read it there safely.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        Function or number
            The inverse Jacobian indexed at `location`, or a number if it is
            constant.
        """
        staggered, shifts = self._staggering(location)
        return _shifted(self._metric_terms(staggered).inverse_jacobian, shifts)

    def normal_ratios_at(self, location):
        """
        Return the ratios `n_i/n_z` of the normal to the grid rows.

        The normal `n` is the gradient of the last computational coordinate.
        There is one ratio for each lateral Dimension. The ratios are 0 in the
        halo, so a stencil can read them there safely.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        dict
            One ratio per lateral Dimension: a Function indexed at `location`,
            or a number if it is constant.
        """
        staggered, shifts = self._staggering(location)
        ratios = self._metric_terms(staggered).normal_ratios
        return {k: _shifted(v, shifts) for k, v in ratios.items()}

    def curvatures_at(self, location):
        """
        Return the second derivatives `d^2 xi_j/(d x_i d x_l)` at a node or a
        half point.

        Parameters
        ----------
        location : dict of {Dimension: expr}
            The point (see `metrics_at`).

        Returns
        -------
        dict
            The terms keyed by `(x_i, x_l, xi_j)`, with both orders of `x_i` and
            `x_l`. Each is a Function indexed at `location`, or a number if it
            is constant.
        """
        staggered, shifts = self._staggering(location)
        curvatures = self._curvature_terms(staggered)
        return {k: _shifted(v, shifts) for k, v in curvatures.items()}

    @cached_property
    def min_spacing(self):
        """
        The smallest physical cell size, used for the stability condition.

        It is `1/sqrt(l)`, where `l` is the largest eigenvalue of the metric
        tensor scaled by the grid spacing.
        """
        # Computing the metric terms is collective, so every rank does it
        metrics = self.metrics
        minimum = np.inf
        if not self.distributor.loc_empty:
            # The metric terms scaled by the spacing, one matrix per node
            matrix = np.empty(self.shape_local + (self.dim, self.dim))
            for (physical, computational), term in metrics.items():
                i = self.dimensions.index(physical)
                j = self.computational_dimensions.index(computational)
                values = np.asarray(term.data, dtype=np.float64) if term.is_Function \
                    else float(term)
                matrix[..., j, i] = values/float(self.spacing[j])

            tensor = matrix @ np.swapaxes(matrix, -1, -2)
            minimum = 1/np.sqrt(np.linalg.eigvalsh(tensor)[..., -1].max())

        return self.dtype(self._reduce(minimum, MPI.MIN))

    def to_computational(self, points):
        """
        Return the computational coordinates of physical points.

        The lateral coordinates do not change. The depth is found along the
        column of each point, by linear interpolation between the two nodes
        around it. Points above the first row or below the last row are
        extrapolated.

        Parameters
        ----------
        points : array_like
            The physical points, of shape `(npoint, ndim)`.

        Returns
        -------
        ndarray
            The computational coordinates, of shape `(npoint, ndim)`.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        *lateral, depth = points.T
        h = float(self.spacing[-1])
        rows = float(self.origin[-1]) + h*np.arange(self.shape[-1])

        # The depth of the nodes in the column of each point
        column = (*[v[:, None] for v in lateral], rows[None])
        nodes = np.broadcast_to(self._mapping.forward(self, column),
                                (len(points), len(rows)))

        # The row just above each point, or the first or last row if the point is
        # outside the grid
        above = np.clip((nodes <= depth[:, None]).sum(axis=1) - 1, 0, len(rows) - 2)
        top, bottom = (np.take_along_axis(nodes, (above + k)[:, None], axis=1)[:, 0]
                       for k in (0, 1))
        eta = rows[above] + h*(depth - top)/(bottom - top)
        return np.stack([*lateral, eta], axis=1)

    def to_physical(self, points):
        """
        Return the physical coordinates of computational points.

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
        return np.stack(self._physical(tuple(points.T)), axis=1)

    def _physical(self, computational):
        """
        Return the physical coordinates of computational coordinates: the
        lateral coordinates unchanged, and the depth given by the map.
        """
        *lateral, _ = computational
        return np.broadcast_arrays(*lateral, self._mapping.forward(self, computational))

    def _metric_terms(self, staggered):
        """Return the `_MetricTerms` at the points staggered along `staggered`."""
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
            physical, computational = self.dimensions[i], self.computational_dimensions[j]
            metrics[(physical, computational)] = self._term(
                f'{computational.name}_{physical.name}', inverse[..., j, i], scale,
                staggered)
        # The normal ratios. Their roundoff is relative to 1
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
        Return the second derivatives `d^2 xi_j/(d x_i d x_l)` at the points
        staggered along `staggered`.
        """
        staggered = tuple(d for d in self.dimensions if d in staggered)
        if staggered in self._curvature_cache:
            return self._curvature_cache[staggered]

        # The Jacobian matrix `A` of the map, with `radius` extra points so that it
        # can be differentiated again
        radius = self._mapping.order//2
        axes = set(range(self.dim))
        matrix = self._map_derivatives(staggered, radius)
        inverse = np.linalg.inv(_crop(matrix, radius, axes))

        # With `M = A^-1`: `d M/d xi_k = -M (d A/d xi_k) M`, so
        # `d^2 xi_j/(d x_i d x_l) = sum_k M_kl (d M/d xi_k)_ji`
        hessian = 0
        for k in range(self.dim):
            derivative = _crop(_difference(matrix, k, float(self.spacing[k]),
                                           self._mapping.order), radius, axes - {k})
            change = -np.einsum('...ja,...ab,...bi->...ji', inverse, derivative, inverse)
            hessian = hessian + inverse[..., k, None, None, :]*change[..., None]

        # The roundoff of the metric terms over one cell
        scale = max(self._reduce(np.abs(inverse).max(initial=0), MPI.MAX) /
                    min(float(h) for h in self.spacing),
                    self._reduce(np.abs(hessian).max(initial=0), MPI.MAX))
        curvatures = {}
        for i, l in combinations_with_replacement(range(self.dim), 2):
            for j in range(self.dim):
                xi, xl = self.dimensions[i], self.dimensions[l]
                computational = self.computational_dimensions[j]
                name = f'{computational.name}_{xi.name}{xl.name}'
                term = self._term(name, hessian[..., j, i, l], scale, staggered)
                curvatures[(xi, xl, computational)] = term
                curvatures[(xl, xi, computational)] = term

        self._curvature_cache[staggered] = curvatures
        return curvatures

    def _map_derivatives(self, staggered, pad):
        """
        Return the Jacobian matrix `d x_i/d xi_j` of the map, by finite
        differences.

        It is computed at the points of this rank staggered along `staggered`,
        plus `pad` points at each end. Entry `[..., i, j]` is `d x_i/d xi_j`.
        """
        radius = self._mapping.order//2
        mesh = np.meshgrid(*self._vectors(staggered, pad + radius), indexing='ij',
                           sparse=True)
        physical = self._physical(mesh)

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
        Return the computational coordinates of the nodes of this rank, one
        vector per Dimension.

        The nodes are staggered by half a cell along `staggered`, with `pad`
        extra nodes at each end. A rank without nodes gets empty vectors.
        """
        if self.distributor.loc_empty:
            return [np.empty(0)]*self.dim

        vectors = []
        for d, numbering, o, h in zip(self.dimensions, self.distributor.glb_numb,
                                      self.origin, self.spacing, strict=True):
            index = np.arange(numbering[0] - pad, numbering[-1] + pad + 1,
                              dtype=np.float64)
            shift = .5 if d in staggered else 0.
            vectors.append(float(o) + float(h)*(index + shift))
        return vectors

    def _check_jacobian(self, jacobian):
        """
        Raise on every rank if the Jacobian is not finite and positive
        somewhere.

        That happens when the map is singular or turns cells inside out.
        """
        valid = bool(np.all(np.isfinite(jacobian)) and np.all(jacobian > 0))
        if not self._reduce(valid, MPI.LAND):
            raise ValueError(f"{type(self._mapping).__name__} has a Jacobian that is not "
                             "finite and positive on the grid: the map is singular or "
                             "turns cells inside out")

    def _term(self, name, values, scale, staggered):
        """
        Return a Function holding `values`, or a number if `values` is constant.

        `values` is constant if it varies by less than the roundoff relative to
        `scale` across all ranks. This makes, e.g., the zero metric terms of a
        flat map exactly 0.
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

        return self._function(f'{name}{_staggering_suffix(staggered)}', values, staggered)

    def _function(self, name, values, staggered):
        """
        Return a Function holding `values`, staggered along `staggered`.

        Its halo is the order of the map. Its name ends with the suffix of the
        grid.
        """
        f = Function(name=f'{name}{self._suffix}', grid=self,
                     space_order=self._mapping.order,
                     staggered=staggered or None, dtype=self.dtype)
        f.data_local[:] = values
        return f

    def _staggering(self, location):
        """
        Return the Dimensions along which `location` is staggered, and the shift
        from `location` to the points of a Function staggered that way.
        """
        staggered, shifts = [], {}
        for d in self.dimensions:
            offset = sympy.nsimplify((location.get(d, d) - d)/d.spacing)
            if offset.q not in (1, 2):
                raise ValueError("The geometric terms are only defined at the nodes and "
                                 f"the half points, not at {location[d]}")
            if offset.q == 2:
                staggered.append(d)
                offset = offset - sympy.Rational(1, 2)
            if offset:
                shifts[d] = d + offset*d.spacing
        return tuple(staggered), shifts

    def _reduce(self, value, op):
        """Reduce `value` over the ranks with the MPI operation `op`."""
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
    """Return `term` indexed at `shifts`, if it is a Function."""
    return term.subs(shifts) if shifts and isinstance(term, Function) else term


def _staggering_suffix(staggered):
    """Return the name suffix of geometric Functions staggered along `staggered`."""
    return ''.join(f'_{d.name}' for d in staggered)
