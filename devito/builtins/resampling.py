from itertools import product

import numpy as np

from devito.mpi import MPI

__all__ = ['resample']


def resample(function, cartesian, transpose=False):
    """
    Resample a Function between a Cartesian Grid and a CurvilinearGrid.

    With `transpose=False`, `function` is set to `cartesian` interpolated at the
    physical position of each node of its CurvilinearGrid. With `transpose=True`,
    `cartesian` is set to the exact transpose of that interpolation applied to
    `function`. The interpolation is multilinear in the Cartesian grid, which
    reduces to a 1D interpolation along each column when the lateral nodes of both
    grids coincide, as for a TerrainFollowing map. Nodes above or below the
    Cartesian domain take the value of its nearest node.

    Multilinear interpolation is second-order accurate for a smooth field, and
    keeps the resampled values within the bounds of the field it reads, as at
    the sharp interfaces of an earth model, where a higher-order polynomial
    interpolation would overshoot, e.g. into negative velocities.

    Under MPI, each rank holds the whole Cartesian field while resampling.

    Parameters
    ----------
    function : Function
        A Function on a CurvilinearGrid.
    cartesian : Function
        A Function on a Cartesian Grid of the same dimension, with any shape,
        extent, origin and domain decomposition.
    transpose : bool, optional, default=False
        Apply the transpose, from `function` to `cartesian`.

    Examples
    --------
    >>> import numpy as np
    >>> from devito import CurvilinearGrid, Function, Grid, TerrainFollowing, resample
    >>> mapping = TerrainFollowing(surface=np.linspace(0., 20., 11))
    >>> grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.), mapping=mapping)
    >>> cartesian = Function(name='c', grid=Grid(shape=(11, 21), extent=(100., 100.)))
    >>> cartesian.data[:] = 1.5
    >>> velocity = Function(name='v', grid=grid)
    >>> resample(velocity, cartesian)
    >>> float(velocity.data.min())
    1.5
    """
    grid = function.grid.root
    if not grid.is_Curvilinear:
        raise ValueError(f"{function} is not defined on a CurvilinearGrid")
    cartesian_grid = cartesian.grid.root
    axes = _resampling_axes(grid, cartesian_grid)
    shape = tuple(cartesian.shape_global)
    dtype = np.result_type(function.dtype, cartesian.dtype, np.float64)

    if not transpose:
        values = _global_values(cartesian, dtype).ravel()
        resampled = 0
        for index, weight in _corners(axes, shape):
            resampled = resampled + weight*values[index]
        if not grid.distributor.loc_empty:
            function.data_local[:] = resampled
        return

    values = np.asarray(function.data_local, dtype=dtype)
    transposed = np.zeros(int(np.prod(shape)), dtype=dtype)
    for index, weight in _corners(axes, shape):
        transposed += np.bincount(index.ravel(), weights=(weight*values).ravel(),
                                  minlength=transposed.size)
    transposed = transposed.reshape(shape)
    if grid.distributor.is_parallel:
        transposed = grid.comm.allreduce(transposed, op=MPI.SUM)
    cartesian.data[:] = transposed


def _resampling_axes(grid, cartesian_grid):
    """
    The cell of `cartesian_grid` holding each local node of the CurvilinearGrid
    `grid`, along each axis, as a list of `(first node index, fraction)` pairs,
    the fraction being None along an axis where every node is on a Cartesian
    node. Positions are clamped to the Cartesian domain, and snapped to a node
    within roundoff, so that a node shared by both grids is copied exactly.
    """
    if cartesian_grid.dim != grid.dim or cartesian_grid.is_Curvilinear:
        raise ValueError(f"Cannot resample from {cartesian_grid}, which is not a "
                         f"{grid.dim}D Cartesian Grid")
    axes = []
    for coordinate, origin, spacing, size in zip(
        grid.coordinates, cartesian_grid.origin, cartesian_grid.spacing,
        cartesian_grid.shape, strict=True
    ):
        values = np.asarray(coordinate.data_local, dtype=np.float64)
        position = (values - np.float64(origin))/np.float64(spacing)
        nearest = np.rint(position)
        position = np.clip(np.where(np.abs(position - nearest) < 1e-6, nearest,
                                    position), 0, size - 1)
        first = np.minimum(np.floor(position).astype(np.int64), max(size - 2, 0))
        fraction = position - first
        axes.append((first, fraction if np.any(fraction) else None))
    return axes


def _corners(axes, shape):
    """
    The corners of the multilinear interpolation cells of `axes` (see
    `_resampling_axes`) in a Cartesian grid of `shape`, one at a time, as
    `(flat node indices, weights)`. An axis without fraction has one corner.
    """
    choices = [(0, 1) if fraction is not None else (0,) for _, fraction in axes]
    for corner in product(*choices):
        weight = 1
        index = []
        for c, (first, fraction), size in zip(corner, axes, shape, strict=True):
            if fraction is not None:
                weight = weight*(fraction if c else 1 - fraction)
            index.append(np.minimum(first + c, size - 1))
        yield np.ravel_multi_index(np.broadcast_arrays(*index), shape), weight


def _global_values(function, dtype):
    """The global domain values of a possibly distributed `function`."""
    values = np.zeros(function.shape_global, dtype=dtype)
    if not function.grid.distributor.loc_empty:
        values[function.local_indices] = function.data_local
    if function.grid.distributor.is_parallel:
        values = function.grid.comm.allreduce(values, op=MPI.SUM)
    return values
