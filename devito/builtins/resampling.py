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
    corners = _resampling_weights(grid, cartesian.grid.root)
    dtype = np.result_type(function.dtype, cartesian.dtype, np.float64)

    if not transpose:
        values = _global_values(cartesian, dtype)
        resampled = sum(w*values[index] for index, w in corners)
        if not grid.distributor.loc_empty:
            function.data_local[:] = resampled
        return

    values = np.asarray(function.data_local, dtype=dtype)
    transposed = np.zeros(cartesian.shape_global, dtype=dtype)
    for index, weight in corners:
        np.add.at(transposed, index, weight*values)
    if grid.distributor.is_parallel:
        transposed = grid.comm.allreduce(transposed, op=MPI.SUM)
    cartesian.data[:] = transposed


def _resampling_weights(grid, cartesian_grid):
    """
    Multilinear interpolation weights from the nodes of `cartesian_grid` to the
    local nodes of the CurvilinearGrid `grid`, as a list of `(cartesian node
    indices, weights)` pairs, one per corner of the interpolation cell. Positions
    are clamped to the Cartesian domain, and snapped to a node within roundoff, so
    that a node shared by both grids is copied exactly.
    """
    if cartesian_grid.dim != grid.dim or cartesian_grid.is_Curvilinear:
        raise ValueError(f"Cannot resample from {cartesian_grid}, which is not a "
                         f"{grid.dim}D Cartesian Grid")
    positions = []
    for coordinate, origin, spacing, size in zip(
        grid.coordinates, cartesian_grid.origin, cartesian_grid.spacing,
        cartesian_grid.shape, strict=True
    ):
        values = np.asarray(coordinate.data_local, dtype=np.float64)
        position = (values - np.float64(origin))/np.float64(spacing)
        nearest = np.rint(position)
        position = np.where(np.abs(position - nearest) < 1e-6, nearest, position)
        positions.append(np.clip(position, 0, size - 1))

    base = [np.minimum(np.floor(p).astype(np.int64), max(size - 2, 0))
            for p, size in zip(positions, cartesian_grid.shape, strict=True)]
    fractions = [p - b for p, b in zip(positions, base, strict=True)]
    corners = []
    for corner in np.ndindex(*(2,)*grid.dim):
        weight = 1
        index = []
        for c, b, f, size in zip(corner, base, fractions, cartesian_grid.shape,
                                 strict=True):
            weight = weight * (f if c else 1 - f)
            index.append(np.minimum(b + c, size - 1))
        corners.append((tuple(index), weight))
    return corners


def _global_values(function, dtype):
    """The global domain values of a possibly distributed `function`."""
    values = np.zeros(function.shape_global, dtype=dtype)
    if not function.grid.distributor.loc_empty:
        values[function.local_indices] = function.data_local
    if function.grid.distributor.is_parallel:
        values = function.grid.comm.allreduce(values, op=MPI.SUM)
    return values
