"""Resample a Function from a Cartesian Grid onto a CurvilinearGrid."""
from itertools import product

import numpy as np

import devito as dv
from devito.mpi import MPI

__all__ = ['resample']


def resample(function, cartesian):
    """
    Set a Function on a CurvilinearGrid from a Function on a Cartesian Grid.

    Each point of `function` gets the value of `cartesian` at its physical
    position, by multilinear interpolation. Points outside the Cartesian grid
    get the value of the nearest Cartesian point. Multilinear interpolation
    never goes outside the range of `cartesian`, so there is no overshoot at
    sharp interfaces.

    Parameters
    ----------
    function : Function
        A Function on a CurvilinearGrid. It cannot be on a SubDomain.
    cartesian : Function
        A Function on a Cartesian Grid with the same number of Dimensions. It is
        either distributed over the same ranks as the CurvilinearGrid, or not
        distributed.

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
    >>> bool(np.allclose(velocity.data, 1.5))
    True
    """
    _check_grids(function, cartesian)
    grid = function.grid

    firsts, fractions = _cells(function, cartesian)

    # The Cartesian grid, split over the ranks of the CurvilinearGrid
    mapped_grid = dv.types.grid.MappedGrid(grid, cartesian.grid)

    # The index and the interpolation weight of the cell of each point, in the local
    # indices of this rank
    firsts = [first - offset for first, offset in zip(firsts, mapped_grid.origin_ioffset,
                                                      strict=True)]
    cell_firsts, cell_fractions = [], []
    for d, first, fraction in zip(grid.dimensions, firsts, fractions, strict=True):
        cell_first = dv.Function(name=f'{function.name}_first_{d.name}', grid=grid,
                                 staggered=function.staggered, space_order=0,
                                 dtype=np.int32)
        cell_first.data_local[:] = first
        cell_fraction = dv.Function(name=f'{function.name}_fraction_{d.name}',
                                    grid=grid, staggered=function.staggered,
                                    space_order=0, dtype=function.dtype)
        cell_fraction.data_local[:] = fraction
        cell_firsts.append(cell_first)
        cell_fractions.append(cell_fraction)

    # The Cartesian values, with a halo wide enough for every cell
    mapped = dv.Function(name=f'{cartesian.name}_mapped', grid=mapped_grid,
                         space_order=_reach(firsts, mapped_grid),
                         dtype=cartesian.dtype)
    mapped.data[:] = cartesian.data

    # Multilinear interpolation from the corners of each cell
    value = 0
    for sides in product((0, 1), repeat=grid.dim):
        weight = 1
        for fraction, side in zip(cell_fractions, sides, strict=True):
            weight *= fraction if side else 1 - fraction
        value += weight*mapped[tuple(first + side for first, side
                                     in zip(cell_firsts, sides, strict=True))]
    dv.Operator(dv.Eq(function, value), name='Resample').apply()


def _check_grids(function, cartesian):
    """Raise a ValueError if `function` cannot be resampled from `cartesian`."""
    grid = function.grid
    if not grid.root.is_Curvilinear:
        raise ValueError(f"{function} is not defined on a CurvilinearGrid")
    if grid is not grid.root:
        raise ValueError(f"{function} is defined on a SubDomain: resample a Function "
                         "on the whole CurvilinearGrid")

    cartesian_grid = cartesian.grid
    if not cartesian_grid.is_Grid or cartesian_grid.is_Curvilinear or \
            cartesian_grid.dim != grid.dim:
        raise ValueError(f"Cannot resample from {cartesian_grid}, which is not a "
                         f"{grid.dim}D Cartesian Grid")

    if cartesian_grid.distributor.is_parallel and \
            MPI.Comm.Compare(grid.comm, cartesian_grid.comm) not in (MPI.IDENT,
                                                                     MPI.CONGRUENT):
        raise ValueError("The Cartesian Grid is distributed over other ranks than "
                         "the CurvilinearGrid")


def _cells(function, cartesian):
    """
    Return the Cartesian cell of each local point of `function`.

    For each Dimension, return the global index of the Cartesian point just
    below each point, and the fractional distance from it.
    """
    grid = function.grid
    cartesian_grid = cartesian.grid
    staggered = tuple(d for d in grid.dimensions if function.indices_ref[d] != d)

    firsts, fractions = [], []
    for d, physical, origin, spacing, size in zip(
            cartesian_grid.dimensions, grid.physical_points(staggered),
            cartesian_grid.origin, cartesian_grid.spacing, cartesian_grid.shape,
            strict=True):
        # The position in Cartesian cells, clamped to the Cartesian grid
        shift = 0.5 if cartesian.indices_ref[d] != d else 0.
        position = (physical - float(origin))/float(spacing) - shift
        position = np.clip(position, 0, size - 1)

        # The last cell contains the last point
        first = np.minimum(np.floor(position).astype(np.int64), size - 2)
        firsts.append(first)
        fractions.append(position - first)
    return firsts, fractions


def _reach(firsts, mapped_grid):
    """
    Return the halo width that `mapped_grid` needs for every rank to read the cells
    of `firsts`, in its local indices.

    A halo exchange only reaches the neighboring ranks, so a ValueError is raised if
    a rank reads past them.
    """
    reach = max([0] + [int(np.max(np.maximum(-first, first + 2 - size), initial=0))
                       for first, size in zip(firsts, mapped_grid.shape_local,
                                              strict=True)])
    smallest = min(mapped_grid.shape_local)
    if mapped_grid.distributor.is_parallel:
        reach = mapped_grid.comm.allreduce(reach, op=MPI.MAX)
        smallest = mapped_grid.comm.allreduce(smallest, op=MPI.MIN)
        if reach > smallest:
            raise ValueError(f"The points of a rank are {reach} Cartesian points away "
                             f"from its own, past the {smallest} of its neighbors: use "
                             "fewer ranks or a finer Cartesian Grid")
    return reach
