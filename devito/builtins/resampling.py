"""
Resampling of Functions from a Cartesian Grid onto a CurvilinearGrid.
"""
from itertools import product

import numpy as np

import devito as dv
from devito.mpi import MPI

__all__ = ['resample']


def resample(function, cartesian):
    """
    Set a Function on a CurvilinearGrid to a Cartesian Function interpolated at
    the physical position of each of its points, the nodes of its
    CurvilinearGrid or its staggered points.

    Points outside the Cartesian domain take the value of its nearest point, so
    that a model need not cover e.g. the mirror rows of a free surface.

    The interpolation is multilinear: second-order accurate for a smooth
    field, and within the bounds of the field it reads, as at the sharp
    interfaces of an earth model, where a higher-order interpolation would
    overshoot, e.g. into negative velocities.

    Parameters
    ----------
    function : Function
        A Function on a CurvilinearGrid, not on one of its SubDomains.
    cartesian : Function
        A Function on a Cartesian Grid of the same dimension, with any shape,
        extent, origin and staggering, distributed over the ranks of the
        CurvilinearGrid or not at all.

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

    # The Cartesian cell of each point of this rank: along each Dimension, its
    # first point and the fraction of the cell past it
    firsts, fractions = _cells(function, cartesian)

    # The Cartesian points, split over the ranks of the CurvilinearGrid
    mapped_grid = dv.types.grid.MappedGrid(grid, cartesian.grid)

    # The cells, as Functions at the points of `function`, in the indices of
    # the Cartesian points of this rank
    cell_firsts, cell_fractions = [], []
    for d, first, fraction, offset in zip(grid.dimensions, firsts, fractions,
                                          mapped_grid.origin_ioffset, strict=True):
        cell_first = dv.Function(name=f'{function.name}_first_{d.name}', grid=grid,
                                 staggered=function.staggered, space_order=0,
                                 dtype=np.int32)
        cell_first.data_local[:] = first - offset
        cell_fraction = dv.Function(name=f'{function.name}_fraction_{d.name}',
                                    grid=grid, staggered=function.staggered,
                                    space_order=0, dtype=function.dtype)
        cell_fraction.data_local[:] = fraction
        cell_firsts.append(cell_first)
        cell_fractions.append(cell_fraction)

    # The Cartesian values, with a halo as wide as the cells reach past the
    # points of a rank, which the Operator exchanges
    mapped = dv.Function(name=f'{cartesian.name}_mapped', grid=mapped_grid,
                         space_order=_reach(function, cell_firsts, mapped_grid),
                         dtype=cartesian.dtype)
    mapped.data[:] = cartesian.data

    # The value at each point is the sum of the corners of its cell, weighted
    # along each Dimension by one minus the fraction for the first point and by
    # the fraction for the next one
    value = 0
    for sides in product((0, 1), repeat=grid.dim):
        weight = 1
        for fraction, side in zip(cell_fractions, sides, strict=True):
            weight *= fraction if side else 1 - fraction
        value += weight*mapped[tuple(first + side for first, side
                                     in zip(cell_firsts, sides, strict=True))]
    dv.Operator(dv.Eq(function, value), name='Resample').apply()


def _check_grids(function, cartesian):
    """Raise a ValueError unless `function` can be resampled from `cartesian`."""
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
    The cell of the points of `cartesian` around each point of `function` on
    this rank: along each Dimension, the global index of its first point and
    the fraction of the cell past it, of the local shape of `function`.
    """
    grid = function.grid
    cartesian_grid = cartesian.grid
    staggered = tuple(d for d in grid.dimensions if function.indices_ref[d] != d)

    firsts, fractions = [], []
    for d, physical, origin, spacing, size in zip(
            cartesian_grid.dimensions, grid.physical_points(staggered),
            cartesian_grid.origin, cartesian_grid.spacing, cartesian_grid.shape,
            strict=True):
        # The position in units of the Cartesian spacing, clamped to the Cartesian
        # domain. A staggered Cartesian Function has its points half a cell past
        # the nodes
        shift = 0.5 if cartesian.indices_ref[d] != d else 0.
        position = (physical - float(origin))/float(spacing) - shift
        position = np.clip(position, 0, size - 1)

        # The last cell holds the last point
        first = np.minimum(np.floor(position).astype(np.int64), size - 2)
        firsts.append(first)
        fractions.append(position - first)
    return firsts, fractions


def _reach(function, cell_firsts, mapped_grid):
    """
    The number of points by which the cells of `cell_firsts`, the Functions of
    their first points, reach past the points of `mapped_grid` a rank holds,
    along any Dimension and on any rank.
    """
    # The reach of the cell of each point, past the first or the last point of
    # its rank, along the Dimension it reaches farthest
    reach = dv.Function(name=f'{function.name}_reach', grid=function.grid,
                        staggered=function.staggered, space_order=0, dtype=np.int32)
    for cell_first, size in zip(cell_firsts, mapped_grid.shape_local, strict=True):
        first = cell_first.data_local
        reach.data_local[:] = np.maximum(reach.data_local,
                                         np.maximum(-first, first + 2 - size))
    return dv.mmax(reach)
