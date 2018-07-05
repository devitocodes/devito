from mpi4py import MPI
import numpy as np

import pytest

from devito import Grid, Function
from devito.distributed import LEFT, RIGHT


@pytest.mark.parallel(nprocs=2)
def test_hello_mpi():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print("Hello, World! I am rank %d of %d on %s" % (rank, size, name), flush=True)


@pytest.mark.parallel(nprocs=[2, 4])
def test_basic_partitioning():
    grid = Grid(shape=(15, 15))
    f = Function(name='f', grid=grid)

    distributor = grid.distributor
    expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
        2: [(15, 8), (15, 7)],
        4: [(8, 8), (8, 7), (7, 8), (7, 7)]
    }
    assert f.shape == expected[distributor.nprocs][distributor.myrank]


@pytest.mark.parallel(nprocs=9)
def test_neighborhood_2d():
    grid = Grid(shape=(3, 3))
    x, y = grid.dimensions

    distributor = grid.distributor
    # Rank map:
    # ---------------y
    # | 0 | 1 | 2 |
    # -------------
    # | 3 | 4 | 5 |
    # -------------
    # | 6 | 7 | 8 |
    # -------------
    # |
    # x
    expected = {
        0: {x: {LEFT: None, RIGHT: 3}, y: {LEFT: None, RIGHT: 1}},
        1: {x: {LEFT: None, RIGHT: 4}, y: {LEFT: 0, RIGHT: 2}},
        2: {x: {LEFT: None, RIGHT: 5}, y: {LEFT: 1, RIGHT: None}},
        3: {x: {LEFT: 0, RIGHT: 6}, y: {LEFT: None, RIGHT: 4}},
        4: {x: {LEFT: 1, RIGHT: 7}, y: {LEFT: 3, RIGHT: 5}},
        5: {x: {LEFT: 2, RIGHT: 8}, y: {LEFT: 4, RIGHT: None}},
        6: {x: {LEFT: 3, RIGHT: None}, y: {LEFT: None, RIGHT: 7}},
        7: {x: {LEFT: 4, RIGHT: None}, y: {LEFT: 6, RIGHT: 8}},
        8: {x: {LEFT: 5, RIGHT: None}, y: {LEFT: 7, RIGHT: None}},
    }
    assert expected[distributor.myrank] == distributor.neighbours


@pytest.mark.parallel(nprocs=2)
def test_halo_exchange_bilateral():
    """
    Test halo exchange between two processes organised in a 1x2 cartesian grid.

    Their initial data_with_halo looks like:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 0 0 0 0 0     0 0 0 0 0 0

    After the halo exchange, the following is expected and tested for:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 0 0 0 0 0     0 0 0 0 0 0
    """
    grid = Grid(shape=(12, 12))
    f = Function(name='f', grid=grid)

    distributor = grid.distributor
    f.data[:] = distributor.myrank + 1

    # Now trigger a halo exchange...
    f.data_with_halo   # noqa

    if distributor.myrank == 0:
        assert np.all(f.data_ro_with_halo[1:-1, -1] == f.data_ro_domain[:, -1] + 1)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
    else:
        assert np.all(f.data_ro_with_halo[1:-1, 0] == f.data_ro_domain[:, 0] - 1)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
    assert np.all(f.data_ro_with_halo[0] == 0.)
    assert np.all(f.data_ro_with_halo[-1] == 0.)
