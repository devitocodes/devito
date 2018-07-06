import numpy as np

import pytest
from conftest import skipif_yask

from devito import Grid, Function, TimeFunction, Eq, Operator
from devito.ir.iet import copy, halo_exchange
from devito.distributed import LEFT, RIGHT
from devito.types import Array


@skipif_yask
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


@skipif_yask
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


@skipif_yask
@pytest.mark.parallel(nprocs=2)
def test_halo_exchange_bilateral():
    """
    Test halo exchange between two processes organised in a 1x2 cartesian grid.

    The initial ``data_with_halo`` looks like:

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
        assert np.all(f.data_ro_with_halo[1:-1, -1] == 2.)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
    else:
        assert np.all(f.data_ro_with_halo[1:-1, 0] == 1.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
    assert np.all(f.data_ro_with_halo[0] == 0.)
    assert np.all(f.data_ro_with_halo[-1] == 0.)


@skipif_yask
@pytest.mark.parallel(nprocs=2)
def test_halo_exchange_bilateral_asymmetric():
    """
    Test halo exchange between two processes organised in a 1x2 cartesian grid.

    In this test, the size of left and right halo regions are different.

    The initial ``data_with_halo`` looks like:

           rank0           rank1
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0

    After the halo exchange, the following is expected and tested for:

           rank0           rank1
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
    """
    grid = Grid(shape=(12, 12))
    f = Function(name='f', grid=grid, space_order=(1, 2, 1))

    distributor = grid.distributor
    f.data[:] = distributor.myrank + 1

    # Now trigger a halo exchange...
    f.data_with_halo   # noqa

    if distributor.myrank == 0:
        assert np.all(f.data_ro_with_halo[2:-1, -1] == 2.)
        assert np.all(f.data_ro_with_halo[:, 0:2] == 0.)
    else:
        assert np.all(f.data_ro_with_halo[2:-1, 0:2] == 1.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
    assert np.all(f.data_ro_with_halo[0:2] == 0.)
    assert np.all(f.data_ro_with_halo[-1] == 0.)


@skipif_yask
@pytest.mark.parallel(nprocs=4)
def test_halo_exchange_quadrilateral():
    """
    Test halo exchange between four processes organised in a 2x2 cartesian grid.

    The initial ``data_with_halo`` looks like:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 0 0 0 0 0     0 0 0 0 0 0

           rank2           rank3
        0 0 0 0 0 0     0 0 0 0 0 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 0 0 0 0 0     0 0 0 0 0 0

    After the halo exchange, the following is expected and tested for:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 3 3 3 3 4     3 4 4 4 4 0

           rank2           rank3
        0 1 1 1 1 2     1 2 2 2 2 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 0 0 0 0 0     0 0 0 0 0 0
    """
    grid = Grid(shape=(12, 12))
    f = Function(name='f', grid=grid)

    distributor = grid.distributor
    f.data[:] = distributor.myrank + 1

    # Now trigger a halo exchange...
    f.data_with_halo   # noqa

    if distributor.myrank == 0:
        assert np.all(f.data_ro_with_halo[0] == 0.)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, -1] == 2.)
        assert np.all(f.data_ro_with_halo[-1, 1:-1] == 3.)
        assert f.data_ro_with_halo[-1, -1] == 4.
    elif distributor.myrank == 1:
        assert np.all(f.data_ro_with_halo[0] == 0.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, 0] == 1.)
        assert np.all(f.data_ro_with_halo[-1, 1:-1] == 4.)
        assert f.data_ro_with_halo[-1, 0] == 3.
    elif distributor.myrank == 2:
        assert np.all(f.data_ro_with_halo[-1] == 0.)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, -1] == 4.)
        assert np.all(f.data_ro_with_halo[0, 1:-1] == 1.)
        assert f.data_ro_with_halo[0, -1] == 2.
    else:
        assert np.all(f.data_ro_with_halo[-1] == 0.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, 0] == 3.)
        assert np.all(f.data_ro_with_halo[0, 1:-1] == 2.)
        assert f.data_ro_with_halo[0, 0] == 1.


class TestCodeGeneration(object):

    def test_iet_copy(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = copy(f, {t: 0})
        assert str(iet.parameters) ==\
'(dst(dst_t, dst_x, dst_y), dst_t_size, dst_x_size,\
 dst_y_size, f(t, x, y), ox, oy, x_size, y_size)'
        assert """for (int x = 0, dst_x = 0; x <= dst_x_size; x += 1, dst_x = dst_x + 1)
  {
    for (int y = 0, dst_y = 0; y <= dst_y_size; y += 1, dst_y = dst_y + 1)
    {
      dst[0][dst_x][dst_y] = f[0][x + ox][y + oy];
    }
  }
}""" in str(iet)

    def test_iet_halo_exchange(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = halo_exchange(f, {t: 0})
        from IPython import embed; embed()


if __name__ == "__main__":
    test_halo_exchange_bilateral_asymmetric()
