import numpy as np
import pytest

from conftest import skipif
from devito import (Grid, Constant, Function, TimeFunction, SparseFunction,
                    SparseTimeFunction, Dimension, ConditionalDimension,
                    SubDimension, Eq, Inc, Operator, norm, inner)
from devito.data import LEFT, RIGHT
from devito.ir.iet import Call, Conditional, Iteration, FindNodes
from devito.mpi import MPI, HaloExchangeBuilder, HaloSchemeEntry
from examples.seismic.acoustic import acoustic_setup

pytestmark = skipif(['yask', 'ops', 'nompi'])


class TestDistributor(object):

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning(self):
        grid = Grid(shape=(15, 15))
        f = Function(name='f', grid=grid)

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(15, 8), (15, 7)],
            4: [(8, 8), (8, 7), (7, 8), (7, 7)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning_fewer_dims(self):
        """Test domain decomposition for Functions defined over a strict subset
        of grid-decomposed dimensions."""
        size_x, size_y = 16, 16
        grid = Grid(shape=(size_x, size_y))
        x, y = grid.dimensions

        # A function with fewer dimensions that in `grid`
        f = Function(name='f', grid=grid, dimensions=(y,), shape=(size_y,))

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8,), (8,)],
            4: [(8,), (8,), (8,), (8,)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(mode=9)
    def test_neighborhood_horizontal_2d(self):
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
        PN = MPI.PROC_NULL
        expected = {
            0: {x: {LEFT: PN, RIGHT: 3}, y: {LEFT: PN, RIGHT: 1}},
            1: {x: {LEFT: PN, RIGHT: 4}, y: {LEFT: 0, RIGHT: 2}},
            2: {x: {LEFT: PN, RIGHT: 5}, y: {LEFT: 1, RIGHT: PN}},
            3: {x: {LEFT: 0, RIGHT: 6}, y: {LEFT: PN, RIGHT: 4}},
            4: {x: {LEFT: 1, RIGHT: 7}, y: {LEFT: 3, RIGHT: 5}},
            5: {x: {LEFT: 2, RIGHT: 8}, y: {LEFT: 4, RIGHT: PN}},
            6: {x: {LEFT: 3, RIGHT: PN}, y: {LEFT: PN, RIGHT: 7}},
            7: {x: {LEFT: 4, RIGHT: PN}, y: {LEFT: 6, RIGHT: 8}},
            8: {x: {LEFT: 5, RIGHT: PN}, y: {LEFT: 7, RIGHT: PN}},
        }
        assert expected[distributor.myrank][x] == distributor.neighborhood[x]
        assert expected[distributor.myrank][y] == distributor.neighborhood[y]

    @pytest.mark.parallel(mode=9)
    def test_neighborhood_diagonal_2d(self):
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
        PN = MPI.PROC_NULL
        expected = {
            0: {(LEFT, LEFT): PN, (LEFT, RIGHT): PN, (RIGHT, LEFT): PN, (RIGHT, RIGHT): 4},  # noqa
            1: {(LEFT, LEFT): PN, (LEFT, RIGHT): PN, (RIGHT, LEFT): 3, (RIGHT, RIGHT): 5},
            2: {(LEFT, LEFT): PN, (LEFT, RIGHT): PN, (RIGHT, LEFT): 4, (RIGHT, RIGHT): PN},  # noqa
            3: {(LEFT, LEFT): PN, (LEFT, RIGHT): 1, (RIGHT, LEFT): PN, (RIGHT, RIGHT): 7},
            4: {(LEFT, LEFT): 0, (LEFT, RIGHT): 2, (RIGHT, LEFT): 6, (RIGHT, RIGHT): 8},
            5: {(LEFT, LEFT): 1, (LEFT, RIGHT): PN, (RIGHT, LEFT): 7, (RIGHT, RIGHT): PN},
            6: {(LEFT, LEFT): PN, (LEFT, RIGHT): 4, (RIGHT, LEFT): PN, (RIGHT, RIGHT): PN},  # noqa
            7: {(LEFT, LEFT): 3, (LEFT, RIGHT): 5, (RIGHT, LEFT): PN, (RIGHT, RIGHT): PN},
            8: {(LEFT, LEFT): 4, (LEFT, RIGHT): PN, (RIGHT, LEFT): PN, (RIGHT, RIGHT): PN}  # noqa
        }
        assert all(expected[distributor.myrank][i] == distributor.neighborhood[i]
                   for i in [(LEFT, LEFT), (LEFT, RIGHT), (RIGHT, LEFT), (RIGHT, RIGHT)])

    @pytest.mark.parallel(mode=[2, 4])
    def test_ctypes_neighborhood(self):
        grid = Grid(shape=(4, 4))
        distributor = grid.distributor

        PN = MPI.PROC_NULL
        attrs = ['ll', 'lc', 'lr', 'cl', 'cc', 'cr', 'rl', 'rc', 'rr']
        expected = {  # nprocs -> [(rank0 xleft xright ...), (rank1 xleft ...), ...]
            2: [(PN, PN, PN, PN, 0, 1, PN, PN, PN),
                (PN, PN, PN, 0, 1, PN, PN, PN, PN)],
            4: [(PN, PN, PN, PN, 0, 1, PN, 2, 3),
                (PN, PN, PN, 0, 1, PN, 2, 3, PN),
                (PN, 0, 1, PN, 2, 3, PN, PN, PN),
                (0, 1, PN, 2, 3, PN, PN, PN, PN)]
        }

        mapper = dict(zip(attrs, expected[distributor.nprocs][distributor.myrank]))
        obj = distributor._obj_neighborhood
        assert all(getattr(obj.value._obj, k) == v for k, v in mapper.items())


class TestFunction(object):

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_bilateral(self):
        """
        Test halo exchange between two processes organised in a 1x2 cartesian grid.

        The initial ``data_with_inhalo`` looks like:

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
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[1:-1, -1] == 2.)
            assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
        else:
            assert np.all(f._data_ro_with_inhalo[1:-1, 0] == 1.)
            assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)
        assert np.all(f._data_ro_with_inhalo[0] == 0.)
        assert np.all(f._data_ro_with_inhalo[-1] == 0.)

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_bilateral_asymmetric(self):
        """
        Test halo exchange between two processes organised in a 1x2 cartesian grid.

        In this test, the size of left and right halo regions are different.

        The initial ``data_with_inhalo`` looks like:

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
        x, y = grid.dimensions

        f = Function(name='f', grid=grid, space_order=(1, 2, 1))
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[2:-1, -1] == 2.)
            assert np.all(f._data_ro_with_inhalo[:, 0:2] == 0.)
        else:
            assert np.all(f._data_ro_with_inhalo[2:-1, 0:2] == 1.)
            assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)
        assert np.all(f._data_ro_with_inhalo[0:2] == 0.)
        assert np.all(f._data_ro_with_inhalo[-1] == 0.)

    @pytest.mark.parallel(mode=4)
    def test_halo_exchange_quadrilateral(self):
        """
        Test halo exchange between four processes organised in a 2x2 cartesian grid.

        The initial ``data_with_inhalo`` looks like:

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
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[0] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, -1] == 2.)
            assert np.all(f._data_ro_with_inhalo[-1, 1:-1] == 3.)
            assert f._data_ro_with_inhalo[-1, -1] == 4.
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[0] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, 0] == 1.)
            assert np.all(f._data_ro_with_inhalo[-1, 1:-1] == 4.)
            assert f._data_ro_with_inhalo[-1, 0] == 3.
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[-1] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, -1] == 4.)
            assert np.all(f._data_ro_with_inhalo[0, 1:-1] == 1.)
            assert f._data_ro_with_inhalo[0, -1] == 2.
        else:
            assert np.all(f._data_ro_with_inhalo[-1] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, 0] == 3.)
            assert np.all(f._data_ro_with_inhalo[0, 1:-1] == 2.)
            assert f._data_ro_with_inhalo[0, 0] == 1.

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('shape,expected', [
        ((15, 15), [((0, 8), (0, 8)), ((0, 8), (8, 15)),
                    ((8, 15), (0, 8)), ((8, 15), (8, 15))]),
    ])
    def test_local_indices(self, shape, expected):
        grid = Grid(shape=shape)
        f = Function(name='f', grid=grid)

        assert all(i == slice(*j)
                   for i, j in zip(f.local_indices, expected[grid.distributor.myrank]))


class TestCodeGeneration(object):

    @pytest.mark.parallel(mode=1)
    def test_iet_copy(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        heb = HaloExchangeBuilder()
        gather = heb._make_copy(f, HaloSchemeEntry([t], []))
        assert str(gather.parameters) == """\
(buf(buf_x, buf_y), buf_x_size, buf_y_size, f(t, x, y), otime, ox, oy)"""
        assert """\
  for (int x = 0; x <= buf_x_size - 1; x += 1)
  {
    for (int y = 0; y <= buf_y_size - 1; y += 1)
    {
      buf[x][y] = f[otime][x + ox][y + oy];
    }
  }""" in str(gather)

    @pytest.mark.parallel(mode=1)
    def test_iet_basic_sendrecv(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        heb = HaloExchangeBuilder()
        sendrecv = heb._make_sendrecv(f, HaloSchemeEntry([t], []))
        assert str(sendrecv.parameters) == """\
(f(t, x, y), buf_x_size, buf_y_size, ogtime, ogx, ogy, ostime, osx, osy,\
 fromrank, torank, comm)"""
        assert str(sendrecv.body[0]) == """\
float (*bufs)[buf_y_size];
float (*bufg)[buf_y_size];
posix_memalign((void**)&bufs, 64, sizeof(float[buf_x_size][buf_y_size]));
posix_memalign((void**)&bufg, 64, sizeof(float[buf_x_size][buf_y_size]));
MPI_Request rrecv;
MPI_Request rsend;
MPI_Irecv((float *)bufs,buf_x_size*buf_y_size,MPI_FLOAT,fromrank,13,comm,&rrecv);
if (torank != MPI_PROC_NULL)
{
  gather3d((float *)bufg,buf_x_size,buf_y_size,f_vec,ogtime,ogx,ogy);
}
MPI_Isend((float *)bufg,buf_x_size*buf_y_size,MPI_FLOAT,torank,13,comm,&rsend);
MPI_Wait(&rsend,MPI_STATUS_IGNORE);
MPI_Wait(&rrecv,MPI_STATUS_IGNORE);
if (fromrank != MPI_PROC_NULL)
{
  scatter3d((float *)bufs,buf_x_size,buf_y_size,f_vec,ostime,osx,osy);
}
free(bufs);
free(bufg);"""

    @pytest.mark.parallel(mode=1)
    def test_iet_basic_haloupdate(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        heb = HaloExchangeBuilder()
        halos = [(x, LEFT), (x, RIGHT), (y, LEFT), (y, RIGHT)]
        haloupdate = heb._make_haloupdate(f, HaloSchemeEntry([t], halos))
        assert str(haloupdate.parameters) == """\
(f(t, x, y), comm, nb, otime)"""
        assert str(haloupdate.body[0]) == """\
sendrecv3d(f_vec,f_vec->hsize[3],f_vec->npsize[2],otime,f_vec->oofs[2],\
f_vec->hofs[4],otime,f_vec->hofs[3],f_vec->hofs[4],nb->rc,nb->lc,comm);
sendrecv3d(f_vec,f_vec->hsize[2],f_vec->npsize[2],otime,f_vec->oofs[3],\
f_vec->hofs[4],otime,f_vec->hofs[2],f_vec->hofs[4],nb->lc,nb->rc,comm);
sendrecv3d(f_vec,f_vec->npsize[1],f_vec->hsize[5],otime,f_vec->hofs[2],\
f_vec->oofs[4],otime,f_vec->hofs[2],f_vec->hofs[5],nb->cr,nb->cl,comm);
sendrecv3d(f_vec,f_vec->npsize[1],f_vec->hsize[4],otime,f_vec->hofs[2],\
f_vec->oofs[5],otime,f_vec->hofs[2],f_vec->hofs[4],nb->cl,nb->cr,comm);"""


class TestSparseFunction(object):

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('coords', [
        ((1., 1.), (1., 3.), (3., 1.), (3., 3.)),
    ])
    def test_ownership(self, coords):
        """Given a sparse point ``p`` with known coordinates, this test checks
        that the MPI rank owning ``p`` is retrieved correctly."""
        grid = Grid(shape=(4, 4), extent=(4.0, 4.0))

        sf = SparseFunction(name='sf', grid=grid, npoint=4, coordinates=coords)

        # The domain decomposition is so that the i-th MPI rank gets exactly one
        # sparse point `p` and, incidentally, `p` is logically owned by `i`
        assert len(sf.gridpoints) == 1
        assert all(grid.distributor.glb_to_rank(i) == grid.distributor.myrank
                   for i in sf.gridpoints)

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('coords,expected', [
        ([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)], [[0.], [1.], [2.], [3.]]),
    ])
    def test_local_indices(self, coords, expected):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        data = np.array([0., 1., 2., 3.])
        coords = np.array(coords)
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords))

        # Each of the 4 MPI ranks get one (randomly chosen) sparse point
        assert sf.npoint == 1

        sf.coordinates.data[:] = coords
        sf.data[:] = data

        expected = np.array(expected[grid.distributor.myrank])
        assert np.all(sf.data == expected)

    @pytest.mark.parallel(mode=4)
    def test_scatter_gather(self):
        """
        Test scattering and gathering of sparse data from and to a single MPI rank.

        The initial data distribution (A, B, C, and D are generic values) looks like:

               rank0           rank1           rank2           rank3
                [D]             [C]             [B]             [A]

        Logically (i.e., given point coordinates and domain decomposition), A belongs
        to rank0, B belongs to rank1, etc. Thus, after scattering, the data distribution
        is expected to be:

               rank0           rank1           rank2           rank3
                [A]             [B]             [C]             [D]

        Then, locally on each rank, a trivial *2 multiplication is performed:

               rank0           rank1           rank2           rank3
               [A*2]           [B*2]           [C*2]           [D*2]

        Finally, we gather the data values and we get:

               rank0           rank1           rank2           rank3
               [D*2]           [C*2]           [B*2]           [A*2]
        """
        grid = Grid(shape=(4, 4), extent=(4.0, 4.0))

        # Initialization
        data = np.array([3, 2, 1, 0])
        coords = np.array([(3., 3.), (3., 1.), (1., 3.), (1., 1.)])
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = data

        # Scatter
        loc_data = sf._dist_scatter()[sf]
        assert len(loc_data) == 1
        assert loc_data[0] == grid.distributor.myrank

        # Do some local computation
        loc_data = loc_data*2

        # Gather
        sf._dist_gather(loc_data)
        assert len(sf.data) == 1
        assert np.all(sf.data == data[sf.local_indices]*2)


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[2, 4, 8, 16, 32])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
        op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 3.)
        if f.grid.distributor.myrank == 0:
            assert f.data_ro_domain[0, 0] == 5.
            assert np.all(f.data_ro_domain[0, 1:] == 7.)
        elif f.grid.distributor.myrank == f.grid.distributor.nprocs - 1:
            assert f.data_ro_domain[0, -1] == 5.
            assert np.all(f.data_ro_domain[0, :-1] == 7.)
        else:
            assert np.all(f.data_ro_domain[0] == 7.)

    @pytest.mark.parallel(mode=2)
    def test_trivial_eq_1d_save(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=5)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[time, x-1] + f[time, x+1] + 1))
        op.apply()

        time_M = op._prepare_arguments()['time_M']

        assert np.all(f.data_ro_domain[1] == 3.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[-1, time_M:] == 31.)
        else:
            assert np.all(f.data_ro_domain[-1, :-time_M] == 31.)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap')])
    def test_trivial_eq_2d(self):
        grid = Grid(shape=(8, 8,))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid, space_order=1)
        f.data_with_halo[:] = 1.

        eqn = Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + f[t, x, y-1] + f[t, x, y+1])
        op = Operator(eqn)
        op.apply(time=1)

        # Expected computed values
        corner, side, interior = 10., 13., 16.

        glb_pos_map = f.grid.distributor.glb_pos_map
        assert np.all(f.data_ro_domain[0, 1:-1, 1:-1] == interior)
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert f.data_ro_domain[0, 0, 0] == corner
            assert np.all(f.data_ro_domain[0, 1:, :1] == side)
            assert np.all(f.data_ro_domain[0, :1, 1:] == side)
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert f.data_ro_domain[0, 0, -1] == corner
            assert np.all(f.data_ro_domain[0, :1, :-1] == side)
            assert np.all(f.data_ro_domain[0, 1:, -1:] == side)
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert f.data_ro_domain[0, -1, 0] == corner
            assert np.all(f.data_ro_domain[0, -1:, 1:] == side)
            assert np.all(f.data_ro_domain[0, :-1, :1] == side)
        else:
            assert f.data_ro_domain[0, -1, -1] == corner
            assert np.all(f.data_ro_domain[0, :-1, -1:] == side)
            assert np.all(f.data_ro_domain[0, -1:, :-1] == side)

    @pytest.mark.parallel(mode=[(8, 'basic'), (8, 'diag'), (8, 'overlap')])
    def test_trivial_eq_3d(self):
        grid = Grid(shape=(8, 8, 8))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid, space_order=1)
        f.data_with_halo[:] = 1.

        eqn = Eq(f.forward, (f[t, x-1, y, z] + f[t, x+1, y, z] +
                             f[t, x, y-1, z] + f[t, x, y+1, z] +
                             f[t, x, y, z-1] + f[t, x, y, z+1]))
        op = Operator(eqn)
        op.apply(time=1)

        # Expected computed values
        corner, side, face, interior = 21., 26., 31., 36.

        # Situation at t=0
        assert np.all(f.data_ro_domain[1] == 6.)
        # Situation at t=1
        # 1) corners
        for i in [[0, 0, 0], [0, 0, -1], [0, -1, 0], [0, -1, -1],
                  [-1, 0, 0], [-1, 0, -1], [-1, -1, 0], [-1, -1, -1]]:
            assert f.data_ro_domain[[0] + i] in [corner, None]
        # 2) sides
        for i in [[0, 0, slice(1, -1)], [0, -1, slice(1, -1)],
                  [0, slice(1, -1), 0], [0, slice(1, -1), -1],
                  [-1, 0, slice(1, -1)], [-1, -1, slice(1, -1)],
                  [-1, slice(1, -1), 0], [-1, slice(1, -1), -1],
                  [slice(1, -1), 0, 0], [slice(1, -1), 0, -1],
                  [slice(1, -1), -1, 0], [slice(1, -1), -1, -1]]:
            assert np.all(f.data_ro_domain[[0] + i] == side)
        # 3) faces
        for i in [[0, slice(1, -1), slice(1, -1)], [-1, slice(1, -1), slice(1, -1)],
                  [slice(1, -1), 0, slice(1, -1)], [slice(1, -1), -1, slice(1, -1)],
                  [slice(1, -1), slice(1, -1), 0], [slice(1, -1), slice(1, -1), -1]]:
            assert np.all(f.data_ro_domain[[0] + i] == face)
        # 4) interior
        assert np.all(f.data_ro_domain[0, 1:-1, 1:-1, 1:-1] == interior)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap')])
    def test_multiple_eqs_funcs(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 0.
        g = TimeFunction(name='g', grid=grid)
        g.data_with_halo[:] = 0.

        op = Operator([Eq(f.forward, f[t, x+1] + g[t, x-1] + 1),
                       Eq(g.forward, f[t, x-1] + g[t, x+1] + 1)])
        op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 1.)
        if f.grid.distributor.myrank == 0:
            assert f.data_ro_domain[0, 0] == 2.
            assert np.all(f.data_ro_domain[0, 1:] == 3.)
        elif f.grid.distributor.myrank == f.grid.distributor.nprocs - 1:
            assert f.data_ro_domain[0, -1] == 2.
            assert np.all(f.data_ro_domain[0, :-1] == 3.)
        else:
            assert np.all(f.data_ro_domain[0] == 3.)

        # Also check that there are no redundant halo exchanges. Here, only
        # two are expected before the `x` Iteration, one for `f` and one for `g`
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

    def test_nostencil_implies_nohaloupdate(self):
        grid = Grid(shape=(12,))

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(f.forward, f + 1.),
                       Eq(g, f + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_avoid_redundant_haloupdate(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        i = Dimension(name='i')
        j = Dimension(name='j')

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(f[t+1, i], 1.),  # no halo update as it's an Inc
                       Eq(g, f[t, j] + 1)])  # access `f` at `t`, not `t+1`!

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_if_distr_but_sequential(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        # There is an anti-dependence between the first and second Eqs, so
        # the compiler places them in different x-loops. However, none of the
        # two loops should be preceded by a halo exchange, though for different
        # reasons:
        # * the equation in the first loop has no stencil
        # * the equation in the second loop is inherently sequential, so the
        #   compiler should be sufficiently smart to see that there is no point
        #   in adding a halo exchange
        op = Operator([Eq(f, f + 1),
                       Eq(f, f[t, x-1] + f[t, x+1] + 1.)])

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == 3
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_stencil_nowrite_implies_haloupdate_anyway(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        # It does a halo update, even though there's no data dependence,
        # because when the halo updates are placed, the compiler conservatively
        # assumes there might have been another equation writing to `f` before.
        op = Operator(Eq(g, f[t, x-1] + f[t, x+1] + 1.))

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=[(2, 'basic'), (2, 'diag'), (2, 'overlap')])
    def test_redo_haloupdate_due_to_antidep(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Eq(g.forward, f[t+1, x-1] + f[t+1, x+1] + g)])
        op.apply(time=0)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        assert np.all(f.data_ro_domain[1] == 1.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(g.data_ro_domain[1, 1:] == 2.)
        else:
            assert np.all(g.data_ro_domain[1, :-1] == 2.)

    def test_haloupdate_not_requried(self):
        grid = Grid(shape=(4, 4))
        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=2, save=None)
        v = TimeFunction(name='v', grid=grid, space_order=0, time_order=0, save=5)
        g = Function(name='g', grid=grid, space_order=0)
        i = Function(name='i', grid=grid, space_order=0)

        shift = Constant(name='shift', dtype=np.int32)

        step = Eq(u.forward, u - u.backward + 1)
        g_inc = Inc(g, u * v.subs(grid.time_dim, grid.time_dim - shift))
        i_inc = Inc(i, (v*v).subs(grid.time_dim, grid.time_dim - shift))

        op = Operator([step, g_inc, i_inc])

        # No stencil in the expressions, so no halo update required!
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=2)
    def test_reapply_with_different_functions(self):
        grid1 = Grid(shape=(30, 30, 30))
        f1 = Function(name='f', grid=grid1, space_order=4)

        op = Operator(Eq(f1, 1.))
        op.apply()

        grid2 = Grid(shape=(40, 40, 40))
        f2 = Function(name='f', grid=grid2, space_order=4)

        # Re-application
        op.apply(f=f2)

        assert np.all(f1.data == 1.)
        assert np.all(f2.data == 1.)

    @pytest.mark.parametrize('expr,expected', [
        ('f[t,x-1,y] + f[t,x+1,y]', {'rc', 'lc'}),
        ('f[t,x,y-1] + f[t,x,y+1]', {'cr', 'cl'}),
        ('f[t,x-1,y-1] + f[t,x,y+1]', {'cl', 'll', 'lc', 'cr'}),
        ('f[t,x-1,y-1] + f[t,x+1,y+1]', {'cl', 'll', 'lc', 'cr', 'rr', 'rc'}),
    ])
    @pytest.mark.parallel(mode=[(1, 'diag')])
    def test_diag_comm_scheme(self, expr, expected):
        """
        Check that the 'diag' mode does not generate more communications
        than strictly necessary.
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa
        t = grid.stepping_dim  # noqa

        f = TimeFunction(name='f', grid=grid)  # noqa

        op = Operator(Eq(f.forward, eval(expr)), dle=('advanced', {'openmp': False}))

        calls = FindNodes(Call).visit(op._func_table['haloupdate3d0'])
        destinations = {i.params[-2].field for i in calls}
        assert destinations == expected


class TestOperatorAdvanced(object):

    @pytest.mark.parallel(mode=4)
    def test_injection_wodup(self):
        """
        Test injection operator when the sparse points don't need to be replicated
        ("wodup" -> w/o duplication) over multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        f = Function(name='f', grid=grid, space_order=0)
        f.data[:] = 0.
        coords = np.array([(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)])
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 4.

        # This is the situation at this point
        # O is a grid point
        # * is a sparse point
        #
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O
        # |     |     |     |
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O

        op = Operator(sf.inject(field=f, expr=sf + 1))
        op.apply()

        assert np.all(f.data == 1.25)

    @pytest.mark.parallel(mode=4)
    def test_injection_wodup_wtime(self):
        """
        Just like ``test_injection_wodup``, but using a SparseTimeFunction
        instead of a SparseFunction. Hence, the data scattering/gathering now
        has to correctly pack/unpack multidimensional arrays.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        save = 3
        f = TimeFunction(name='f', grid=grid, save=save, space_order=0)
        f.data[:] = 0.
        coords = np.array([(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)])
        sf = SparseTimeFunction(name='sf', grid=grid, nt=save,
                                npoint=len(coords), coordinates=coords)
        sf.data[0, :] = 4.
        sf.data[1, :] = 8.
        sf.data[2, :] = 12.

        op = Operator(sf.inject(field=f, expr=sf + 1))
        op.apply()

        assert np.all(f.data[0] == 1.25)
        assert np.all(f.data[1] == 2.25)
        assert np.all(f.data[2] == 3.25)

    @pytest.mark.parallel(mode=4)
    def test_injection_dup(self):
        """
        Test injection operator when the sparse points are replicated over
        multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = 0.
        coords = [(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)]
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 4.

        # Global view (left) and local view (right, after domain decomposition)
        # O is a grid point
        # x is a halo point
        # A, B, C, D are sparse points
        #                               Rank0           Rank1
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |  A  |     |     |           |  A  |     |   |     |     |
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |     |  C  |  B  |     -->   |     |  C  |   |  C  |  B  |
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        # |     |  D  |     |           Rank2           Rank3
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        #                               |     |  C  |   |  C  |  B  |
        #                               O --- O --- x   x --- O --- O
        #                               |     |  D  |   |  D  |     |
        #                               O --- O --- x   x --- O --- O
        #
        # Expected `f.data` (global view)
        #
        # 1.25 --- 1.25 --- 0.00 --- 0.00
        #  |        |        |        |
        # 1.25 --- 2.50 --- 2.50 --- 1.25
        #  |        |        |        |
        # 0.00 --- 2.50 --- 3.75 --- 1.25
        #  |        |        |        |
        # 0.00 --- 1.25 --- 1.25 --- 0.00

        op = Operator(sf.inject(field=f, expr=sf + 1))
        op.apply()

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:  # rank0
            assert np.all(f.data_ro_domain == [[1.25, 1.25], [1.25, 2.5]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:  # rank1
            assert np.all(f.data_ro_domain == [[0., 0.], [2.5, 1.25]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(f.data_ro_domain == [[0., 2.5], [0., 1.25]])
        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(f.data_ro_domain == [[3.75, 1.25], [1.25, 0.]])

    @pytest.mark.parallel(mode=4)
    def test_interpolation_wodup(self):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        f = Function(name='f', grid=grid, space_order=0)
        f.data[:] = 4.
        coords = [(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)]
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 0.

        # This is the situation at this point
        # O is a grid point
        # * is a sparse point
        #
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O
        # |     |     |     |
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O

        op = Operator(sf.interpolate(expr=f))
        op.apply()

        assert np.all(sf.data == 4.)

    @pytest.mark.parallel(mode=4)
    def test_interpolation_dup(self):
        """
        Test interpolation operator when the sparse points are replicated over
        multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
        x, y = grid.dimensions

        # Init Function+data
        f = Function(name='f', grid=grid)
        f.data[:] = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
        coords = np.array([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)])
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 0.

        # Global view (left) and local view (right, after domain decomposition)
        # O is a grid point
        # x is a halo point
        # A, B, C, D are sparse points
        #                               Rank0           Rank1
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |  A  |     |     |           |  A  |     |   |     |     |
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |     |  C  |  B  |     -->   |     |  C  |   |  C  |  B  |
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        # |     |  D  |     |           Rank2           Rank3
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        #                               |     |  C  |   |  C  |  B  |
        #                               O --- O --- x   x --- O --- O
        #                               |     |  D  |   |  D  |     |
        #                               O --- O --- x   x --- O --- O
        #
        # The initial `f.data` is (global view)
        #
        # 1. --- 1. --- 1. --- 1.
        # |      |      |      |
        # 2. --- 2. --- 2. --- 2.
        # |      |      |      |
        # 3. --- 3. --- 3. --- 3.
        # |      |      |      |
        # 4. --- 4. --- 4. --- 4.
        #
        # Expected `sf.data` (global view)
        #
        # 1.5 --- 2.5 --- 2.5 --- 3.5

        op = Operator(sf.interpolate(expr=f))
        op.apply()

        assert np.all(sf.data == [1.5, 2.5, 2.5, 3.5][grid.distributor.myrank])

    @pytest.mark.parallel(mode=2)
    def test_subsampling(self):
        grid = Grid(shape=(40,))
        x = grid.dimensions[0]
        t = grid.stepping_dim
        time = grid.time_dim

        nt = 9

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        # Setup subsampled function
        factor = 4
        nsamples = (nt+factor-1)//factor
        times = ConditionalDimension('t_sub', parent=time, factor=factor)
        fsave = TimeFunction(name='fsave', grid=grid, save=nsamples, time_dim=times)

        eqns = [Eq(f.forward, f[t, x-1] + f[t, x+1]), Eq(fsave, f)]
        op = Operator(eqns)
        op.apply(time=nt-1)

        assert np.all(f.data_ro_domain[0] == fsave.data_ro_domain[nsamples-1])
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(fsave.data_ro_domain[nsamples-1, nt-1:] == 256.)
        else:
            assert np.all(fsave.data_ro_domain[nsamples-1, :-(nt-1)] == 256.)

        # Also check there are no redundant halo exchanges
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        # In particular, there is no need for a halo exchange within the conditional
        conditional = FindNodes(Conditional).visit(op)
        assert len(conditional) == 1
        assert len(FindNodes(Call).visit(conditional[0])) == 0

    @pytest.mark.parallel(mode=2)
    def test_arguments_subrange(self):
        """
        Test op.apply when a subrange is specified for a distributed dimension.
        """
        grid = Grid(shape=(16,))
        x = grid.dimensions[0]

        f = TimeFunction(name='f', grid=grid)

        op = Operator(Eq(f.forward, f + 1.))
        op.apply(time=0, x_m=4, x_M=11)

        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[1, :4] == 0.)
            assert np.all(f.data_ro_domain[1, 4:] == 1.)
        else:
            assert np.all(f.data_ro_domain[1, :-4] == 1.)
            assert np.all(f.data_ro_domain[1, -4:] == 0.)

    @pytest.mark.parallel(mode=2)
    def test_bcs_basic(self):
        """
        Test MPI in presence of boundary condition loops. Here, no halo exchange
        is expected (as there is no stencil in the computed expression) but we
        check that:

            * the left BC loop is computed by the leftmost rank only
            * the right BC loop is computed by the rightmost rank only
        """
        grid = Grid(shape=(20,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        thickness = 4

        u = TimeFunction(name='u', grid=grid, time_order=1)

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)
        xright = SubDimension.right(name='xright', parent=x, thickness=thickness)

        t_in_centre = Eq(u[t+1, xi], 1)
        leftbc = Eq(u[t+1, xleft], u[t+1, xleft+1] + 1)
        rightbc = Eq(u[t+1, xright], u[t+1, xright-1] + 1)

        op = Operator([t_in_centre, leftbc, rightbc])

        op.apply(time_m=1, time_M=1)

        glb_pos_map = u.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(u.data_ro_domain[0, thickness:] == 1.)
            assert np.all(u.data_ro_domain[0, :thickness] == range(thickness+1, 1, -1))
        else:
            assert np.all(u.data_ro_domain[0, :-thickness] == 1.)
            assert np.all(u.data_ro_domain[0, -thickness:] == range(2, thickness+2))

    @pytest.mark.parallel(mode=2)
    def test_interior_w_stencil(self):
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u[t, x-1] + u[t, x+1] + 1, subdomain=grid.interior))
        op.apply(time=1)

        glb_pos_map = u.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(u.data_ro_domain[0, 1] == 2.)
            assert np.all(u.data_ro_domain[0, 2:] == 3.)
        else:
            assert np.all(u.data_ro_domain[0, -2] == 2.)
            assert np.all(u.data_ro_domain[0, :-2] == 3.)

    @pytest.mark.parallel(mode=4)
    def test_misc_dims(self):
        """
        Test MPI in presence of Functions with mixed distributed/replicated
        Dimensions, with only a strict subset of the Grid dimensions used.
        """
        dx = Dimension(name='dx')
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2, save=4)
        c = Function(name='c', grid=grid, dimensions=(x, dx), shape=(4, 5))

        step = Eq(u.forward, (
            u[time, x-2, y] * c[x, 0]
            + u[time, x-1, y] * c[x, 1]
            + u[time, x, y] * c[x, 2]
            + u[time, x+1, y] * c[x, 3]
            + u[time, x+2, y] * c[x, 4]))

        for i in range(4):
            c.data[i, 0] = 1.0+i
            c.data[i, 1] = 1.0+i
            c.data[i, 2] = 3.0+i
            c.data[i, 3] = 6.0+i
            c.data[i, 4] = 5.0+i

        u.data[:] = 0.0
        u.data[0, 2, :] = 2.0

        op = Operator(step)

        op(time_m=0, time_M=0)

        if LEFT in glb_pos_map[x]:
            assert(np.all(u.data[1, 0, :] == 10.0))
            assert(np.all(u.data[1, 1, :] == 14.0))
        else:
            assert(np.all(u.data[1, 2, :] == 10.0))
            assert(np.all(u.data[1, 3, :] == 8.0))

    @pytest.mark.parallel(mode=9)
    def test_nontrivial_operator(self):
        """
        Test MPI in a non-trivial scenario: ::

            * 9 processes logically organised in a 3x3 cartesian grid (as opposed to
              most tests in this module, which only use 2 or 4 processed);
            * star-like stencil expression;
            * non-trivial Higdon-like BCs;
            * simultaneous presence of TimeFunction(grid), Function(grid), and
              Function(dimensions)
        """
        size_x, size_y = 9, 9
        tkn = 2

        # Grid and Dimensions
        grid = Grid(shape=(size_x, size_y,))
        x, y = grid.dimensions
        t = grid.stepping_dim

        # SubDimensions to implement BCs
        xl, yl = [SubDimension.left('%sl' % d.name, d, tkn) for d in [x, y]]
        xi, yi = [SubDimension.middle('%si' % d.name, d, tkn, tkn) for d in [x, y]]
        xr, yr = [SubDimension.right('%sr' % d.name, d, tkn) for d in [x, y]]

        # Functions
        u = TimeFunction(name='f', grid=grid)
        m = Function(name='m', grid=grid)
        c = Function(name='c', grid=grid, dimensions=(x,), shape=(size_x,))

        # Data initialization
        u.data_with_halo[:] = 0.
        m.data_with_halo[:] = 1.
        c.data_with_halo[:] = 0.

        # Equations
        c_init = Eq(c, 1.)
        eqn = Eq(u[t+1, xi, yi], u[t, xi, yi] + m[xi, yi] + c[xi] + 1.)
        bc_left = Eq(u[t+1, xl, yi], u[t+1, xl+1, yi] + 1.)
        bc_right = Eq(u[t+1, xr, yi], u[t+1, xr-1, yi] + 1.)
        bc_top = Eq(u[t+1, xi, yl], u[t+1, xi, yl+1] + 1.)
        bc_bottom = Eq(u[t+1, xi, yr], u[t+1, xi, yr-1] + 1.)

        op = Operator([c_init, eqn, bc_left, bc_right, bc_top, bc_bottom])
        op.apply(time=0)

        # Expected (global view):
        # 0 0 5 5 5 5 5 0 0
        # 0 0 4 4 4 4 4 0 0
        # 5 4 3 3 3 3 3 4 5
        # 5 4 3 3 3 3 3 4 5
        # 5 4 3 3 3 3 3 4 5
        # 5 4 3 3 3 3 3 4 5
        # 0 0 4 4 4 4 4 0 0
        # 0 0 5 5 5 5 5 0 0

        assert np.all(u.data_ro_domain[0] == 0)  # The write occures at t=1

        glb_pos_map = u.grid.distributor.glb_pos_map
        # Check cornes
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[0, 0, 5], [0, 0, 4], [5, 4, 3]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 0, 0], [4, 0, 0], [3, 4, 5]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 4, 3], [0, 0, 4], [0, 0, 5]])
        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[3, 4, 5], [4, 0, 0], [5, 0, 0]])
        # Check sides
        if not glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 4, 3], [5, 4, 3], [5, 4, 3]])
        elif not glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[3, 4, 5], [3, 4, 5], [3, 4, 5]])
        elif LEFT in glb_pos_map[x] and not glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 5, 5], [4, 4, 4], [3, 3, 3]])
        elif RIGHT in glb_pos_map[x] and not glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[3, 3, 3], [4, 4, 4], [5, 5, 5]])
        # Check center
        if not glb_pos_map[x] and not glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == 3)


class TestIsotropicAcoustic(object):

    """
    Test the isotropic acoustic wave equation with MPI.
    """

    @pytest.mark.parametrize('shape,kernel,space_order,nbpml,save', [
        ((60, ), 'OT2', 4, 10, False),
        ((60, 70), 'OT2', 8, 10, False),
    ])
    @pytest.mark.parallel(mode=1)
    def test_adjoint_codegen(self, shape, kernel, space_order, nbpml, save):
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape], kernel=kernel,
                                nbpml=nbpml, tn=500, space_order=space_order, nrec=130,
                                preset='layers-isotropic', dtype=np.float64)
        op_fwd = solver.op_fwd(save=save)
        fwd_calls = FindNodes(Call).visit(op_fwd)

        op_adj = solver.op_adj()
        adj_calls = FindNodes(Call).visit(op_adj)

        assert len(fwd_calls) == 2
        assert len(adj_calls) == 2

    @pytest.mark.parametrize('shape,kernel,space_order,nbpml,save,Eu,Erec,Ev,Esrca', [
        ((60, ), 'OT2', 4, 10, False, 385.853, 12937.250, 63818503.321, 101159204.362),
        ((60, 70), 'OT2', 8, 10, False, 351.217, 867.420, 405805.482, 239444.952),
        ((60, 70, 80), 'OT2', 12, 10, False, 153.122, 205.902, 27484.635, 11736.917)
    ])
    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap'), (8, 'basic')])
    def test_adjoint_F(self, shape, kernel, space_order, nbpml, save,
                       Eu, Erec, Ev, Esrca):
        """
        Unlike `test_adjoint_F` in test_adjoint.py, here we explicitly check the norms
        of all Operator-evaluated Functions. The numbers we check against are derived
        "manually" from sequential runs of test_adjoint::test_adjoint_F
        """
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape], kernel=kernel,
                                nbpml=nbpml, tn=tn, space_order=space_order, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)
        # Run forward operator
        rec, u, _ = solver.forward(save=save)

        assert np.isclose(norm(u), Eu, rtol=Eu*1.e-8)
        assert np.isclose(norm(rec), Erec, rtol=Erec*1.e-8)

        # Run adjoint operator
        srca, v, _ = solver.adjoint(rec=rec)

        assert np.isclose(norm(v), Ev, rtol=Ev*1.e-8)
        assert np.isclose(norm(srca), Esrca, rtol=Esrca*1.e-8)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = inner(srca, solver.geometry.src)
        term2 = norm(rec)**2
        assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)


if __name__ == "__main__":
    from devito import configuration
    configuration['mpi'] = True
    # TestDecomposition().test_reshape_left_right()
    # TestOperatorSimple().test_trivial_eq_2d()
    # TestOperatorSimple().test_num_comms('f[t,x-1,y] + f[t,x+1,y]', {'rc', 'lc'})
    # TestFunction().test_halo_exchange_bilateral()
    # TestSparseFunction().test_ownership(((1., 1.), (1., 3.), (3., 1.), (3., 3.)))
    # TestSparseFunction().test_local_indices([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)], [[0.], [1.], [2.], [3.]])  # noqa
    # TestSparseFunction().test_scatter_gather()
    # TestOperatorAdvanced().test_nontrivial_operator()
    # TestOperatorAdvanced().test_interpolation_dup()
    TestOperatorAdvanced().test_injection_wodup()
    # TestIsotropicAcoustic().test_adjoint_F((60, 70, 80), 'OT2', 12, 10, False,
    #                                        153.122, 205.902, 27484.635, 11736.917)
