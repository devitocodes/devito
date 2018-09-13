import numpy as np
from mpi4py import MPI

import pytest
from conftest import skipif_yask

from devito import (Grid, Function, TimeFunction, Dimension, ConditionalDimension,
                    SubDimension, Eq, Inc, Operator)
from devito.ir.iet import Call, Conditional, FindNodes
from devito.mpi import copy, sendrecv, update_halo
from devito.parameters import configuration
from devito.types import LEFT, RIGHT


def setup_module(module):
    configuration['mpi'] = True


def teardown_module(module):
    configuration['mpi'] = False


@skipif_yask
class TestPythonMPI(object):

    @pytest.mark.parallel(nprocs=[2, 4])
    def test_partitioning(self):
        grid = Grid(shape=(15, 15))
        f = Function(name='f', grid=grid)

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(15, 8), (15, 7)],
            4: [(8, 8), (8, 7), (7, 8), (7, 7)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(nprocs=[2, 4])
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

    @pytest.mark.parallel(nprocs=9)
    def test_neighborhood_2d(self):
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
            0: {x: {LEFT: MPI.PROC_NULL, RIGHT: 3}, y: {LEFT: MPI.PROC_NULL, RIGHT: 1}},
            1: {x: {LEFT: MPI.PROC_NULL, RIGHT: 4}, y: {LEFT: 0, RIGHT: 2}},
            2: {x: {LEFT: MPI.PROC_NULL, RIGHT: 5}, y: {LEFT: 1, RIGHT: MPI.PROC_NULL}},
            3: {x: {LEFT: 0, RIGHT: 6}, y: {LEFT: MPI.PROC_NULL, RIGHT: 4}},
            4: {x: {LEFT: 1, RIGHT: 7}, y: {LEFT: 3, RIGHT: 5}},
            5: {x: {LEFT: 2, RIGHT: 8}, y: {LEFT: 4, RIGHT: MPI.PROC_NULL}},
            6: {x: {LEFT: 3, RIGHT: MPI.PROC_NULL}, y: {LEFT: MPI.PROC_NULL, RIGHT: 7}},
            7: {x: {LEFT: 4, RIGHT: MPI.PROC_NULL}, y: {LEFT: 6, RIGHT: 8}},
            8: {x: {LEFT: 5, RIGHT: MPI.PROC_NULL}, y: {LEFT: 7, RIGHT: MPI.PROC_NULL}},
        }
        assert expected[distributor.myrank] == distributor.neighbours

    @pytest.mark.parallel(nprocs=2)
    def test_halo_exchange_bilateral(self):
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

    @pytest.mark.parallel(nprocs=2)
    def test_halo_exchange_bilateral_asymmetric(self):
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

    @pytest.mark.parallel(nprocs=4)
    def test_halo_exchange_quadrilateral(self):
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

    @skipif_yask
    @pytest.mark.parallel(nprocs=[2, 4])
    def test_ctypes_neighbours(self):
        grid = Grid(shape=(4, 4))
        distributor = grid.distributor

        PN = MPI.PROC_NULL
        attrs = ['xleft', 'xright', 'yleft', 'yright']
        expected = {  # nprocs -> [(rank0 xleft xright ...), (rank1 xleft ...), ...]
            2: [(PN, PN, PN, 1), (PN, PN, 0, PN)],
            4: [(PN, 2, PN, 1), (PN, 3, 0, PN), (0, PN, PN, 3), (1, PN, 2, PN)]
        }

        mapper = dict(zip(attrs, expected[distributor.nprocs][distributor.myrank]))
        _, _, obj = distributor._C_neighbours
        assert all(getattr(obj.value._obj, k) == v for k, v in mapper.items())


@skipif_yask
class TestCodeGeneration(object):

    def test_iet_copy(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = copy(f, [t])
        assert str(iet.parameters) == """\
(buf(buf_x, buf_y), buf_x_size, buf_y_size, dat(dat_time, dat_x, dat_y),\
 dat_time_size, dat_x_size, dat_y_size, otime, ox, oy)"""
        assert """\
  for (int x = 0; x <= buf_x_size - 1; x += 1)
  {
    for (int y = 0; y <= buf_y_size - 1; y += 1)
    {
      buf[x][y] = dat[otime][x + ox][y + oy];
    }
  }""" in str(iet)

    def test_iet_sendrecv(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = sendrecv(f, [t])
        assert str(iet.parameters) == """\
(dat(dat_time, dat_x, dat_y), dat_time_size, dat_x_size, dat_y_size,\
 buf_x_size, buf_y_size, ogtime, ogx, ogy, ostime, osx, osy, fromrank, torank, comm)"""
        assert str(iet.body[0]) == """\
float (*restrict dat)[dat_x_size][dat_y_size] __attribute__((aligned(64))) =\
 (float (*)[dat_x_size][dat_y_size]) dat_vec;
float bufs[buf_x_size][buf_y_size] __attribute__((aligned(64)));
MPI_Request rrecv;
float bufg[buf_x_size][buf_y_size] __attribute__((aligned(64)));
MPI_Request rsend;
MPI_Status srecv;
MPI_Irecv((float*)bufs,buf_x_size*buf_y_size,MPI_FLOAT,fromrank,13,comm,&rrecv);
gather_f((float*)bufg,buf_x_size,buf_y_size,(float*)dat,dat_time_size,dat_x_size,\
dat_y_size,ogtime,ogx,ogy);
MPI_Isend((float*)bufg,buf_x_size*buf_y_size,MPI_FLOAT,torank,13,comm,&rsend);
MPI_Wait(&rsend,MPI_STATUS_IGNORE);
MPI_Wait(&rrecv,&srecv);
if (fromrank != MPI_PROC_NULL)
{
  scatter_f((float*)bufs,buf_x_size,buf_y_size,(float*)dat,dat_time_size,dat_x_size,\
dat_y_size,ostime,osx,osy);
}"""

    def test_iet_update_halo(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = update_halo(f, [t])
        assert str(iet.parameters) == """\
(f(t, x, y), mxl, mxr, myl, myr, comm, nb, otime, t_size, x_size, y_size)"""
        assert """\
MPI_Comm *comm = (MPI_Comm*) _comm;
struct neighbours *nb = (struct neighbours*) _nb;
if (mxl)
{
  sendrecv(f_vec,t_size,x_size + 1 + 1,y_size + 1 + 1,1,y_size + 1 + 1,\
otime,1,0,otime,x_size + 1,0,nb->xright,nb->xleft,comm);
}
if (mxr)
{
  sendrecv(f_vec,t_size,x_size + 1 + 1,y_size + 1 + 1,1,y_size + 1 + 1,\
otime,x_size,0,otime,0,0,nb->xleft,nb->xright,comm);
}
if (myl)
{
  sendrecv(f_vec,t_size,x_size + 1 + 1,y_size + 1 + 1,x_size + 1 + 1,1,\
otime,0,1,otime,0,y_size + 1,nb->yright,nb->yleft,comm);
}
if (myr)
{
  sendrecv(f_vec,t_size,x_size + 1 + 1,y_size + 1 + 1,x_size + 1 + 1,1,\
otime,0,y_size,otime,0,0,nb->yleft,nb->yright,comm);
}"""


@skipif_yask
class TestOperatorSimple(object):

    @pytest.mark.parallel(nprocs=[2, 4, 8, 16, 32])
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

    @pytest.mark.parallel(nprocs=2)
    def test_trivial_eq_1d_save(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=5)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[time, x-1] + f[time, x+1] + 1))
        op.apply()

        time_M = op.prepare_arguments()['time_M']

        assert np.all(f.data_ro_domain[1] == 3.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[-1, time_M:] == 31.)
        else:
            assert np.all(f.data_ro_domain[-1, :-time_M] == 31.)

    @pytest.mark.parallel(nprocs=4)
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

        assert np.all(f.data_ro_interior[0] == interior)
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

    @pytest.mark.parallel(nprocs=4)
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

    def test_stencil_nowrite_implies_haloupdate(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator(Eq(g, f[t, x-1] + f[t, x+1] + 1.))

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    def test_avoid_redundant_haloupdate(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        i = Dimension(name='i')
        j = Dimension(name='j')

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(f[t+1, i], f[t+1, i] + 1.),  # no halo update as it's an Inc
                       Eq(g, f[t, j] + 1)])  # access `f` at `t`, not `t+1`!

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(nprocs=2)
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


@skipif_yask
class TestOperatorAdvanced(object):

    @pytest.mark.parallel(nprocs=2)
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

    @pytest.mark.parallel(nprocs=2)
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

    @pytest.mark.parallel(nprocs=2)
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

    @pytest.mark.parallel(nprocs=9)
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
    Test the acoustic wave model with MPI.
    """

    # TODO: Cannot mark the following test as `xfail` since this marker
    # doesn't cope well with the `parallel` mark. Leaving it commented out
    # for the time being...
    # @pytest.mark.parametrize('shape, kernel, space_order, nbpml', [
    #     # 1 tests with varying time and space orders
    #     ((60, ), 'OT2', 4, 10),
    # ])
    # @pytest.mark.parallel(nprocs=2)
    # def test_adjoint_F(self, shape, kernel, space_order, nbpml):
    #     from test_adjoint import TestAdjoint
    #     TestAdjoint().test_adjoint_F('layers', shape, kernel, space_order, nbpml)

    pass


if __name__ == "__main__":
    configuration['mpi'] = True
    TestOperatorAdvanced().test_nontrivial_operator()
