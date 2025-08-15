import numpy as np
import pytest
from functools import cached_property

from conftest import _R, assert_blocking, assert_structure
from devito import (Grid, Constant, Function, TimeFunction, SparseFunction,
                    SparseTimeFunction, VectorTimeFunction, TensorTimeFunction,
                    Dimension, ConditionalDimension, div, solve, diag, grad,
                    SubDimension, SubDomain, Eq, Ne, Inc, NODE, Operator, norm,
                    inner, configuration, switchconfig, generic_derivative,
                    PrecomputedSparseFunction, DefaultDimension, Buffer,
                    CustomDimension)
from devito.arch.compiler import OneapiCompiler
from devito.data import LEFT, RIGHT
from devito.ir.iet import (Call, Conditional, Iteration, FindNodes, FindSymbols,
                           retrieve_iteration_tree)
from devito.mpi import MPI
from devito.mpi.routines import (HaloUpdateCall, HaloUpdateList, MPICall,
                                 ComputeCall)
from devito.mpi.distributed import CustomTopology
from devito.tools import Bunch

from examples.seismic.acoustic import acoustic_setup
from test_dse import TestTTI


class TestDistributor:

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning(self, mode):
        grid = Grid(shape=(15, 15))
        f = Function(name='f', grid=grid)

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8, 15), (7, 15)],
            4: [(8, 8), (8, 7), (7, 8), (7, 7)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]
        assert f.size_global == 225
        assert distributor.nprocs_local == distributor.nprocs

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning_fewer_dims(self, mode):
        """Test domain decomposition for Functions defined over a strict subset
        of grid-decomposed dimensions."""
        size_x, size_y = 16, 16
        grid = Grid(shape=(size_x, size_y))
        x, y = grid.dimensions

        # A function with fewer dimensions that in `grid`
        f = Function(name='f', grid=grid, dimensions=(x,), shape=(size_x,))

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8,), (8,)],
            4: [(8,), (8,), (8,), (8,)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning_fewer_dims_timefunc(self, mode):
        """Test domain decomposition for Functions defined over a strict subset
        of grid-decomposed dimensions."""
        size_x, size_y = 16, 16
        grid = Grid(shape=(size_x, size_y))
        x, y = grid.dimensions

        # A function with fewer dimensions that in `grid`
        f = TimeFunction(
            name='f',
            grid=grid,
            dimensions=(grid.time_dim, x,),
            shape=(10, size_x,),
        )

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8,), (8,)],
            4: [(8,), (8,), (8,), (8,)]
        }
        assert len(f.shape) == 2
        assert f.shape[0] == 10
        assert f.shape[1:] == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(mode=9)
    def test_neighborhood_horizontal_2d(self, mode):
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
    def test_neighborhood_diagonal_2d(self, mode):
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
    def test_ctypes_neighborhood(self, mode):
        grid = Grid(shape=(4, 4))
        distributor = grid.distributor

        PN = MPI.PROC_NULL
        attrs = ['ll', 'lc', 'lr', 'cl', 'cc', 'cr', 'rl', 'rc', 'rr']
        expected = {  # nprocs -> [(rank0 xleft xright ...), (rank1 xleft ...), ...]
            2: [(PN, PN, PN, PN, 0, PN, PN, 1, PN),
                (PN, 0, PN, PN, 1, PN, PN, PN, PN)],
            4: [(PN, PN, PN, PN, 0, 1, PN, 2, 3),
                (PN, PN, PN, 0, 1, PN, 2, 3, PN),
                (PN, 0, 1, PN, 2, 3, PN, PN, PN),
                (0, 1, PN, 2, 3, PN, PN, PN, PN)]
        }

        mapper = dict(zip(attrs, expected[distributor.nprocs][distributor.myrank]))
        obj = distributor._obj_neighborhood
        value = obj._arg_defaults()[obj.name]
        assert all(getattr(value._obj, k) == v for k, v in mapper.items())

    @pytest.mark.parallel(mode=[4])
    def test_custom_topology(self, mode):
        grid = Grid(shape=(15, 15))
        f = Function(name='f', grid=grid)

        # Default topology, computed by Devito
        distributor = grid.distributor
        assert distributor.topology == (2, 2)
        expected = [(8, 8), (8, 7), (7, 8), (7, 7)]
        assert f.shape == expected[distributor.myrank]
        assert f.size_global == 225

        # Now with a custom topology
        grid2 = Grid(shape=(15, 15), topology=(4, 1))
        f2 = Function(name='f', grid=grid2)

        distributor = grid2.distributor
        assert distributor.topology == (4, 1)
        expected = [(4, 15), (4, 15), (4, 15), (3, 15)]
        assert f2.shape == expected[distributor.myrank]
        assert f2.size_global == f.size_global

    @pytest.mark.parametrize('comm_size, topology, dist_topology', [
        (2, (1, '*'), (1, 2)),
        (2, ('*', '*'), (2, 1)),
        (1, (1, '*', '*'), (1, 1, 1)),
        (2, (1, '*', '*'), (1, 2, 1)),
        (2, (2, '*', '*'), (2, 1, 1)),
        (3, (1, '*', '*'), (1, 3, 1)),
        (3, ('*', '*', 1), (3, 1, 1)),
        (4, (2, '*', '*'), (2, 2, 1)),
        (4, ('*', '*', 2), (2, 1, 2)),
        (6, ('*', '*', 1), (3, 2, 1)),
        (6, (1, '*', '*'), (1, 3, 2)),
        (6, ('*', '*', '*'), (3, 2, 1)),
        (12, ('*', '*', '*'), (3, 2, 2)),
        (12, ('*', 3, '*'), (2, 3, 2)),
        (18, ('*', '*', '*'), (3, 3, 2)),
        (18, ('*', '*', 9), (2, 1, 9)),
        (18, ('*', '*', 3), (3, 2, 3)),
        (24, ('*', '*', '*'), (6, 2, 2)),
        (32, ('*', '*', '*'), (4, 4, 2)),
        (8, ('*', 1, '*'), (4, 1, 2)),
        (8, ('*', '*', 1), (4, 2, 1)),
        (8, ('*', '*', '*'), (2, 2, 2)),
        (9, ('*', '*', '*'), (3, 3, 1)),
        (11, (1, '*', '*'), (1, 11, 1)),
        (22, ('*', '*', '*'), (11, 2, 1)),
        (16, ('*', 1, '*'), (4, 1, 4)),
        (32, ('*', '*', 1), (8, 4, 1)),
        (64, ('*', '*', 1), (8, 8, 1)),
        (64, ('*', 2, 4), (8, 2, 4)),
        (128, ('*', '*', 1), (16, 8, 1)),
        (231, ('*', '*', '*'), (11, 7, 3)),
        (256, (1, '*', '*'), (1, 16, 16)),
        (256, ('*', '*', '*'), (8, 8, 4)),
        (256, ('*', '*', 2), (16, 8, 2)),
        (256, ('*', 32, 2), (4, 32, 2)),
    ])
    @pytest.mark.parallel(mode=[2])
    def test_custom_topology_v2(self, comm_size, topology, dist_topology, mode):
        dummy_comm = Bunch(size=comm_size)
        custom_topology = CustomTopology(topology, dummy_comm)
        assert custom_topology == dist_topology

    @pytest.mark.parallel(mode=[(4, 'diag2')])
    @switchconfig(topology='y')
    def test_custom_topology_fallback(self, mode):
        grid = Grid(shape=(16,))
        f = Function(name='f', grid=grid)

        # The input topology was `y` but Grid only has one axis, so we decompose
        # along that instead
        assert f.shape == (4,)


class TestSubDistributor:

    sd_specs = [('middle', 2, 1), ('left', 2), ('right', 3)]

    @pytest.mark.parametrize('sd_x', sd_specs)
    @pytest.mark.parametrize('sd_y', sd_specs)
    @pytest.mark.parallel(mode=[2])
    def test_intervals(self, sd_x, sd_y, mode):
        """
        Check the interval of indices spanned by the SubDomain is correctly calculated
        within SubDistributor. Also check that the interval of indices spanned by the
        SubDomain on each rank is correct.
        """
        def check_interval(interval, spec, size, vmin=None, vmax=None):
            if spec[0] == 'middle':
                start = spec[1]
                end = size - spec[2] - 1
            elif spec[0] == 'left':
                start = 0
                end = spec[1] - 1
            else:
                start = size - spec[1]
                end = size - 1

            if vmin:
                assert interval.start == max(start, vmin)
            else:
                assert interval.start == start

            if vmax:
                assert interval.end == min(end, vmax)
            else:
                assert interval.end == end

        class MyDomain(SubDomain):

            name = 'mydomain'

            def define(self, dimensions):
                x, y = dimensions
                return {x: sd_x, y: sd_y}

        grid = Grid(shape=(10, 10))
        md = MyDomain(grid=grid)
        d = md.distributor

        check_interval(d.subdomain_interval[0], sd_x, grid.shape[0])
        check_interval(d.subdomain_interval[1], sd_y, grid.shape[1])

        if not d.intervals[0].is_empty:
            check_interval(d.intervals[0], sd_x, grid.shape[0],
                           vmin=grid.origin_ioffset[0],
                           vmax=grid.origin_ioffset[0]+grid.shape_local[0]-1)
        if not d.intervals[1].is_empty:
            check_interval(d.intervals[1], sd_y, grid.shape[1],
                           vmin=grid.origin_ioffset[1],
                           vmax=grid.origin_ioffset[1]+grid.shape_local[1]-1)

    @pytest.mark.parametrize('sd', sd_specs)
    @pytest.mark.parallel(mode=[3])
    def test_crosses(self, sd, mode):
        """
        Check that the edges of the rank crossed by the subdomain are correctly
        identified.
        """
        class MyDomain(SubDomain):

            name = 'mydomain'

            def define(self, dimensions):
                x, y = dimensions
                return {x: sd, y: y}

        grid = Grid(shape=(10, 10))
        md = MyDomain(grid=grid)
        xi, yi = md.dimensions
        d = md.distributor

        rank = grid.distributor.comm.rank
        if sd[0] == 'middle':
            if rank == 0:
                check = {LEFT: False, RIGHT: True}
            elif rank == 1:
                check = {LEFT: True, RIGHT: True}
            else:
                check = {LEFT: True, RIGHT: False}
        else:
            check = {LEFT: False, RIGHT: False}

        assert d.crosses[xi] == check

    @pytest.mark.parametrize('sd', sd_specs)
    @pytest.mark.parallel(mode=[3])
    def test_decomposition(self, sd, mode):
        """
        Check that the subdomain is correctly decomposed.
        """
        class MyDomain(SubDomain):

            name = 'mydomain'

            def define(self, dimensions):
                x, y = dimensions
                return {x: sd, y: y}

        grid = Grid(shape=(10, 10))
        md = MyDomain(grid=grid)
        d = md.distributor

        for dec, pdec, sdi, sh in zip(d.decomposition, d.parent.decomposition,
                                      d.subdomain_interval, grid.shape):
            # Get the global min and max
            lower_bounds = [np.amin(i) for i in dec if i.size != 0]
            upper_bounds = [np.amax(i) for i in dec if i.size != 0]

            parent_lower_bounds = [np.amin(i) for i in pdec]
            parent_upper_bounds = [np.amax(i) for i in pdec]

            dM = np.amax(upper_bounds)
            dm = np.amin(lower_bounds)

            irange = np.arange(dm, dM+1)

            # Indices increase monotonically over correct range
            assert np.all(np.concatenate(dec) == irange)

            # Inner boundaries line up with parent decomposition
            # Skip the check if the subdomain means there are no inner boundaries
            if len(lower_bounds) > 1:
                assert np.all(parent_lower_bounds[1:] == lower_bounds[1:])
            if len(upper_bounds) > 1:
                assert np.all(parent_upper_bounds[:-1] == upper_bounds[:-1])

            # Outer boundaries line up with subdomain
            if sdi is None:
                assert lower_bounds[0] == 0
                assert upper_bounds[-1] == sh - 1
            else:
                assert lower_bounds[0] == sdi.start
                assert upper_bounds[-1] == sdi.end


class TestFunction:

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_bilateral(self, mode):
        """
        Test halo exchange between two processes organised in a 2x1 cartesian grid.

        On the left, the initial ``data_with_inhalo``; on the right, the situation
        after the halo exchange.

               rank0               rank0
            0 0 0 0 0 0         0 0 0 0 0 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 0 0 0 0 0         0 2 2 2 2 0
                         ---->
               rank1               rank1
            0 0 0 0 0 0         0 1 1 1 1 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 0 0 0 0 0         0 0 0 0 0 0
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[:] == 1.)
            assert np.all(f._data_ro_with_inhalo[-1, 1:-1] == 2.)
            assert np.all(f._data_ro_with_inhalo[0, :] == 0.)
        else:
            assert np.all(f.data_ro_domain[:] == 2.)
            assert np.all(f._data_ro_with_inhalo[0, 1:-1] == 1.)
            assert np.all(f._data_ro_with_inhalo[-1, :] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)

    @pytest.mark.parametrize("paddings", [
        (None, None),
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((1, 0), (0, 1)),
    ])
    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_bilateral_asymmetric(self, paddings, mode):
        """
        Test halo exchange between two processes organised in a 2x1 cartesian grid.

        In this test, the size of left and right halo regions have different size.

        On the left, the initial ``data_with_inhalo``; on the right, the situation
        after the halo exchange.

                rank0                 rank0
            0 0 0 0 0 0 0         0 0 0 0 0 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 0 0 0 0 0 0         0 2 2 2 2 0 0
            0 0 0 0 0 0 0         0 2 2 2 2 0 0
                           ---->
                rank1                 rank1
            0 0 0 0 0 0 0         0 1 1 1 1 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 0 0 0 0 0 0         0 0 0 0 0 0 0
            0 0 0 0 0 0 0         0 0 0 0 0 0 0
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        padding = paddings[grid.distributor.comm.rank]

        f = Function(name='f', grid=grid, space_order=(1, 1, 2), padding=padding)
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[:] == 1.)
            assert np.all(f._data_ro_with_inhalo[-2:, 1:-2] == 2.)
            assert np.all(f._data_ro_with_inhalo[0:1, :] == 0.)
        else:
            assert np.all(f.data_ro_domain[:] == 2.)
            assert np.all(f._data_ro_with_inhalo[:1, 1:-2] == 1.)
            assert np.all(f._data_ro_with_inhalo[-2:, :] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, :1] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, -2:] == 0.)

    @pytest.mark.parallel(mode=4)
    def test_halo_exchange_quadrilateral(self, mode):
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
    def test_local_indices(self, shape, expected, mode):
        grid = Grid(shape=shape)
        f = Function(name='f', grid=grid)

        assert all(i == slice(*j)
                   for i, j in zip(f.local_indices, expected[grid.distributor.myrank]))

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('shape', [(1,), (2, 3), (4, 5, 6)])
    def test_mpi4py_nodevmpi(self, shape, mode):

        with switchconfig(mpi=False):
            # Mimic external mpi init
            MPI.Init()
            # Check that internal Function work correctly
            grid = Grid(shape=shape)
            f = Function(name="f", grid=grid, space_order=1)
            assert f.data.shape == shape
            assert f.data_with_halo.shape == tuple(s+2 for s in shape)
            assert f.data._local.shape == shape
            MPI.Finalize()


class TestSparseFunction:

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('shape, coords, points', [
        ((4, 4), ((1., 1.), (1., 3.), (3., 1.), (3., 3.)), 1),
        ((8, ), ((1.,), (3.,), (5.,), (7.,)), 1),
        ((8, ), ((1.,), (2.,), (3.,), (4.,), (5.,), (6.,), (7.,), (8.,)), 2)
    ])
    def test_ownership(self, shape, coords, points, mode):
        """Given a sparse point ``p`` with known coordinates, this test checks
        that the MPI rank owning ``p`` is retrieved correctly."""
        grid = Grid(shape=shape, extent=shape)

        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)

        # The domain decomposition is so that the i-th MPI rank gets exactly one
        # sparse point `p` and, incidentally, `p` is logically owned by `i`
        assert len(sf.gridpoints) == points
        ownership = grid.distributor.glb_to_rank(sf.gridpoints)
        assert list(ownership.keys()) == [grid.distributor.myrank]

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('coords,expected,expectedinds', [
        ([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)], [[0.], [1.], [2.], [3.]],
         [(slice(0, 1), ), (slice(1, 2), ), (slice(2, 3), ), (slice(3, 4), )]),
        ([(1.5, 1.5), ], [[], [], [], [0.]],
         [(slice(0, -1), ), (slice(0, -1), ), (slice(0, -1), ), (slice(0, 1), )])
    ])
    def test_local_indices(self, coords, expected, expectedinds, mode):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        data = np.array([0., 1., 2., 3.])
        coords = np.array(coords)
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords))

        # Each of the 4 MPI ranks get one (randomly chosen) sparse point if using 4 points
        # If we are only using 1 point, only the last mpi rank receives a sparse point
        npointexpected = 1 if len(coords) == 4 else (grid.distributor.myrank+1)//4
        assert sf.npoint == npointexpected

        sf.coordinates.data[:] = coords
        sf.data[:] = data
        expected = np.array(expected[grid.distributor.myrank])
        assert np.all(sf.data == expected)
        expectedinds = expectedinds[grid.distributor.myrank]
        assert sf.local_indices == expectedinds

    @pytest.mark.parallel(mode=4)
    def test_scatter_gather(self, mode):
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
        loc_coords = sf._dist_scatter()[sf.coordinates]
        assert len(loc_data) == 1
        assert loc_data[0] == grid.distributor.myrank
        # Do some local computation
        loc_data = loc_data*2

        # Gather
        sf._dist_data_gather(loc_data)
        assert len(sf.data) == 1
        assert np.all(sf.data == data[sf.local_indices]*2)

        sf._dist_subfunc_gather(loc_coords, sf.coordinates)
        assert sf.coordinates.data.shape == (1, 2)
        assert np.all(sf.coordinates.data == coords[sf.local_indices[0], :])

    @pytest.mark.parallel(mode=4)
    @switchconfig(condition=isinstance(configuration['compiler'],
                  (OneapiCompiler)), safe_math=True)
    def test_sparse_coords(self, mode):
        grid = Grid(shape=(21, 31, 21), extent=(20, 30, 20))
        x, y, z = grid.dimensions

        coords = np.zeros((21*31, 3))
        coords[:, 0] = np.asarray([i for i in range(21) for j in range(31)])
        coords[:, 1] = np.asarray([j for i in range(21) for j in range(31)])
        sf = SparseFunction(name="s", grid=grid, coordinates=coords, npoint=21*31)

        u = Function(name="u", grid=grid, space_order=1)
        u.data[:, :, 0] = np.reshape(np.asarray([i+j for i in range(21)
                                                 for j in range(31)]), (21, 31))

        op = Operator(sf.interpolate(u))
        op.apply()

        for i in range(21*31):
            coords_loc = sf.coordinates.data[i, 1]
            if coords_loc is not None:
                coords_loc += sf.coordinates.data[i, 0]
            assert sf.data[i] == coords_loc

    @pytest.mark.parallel(mode=4)
    def test_sparse_coords_issue1823(self, mode):
        grid = Grid((101, 101, 101), extent=(1000, 1000, 1000))
        coords = np.array([[1000., 0., 900.], [1000., 300., 700.],
                           [1000., 500., 500.], [1000., 700., 300.],
                           [1000., 900., 0.], [1000., 0., 850.]])
        rec = SparseTimeFunction(name="s", grid=grid, coordinates=coords,
                                 nt=10, npoint=6)
        ref = SparseTimeFunction(name="s1", grid=grid, coordinates=coords,
                                 nt=10, npoint=6)
        u = TimeFunction(name="u", grid=grid, space_order=1)

        Operator([Eq(u, u+1)]+rec.interpolate(u))()

        assert np.allclose(rec.coordinates.data[:], ref.coordinates.data)

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('r', [2])
    def test_precomputed_sparse(self, r, mode):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        coords = np.array([(1.0, 1.0), (2.0, 2.0), (1.0, 2.0), (2.0, 1.0)])
        points = np.array([(1, 1), (2, 2), (1, 2), (2, 1)])
        coeffs = np.ones((4, 2, r))

        sf1 = PrecomputedSparseFunction(name="sf1", grid=grid, coordinates=coords,
                                        npoint=4, interpolation_coeffs=coeffs, r=r)
        sf2 = PrecomputedSparseFunction(name="sf2", grid=grid, gridpoints=points,
                                        npoint=4, interpolation_coeffs=coeffs, r=r)

        assert sf1.npoint == 1
        assert sf2.npoint == 1
        assert np.all(sf1.coordinates.data.shape == (1, 2))
        assert np.all(sf2.gridpoints.data.shape == (1, 2))
        assert np.all(sf1._coords_indices == sf2.gridpoints_data)
        assert np.all(sf1.interpolation_coeffs.shape == (1, 2, r))
        assert np.all(sf2.interpolation_coeffs.shape == (1, 2, r))

        u = Function(name="u", grid=grid, space_order=r)
        u._data_with_outhalo[:] = 1
        Operator(sf2.interpolate(u))()
        assert np.all(sf2.data == 4)
        Operator(sf1.interpolate(u))()
        assert np.all(sf1.data == 4)

    @pytest.mark.parallel(mode=4)
    def test_sparse_first(self, mode):
        """
        Tests custom sprase function with sparse dimension as first index.
        """

        class SparseFirst(SparseFunction):
            """ Custom sparse class with the sparse dimension as the first one"""
            _sparse_position = 0

        dr = Dimension("cd")
        ds = DefaultDimension("ps", default_value=3)
        grid = Grid((11, 11))
        dims = grid.dimensions
        s = SparseFirst(name="s", grid=grid, npoint=4, dimensions=(dr, ds), shape=(4, 3),
                        coordinates=[[.1, .1], [.2, .2], [.3, .3], [.5, .5]])

        # Check dimensions and shape are correctly initialized
        assert s.indices[s._sparse_position] == dr
        assert s.shape == (1, 3)
        assert s.coordinates.indices[0] == dr

        # Operator
        u = TimeFunction(name="u", grid=grid, time_order=1)
        fs = Function(name="fs", grid=grid, dimensions=(*dims, ds), shape=(11, 11, 3))

        eqs = [Eq(u.forward, u+1), Eq(fs, u)]
        # No time dependence so need the implicit dim
        rec = s.interpolate(expr=s+fs, implicit_dims=grid.stepping_dim)
        op = Operator(eqs + rec)

        # Check generated code -- expected one halo exchange
        assert len(FindNodes(Call).visit(op)) == 1

        op(time_M=10)
        expected = 10*11/2  # n (n+1)/2
        assert np.allclose(s.data, expected)

    @pytest.mark.parallel(mode=[(4, 'diag2')])
    def test_no_grid_dim_slow(self, mode):
        shape = (12, 13, 14)
        nfreq = 5
        nrec = 2

        grid = Grid(shape=shape)
        f = DefaultDimension(name="f", default_value=nfreq)

        u = Function(name="u", grid=grid, dimensions=(*grid.dimensions, f),
                     shape=(*shape, nfreq), space_order=2)
        u.data.fill(1)

        class CoordSlowSparseFunction(SparseFunction):
            _sparse_position = 0

        r = Dimension(name="r")
        s = CoordSlowSparseFunction(name="s", grid=grid, dimensions=(r, f),
                                    shape=(nrec, nfreq), npoint=nrec)

        rec_eq = s.interpolate(expr=u)

        op = Operator([Eq(u, 1)] + rec_eq)

        # Check generated code -- expected one halo exchange
        assert len(FindNodes(Call).visit(op)) == 1

        op.apply()
        assert np.all(s.data == 1)

    @pytest.mark.parallel(mode=4)
    def test_no_grid_dim_slow_time(self, mode):
        shape = (12, 13, 14)
        nfreq = 5
        nrec = 2

        grid = Grid(shape=shape)
        t = grid.stepping_dim
        f = DefaultDimension(name="f", default_value=nfreq)

        u = TimeFunction(name="u", grid=grid, dimensions=(t, *grid.dimensions, f),
                         shape=(2, *shape, nfreq), space_order=2)

        class CoordSlowSparseFunction(SparseTimeFunction):
            _sparse_position = 0

        r = Dimension(name="r")
        s = CoordSlowSparseFunction(name="s", grid=grid, dimensions=(r, f),
                                    shape=(nrec, nfreq), npoint=nrec, nt=5)

        rec_eq = s.interpolate(expr=u)

        op = Operator([Eq(u, 1)] + rec_eq)

        # Check generated code -- expected one halo exchange
        assert len(FindNodes(Call).visit(op)) == 1

        op.apply(time_M=5)
        assert np.all(s.data == 1)


class TestOperatorSimple:

    @pytest.mark.parallel(mode=[2, 4, 8])
    def test_trivial_eq_1d(self, mode):
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

    @pytest.mark.parallel(mode=[2])
    def test_trivial_eq_1d_asymmetric(self, mode):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x+1] + 1))
        op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 2.)
        if f.grid.distributor.myrank == 0:
            assert np.all(f.data_ro_domain[0] == 3.)
        else:
            assert np.all(f.data_ro_domain[0, :-1] == 3.)
            assert f.data_ro_domain[0, -1] == 2.

    @pytest.mark.parallel(mode=2)
    def test_trivial_eq_1d_save(self, mode):
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

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap'),
                                (4, 'overlap2'), (4, 'diag2'), (4, 'full')])
    def test_trivial_eq_2d(self, mode):
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

    @pytest.mark.parallel(mode=[(8, 'basic'), (8, 'diag'), (8, 'overlap'),
                                (8, 'overlap2'), (8, 'diag2'), (8, 'full')])
    def test_trivial_eq_3d(self, mode):
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

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag')])
    def test_multiple_eqs_funcs(self, mode):
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
        # one is expected before the `x` Iteration, with two components, namely
        # `f` and `g`
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        assert calls[0].ncomps == 2

    @pytest.mark.parallel(mode=2)
    def test_reapply_with_different_functions(self, mode):
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


def check_halo_exchanges(op, exp0, exp1):
    calls = FindNodes(HaloUpdateCall).visit(op)
    assert len(calls) == exp0
    tloop = get_time_loop(op)
    calls = FindNodes(HaloUpdateCall).visit(tloop)
    assert len(calls) == exp1

    return calls, tloop


class TestCodeGeneration:

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_as_nostencil_basic(self, mode):
        grid = Grid(shape=(12,))

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(f.forward, f + 1.),
                       Eq(g, f + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_as_nostencil_advanced(self, mode):
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

    @pytest.mark.parallel(mode=1)
    def test_avoid_redundant_haloupdate(self, mode):
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
        calls = FindNodes(Call).visit(get_time_loop(op))
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_avoid_redundant_haloupdate_cond(self, mode):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        i = Dimension(name='i')
        j = Dimension(name='j')

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        t_sub = ConditionalDimension(name='t_sub', parent=t, factor=2)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(f[t+1, i], 1.),  # no halo update as it's an Inc
                       # access `f` at `t`, not `t+1` through factor subdim!
                       Eq(g, f[t, j] + 1, implicit_dim=t_sub)])

        calls = FindNodes(Call).visit(get_time_loop(op))
        assert len(calls) == 1
        assert calls[0].functions[0] is f

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_if_distr_but_sequential(self, mode):
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
    def test_avoid_haloupdate_with_local_subdims(self, mode):
        grid = Grid(shape=(4,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        thickness = 4

        u = TimeFunction(name='u', grid=grid, time_order=1)

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)

        eq_centre = Eq(u[t+1, xi], u[t, xi-1] + u[t, xi+1] + 1.)
        eq_left = Eq(u[t+1, xleft], u[t+1, xleft+1] + u[t, xleft+1] + 1.)

        # There is only one halo update -- for the `eq_centre` expression.
        # `eq_left` gets no halo update since it's a left-SubDimension, which by
        # default (i.e., unless one passes `local=False` to SubDimension.left) is
        # assumed to be a local Dimension.
        op = Operator([eq_centre, eq_left])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_with_local_customdim(self, mode):
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        d = CustomDimension(name='d', symbolic_min=1, symbolic_max=3, parent=y)

        u = TimeFunction(name='u', grid=grid, space_order=4)

        eq = Eq(u.forward.subs(y, -d), u.subs(y, d - 1) + 1)

        op = Operator(eq)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_with_constant_index(self, mode):
        grid = Grid(shape=(4,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)

        eq = Eq(u.forward, u[t, 1] + u[t, 1 + x.symbolic_min] + u[t, x])
        op = Operator(eq)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_do_haloupdate_with_constant_locindex(self, mode):
        """
        Like `test_avoid_haloupdate_with_constant_index`, there is again
        a constant index, but this time along a loc-index (`t` Dimension),
        which needs to be handled by the `compute_loc_indices` function.
        The actual halo update is induced by u.dx.
        """
        grid = Grid(shape=(4,))
        x = grid.dimensions[0]

        u = TimeFunction(name='u', grid=grid)

        eq = Eq(u.forward, u[0, x] + u.dx)
        op = Operator(eq)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_if_no_flowdep(self, mode):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        i = Dimension(name='i')

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(g[i], f[t, h[i]] + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        calls = FindNodes(Call).visit(get_time_loop(op))
        assert len(calls) == 1

        # Below, there is a flow-dependence along `x`, so a further halo update
        # before the Inc is required
        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(g[i], f[t+1, h[i]] + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_with_subdims(self, mode):
        """
        This test stems from https://github.com/devitocodes/devito/issues/1119

        Ensure SubDimensions are treated just like any other Dimensions when it
        gets to placing halo exchanges.
        """
        grid = Grid(shape=(20, 20, 20))

        u = TimeFunction(name="u", grid=grid, space_order=2)
        U = TimeFunction(name="U", grid=grid, space_order=2)

        eqns = [Eq(u.forward, u.dx, subdomain=grid.interior),
                Eq(U.forward, U.dx + u.forward)]

        op = Operator(eqns)

        assert len(op._func_table) == 4

        # There are exactly two halo exchange calls in the Operator body
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2
        assert calls[0].name == 'haloupdate0'
        assert calls[1].name == 'haloupdate0'

        # ... and none in the created efuncs
        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0'})
        calls = FindNodes(Call).visit(bns['x0_blk0'])
        assert len(calls) == 0
        calls = FindNodes(Call).visit(bns['x1_blk0'])
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_from_innerloop(self, mode):
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid, space_order=4)
        g = Function(name='g', grid=grid, space_order=2)

        eqns = [Eq(g, f.dzl + f.dzr + 1),
                Eq(f, g)]

        op = Operator(eqns, opt=('advanced', {'openmp': False}))

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

        # Also make sure the Call is at the right place in the IET
        assert op.body.body[-1].body[1].body[0].body[0].body[0].body[0].is_Call
        assert op.body.body[-1].body[1].body[0].body[0].body[1].is_Iteration

    @pytest.mark.parallel(mode=2)
    def test_unhoist_haloupdate_if_invariant(self, mode):
        """
        Test an Operator that computes coupled equations in which the first
        one *does require* a halo update on a Dimension-invariant Function.
        """
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)

        f.data_with_halo[:] = 2.
        u.data_with_halo[:] = 1.

        eqns = [Eq(u.forward, u + f[x-1] + f[x+1] + 1.),
                Eq(f, u[t+1, x-1] + u[t+1, x+1] + 1.)]

        op = Operator(eqns)
        op.apply(time=1)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        glb_pos_map = grid.distributor.glb_pos_map
        R = 1e-07
        if LEFT in glb_pos_map[x]:
            assert np.allclose(f.data_ro_domain[:5], [30., 56., 62., 67., 67.], rtol=R)
        else:
            assert np.allclose(f.data_ro_domain[5:], [67., 67., 62., 56., 30.], rtol=R)

    @pytest.mark.parallel(mode=[(2, 'basic'), (2, 'diag')])
    def test_redo_haloupdate_due_to_antidep(self, mode):
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

    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_avoid_fullmode_if_crossloop_dep(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        f.data_with_halo[:] = 0.
        g.data_with_halo[:] = 1.

        op = Operator([Eq(f, g[x, y-1] + g[x, y+1]),
                       Eq(g, f)])

        # Exactly 4 routines will be generated for the basic mode
        assert len(op._func_table) == 4

        # Also check the numerical values
        op.apply()
        assert np.all(f.data[:] == 2.)

    @pytest.mark.parallel(mode=2)
    def test_avoid_haloupdate_if_flowdep_along_other_dim(self, mode):
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        xl = SubDimension.left(name='xl', parent=x, thickness=2)

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        f.data_with_halo[:] = 1.2
        g.data_with_halo[:] = 2.

        # Note: the subdomain is used to prevent the compiler from fusing the
        # third Eq in the first Eq's loop
        eqns = [Eq(f.forward, f[t, x-1] + f[t, x+1]),
                Eq(f[t+1, xl], f[t+1, xl] + 1.),
                Eq(g.forward, f[t, x-1] + f[t, x+1], subdomain=grid.interior)]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        calls = FindNodes(Call).visit(get_time_loop(op))
        assert len(calls) == 1

        op.apply(time_M=1)
        glb_pos_map = f.grid.distributor.glb_pos_map
        R = 1e-07  # Can't use np.all due to rounding error at the tails
        if LEFT in glb_pos_map[x]:
            assert np.allclose(f.data_ro_domain[0, :5], [5.6, 6.8, 5.8, 4.8, 4.8],
                               rtol=R)
            assert np.allclose(g.data_ro_domain[0, :5], [2., 5.8, 5.8, 4.8, 4.8],
                               rtol=R)
        else:
            assert np.allclose(f.data_ro_domain[0, 5:], [4.8, 4.8, 4.8, 4.8, 3.6],
                               rtol=R)
            assert np.allclose(g.data_ro_domain[0, 5:], [4.8, 4.8, 4.8, 4.8, 2.],
                               rtol=R)

    @pytest.mark.parallel(mode=2)
    def test_avoid_merging_if_no_locindices(self, mode):
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = Function(name='f', grid=grid, space_order=1)
        g = TimeFunction(name='g', grid=grid, space_order=1, time_order=1)

        f.data_with_halo[:] = 0.
        g.data_with_halo[:] = 1.

        eqns = [Eq(f, g[t, x+1] + g[t, x-1]),
                Eq(g.forward, f[x-1] + f[x+1])]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        titer = get_time_loop(op)
        assert titer.dim is grid.time_dim
        assert len(titer.nodes[0].body[0].body[0].body[0].body) == 1
        assert titer.nodes[0].body[0].body[0].body[0].body[0].is_Call
        parent = titer.nodes[0].body[0].body[0].body[1]
        if configuration['language'] == 'openmp':
            assert parent.body[0].body[0].is_Iteration
        else:
            assert parent.is_Iteration

        op.apply(time_M=1)

        glb_pos_map = f.grid.distributor.glb_pos_map
        R = 1e-07  # Can't use np.all due to rounding error at the tails
        if LEFT in glb_pos_map[x]:
            assert np.allclose(f.data_ro_domain[:5], [5., 6., 8., 8., 8.], rtol=R)
            assert np.allclose(g.data_ro_domain[0, :5], [6., 13., 14., 16., 16.], rtol=R)
        else:
            assert np.allclose(f.data_ro_domain[5:], [8., 8., 8., 6., 5.], rtol=R)
            assert np.allclose(g.data_ro_domain[0, 5:], [16., 16., 14., 13., 6.], rtol=R)

    @pytest.mark.parallel(mode=1)
    def test_avoid_merging_if_diff_functions(self, mode):
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions

        u = TimeFunction(name="u", grid=grid, space_order=2)
        U = TimeFunction(name="U", grid=grid, space_order=2)
        src = SparseTimeFunction(name="src", grid=grid, nt=3, npoint=1,
                                 coordinates=np.array([(0.5, 0.5, 0.5)]))

        eqns = [Eq(u.forward, u.dx)]
        eqns += src.inject(field=u.forward, expr=src)
        eqns += [Eq(U.forward, U.dx + u.forward)]

        op = Operator(eqns)
        op.cfunction

        check_halo_exchanges(op, 2, 2)

    @pytest.mark.parallel(mode=1)
    def test_merge_haloupdate_if_diff_locindices(self, mode):
        grid = Grid(shape=(101, 101))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = Function(name="f", grid=grid)
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

        cond = ConditionalDimension(name='cond', parent=y, condition=y < 10)

        eqns = [
            Eq(f, u[t, x+2, y]),
            Eq(u.forward, u[t-1, x+2, y], implicit_dims=[cond])
        ]

        op = Operator(eqns)

        assert len(FindNodes(HaloUpdateCall).visit(op)) == 1
        op.cfunction

    @pytest.mark.parallel(mode=2)
    def test_merge_and_hoist_haloupdate_if_diff_locindices(self, mode):
        """
        This test is a revisited, more complex version of
        `test_merge_haloupdate_if_diff_locindices`, also checking hoisting.
        And in addition to checking the generated code, it also checks the
        numerical output.

        In the Operator there are three Eqs:

        * the first one does *not* require a halo update
        * the second one requires a halo update for `f` at `t+1`
        * the third one requires a halo update for `f` at `t`

        Also:

        * the second and third Eqs cannot be fused in the same loop

        In the IET we end up with *two* HaloSpots, one placed before the
        time loop, and one placed before the second Eq. The third Eq,
        reading from f[t0], will seamlessy find its halo up-to-date,
        due to the f[t1] being updated in the previous time iteration.
        """
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid, space_order=2)
        g = TimeFunction(name='g', grid=grid)
        h = TimeFunction(name='h', grid=grid)

        f.data_with_halo[:] = 1.2
        g.data_with_halo[:] = 2.
        h.data_with_halo[:] = 3.1

        # Note: the subdomain is used to prevent the compiler from fusing the
        # third Eq in the second Eq's loop
        eqns = [Eq(f.forward, f + 1.),
                Eq(g.forward, f[t+1, x-1] + f[t+1, x+1]),
                Eq(h.forward, f[t, x-2] + f[t, x+2], subdomain=grid.interior)]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2
        assert calls[0].arguments[3].args[0] is t.symbolic_min

        assert len(FindNodes(HaloUpdateCall).visit(op.body.body[1].body[0])) == 1
        assert len(FindNodes(HaloUpdateCall).visit(op.body.body[1].body[1])) == 1
        assert len(FindNodes(HaloUpdateCall).visit(op.body.body[1].body[2])) == 0

        op.apply(time_M=1)

        glb_pos_map = f.grid.distributor.glb_pos_map
        R = 1e-07  # Can't use np.all due to rounding error at the tails
        if LEFT in glb_pos_map[x]:
            assert np.allclose(g.data_ro_domain[0, :5], [4.4, 6.4, 6.4, 6.4, 6.4], rtol=R)
            assert np.allclose(h.data_ro_domain[0, :5], [3.1, 3.4, 4.4, 4.4, 4.4], rtol=R)
        else:
            assert np.allclose(g.data_ro_domain[0, 5:], [6.4, 6.4, 6.4, 6.4, 4.4], rtol=R)
            assert np.allclose(h.data_ro_domain[0, 5:], [4.4, 4.4, 4.4, 3.4, 3.1], rtol=R)

    @pytest.mark.parallel(mode=1)
    def test_merge_haloupdate_if_diff_but_equivalent_locindices(self, mode):
        grid = Grid(shape=(65, 65, 65))

        v1 = TimeFunction(name='v1', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))
        v2 = TimeFunction(name='v2', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))

        rec = SparseTimeFunction(name='rec', grid=grid, nt=500, npoint=65)  # noqa

        eqns = [Eq(v1.forward, v2.laplace),
                Eq(v2.forward, v1.forward.laplace + v2),
                rec.interpolate(expr=v1)]

        op = Operator(eqns)
        op.cfunction

        calls, _ = check_halo_exchanges(op, 2, 2)
        for i, v in enumerate([v2, v1]):
            assert calls[i].arguments[0] is v

    @pytest.mark.parallel(mode=1)
    @switchconfig(autopadding=True)
    def test_process_but_avoid_haloupdate_along_replicated(self, mode):
        dx = Dimension(name='dx')
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        u = TimeFunction(name='u', grid=grid, space_order=4)
        c = Function(name='c', grid=grid, dimensions=(x, dx), shape=(10, 5))

        cases = [
            Eq(u.forward, (u.dx*c).dx + 1),
            Eq(u.forward, (u.dx*c[x, 0]).dx + 1)
        ]

        for eq in cases:
            op = Operator(eq, opt=('advanced', {'cire-mingain': 1}))

            calls = [i for i in FindNodes(Call).visit(op)
                     if isinstance(i, HaloUpdateCall)]
            assert len(calls) == 1
            assert calls[0].arguments[0] is u

    @pytest.mark.parallel(mode=1)
    def test_conditional_dimension(self, mode):
        """
        Test the case of Functions in the condition of a ConditionalDimension.
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = TimeFunction(name='h', grid=grid, space_order=2)

        cd = ConditionalDimension(name='cd', parent=x, condition=~Ne(g, h[t, x+1, y]))

        eqns = [Eq(f.forward, f + 1, implicit_dims=cd),
                Eq(g, h + 1)]

        op = Operator(eqns)

        # No halo update here because the `x` Iteration is SEQUENTIAL
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_conditional_dimension_v2(self, mode):
        """
        Make sure optimizations don't move around halo exchanges if embedded
        within conditionals.
        """
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, space_order=2)
        h = TimeFunction(name='h', grid=grid, space_order=2)

        cd0 = ConditionalDimension(name='cd0', parent=time, condition=time <= 2)
        cd1 = ConditionalDimension(name='cd1', parent=time, condition=time > 2)

        eqns = [Eq(f.forward, f.dx2 + 1, implicit_dims=cd0),
                Eq(h.forward, h.dx2 + 1, implicit_dims=cd1)]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2
        conds = FindNodes(Conditional).visit(op)
        assert len(conds) == 2
        assert all(isinstance(i.then_body[0].body[0].body[0].body[0],
                              HaloUpdateList) for i in conds)

    @pytest.mark.parametrize('expr,expected', [
        ('f[t,x-1,y] + f[t,x+1,y]', {'rc', 'lc'}),
        ('f[t,x,y-1] + f[t,x,y+1]', {'cr', 'cl'}),
        ('f[t,x-1,y-1] + f[t,x,y+1]', {'cr', 'rr', 'rc', 'cl'}),
        ('f[t,x-1,y-1] + f[t,x+1,y+1]', {'cr', 'rr', 'rc', 'cl', 'll', 'lc'}),
    ])
    @pytest.mark.parallel(mode=[(1, 'diag')])
    def test_diag_comm_scheme(self, expr, expected, mode):
        """
        Check that the 'diag' mode does not generate more communications
        than strictly necessary.
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa
        t = grid.stepping_dim  # noqa

        f = TimeFunction(name='f', grid=grid)  # noqa

        op = Operator(Eq(f.forward, eval(expr)), opt=('advanced', {'openmp': False}))

        calls = FindNodes(Call).visit(op._func_table['haloupdate0'])
        destinations = {str(i.arguments[-2].field) for i in calls}
        assert destinations == expected

    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_poke_progress(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        eqn = Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + f[t, x, y-1] + f[t, x, y+1])
        op = Operator(eqn, opt=('advanced', {'par-dynamic-work': 0}))

        trees = retrieve_iteration_tree(op._func_table['compute0'].root)
        assert len(trees) == 2
        tree = trees[0]
        # Make sure `pokempi0` is the last node within the outer Iteration
        assert len(tree) == 2
        assert len(tree.root.nodes) == 2
        call = tree.root.nodes[1]
        assert call.name == 'pokempi0'
        assert call.arguments[0].name == 'msg0'
        if configuration['language'] == 'openmp':
            # W/ OpenMP, we prod until all comms have completed
            assert call.then_body[0].body[0].is_While
            # W/ OpenMP, we expect dynamic thread scheduling
            assert 'dynamic,1' in tree.root.pragmas[0].ccode.value
        else:
            # W/o OpenMP, it's a different story
            assert call._single_thread

        # Now we do as before, but enforcing loop blocking (by default off,
        # as heuristically it is not enabled when the Iteration nest has depth < 3)
        op = Operator(eqn, opt=('advanced', {'blockinner': True, 'par-dynamic-work': 0}))

        bns, _ = assert_blocking(op._func_table['compute0'].root, {'x0_blk0'})

        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 2
        tree = trees[1]
        # Make sure `pokempi0` is the last node within the inner Iteration over blocks
        assert len(tree) == 2
        assert len(tree.root.nodes[0].nodes) == 2
        call = tree.root.nodes[0].nodes[1]
        assert call.name == 'pokempi0'
        assert call.arguments[0].name == 'msg0'
        if configuration['language'] == 'openmp':
            # W/ OpenMP, we prod until all comms have completed
            assert call.then_body[0].body[0].is_While
            # W/ OpenMP, we expect dynamic thread scheduling
            assert 'dynamic,1' in tree.root.pragmas[0].ccode.value
        else:
            # W/o OpenMP, it's a different story
            assert call._single_thread

    @pytest.mark.parallel(mode=[(1, 'diag2')])
    def test_diag2_quality(self, mode):
        grid = Grid(shape=(10, 10, 10))

        f = TimeFunction(name='f', grid=grid, space_order=2)

        eqn = Eq(f.forward, f.dx2 + 1.)

        op = Operator(eqn)

        assert len(op._func_table) == 4  # gather, scatter, haloupdate, halowait
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2
        assert calls[0].name == 'haloupdate0'
        assert calls[1].name == 'halowait0'
        assert_blocking(op, {'x0_blk0'})

    @pytest.mark.parallel(mode=[
        (1, 'basic'),
        (1, 'diag'),
        (1, 'overlap'),
        (1, 'overlap2'),
        (1, 'diag2'),
        (1, 'full'),
    ])
    def test_min_code_size(self, mode):
        grid = Grid(shape=(10, 10, 10))

        f = TimeFunction(name='f', grid=grid, space_order=2)
        g = TimeFunction(name='g', grid=grid, space_order=2)

        eqns = [Eq(f.forward, f.dx2 + 1.),
                Eq(g.forward, g.dx2 + 1.)]

        op = Operator(eqns)

        op.cfunction

        calls = FindNodes(Call).visit(op)

        if configuration['mpi'] in ('basic', 'diag'):
            assert len(op._func_table) == 4  # gather, scatter, sendrecv, haloupdate
            assert len(calls) == 1
            assert calls[0].name == 'haloupdate0'
            assert calls[0].ncomps == 2
        elif configuration['mpi'] in ('overlap'):
            assert len(op._func_table) == 8
            assert len(calls) == 4  # haloupdate, compute, halowait, remainder
            assert 'haloupdate1' not in op._func_table
        elif configuration['mpi'] in ('overlap2'):
            assert len(op._func_table) == 6
            assert len(calls) == 4  # haloupdate, compute, halowait, remainder
            assert 'haloupdate1' not in op._func_table
        elif configuration['mpi'] in ('diag2'):
            assert len(op._func_table) == 4
            assert len(calls) == 2
            assert calls[0].name == 'haloupdate0'
            assert calls[0].ncomps == 2
            assert calls[1].name == 'halowait0'
        elif configuration['mpi'] in ('full'):
            assert len(op._func_table) == 7
            assert len(calls) == 4
            assert 'haloupdate1' not in op._func_table
            assert len(FindNodes(ComputeCall).visit(op)) == 1

    @pytest.mark.parallel(mode=[(1, 'diag2')])
    def test_many_functions(self, mode):
        grid = Grid(shape=(10, 10, 10))

        eqns = []
        for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
            w = TimeFunction(name=i, grid=grid, space_order=2)
            eqns.append(Eq(w.forward, w.dx + 1.))

        op = Operator(eqns)

        op.cfunction

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2
        assert calls[0].ncomps == 7

    @switchconfig(profiling='advanced2')
    @pytest.mark.parallel(mode=[
        (1, 'full'),
    ])
    def test_profiled_regions(self, mode):
        grid = Grid(shape=(10, 10, 10))

        f = TimeFunction(name='f', grid=grid, space_order=2)
        g = TimeFunction(name='g', grid=grid, space_order=2)

        eqns = [Eq(f.forward, f.dx2 + 1.),
                Eq(g.forward, g.dx2 + 1.)]

        op = Operator(eqns)
        assert op._profiler.all_sections == ['section0', 'haloupdate0', 'halowait0',
                                             'remainder0', 'compute0']

    @pytest.mark.parallel(mode=1)
    def test_enforce_haloupdate_if_unwritten_function(self, mode):
        grid = Grid(shape=(16, 16))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        w = TimeFunction(name='w', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=10, space_order=4)

        eqns = [Eq(w.forward, v.forward.dx + w + 1., subdomain=grid.interior),
                Eq(u.forward, u + 1.),
                Eq(v.forward, u.forward + usave.dx4, subdomain=grid.interior)]

        key = lambda f: f is not usave

        op = Operator(eqns, opt=('advanced', {'dist-drop-unwritten': key}))

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2   # One for `v` and one for `usave`

    @pytest.mark.parallel(mode=1)
    def test_haloupdate_buffer1(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        u = TimeFunction(name='u', grid=grid, time_order=1, save=Buffer(1))
        v = TimeFunction(name='v', grid=grid, time_order=1, save=Buffer(1))

        eqns = [Eq(u.forward, div(v) + 1.),
                Eq(v.forward, div(u.forward) + 1.)]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        # There should be two separate calls
        # halo(v), eq_u, halo_u, eq(v)
        assert len(calls) == 2

        # Also ensure the compiler is doing its job removing unnecessary
        # ModuloDimensions
        assert len([i for i in FindSymbols('dimensions').visit(op) if i.is_Modulo]) == 0

    @pytest.mark.parallel(mode=1)
    @pytest.mark.parametrize('sz,fwd,expr,exp0,exp1,args', [
        (1, True, 'rec.interpolate(v2)', 3, 2, ('v1', 'v2')),
        (1, True, 'Eq(v3.forward, v2.laplace + 1)', 1, 1, ('v2',)),
        (1, True, 'Eq(v3.forward, v2.forward.laplace + 1)', 3, 2, ('v1', 'v2',)),
        (2, True, 'Eq(v3.forward, v2.forward.laplace + 1)', 3, 2, ('v1', 'v2',)),
        (1, False, 'rec.interpolate(v2)', 3, 2, ('v1', 'v2')),
        (1, False, 'Eq(v3.backward, v2.laplace + 1)', 1, 1, ('v2',)),
        (1, False, 'Eq(v3.backward, v2.backward.laplace + 1)', 3, 2, ('v1', 'v2',)),
        (2, False, 'Eq(v3.backward, v2.backward.laplace + 1)', 3, 2, ('v1', 'v2',)),
    ])
    def test_haloupdate_buffer_cases(self, sz, fwd, expr, exp0, exp1, args, mode):
        grid = Grid((65, 65, 65), topology=('*', 1, '*'))

        v1 = TimeFunction(name='v1', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))
        v2 = TimeFunction(name='v2', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))
        v3 = TimeFunction(name='v3', grid=grid, space_order=2, time_order=1,  # noqa
                          save=Buffer(1))

        rec = SparseTimeFunction(name='rec', grid=grid, nt=500, npoint=65)  # noqa

        if fwd:
            eqns = [Eq(v1.forward, v2.laplace + v1),
                    Eq(v2.forward, v1.forward.dx2 + v2),
                    eval(expr)]
        else:
            eqns = [Eq(v1.backward, v2.laplace + v1),
                    Eq(v2.backward, v1.backward.dx2 + v2),
                    eval(expr)]

        op = Operator(eqns)
        op.cfunction

        calls, _ = check_halo_exchanges(op, exp0, exp1)
        for i, v in enumerate(args):
            assert calls[i].arguments[0] is eval(v)

    @pytest.mark.parallel(mode=1)
    def test_avoid_hoisting_if_antidep(self, mode):
        grid = Grid(shape=(65, 65, 65))

        v1 = TimeFunction(name='v1', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))
        v2 = TimeFunction(name='v2', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))
        v3 = TimeFunction(name='v3', grid=grid, space_order=2, time_order=1,
                          save=Buffer(1))

        eqns = [Eq(v1, v2.laplace + v3),
                Eq(v2, v1.dx2 + v2),
                Eq(v3, v2.laplace + v1)]

        op = Operator(eqns)
        op.cfunction

        calls, _ = check_halo_exchanges(op, 3, 2)
        # More specifically, we ensure HaloSpot(v2) is on the last loop nest
        assert calls[0].arguments[0] is v1
        assert calls[1].arguments[0] is v2

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_if_in_the_middle(self, mode):
        grid = Grid(shape=(65, 65, 65))

        v1 = TimeFunction(name='v1', grid=grid, space_order=2, time_order=1)
        v2 = TimeFunction(name='v2', grid=grid, space_order=2, time_order=1)

        rec = SparseTimeFunction(name='rec', grid=grid, nt=500, npoint=65)  # noqa

        eqns = [Eq(v1.forward, v2.laplace),
                Eq(v2.forward, v1.laplace + v2),
                rec.interpolate(expr=v1.forward)]

        op = Operator(eqns)
        op.cfunction

        calls, _ = check_halo_exchanges(op, 3, 2)
        assert calls[0].arguments[0] is v2
        assert calls[1].arguments[0] is v1

    @pytest.mark.parallel(mode=2)
    def test_merge_smart_if_within_conditional(self, mode):
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        t_sub = ConditionalDimension(name='t_sub', parent=time, factor=3)

        f = TimeFunction(name='f', grid=grid, space_order=4)
        s = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=100)

        eq = Eq(f.forward, f.laplace + .002)

        rec = s.interpolate(expr=f, implicit_dims=t_sub)

        op = Operator(rec + [eq])

        # Check generated code -- the halo exchange is expected at the top of
        # the time loop, outside of any conditional
        calls, tloop = check_halo_exchanges(op, 1, 1)
        assert tloop.nodes[0].body[0].body[0] is calls[0]

        op.apply(time_M=3)

        assert np.isclose(norm(f), 254292.75, atol=1e-1, rtol=0)
        assert np.isclose(norm(s), 191.44644, atol=1e-1, rtol=0)

        # Analogous to above, but going backwards in time
        eq1 = Eq(f.backward, f.laplace + .002)

        op1 = Operator(rec + [eq1])
        op1.cfunction

        check_halo_exchanges(op1, 1, 1)

        # Check that the halo update is still at the top of the time loop and
        # so outside of any conditional
        for n in FindNodes(Conditional).visit(op1):
            assert len(FindNodes(HaloUpdateCall).visit(n)) == 0

    @pytest.mark.parallel(mode=2)
    def test_allreduce_time(self, mode):
        space_order = 8
        nx, ny = 11, 11

        grid = Grid(shape=(nx, ny))
        tt = grid.time_dim
        nt = 10

        ux = TimeFunction(name="ux", grid=grid, time_order=1, space_order=space_order)
        g = TimeFunction(name="g", grid=grid, dimensions=(tt, ), shape=(nt,))

        op = Operator([Eq(ux.forward, ux + tt), Inc(g, ux)], name="Op")
        assert_structure(op, ['t,x,y', 't'], 'txy')

        # Reduce should be in time loop but not in space loop
        iters = FindNodes(Iteration).visit(op)
        for i in iters:
            if i.dim.is_Time:
                assert len(FindNodes(Call).visit(i)) == 1  # one allreduce
            else:
                assert len(FindNodes(Call).visit(i)) == 0

        op.apply(time_m=0, time_M=nt-1)
        assert np.isclose(np.max(g.data), 4356.0)

    @pytest.mark.parallel(mode=2)
    def test_multi_allreduce_time(self, mode):
        space_order = 8
        nx, ny = 11, 11

        grid = Grid(shape=(nx, ny))
        tt = grid.time_dim
        nt = 10

        ux = TimeFunction(name="ux", grid=grid, time_order=1, space_order=space_order)
        g = TimeFunction(name="g", grid=grid, dimensions=(tt, ), shape=(nt,))
        h = TimeFunction(name="h", grid=grid, dimensions=(tt, ), shape=(nt,))

        op = Operator([Eq(ux.forward, ux + tt), Inc(g, ux), Inc(h, ux)], name="Op")
        assert_structure(op, ['t,x,y', 't'], 'txy')

        # Make sure the two allreduce calls are in the time the loop
        iters = FindNodes(Iteration).visit(op)
        for i in iters:
            if i.dim.is_Time:
                assert len(FindNodes(Call).visit(i)) == 2  # Two allreduce
            else:
                assert len(FindNodes(Call).visit(i)) == 0

        op.apply(time_m=0, time_M=nt-1)
        assert np.isclose(np.max(g.data), 4356.0)
        assert np.isclose(np.max(h.data), 4356.0)

    @pytest.mark.parallel(mode=2)
    def test_multi_allreduce_time_cond(self, mode):
        space_order = 8
        nx, ny = 11, 11

        grid = Grid(shape=(nx, ny))
        tt = grid.time_dim
        nt = 20
        ct = ConditionalDimension(name="ct", parent=tt, factor=2)

        ux = TimeFunction(name="ux", grid=grid, time_order=1, space_order=space_order)
        g = TimeFunction(name="g", grid=grid, dimensions=(ct, ), shape=(int(nt/2),),
                         time_dim=ct)
        h = TimeFunction(name="h", grid=grid, dimensions=(ct, ), shape=(int(nt/2),),
                         time_dim=ct)

        op = Operator([Eq(g, 0), Eq(ux.forward, tt), Inc(g, ux), Inc(h, ux)], name="Op")
        assert_structure(op, ['t', 't,x,y', 't,x,y'], 'txyxy')

        # Make sure the two allreduce calls are in the time the loop
        iters = FindNodes(Iteration).visit(op)
        for i in iters:
            if i.dim.is_Time:
                assert len(FindNodes(Call).visit(i)) == 2  # Two allreduce
            else:
                assert len(FindNodes(Call).visit(i)) == 0

        op.apply(time_m=0, time_M=nt-1)

        expected = [nx * ny * max(t-1, 0) for t in range(0, nt, 2)]
        assert np.allclose(g.data, expected)
        assert np.allclose(h.data, expected)

    @pytest.mark.parallel(mode=2)
    def test_allreduce_multicond(self, mode):
        space_order = 8
        nx, ny = 11, 11

        grid = Grid(shape=(nx, ny))
        tt = grid.time_dim
        nt = 20
        ct = ConditionalDimension(name="ct", parent=tt, factor=2)
        ct2 = ConditionalDimension(name="ct2", parent=tt, factor=8)

        ux = TimeFunction(name="ux", grid=grid, time_order=1, space_order=space_order)
        uy = TimeFunction(name="uy", grid=grid, time_order=1, space_order=space_order)
        g = TimeFunction(name="g", grid=grid, dimensions=(ct, ), shape=(int(nt/2),),
                         time_dim=ct)
        h = TimeFunction(name="h", grid=grid, dimensions=(ct, ), shape=(int(nt/2),),
                         time_dim=ct)

        op = Operator([Eq(g, 0), Eq(ux.forward, tt), Inc(g, ux), Inc(h, ux),
                       Eq(uy, tt, implicit_dims=ct2)],
                      name="Op")
        assert_structure(op, ['t', 't,x,y', 't,x,y', 't,x,y'], 'txyxyxy')

        # Make sure the two allreduce calls are in the time the loop
        iters = FindNodes(Iteration).visit(op)
        for i in iters:
            if i.dim.is_Time:
                assert len(FindNodes(Call).visit(i)) == 2  # Two allreduce
            else:
                assert len(FindNodes(Call).visit(i)) == 0

        # Check conditionals
        conds = FindNodes(Conditional).visit(op)
        assert len(conds) == 3
        # First one is just g initialization
        sym0 = FindSymbols().visit(conds[0])
        assert set(sym0) == {ct.symbolic_factor, tt, g}
        assert grid.distributor._obj_comm not in sym0
        # Second one is g and h and allreduce
        sym1 = FindSymbols().visit(conds[1])
        assert g in sym1
        assert h in sym1
        assert ux in sym1
        # The allreduce
        assert grid.distributor._obj_comm in sym1
        # Last one is only uy
        sym2 = FindSymbols().visit(conds[-1])
        assert g not in sym2
        assert h not in sym2
        assert uy in sym2
        assert grid.distributor._obj_comm not in sym2

        op.apply(time_m=0, time_M=nt-1)

        expected = [nx * ny * max(t-1, 0) for t in range(0, nt, 2)]
        assert np.allclose(g.data, expected)
        assert np.allclose(h.data, expected)


class TestOperatorAdvanced:

    @pytest.mark.parallel(mode=4)
    def test_injection_wodup(self, mode):
        """
        Test injection operator when the sparse points don't need to be replicated
        ("wodup" -> w/o duplication) over multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        f = Function(name='f', grid=grid, space_order=1)
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
    @switchconfig(condition=isinstance(configuration['compiler'],
                  (OneapiCompiler)), safe_math=True)
    def test_injection_wodup_wtime(self, mode):
        """
        Just like ``test_injection_wodup``, but using a SparseTimeFunction
        instead of a SparseFunction. Hence, the data scattering/gathering now
        has to correctly pack/unpack multidimensional arrays.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        save = 3
        f = TimeFunction(name='f', grid=grid, save=save, space_order=1)
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
    def test_injection_dup(self, mode):
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
    def test_interpolation_wodup(self, mode):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        f = Function(name='f', grid=grid, space_order=1)
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
    def test_interpolation_dup(self, mode):
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
    def test_subsampling(self, mode):
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
    def test_arguments_subrange(self, mode):
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
    def test_bcs_basic(self, mode):
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
    def test_interior_w_stencil(self, mode):
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
    def test_misc_dims(self, mode):
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
    def test_nontrivial_operator(self, mode):
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

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'overlap'), (4, 'full')])
    def test_coupled_eqs_mixed_dims(self, mode):
        """
        Test an Operator that computes coupled equations over partly disjoint sets
        of Dimensions (e.g., one Eq over [x, y, z], the other Eq over [x, yi, zi]).
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        xi, yi = grid.interior.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=2)
        v = TimeFunction(name='v', grid=grid, space_order=2)

        u.data_with_halo[:] = 1.

        eqns = [Eq(u[t+1, x, y], u[t, x-1, y] + u[t, x, y] + u[t, x+1, y] + v + 1),
                Eq(v[t+1, x, yi],
                   (v[t, x, yi] + u[t, x, yi-1] + u[t, x, yi] + u[t, x, yi+1] + 1))]

        # `u`'s stencil is
        #
        #   *
        # * C *
        #   *
        #
        # Where C is a generic point (x, y) and * are the accessed neighbours

        op = Operator(eqns)
        op.apply(time=0)

        assert np.all(u.data_ro_domain[1] == 4.)

        glb_pos_map = v.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[0, 4], [0, 4]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[4, 0], [4, 0]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[0, 4], [0, 4]])
        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[4, 0], [4, 0]])

        # Same checks as above, but exploiting the user API
        assert np.all(v.data_ro_domain[1, :, 0] == 0.)
        assert np.all(v.data_ro_domain[1, :, 1] == 4.)
        assert np.all(v.data_ro_domain[1, :, 2] == 4.)
        assert np.all(v.data_ro_domain[1, :, 3] == 0.)

    @pytest.mark.parallel(mode=2)
    def test_haloupdate_same_timestep(self, mode):
        """
        Test an Operator that computes coupled equations in which the second
        one requires a halo update right after the computation of the first one.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.
        v = TimeFunction(name='v', grid=grid)
        v.data_with_halo[:] = 0.

        eqns = [Eq(u.forward, u + v + 1.),
                Eq(v.forward, u[t+1, x, y-1] + u[t+1, x, y] + u[t+1, x, y+1])]

        op = Operator(eqns)
        op.apply(time=0)

        assert np.all(v.data_ro_domain[-1, :, 1:-1] == 6.)

    @pytest.mark.parallel(mode=2)
    def test_haloupdate_same_timestep_v2(self, mode):
        """
        Similar to test_haloupdate_same_timestep, but switching the expression that
        writes to subsequent time step. Also checks halo update call placement.
        MFE for issue #1483
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.
        v = TimeFunction(name='v', grid=grid)
        v.data_with_halo[:] = 0.

        eqns = [Eq(u, u + v + 1.),
                Eq(v.forward, u[t, x, y-1] + u[t, x, y] + u[t, x, y+1])]

        op = Operator(eqns)

        titer = op.body.body[-1].body[0]
        assert titer.dim is grid.time_dim
        assert titer.nodes[0].body[0].body[0].is_List
        assert len(titer.nodes[0].body[0].body[0].body[0].body) == 1
        assert titer.nodes[0].body[0].body[0].body[0].body[0].is_Call

        op.apply(time=0)

        assert np.all(v.data_ro_domain[-1, :, 1:-1] == 6.)

    @pytest.mark.parallel(mode=4)
    def test_haloupdate_multi_op(self, mode):
        """
        Test that halo updates are carried out correctly when multiple operators
        are applied consecutively.
        """
        a = np.arange(64).reshape((8, 8))
        grid = Grid(shape=a.shape, extent=(8, 8))

        so = 3
        dims = grid.dimensions
        f = Function(name='f', grid=grid, space_order=so)
        f.data[:] = a

        fo = Function(name='fo', grid=grid, space_order=so)

        for d in dims:
            rhs = generic_derivative(f, d, so, 1)
            expr = Eq(fo, rhs)
            op = Operator(expr)
            op.apply()
            f.data[:, :] = fo.data[:, :]

        assert (np.isclose(norm(f), 17.86754, atol=1e-4, rtol=0))

    @pytest.mark.parallel(mode=1)
    def test_haloupdate_issue_1613(self, mode):
        """
        Test the HaloScheme construction and generation when using u.dt2.

        This stems from issue #1613.
        """
        configuration['mpi'] = True

        grid = Grid(shape=(10, 10))
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=2)

        eqns = [Eq(u.forward, u.dt2 + u.dx)]

        op = Operator(eqns)

        # The loc_indices must be along `t` (ie `t0`)
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        dims = [i for i in calls[0].arguments if isinstance(i, Dimension)]
        assert len(dims) == 1
        assert dims[0].is_Modulo
        assert dims[0].origin is t

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag2'), (4, 'overlap2')])
    def test_cire(self, mode):
        """
        Check correctness when the DSE extracts aliases and places them
        into offset-ed loop (nest). For example, the compiler may generate:

            for i = i_m - 1 to i_M + 1
              tmp[i] = f(a[i-1], a[i], a[i+1], ...)
            for i = i_m to i_M
              u[i] = g(tmp[i-1], tmp[i], ... a[i], ...)

        If the employed MPI scheme doesn't use comp/comm overlap (i.e., `basic`,
        `diag`), then it's not so different than most of the other tests seen in
        this module. However, with comp/comm overlap, which exploits the same loops
        to compute the boundary ("OWNED") regions, the situation is more delicate.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        eqn = Eq(u.forward, _R(_R(u[t, x, y] + u[t, x+1, y+1])*3.*f +
                               _R(u[t, x+2, y+2] + u[t, x+3, y+3])*3.*f) + 1.)
        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('advanced', {'cire-mingain': 0,
                                              'cire-schedule': 1}))

        assert len([i for i in FindSymbols().visit(op1.body) if i.is_Array]) == 1

        op0(time_M=1)
        u0_norm = norm(u)

        u._data_with_inhalo[:] = 0.
        op1(time_M=1)
        u1_norm = norm(u)

        assert u0_norm == u1_norm

    @pytest.mark.parallel(mode=[(4, 'overlap2'), (4, 'diag2')])
    def test_cire_with_shifted_diagonal_halo_touch(self, mode):
        """
        Like ``test_cire`` but now the diagonal halos required to compute
        the aliases are shifted due to the iteration space. Basically, this
        is checking that ``TimedAccess.touched_halo`` does the right thing
        using the information stored in ``.intervals``.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        eqn = Eq(u.forward, _R(_R(u[t, x, y] + u[t, x+2, y])*3.*f +
                               _R(u[t, x+1, y+1] + u[t, x+3, y+1])*3.*f) + 1.)
        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('advanced', {'cire-mingain': 0,
                                              'cire-schedule': 1}))

        assert len([i for i in FindSymbols().visit(op1.body) if i.is_Array]) == 1

        op0(time_M=1)
        u0_norm = norm(u)

        u._data_with_inhalo[:] = 0.
        op1(time_M=1)
        u1_norm = norm(u)

        assert u0_norm == u1_norm

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('opt_options', [
        {'cire-rotate': True},  # Issue #1490
        {'min-storage': True},  # Issue #1491
    ])
    def test_cire_options(self, opt_options, mode):
        """
        MFEs for issues #1490 and #1491.
        """
        grid = Grid(shape=(128, 128, 128), dtype=np.float64)

        p = TimeFunction(name='p', grid=grid, time_order=2, space_order=8)
        p1 = TimeFunction(name='p', grid=grid, time_order=2, space_order=8)

        p.data[0, 40:80, 40:80, 40:80] = 0.12
        p1.data[0, 40:80, 40:80, 40:80] = 0.12

        eqn = Eq(p.forward, (p.dx).dx + (p.dy).dy + (p.dz).dz)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('advanced', opt_options))

        # Check generated code
        bns, _ = assert_blocking(op1, {'x0_blk0'})
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 3
        assert 'haloupdate0' in op1._func_table
        # We expect exactly one halo exchange
        calls = FindNodes(MPICall).visit(op1)
        assert len(calls) == 1
        assert calls[0].name == 'haloupdate0'

        op0.apply(time_M=1)
        op1.apply(time_M=1, p=p1)

        # TODO: we will tighten the tolerance, or switch to single precision,
        # or both, once issue #1438 is fixed
        assert np.allclose(p.data, p1.data, rtol=10e-11)

    @pytest.mark.parallel(mode=[(4, 'full')])
    def test_staggering(self, mode):
        """
        Test MPI in presence of staggered grids.

        The equations are semantically meaningless, but they are designed to
        generate the kind of loop nest structure which is typical of *-elastic
        problems (e.g., visco-elastic).
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions

        so = 2
        ux = TimeFunction(name='ux', grid=grid, staggered=x, space_order=so)
        uxx = TimeFunction(name='uxx', grid=grid, staggered=NODE, space_order=so)
        uxy = TimeFunction(name='uxy', grid=grid, staggered=(x, y), space_order=so)

        eqns = [Eq(ux.forward, ux + 0.2*uxx.dx + uxy.dy + 0.5),
                Eq(uxx.forward, uxx + ux.forward.dx + ux.forward.dy + 1.),
                Eq(uxy.forward, 40.*uxy + ux.forward.dx + ux.forward.dy + 3.)]

        op = Operator(eqns)

        op(time_M=2)
        # Expected norms computed "manually" from sequential runs
        assert np.isclose(norm(ux), 7003.098, rtol=1.e-4)
        assert np.isclose(norm(uxx), 78902.21, rtol=1.e-4)
        assert np.isclose(norm(uxy), 71852.62, rtol=1.e-4)

    @pytest.mark.parallel(mode=2)
    def test_op_new_dist(self, mode):
        """
        Test that an operator made with one distributor produces correct results
        when executed with a different distributor.
        """
        grid = Grid(shape=(10, 10), comm=MPI.COMM_SELF)
        grid2 = Grid(shape=(10, 10), comm=MPI.COMM_WORLD)

        u = TimeFunction(name='u', grid=grid, space_order=2)
        u2 = TimeFunction(name='u2', grid=grid2, space_order=2)

        x, y = np.ix_(np.linspace(-1, 1, grid.shape[0]),
                      np.linspace(-1, 1, grid.shape[1]))
        dx = x - 0.5
        dy = y
        u.data[0, :, :] = 1.0 * ((dx*dx + dy*dy) < 0.125)
        u2.data[0, :, :] = 1.0 * ((dx*dx + dy*dy) < 0.125)

        # Create some operator that requires MPI communication
        eqn = Eq(u.forward, u + 0.000001 * u.laplace)
        op = Operator(eqn)

        op.apply(u=u, time_M=10)
        op.apply(u=u2, time_M=10)

        assert abs(norm(u) - norm(u2)) < 1.e-3

    @pytest.mark.parallel(mode=[(4, 'full')])
    def test_misc_subdims(self, mode):
        """
        Test MPI full mode with an Operator having:

            * A middle SubDimension in which at least one of the extremes has
              thickness 0;
            * A left SubDimension.

        Thus, only one of the two distributed Dimensions (x and y) induces
        a halo exchange

        Derived from issue https://github.com/devitocodes/devito/issues/1121
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.

        xi = SubDimension.middle(name='xi', parent=x, thickness_left=0, thickness_right=1)
        yl = SubDimension.left(name='yl', parent=y, thickness=1)

        # A 5 point stencil expression
        eqn = Eq(u[t+1, xi, yl], (u[t, xi, yl] + u[t, xi-1, yl] + u[t, xi+1, yl] +
                                  u[t, xi, yl-1] + u[t, xi, yl+1]))

        op = Operator(eqn)

        # Halo exchanges metadata check-up
        msgs = [i for i in op.parameters if i.name.startswith('msg')]
        assert len(msgs) == 1
        msg = msgs.pop()
        assert len(msg.halos) == 2

        op(time_M=0)

        # Also try running it
        assert np.all(u.data[1, :-1, :1] == 5.)
        assert np.all(u.data[1, -1:] == 1.)
        assert np.all(u.data[1, :, 1:] == 1.)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'full')])
    def test_misc_subdims_3D(self, mode):
        """
        Test `SubDims` in 3D (so that spatial blocking is introduced).

        Derived from issue https://github.com/devitocodes/devito/issues/1309
        """
        grid = Grid(shape=(12, 12, 12))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.

        xi = SubDimension.middle(name='xi', parent=x, thickness_left=2, thickness_right=2)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=2, thickness_right=2)
        zi = SubDimension.middle(name='zi', parent=z, thickness_left=2, thickness_right=2)

        # A 7 point stencil expression
        eqn = Eq(u[t+1, xi, yi, zi], (u[t, xi, yi, zi]
                                      + u[t, xi-1, yi, zi] + u[t, xi+1, yi, zi]
                                      + u[t, xi, yi-1, zi] + u[t, xi, yi+1, zi]
                                      + u[t, xi, yi, zi-1] + u[t, xi, yi, zi+1]))

        op = Operator(eqn)

        op(time_M=0)

        # Also try running it
        assert np.all(u.data[1, 2:-2, 2:-2, 2:-2] == 7.)
        assert np.all(u.data[1, 0:2, :, :] == 1.)
        assert np.all(u.data[1, -2:, :, :] == 1.)
        assert np.all(u.data[1, :, 0:2, :] == 1.)
        assert np.all(u.data[1, :, -2:, :] == 1.)
        assert np.all(u.data[1, :, :, 0:2] == 1.)
        assert np.all(u.data[1, :, :, -2:] == 1.)

    @pytest.mark.parallel(mode=[(4, 'full')])
    def test_custom_subdomain(self, mode):
        """
        This test uses a custom SubDomain such that we end up with two loop
        nests with a data dependence across them inducing two halo exchanges,
        one for each loop nests. A crucial aspect of this test is that the data
        dependence is across a Dimension (xl) that does *not* require a halo
        exchange (xl is a local SubDimension). Unlike more typical cases of
        when a dependence occurs along time, here the dependence distance goes
        to infinity (there are indeed two consecutive separate loop nests), so
        a halo exchange *is* required even though the halo Dimension (y) is not
        the one inducing the dependence.
        """

        class mydomain(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 2), y: y}

        mydomain = mydomain()

        grid = Grid(shape=(8, 8), subdomains=mydomain)

        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid)

        u.data[:] = 0.6
        v.data[:] = 0.4

        eqns = [Eq(u.forward, u[t, x, y-1] + 1, subdomain=grid.subdomains['d1']),
                Eq(v.forward, u[t+1, x+1, y+1] + 1, subdomain=grid.subdomains['d1'])]

        op = Operator(eqns, subs=grid.spacing_map)

        # We expect 2 halo-exchange sets of calls, for a total of 8 calls
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 8

        # Compilation + run
        op(time_M=4)

        # Check numerical values
        assert np.isclose(norm(u), 23.70654, atol=1e-5, rtol=0)
        assert np.isclose(norm(v), 21.14994, atol=1e-5, rtol=0)

    @pytest.mark.parallel(mode=2)
    def test_overriding_from_different_grid(self, mode):
        """
        MFE for issue #1629.
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        xi = SubDimension.middle(name='xi', parent=x, thickness_left=3, thickness_right=3)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=3, thickness_right=3)
        u = TimeFunction(name='u', grid=grid, space_order=2, time_order=0)

        eqn = Eq(u.forward, u + 1).subs({x: xi, y: yi})
        op = Operator(eqn)

        grid2 = Grid(shape=(10, 10), dimensions=(x, y))
        u2 = TimeFunction(name='u', grid=grid2, space_order=2, time_order=0)

        op.apply(time_M=0, u=u2)
        assert np.all(u2.data[0, 3:-3, 3:-3] == 1.)

        grid3 = Grid(shape=(10, 10))
        u3 = TimeFunction(name='u', grid=grid3, space_order=2, time_order=0)

        op.apply(time_M=0, u=u3)
        assert np.all(u3.data[0, 3:-3, 3:-3] == 1.)

    @pytest.mark.parallel(mode=4)
    def test_fission_due_to_antidep(self, mode):
        grid = Grid(shape=(16, 16, 64), dtype=np.float64)

        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name='u1', grid=grid, space_order=4)
        v = TimeFunction(name='v', grid=grid, space_order=4)
        v1 = TimeFunction(name='v1', grid=grid, space_order=4)

        eqns = [Eq(u.forward, v.laplace),
                Eq(v.forward, u.forward.dz2)]

        op1 = Operator(eqns, opt=('advanced', {'openmp': True}))

        # First, check the generated code
        assert_structure(op1, ['t',
                               't,x0_blk0,y0_blk0,x,y,z',
                               't,x0_blk0,y0_blk0,x,y,z'],
                         't,x0_blk0,y0_blk0,x,y,z,z')

        def init(f, v=1):
            f.data[:] = np.indices(grid.shape).sum(axis=0) % (.004*v) + .01

        init(u1)
        init(v1, 2)
        op1(u=u1, v=v1, time_M=5, h_z=20.)

        assert np.isclose(norm(u1), 12445251.87, rtol=1e-7)
        assert np.isclose(norm(v1), 147063.38, rtol=1e-7)

    @pytest.mark.parallel(mode=1)
    def test_interpolation_at_uforward(self, mode):
        grid = Grid(shape=(10, 10, 10))
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=2, time_order=2)

        rec = SparseTimeFunction(name="rec", grid=grid, npoint=1, nt=10)

        eqns = [Eq(u.forward, u.laplace + u.backward + 1),
                rec.interpolate(expr=u.forward)]

        op = Operator(eqns)

        op.cfunction

        calls, _ = check_halo_exchanges(op, 2, 1)
        args = calls[0].arguments
        assert args[-2].name == 't2'
        assert args[-2].origin == t + 1


def gen_serial_norms(shape, so):
    """
    Computes the norms of the outputs in serial mode to compare with
    """
    day = np.datetime64('today')
    try:
        l = np.load("norms%s.npy" % len(shape), allow_pickle=True)
        assert l[-1] == day
    except:
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape],
                                tn=tn, space_order=so, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)
        # Run forward operator
        rec, u, _ = solver.forward()
        Eu = norm(u)
        Erec = norm(rec)

        # Run adjoint operator
        srca, v, _ = solver.adjoint(rec=rec)
        Ev = norm(v)
        Esrca = norm(srca)

        np.save("norms%s.npy" % len(shape), (Eu, Erec, Ev, Esrca, day), allow_pickle=True)


class TestIsotropicAcoustic:

    """
    Test the isotropic acoustic wave equation with MPI.
    """
    _shapes = {1: (60,), 2: (60, 70), 3: (60, 70, 80)}
    _so = {1: 12, 2: 8, 3: 4}
    gen_serial_norms((60,), 12)
    gen_serial_norms((60, 70), 8)
    gen_serial_norms((60, 70, 80), 4)

    @cached_property
    def norms(self):
        return {1: np.load("norms1.npy", allow_pickle=True)[:-1],
                2: np.load("norms2.npy", allow_pickle=True)[:-1],
                3: np.load("norms3.npy", allow_pickle=True)[:-1]}

    @pytest.mark.parametrize('shape,kernel,space_order,save', [
        ((60, ), 'OT2', 4, False),
        ((60, 70), 'OT2', 8, False),
    ])
    @pytest.mark.parallel(mode=1)
    def test_adjoint_codegen(self, shape, kernel, space_order, save, mode):
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape], kernel=kernel,
                                tn=500, space_order=space_order, nrec=130,
                                preset='layers-isotropic', dtype=np.float64)
        op_fwd = solver.op_fwd(save=save)
        fwd_calls = FindNodes(Call).visit(op_fwd)

        op_adj = solver.op_adj()
        adj_calls = FindNodes(Call).visit(op_adj)

        assert len(fwd_calls) == 1
        assert len(adj_calls) == 1

    def run_adjoint_F(self, nd):
        """
        Unlike `test_adjoint_F` in test_adjoint.py, here we explicitly check the norms
        of all Operator-evaluated Functions. The numbers we check against are derived
        "manually" from sequential runs of test_adjoint::test_adjoint_F
        """
        Eu, Erec, Ev, Esrca = self.norms[nd]
        shape = self._shapes[nd]
        so = self._so[nd]
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape],
                                tn=tn, space_order=so, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)

        # Run forward operator
        src = solver.geometry.src
        rec, u, _ = solver.forward(src=src)

        assert np.isclose(norm(u) / Eu, 1.0)
        assert np.isclose(norm(rec) / Erec, 1.0)

        # Run adjoint operator
        srca = src.func(name='srca')
        srca, v, _ = solver.adjoint(srca=srca, rec=rec)

        assert np.isclose(norm(v) / Ev, 1.0)
        assert np.isclose(norm(srca) / Esrca, 1.0)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = inner(srca, solver.geometry.src)
        term2 = norm(rec)**2
        assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)

    @pytest.mark.parametrize('nd', [1, 2, 3])
    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap'),
                                (4, 'overlap2'), (4, 'full')])
    def test_adjoint_F(self, nd, mode):
        self.run_adjoint_F(nd)

    @pytest.mark.parallel(mode=[(8, 'diag2'), (8, 'full')])
    @switchconfig(language='C')
    def test_adjoint_F_no_omp(self, mode):
        """
        ``run_adjoint_F`` with OpenMP disabled. By disabling OpenMP, we can
        practically scale up to higher process counts.
        """
        self.run_adjoint_F(3)


class TestElasticLike:

    @pytest.mark.parallel(mode=[(1, 'diag')])
    def test_elastic_structure(self, mode):

        so = 4
        grid = Grid(shape=(3, 3))

        v = VectorTimeFunction(name='v', grid=grid, space_order=so)
        tau = TensorTimeFunction(name='t', grid=grid, space_order=so)

        damp = Function(name='damp', grid=grid)
        l = Function(name='lam', grid=grid)
        mu = Function(name='mu', grid=grid)
        ro = Function(name='b', grid=grid)

        rec = SparseTimeFunction(name="rec", grid=grid, npoint=1, nt=10)
        rec_term = rec.interpolate(expr=v[0] + v[1])

        # First order elastic wave equation
        pde_v = v.dt - ro * div(tau)
        pde_tau = (tau.dt - l * diag(div(v.forward)) -
                   mu * (grad(v.forward) + grad(v.forward).transpose(inner=False)))

        # Time update
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))
        u_t = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

        op = Operator([u_v] + [u_t] + rec_term)
        op.cfunction

        assert len(op._func_table) == 11

        calls = [i for i in FindNodes(Call).visit(op) if isinstance(i, HaloUpdateCall)]
        assert len(calls) == 5
        assert len(FindNodes(HaloUpdateCall).visit(op.body.body[1].body[1].body[0])) == 1
        assert len(FindNodes(HaloUpdateCall).visit(op.body.body[1].body[1].body[1])) == 4
        assert len(FindNodes(HaloUpdateCall).visit(op.body.body[1].body[1].body[2])) == 0

        assert calls[0].arguments[0] is v[0]
        assert calls[0].arguments[1] is v[1]
        assert calls[1].arguments[0] is tau[0, 0]
        assert calls[2].arguments[0] is tau[0, 1]
        assert calls[3].arguments[0] is tau[1, 1]
        assert calls[4].arguments[0] is v[0]
        assert calls[4].arguments[1] is v[1]

    @pytest.fixture
    def setup(self):
        """
        This fixture sets up the grid, fields, elastic-like
        equations and receivers for test_issue_2448_*.
        """
        shape = (2,)
        so = 2
        tn = 30

        grid = Grid(shape=shape)

        # Velocity and pressure fields
        v = TimeFunction(name='v', grid=grid, space_order=so)
        tau = TimeFunction(name='tau', grid=grid, space_order=so)

        # First order elastic-like dependencies equations
        pde_v = v.dt - (tau.dx)
        pde_tau = tau.dt - ((v.forward).dx)
        u_v = Eq(v.forward, solve(pde_v, v.forward))
        u_tau = Eq(tau.forward, solve(pde_tau, tau.forward))

        rec = SparseTimeFunction(name="rec", grid=grid, npoint=1, nt=tn)
        rec.coordinates.data[:, 0] = np.linspace(0., shape[0], num=1)

        return grid, v, tau, u_v, u_tau, rec

    @pytest.mark.parallel(mode=1)
    def test_issue_2448_v0(self, mode, setup):
        _, v, tau, u_v, u_tau, rec = setup

        rec_term0 = rec.interpolate(expr=v)

        op0 = Operator([u_v, u_tau, rec_term0])

        check_halo_exchanges(op0, 3, 2)
        calls = [i for i in FindNodes(Call).visit(op0) if isinstance(i, HaloUpdateCall)]

        assert len(calls) == 3
        assert len(FindNodes(HaloUpdateCall).visit(op0.body.body[1].body[1].body[0])) == 1
        assert len(FindNodes(HaloUpdateCall).visit(op0.body.body[1].body[1].body[1])) == 2
        assert calls[0].arguments[0] is v
        assert calls[1].arguments[0] is tau
        assert calls[2].arguments[0] is v

    @pytest.mark.parallel(mode=1)
    def test_issue_2448_v1(self, mode, setup):
        _, v, tau, u_v, u_tau, rec = setup

        rec_term1 = rec.interpolate(expr=v.forward)

        op1 = Operator([u_v, u_tau, rec_term1])
        op1.cfunction

        calls, _ = check_halo_exchanges(op1, 2, 2)
        assert calls[0].arguments[0] is tau
        assert calls[1].arguments[0] is v

    @pytest.mark.parallel(mode=1)
    def test_issue_2448_v2(self, mode, setup):
        grid, v, tau, u_v, u_tau, rec = setup

        # Additional velocity and pressure fields
        v2 = TimeFunction(name='v2', grid=grid, space_order=2)
        tau2 = TimeFunction(name='tau2', grid=grid, space_order=2)

        # First order elastic-like dependencies equations
        pde_v2 = v2.dt - (tau2.dx)
        pde_tau2 = tau2.dt - ((v2.forward).dx)
        u_v2 = Eq(v2.forward, solve(pde_v2, v2.forward))
        u_tau2 = Eq(tau2.forward, solve(pde_tau2, tau2.forward))

        rec2 = SparseTimeFunction(name="rec2", grid=grid, npoint=1, nt=30)
        rec2.coordinates.data[:, 0] = np.linspace(0., grid.shape[0], num=1)

        rec_term0 = rec.interpolate(expr=v)
        rec_term2 = rec2.interpolate(expr=v2)

        op2 = Operator([u_v, u_v2, u_tau, u_tau2, rec_term0, rec_term2])

        calls = [i for i in FindNodes(Call).visit(op2) if isinstance(i, HaloUpdateCall)]

        assert len(calls) == 5
        assert len(FindNodes(HaloUpdateCall).visit(op2.body.body[1].body[1].body[0])) == 2
        assert len(FindNodes(HaloUpdateCall).visit(op2.body.body[1].body[1].body[1])) == 3
        assert calls[0].arguments[0] is v
        assert calls[1].arguments[0] is v2
        assert calls[2].arguments[0] is tau
        assert calls[2].arguments[1] is tau2
        assert calls[3].arguments[0] is v
        assert calls[4].arguments[0] is v2

    @pytest.mark.parallel(mode=1)
    def test_issue_2448_v3(self, mode, setup):
        grid, v, tau, u_v, u_tau, rec = setup

        # Additional velocity and pressure fields
        v2 = TimeFunction(name='v2', grid=grid, space_order=2)
        tau2 = TimeFunction(name='tau2', grid=grid, space_order=2)

        # First order elastic-like dependencies equations
        pde_v2 = v2.dt - (tau2.dx)
        pde_tau2 = tau2.dt - ((v2.forward).dx)
        u_v2 = Eq(v2.forward, solve(pde_v2, v2.forward))
        u_tau2 = Eq(tau2.forward, solve(pde_tau2, tau2.forward))

        rec2 = SparseTimeFunction(name="rec2", grid=grid, npoint=1, nt=30)
        rec2.coordinates.data[:, 0] = np.linspace(0., grid.shape[0], num=1)

        rec_term0 = rec.interpolate(expr=v)
        rec_term3 = rec2.interpolate(expr=v2.forward)

        op3 = Operator([u_v, u_v2, u_tau, u_tau2, rec_term0, rec_term3])
        op3.cfunction

        calls = [i for i in FindNodes(Call).visit(op3) if isinstance(i, HaloUpdateCall)]

        assert len(calls) == 4
        assert len(FindNodes(HaloUpdateCall).visit(op3.body.body[1].body[1].body[0])) == 1
        assert len(FindNodes(HaloUpdateCall).visit(op3.body.body[1].body[1].body[1])) == 3
        assert calls[0].arguments[0] is v
        assert calls[1].arguments[0] is tau
        assert calls[1].arguments[1] is tau2
        assert calls[2].arguments[0] is v
        assert calls[3].arguments[0] is v2

    @pytest.mark.parallel(mode=1)
    def test_issue_2448_backward(self, mode):
        """
        Similar to test_issue_2448, but with backward instead of forward
        so that the hoisted halo has different starting point
        """
        shape = (2,)
        so = 2

        grid = Grid(shape=shape)
        t = grid.stepping_dim

        tn = 7

        # Velocity and pressure fields
        v = TimeFunction(name='v', grid=grid, space_order=so)
        v.data_with_halo[0, :] = 1.
        v.data_with_halo[1, :] = 3.

        tau = TimeFunction(name='tau', grid=grid, space_order=so)
        tau.data_with_halo[:] = 1.

        # First order elastic-like dependencies equations
        pde_v = v.dt - (tau.dx)
        pde_tau = tau.dt - ((v.backward).dx)

        u_v = Eq(v.backward, solve(pde_v, v))
        u_tau = Eq(tau.backward, solve(pde_tau, tau))

        # Test two variants of receiver interpolation
        nrec = 1
        rec = SparseTimeFunction(name="rec", grid=grid, npoint=nrec, nt=tn)
        rec.coordinates.data[:, 0] = np.linspace(0., shape[0], num=nrec)

        # Test receiver interpolation 0, here we have a halo exchange hoisted
        op0 = Operator([u_v] + [u_tau] + rec.interpolate(expr=v))

        calls = [i for i in FindNodes(Call).visit(op0)
                 if isinstance(i, HaloUpdateCall)]

        assert len(calls) == 3
        assert len(FindNodes(HaloUpdateCall).visit(op0.body.body[1].body[1].body[0])) == 1
        assert len(FindNodes(HaloUpdateCall).visit(op0.body.body[1].body[1].body[1])) == 2
        assert calls[0].arguments[0] is v
        assert calls[0].arguments[3].args[0] is t.symbolic_max
        assert calls[1].arguments[0] is tau
        assert calls[2].arguments[0] is v


class TestTTIOp:

    @pytest.mark.skipif(TestTTI is None, reason="Requires installing the tests")
    @pytest.mark.parallel(mode=1)
    def test_halo_structure(self, mode):
        solver = TestTTI().tti_operator(opt='advanced', space_order=8)
        op = solver.op_fwd(save=False)

        calls = [i for i in FindNodes(Call).visit(op) if isinstance(i, HaloUpdateCall)]

        assert len(calls) == 1
        assert calls[0].functions[0].name == 'u'
        assert calls[0].functions[1].name == 'v'


def get_time_loop(op):
    iters = FindNodes(Iteration).visit(op)
    for i in iters:
        if i.dim.is_Time:
            return i
    assert False


if __name__ == "__main__":
    # configuration['mpi'] = 'overlap'
    # TestDecomposition().test_reshape_left_right()
    # TestOperatorSimple().test_trivial_eq_2d()
    # TestFunction().test_halo_exchange_bilateral()
    # TestSparseFunction().test_sparse_coords()
    # TestSparseFunction().test_precomputed_sparse(2)
    # TestOperatorAdvanced().test_fission_due_to_antidep()
    TestOperatorAdvanced().test_injection_wodup_wtime()
    # TestIsotropicAcoustic().test_adjoint_F(1)
