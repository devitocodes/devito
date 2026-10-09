import numpy as np
import pytest

from conftest import assert_blocking, opts_device_tiling, skipif
from devito import (
    ConditionalDimension, Eq, Function, Grid, Max, Operator, SparseTimeFunction,
    TimeFunction, norm, solve, switchconfig
)
from devito.data import LEFT
from devito.exceptions import InvalidOperator
from devito.ir.iet import (
    BusyWaitCall, FindNodes, Iteration, MapNodes, TimedList, retrieve_iteration_tree
)
from examples.seismic import Receiver, RickerSource, TimeAxis

pytestmark = skipif(['nodevice'], whole_module=True)


class TestCodeGeneration:

    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].ccode.value ==\
            'acc parallel loop collapse(3) present(u)'
        assert op.body.maps[0].ccode.value ==\
            ('acc enter data copyin(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body.unmaps[0].ccode.value ==\
            ('acc exit data copyout(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body.unmaps[1].ccode.value ==\
            ('acc exit data delete(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]]) if(devicerm)')

        # Currently, advanced-fsg mode == advanced mode
        op1 = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc',
                       opt='advanced-fsg')
        assert str(op) == str(op1)

    def test_basic_customop(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1),
                      platform='nvidiaX', language='openacc', opt='openacc')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].ccode.value ==\
            'acc parallel loop collapse(3) present(u)'

        try:
            Operator(Eq(u.forward, u + 1),
                     platform='nvidiaX', language='openacc', opt='openmp')
        except InvalidOperator:
            assert True
        except Exception as e:
            raise AssertionError('Assert False') from e

    @pytest.mark.parametrize('opt', opts_device_tiling)
    def test_blocking(self, opt):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u.dx + 1),
                      platform='nvidiaX', language='openacc', opt=opt)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree) == 7
        assert all(i.dim.is_Block for i in tree[1:7])

        assert op.parameters[4] is tree[1].step
        assert op.parameters[7] is tree[2].step
        assert op.parameters[10] is tree[3].step

        assert tree[1].pragmas[0].ccode.value ==\
            'acc parallel loop collapse(3) present(u)'

    @pytest.mark.parametrize('par_tile', [True, (32, 4), (32, 4, 4), (32, 4, 4, 8)])
    def test_tile_insteadof_collapse(self, par_tile):
        grid = Grid(shape=(3, 3, 3))
        t = grid.stepping_dim
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid)
        src = SparseTimeFunction(name="src", grid=grid, nt=3, npoint=1)

        eqns = [Eq(u.forward, u + 1,),
                Eq(u[t+1, 0, y, z], u[t, 0, y, z] + 1.)]
        eqns += src.inject(field=u.forward, expr=src)

        op = Operator(eqns, platform='nvidiaX', language='openacc',
                      opt=('advanced', {'par-tile': par_tile}))

        trees = retrieve_iteration_tree(op)
        stile = (32, 4, 4, 4) if par_tile != (32, 4, 4, 8) else (32, 4, 4, 8)
        assert len(trees) == 4

        assert trees[0][1].pragmas[0].ccode.value ==\
            'acc parallel loop tile(32,4,4) present(u)'
        assert trees[1][1].pragmas[0].ccode.value ==\
            'acc parallel loop tile(32,4) present(u)'
        strtile = ','.join([str(i) for i in stile])
        pres = 'src,src_gp,src_wx,src_wy,src_wz,u'
        assert trees[3][1].pragmas[0].ccode.value ==\
            f'acc parallel loop tile({strtile}) present({pres})'

    @pytest.mark.parametrize('par_tile', [((32, 4, 4), (8, 8)), ((32, 4), (8, 8)),
                                          ((32, 4, 4), (8, 8, 8)),
                                          ((32, 4, 4), (8, 8), None)])
    def test_multiple_tile_sizes(self, par_tile):
        grid = Grid(shape=(3, 3, 3))
        t = grid.stepping_dim
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid)
        src = SparseTimeFunction(name="src", grid=grid, nt=3, npoint=1)

        eqns = [Eq(u.forward, u + 1,),
                Eq(u[t+1, 0, y, z], u[t, 0, y, z] + 1.)]
        eqns += src.inject(field=u.forward, expr=src)

        op = Operator(eqns, platform='nvidiaX', language='openacc',
                      opt=('advanced', {'par-tile': par_tile}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 4

        assert trees[0][1].pragmas[0].ccode.value ==\
            'acc parallel loop tile(32,4,4) present(u)'
        assert trees[1][1].pragmas[0].ccode.value ==\
            'acc parallel loop tile(8,8) present(u)'
        sclause = 'collapse(4)' if par_tile[-1] is None else 'tile(8,8,8,8)'
        assert trees[3][1].pragmas[0].ccode.value ==\
            f'acc parallel loop {sclause} present(src,src_gp,src_wx,src_wy,src_wz,u)'

    def test_short_multi_tile_keeps_outer_dim_on_device(self):
        """
        A short multi `par-tile` leaves the outermost Dimension unblocked.
        Its full loop must stay inside the OpenACC region.
        """
        grid = Grid(shape=(8, 8, 8))

        u = TimeFunction(name="u", grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eqns = [Eq(u.forward, u.dx),
                Eq(v.forward, u.forward.dx)]

        # The second entry is 2D, while the nest it lands on is 3D
        par_tile = ((32, 4, 4), (16, 4))

        op = Operator(eqns, platform='nvidiaX', language='openacc',
                      opt=(
                          'advanced',
                          {'par-tile': par_tile, 'blocklevels': 1, 'blockinner': True}))

        bns, _ = assert_blocking(op, {'x0_blk0', 'y1_blk0'})

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert all(tree[0].dim is grid.time_dim for tree in trees)
        assert trees[0][1].pragmas[0].ccode.value ==\
            'acc parallel loop tile(32,4,4) present(u)'

        x = grid.dimensions[0]
        assert trees[1][1].dim is x
        assert trees[1][1].limits == (x.symbolic_min, x.symbolic_max, 1)
        assert trees[1][1].pragmas[0].ccode.value ==\
            'acc parallel loop tile(16,4,4) present(u,v)'

        expected = ((4, 4, 32), (4, 16))
        for root, v in zip(bns.values(), expected, strict=True):
            iters = FindNodes(Iteration).visit(root)
            iters = [i for i in iters if i.dim.is_Block and i.dim._depth == 1]
            assert len(iters) == len(v)
            assert all(i.step == j for i, j in zip(iters, v, strict=True))

    def test_multi_tile_blocking_structure(self):
        grid = Grid(shape=(8, 8, 8))

        u = TimeFunction(name="u", grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eqns = [Eq(u.forward, u.dx),
                Eq(v.forward, u.forward.dx)]

        par_tile = ((32, 4, 4), (16, 4, 4))
        expected = ((4, 4, 32), (4, 4, 16))

        op = Operator(eqns, platform='nvidiaX', language='openacc',
                      opt=(
                          'advanced',
                          {'par-tile': par_tile, 'blocklevels': 1, 'blockinner': True}))

        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0'})
        assert len(bns) == len(expected)
        assert bns['x0_blk0'].pragmas[0].ccode.value ==\
            'acc parallel loop tile(32,4,4) present(u)'
        assert bns['x1_blk0'].pragmas[0].ccode.value ==\
            'acc parallel loop tile(16,4,4) present(u,v)'
        for root, v in zip(bns.values(), expected, strict=True):
            iters = FindNodes(Iteration).visit(root)
            iters = [i for i in iters if i.dim.is_Block and i.dim._depth == 1]
            assert len(iters) == len(v)
            assert all(i.step == j for i, j in zip(iters, v, strict=True))

    def test_std_max(self):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions

        u = Function(name='u', grid=grid)

        op = Operator(Eq(u, Max(1.2 * x / y, 2.3 * y / x)),
                      platform='nvidiaX', language='openacc')

        assert '<algorithm>' in str(op)


class TestOperator:

    def test_op_apply(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, dtype=np.int32)

        op = Operator(Eq(u.forward, u + 1))

        # Make sure we've indeed generated OpenACC code
        assert 'acc parallel' in str(op)

        time_steps = 1000
        op.apply(time_M=time_steps)

        assert np.all(np.array(u.data[0, :, :, :]) == time_steps)

    def iso_acoustic(self, opt):
        shape = (101, 101)
        extent = (1000, 1000)
        origin = (0., 0.)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

        t0 = 0.
        tn = 1000.
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name='src', grid=grid, f0=f0,
                           npoint=1, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size*.5
        src.coordinates.data[0, -1] = 20.

        rec = Receiver(name='rec', grid=grid, npoint=101, time_range=time_range)
        rec.coordinates.data[:, 0] = np.linspace(0, domain_size[0], num=101)
        rec.coordinates.data[:, 1] = 20.

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        pde = m * u.dt2 - u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
        rec_term = rec.interpolate(expr=u.forward)

        op = Operator([stencil] + src_term + rec_term, opt=opt, language='openacc')

        # Make sure we've indeed generated OpenACC code
        assert 'acc parallel' in str(op)

        op(time=time_range.num-1, dt=dt)

        assert np.isclose(norm(rec), 490.56, atol=1e-2, rtol=0)

    @pytest.mark.parametrize('opt', [
        'advanced',
        ('advanced', {'blocklevels': 1, 'linearize': True}),
    ])
    def test_iso_acoustic(self, opt):
        TestOperator().iso_acoustic(opt)

    @pytest.mark.parametrize('profiling', ['advanced1', 'advanced2'])
    @pytest.mark.parametrize('async_degree,npthreads', [(1, None), (3, None), (1, 1)])
    def test_wait_timing(self, profiling, async_degree, npthreads):
        grid = Grid(shape=(4, 4))
        u = TimeFunction(name='u', grid=grid)
        ts = ConditionalDimension(name='ts', parent=grid.time_dim, factor=2)
        usave = TimeFunction(name='usave', grid=grid, save=3, time_dim=ts)

        with switchconfig(profiling=profiling):
            op = Operator([Eq(u.forward, u + 1), Eq(usave, u)],
                          name='wait_timing', platform='nvidiaX', language='openacc',
                          opt=('buffering', 'tasking', 'orchestrate',
                               {'buf-async-degree': async_degree,
                                'npthreads': npthreads}))

        timed_calls = MapNodes(TimedList, BusyWaitCall, mode='immediate').visit(op)
        waits = {c.name: timer for timer, calls in timed_calls.items() for c in calls}
        assert set(waits) == {'release_lock0', 'activate0', 'shutdown0'}
        assert all(t is not None and t.name.startswith('busywait')
                   for t in waits.values())

        standalone = {name for name, timer in waits.items()
                      if timer.name in op._profiler._sections}
        expected = {'shutdown0', 'activate0'} if npthreads else {'shutdown0'}
        assert standalone == expected
        for name in [*waits, 'copy_to_host0']:
            assert not FindNodes(TimedList).visit(op._func_table[name].root)

        # Two propagations through the same Operator, followed by an empty loop.
        runs = [(0, 4, [4, 5], [0, 2, 4]),
                (0, 2, [2, 3], [0, 2, 0]),
                (1, 0, [0, 0], [0, 0, 0])]
        for time_m, time_M, expected_u, expected_save in runs:
            u.data[:] = 0
            usave.data[:] = 0
            summary = op.apply(time_m=time_m, time_M=time_M)

            assert np.all(u.data == np.array(expected_u)[:, None, None])
            assert np.all(usave.data == np.array(expected_save)[:, None, None])

            # Standalone waits belong to the global total; nested waits are
            # subsections of their caller and must not be counted a second time.
            reported_waits = [k for k in summary if k.name.startswith('busywait')]
            assert reported_waits
            if time_m > time_M:
                assert len(reported_waits) == 1
            assert summary.globals_nosetup_all.time == pytest.approx(
                sum(v.time for v in summary.values())
            )


class TestMPI:

    @pytest.mark.parallel(mode=2)
    def test_basic(self, mode):
        grid = Grid(shape=(6, 6))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=2)
        u.data[:] = 1.

        expr = u[t, x, y-1] + u[t, x-1, y] + u[t, x, y] + u[t, x, y+1] + u[t, x+1, y]
        op = Operator(Eq(u.forward, expr), platform='nvidiaX', language='openacc')

        # Make sure we've indeed generated OpenACC+MPI code
        assert 'acc parallel' in str(op)
        assert len(op._func_table) == 4

        op(time_M=1)

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(u.data[0] == [[11., 16., 17., 17., 16., 11.],
                                        [16., 23., 24., 24., 23., 16.],
                                        [17., 24., 25., 25., 24., 17.]])
        else:
            assert np.all(u.data[0] == [[17., 24., 25., 25., 24., 17.],
                                        [16., 23., 24., 24., 23., 16.],
                                        [11., 16., 17., 17., 16., 11.]])

    @pytest.mark.parallel(mode=2)
    def test_iso_ac(self, mode):
        TestOperator().iso_acoustic(opt='advanced')
