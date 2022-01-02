import pytest
import numpy as np

from devito import (Grid, Function, TimeFunction, SparseTimeFunction, Eq, Operator,
                    norm, solve)
from conftest import skipif, opts_device_tiling
from devito.data import LEFT
from devito.exceptions import InvalidOperator
from devito.ir.iet import FindNodes, Section, retrieve_iteration_tree
from examples.seismic import TimeAxis, RickerSource, Receiver

pytestmark = skipif(['nodevice'], whole_module=True)


class TestCodeGeneration(object):

    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'acc parallel loop collapse(3) present(u)'
        assert op.body.maps[0].pragmas[0].value ==\
            ('acc enter data copyin(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body.unmaps[0].pragmas[0].value ==\
            ('acc exit data copyout(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body.unmaps[1].pragmas[0].value ==\
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

        assert trees[0][1].pragmas[0].value ==\
            'acc parallel loop collapse(3) present(u)'

        try:
            Operator(Eq(u.forward, u + 1),
                     platform='nvidiaX', language='openacc', opt='openmp')
        except InvalidOperator:
            assert True
        except:
            assert False

    @pytest.mark.parametrize('opt', opts_device_tiling)
    def test_blocking(self, opt):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1),
                      platform='nvidiaX', language='openacc', opt=opt)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree) == 7
        assert all(i.dim.is_Block for i in tree[1:7])

        assert op.parameters[3] is tree[1].step
        assert op.parameters[6] is tree[2].step
        assert op.parameters[9] is tree[3].step

        assert tree[1].pragmas[0].value ==\
            'acc parallel loop collapse(3) present(u)'

    def test_streaming_postponed_deletion(self):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=10)

        eqns = [Eq(u.forward, u + usave),
                Eq(v.forward, v + u.forward.dx + usave)]

        op = Operator(eqns, platform='nvidiaX', language='openacc',
                      opt=('streaming', 'orchestrate'))

        sections = FindNodes(Section).visit(op)
        assert len(sections) == 2
        assert str(sections[1].body[0].body[0].body[-1]) ==\
            ('#pragma acc exit data delete(usave[time:1][0:usave_vec->size[1]]'
             '[0:usave_vec->size[2]][0:usave_vec->size[3]])')

    def test_streaming_with_host_loop(self):
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=10)

        eqns = [Eq(f, u),
                Eq(u.forward, f + 1)]

        op = Operator(eqns, platform='nvidiaX', language='openacc',
                      opt=('streaming', 'orchestrate'))

        # Check generated code
        assert len(op._func_table) == 3
        assert 'init_device0' in op._func_table
        assert 'init_tsdata0' in op._func_table
        assert 'prefetch_host_to_device0' in op._func_table
        sections = FindNodes(Section).visit(op)
        assert len(sections) == 2
        s = sections[0].body[0].body[0]
        assert str(s.body[3].body[-1]) == ('#pragma acc exit data delete'
                                           '(u[time:1][0:u_vec->size[1]][0:u_vec'
                                           '->size[2]][0:u_vec->size[3]])')
        assert str(s.body[2]) == ('#pragma acc data present(u[time:1][0:u_vec->'
                                  'size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3
        assert 'present(f)' in str(trees[0][1].pragmas[0])

    @pytest.mark.parametrize('par_tile', [True, (32, 4, 4)])
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
        assert len(trees) == 4

        assert trees[0][1].pragmas[0].value ==\
            'acc parallel loop tile(32,4,4) present(u)'
        assert trees[1][1].pragmas[0].value ==\
            'acc parallel loop tile(32,4) present(u)'
        # Only the AFFINE Iterations are tiled
        assert trees[3][1].pragmas[0].value ==\
            'acc parallel loop collapse(1) present(src,src_coords,u)'


class TestOperator(object):

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


class TestMPI(object):

    @pytest.mark.parallel(mode=2)
    def test_basic(self):
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
    def test_iso_ac(self):
        TestOperator().iso_acoustic(opt='advanced')
