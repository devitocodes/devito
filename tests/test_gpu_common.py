import pytest
import numpy as np

from devito import (Constant, Eq, Inc, Grid, Function, ConditionalDimension,
                    SubDimension, SubDomain, TimeFunction, Operator)
from devito.arch import get_gpu_info
from devito.ir import Expression, Section, FindNodes, FindSymbols, retrieve_iteration_tree
from devito.passes.iet.languages.openmp import OmpIteration
from devito.types import DeviceID, DeviceRM, Lock, PThreadArray

from conftest import skipif

pytestmark = skipif(['nodevice'], whole_module=True)


class TestGPUInfo(object):

    def test_get_gpu_info(self):
        info = get_gpu_info()
        try:
            assert 'tesla' in info['architecture'].lower()
        except TypeError:
            # There might be than one GPUs, but for now we don't care
            # as we're not really exploiting this info yet...
            pass


class TestCodeGeneration(object):

    def test_maxpar_option(self):
        """
        Make sure the `cire-maxpar` option is set to True by default.
        """
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=2)

        eq = Eq(u.forward, u.dy.dy)

        op = Operator(eq)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert trees[0][0] is trees[1][0]
        assert trees[0][1] is not trees[1][1]


class Bundle(SubDomain):
    """
    We use this SubDomain to enforce Eqs to end up in different loops.
    """

    name = 'bundle'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 0, 0), y: ('middle', 0, 0), z: ('middle', 0, 0)}


class TestStreaming(object):

    def test_tasking_in_isolation(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)

        tmp = Function(name='tmp', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(tmp, v),
                Eq(v.forward, v + 1),
                Eq(u.forward, tmp, subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op)) == 5
        assert len([i for i in FindSymbols().visit(op) if isinstance(i, Lock)]) == 1
        sections = FindNodes(Section).visit(op)
        assert len(sections) == 3
        assert str(sections[0].body[0].body[0].body[0].body[0]) == 'while(lock0[0] == 0);'
        body = sections[2].body[0].body[0]
        assert (str(body.body[1].condition) ==
                'Ne(lock0[0], 2) | Ne(FieldFromComposite(sdata0[wi0]), 1)')
        assert str(body.body[2]) == 'sdata0[wi0].time = time;'
        assert str(body.body[3]) == 'lock0[0] = 0;'
        assert str(body.body[4]) == 'sdata0[wi0].flag = 2;'

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 8)

    def test_tasking_fused(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)

        tmp = Function(name='tmp', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)
        w = TimeFunction(name='w', grid=grid)

        eqns = [Eq(w.forward, w + 1),
                Eq(tmp, w.forward),
                Eq(u.forward, tmp, subdomain=bundle0),
                Eq(v.forward, tmp, subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'fuse', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op)) == 4
        locks = [i for i in FindSymbols().visit(op) if isinstance(i, Lock)]
        assert len(locks) == 1  # Only 1 because it's only `tmp` that needs protection
        assert len(op._func_table) == 2
        exprs = FindNodes(Expression).visit(op._func_table['copy_device_to_host0'].root)
        assert len(exprs) == 19
        assert str(exprs[12]) == 'int id = sdata0->id;'
        assert str(exprs[13]) == 'const int time = sdata0->time;'
        assert str(exprs[14]) == 'lock0[0] = 1;'
        assert exprs[15].write is u
        assert exprs[16].write is v
        assert str(exprs[17]) == 'lock0[0] = 2;'
        assert str(exprs[18]) == 'sdata0->flag = 1;'

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 9)
        assert np.all(v.data[nt-1] == 9)

    def test_tasking_unfused_two_locks(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)

        tmp0 = Function(name='tmp0', grid=grid)
        tmp1 = Function(name='tmp1', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)
        w = TimeFunction(name='w', grid=grid)

        eqns = [Eq(w.forward, w + 1),
                Eq(tmp0, w.forward),
                Eq(tmp1, w.forward),
                Eq(u.forward, tmp0, subdomain=bundle0),
                Eq(v.forward, tmp1, subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'fuse', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op)) == 6
        assert len([i for i in FindSymbols().visit(op) if isinstance(i, Lock)]) == 2
        sections = FindNodes(Section).visit(op)
        assert len(sections) == 3
        assert (str(sections[0].body[0].body[0].body[0].body[0]) ==
                'while(lock0[0] == 0 || lock1[0] == 0);')  # Wait-lock
        body = sections[1].body[0].body[0]
        assert (str(body.body[1].condition) ==
                'Ne(lock0[0], 2) | Ne(FieldFromComposite(sdata0[wi0]), 1)')  # Wait-thread
        assert (str(body.body[1].body[0]) ==
                'wi0 = (wi0 + 1)%(npthreads0);')
        assert str(body.body[2]) == 'sdata0[wi0].time = time;'
        assert str(body.body[3]) == 'lock0[0] = 0;'  # Set-lock
        assert str(body.body[4]) == 'sdata0[wi0].flag = 2;'
        body = sections[2].body[0].body[0]
        assert (str(body.body[1].condition) ==
                'Ne(lock1[0], 2) | Ne(FieldFromComposite(sdata1[wi1]), 1)')  # Wait-thread
        assert (str(body.body[1].body[0]) ==
                'wi1 = (wi1 + 1)%(npthreads1);')
        assert str(body.body[2]) == 'sdata1[wi1].time = time;'
        assert str(body.body[3]) == 'lock1[0] = 0;'  # Set-lock
        assert str(body.body[4]) == 'sdata1[wi1].flag = 2;'
        assert len(op._func_table) == 4
        exprs = FindNodes(Expression).visit(op._func_table['copy_device_to_host0'].root)
        assert len(exprs) == 18
        assert str(exprs[14]) == 'lock0[0] = 1;'
        assert exprs[15].write is u
        exprs = FindNodes(Expression).visit(op._func_table['copy_device_to_host1'].root)
        assert str(exprs[14]) == 'lock1[0] = 1;'
        assert exprs[15].write is v

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 9)
        assert np.all(v.data[nt-1] == 9)

    @pytest.mark.parametrize('opt', [
        ('tasking', 'orchestrate'),
        ('tasking', 'streaming', 'orchestrate'),
    ])
    def test_attempt_tasking_but_no_temporaries(self, opt):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, save=10)

        op = Operator(Eq(u.forward, u + 1), opt=opt)

        piters = FindNodes(OmpIteration).visit(op)
        assert len(piters) == 0

        op = Operator(Eq(u.forward, u + 1), opt=(opt, {'par-disabled': False}))

        # Degenerates to host execution with no data movement, since `u` is
        # a host Function
        piters = FindNodes(OmpIteration).visit(op)
        assert len(piters) == 1
        assert type(piters.pop()) == OmpIteration

    def test_tasking_multi_output(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)
        t = grid.stepping_dim
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid, time_order=2)
        u1 = TimeFunction(name='u', grid=grid, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        usave1 = TimeFunction(name='usave', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + 1),
                Eq(usave, u.forward + u + u.backward + u[t, x-1, y, z],
                   subdomain=bundle0)]

        op0 = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
        op1 = Operator(eqns, opt=('tasking', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 4
        assert len([i for i in FindSymbols().visit(op1) if isinstance(i, Lock)]) == 1
        sections = FindNodes(Section).visit(op1)
        assert len(sections) == 2
        assert str(sections[0].body[0].body[0].body[0].body[0]) ==\
            'while(lock0[t2] == 0);'
        for i in range(3):
            assert 'lock0[t' in str(sections[1].body[0].body[0].body[6 + i])  # Set-lock
        assert str(sections[1].body[0].body[0].body[9]) == 'sdata0[wi0].flag = 2;'
        assert len(op1._func_table) == 2
        exprs = FindNodes(Expression).visit(op1._func_table['copy_device_to_host0'].root)
        assert len(exprs) == 25
        for i in range(3):
            assert 'lock0[t' in str(exprs[17 + i])
        assert exprs[20].write is usave

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, usave=usave1)

        assert np.all(u.data[:] == u1.data[:])
        assert np.all(usave.data[:] == usave1.data[:])

    def test_tasking_lock_placement(self):
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid, space_order=2)
        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=10)

        eqns = [Eq(f, u + 1),
                Eq(u.forward, f.dx + u + 1),
                Eq(usave, u)]

        op = Operator(eqns, opt=('tasking', 'orchestrate'))

        # Check generated code -- the wait-lock is expected in section1
        assert len(retrieve_iteration_tree(op)) == 5
        assert len([i for i in FindSymbols().visit(op) if isinstance(i, Lock)]) == 1
        sections = FindNodes(Section).visit(op)
        assert len(sections) == 3
        assert sections[0].body[0].body[0].body[0].is_Iteration
        assert str(sections[1].body[0].body[0].body[0].body[0]) ==\
            'while(lock0[t1] == 0);'

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_streaming_simple(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=nt)

        for i in range(nt):
            usave.data[i, :] = i

        eqn = Eq(u.forward, u + usave)

        op = Operator(eqn, opt=('streaming', 'orchestrate'))

        op.apply(time_M=nt-2)

        assert np.all(u.data[0] == 28)
        assert np.all(u.data[1] == 36)

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_streaming_two_buffers(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        vsave = TimeFunction(name='vsave', grid=grid, save=nt)

        for i in range(nt):
            usave.data[i, :] = i
            vsave.data[i, :] = i

        eqn = Eq(u.forward, u + usave + vsave)

        op = Operator(eqn, opt=('streaming', 'orchestrate'))

        op.apply(time_M=nt-2)

        assert np.all(u.data[0] == 56)
        assert np.all(u.data[1] == 72)

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_streaming_multi_input(self):
        nt = 100
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, save=nt, time_order=2, space_order=2)
        v = TimeFunction(name='v', grid=grid, save=None, time_order=2, space_order=2)
        grad = Function(name='grad', grid=grid)
        grad1 = Function(name='grad', grid=grid)

        v.data[:] = 0.02
        for i in range(nt):
            u.data[i, :] = i + 0.1

        eqn = Eq(grad, grad - u.dt2 * v)

        op0 = Operator(eqn, opt=('noop', {'gpu-fit': u}))
        op1 = Operator(eqn, opt=('streaming', 'orchestrate'))

        op0.apply(time_M=nt-2, dt=0.1)
        op1.apply(time_M=nt-2, dt=0.1, grad=grad1)

        assert np.all(grad.data == grad1.data)

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_streaming_postponed_deletion(self):
        nt = 10
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid)
        v1 = TimeFunction(name='v', grid=grid)

        for i in range(nt):
            usave.data[i, :] = i

        eqns = [Eq(u.forward, u + usave),
                Eq(v.forward, v + u.forward.dx + usave)]

        op0 = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
        op1 = Operator(eqns, opt=('streaming', 'orchestrate'))

        op0.apply(time_M=nt-1)
        op1.apply(time_M=nt-1, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_streaming_with_host_loop(self):
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=10)

        eqns = [Eq(f, u),
                Eq(u.forward, f + 1)]

        op = Operator(eqns, opt=('streaming', 'orchestrate'))

        assert len(op._func_table) == 3
        assert 'init_device0' in op._func_table
        assert 'prefetch_host_to_device0' in op._func_table

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_composite_streaming_tasking(self):
        nt = 10
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid)
        u1 = TimeFunction(name='u', grid=grid)
        fsave = TimeFunction(name='fsave', grid=grid, save=nt)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        usave1 = TimeFunction(name='usave', grid=grid, save=nt)

        for i in range(nt):
            fsave.data[i, :] = i

        eqns = [Eq(u.forward, u + fsave + 1),
                Eq(usave, u)]

        op0 = Operator(eqns, opt=('noop', {'gpu-fit': (fsave, usave)}))
        op1 = Operator(eqns, opt=('tasking', 'streaming', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op0)) == 1
        assert len(retrieve_iteration_tree(op1)) == 4
        symbols = FindSymbols().visit(op1)
        assert len([i for i in symbols if isinstance(i, Lock)]) == 1
        threads = [i for i in symbols if isinstance(i, PThreadArray)]
        assert len(threads) == 2
        assert threads[0].size == 1
        assert threads[1].size.data == 2

        op0.apply(time_M=nt-1)
        op1.apply(time_M=nt-1, u=u1, usave=usave1)

        assert np.all(u.data == u1.data)
        assert np.all(usave.data == usave1.data)

    def test_composite_buffering_tasking(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(4, 4, 4), subdomains=bundle0)

        u = TimeFunction(name='u', grid=grid, time_order=2)
        u1 = TimeFunction(name='u', grid=grid, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        usave1 = TimeFunction(name='usave', grid=grid, save=nt)

        eqns = [Eq(u.forward, u*1.1 + 1),
                Eq(usave, u.dt2, subdomain=bundle0)]

        op0 = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
        op1 = Operator(eqns, opt=('buffering', 'tasking', 'orchestrate'))

        # Check generated code -- thanks to buffering only expect 1 lock!
        assert len(retrieve_iteration_tree(op0)) == 2
        assert len(retrieve_iteration_tree(op1)) == 5
        symbols = FindSymbols().visit(op1)
        assert len([i for i in symbols if isinstance(i, Lock)]) == 1
        threads = [i for i in symbols if isinstance(i, PThreadArray)]
        assert len(threads) == 1
        assert threads[0].size.data == 1

        op0.apply(time_M=nt-1, dt=0.1)
        op1.apply(time_M=nt-1, dt=0.1, u=u1, usave=usave1)

        assert np.all(u.data == u1.data)
        assert np.all(usave.data == usave1.data)

    def test_composite_buffering_tasking_multi_output(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(4, 4, 4), subdomains=bundle0)

        u = TimeFunction(name='u', grid=grid, time_order=2)
        v = TimeFunction(name='v', grid=grid, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        vsave = TimeFunction(name='vsave', grid=grid, save=nt)

        u1 = TimeFunction(name='u', grid=grid, time_order=2)
        v1 = TimeFunction(name='v', grid=grid, time_order=2)
        usave1 = TimeFunction(name='usave', grid=grid, save=nt)
        vsave1 = TimeFunction(name='vsave', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + 1),
                Eq(v.forward, v + 1),
                Eq(usave, u, subdomain=bundle0),
                Eq(vsave, v, subdomain=bundle0)]

        op0 = Operator(eqns, opt=('noop', {'gpu-fit': (usave, vsave)}))
        op1 = Operator(eqns, opt=('buffering', 'tasking', 'topofuse', 'orchestrate'))

        # Check generated code -- thanks to buffering only expect 1 lock!
        assert len(retrieve_iteration_tree(op0)) == 2
        assert len(retrieve_iteration_tree(op1)) == 7
        symbols = FindSymbols().visit(op1)
        assert len([i for i in symbols if isinstance(i, Lock)]) == 2
        threads = [i for i in symbols if isinstance(i, PThreadArray)]
        assert len(threads) == 2
        assert threads[0].size.data == 1
        assert threads[1].size.data == 1
        assert len(op1._func_table) == 4  # usave and vsave eqns are in two diff efuncs

        op0.apply(time_M=nt-1)
        op1.apply(time_M=nt-1, u=u1, v=v1, usave=usave1, vsave=vsave1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)
        assert np.all(usave.data == usave1.data)
        assert np.all(vsave.data == vsave1.data)

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    def test_composite_full(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v1 = TimeFunction(name='v', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + v + 1),
                Eq(v.forward, u + v + v.backward)]

        op0 = Operator(eqns, opt=('noop', {'gpu-fit': (u, v)}))
        op1 = Operator(eqns, opt=('buffering', 'tasking', 'streaming', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 9
        assert len([i for i in FindSymbols().visit(op1) if isinstance(i, Lock)]) == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_tasking_over_compiler_generated(self):
        nt = 10
        bundle0 = Bundle()
        grid = Grid(shape=(4, 4, 4), subdomains=bundle0)

        u = TimeFunction(name='u', grid=grid, space_order=2)
        u1 = TimeFunction(name='u', grid=grid, space_order=2)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        usave1 = TimeFunction(name='usave', grid=grid, save=nt)

        eqns = [Eq(u.forward, u.dx.dx*0.042 + 1),
                Eq(usave, u, subdomain=bundle0)]

        op0 = Operator(eqns, opt=('cire-sops', {'gpu-fit': usave}))
        op1 = Operator(eqns, opt=('cire-sops', 'tasking', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 5
        assert len([i for i in FindSymbols().visit(op1) if isinstance(i, Lock)]) == 1
        sections = FindNodes(Section).visit(op1)
        assert len(sections) == 3
        assert 'while(lock0[t' in str(sections[1].body[0].body[0].body[0])

        op0.apply(time_M=nt-1)
        op1.apply(time_M=nt-1, u=u1, usave=usave1)

        assert np.all(u.data == u1.data)
        assert np.all(usave.data == usave1.data)

    @pytest.mark.parametrize('opt,gpu_fit,async_degree', [
        (('tasking', 'orchestrate'), True, None),
        (('buffering', 'tasking', 'orchestrate'), True, None),
        (('buffering', 'tasking', 'orchestrate'), False, None),
        (('buffering', 'tasking', 'orchestrate'), False, 3),
    ])
    def test_save(self, opt, gpu_fit, async_degree):
        nt = 10
        grid = Grid(shape=(300, 300, 300))

        time_dim = grid.time_dim

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)
        # For the given `nt` and grid shape, `usave` is roughly 4*5*300**3=~ .5GB of data

        op = Operator([Eq(u.forward, u + 1), Eq(usave, u.forward)],
                      opt=(opt, {'gpu-fit': usave if gpu_fit else None,
                                 'buf-async-degree': async_degree}))

        op.apply(time_M=nt-1)

        assert all(np.all(usave.data[i] == 2*i + 1) for i in range(usave.save))

    def test_save_multi_output(self):
        nt = 10
        grid = Grid(shape=(150, 150, 150))

        time_dim = grid.time_dim

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)
        vsave = TimeFunction(name='vsave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)

        eqns = [Eq(u.forward, u + 1),
                Eq(usave, u.forward),
                Eq(vsave, u.forward)]

        op = Operator(eqns, opt=('buffering', 'tasking', 'topofuse', 'orchestrate'))

        # Check generated code
        assert len(op._func_table) == 4  # usave and vsave eqns are in separate tasks

        op.apply(time_M=nt-1)

        assert all(np.all(usave.data[i] == 2*i + 1) for i in range(usave.save))
        assert all(np.all(vsave.data[i] == 2*i + 1) for i in range(vsave.save))

    def test_save_w_shifting(self):
        factor = 4
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=2, time_dim=time_subsampled)

        save_shift = Constant(name='save_shift', dtype=np.int32)

        eqns = [Eq(u.forward, u + 1.),
                Eq(usave.subs(time_subsampled, time_subsampled - save_shift), u)]

        op = Operator(eqns, opt=('buffering', 'tasking', 'orchestrate'))

        # Starting at time_m=10, so time_subsampled - save_shift is in range
        op.apply(time_m=10, time_M=nt-2, save_shift=3)
        assert np.all(np.allclose(u.data[0], 8))
        assert np.all([np.allclose(usave.data[i], 2+i*factor) for i in range(2)])

    def test_save_w_nonaffine_time(self):
        factor = 4
        grid = Grid(shape=(11, 11))
        x, y = grid.dimensions
        t = grid.stepping_dim
        time = grid.time_dim

        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)

        f = Function(name='f', grid=grid, dtype=np.int32)
        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=2, time_dim=time_subsampled)

        save_shift = Constant(name='save_shift', dtype=np.int32)

        eqns = [Eq(u.forward, u[t, f[x, x], f[y, y]] + 1.),
                Eq(usave.subs(time_subsampled, time_subsampled - save_shift), u)]

        op = Operator(eqns, opt=('buffering', 'tasking', 'orchestrate'))

        # We just check the generated code here
        assert len([i for i in FindSymbols().visit(op) if isinstance(i, Lock)]) == 1
        assert len(op._func_table) == 2

    def test_save_w_subdims(self):
        nt = 10
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        time_dim = grid.time_dim
        xi = SubDimension.middle(name='xi', parent=x, thickness_left=3, thickness_right=3)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=3, thickness_right=3)

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)

        eqns = [Eq(u.forward, u + 1),
                Eq(usave, u.forward)]
        eqns = [e.xreplace({x: xi, y: yi}) for e in eqns]

        op = Operator(eqns, opt=('buffering', 'tasking', 'orchestrate'))

        op.apply(time_M=nt-1)

        for i in range(usave.save):
            assert np.all(usave.data[i, 3:-3, 3:-3] == 2*i + 1)
            assert np.all(usave.data[i, :3, :] == 0)
            assert np.all(usave.data[i, -3:, :] == 0)
            assert np.all(usave.data[i, :, :3] == 0)
            assert np.all(usave.data[i, :, -3:] == 0)

    @skipif('device-openmp')  # TODO: Still unsupported with OpenMP, but soon will be
    @pytest.mark.parametrize('opt,gpu_fit', [
        (('streaming', 'orchestrate'), True),
        (('streaming', 'orchestrate'), False)
    ])
    def test_xcor_from_saved(self, opt, gpu_fit):
        nt = 10
        grid = Grid(shape=(300, 300, 300))
        time_dim = grid.time_dim

        period = 2
        factor = Constant(name='factor', value=period, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        g = Function(name='g', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)
        # For the given `nt` and grid shape, `usave` is roughly 4*5*300**3=~ .5GB of data

        for i in range(int(nt//period)):
            usave.data[i, :] = i
        v.data[:] = i*2 + 1

        # Assuming nt//period=5, we are computing, over 5 iterations:
        # g = 4*4  [time=8] + 3*3 [time=6] + 2*2 [time=4] + 1*1 [time=2]
        op = Operator([Eq(v.backward, v - 1), Inc(g, usave*(v/2))],
                      opt=(opt, {'gpu-fit': usave if gpu_fit else None}))

        op.apply(time_M=nt-1)

        assert np.all(g.data == 30)


class TestAPI(object):

    def get_param(self, op, param):
        for i in op.parameters:
            if isinstance(i, param):
                return i
        return None

    def check_deviceid(self):
        grid = Grid(shape=(6, 6))

        u = TimeFunction(name='u', grid=grid, space_order=2, save=10)

        op = Operator(Eq(u.forward, u.dx + 1))

        deviceid = self.get_param(op, DeviceID)
        assert deviceid is not None
        assert deviceid.data == -1
        assert op.arguments()[deviceid.name] == -1
        assert op.arguments(deviceid=0)[deviceid.name] == 0

    def test_deviceid(self):
        self.check_deviceid()

    @pytest.mark.parallel(mode=1)
    def test_deviceid_w_mpi(self):
        self.check_deviceid()

    def test_devicerm(self):
        grid = Grid(shape=(6, 6))

        u = TimeFunction(name='u', grid=grid, space_order=2)
        f = Function(name='f', grid=grid)

        op = Operator(Eq(u.forward, u.dx + f))

        devicerm = self.get_param(op, DeviceRM)
        assert devicerm is not None
        assert devicerm.data == 1  # Always evict, by default
        assert op.arguments(time_M=2)[devicerm.name] == 1
        assert op.arguments(time_M=2, devicerm=0)[devicerm.name] == 0
        assert op.arguments(time_M=2, devicerm=1)[devicerm.name] == 1
        assert op.arguments(time_M=2, devicerm=224)[devicerm.name] == 1
        assert op.arguments(time_M=2, devicerm=True)[devicerm.name] == 1
        assert op.arguments(time_M=2, devicerm=False)[devicerm.name] == 0
