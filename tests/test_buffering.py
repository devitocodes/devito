import pytest
import numpy as np

from conftest import skipif
from devito import (Constant, Grid, TimeFunction, Operator, Eq, SubDimension,
                    SubDomain, ConditionalDimension, configuration, switchconfig)
from devito.arch.archinfo import AppleArm
from devito.ir import FindSymbols, retrieve_iteration_tree
from devito.exceptions import CompilationError


def test_read_write():
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)

    eqn = Eq(u.forward, u + 1)

    op0 = Operator(eqn, opt='noop')
    op1 = Operator(eqn, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1
    assert buffers.pop().symbolic_shape[0] == 2

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_write_only():
    nt = 10
    grid = Grid(shape=(4, 4))
    time = grid.time_dim

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    eqns = [Eq(v.forward, v + 1, implicit_dims=time),
            Eq(u, v)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1, v=v1)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


def test_read_only():
    nt = 10
    grid = Grid(shape=(2, 2))

    u = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(nt):
        u.data[i, :] = i

    eqns = [Eq(v.forward, v + u.backward + u + u.forward + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 4
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, v=v1)

    assert np.all(v.data == v1.data)


def test_read_only_w_offset():
    nt = 10
    grid = Grid(shape=(2, 2))

    u = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(nt):
        u.data[i, :] = i

    eqns = [Eq(v.forward, v + u.backward + u + u.forward + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    op0.apply(time_M=nt-2, time_m=4)
    op1.apply(time_M=nt-2, time_m=4, v=v1)

    assert np.all(v.data == v1.data)


def test_read_only_backwards():
    nt = 10
    grid = Grid(shape=(2, 2))

    u = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(nt):
        u.data[i, :] = i

    eqns = [Eq(v.backward, v + u.backward + u + u.forward + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 4
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_m=1)
    op1.apply(time_m=1, v=v1)

    assert np.all(v.data == v1.data)


def test_read_only_backwards_unstructured():
    """
    Instead of the class `time-1`, `time`, and `time+1`, here we access the
    buffered Function via `time-2`, `time-1` and `time+2`.
    """
    nt = 10
    grid = Grid(shape=(2, 2))

    u = TimeFunction(name='u', grid=grid, save=nt, space_order=0)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(nt):
        u.data[i, :] = i

    eqns = [Eq(v.backward, v + u.backward.backward + u.backward + u.forward.forward + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_m=2)
    op1.apply(time_m=2, v=v1)

    assert np.all(v.data == v1.data)


@pytest.mark.parametrize('async_degree', [2, 4])
def test_async_degree(async_degree):
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)

    eqn = Eq(u.forward, u + 1)

    op0 = Operator(eqn, opt='noop')
    op1 = Operator(eqn, opt=('buffering', {'buf-async-degree': async_degree}))

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1
    assert buffers.pop().symbolic_shape[0] == async_degree

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_two_homogeneous_buffers():
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid, save=nt)
    v1 = TimeFunction(name='v', grid=grid, save=nt)

    eqns = [Eq(u.forward, u + v + u.backward + v.backward + 1.),
            Eq(v.forward, u + v + u.backward + v.backward + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')
    op2 = Operator(eqns, opt=('buffering', 'fuse'))

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 5
    assert len(retrieve_iteration_tree(op2)) == 3
    buffers = [i for i in FindSymbols().visit(op1.body) if i.is_Array and i._mem_heap]
    assert len(buffers) == 2

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1, v=v1)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


def test_two_heterogeneous_buffers():
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid, save=nt)
    v1 = TimeFunction(name='v', grid=grid, save=nt)

    for i in range(nt):
        u.data[i, :] = i
        u1.data[i, :] = i

    eqns = [Eq(u.forward, u + v + 1),
            Eq(v.forward, u + v + v.backward)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 5
    buffers = [i for i in FindSymbols().visit(op1.body) if i.is_Array and i._mem_heap]
    assert len(buffers) == 2

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1, v=v1)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


def test_over_one_subdomain():

    class sd0(SubDomain):
        name = 'd0'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 3, 3), y: ('middle', 3, 3)}

    s_d0 = sd0()
    nt = 10
    grid = Grid(shape=(10, 10), subdomains=(s_d0,))

    u = TimeFunction(name="u", grid=grid, save=nt)
    u1 = TimeFunction(name="u", grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    eqns = [Eq(v.forward, v + 1, subdomain=s_d0),
            Eq(u, v, subdomain=s_d0)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1, v=v1)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


def test_over_one_subdomain_read_only():

    class sd0(SubDomain):
        name = 'd0'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 3, 3), y: ('middle', 3, 3)}

    s_d0 = sd0()
    nt = 10
    grid = Grid(shape=(10, 10), subdomains=(s_d0,))

    u = TimeFunction(name="u", grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(nt):
        u.data[i, :] = i

    eqns = [Eq(v.forward, v + u + u.forward + 2., subdomain=s_d0)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, v=v1)

    assert np.all(v.data == v1.data)


def test_over_two_subdomains_illegal():
    """
    Cannot use buffering when:

        * an Eq writes to `f` using one set of SubDimensions
        * another Eq reads from `f` through a different set of SubDimensions

    as the second Eq may want to read unwritten memory (i.e., zero-valued)
    in the buffered Function, while with buffering it might end up reading values
    written in a previous iteration, thus breaking a storage-related RAW dependence.
    """

    class sd0(SubDomain):
        name = 'd0'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 3, 3), y: ('middle', 3, 3)}

    class sd1(SubDomain):
        name = 'd0'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', 2, 2)}

    s_d0 = sd0()
    s_d1 = sd1()
    nt = 10
    grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

    u = TimeFunction(name="u", grid=grid, save=nt)

    eqns = [Eq(u.forward, u + 1, subdomain=s_d0),
            Eq(u.forward, u.forward + 1, subdomain=s_d1)]

    with pytest.raises(CompilationError):
        Operator(eqns, opt='buffering')


@pytest.mark.xfail(reason="Cannot deal with non-overlapping SubDimensions yet")
def test_over_two_subdomains():

    class sd0(SubDomain):
        name = 'd0'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('left', 2), y: ('left', 2)}

    class sd1(SubDomain):
        name = 'd0'

        def define(self, dimensions):
            x, y = dimensions
            return {x: ('middle', 2, 2), y: ('middle', 2, 2)}

    s_d0 = sd0()
    s_d1 = sd1()
    nt = 10
    grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

    u = TimeFunction(name="u", grid=grid, save=nt)
    u1 = TimeFunction(name="u", grid=grid, save=nt)

    eqns = [Eq(u.forward, u + 1, subdomain=s_d0),
            Eq(u.forward, u.forward + u + 1, subdomain=s_d1)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_subdims():
    nt = 10
    grid = Grid(shape=(10, 10, 10))
    x, y, z = grid.dimensions
    xi = SubDimension.middle(name='xi', parent=x, thickness_left=2, thickness_right=2)
    yi = SubDimension.middle(name='yi', parent=y, thickness_left=2, thickness_right=2)
    zi = SubDimension.middle(name='zi', parent=z, thickness_left=2, thickness_right=2)

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)

    eqn = Eq(u.forward, u + 1).xreplace({x: xi, y: yi, z: zi})

    op0 = Operator(eqn, opt='noop')
    op1 = Operator(eqn, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    assert len([i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_conddim_backwards():
    nt = 10
    grid = Grid(shape=(4, 4))
    time_dim = grid.time_dim
    x, y = grid.dimensions

    factor = Constant(name='factor', value=2, dtype=np.int32)
    time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

    u = TimeFunction(name='u', grid=grid, time_order=0, save=nt, time_dim=time_sub,
                     space_order=0)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(u.save):
        u.data[i, :] = i

    eqns = [Eq(v.backward, v.backward + v + u + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 4
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_m=1, time_M=9)
    op1.apply(time_m=1, time_M=9, v=v1)

    assert np.all(v.data == v1.data)


def test_conddim_backwards_multi_slots():
    nt = 10
    grid = Grid(shape=(4, 4))
    time_dim = grid.time_dim
    x, y = grid.dimensions

    factor = Constant(name='factor', value=2, dtype=np.int32)
    time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

    u = TimeFunction(name='u', grid=grid, time_order=0, save=nt, time_dim=time_sub,
                     space_order=0)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(u.save):
        u.data[i, :] = i

    eqns = [Eq(v.backward, v + u.backward + u + u.forward + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 4
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_m=1, time_M=9)
    op1.apply(time_m=1, time_M=9, v=v1)

    assert np.all(v.data == v1.data)


def test_conddim_backwards_unstructured():
    nt = 10
    grid = Grid(shape=(4, 4))
    time_dim = grid.time_dim
    x, y = grid.dimensions

    factor = Constant(name='factor', value=2, dtype=np.int32)
    time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

    u = TimeFunction(name='u', grid=grid, space_order=0, time_order=0, save=nt,
                     time_dim=time_sub)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)

    for i in range(u.save):
        u.data[i, :] = i

    ub = u[time_sub - 1, x, y]
    ubb = u[time_sub - 2, x, y]
    uff = u[time_sub + 2, x, y]

    eqns = [Eq(v.backward, v.backward + v + ubb + ub + uff + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 4
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    # Note 1: cannot use time_m<4 or time_M>14 or there would be OOB accesses
    # due to `ubb` and `uff`, which read two steps away from the current point,
    # while `u` has in total `nt=10` entries (so last one has index 9). In
    # particular, at `time_M=14` we will read from `uff = u[time/factor + 2] =
    # u[14/2+2] = u[9]`, which is the last available entry in `u`. Likewise,
    # at `time_m=4` we will read from `ubb = u[time/factor - 2`] = u[4/2 - 2] =
    # u[0]`, which is clearly the last accessible entry in `u` while iterating
    # in the backward direction
    # Note 2: Given `factor=2`, we always write to `v` when `time % 2 == 0`, which
    # means that we always write to `v[t1] = v[(time+1)%2] = v[1]`, while `v[0]`
    # remains zero-valued. So the fact that the Eq is also reading from `v` is
    # only relevant to induce the backward iteration direction
    op0.apply(time_m=4, time_M=14)
    op1.apply(time_m=4, time_M=14, v=v1)

    assert np.all(v.data == v1.data)


def test_conddim_w_shifting():
    nt = 50
    grid = Grid(shape=(5, 5))
    time = grid.time_dim

    factor = Constant(name='factor', value=5, dtype=np.int32)
    t_sub = ConditionalDimension('t_sub', parent=time, factor=factor)
    save_shift = Constant(name='save_shift', dtype=np.int32)

    u = TimeFunction(name='u', grid=grid, time_order=0)
    u1 = TimeFunction(name='u', grid=grid, time_order=0)
    usave = TimeFunction(name='usave', grid=grid, space_order=0, time_order=0,
                         save=(int(nt//factor.data)), time_dim=t_sub)

    for i in range(usave.save):
        usave.data[i, :] = i

    eqns = Eq(u.forward, u + usave.subs(t_sub, t_sub - save_shift))

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 4
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    # From time_m=15 to time_M=35 with a factor=5 -- it means that, thanks
    # to t_sub, we enter the Eq exactly (35-15)/5 + 1 = 5 times. We set
    # save_shift=1 so instead of accessing the range usave[15/5:35/5+1],
    # we rather access the range usave[15/5-1:35:5], which means accessing
    # the usave values 2, 3, 4, 5, 6.
    op0.apply(time_m=15, time_M=35, save_shift=1)
    op1.apply(time_m=15, time_M=35, save_shift=1, u=u1)
    assert np.allclose(u.data, 20)
    assert np.all(u.data == u1.data)

    # Again, but with a different shift
    op1.apply(time_m=15, time_M=35, save_shift=-2, u=u1)
    assert np.allclose(u1.data, 20 + 35)


def test_multi_access():
    nt = 10
    grid = Grid(shape=(2, 2))

    u = TimeFunction(name='u', grid=grid, save=nt, space_order=0)
    v = TimeFunction(name='v', grid=grid)
    v1 = TimeFunction(name='v', grid=grid)
    w = TimeFunction(name='w', grid=grid)
    w1 = TimeFunction(name='w', grid=grid)

    for i in range(nt):
        u.data[i, :] = i

    eqns = [Eq(v.forward, v + u.forward + 1.),
            Eq(w.forward, w + u + 1.)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, v=v1, w=w1)

    assert np.all(v.data == v1.data)
    assert np.all(w.data == w1.data)


def test_issue_1901():
    grid = Grid(shape=(2, 2))
    time = grid.time_dim
    x, y = grid.dimensions

    usave = TimeFunction(name='usave', grid=grid, save=10, space_order=0)
    v = TimeFunction(name='v', grid=grid)

    eq = [Eq(v[time, x, y], usave)]

    op = Operator(eq, opt='buffering')

    trees = retrieve_iteration_tree(op)
    assert len(trees) == 3
    assert trees[2].root.dim is time
    assert not trees[2].root.is_Parallel
    assert trees[2].root.is_Sequential  # Obv


def test_everything():
    nt = 50
    grid = Grid(shape=(6, 6))
    x, y = grid.dimensions
    time = grid.time_dim
    xi = SubDimension.middle(name='xi', parent=x, thickness_left=2, thickness_right=2)
    yi = SubDimension.middle(name='yi', parent=y, thickness_left=2, thickness_right=2)

    factor = Constant(name='factor', value=5, dtype=np.int32)
    t_sub = ConditionalDimension('t_sub', parent=time, factor=factor)
    save_shift = Constant(name='save_shift', dtype=np.int32)

    u = TimeFunction(name='u', grid=grid, time_order=0)
    u1 = TimeFunction(name='u', grid=grid, time_order=0)
    va = TimeFunction(name='va', grid=grid, time_order=0,
                      save=(int(nt//factor.data)), time_dim=t_sub)
    vb = TimeFunction(name='vb', grid=grid, time_order=0,
                      save=(int(nt//factor.data)), time_dim=t_sub)

    for i in range(va.save):
        va.data[i, :] = i
        vb.data[i, :] = i*2 - 1

    vas = va.subs(t_sub, t_sub - save_shift)
    vasb = va.subs(t_sub, t_sub - 1 - save_shift)
    vasf = va.subs(t_sub, t_sub + 1 - save_shift)

    eqns = [Eq(u.forward, u + (vasb + vas + vasf)*2. + vb)]

    eqns = [e.xreplace({x: xi, y: yi}) for e in eqns]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len([i for i in FindSymbols().visit(op1.body) if i.is_Array
                and i._mem_heap]) == 2

    op0.apply(time_m=15, time_M=35, save_shift=0)
    op1.apply(time_m=15, time_M=35, save_shift=0, u=u1)

    assert np.all(u.data == u1.data)


@pytest.mark.parametrize('subdomain', ['domain', 'interior'])
@switchconfig(safe_math=True, condition=isinstance(configuration['platform'], AppleArm))
def test_stencil_issue_1915(subdomain):
    nt = 5
    grid = Grid(shape=(6, 6))

    u = TimeFunction(name='u', grid=grid, space_order=4)
    u1 = TimeFunction(name='u', grid=grid, space_order=4)
    usave = TimeFunction(name='usave', grid=grid, space_order=4, save=nt)
    usave1 = TimeFunction(name='usave', grid=grid, space_order=4, save=nt)

    subdomain = grid.subdomains[subdomain]

    eqns = [Eq(u.forward, u.dx + 1, subdomain=subdomain),
            Eq(usave, u.forward, subdomain=subdomain)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1, usave=usave1)

    assert np.all(u.data == u1.data)


@skipif('cpu64-icc')
@pytest.mark.parametrize('subdomain', ['domain', 'interior'])
def test_stencil_issue_1915_v2(subdomain):
    """
    Follow up of test_stencil_issue_1915, now with reverse propagation.
    """
    nt = 5
    grid = Grid(shape=(6, 6))
    time = grid.time_dim
    x, y = grid.dimensions

    u = TimeFunction(name='u', grid=grid, space_order=4)
    u1 = TimeFunction(name='u', grid=grid, space_order=4)

    usave = TimeFunction(name='usave', grid=grid, space_order=4, save=nt)

    for i in range(nt):
        usave.data[i] = i

    subdomain = grid.subdomains[subdomain]

    eqn = Eq(u, usave[time, x, y-1] + usave + usave[time, x, y+1], subdomain=subdomain)

    op0 = Operator(eqn, opt='noop')
    op1 = Operator(eqn, opt='buffering')

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_buffer_reuse():
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid)
    usave = TimeFunction(name='usave', grid=grid, save=nt)
    vsave = TimeFunction(name='vsave', grid=grid, save=nt)

    eqns = [Eq(u.forward, u + 1),
            Eq(usave, u.forward),
            Eq(vsave, u.forward + 1)]

    op = Operator(eqns, opt=('buffering', {'buf-reuse': True}))

    # Check generated code
    assert len(retrieve_iteration_tree(op)) == 5
    buffers = [i for i in FindSymbols().visit(op) if i.is_Array and i._mem_heap]
    assert len(buffers) == 1

    op.apply(time_M=nt-1)

    assert all(np.all(usave.data[i-1] == i) for i in range(1, nt + 1))
    assert all(np.all(vsave.data[i-1] == i + 1) for i in range(1, nt + 1))
