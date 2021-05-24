import pytest
import numpy as np

from devito import (Grid, TimeFunction, SparseTimeFunction, Operator, Eq,
                    SubDimension, SubDomain, configuration)
from devito.ir import FindSymbols, retrieve_iteration_tree
from devito.exceptions import InvalidOperator


def test_read_write():
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)

    eqn = Eq(u.forward, u + 1)

    op0 = Operator(eqn, opt='noop')
    op1 = Operator(eqn, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 2
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
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
    assert len(retrieve_iteration_tree(op1)) == 1
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
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
    assert len(retrieve_iteration_tree(op1)) == 2
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
    assert len(buffers) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, v=v1)

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
    assert len(retrieve_iteration_tree(op1)) == 2
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
    assert len(buffers) == 1

    op0.apply(time_m=1)
    op1.apply(time_m=1, v=v1)

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
    assert len(retrieve_iteration_tree(op1)) == 2
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
    assert len(buffers) == 1
    assert buffers.pop().symbolic_shape[0] == async_degree

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_two_heterogeneous_buffers():
    nt = 10
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, save=nt)
    u1 = TimeFunction(name='u', grid=grid, save=nt)
    v = TimeFunction(name='v', grid=grid, save=nt)
    v1 = TimeFunction(name='v', grid=grid, save=nt)

    eqns = [Eq(u.forward, u + v + 1),
            Eq(v.forward, u + v + v.backward)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) == 3
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
    assert len(buffers) == 2

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1, v=v1)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


def test_over_injection():
    nt = 10
    grid = Grid(shape=(4, 4))

    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
    rec = SparseTimeFunction(name='rec', grid=grid, npoint=1, nt=nt)
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2, save=nt)
    u1 = TimeFunction(name="u", grid=grid, time_order=2, space_order=2, save=nt)

    src.data[:] = 1.

    eqns = ([Eq(u.forward, u + 1)] +
            src.inject(field=u.forward, expr=src) +
            rec.interpolate(expr=u.forward))

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt='buffering')

    # Check generated code
    assert len(retrieve_iteration_tree(op1)) ==\
        5 + bool(configuration['language'] != 'C')
    buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
    assert len(buffers) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


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

    try:
        Operator(eqns, opt='buffering')
    except InvalidOperator:
        assert True
    except:
        assert False


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


def test_subdimensions():
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
    assert len(retrieve_iteration_tree(op1)) == 2
    assert len([i for i in FindSymbols().visit(op1) if i.is_Array]) == 1

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)
