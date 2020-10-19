import pytest
import numpy as np

from devito import Grid, Eq, Function, TimeFunction, Operator
from devito.ir import FindSymbols, retrieve_iteration_tree


def test_basic():
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


def test_unread_buffered_function():
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
