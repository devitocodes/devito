import pytest
import numpy as np
import scipy.sparse

from devito import (Grid, Function, TimeFunction, SparseTimeFunction, Operator, Eq,
                    MatrixSparseTimeFunction)
from devito.ir import Call, Callable, DummyExpr, Expression, FindNodes
from devito.operator import SymbolRegistry
from devito.passes import Graph, linearize


def test_basic():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid)
    u1 = TimeFunction(name='u', grid=grid)

    eqn = Eq(u.forward, u + 1)

    op0 = Operator(eqn)
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


@pytest.mark.parallel(mode=[(1, 'basic'), (1, 'diag2'), (1, 'full')])
def test_mpi():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dx2 + 1.)

    op0 = Operator(eqn)
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


def test_cire():
    grid = Grid(shape=(4, 4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dy.dy + 1.)

    op0 = Operator(eqn, opt=('advanced', {'cire-mingain': 0}))
    op1 = Operator(eqn, opt=('advanced', {'linearize': True, 'cire-mingain': 0}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


def test_nested_indexeds():
    grid = Grid(shape=(4, 4))
    t = grid.stepping_dim
    x, y = grid.dimensions

    f = Function(name='f', grid=grid, dtype=np.int32)
    g = Function(name='g', grid=grid, dimensions=(x,), shape=(4,), dtype=np.int32)
    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u[t, f[g[x], g[x]], y] + 1.)

    op0 = Operator(eqn)
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


def test_interpolation():
    nt = 10
    grid = Grid(shape=(4, 4))

    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
    rec = SparseTimeFunction(name='rec', grid=grid, npoint=1, nt=nt)
    u = TimeFunction(name="u", grid=grid, time_order=2)
    u1 = TimeFunction(name="u", grid=grid, time_order=2)

    src.data[:] = 1.

    eqns = ([Eq(u.forward, u + 1)] +
            src.inject(field=u.forward, expr=src) +
            rec.interpolate(expr=u.forward))

    op0 = Operator(eqns, opt='advanced')
    op1 = Operator(eqns, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_interpolation_msf():
    grid = Grid(shape=(4, 4))

    r = 2  # Because we interpolate across 2 neighbouring points in each dimension
    nt = 10

    m0 = TimeFunction(name="m0", grid=grid, space_order=0, save=nt, time_order=0)
    m1 = TimeFunction(name="m1", grid=grid, space_order=0, save=nt, time_order=0)

    mat = scipy.sparse.coo_matrix((0, 0), dtype=np.float32)
    sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=mat, nt=nt)

    eqns = sf.inject(field=m0.forward, expr=sf.dt2)
    eqns += sf.inject(field=m1.forward, expr=sf.dt2)

    op0 = Operator(eqns)
    op1 = Operator(eqns, opt=('advanced', {'linearize': True}))

    assert 'm0L0' in str(op1)

    # There used to be a bug causing the jit compilation to fail because of
    # the writing to `const int` variables
    assert op0.cfunction
    assert op1.cfunction


@pytest.mark.parallel(mode=[(1, 'diag2')])
def test_codegen_quality0():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dx2 + 1.)

    op = Operator(eqn, opt=('advanced', {'linearize': True}))

    assert 'uL0' in str(op)

    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 6
    assert all('const unsigned int' in str(i) for i in exprs[:-2])

    # Only four access macros necessary, namely `uL0`, `aL0`, `bufL0`, `bufL1` (the
    # other three obviously are _POSIX_C_SOURCE, START_TIMER, STOP_TIMER)
    assert len(op._headers) == 7


def test_codegen_quality1():
    grid = Grid(shape=(4, 4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dy.dy + 1.)

    op = Operator(eqn, opt=('advanced', {'linearize': True, 'cire-mingain': 0}))

    assert 'uL0' in str(op)

    # 11 expressions in total are expected, 8 of which are for the linearized accesses
    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 11
    assert all('const unsigned int' in str(i) for i in exprs[:-3])
    assert all('const unsigned int' not in str(i) for i in exprs[-3:])

    # Only two access macros necessary, namely `uL0` and `r1L0` (the other five
    # obviously are _POSIX_C_SOURCE, MIN, MAX, START_TIMER, STOP_TIMER)
    assert len(op._headers) == 7


def test_pow():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, 1./(u*u) + 1.)

    op = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Make sure linearize() doesn't cause `a*a` -> `Pow(a, 2)`
    assert 'uL0' in str(op)
    expr = FindNodes(Expression).visit(op)[-1].expr
    assert expr.rhs.is_Add
    assert expr.rhs.args[1].is_Pow
    assert expr.rhs.args[1].args[0].is_Mul
    assert expr.rhs.args[1].args[1] == -1


def test_different_halos():
    grid = Grid(shape=(8, 8, 8))

    f = Function(name='f', grid=grid, space_order=8)
    g = Function(name='g', grid=grid, space_order=16)
    u = TimeFunction(name='u', grid=grid, space_order=12)
    u1 = TimeFunction(name='u', grid=grid, space_order=12)

    f.data[:] = 1.
    g.data[:] = 2.

    eqn = Eq(u.forward, u + f + g + 1)

    op0 = Operator(eqn)
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=4)
    op1.apply(time_M=4, u=u1)

    assert np.all(u.data == u1.data)


def test_strides_forwarding():
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid)

    bar = Callable('bar', DummyExpr(f[0, 0], 0), 'void', parameters=[f.indexed])
    call = Call(bar.name, [f.indexed])
    foo = Callable('foo', call, 'void', parameters=[f])

    # Emulate what the compiler would do
    graph = Graph(foo)
    graph.efuncs['bar'] = bar

    linearize(graph, mode=True, sregistry=SymbolRegistry())

    # Since `f` is passed via `f.indexed`, we expect the stride exprs to be
    # lifted in `foo` and then passed down to `bar` as arguments
    foo = graph.root
    bar = graph.efuncs['bar']

    assert foo.body.body[0].write.name == 'y_fsz0'
    assert foo.body.body[2].write.name == 'y_stride0'
    assert len(foo.body.body[4].arguments) == 2

    assert len(bar.parameters) == 2
    assert bar.parameters[1].name == 'y_stride0'
    assert len(bar.body.body) == 1
