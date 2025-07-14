import pytest
import numpy as np
import scipy.sparse

from devito import (Grid, Function, TimeFunction, SparseTimeFunction, Operator, Eq,
                    Inc, MatrixSparseTimeFunction, sin, switchconfig, configuration)
from devito.ir import Call, Callable, DummyExpr, Expression, FindNodes, SymbolRegistry
from devito.passes import Graph, linearize, generate_macros
from devito.types import Array, Bundle, DefaultDimension


def test_basic():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid)
    u1 = TimeFunction(name='u', grid=grid)

    eqn = Eq(u.forward, u + 1)

    op0 = Operator(eqn, opt=('advanced', {'linearize': False}))
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


@pytest.mark.parallel(mode=[(1, 'basic'), (1, 'diag2'), (1, 'full')])
def test_mpi(mode):
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dx2 + 1.)

    op0 = Operator(eqn, opt=('advanced', {'linearize': False}))
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.allclose(u.data, u1.data, rtol=1e-5)


def test_cire():
    grid = Grid(shape=(4, 4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dy.dy + 1.)

    op0 = Operator(eqn, opt=('advanced', {'linearize': False, 'cire-mingain': 0}))
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

    op0 = Operator(eqn, opt=('advanced', {'linearize': False}))
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

    op0 = Operator(eqns, opt=('advanced', {'linearize': False}))
    op1 = Operator(eqns, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=nt-2)
    op1.apply(time_M=nt-2, u=u1)

    assert np.all(u.data == u1.data)


def test_interpolation_enforcing_int64_indexing():
    grid = Grid(shape=(4, 4))

    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=10)
    rec = SparseTimeFunction(name='rec', grid=grid, npoint=1, nt=10)
    u = TimeFunction(name="u", grid=grid, time_order=2)

    eqns = ([Eq(u.forward, u + 1)] +
            src.inject(field=u.forward, expr=src) +
            rec.interpolate(expr=u.forward))

    op = Operator(eqns, opt=('advanced', {'linearize': True,
                                          'index-mode': 'int32'}))

    # Check generated code
    assert 'uL0' in str(op)
    assert 'int x_stride0' in str(op)  # for `u`
    assert 'long p_rec_stride0' in str(op)  # for `rec`
    assert 'long p_src_stride0' in str(op)  # for `src`


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

    op0 = Operator(eqns, opt=('advanced', {'linearize': False}))
    op1 = Operator(eqns, opt=('advanced', {'linearize': True}))

    assert 'm0L0' in str(op1)

    # There used to be a bug causing the jit compilation to fail because of
    # the writing to `const int` variables
    assert op0.cfunction
    assert op1.cfunction


@pytest.mark.parallel(mode=[(2, 'diag2')])
def test_codegen_quality0(mode):
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dx2 + 1.)

    op = Operator(eqn, opt=('advanced', {'linearize': True}))
    assert 'uL0' in str(op)

    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 6
    assert all('const int' in str(i) for i in exprs[:-2])

    # Only four access macros necessary, namely `uL0`, `bufL0`, `bufL1`
    # for the efunc args
    # (the other three obviously are _POSIX_C_SOURCE, START, STOP)
    assert len(op.headers) == 6


def test_codegen_quality1():
    grid = Grid(shape=(4, 4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dy.dy + 1.)

    op = Operator(eqn, opt=('advanced', {'linearize': True, 'cire-mingain': 0}))

    assert 'uL0' in str(op)

    # 11 expressions in total are expected, 8 of which are for the linearized accesses
    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 11
    assert all('const int' in str(i) for i in exprs[:-3])
    assert all('const int' not in str(i) for i in exprs[-3:])

    # Only two access macros necessary, namely `uL0` and `r1L0` (the other five
    # obviously are _POSIX_C_SOURCE, MIN, MAX, START, STOP)
    assert len(op.headers) == 6


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

    op0 = Operator(eqn, opt=('advanced', {'linearize': False}))
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=4)
    op1.apply(time_M=4, u=u1)

    assert np.all(u.data == u1.data)


def test_unsubstituted_indexeds():
    """
    This issue emerged in the context of PR #1828, after the introduction
    of Uxreplace to substitute Indexeds with FIndexeds. Basically what happened
    was that `FindSymbols('indexeds')` was missing syntactically identical
    objects that however look the same. For example, as in this test,
    we end up with two `r0[x, y, z]`, but the former's `x` and `y` are
    SpaceDimensions, while the latter's are BlockDimensions. This means
    that the two objects, while looking identical, are different, and in
    partical they hash differently, hence we need two entries in a mapper
    to perform an Uxreplace. But FindSymbols made us detect only one entry...
    """
    grid = Grid(shape=(8, 8, 8))

    f = Function(name='f', grid=grid)
    p = TimeFunction(name='p', grid=grid)
    p1 = TimeFunction(name='p', grid=grid)

    f.data[:] = 0.12
    p.data[:] = 1.
    p1.data[:] = 1.

    eq = Eq(p.forward, sin(f)*p*f)

    op0 = Operator(eq, opt=('advanced', {'linearize': False}))
    op1 = Operator(eq, opt=('advanced', {'linearize': True}))

    # NOTE: Eventually we compare the numerical output, but truly the most
    # import check is implicit to op1.apply, and it's the fact that op1
    # actually jit-compiles successfully, meaning that all substitutions
    # were performed correctly
    op0.apply(time_M=2)
    op1.apply(time_M=2, p=p1)

    assert np.allclose(p.data, p1.data, rtol=1e-7)


def test_strides_forwarding0():
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid)

    bar = Callable('bar', DummyExpr(f[0, 0], 0), 'void', parameters=[f.indexed])
    call = Call(bar.name, [f.indexed])
    foo = Callable('foo', call, 'void', parameters=[f])

    # Emulate what the compiler would do
    graph = Graph(foo)
    graph.efuncs['bar'] = bar

    linearize(graph, callback=True, options={'index-mode': 'int32'},
              sregistry=SymbolRegistry())

    # Since `f` is passed via `f.indexed`, we expect the stride exprs to be
    # lifted in `foo` and then passed down to `bar` as arguments
    foo = graph.root
    bar = graph.efuncs['bar']

    assert foo.body.strides[0].write.name == 'y_fsz0'
    assert foo.body.strides[2].write.name == 'y_stride0'
    assert len(foo.body.body[0].arguments) == 2

    assert len(bar.parameters) == 2
    assert bar.parameters[1].name == 'y_stride0'
    assert len(bar.body.body) == 1


def test_strides_forwarding1():
    grid = Grid(shape=(4, 4))

    a = Array(name='a', dimensions=grid.dimensions, shape=grid.shape)

    bar = Callable('bar', DummyExpr(a[0, 0], 0), 'void', parameters=[a.indexed])
    call = Call(bar.name, [a.indexed])
    foo = Callable('foo', call, 'void', parameters=[a])

    # Emulate what the compiler would do
    graph = Graph(foo)
    graph.efuncs['bar'] = bar

    linearize(graph, callback=True, options={'index-mode': 'int32'},
              sregistry=SymbolRegistry())

    # `a` is passed via `a.indexed`, so the stride exprs are expected to be
    # placed in `foo` and then passed down to `bar` as arguments
    foo = graph.root
    bar = graph.efuncs['bar']

    assert len(foo.body.body) == 1
    assert foo.body.body[0].is_Call
    assert len(foo.body.strides) == 3
    assert foo.body.strides[0].write.name == 'y_fsz0'
    assert foo.body.strides[2].write.name == 'y_stride0'

    assert len(bar.parameters) == 2
    assert bar.parameters[0].name == 'a'
    assert bar.parameters[1].name == 'y_stride0'
    assert len(bar.body.body) == 1


def test_strides_forwarding2():
    grid = Grid(shape=(4, 4))

    a = Function(name='a', grid=grid)

    # Construct the following Calls tree
    # root
    #   foo0
    #     bar0
    #   foo1
    #     bar1
    bar0 = Callable('bar0', DummyExpr(a[0, 0], 0), 'void', parameters=[a.indexed])
    call = Call(bar0.name, [a.indexed])
    foo0 = Callable('foo0', call, 'void', parameters=[a])

    bar1 = Callable('bar1', DummyExpr(a[0, 0], 0), 'void', parameters=[a.indexed])
    call = Call(bar1.name, [a.indexed])
    foo1 = Callable('foo1', call, 'void', parameters=[a])

    calls = [Call(foo0.name, a), Call(foo1.name, a)]
    root = Callable('root', calls, 'void', parameters=[a])

    # Emulate what the compiler would do
    graph = Graph(root)
    graph.efuncs['bar0'] = bar0
    graph.efuncs['bar1'] = bar1
    graph.efuncs['foo0'] = foo0
    graph.efuncs['foo1'] = foo1

    linearize(graph, callback=True, options={'index-mode': 'int32'},
              sregistry=SymbolRegistry())

    # Both foo's are expected to define `a`!
    root = graph.root
    foo0 = graph.efuncs['foo0']
    foo1 = graph.efuncs['foo1']
    bar0 = graph.efuncs['bar0']
    bar1 = graph.efuncs['bar1']

    assert all(i.is_Call for i in root.body.body)

    for foo in [foo0, foo1]:
        assert foo.body.strides[0].write.name == 'y_fsz0'
        assert foo.body.strides[2].write.name == 'y_stride0'
        assert len(foo.body.body[0].arguments) == 2

    for bar in [bar0, bar1]:
        assert len(bar.parameters) == 2
        assert bar.parameters[1].name == 'y_stride0'
        assert len(bar.body.body) == 1


def test_strides_forwarding3():
    grid = Grid(shape=(4, 4))

    a = Function(name='a', grid=grid)
    a1 = a._rebuild(name='f0', alias=True)

    # Construct the following Calls tree
    # foo
    #   bar
    bar = Callable('bar', DummyExpr(a1[0, 0], 0), 'void', parameters=[a1.indexed])
    call = Call(bar.name, [a.indexed])
    root = Callable('foo', call, 'void', parameters=[a])

    # Emulate what the compiler would do
    graph = Graph(root)
    graph.efuncs['bar'] = bar

    linearize(graph, callback=True, options={'index-mode': 'int64'},
              sregistry=SymbolRegistry())

    # Both foo's are expected to define `a`!
    root = graph.root
    bar = graph.efuncs['bar']

    assert root.body.strides[0].write.name == 'y_fsz0'
    assert root.body.strides[0].write.dtype is np.int64
    assert root.body.strides[2].write.name == 'y_stride0'
    assert root.body.strides[2].write.dtype is np.int64

    assert bar.parameters[1].name == 'y_stride0'


def test_strides_forwarding4():
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid)

    # Construct the following Calls tree
    # foo
    #   bar
    call0 = Call('sin', (f[0, 0],))
    bar = Callable('bar', call0, 'void', parameters=[f.indexed])
    call1 = Call(bar.name, [f.indexed])
    root = Callable('foo', call1, 'void', parameters=[f])

    # Emulate what the compiler would do
    graph = Graph(root)
    graph.efuncs['bar'] = bar

    linearize(graph, callback=True, options={'index-mode': 'int64'},
              sregistry=SymbolRegistry())

    root = graph.root
    bar = graph.efuncs['bar']

    assert root.body.strides[0].write.name == 'y_fsz0'
    assert root.body.strides[2].write.name == 'y_stride0'
    assert root.body.body[0].arguments[1].name == 'y_stride0'
    assert bar.parameters[1].name == 'y_stride0'


def test_issue_1838():
    """
    MFE for issue #1838.
    """
    space_order = 4

    grid = Grid(shape=(4, 4, 4))

    f = Function(name='f', grid=grid, space_order=space_order)
    b = Function(name='b', grid=grid, space_order=space_order)
    p0 = TimeFunction(name='p0', grid=grid, space_order=space_order)
    p1 = TimeFunction(name='p0', grid=grid, space_order=space_order)

    f.data[:] = 2.1
    b.data[:] = 1.3
    p0.data[:, 2, 2, 2] = .3
    p1.data[:, 2, 2, 2] = .3

    eq = Eq(p0.forward, (sin(b)*p0.dx).dx + (sin(b)*p0.dx).dy + (sin(b)*p0.dx).dz + p0)

    op0 = Operator(eq, opt=('advanced', {'linearize': False}))
    op1 = Operator(eq, opt=('advanced', {'linearize': True}))

    op0.apply(time_M=3, dt=1.)
    op1.apply(time_M=3, dt=1., p0=p1)

    # Check generated code
    assert "r0L0(x,y,z) r0[(x)*y_stride1 + (y)*z_stride1 + (z)]" in str(op1)
    assert "r4L0(x,y,z) r4[(x)*y_stride2 + (y)*z_stride1 + (z)]" in str(op1)

    assert np.allclose(p0.data, p1.data, rtol=1e-6)


def test_memspace_stack():
    grid = Grid(shape=(4, 4))
    dimensions = grid.dimensions

    a = Array(name='a', dimensions=dimensions, dtype=grid.dtype, scope='stack')
    u = TimeFunction(name='u', grid=grid)

    eqn = Eq(u.forward, u + a.indexify() + 1.)

    op = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' in str(op)
    assert 'aL0' not in str(op)
    assert 'a[x][y]' in str(op)


def test_call_retval_indexed():
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid)
    g = Function(name='v', grid=grid)

    call = Call('bar', [f.indexed], retobj=g.indexify())
    foo = Callable('foo', call, 'void', parameters=[f])

    # Emulate what the compiler would do
    graph = Graph(foo)

    sregistry = SymbolRegistry()
    linearize(graph, callback=True, options={'index-mode': 'int64'},
              sregistry=sregistry)
    generate_macros(graph, sregistry=sregistry)

    foo = graph.root

    assert foo.body.strides[0].write.name == 'y_fsz0'
    assert foo.body.strides[2].write.name == 'y_stride0'
    assert str(foo.body.body[-1]) == 'vL0(x, y) = bar(f);'


def test_bundle():
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid)
    g = Function(name='g', grid=grid)

    fg = Bundle(name='fg', components=(f, g))

    bar = Callable('bar', DummyExpr(fg[0, 0, 0], 0), 'void', parameters=[fg.indexed])
    call = Call('bar', [fg.indexed])
    foo = Callable('foo', call, 'void', parameters=[f, g])

    # Emulate what the compiler would do
    graph = Graph(foo)
    graph.efuncs['bar'] = bar

    linearize(graph, callback=True, options={'index-mode': 'int64'},
              sregistry=SymbolRegistry())

    foo = graph.root
    bar = graph.efuncs['bar']

    # Instead of propagating the components, we propagate the necessary strides!
    assert f not in bar.parameters
    assert g not in bar.parameters

    assert foo.body.strides[0].write.name == 'y_fsz0'
    y_stride0 = foo.body.strides[2].write
    assert y_stride0.name == 'y_stride0'
    assert y_stride0 in bar.parameters


def test_inc_w_default_dims():
    grid = Grid(shape=(5, 6))
    k = DefaultDimension(name="k", default_value=7)
    x, y = grid.dimensions

    f = Function(name="f", grid=grid, dimensions=(x, y, k),
                 shape=grid.shape + (k._default_value,))
    g = Function(name="g", grid=grid)

    f.data[:] = 1

    eq = Inc(g, f)

    op = Operator(eq, opt=('advanced', {'linearize': True}))

    # NOTE: Eventually we compare the numerical output, but truly the most
    # import check is implicit to op1.apply, and it's the fact that op1
    # actually jit-compiles successfully, with the openmp reduction clause
    # getting "linearized" just like everything else in the Operator
    op.apply()

    assert np.all(g.data == 7)

    # Similar, but now reducing into a specific item along one Dimension
    g.data[:] = 0.

    eq = Inc(g[3, y], f)

    op = Operator(eq, opt=('advanced', {'linearize': True}))

    op.apply()

    assert np.all(g.data[:3] == 0)
    assert f.shape[0]*k._default_value == 35
    assert np.all(g.data[3] == f.shape[0]*k._default_value)
    assert np.all(g.data[4:] == 0)


@pytest.mark.parametrize('autopadding', [False, True, np.float64])
def test_different_dtype(autopadding):

    @switchconfig(autopadding=autopadding)
    def _test_different_dtype():
        space_order = 4

        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid, space_order=space_order)
        b = Function(name='b', grid=grid, space_order=space_order, dtype=np.float64)

        f.data[:] = 2.1
        b.data[:] = 1.3

        eq = Eq(f, b.dx + f.dy)

        op1 = Operator(eq, opt=('advanced', {'linearize': True}))

        # Check generated code has different strides for different dtypes
        assert "bL0(x,y) b[(x)*y_stride0 + (y)]" in str(op1)
        assert "L0(x,y) f[(x)*y_stride0 + (y)]" in str(op1)

    _test_different_dtype()


@pytest.mark.parametrize('order', [2, 4])
def test_int64_array(order):

    grid = Grid(shape=(4, 4))
    f = Function(name='f', grid=grid, space_order=order)

    a = Array(name='a', dimensions=grid.dimensions, shape=grid.shape,
              halo=f.halo)

    eqs = [Eq(f, a.indexify() + 1)]
    op = Operator(eqs, opt=('advanced', {'linearize': True, 'index-mode': 'int64'}))
    if 'CXX' in configuration['language']:
        long = 'static_cast<long>'
        assert f'({2*order} + {long}(y_size))*({2*order} + {long}(x_size)))' in str(op)
    else:
        long = '(long)'
        assert f'({2*order} + {long}y_size)*({2*order} + {long}x_size))' in str(op)
