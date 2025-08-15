from ctypes import c_void_p

import sympy
import pytest
import numpy as np

from sympy import Expr, Number, Symbol
from devito import (Constant, Dimension, Grid, Function, solve, TimeFunction, Eq,  # noqa
                    Operator, SubDimension, norm, Le, Ge, Gt, Lt, Abs, sin, cos,
                    Min, Max, Real, Imag, Conj, SubDomain, configuration)
from devito.finite_differences.differentiable import SafeInv, Weights, Mul
from devito.ir import Expression, FindNodes, ccode
from devito.mpi.halo_scheme import HaloTouch
from devito.symbolics import (
    retrieve_functions, retrieve_indexed, evalrel, CallFromPointer, Cast, # noqa
    DefFunction, FieldFromPointer, INT, FieldFromComposite, IntDiv, Namespace,
    Rvalue, ReservedWord, ListInitializer, uxreplace, pow_to_mul,
    retrieve_derivatives, BaseCast, SizeOf, VectorAccess
)
from devito.tools import as_tuple, CustomDtype
from devito.types import (Array, Bundle, FIndexed, LocalObject, Object,
                          ComponentAccess, StencilDimension, Symbol as dSymbol)
from devito.types.basic import AbstractSymbol


def test_float_indices():
    """
    Test that indices only contain Integers.
    """
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + 1.0 * x.spacing
    u = Function(name="u", grid=grid, space_order=2)
    indices = u.subs({x: x0}).indexify().indices[0]
    assert len(indices.atoms(sympy.Float)) == 0
    assert indices == x + 1

    indices = u.subs({x: 1.0}).indexify().indices[0]
    assert len(indices.atoms(sympy.Float)) == 0
    assert indices == 1


def test_func_of_indices():
    """
    Test that origin is correctly processed with functions
    """
    grid = Grid((10,))
    x = grid.dimensions[0]
    u = Function(name="u", grid=grid, space_order=2, staggered=x)
    us = u.subs({u.indices[0]: INT(Abs(u.indices[0]))})
    assert us.indices[0] == INT(Abs(x + x.spacing/2))
    assert us.indexify().indices[0] == INT(Abs(x))


@pytest.mark.parametrize('dtype,expected', [
    (np.float32, "float r0 = 1.0F/h_x;"),
    (np.float64, "double r0 = 1.0/h_x;")
])
def test_floatification_issue_1627(dtype, expected):
    """
    MFE for issue #1627.
    """
    grid = Grid(shape=(10, 10), dtype=dtype)
    x, y = grid.dimensions

    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, ((u/x.spacing) + 2.0)/x.spacing)

    op = Operator(eq, opt=('advanced', {'linearize': False}))

    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 2
    assert str(exprs[0]) == expected


def test_sympy_assumptions():
    """
    Ensure that AbstractSymbol assumptions are set correctly and
    preserved during rebuild.
    """
    s0 = AbstractSymbol('s')
    s1 = AbstractSymbol('s', nonnegative=True, integer=False, real=True)

    assert s0.is_negative is None
    assert s0.is_positive is None
    assert s0.is_integer is None
    assert s0.is_real is True
    assert s1.is_negative is False
    assert s1.is_positive is True
    assert s1.is_integer is False
    assert s1.is_real is True

    s0r = s0._rebuild()
    s1r = s1._rebuild()

    assert s0.assumptions0 == s0r.assumptions0
    assert s0 == s0r

    assert s1.assumptions0 == s1r.assumptions0
    assert s1 == s1r


def test_modified_sympy_assumptions():
    """
    Check that sympy assumptions can be changed during a rebuild.
    """
    s0 = AbstractSymbol('s')
    s1 = AbstractSymbol('s', nonnegative=True, integer=False, real=True)

    s2 = s0._rebuild(nonnegative=True, integer=False, real=True)

    assert s2.assumptions0 == s1.assumptions0
    assert s2 == s1


def test_real():
    for dtype in [np.float32, np.complex64]:
        c = Constant(name='c', dtype=dtype)
        assert c.is_real is not np.iscomplexobj(dtype(0))
        assert c.is_imaginary is np.iscomplexobj(dtype(0))
        f = Function(name='f', dtype=dtype, grid=Grid((11,)))
        assert f.is_real is not np.iscomplexobj(dtype(0))
        assert f.is_imaginary is np.iscomplexobj(dtype(0))
        s = dSymbol(name='s', dtype=dtype)
        assert s.is_real is not np.iscomplexobj(dtype(0))
        assert s.is_imaginary is np.iscomplexobj(dtype(0))


def test_constant():
    c = Constant(name='c')

    assert c.free_symbols == {c}
    assert c.bound_symbols == set()


def test_dimension():
    d = Dimension(name='d')

    assert d.free_symbols == {d}
    assert d.bound_symbols == {d.symbolic_min, d.symbolic_max, d.symbolic_size}


def test_subdimension():
    d = Dimension(name='d')

    di = SubDimension.middle(name='di', parent=d, thickness_left=4, thickness_right=4)
    assert di.free_symbols == {di}
    assert di.bound_symbols == {d.symbolic_min, d.symbolic_max} | set(di.thickness)

    dl = SubDimension.left(name='dl', parent=d, thickness=4)
    assert dl.free_symbols == {dl}
    assert dl.bound_symbols == {d.symbolic_min, dl.thickness.left}

    dr = SubDimension.right(name='dr', parent=d, thickness=4)
    assert dr.free_symbols == {dr}
    assert dr.bound_symbols == {d.symbolic_max, dr.thickness.right}


def test_timefunction():
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions
    t = grid.stepping_dim

    f = TimeFunction(name='f', grid=grid)

    assert f.free_symbols == {t, x, y}
    assert f.bound_symbols == ({f.indexed, f._C_symbol} |
                               t.bound_symbols | x.bound_symbols | y.bound_symbols)


def test_indexed():
    grid = Grid(shape=(10, 10))
    x, y = grid.dimensions

    u = Function(name='u', grid=grid)

    assert u.free_symbols == {x, y}
    assert u.indexed.free_symbols == {u.indexed}

    # Test reconstruction
    assert u.indexed.func() == u.indexed
    ru = u.indexed.func(*u.indexed.args)
    assert ru is not u.indexed
    assert ru == u.indexed
    assert ru.function is u

    ub = Array(name='ub', dtype=u.dtype, dimensions=u.dimensions)

    assert ub.free_symbols == {x, y}
    assert ub.indexed.free_symbols == {ub.indexed}


def test_indexed_staggered():
    grid = Grid(shape=(10, 10))
    x, y = grid.dimensions
    hx, hy = x.spacing, y.spacing

    u = Function(name='u', grid=grid, staggered=(x, y))
    u0 = u.subs({x: 1, y: 2})
    assert u0.indices == (1 + hx / 2, 2 + hy / 2)
    assert u0.indexify().indices == (1, 2)


def test_bundle():
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid)
    g = Function(name='g', grid=grid)

    fg = Bundle(name='fg', components=(f, g))

    # Test reconstruction
    fg._rebuild().components == fg.components


def test_call_from_pointer():
    s = Symbol('s')

    # Test construction
    cfp0 = CallFromPointer('foo', 's')
    cfp1 = CallFromPointer('foo', s)
    assert str(cfp0) == str(cfp1) == 's->foo()'
    assert cfp0 != cfp1  # As `cfc0`'s underlying Symbol is a types.Symbol
    cfp2 = CallFromPointer('foo', s, [0])
    assert str(cfp2) == 's->foo(0)'
    cfp3 = CallFromPointer('foo', s, [0, 'a'])
    assert str(cfp3) == 's->foo(0, a)'

    # Test hashing
    assert hash(cfp0) != hash(cfp1)  # Same reason as above
    assert hash(cfp0) != hash(cfp2)
    assert hash(cfp2) != hash(cfp3)

    # Test reconstruction
    cfp4 = cfp3.func(*cfp3.args)
    assert cfp3 == cfp4

    # Free symbols
    a = cfp3.args[2][1]
    assert str(a) == 'a'
    assert cfp3.free_symbols == {s, a}


def test_field_from_pointer():
    s = Symbol('s')

    # Test construction
    ffp0 = FieldFromPointer('foo', 's')
    ffp1 = FieldFromPointer('foo', s)
    assert str(ffp0) == str(ffp1) == 's->foo'
    assert ffp0 != ffp1  # As `ffc0`'s underlying Symbol is a types.Symbol
    ffp2 = FieldFromPointer('bar', 's')
    assert ffp0 != ffp2

    # Test hashing
    assert hash(ffp0) != hash(ffp1)  # Same reason as above
    assert hash(ffp0) != hash(ffp2)

    # Test reconstruction
    ffp3 = ffp0.func(*ffp0.args)
    assert ffp0 == ffp3

    # Free symbols
    assert ffp1.free_symbols == {s}


def test_field_from_composite():
    s = Symbol('s')

    # Test construction
    ffc0 = FieldFromComposite('foo', 's')
    ffc1 = FieldFromComposite('foo', s)
    assert str(ffc0) == str(ffc1) == 's.foo'
    assert ffc0 != ffc1  # As `ffc0`'s underlying Symbol is a types.Symbol
    ffc2 = FieldFromComposite('bar', 's')
    assert ffc0 != ffc2

    # Test hashing
    assert hash(ffc0) != hash(ffc1)  # Same reason as above
    assert hash(ffc0) != hash(ffc2)

    # Test reconstruction
    ffc3 = ffc0.func(*ffc0.args)
    assert ffc0 == ffc3

    # Free symbols
    assert ffc1.free_symbols == {s}


def test_extended_sympy_arithmetic():
    # NOTE: `s` gets turned into a devito.Symbol, whose dtype
    # defaults to np.int32
    cfp = CallFromPointer('foo', 's')
    ffp = FieldFromPointer('foo', 's')
    ffc = FieldFromComposite('foo', 's')

    assert ccode(cfp + 1) == 's->foo() + 1'
    assert ccode(ffp + 1) == 's->foo + 1'
    assert ccode(ffc + 1) == 's.foo + 1'

    grid = Grid(shape=(4, 4))
    u = Function(name='u', grid=grid)
    foo = FieldFromPointer('foo', u._C_symbol)
    assert ccode(-1 + foo) == 'u_vec->foo - 1'

    # Now wrapping an object whose dtype is not numeric, hence
    # noncommutative
    o = Object(name='o', dtype=c_void_p)
    bar = FieldFromPointer('bar', o)
    assert ccode(-1 + bar) == '-1 + o->bar'


def test_integer_abs():
    i1 = Dimension(name="i1")
    assert ccode(Abs(i1 - 1)) == "abs(i1 - 1)"
    assert ccode(Abs(i1 - .5)) == "fabsf(i1 - 5.0e-1F)"
    assert ccode(
        Abs(i1 - Constant('half', dtype=np.float64, default_value=0.5))
    ) == "fabs(i1 - half)"


def test_cos_vs_cosf():
    a = dSymbol('a', dtype=np.float32)
    assert ccode(cos(a)) == "cosf(a)"

    b = dSymbol('b', dtype=np.float64)
    assert ccode(cos(b)) == "cos(b)"

    # Doesn't make much sense, but it's legal
    c = dSymbol('c', dtype=np.int32)
    assert ccode(cos(c)) == "cos(c)"


def test_intdiv():
    a = Symbol('a')
    b = Symbol('b')

    # IntDiv by 1 automatically simplified
    v = IntDiv(a, 1)
    assert v is a

    v = IntDiv(a, 2)
    assert ccode(v) == 'a / 2'

    # Within larger expressions -> parentheses
    v = b*IntDiv(a, 2)
    assert ccode(v) == 'b*(a / 2)'
    v = 3*IntDiv(a, 2)
    assert ccode(v) == '3*(a / 2)'
    v = b*IntDiv(a, 2) + 3
    assert ccode(v) == 'b*(a / 2) + 3'

    # IntDiv by zero or non-integer fails
    with pytest.raises(ValueError):
        IntDiv(a, 0)
    with pytest.raises(ValueError):
        IntDiv(a, 3.5)

    v = b*IntDiv(a + b, 2) + 3
    assert ccode(v) == 'b*((a + b) / 2) + 3'


def test_safeinv():
    grid = Grid(shape=(11, 11))
    x, y = grid.dimensions

    u1 = Function(name='u', grid=grid)
    u2 = Function(name='u', grid=grid, dtype=np.float64)

    op1 = Operator(Eq(u1, SafeInv(u1, u1)))
    op2 = Operator(Eq(u2, SafeInv(u2, u2)))

    assert 'SAFEINV' in str(op1)
    assert 'SAFEINV' in str(op2)


def test_def_function():
    foo0 = DefFunction('foo', arguments=['a', 'b'], template=['int'])
    foo1 = DefFunction('foo', arguments=['a', 'b'], template=['int'])
    foo2 = DefFunction('foo', arguments=['a', 'b'])
    foo3 = DefFunction('foo', arguments=['a'])

    # Code generation
    assert str(foo0) == 'foo<int>(a, b)'
    assert str(foo3) == 'foo(a)'

    # Hashing and equality
    assert hash(foo0) == hash(foo1)
    assert foo0 == foo1
    assert hash(foo0) != hash(foo2)
    assert hash(foo2) != hash(foo3)

    # Reconstruction
    assert foo0 == foo0._rebuild()
    assert str(foo0._rebuild('bar', template=['float'])) == 'bar<float>(a, b)'


def test_namespace():
    ns0 = Namespace(['std', 'algorithms', 'parallel'])
    assert str(ns0) == 'std::algorithms::parallel'

    ns1 = Namespace(['std'])
    ns2 = Namespace(['std', 'algorithms', 'parallel'])

    # Test hashing and equality
    assert hash(ns0) != hash(ns1)  # Same reason as above
    assert ns0 != ns1
    assert hash(ns0) == hash(ns2)
    assert ns0 == ns2

    # Free symbols
    assert not ns0.free_symbols


def test_rvalue():
    ctype = ReservedWord('dummytype')
    ns = Namespace(['my', 'namespace'])
    init = ListInitializer(())

    assert str(Rvalue(ctype, ns, init)) == 'my::namespace::dummytype{}'


def test_basecast():
    s = Symbol(name='s', dtype=np.float32)

    class BarCast(BaseCast):
        _dtype = 'bar'

    v = BarCast(s, '**')
    assert ccode(v) == '(bar**)s'

    # Reconstruction
    assert ccode(v.func(*v.args)) == '(bar**)s'

    v1 = BarCast(s, '****')
    assert v != v1


def test_str_cast():
    s = Symbol(name='s', dtype=np.float32)

    v = Cast(s, 'foo')
    assert not v.stars
    assert v.dtype == 'foo'
    assert v._op == '(foo)'
    assert ccode(v) == '(foo)s'

    v = Cast(s, 'foo*')
    assert v.stars == '*'
    assert v.dtype == 'foo'
    assert v._op == '(foo*)'
    assert ccode(v) == '(foo*)s'

    v = Cast(s, 'foo **')
    assert v.stars == '**'
    assert v.dtype == 'foo'
    assert v._op == '(foo**)'
    assert ccode(v) == '(foo**)s'


def test_findexed():
    grid = Grid(shape=(3, 3, 3))
    x, y, z = grid.dimensions

    f = Function(name='f', grid=grid)

    strides_map = {x: 1, y: 2, z: 3}
    fi = FIndexed(f.base, x+1, y, z-2, strides_map=strides_map)
    assert ccode(fi) == 'f(x + 1, y, z - 2)'

    # Binding
    _, fi1 = fi.bind('fL')
    assert fi1.base is f.base
    assert ccode(fi1) == 'fL(x + 1, y, z - 2)'

    # Reconstruction
    strides_map = {x: 3, y: 2, z: 1}
    new_fi = fi.func(strides_map=strides_map, accessor=fi1.accessor)

    assert new_fi.name == fi.name == 'f'
    assert new_fi.accessor == fi1.accessor
    assert new_fi.accessor.name == 'fL'
    assert new_fi.indices == fi.indices
    assert new_fi.strides_map == strides_map


def test_component_access():
    grid = Grid(shape=(3, 3, 3))
    x, y, z = grid.dimensions

    f = Function(name='f', grid=grid)

    cf0 = ComponentAccess(f.indexify(), 0)
    cf1 = ComponentAccess(f.indexify(), 1)

    assert ccode(cf0) == 'f[x][y][z].x'
    assert ccode(cf1) == 'f[x][y][z].y'

    # Reconstruction
    cf2 = cf1.func(*cf1.args)
    assert cf2.index == cf1.index
    assert cf2 == cf1


def test_vector_access():
    grid = Grid(shape=(3, 3, 3))

    f = Function(name='f', grid=grid)
    g = Function(name='g', grid=grid)

    v = VectorAccess(f.indexify())

    assert v.base == f.indexify()
    assert v.function is f

    # Code generation
    assert ccode(v) == 'VL<f[x, y, z]>'

    # Reconstruction
    v1 = v.func(g.indexify())
    assert ccode(v1) == 'VL<g[x, y, z]>'


def test_halo_touch():
    grid = Grid(shape=(3, 3))
    x, y = grid.dimensions

    f = Function(name='f', grid=grid)
    g = Function(name='g', grid=grid)

    # Hashing and equality
    ht0 = HaloTouch(f[x, y], g[x, y])
    ht1 = HaloTouch(f[x, y], g[x, y])
    ht2 = HaloTouch(f[x, y], g[x + 1, y + 1])
    assert hash(ht0) == hash(ht1)
    assert ht0 == ht1
    assert ht0 != ht2
    assert hash(ht0) != hash(ht2)

    # Reconstruction
    assert ht0 == ht0._rebuild()
    assert hash(ht0) == hash(ht0._rebuild())


def test_canonical_ordering_of_weights():
    grid = Grid(shape=(3, 3, 3))
    x, y, z = grid.dimensions

    f = Function(name='f', grid=grid)

    i = StencilDimension('i0', 0, 2)
    w = Weights(name='w0', dimensions=i, initvalue=[1.0, 2.0, 3.0])

    fi = f[x, y + i, z]
    wi = w[i]
    cf = ComponentAccess(fi, 0)

    assert (ccode(1.0*f[x, y, z] + 2.0*f[x, y + 1, z] + 3.0*f[x, y + 2, z]) ==
            '1.0F*f[x][y][z] + 2.0F*f[x][y + 1][z] + 3.0F*f[x][y + 2][z]')
    assert ccode(fi*wi) == 'w0[i0]*f[x][y + i0][z]'
    assert ccode(cf*wi) == 'w0[i0]*f[x][y + i0][z].x'


def test_symbolic_printing():
    b = Symbol('b')

    v = CallFromPointer('foo', 's') + b
    assert str(v) == 'b + s->foo()'

    class MyLocalObject(LocalObject, Expr):
        pass

    lo = MyLocalObject(name='lo')
    assert str(lo + 2) == '2 + lo'

    grid = Grid((10,))
    f = Function(name="f", grid=grid)
    fi = FIndexed(f.base, grid.dimensions[0])
    df = DefFunction('aaa', arguments=[fi])
    assert ccode(df) == 'aaa(f(x))'


def test_is_on_grid():
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + .5 * x.spacing
    u = Function(name="u", grid=grid, space_order=2)

    assert u._grid_map == {}
    assert u.subs({x: x0})._grid_map == {x: x0}
    assert all(uu._grid_map == {} for uu in retrieve_functions(u.subs({x: x0}).evaluate))


@pytest.mark.parametrize('expr,expected', [
    ('f[x+2]*g[x+4] + f[x+3]*g[x+5] + f[x+4] + f[x+1]',
     ['f[x+2]', 'g[x+4]', 'f[x+3]', 'g[x+5]', 'f[x+1]', 'f[x+4]']),
    ('f[x]*g[x+2] + f[x+1]*g[x+3]', ['f[x]', 'g[x+2]', 'f[x+1]', 'g[x+3]']),
])
def test_canonical_ordering(expr, expected):
    """
    Test that the `expr.args` are stored in canonical ordering.
    """
    grid = Grid(shape=(10,))
    x, = grid.dimensions  # noqa

    f = Function(name='f', grid=grid)  # noqa
    g = Function(name='g', grid=grid)  # noqa

    expr = eval(expr)
    for n, i in enumerate(list(expected)):
        expected[n] = eval(i)

    assert retrieve_indexed(expr) == expected


def test_solve_time():
    """
    Tests that solves only evaluate the time derivative.
    """
    grid = Grid(shape=(11, 11))
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)
    m = Function(name="m", grid=grid, space_order=4)
    dt = grid.time_dim.spacing
    eq = m * u.dt2 + u.dx
    sol = solve(eq, u.forward)
    # Check u.dx is not evaluated. Need to simplify because the solution
    # contains some Dummy in the Derivative subs that make equality break.
    assert len(retrieve_derivatives(sol)) == 1
    assert sympy.simplify(u.dx - retrieve_derivatives(sol)[0]) == 0
    assert sympy.simplify(sympy.expand(sol - (-dt**2*u.dx/m + 2.0*u - u.backward))) == 0


class TestUxreplace:

    @pytest.mark.parametrize('expr,subs,expected', [
        ('f', '{f: g}', 'g'),
        ('f[x, y+1]', '{f.indexed: g.indexed}', 'g[x, y+1]'),
        ('cos(f)', '{cos: sin}', 'sin(f)'),
        ('cos(f + sin(g))', '{cos: sin, sin: cos}', 'sin(f + cos(g))'),
        ('FIndexed(f.indexed, x, y)', '{x: 0}', 'FIndexed(f.indexed, 0, y)'),
    ])
    def test_expressions(self, expr, subs, expected):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa

        f = Function(name='f', grid=grid)  # noqa
        g = Function(name='g', grid=grid)  # noqa

        assert uxreplace(eval(expr), eval(subs)) == eval(expected)

    def test_custom_reconstructable(self):

        class MyDefFunction(DefFunction):
            __rargs__ = ('name', 'arguments')
            __rkwargs__ = ('p0', 'p1', 'p2')

            def __new__(cls, name=None, arguments=None, p0=None, p1=None, p2=None):
                obj = super().__new__(cls, name=name, arguments=arguments)
                obj.p0 = p0
                obj.p1 = as_tuple(p1)
                obj.p2 = p2
                return obj

        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        func = MyDefFunction(name='foo', arguments=f.indexify(),
                             p0=f, p1=f, p2='bar')

        mapper = {f: g, f.indexify(): g.indexify()}
        func1 = uxreplace(func, mapper)

        assert func1.arguments == (g.indexify(),)
        assert func1.p0 is g
        assert func1.p1 == (g,)
        assert func1.p2 == 'bar'

    def test_reduce_to_number(self):
        grid = Grid(shape=(4, 4))
        x, _ = grid.dimensions
        h_x = x.spacing

        # Emulate lowered coefficient
        w = -0.0354212/(h_x*h_x)
        w_lowered = pow_to_mul(w)

        w_sub = uxreplace(w_lowered, {h_x: Number(3)})

        assert np.isclose(w_sub, -0.003935689)
        assert not w_sub.is_Mul
        assert w_sub.is_Number

    def test_halo_touch(self):
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        ht0 = HaloTouch(f[x, y])
        ht1 = uxreplace(ht0, {f.indexed: g.indexed})

        assert ht1.args == (g[x, y],)


def test_minmax():
    grid = Grid(shape=(5, 5))
    x, y = grid.dimensions

    f = Function(name="f", grid=grid)
    s = dSymbol(name="s")

    eqns = [Eq(s, 2),
            Eq(f, Max(y, s, 4))]

    op = Operator(eqns)

    op.apply()
    assert np.all(f.data == 4)


@pytest.mark.parametrize('dtype,expected', [
    (np.float32, ("fmaxf(", "fminf(")),
    (np.float64, ("fmax(", "fmin(")),
])
def test_minmax_precision(dtype, expected):
    grid = Grid(shape=(5, 5), dtype=dtype)

    f = Function(name="f", grid=grid)
    g = Function(name="g", grid=grid)

    eqn = Eq(f, Min(g, 4.0) + Max(g, 2.0))

    op = Operator(eqn)

    g.data[:] = 3.0

    op.apply()

    # Check generated code -- ensure it's using the fp64 versions of min/max,
    # that is fminf/fmaxf
    if 'CXX' in configuration['language']:
        expected = [f"std::{e.replace('f(', '(')}" for e in expected]
    assert all(i in str(op) for i in expected)

    assert np.all(f.data == 6.0)


@pytest.mark.parametrize('dtype,expected', [
    (np.float32, "powf("),
    (np.float64, "pow("),
])
def test_pow_precision(dtype, expected):
    grid = Grid(shape=(5, 5), dtype=dtype)

    f = Function(name="f", grid=grid)
    g = Function(name="g", grid=grid)

    eqn = Eq(f, g**1.5)

    op = Operator(eqn)

    g.data[:] = 4.0

    op.apply()

    if 'CXX' in configuration['language']:
        expected = "std::pow"

    assert expected in str(op)
    assert np.allclose(f.data, 8.0, rtol=np.finfo(dtype).eps)


@pytest.mark.parametrize('dtype,expected', [
    (np.float32, "fabsf("),
    (np.float64, "fabs("),
])
def test_abs_precision(dtype, expected):
    grid = Grid(shape=(5, 5), dtype=dtype)

    f = Function(name="f", grid=grid)
    g = Function(name="g", grid=grid)

    eqn = Eq(f, abs(g))

    op = Operator(eqn)

    g.data[:] = -1.0

    op.apply()

    if 'CXX' in configuration['language']:
        expected = "std::fabs"

    assert expected in str(op)
    assert np.all(f.data == 1.0)


class TestRelationsWithAssumptions:

    def test_multibounds_op(self):
        """
        Tests evalrel function on a simple example.
        """

        grid = Grid(shape=(16, 16, 16))

        a = Function(name='a', grid=grid)
        b = Function(name='b', grid=grid)
        c = Function(name='c', grid=grid)
        d = Function(name='d', grid=grid)
        a.data[:] = 5

        b = pow(a, 2)
        c = a + 10
        d = 2*a

        f = TimeFunction(name='f', grid=grid, space_order=2)

        f.data[:] = 0.1
        eqns = [Eq(f.forward, f.laplace + f * evalrel(min, [f, b, c, d]))]

        op = Operator(eqns, opt=('advanced'))
        op.apply(time_M=5)
        fnorm = norm(f)

        c2 = Function(name='c2', grid=grid)
        d2 = Function(name='d2', grid=grid)

        f.data[:] = 0.1
        eqns = [Eq(f.forward, f.laplace + f * evalrel(min, [f, b, c2, d2], [Ge(d, c)]))]
        op = Operator(eqns, opt=('advanced'))
        op.apply(time_M=5)
        fnorm2 = norm(f)

        assert fnorm == fnorm2

    @pytest.mark.parametrize('op, expr, assumptions, expected', [
        ([min, '[a, b, c, d]', '[]', 'Min(a, Min(b, Min(c, d)))']),
        ([max, '[a, b, c, d]', '[]', 'Max(a, Max(b, Max(c, d)))']),
        ([min, '[a]', '[]', 'a']),
        ([min, '[a, b]', '[Le(d, a), Ge(c, b)]', 'Min(a, b)']),
        ([min, '[a, b, c]', '[]', 'Min(a, Min(b, c))']),
        ([min, '[a, b, c, d]', '[Le(d, a), Ge(c, b)]', 'Min(b, d)']),
        ([min, '[a, b, c, d]', '[Ge(a, b), Ge(d, a), Ge(b, c)]', 'c']),
        ([max, '[a]', '[Le(a, a)]', 'a']),
        ([max, '[a, b]', '[Le(a, b)]', 'b']),
        ([max, '[a, b, c]', '[Le(c, b), Le(c, a)]', 'Max(a, b)']),
        ([max, '[a, b, c, d]', '[Ge(a, b), Ge(d, a), Ge(b, c)]', 'd']),
        ([max, '[a, b, c, d]', '[Ge(a, b), Le(b, c)]', 'Max(a, Max(c, d))']),
        ([max, '[a, b, c, d]', '[Ge(a, b), Le(c, b)]', 'Max(a, d)']),
        ([max, '[a, b, c, d]', '[Ge(b, a), Ge(a, b)]', 'Max(a, Max(c, d))']),
        ([min, '[a, b, c, d]', '[Ge(b, a), Ge(a, b), Le(c, b), Le(b, a)]', 'Min(c, d)']),
        ([min, '[a, b, c, d]', '[Ge(b, a), Ge(a, b), Le(c, b), Le(b, d)]', 'c']),
        ([min, '[a, b, c, d]', '[Ge(b, a + d)]', 'Min(a, Min(c, d))']),
        ([min, '[a, b, c, d]', '[Lt(b + a, d)]', 'Min(a, Min(b, c))']),
        ([max, '[a, b, c, d]', '[Lt(b + a, d)]', 'Max(c, d)']),
        ([max, '[a, b, c, d]', '[Gt(a, b + c + d)]', 'a']),
    ])
    def test_relations_w_complex_assumptions(self, op, expr, assumptions, expected):
        """
        Tests evalmin/evalmax with multiple args and assumtpions"""
        a = Symbol('a', positive=True)  # noqa
        b = Symbol('b', positive=True)  # noqa
        c = Symbol('c', positive=True)  # noqa
        d = Symbol('d', positive=True)  # noqa

        eqn = eval(expr)
        assumptions = eval(assumptions)
        assert str(evalrel(op, eqn, assumptions)) == expected

    @pytest.mark.parametrize('op, expr, assumptions, expected', [
        ([min, '[a, b, c, d]', '[Ge(b, a), Ge(a, b), Le(c, b), Le(b, d)]', 'c']),
        ([min, '[a, b, c, d]', '[Ge(b, a + d)]', 'Min(a, Min(b, Min(c, d)))']),
        ([min, '[a, b, c, d]', '[Ge(c, a + d)]', 'Min(a, b)']),
        ([max, '[a, b, c, d]', '[Ge(c, a + d), Gt(b, a + d)]', 'Max(b, d)']),
        ([max, '[a, b, c, d]', '[Ge(a + d, b), Gt(b, a + d)]',
         'Max(a, Max(b, Max(c, d)))']),
        ([max, '[a, b, c, d]', '[Le(c, a + d)]', 'Max(a, Max(b, Max(c, d)))']),
        ([max, '[a, b, c, d]', '[Le(c, d), Le(a, b)]', 'Max(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c, d), Le(d, c)]', 'Max(a, Max(b, c))']),
        ([min, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'Min(b, d)']),
        ([min, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'Min(b, d)']),
        ([min, '[a, b, c, d]', '[Gt(c, d).negated, Ge(a, b).negated]', 'Min(a, c)']),
        ([min, '[a, b, c, d]', '[Gt(c, d).reversed, Ge(a, b).reversed]', 'Min(b, d)']),
        ([min, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'Min(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'Max(a, c)']),
        ([max, '[a, b, c, d]', '[Gt(c, d).negated, Ge(a, b).negated]', 'Max(b, d)']),
        ([max, '[a, b, c, d]', '[Gt(c, d).reversed, Ge(a, b).reversed]', 'Max(a, c)']),
        ([max, '[a, b, c, d]', '[Lt(c, d).reversed, Le(a, b).reversed]', 'Max(b, d)']),
        ([max, '[a, b, c, d]', '[Gt(c, d + a).negated]', 'Max(a, Max(b, Max(c, d)))']),
        ([max, '[a, b, c, d]', '[Lt(c, d + a).negated]', 'Max(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c, d + a).negated]', 'Max(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c + b, d + a).negated]',
          'Max(a, Max(b, Max(c, d)))']),
        ([max, '[a, b, c, d, e]', '[Gt(a, b + c + e)]',
          'Max(a, Max(b, Max(c, Max(d, e))))']),
        ([max, '[a, b, c, d, e]', '[Ge(c, a), Ge(b, a), Ge(a, c), Ge(e, c), Ge(d, e)]',
          'Max(b, d)']),
    ])
    def test_relations_w_complex_assumptions_II(self, op, expr, assumptions, expected):
        """
        Tests evalmin/evalmax with multiple args and assumtpions"""
        a = Symbol('a', positive=False)  # noqa
        b = Symbol('b', positive=False)  # noqa
        c = Symbol('c', positive=True)  # noqa
        d = Symbol('d', positive=True)  # noqa
        e = Symbol('e', positive=True)  # noqa

        eqn = eval(expr)
        assumptions = eval(assumptions)
        assert str(evalrel(op, eqn, assumptions)) == expected

    @pytest.mark.parametrize('op, expr, assumptions, expected', [
        ([min, '[a, b, c, d]', '[Ge(b, a)]', 'a']),
        ([min, '[a, b, c, d]', '[Ge(b, d)]', 'Min(a, d)']),
        ([min, '[a, b, c, d]', '[Ge(c, a + d)]', 'Min(a, b)']),
        ([max, '[a, b, c, d, e]', 'None', 'Max(e, d)']),
    ])
    def test_assumptions(self, op, expr, assumptions, expected):
        """
        Tests evalmin/evalmax with multiple args and assumtpions"""
        a = Symbol('a', positive=False)
        b = Symbol('b', positive=False)
        c = Symbol('c', positive=True)
        d = Symbol('d', positive=True)
        e = Symbol('e', positive=True)

        c = a + 9  # noqa
        d = b + 10  # noqa
        e = c + 1  # noqa
        eqn = eval(expr)
        assumptions = eval(assumptions)
        expected = eval(expected)
        assert evalrel(op, eqn, assumptions) == expected


def test_issue_2577a():
    u = TimeFunction(name='u', grid=Grid((2,)))
    x = u.grid.dimensions[0]
    expr = Mul(-1, -1., x, u)
    assert expr.args == (x, u)
    eq = Eq(u.forward, expr)
    op = Operator(eq)

    assert '--' not in str(op.ccode)


def test_issue_2577b():
    class SD0(SubDomain):
        name = 'sd0'

        def define(self, dimensions):
            x, = dimensions
            return {x: ('middle', 1, 1)}

    grid = Grid(shape=(11,))

    sd0 = SD0(grid=grid)

    u = Function(name='u', grid=grid, space_order=2)

    eq_u = Eq(u, -(u*u).dxc, subdomain=sd0)

    op = Operator(eq_u)
    assert '--' not in str(op.ccode)


def test_print_div():
    a = SizeOf(np.int32)
    b = SizeOf(np.int64)
    cstr = ccode(a / b)
    assert cstr == 'sizeof(int)/sizeof(long)'


def test_customdtype_complex():
    """
    Test that `CustomDtype` doesn't brak is_imag
    """
    grid = Grid(shape=(4, 4))

    f = Function(name='f', grid=grid, dtype=CustomDtype('notnumpy'))

    assert not f.is_imaginary
    assert f.is_real


class TestComplexParts:
    def setup_basic(self, dtype):
        grid = Grid(shape=(5,), extent=(4.,))
        f = Function(name='f', grid=grid, dtype=dtype)
        f.data_with_halo[:] = np.arange(7) + 1j*np.arange(7, 14)[::-1]

        f_real = Function(name='f_real', grid=grid)
        f_imag = Function(name='f_imag', grid=grid)
        return f, f_real, f_imag

    def test_devito_print(self):
        f, _, _ = self.setup_basic(np.complex64)

        assert str(Real(f)) == 'Real(f(x))'
        assert str(Imag(f)) == 'Imag(f(x))'

    def test_printing(self):
        f, f_real, f_imag = self.setup_basic(np.complex64)

        eq_re = Eq(f_real, Real(f))
        eq_im = Eq(f_imag, Imag(f))

        op = Operator([eq_re, eq_im])

        if configuration['language'] in ('CXX', 'CXXopenmp'):
            assert "f_real[x + 1] = std::real(f[x + 1])" in str(op.ccode)
            assert "f_imag[x + 1] = std::imag(f[x + 1])" in str(op.ccode)

        else:
            assert "f_real[x + 1] = crealf(f[x + 1])" in str(op.ccode)
            assert "f_imag[x + 1] = cimagf(f[x + 1])" in str(op.ccode)

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_trivial(self, dtype):
        f, f_real, f_imag = self.setup_basic(dtype)

        eq_re = Eq(f_real, Real(f+1.))
        eq_im = Eq(f_imag, Imag(f+1.))

        Operator([eq_re, eq_im])()

        rcheck = np.array([2., 3., 4., 5., 6.])
        icheck = np.array([12., 11., 10., 9., 8.])
        assert np.all(np.isclose(f_real.data, rcheck))
        assert np.all(np.isclose(f_imag.data, icheck))

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_trivial_imag(self, dtype):
        f, f_real, f_imag = self.setup_basic(dtype)

        eq_re = Eq(f_real, Real(f+1j))
        eq_im = Eq(f_imag, Imag(f+1j))

        Operator([eq_re, eq_im])()

        rcheck = np.array([1., 2., 3., 4., 5.])
        icheck = np.array([13., 12., 11., 10., 9.])
        assert np.all(np.isclose(f_real.data, rcheck))
        assert np.all(np.isclose(f_imag.data, icheck))

    def test_deriv(self):
        f, f_real, f_imag = self.setup_basic(np.complex64)

        eq_re = Eq(f_real, Real(f.dx))
        eq_im = Eq(f_imag, Imag(f.dx))

        Operator([eq_re, eq_im])()

        assert np.all(np.isclose(f_real.data, 1.))
        assert np.all(np.isclose(f_imag.data, -1.))

    def test_outer_deriv(self):
        f, f_real, f_imag = self.setup_basic(np.complex64)

        eq_re = Eq(f_real, Real(f).dx)
        eq_im = Eq(f_imag, Imag(f).dx)

        Operator([eq_re, eq_im])()

        assert np.all(np.isclose(f_real.data, 1.))
        assert np.all(np.isclose(f_imag.data, -1.))

    def test_mul(self):
        grid = Grid(shape=(5,))

        f = Function(name='f', grid=grid, dtype=np.complex64)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid, dtype=np.complex64)
        f.data[:] = 1 + 1j
        g.data[:] = 2
        h.data[:] = 2j

        fg_re = Function(name='fg_re', grid=grid)
        fg_im = Function(name='fg_im', grid=grid)
        fh_re = Function(name='fh_re', grid=grid)
        fh_im = Function(name='fh_im', grid=grid)

        eq_fg_re = Eq(fg_re, Real(f*g))
        eq_fg_im = Eq(fg_im, Imag(f*g))
        eq_fh_re = Eq(fh_re, Real(f*h))
        eq_fh_im = Eq(fh_im, Imag(f*h))

        Operator([eq_fg_re, eq_fg_im, eq_fh_re, eq_fh_im])()

        assert np.all(np.isclose(fg_re.data, 2.))
        assert np.all(np.isclose(fg_im.data, 2.))

        assert np.all(np.isclose(fh_re.data, -2.))
        assert np.all(np.isclose(fh_im.data, 2.))

    def test_conj(self):
        grid = Grid(shape=(5,))
        f = Function(name='f', grid=grid, dtype=np.complex64)
        g = Function(name='g', grid=grid, dtype=np.complex64)

        f.data[:] = np.arange(5) + 1j*np.arange(5)[::-1]

        Operator([Eq(g, Conj(f))])()

        assert np.all(np.isclose(g.data, np.conj(f.data)))
