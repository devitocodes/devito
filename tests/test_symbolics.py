import sympy
import pytest
import numpy as np

from sympy import Symbol
from devito import (Constant, Dimension, Grid, Function, solve, TimeFunction, Eq,  # noqa
                    Operator, SubDimension, norm, Le, Ge, Gt, Lt)
from devito.ir import Expression, FindNodes
from devito.symbolics import (retrieve_functions, retrieve_indexed, evalrel,  # noqa
                              CallFromPointer, Cast, FieldFromPointer,
                              FieldFromComposite, IntDiv, MIN, MAX, ccode)
from devito.types import Array


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

    op = Operator(eq)

    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 2
    assert str(exprs[0]) == expected


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
    assert di.bound_symbols == {d.symbolic_min, d.symbolic_max} | set(di._thickness_map)

    dl = SubDimension.left(name='dl', parent=d, thickness=4)
    assert dl.free_symbols == {dl}
    assert dl.bound_symbols == {d.symbolic_min, dl.thickness.left[0]}

    dr = SubDimension.right(name='dr', parent=d, thickness=4)
    assert dr.free_symbols == {dr}
    assert dr.bound_symbols == {d.symbolic_max, dr.thickness.right[0]}


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

    ub = Array(name='ub', dtype=u.dtype, dimensions=u.dimensions)

    assert ub.free_symbols == {x, y}
    assert ub.indexed.free_symbols == {ub.indexed, x.symbolic_size, y.symbolic_size}


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
    cfp = CallFromPointer('foo', 's')
    ffp = FieldFromPointer('foo', 's')
    ffc = FieldFromComposite('foo', 's')

    assert ccode(cfp + 1) == '1 + s->foo()'
    assert ccode(ffp + 1) == '1 + s->foo'
    assert ccode(ffc + 1) == '1 + s.foo'


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


def test_cast():
    s = Symbol(name='s', dtype=np.float32)

    class BarCast(Cast):
        _base_typ = 'bar'

    v = BarCast(s, '**')
    assert ccode(v) == '(bar**)s'

    # Reconstruction
    assert ccode(v.func(*v.args)) == '(bar**)s'

    v1 = BarCast(s, '****')
    assert v != v1


def test_is_on_grid():
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + .5 * x.spacing
    u = Function(name="u", grid=grid, space_order=2)

    assert u._is_on_grid
    assert not u.subs({x: x0})._is_on_grid
    assert all(uu._is_on_grid for uu in retrieve_functions(u.subs({x: x0}).evaluate))


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
    # contains some Dummy in the Derivatibe subs that make equality break.
    # TODO: replace by retreive_derivatives after Fabio's PR
    assert sympy.simplify(u.dx - sol.args[2].args[0].args[1]) == 0
    assert sympy.simplify(sol - (-dt**2*u.dx/m + 2.0*u - u.backward)) == 0


class TestRelationsWithAssumptions(object):

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
        ([min, '[a, b, c, d]', '[]', 'MIN(a, MIN(b, MIN(c, d)))']),
        ([max, '[a, b, c, d]', '[]', 'MAX(a, MAX(b, MAX(c, d)))']),
        ([min, '[a]', '[]', 'a']),
        ([min, '[a, b]', '[Le(d, a), Ge(c, b)]', 'MIN(a, b)']),
        ([min, '[a, b, c]', '[]', 'MIN(a, MIN(b, c))']),
        ([min, '[a, b, c, d]', '[Le(d, a), Ge(c, b)]', 'MIN(b, d)']),
        ([min, '[a, b, c, d]', '[Ge(a, b), Ge(d, a), Ge(b, c)]', 'c']),
        ([max, '[a]', '[Le(a, a)]', 'a']),
        ([max, '[a, b]', '[Le(a, b)]', 'b']),
        ([max, '[a, b, c]', '[Le(c, b), Le(c, a)]', 'MAX(a, b)']),
        ([max, '[a, b, c, d]', '[Ge(a, b), Ge(d, a), Ge(b, c)]', 'd']),
        ([max, '[a, b, c, d]', '[Ge(a, b), Le(b, c)]', 'MAX(a, MAX(c, d))']),
        ([max, '[a, b, c, d]', '[Ge(a, b), Le(c, b)]', 'MAX(a, d)']),
        ([max, '[a, b, c, d]', '[Ge(b, a), Ge(a, b)]', 'MAX(a, MAX(c, d))']),
        ([min, '[a, b, c, d]', '[Ge(b, a), Ge(a, b), Le(c, b), Le(b, a)]', 'MIN(c, d)']),
        ([min, '[a, b, c, d]', '[Ge(b, a), Ge(a, b), Le(c, b), Le(b, d)]', 'c']),
        ([min, '[a, b, c, d]', '[Ge(b, a + d)]', 'MIN(a, MIN(c, d))']),
        ([min, '[a, b, c, d]', '[Lt(b + a, d)]', 'MIN(a, MIN(b, c))']),
        ([max, '[a, b, c, d]', '[Lt(b + a, d)]', 'MAX(c, d)']),
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
        expected = eval(expected)
        assert evalrel(op, eqn, assumptions) == expected

    @pytest.mark.parametrize('op, expr, assumptions, expected', [
        ([min, '[a, b, c, d]', '[Ge(b, a), Ge(a, b), Le(c, b), Le(b, d)]', 'c']),
        ([min, '[a, b, c, d]', '[Ge(b, a + d)]', 'MIN(a, MIN(b, MIN(c, d)))']),
        ([min, '[a, b, c, d]', '[Ge(c, a + d)]', 'MIN(a, b)']),
        ([max, '[a, b, c, d]', '[Ge(c, a + d), Gt(b, a + d)]', 'MAX(b, d)']),
        ([max, '[a, b, c, d]', '[Ge(a + d, b), Gt(b, a + d)]',
         'MAX(a, MAX(b, MAX(c, d)))']),
        ([max, '[a, b, c, d]', '[Le(c, a + d)]', 'MAX(a, MAX(b, MAX(c, d)))']),
        ([max, '[a, b, c, d]', '[Le(c, d), Le(a, b)]', 'MAX(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c, d), Le(d, c)]', 'MAX(a, MAX(b, c))']),
        ([min, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'MIN(b, d)']),
        ([min, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'MIN(b, d)']),
        ([min, '[a, b, c, d]', '[Gt(c, d).negated, Ge(a, b).negated]', 'MIN(a, c)']),
        ([min, '[a, b, c, d]', '[Gt(c, d).reversed, Ge(a, b).reversed]', 'MIN(b, d)']),
        ([min, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'MIN(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c, d).negated, Le(a, b).negated]', 'MAX(a, c)']),
        ([max, '[a, b, c, d]', '[Gt(c, d).negated, Ge(a, b).negated]', 'MAX(b, d)']),
        ([max, '[a, b, c, d]', '[Gt(c, d).reversed, Ge(a, b).reversed]', 'MAX(a, c)']),
        ([max, '[a, b, c, d]', '[Lt(c, d).reversed, Le(a, b).reversed]', 'MAX(b, d)']),
        ([max, '[a, b, c, d]', '[Gt(c, d + a).negated]', 'MAX(a, MAX(b, MAX(c, d)))']),
        ([max, '[a, b, c, d]', '[Lt(c, d + a).negated]', 'MAX(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c, d + a).negated]', 'MAX(b, d)']),
        ([max, '[a, b, c, d]', '[Le(c + b, d + a).negated]',
          'MAX(a, MAX(b, MAX(c, d)))']),
        ([max, '[a, b, c, d, e]', '[Gt(a, b + c + e)]',
          'MAX(a, MAX(b, MAX(c, MAX(d, e))))']),
        ([max, '[a, b, c, d, e]', '[Ge(c, a), Ge(b, a), Ge(a, c), Ge(e, c), Ge(d, e)]',
          'MAX(b, d)']),
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
        expected = eval(expected)
        assert evalrel(op, eqn, assumptions) == expected

    @pytest.mark.parametrize('op, expr, assumptions, expected', [
        ([min, '[a, b, c, d]', '[Ge(b, a)]', 'a']),
        ([min, '[a, b, c, d]', '[Ge(b, d)]', 'MIN(a, d)']),
        ([min, '[a, b, c, d]', '[Ge(c, a + d)]', 'MIN(a, b)']),
        ([max, '[a, b, c, d, e]', 'None', 'MAX(e, d)']),
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
