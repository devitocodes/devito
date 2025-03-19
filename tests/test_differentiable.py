from itertools import product

import sympy
import pytest

from devito import Function, Grid, Differentiable, NODE
from devito.finite_differences.differentiable import (Add, Mul, Pow, diffify,
                                                      interp_for_fd, SafeInv)


def test_differentiable():
    a = Function(name="a", grid=Grid((10, 10)))
    e = Function(name="e", grid=Grid((10, 10)))

    assert isinstance(1.2 * a.dx, Mul)
    assert isinstance(e + a, Add)
    assert isinstance(e * a, Mul)
    assert isinstance(a * a, Pow)
    assert isinstance(1 / (a * a), Pow)
    assert (a + e*a).dtype == a.dtype

    addition = a + 1.2 * a.dx
    assert isinstance(addition, Add)
    assert all(isinstance(a, Differentiable) for a in addition.args)
    assert addition.dtype == a.dtype

    addition2 = a + e * a.dx
    assert isinstance(addition2, Add)
    assert all(isinstance(a, Differentiable) for a in addition2.args)
    assert addition2.dtype == a.dtype


def test_diffify():
    a = Function(name="a", grid=Grid((10, 10)))
    e = Function(name="e", grid=Grid((10, 10)))

    assert isinstance(diffify(sympy.Mul(*[1.2, a.dx])), Mul)
    assert isinstance(diffify(sympy.Add(*[a, e])), Add)
    assert isinstance(diffify(sympy.Mul(*[e, a])), Mul)
    assert isinstance(diffify(sympy.Mul(*[a, a])), Pow)
    assert isinstance(diffify(sympy.Pow(*[a*a, -1])), Pow)

    addition = diffify(sympy.Add(*[a, sympy.Mul(*[1.2, a.dx])]))
    assert isinstance(addition, Add)
    assert all(isinstance(a, Differentiable) for a in addition.args)

    addition2 = diffify(sympy.Add(*[a, sympy.Mul(*[e, a.dx])]))
    assert isinstance(addition2, Add)
    assert all(isinstance(a, Differentiable) for a in addition2.args)


def test_shift():
    a = Function(name="a", grid=Grid((10, 10)))
    x = a.dimensions[0]
    assert a.shift(x, x.spacing) == a._subs(x, x + x.spacing)
    assert a.shift(x, x.spacing).shift(x, -x.spacing) == a
    assert a.shift(x, x.spacing).shift(x, x.spacing) == a.shift(x, 2*x.spacing)
    assert a.dx.evaluate.shift(x, x.spacing) == a.shift(x, x.spacing).dx.evaluate
    assert a.shift(x, .5 * x.spacing)._grid_map == {x: x + .5 * x.spacing}


def test_interp():
    grid = Grid((10, 10))
    x = grid.dimensions[0]
    a = Function(name="a", grid=grid, staggered=NODE)
    sa = Function(name="as", grid=grid, staggered=x)

    def sp_diff(a, b):
        a = getattr(a, 'evaluate', a)
        b = getattr(b, 'evaluate', b)
        return sympy.simplify(a - b) == 0

    # Base case, no interp
    assert interp_for_fd(a, {}) == a
    assert interp_for_fd(a, {x: x}) == a
    assert interp_for_fd(sa, {}) == sa
    assert interp_for_fd(sa, {x: x + x.spacing/2}) == sa

    # Base case, interp
    assert sp_diff(interp_for_fd(a, {x: x + x.spacing/2}),
                   .5*a + .5*a.shift(x, x.spacing))
    assert sp_diff(interp_for_fd(sa, {x: x}),
                   .5*sa + .5*sa.shift(x, -x.spacing))

    # Mul case, split interp
    assert sp_diff(interp_for_fd(a*sa, {x: x + x.spacing/2}),
                   sa * interp_for_fd(a, {x: x + x.spacing/2}))
    assert sp_diff(interp_for_fd(a*sa, {x: x}),
                   a * interp_for_fd(sa, {x: x}))

    # Add case, split interp
    assert sp_diff(interp_for_fd(a + sa, {x: x + x.spacing/2}),
                   sa + interp_for_fd(a, {x: x + x.spacing/2}))
    assert sp_diff(interp_for_fd(a + sa, {x: x}),
                   a + interp_for_fd(sa, {x: x}))


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_avg_mode(ndim):
    grid = Grid([11]*ndim)
    v = Function(name='v', grid=grid, staggered=grid.dimensions)
    a0 = Function(name="a0", grid=grid)
    a = Function(name="a", grid=grid, parameter=True)
    b = Function(name="b", grid=grid, parameter=True, avg_mode='harmonic')

    a0_avg = a0._eval_at(v)
    a_avg = a._eval_at(v).evaluate
    b_avg = b._eval_at(v).evaluate

    assert a0_avg == a0

    # Indices around the point at the center of a cell
    all_shift = tuple(product(*[[0, 1] for _ in range(ndim)]))
    args = [{d: d + i * d.spacing for d, i in zip(grid.dimensions, s)} for s in all_shift]

    # Default is arithmetic average
    assert sympy.simplify(a_avg - 0.5**ndim * sum(a.subs(arg) for arg in args)) == 0

    # Harmonic average, h(a[.5]) = 1/(.5/a[0] + .5/a[1])
    expected = 1/(0.5**ndim * sum(1/b.subs(arg) for arg in args))
    assert sympy.simplify(1/b_avg.args[0] - expected) == 0
    assert isinstance(b_avg, SafeInv)
    assert b_avg.base == b
