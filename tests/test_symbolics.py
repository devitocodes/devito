import sympy
import time
import pytest

from devito import Grid, Function, solve, div, grad, TimeFunction
from devito.symbolics import retrieve_functions, retrieve_indexed


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


def test_is_on_grid():
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + .5 * x.spacing
    u = Function(name="u", grid=grid, space_order=2)

    assert u._is_on_grid
    assert not u.subs({x: x0})._is_on_grid
    assert all(uu._is_on_grid for uu in retrieve_functions(u.subs({x: x0}).evaluate))


@pytest.mark.parametrize('so', [2, 4])
def test_solve(so):
    """
    Test that our solve produces the correct output and faster than sympy's
    default behavior for an affine equation (i.e. PDE time steppers).
    """
    grid = Grid((10, 10, 10))
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=so)
    v = Function(name="v", grid=grid, space_order=so)
    eq = u.dt2 - div(v * grad(u))

    # Standard sympy solve
    t0 = time.time()
    sol1 = sympy.solve(eq.evaluate, u.forward, rational=False, simplify=False)[0]
    t1 = time.time() - t0

    # Devito custom solve for linear equation in the target ax + b (most PDE tie steppers)
    t0 = time.time()
    sol2 = solve(eq.evaluate, u.forward)
    t12 = time.time() - t0

    diff = sympy.simplify(sol1 - sol2)
    # Difference can end up with super small coeffs with different evaluation
    # so zero out everything very small
    assert diff.xreplace({k: 0 if abs(k) < 1e-10 else k
                          for k in diff.atoms(sympy.Float)}) == 0
    # Make sure faster (actually much more than 10 for very complex cases)
    assert t12 < t1/10


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
