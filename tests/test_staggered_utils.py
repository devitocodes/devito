import numpy as np
import pytest
from sympy import simplify

from devito import (
    CELL, NODE, Eq, Function, Grid, Operator, TimeFunction, VectorTimeFunction, div
)
from devito.tools import as_tuple, powerset


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_indices(ndim):
    """
    Test that inidces are shifted by half a grid point for staggered Function
    """
    grid = Grid(tuple([10]*ndim))
    dims = grid.dimensions
    for d in list(powerset(dims))[1:]:
        f = Function(name="f", grid=grid, staggered=d)
        for dd in d:
            assert f.indices_ref[dd] == dd + dd.spacing / 2


def test_indices_differentiable():
    """
    Test that differentiable object have correct indices and indices_ref
    """
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + x.spacing/2
    f = Function(name="f", grid=grid, staggered=x)
    assert f.indices_ref[x] == x0
    assert (1 * f).indices_ref[x] == x0
    assert (1. * f).indices_ref[x] == x0
    assert (1 + f).indices_ref[x] == x0
    assert (1 / f).indices_ref[x] == x0


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_avg(ndim):
    """
    Test automatic averaging of Function at undefined grid points
    """
    grid = Grid(tuple([10]*ndim))
    dims = list(powerset(grid.dimensions))[1:]
    for d in dims:
        f = Function(name="f", grid=grid, staggered=d)
        # f at nod (x, y, z)
        shifted = f
        for dd in d:
            shifted = shifted.subs({dd: dd - dd.spacing/2})
        assert all(
            i == dd for i, dd in zip(shifted.indices, grid.dimensions, strict=True)
        )
        # Average automatically i.e.:
        # f not defined at x so f(x, y) = 0.5*f(x - h_x/2, y) + 0.5*f(x + h_x/2, y)
        avg = f
        for dd in d:
            avg = .5 * (avg + avg.subs({dd: dd - dd.spacing}))
        assert simplify(shifted.evaluate - avg) == 0


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_is_param(ndim):
    """
    Test that only parameter are evaluated at the variable and Function and FD indices
    stay unchanged
    """
    grid = Grid(tuple([10]*ndim))
    dims = list(powerset(grid.dimensions))[1:]
    var = Function(name="f", grid=grid, staggered=NODE)
    for d in dims:
        f = Function(name="f", grid=grid, staggered=d)
        # Parameter, automatic averaging
        avg = f
        for dd in d:
            avg = .5 * (avg + avg.subs({dd: dd - dd.spacing}))
        assert simplify(f._eval_at(var).evaluate - avg) == 0


@pytest.mark.parametrize('expr, expected', [
    ('(a*b)._gather_for_diff', 'a.subs({x: x0}) * b'),
    ('(d*b)._gather_for_diff', 'd.subs({x: x0}) * b'),
    ('(d.dx*b)._gather_for_diff', 'd.dx * b.subs({x0: x})'),
    ('(b*c)._gather_for_diff', 'b * c.subs({x: x0, y0: y})')])
def test_gather_for_diff(expr, expected):
    grid = Grid((10, 10))
    x, y = grid.dimensions
    x0 = x + x.spacing/2  # noqa
    y0 = y + y.spacing/2  # noqa
    a = Function(name="a", grid=grid, staggered=NODE)  # noqa
    b = Function(name="b", grid=grid, staggered=x)  # noqa
    c = Function(name="c", grid=grid, staggered=y)  # noqa
    d = Function(name="d", grid=grid)  # noqa

    assert eval(expr) == eval(expected)


@pytest.mark.parametrize('expr, expected', [
    ('((a + b).dx._eval_at(a)).is_Add', 'True'),
    ('(a + b).dx._eval_at(a)', 'a.dx + b.dx._eval_at(a)'),
    ('(a*b).dx._eval_at(a).expr', 'a.subs({x: x0}) * b'),
    ('(a * b.dx).dx._eval_at(b).expr._eval_deriv ',
     'a.subs({x: x0}) * b.dx.evaluate')])
def test_stagg_fd_composite(expr, expected):
    grid = Grid((10, 10))
    x, y = grid.dimensions
    x0 = x + x.spacing/2  # noqa
    y0 = y + y.spacing/2  # noqa
    a = Function(name="a", grid=grid, staggered=NODE)  # noqa
    b = Function(name="b", grid=grid, staggered=x)  # noqa
    assert eval(expr) == eval(expected)


def test_staggered_div():
    """
    Test that div works properly on expressions.
    From @speglish issue #1248
    """
    grid = Grid(shape=(5, 5))

    v = VectorTimeFunction(name="v", grid=grid, time_order=1, space_order=4)
    p1 = TimeFunction(name="p1", grid=grid, time_order=1, space_order=4, staggered=NODE)
    p2 = TimeFunction(name="p2", grid=grid, time_order=1, space_order=4, staggered=NODE)

    # Test that 1.*v and 1*v are doing the same

    v[0].data[:] = 1.
    v[1].data[:] = 1.

    eq1 = Eq(p1, div(1*v))
    eq2 = Eq(p2, div(1.*v))

    op1 = Operator([eq1])
    op2 = Operator([eq2])

    op1.apply(time_M=0)
    op2.apply(time_M=0)
    assert np.allclose(p1.data[:], p2.data[:], atol=0, rtol=1e-5)

    # Test div on expression
    v[0].data[:] = 5.
    v[1].data[:] = 5.

    A = Function(name="A", grid=grid, space_order=4, staggred=NODE)
    A._data_with_outhalo[:] = .5

    av = VectorTimeFunction(name="av", grid=grid, time_order=1, space_order=4)

    # Operator with A (precomputed A*v)
    eq1 = Eq(av, A*v)
    eq2 = Eq(p1, div(av))
    op = Operator([eq1, eq2])
    op.apply(time_M=0)

    # Operator with div(A*v) directly
    eq3 = Eq(p2, div(A*v))
    op2 = Operator([eq3])
    op2.apply(time_M=0)

    assert np.allclose(p1.data[:], p2.data[:], atol=0, rtol=1e-5)


@pytest.mark.parametrize('stagg', [
    'NODE', 'CELL', 'x', 'y', 'z',
    '(x, y)', '(x, z)', '(y, z)', '(x, y, z)'])
def test_staggered_rebuild(stagg):
    grid = Grid(shape=(5, 5, 5))
    x, y, z = grid.dimensions  # noqa
    stagg = eval(stagg)

    f = Function(name='f', grid=grid, space_order=4, staggered=stagg)
    assert tuple(f.staggered.getters.keys()) == grid.dimensions

    f2 = f.func(name="f2")

    assert f2.dimensions == f.dimensions
    assert tuple(f2.staggered) == tuple(f.staggered)
    assert tuple(f2.staggered.getters.keys()) == f.dimensions

    # Check that rebuild correctly set the staggered indices
    # with the new dimensions
    for (d, nd) in zip(grid.dimensions, f.dimensions, strict=True):
        if d in as_tuple(stagg) or stagg is CELL:
            assert f2.indices[nd] == nd + nd.spacing / 2
        else:
            assert f2.indices[nd] == nd


def test_eval_at_different_dim():
    grid = Grid(shape=(31, 17, 25))
    nt = 5
    x, _, _ = grid.dimensions

    v = TimeFunction(name="v", grid=grid, staggered=x)
    tau = TimeFunction(name="tau", grid=grid, save=nt)

    eq = Eq(tau.forward, v).evaluate

    assert grid.time_dim not in eq.rhs.free_symbols


def test_new_from_staggering():
    grid = Grid(shape=(31, 17, 25))
    x, _, _ = grid.dimensions

    f = TimeFunction(name="f", grid=grid, staggered=x)
    # This used to fail since f.staggered as 4 elements (0, 1, 0, 0)
    # but it is processed for Dimension only.
    # Now properly  converts Staggering to the ref (x,) at init
    g = TimeFunction(name="g", grid=grid, staggered=f.staggered)

    assert g.staggered._ref == (x,)
    assert g.staggered == (0, 1, 0, 0)
    assert g.staggered == f.staggered
