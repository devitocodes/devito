import pytest
import numpy as np

from devito import (Function, Grid, NODE, VectorTimeFunction,
                    TimeFunction, Eq, Operator, div)
from devito.tools import powerset


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
            shifted = shifted.subs({dd: dd - dd.spacing/2}, on_grid=False)
        assert all(i == dd for i, dd in zip(shifted.indices, grid.dimensions))
        # Average automatically i.e.:
        # f not defined at x so f(x, y) = 0.5*f(x - h_x/2, y) + 0.5*f(x + h_x/2, y)
        avg = f
        for dd in d:
            avg = .5 * (avg + avg.subs({dd: dd - dd.spacing}))
        assert shifted.evaluate == avg


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_is_param(ndim):
    """
    Test that only parameter are evaluated at the variable anf Function and FD indices
    stay unchanged
    """
    grid = Grid(tuple([10]*ndim))
    dims = list(powerset(grid.dimensions))[1:]
    var = Function(name="f", grid=grid, staggered=NODE)
    for d in dims:
        f = Function(name="f", grid=grid, staggered=d)
        f2 = Function(name="f2", grid=grid, staggered=d, parameter=True)

        # Not a parameter stay untouched (or FD would be destroyed by _eval_at)
        assert f._eval_at(var).evaluate == f
        # Parameter, automatic averaging
        avg = f2
        for dd in d:
            avg = .5 * (avg + avg.subs({dd: dd - dd.spacing}))
        assert f2._eval_at(var).evaluate == avg


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
    c = Function(name="c", grid=grid, staggered=y, parameter=True)  # noqa
    d = Function(name="d", grid=grid)  # noqa

    assert eval(expr) == eval(expected)


@pytest.mark.parametrize('expr, expected', [
    ('((a + b).dx._eval_at(a)).is_Add', 'True'),
    ('(a + b).dx._eval_at(a)', 'a.dx._eval_at(a) + b.dx._eval_at(a)'),
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

    A = Function(name="A", grid=grid, space_order=4, staggred=NODE, parameter=True)
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
