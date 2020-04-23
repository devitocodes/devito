import pytest

from devito import Function, Grid, NODE
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
    f =  Function(name="f", grid=grid, staggered=x)
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
