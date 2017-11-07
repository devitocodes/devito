import numpy as np
import pytest
from conftest import skipif_yask

from devito.cgen_utils import FLOAT
from devito import Grid, Operator, Function, SparseFunction


@pytest.fixture
def a(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = Function(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


def unit_box(name='a', shape=(11, 11)):
    """Create a field with value 0. to 1. in each dimension"""
    grid = Grid(shape=shape)
    a = Function(name=name, grid=grid)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def points(grid, ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseFunction(name=name, grid=grid, npoint=npoints)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


@skipif_yask
@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    expr = p.interpolate(a)
    Operator(expr)(a=a)

    assert np.allclose(p.data[:], xcoords, rtol=1e-6)


@skipif_yask
@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_cumm(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a, cummulative=True)
    Operator(expr)(a=a)

    assert np.allclose(p.data[:], xcoords + 1., rtol=1e-6)


@skipif_yask
@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject(shape, coords, result, npoints=19):
    """Test point injection with a set of points forming a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    p = points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(a, FLOAT(1.))

    Operator(expr)(a=a)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@skipif_yask
@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject_from_field(shape, coords, result, npoints=19):
    """Test point injection from a second field along a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    b = Function(name='b', grid=a.grid)
    b.data[:] = 1.
    p = points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(field=a, expr=b)
    Operator(expr)(a=a, b=b)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@skipif_yask
@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_adjoint_inject_interpolate(shape, coords,
                                    npoints=19):

    a = unit_box(shape=shape)
    a.data[:] = 0.
    c = unit_box(shape=shape, name='c')
    c.data[:] = 27.
    # Inject receiver
    p = points(a.grid, ranges=coords, npoints=npoints)
    p.data[:] = 1.2
    expr = p.inject(field=a, expr=p)
    # Read receiver
    p2 = points(a.grid, name='points2', ranges=coords, npoints=npoints)
    expr2 = p2.interpolate(expr=c)
    Operator(expr + expr2)(a=a, c=c)
    # < P x, y > - < x, P^T y>
    # Px => p2
    # y => p
    # x => c
    # P^T y => a
    term1 = np.dot(p2.data.reshape(-1), p.data.reshape(-1))
    term2 = np.dot(c.data.reshape(-1), a.data.reshape(-1))
    assert np.isclose((term1-term2) / term1, 0., atol=1.e-6)
