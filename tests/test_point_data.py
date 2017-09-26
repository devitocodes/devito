import numpy as np
import pytest

from devito.cgen_utils import FLOAT
from devito import Grid, Operator, DenseData, PointData, x, y, z


@pytest.fixture
def a(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = DenseData(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


def unit_box(name='a', shape=(11, 11)):
    """Create a field with value 0. to 1. in each dimension"""
    grid = Grid(shape=shape)
    a = DenseData(name=name, grid=grid)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def points(ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = PointData(name=name, nt=1, npoint=npoints, ndim=len(ranges))
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    spacing = a.data[tuple([1 for _ in shape])]
    p = points(coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    expr = p.interpolate(a)
    Operator(expr, subs={x.spacing: spacing, y.spacing: spacing,
                         z.spacing: spacing})(a=a, time=1)

    assert np.allclose(p.data[0, :], xcoords, rtol=1e-6)


@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject(shape, coords, result, npoints=19):
    """Test point injection with a set of points forming a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    spacing = a.data[tuple([1 for _ in shape])]
    a.data[:] = 0.
    p = points(ranges=coords, npoints=npoints)

    expr = p.inject(a, FLOAT(1.))

    Operator(expr, subs={x.spacing: spacing, y.spacing: spacing,
                         z.spacing: spacing})(a=a)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject_from_field(shape, coords, result, npoints=19):
    """Test point injection from a second field along a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    spacing = a.data[tuple([1 for _ in shape])]
    a.data[:] = 0.
    b = DenseData(name='b', grid=a.grid)
    b.data[:] = 1.
    p = points(ranges=coords, npoints=npoints)

    expr = p.inject(field=a, expr=b)
    Operator(expr, subs={x.spacing: spacing, y.spacing: spacing,
                         z.spacing: spacing})(a=a, b=b)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_adjoint_inject_interpolate(shape, coords,
                                    npoints=19):

    a = unit_box(shape=shape)
    spacing = a.data[tuple([1 for _ in shape])]
    a.data[:] = 0.
    c = unit_box(shape=shape, name='c')
    c.data[:] = 27.
    # Inject receiver
    p = points(ranges=coords, npoints=npoints)
    p.data[:] = 1.2
    expr = p.inject(field=a, expr=p)
    # Read receiver
    p2 = points(name='points2', ranges=coords, npoints=npoints)
    expr2 = p2.interpolate(expr=c)
    Operator(expr + expr2, subs={x.spacing: spacing, y.spacing: spacing,
                                 z.spacing: spacing})(a=a, c=c, time=1)
    # < P x, y > - < x, P^T y>
    # Px => p2
    # y => p
    # x => c
    # P^T y => a
    term1 = np.dot(p2.data.reshape(-1), p.data.reshape(-1))
    term2 = np.dot(c.data.reshape(-1), a.data.reshape(-1))
    assert np.isclose((term1-term2) / term1, 0., atol=1.e-6)
