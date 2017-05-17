import numpy as np
import pytest
from sympy import Function
from sympy.abc import h

from devito import Operator, DenseData, PointData


@pytest.fixture
def a(shape=(11, 11)):
    x = np.linspace(0., 1., shape[0])
    y = np.linspace(0., 1., shape[1])
    a = DenseData(name='a', shape=shape)
    a.data[:] = np.meshgrid(x, y)[1]
    return a


def unit_box(name='a', shape=(11, 11)):
    """Create a field with value 0. to 1. in each dimension"""
    a = DenseData(name='a', shape=shape)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def points(ranges, npoints):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = PointData(name='points', nt=1, npoint=npoints, ndim=len(ranges))
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
    Operator(expr, subs={h: spacing})(a=a, t=1)

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

    expr = p.inject(a, Function('FLOAT')(1.))
    Operator(expr, subs={h: spacing})(a=a, t=1)

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
    b = DenseData(name='b', shape=a.data.shape)
    b.data[:] = 1.
    p = points(ranges=coords, npoints=npoints)

    expr = p.inject(field=a, expr=b)
    Operator(expr, subs={h: spacing})(a=a, b=b, t=1)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)
