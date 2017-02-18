import numpy as np
import pytest
from sympy import Eq
from sympy.abc import h

from devito import Operator, DenseData, PointData, TimeData, t
from examples.source_type import SourceLike


@pytest.fixture
def a(shape=(11, 11)):
    x = np.linspace(0., 1., shape[0])
    y = np.linspace(0., 1., shape[1])
    a = DenseData(name='a', shape=shape)
    a.data[:] = np.meshgrid(x, y)[1]
    return a


@pytest.fixture
def points(npoints=20):
    x = np.linspace(.05, .9, npoints)
    y = np.linspace(.01, .8, npoints)
    coords = np.concatenate((x, y)).reshape(npoints, 2)
    return SourceLike(name='points', nt=1, npoint=npoints,
                      ndim=2, coordinates=coords, h=0.1, nbpml=0)


def test_interpolate_2d(a, points):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    spacing = a.data[1, 1]
    xcoords = points.coordinates.data[:, 0]
    expr = Eq(points, points.interpolate(a))
    Operator(expr, subs={h: spacing})(t=1)
    assert np.allclose(points.data, xcoords, rtol=1e-7)
