import numpy as np
import pytest
from devito import (Grid, Operator, Dimension, SparseFunction, SparseTimeFunction,
                    Function, TimeFunction)


def unit_box(name='a', shape=(11, 11), grid=None):
    """
    Create a field with value 0. to 1. in each dimension
    """
    grid = grid or Grid(shape=shape)
    a = Function(name=name, grid=grid)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def exp_box(name='a', shape=(11, 11), grid=None):
    """
    Create a field with exponential values
    """
    def init_data(f):
        for i in range(f.data.shape[0]):
            for j in range(f.data.shape[1]):
                f.data[i, j] = (grid.spacing[1]*j)**2
        return f

    if len(shape) == 2:
        grid = grid or Grid(shape=shape, extent=(shape[0] - 1, shape[1] - 1))
    else:
        grid = grid or Grid(shape=shape, extent=(shape[0] - 1,
                            shape[1] - 1, shape[-1] - 1))

    a = Function(name=name, grid=grid, space_order=0)
    a = init_data(a)
    return a


def unit_box_time(name='a', shape=(11, 11),):
    """
    Create a field with value 0. to 1. in each dimension
    """
    grid = Grid(shape=shape)
    a = TimeFunction(name=name, grid=grid, time_order=1)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[0, :] = np.meshgrid(*dims)[1]
    a.data[1, :] = np.meshgrid(*dims)[1]
    return a


def exp_box_time(name='a', shape=(11, 11), grid=None):
    """
    Create a field with exponential values
    """
    def init_data(f):
        for i in range(f.data.shape[1]):
            for j in range(f.data.shape[2]):
                f.data[0, i, j] = (grid.spacing[1]*j)**2
                f.data[1, i, j] = (grid.spacing[1]*j)**2
        return f

    if len(shape) == 2:
        grid = grid or Grid(shape=shape, extent=(shape[0] - 1, shape[1] - 1))
    else:
        grid = grid or Grid(shape=shape, extent=(shape[0] - 1,
                            shape[1] - 1, shape[-1] - 1))

    a = TimeFunction(name=name, grid=grid, time_order=1)
    a = init_data(a)
    return a


def points(grid, ranges, npoints, name='points', cubic=None, sinc=None):
    """
    Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseFunction(name=name, grid=grid, npoint=npoints, cubic=cubic, sinc=sinc)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def time_points(grid, ranges, npoints, name='points', nt=10, cubic=None, sinc=None):
    """
    Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseTimeFunction(name=name, grid=grid, npoint=npoints,
                                nt=nt, cubic=cubic, sinc=sinc)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def custom_points(grid, ranges, npoints, name='points', cubic=None, sinc=None):
    """
    Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    scale = Dimension(name="scale")
    dim = Dimension(name="dim")
    points = SparseFunction(name=name, grid=grid, dimensions=(scale, dim),
                            shape=(3, npoints), npoint=npoints, cubic=cubic, sinc=sinc)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


@pytest.mark.parametrize('shape, coords', [
    ((101, 101), [(3.4, 95.), (3., 92)]),
    ((101, 101, 101), [(3.4, 95.), (3., 92), (10., 10.)])
])
def test_interpolate(shape, coords, npoints=20):
    """
    Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """

    a = exp_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints, cubic=True)
    expr = p.interpolate(a)
    op = Operator(expr)

    op(a=a)

    expected_values = [point[1]**2 for point in p.coordinates.data]

    assert np.allclose(p.data[:], expected_values, atol=1e-5)


@pytest.mark.parametrize('shape, coords', [
    ((101, 101), [(3.4, 95.), (3., 92)]),
    ((101, 101, 101), [(3.4, 95.), (3., 92), (10., 10.)])
])
def test_interpolate_cumm(shape, coords, npoints=20):
    """
    Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = exp_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints, cubic=True)

    p.data[:] = 1.
    expr = p.interpolate(a, increment=True)
    Operator(expr)(a=a)

    expected_values = [((point[1]**2) + 1) for point in p.coordinates.data]

    assert np.allclose(p.data[:], expected_values, rtol=1e-5)


@pytest.mark.parametrize('shape, coords', [
    ((101, 101), [(3.4, 95.), (3., 92)]),
    ((101, 101, 101), [(3.4, 95.), (3., 92), (10., 10.)])
])
def test_interpolate_time_shift(shape, coords, npoints=20):
    """
    Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    This test verifies the optional time shifting for SparseTimeFunctions
    """
    a = exp_box_time(shape=shape)
    p = time_points(a.grid, coords, npoints=npoints, nt=10, cubic=True)

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1)
    Operator(expr)(a=a)

    expected_values = [(point[1]**2) for point in p.coordinates.data]

    assert np.allclose(p.data[0, :], expected_values, rtol=1e-5)

    p.data[:] = 1.
    expr = p.interpolate(a, p_t=p.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[1, :], expected_values, rtol=1e-5)

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1,
                         p_t=p.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[1, :], expected_values, rtol=1e-5)


@pytest.mark.parametrize('shape, coords', [
    ((101, 101), [(3.4, 95.), (3., 92)]),
    ((101, 101, 101), [(3.4, 95.), (3., 92), (10., 10.)])
])
def test_interpolate_array(shape, coords, npoints=20):
    """
    Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = exp_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints, cubic=True)

    expr = p.interpolate(a)
    Operator(expr)(a=a, points=p.data[:])

    expected_values = [(point[1]**2) for point in p.coordinates.data]

    assert np.allclose(p.data[:], expected_values, rtol=1e-5)


@pytest.mark.parametrize('shape, coords', [
    ((101, 101), [(3.4, 95.), (3., 92)]),
    ((101, 101, 101), [(3.4, 95.), (3., 92), (10., 10.)])
])
def test_interpolate_custom(shape, coords, npoints=20):
    """
    Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = exp_box(shape=shape)
    p = custom_points(a.grid, coords, npoints=npoints, cubic=True)

    p.data[:] = 1.
    expr = p.interpolate(a * p.indices[0])
    op = Operator(expr)

    op(a=a)
    expected_values = [(point[1]**2)*0.0 for point in p.coordinates.data]

    expected_values2 = [(point[1]**2)*1.0 for point in p.coordinates.data]

    expected_values3 = [(point[1]**2)*2.0 for point in p.coordinates.data]

    assert np.allclose(p.data[0, :], expected_values, rtol=1e-5)
    assert np.allclose(p.data[1, :], expected_values2, rtol=1e-5)
    assert np.allclose(p.data[2, :], expected_values3, rtol=1e-5)


def test_interpolation_dx():
    """
    Test interpolation of a SparseFunction from a Derivative of
    a Function.
    """
    u = unit_box(shape=(11, 11))
    sf1 = SparseFunction(name='s', grid=u.grid, npoint=1, cubic=True)
    sf1.coordinates.data[0, :] = (0.5, 0.5)

    op = Operator(sf1.interpolate(u.dx))

    assert sf1.data.shape == (1,)
    u.data[:] = 0.0
    u.data[5, 5] = 4.0
    u.data[4, 5] = 2.0
    u.data[6, 5] = 2.0

    op.apply()
    # Exactly in the middle of 4 points, only 1 nonzero is 4
    assert sf1.data[0] == pytest.approx(-20.0)


@pytest.mark.parametrize('shape, coords', [
    ((101, 101), [(3.4, 95.), (3., 92)]),
    ((101, 101, 101), [(3.4, 95.), (3., 92), (10., 10.)])
])
def test_interpolate_indexed(shape, coords, npoints=20):
    """
    Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid. Unlike other tests,
    here we interpolate an expression built using the indexed notation.
    """
    a = exp_box(shape=shape)
    p = custom_points(a.grid, coords, npoints=npoints, cubic=True)
    print(f"{p.interpolator}")

    p.data[:] = 1.
    expr = p.interpolate(a[a.grid.dimensions] * p.indices[0])
    op = Operator(expr)
    op(a=a)

    expected_values = [(point[1]**2)*0.0 for point in p.coordinates.data]

    expected_values1 = [(point[1]**2)*1.0 for point in p.coordinates.data]

    expected_values2 = [(point[1]**2)*2.0 for point in p.coordinates.data]

    assert np.allclose(p.data[0, :], expected_values, rtol=1e-5)
    assert np.allclose(p.data[1, :], expected_values1, rtol=1e-5)
    assert np.allclose(p.data[2, :], expected_values2, rtol=1e-5)
