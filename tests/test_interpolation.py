import numpy as np
import pytest
from conftest import skipif_yask, unit_box, points, unit_box_time, time_points
from math import sin, floor

from devito.cgen_utils import FLOAT
from devito import (Grid, Operator, Function, SparseFunction, Dimension,
                    TimeFunction, PrecomputedSparseFunction,
                    PrecomputedSparseTimeFunction)
from examples.seismic import demo_model, TimeAxis, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver


@pytest.fixture
def a(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = Function(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


@pytest.fixture
def at(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = TimeFunction(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


def custom_points(grid, ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    scale = Dimension(name="scale")
    dim = Dimension(name="dim")
    points = SparseFunction(name=name, grid=grid, dimensions=(scale, dim),
                            shape=(3, npoints), npoint=npoints)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def precompute_linear_interpolation(points, grid, origin):
    """ Sample precompute function that, given point and grid information
        precomputes gridpoints and coefficients according to a linear
        scheme to be used in PrecomputedSparseFunction.
    """
    gridpoints = [tuple(floor((point[i]-origin[i])/grid.spacing[i])
                        for i in range(len(point))) for point in points]

    coefficients = np.zeros((len(points), 2, 2))
    for i, point in enumerate(points):
        for d in range(grid.dim):
            coefficients[i, d, 0] = ((gridpoints[i][d] + 1)*grid.spacing[d] -
                                     point[d])/grid.spacing[d]
            coefficients[i, d, 1] = (point[d]-gridpoints[i][d]*grid.spacing[d])\
                / grid.spacing[d]
    return gridpoints, coefficients


@skipif_yask
def test_precomputed_interpolation():
    """ Test interpolation with PrecomputedSparseFunction which accepts
        precomputed values for coefficients
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)
    r = 2  # Constant for linear interpolation
    #  because we interpolate across 2 neighbouring points in each dimension

    def init(data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = sin(grid.spacing[0]*i) + sin(grid.spacing[1]*j)
        return data

    m = Function(name='m', grid=grid, initializer=init, space_order=0)

    gridpoints, coefficients = precompute_linear_interpolation(points, grid, origin)

    sf = PrecomputedSparseFunction(name='s', grid=grid, r=r, npoint=len(points),
                                   gridpoints=gridpoints, coefficients=coefficients)
    eqn = sf.interpolate(m)
    op = Operator(eqn)
    op()
    expected_values = [sin(point[0]) + sin(point[1]) for point in points]
    assert(all(np.isclose(sf.data, expected_values, rtol=1e-6)))


@skipif_yask
def test_precomputed_interpolation_time():
    """ Test interpolation with PrecomputedSparseFunction which accepts
        precomputed values for coefficients, but this time with a TimeFunction
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)
    r = 2  # Constant for linear interpolation
    #  because we interpolate across 2 neighbouring points in each dimension

    u = TimeFunction(name='u', grid=grid, space_order=0, save=5)
    for it in range(5):
        u.data[it, :] = it

    gridpoints, coefficients = precompute_linear_interpolation(points, grid, origin)

    sf = PrecomputedSparseTimeFunction(name='s', grid=grid, r=r, npoint=len(points),
                                       nt=5, gridpoints=gridpoints,
                                       coefficients=coefficients)

    assert sf.data.shape == (5, 3)

    eqn = sf.interpolate(u)
    op = Operator(eqn)
    op(time_m=0, time_M=4)

    for it in range(5):
        assert all(np.isclose(sf.data[it, :], it))


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
    expr = p.interpolate(a, increment=True)
    Operator(expr)(a=a)

    assert np.allclose(p.data[:], xcoords + 1., rtol=1e-6)


@skipif_yask
@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_time_shift(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    This test verifies the optional time shifting for SparseTimeFunctions
    """
    a = unit_box_time(shape=shape)
    p = time_points(a.grid, coords, npoints=npoints, nt=10)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[0, :], xcoords, rtol=1e-6)

    p.data[:] = 1.
    expr = p.interpolate(a, p_t=p.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[1, :], xcoords, rtol=1e-6)

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1,
                         p_t=p.indices[0]+1)
    Operator(expr)(a=a)

    assert np.allclose(p.data[1, :], xcoords, rtol=1e-6)


@skipif_yask
@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_array(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    expr = p.interpolate(a)
    Operator(expr)(a=a, points=p.data[:])

    assert np.allclose(p.data[:], xcoords, rtol=1e-6)


@skipif_yask
@pytest.mark.parametrize('shape, coords', [
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_custom(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid.
    """
    a = unit_box(shape=shape)
    p = custom_points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a * p.indices[0])
    Operator(expr)(a=a)

    assert np.allclose(p.data[0, :], 0.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[1, :], 1.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[2, :], 2.0 * xcoords, rtol=1e-6)


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
def test_inject_time_shift(shape, coords, result, npoints=19):
    """Test generic point injection testing the x-coordinate of an
    abitrary set of points going across the grid.
    This test verifies the optional time shifting for SparseTimeFunctions
    """
    a = unit_box_time(shape=shape)
    a.data[:] = 0.
    p = time_points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(a, FLOAT(1.), u_t=a.indices[0]+1)

    Operator(expr)(a=a, time=1)

    indices = [slice(1, 1, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)

    a.data[:] = 0.
    expr = p.inject(a, FLOAT(1.), p_t=p.indices[0]+1)

    Operator(expr)(a=a, time=1)

    indices = [slice(0, 0, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)

    a.data[:] = 0.
    expr = p.inject(a, FLOAT(1.), u_t=a.indices[0]+1, p_t=p.indices[0]+1)

    Operator(expr)(a=a, time=1)

    indices = [slice(1, 1, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@skipif_yask
@pytest.mark.parametrize('shape, coords, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 0.5)
])
def test_inject_array(shape, coords, result, npoints=19):
    """Test point injection with a set of points forming a line
    through the middle of the grid.
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    p = points(a.grid, ranges=coords, npoints=npoints)
    p2 = points(a.grid, ranges=coords, npoints=npoints, name='p2')
    p2.data[:] = 1.
    expr = p.inject(a, p)

    Operator(expr)(a=a, points=p2.data[:])

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
@pytest.mark.parametrize('shape', [(50, 50, 50)])
def test_position(shape):
    t0 = 0.0  # Start time
    tn = 500.  # Final time
    nrec = 130  # Number of receivers

    # Create model from preset
    model = demo_model('constant-isotropic', spacing=[15. for _ in shape],
                       shape=shape, nbpml=10)

    # Derive timestepping from model spacing
    dt = model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 30.

    # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=2,
                                space_order=4)

    rec, u, _ = solver.forward(save=False)

    # Define source geometry (center of domain, just below surface) with 100. origin
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5 + 100.
    src.coordinates.data[0, -1] = 130.

    # Define receiver geometry (same as source, but spread across x)
    rec2 = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec2.coordinates.data[:, 0] = np.linspace(100., 100. + model.domain_size[0],
                                              num=nrec)
    rec2.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    rec1, u1, _ = solver.forward(save=False, src=src, rec=rec2,
                                 o_x=100., o_y=100., o_z=100.)

    assert(np.allclose(rec.data, rec1.data, atol=1e-5))
