from math import sin, floor

import numpy as np
import pytest
from sympy import Float

from devito import (Grid, Operator, Dimension, SparseFunction, SparseTimeFunction,
                    Function, TimeFunction, DefaultDimension, Eq, switchconfig,
                    PrecomputedSparseFunction, PrecomputedSparseTimeFunction,
                    MatrixSparseTimeFunction)
from examples.seismic import (demo_model, TimeAxis, RickerSource, Receiver,
                              AcquisitionGeometry)
from examples.seismic.acoustic import AcousticWaveSolver
import scipy.sparse


def unit_box(name='a', shape=(11, 11), grid=None, space_order=1):
    """Create a field with value 0. to 1. in each dimension"""
    grid = grid or Grid(shape=shape)
    a = Function(name=name, grid=grid, space_order=space_order)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def unit_box_time(name='a', shape=(11, 11), space_order=1):
    """Create a field with value 0. to 1. in each dimension"""
    grid = Grid(shape=shape)
    a = TimeFunction(name=name, grid=grid, time_order=1, space_order=space_order)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[0, :] = np.meshgrid(*dims)[1]
    a.data[1, :] = np.meshgrid(*dims)[1]
    return a


def points(grid, ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseFunction(name=name, grid=grid, npoint=npoints)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def time_points(grid, ranges, npoints, name='points', nt=10):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseTimeFunction(name=name, grid=grid, npoint=npoints, nt=nt)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def a(shape=(11, 11)):
    grid = Grid(shape=shape)
    a = Function(name='a', grid=grid)
    xarr = np.linspace(0., 1., shape[0])
    yarr = np.linspace(0., 1., shape[1])
    a.data[:] = np.meshgrid(xarr, yarr)[1]
    return a


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


def precompute_linear_interpolation(points, grid, origin, r=2):
    """
    Sample precompute function that, given point and grid information
    precomputes gridpoints and interpolation coefficients according to a linear
    scheme to be used in PrecomputedSparseFunction.

    Allow larger radius with zero weights for testing.
    """
    gridpoints = [tuple(floor((point[i]-origin[i])/grid.spacing[i])
                        for i in range(len(point))) for point in points]

    interpolation_coeffs = np.zeros((len(points), grid.dim, r))
    rs = r // 2 - 1
    for i, point in enumerate(points):
        for d in range(grid.dim):
            gd = gridpoints[i][d]
            interpolation_coeffs[i, d, rs] = ((gd + 1)*grid.spacing[d] -
                                              point[d])/grid.spacing[d]
            interpolation_coeffs[i, d, rs+1] = (point[d]-gd*grid.spacing[d])\
                / grid.spacing[d]
    return gridpoints, interpolation_coeffs


@pytest.mark.parametrize('r', [2, 4, 6])
def test_precomputed_interpolation(r):
    """ Test interpolation with PrecomputedSparseFunction which accepts
        precomputed values for interpolation coefficients
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)

    def init(data):
        # This is data with halo so need to shift to match the m.data expectations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = sin(grid.spacing[0]*(i-r)) + sin(grid.spacing[1]*(j-r))
        return data

    m = Function(name='m', grid=grid, initializer=init, space_order=r)

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(points,
                                                                       grid, origin,
                                                                       r=r)

    sf = PrecomputedSparseFunction(name='s', grid=grid, r=r, npoint=len(points),
                                   gridpoints=gridpoints,
                                   interpolation_coeffs=interpolation_coeffs)
    eqn = sf.interpolate(m)
    op = Operator(eqn)

    op()
    expected_values = [sin(point[0]) + sin(point[1]) for point in points]
    assert(all(np.isclose(sf.data, expected_values, rtol=1e-6)))


@pytest.mark.parametrize('r', [2, 4, 6])
def test_precomputed_interpolation_time(r):
    """ Test interpolation with PrecomputedSparseFunction which accepts
        precomputed values for interpolation coefficients, but this time
        with a TimeFunction
    """
    shape = (101, 101)
    points = [(.05, .9), (.01, .8), (0.07, 0.84)]
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)

    u = TimeFunction(name='u', grid=grid, space_order=r, save=5)
    for it in range(5):
        u.data[it, :] = it

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(points,
                                                                       grid, origin,
                                                                       r=r)

    sf = PrecomputedSparseTimeFunction(name='s', grid=grid, r=r, npoint=len(points),
                                       nt=5, gridpoints=gridpoints,
                                       interpolation_coeffs=interpolation_coeffs)

    assert sf.data.shape == (5, 3)

    eqn = sf.interpolate(u)
    op = Operator(eqn)

    op(time_m=0, time_M=4)

    for it in range(5):
        assert np.allclose(sf.data[it, :], it)


@pytest.mark.parametrize('r', [2, 4, 6])
def test_precomputed_injection(r):
    """Test injection with PrecomputedSparseFunction which accepts
       precomputed values for interpolation coefficients
    """
    shape = (11, 11)
    coords = [(.05, .95), (.45, .45)]
    origin = (0, 0)
    result = 0.25

    m = unit_box(shape=shape, space_order=r)
    m.data[:] = 0.

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(coords,
                                                                       m.grid, origin,
                                                                       r=r)

    sf = PrecomputedSparseFunction(name='s', grid=m.grid, r=r, npoint=len(coords),
                                   gridpoints=gridpoints,
                                   interpolation_coeffs=interpolation_coeffs)

    expr = sf.inject(m, Float(1.))

    op = Operator(expr)

    op()
    indices = [slice(0, 2, 1), slice(9, 11, 1)]
    assert np.allclose(m.data[indices], result, rtol=1.e-5)

    indices = [slice(4, 6, 1) for _ in coords]
    assert np.allclose(m.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('r', [2, 4, 6])
def test_precomputed_injection_time(r):
    """Test injection with PrecomputedSparseFunction which accepts
       precomputed values for interpolation coefficients
    """
    shape = (11, 11)
    coords = [(.05, .95), (.45, .45)]
    origin = (0, 0)
    result = 0.25
    nt = 20

    m = unit_box_time(shape=shape, space_order=r)
    m.data[:] = 0.

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(coords,
                                                                       m.grid, origin,
                                                                       r=r)

    sf = PrecomputedSparseTimeFunction(name='s', grid=m.grid, r=r, npoint=len(coords),
                                       gridpoints=gridpoints, nt=nt,
                                       interpolation_coeffs=interpolation_coeffs)
    sf.data.fill(1.)
    expr = sf.inject(m, sf)

    op = Operator(expr)

    op()
    for ti in range(2):
        indices = [slice(0, 2, 1), slice(9, 11, 1)]
        assert np.allclose(m.data[ti][indices], nt*result/2, rtol=1.e-5)

        indices = [slice(4, 6, 1) for _ in coords]
        assert np.allclose(m.data[ti][indices], nt*result/2, rtol=1.e-5)


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
    op = Operator(expr)

    op(a=a)
    assert np.allclose(p.data[:], xcoords, rtol=1e-6)


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
    op = Operator(expr)

    op(a=a)

    assert np.allclose(p.data[:], xcoords + 1., rtol=1e-6)


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
    op = Operator(expr)

    op(a=a)

    assert np.allclose(p.data[0, :], xcoords, rtol=1e-6)

    p.data[:] = 1.
    expr = p.interpolate(a, p_t=p.indices[0]+1)
    op = Operator(expr)

    op(a=a)

    assert np.allclose(p.data[1, :], xcoords, rtol=1e-6)

    p.data[:] = 1.
    expr = p.interpolate(a, u_t=a.indices[0]+1,
                         p_t=p.indices[0]+1)
    op = Operator(expr)

    op(a=a)

    assert np.allclose(p.data[1, :], xcoords, rtol=1e-6)


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
    op = Operator(expr)

    op(a=a, points=p.data[:])

    assert np.allclose(p.data[:], xcoords, rtol=1e-6)


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
    op = Operator(expr)

    op(a=a)

    assert np.allclose(p.data[0, :], 0.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[1, :], 1.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[2, :], 2.0 * xcoords, rtol=1e-6)


def test_interpolation_dx():
    """
    Test interpolation of a SparseFunction from a Derivative of
    a Function.
    """
    u = unit_box(shape=(11, 11))
    sf1 = SparseFunction(name='s', grid=u.grid, npoint=1)
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
    ((11, 11), [(.05, .9), (.01, .8)]),
    ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
])
def test_interpolate_indexed(shape, coords, npoints=20):
    """Test generic point interpolation testing the x-coordinate of an
    abitrary set of points going across the grid. Unlike other tests,
    here we interpolate an expression built using the indexed notation.
    """
    a = unit_box(shape=shape)
    p = custom_points(a.grid, coords, npoints=npoints)
    xcoords = p.coordinates.data[:, 0]

    p.data[:] = 1.
    expr = p.interpolate(a[a.grid.dimensions] * p.indices[0])
    op = Operator(expr)

    op(a=a)

    assert np.allclose(p.data[0, :], 0.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[1, :], 1.0 * xcoords, rtol=1e-6)
    assert np.allclose(p.data[2, :], 2.0 * xcoords, rtol=1e-6)


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

    expr = p.inject(a, Float(1.))

    op = Operator(expr)

    op(a=a)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape, coords, nexpr, result', [
    ((11, 11), [(.05, .95), (.45, .45)], 1, 1.),
    ((11, 11), [(.05, .95), (.45, .45)], 2, 1.),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 1, 0.5),
    ((11, 11, 11), [(.05, .95), (.45, .45), (.45, .45)], 2, 0.5)
])
def test_multi_inject(shape, coords, nexpr, result, npoints=19):
    """Test point injection with a set of points forming a line
    through the middle of the grid.
    """
    a1 = unit_box(name='a1', shape=shape)
    a2 = unit_box(name='a2', shape=shape, grid=a1.grid)
    a1.data[:] = 0.
    a2.data[:] = 0.
    p = points(a1.grid, ranges=coords, npoints=npoints)

    iexpr = Float(1.) if nexpr == 1 else (Float(1.), Float(2.))
    expr = p.inject((a1, a2), iexpr)

    op = Operator(expr)

    op(a1=a1, a2=a2)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    result = (result, result) if nexpr == 1 else (result, 2 * result)
    for r, a in zip(result, (a1, a2)):
        assert np.allclose(a.data[indices], r, rtol=1.e-5)


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

    expr = p.inject(a, Float(1.), u_t=a.indices[0]+1)

    op = Operator(expr)

    op(a=a, time=1)

    indices = [slice(1, 1, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)

    a.data[:] = 0.
    expr = p.inject(a, Float(1.), p_t=p.indices[0]+1)

    op = Operator(expr)

    op(a=a, time=1)

    indices = [slice(0, 0, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)

    a.data[:] = 0.
    expr = p.inject(a, Float(1.), u_t=a.indices[0]+1, p_t=p.indices[0]+1)

    op = Operator(expr)

    op(a=a, time=1)

    indices = [slice(1, 1, 1)] + [slice(4, 6, 1) for _ in coords]
    indices[1] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


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

    op = Operator(expr)

    op(a=a, points=p2.data[:])

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
    a.data[:] = 0.
    b = Function(name='b', grid=a.grid)
    b.data[:] = 1.
    p = points(a.grid, ranges=coords, npoints=npoints)

    expr = p.inject(field=a, expr=b)
    op = Operator(expr)

    op(a=a, b=b)

    indices = [slice(4, 6, 1) for _ in coords]
    indices[0] = slice(1, -1, 1)
    assert np.allclose(a.data[indices], result, rtol=1.e-5)


@pytest.mark.parametrize('shape', [(50, 50, 50)])
def test_position(shape):
    t0 = 0.0  # Start time
    tn = 500.  # Final time
    nrec = 130  # Number of receivers

    # Create model from preset
    model = demo_model('constant-isotropic', spacing=[15. for _ in shape],
                       shape=shape, nbl=10)

    # Derive timestepping from model spacing
    dt = model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(shape)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = 30.

    rec_coordinates = np.empty((nrec, len(shape)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1:] = src_coordinates[0, 1:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=t0, tn=tn, src_type='Ricker', f0=0.010)
    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, time_order=2, space_order=4)

    rec, u, _ = solver.forward(save=False)

    # Define source geometry (center of domain, just below surface) with 100. origin
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5 + 100.
    src.coordinates.data[0, -1] = 130.

    # Define receiver geometry (same as source, but spread across `x, y`)
    rec2 = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec2.coordinates.data[:, 0] = np.linspace(100., 100. + model.domain_size[0],
                                              num=nrec)
    rec2.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    ox_g, oy_g, oz_g = tuple(o + 100. for o in model.grid.origin)

    rec1, u1, _ = solver.forward(save=False, src=src, rec=rec2,
                                 o_x=ox_g, o_y=oy_g, o_z=oz_g)

    assert(np.allclose(rec.data, rec1.data, atol=1e-5))


def test_edge_sparse():
    """
    Test that interpolation uses the correct point for the edge case
    where the sparse point is at the origin with non rational grid spacing.
    Due to round up error the interpolation would use the halo point instead of
    the point (0, 0) without the factorizaion of the expressions.
    """
    grid = Grid(shape=(16, 16), extent=(225., 225.), origin=(25., 35.))
    u = unit_box(shape=(16, 16), grid=grid)
    u._data_with_outhalo[:u.space_order, :] = -1
    u._data_with_outhalo[:, :u.space_order] = -1
    sf1 = SparseFunction(name='s', grid=u.grid, npoint=1)
    sf1.coordinates.data[0, :] = (25.0, 35.0)

    expr = sf1.interpolate(u)
    subs = {d.spacing: v for d, v in zip(u.grid.dimensions, u.grid.spacing)}
    op = Operator(expr, subs=subs)

    op()
    assert sf1.data[0] == 0


def test_msf_interpolate():
    """ Test interpolation with MatrixSparseTimeFunction which accepts
        precomputed values for interpolation coefficients, but this time
        with a TimeFunction
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

    gridpoints, interpolation_coeffs = precompute_linear_interpolation(points,
                                                                       grid, origin)

    matrix = scipy.sparse.eye(len(points))

    sf = MatrixSparseTimeFunction(
        name='s', grid=grid, r=r, matrix=matrix, nt=5
    )

    sf.gridpoints.data[:] = gridpoints
    sf.coefficients_x.data[:] = interpolation_coeffs[:, 0, :]
    sf.coefficients_y.data[:] = interpolation_coeffs[:, 0, :]

    assert sf.data.shape == (5, 3)

    eqn = sf.interpolate(u)
    op = Operator(eqn)

    sf.manual_scatter()
    op(time_m=0, time_M=4)
    sf.manual_gather()

    for it in range(5):
        assert np.allclose(sf.data[it, :], it)

    # Now test injection
    u.data[:] = 0

    eqn_inject = sf.inject(field=u, expr=sf)
    op2 = Operator(eqn_inject)
    op2(time_m=0, time_M=4)

    # There should be 4 points touched for each source point
    # (5, 90), (1, 80), (7, 84) and x+1, y+1 for each
    nzt, nzx, nzy = np.nonzero(u.data)
    assert np.all(np.unique(nzx) == np.array([1, 2, 5, 6, 7, 8]))
    assert np.all(np.unique(nzy) == np.array([80, 81, 84, 85, 90, 91]))
    assert np.all(np.unique(nzt) == np.array([1, 2, 3, 4]))
    # 12 points x 4 timesteps
    assert nzt.size == 48


def test_sparse_first():
    """
    Tests custom sprase function with sparse dimension as first index.
    """

    class SparseFirst(SparseFunction):
        """ Custom sparse class with the sparse dimension as the first one"""
        _sparse_position = 0

    dr = Dimension("cd")
    ds = DefaultDimension("ps", default_value=3)
    grid = Grid((11, 11))
    dims = grid.dimensions
    s = SparseFirst(name="s", grid=grid, npoint=2, dimensions=(dr, ds), shape=(2, 3),
                    coordinates=[[.5, .5], [.2, .2]])

    # Check dimensions and shape are correctly initialized
    assert s.indices[s._sparse_position] == dr
    assert s.shape == (2, 3)
    assert s.coordinates.indices[0] == dr

    # Operator
    u = TimeFunction(name="u", grid=grid, time_order=1)
    fs = Function(name="fs", grid=grid, dimensions=(*dims, ds), shape=(11, 11, 3))

    eqs = [Eq(u.forward, u+1), Eq(fs, u)]
    # No time dependence so need the implicit dim
    rec = s.interpolate(expr=s+fs, implicit_dims=grid.stepping_dim)
    op = Operator(eqs + rec)

    op(time_M=10)
    expected = 10*11/2  # n (n+1)/2
    assert np.allclose(s.data, expected)


@switchconfig(safe_math=True)
def test_inject_function():
    nt = 11

    grid = Grid(shape=(5, 5))
    u = TimeFunction(name="u", grid=grid, time_order=2)
    src = SparseTimeFunction(name="src", grid=grid, nt=nt, npoint=1,
                             coordinates=[[0.5, 0.5]])

    nfreq = 5
    freq_dim = DefaultDimension(name="freq", default_value=nfreq)
    omega = Function(name="omega", dimensions=(freq_dim,), shape=(nfreq,), grid=grid)
    omega.data.fill(1.)

    inj = src.inject(field=u.forward, expr=omega)

    op = Operator([inj])

    op(time_M=0)
    assert u.data[1, 2, 2] == nfreq
    assert np.all(u.data[0] == 0)
    assert np.all(u.data[2] == 0)
    for i in [0, 1, 3, 4]:
        for j in [0, 1, 3, 4]:
            assert u.data[1, i, j] == 0


def test_interpolation_radius():
    nt = 11

    grid = Grid(shape=(5, 5))
    u = TimeFunction(name="u", grid=grid, space_order=0)
    src = SparseTimeFunction(name="src", grid=grid, nt=nt, npoint=1)
    try:
        src.interpolate(u)
        assert False
    except ValueError:
        assert True
