import numpy as np
from numpy import sin, floor
import pytest
from sympy import Float

from conftest import assert_structure
from devito import (Grid, Operator, Dimension, SparseFunction, SparseTimeFunction,
                    Function, TimeFunction, DefaultDimension, Eq, switchconfig,
                    PrecomputedSparseFunction, PrecomputedSparseTimeFunction,
                    MatrixSparseTimeFunction, SubDomain)
from devito.operations.interpolators import LinearInterpolator, SincInterpolator
from examples.seismic import (demo_model, TimeAxis, RickerSource, Receiver,
                              AcquisitionGeometry)
from examples.seismic.acoustic import AcousticWaveSolver, acoustic_setup
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
    gridpoints = np.array([tuple(floor((point[i]-origin[i])/grid.spacing[i])
                           for i in range(len(point))) for point in points])

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
    points = np.array([(.05, .9), (.01, .8), (0.07, 0.84)])
    origin = (0, 0)

    grid = Grid(shape=shape, origin=origin)

    def init(data):
        # This is data with halo so need to shift to match the m.data expectations
        print(grid.spacing)
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
    points = np.array([(.05, .9), (.01, .8), (0.07, 0.84)])
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


@pytest.mark.parametrize('r, interp', [(2, 'linear'), (4, 'sinc')])
def test_interpolation_radius(r, interp):
    nt = 11

    grid = Grid(shape=(5, 5))
    u = TimeFunction(name="u", grid=grid, space_order=0)
    src = SparseTimeFunction(name="src", grid=grid, nt=nt, npoint=1,
                             r=r, interpolation=interp)
    try:
        src.interpolate(u)
        assert False
    except ValueError:
        assert True


def test_interp_default():
    nt = 3
    grid = Grid(shape=(5, 5))

    src = SparseTimeFunction(name="src", grid=grid, nt=nt, npoint=1)
    assert isinstance(src.interpolator, LinearInterpolator)
    assert src.r == 1

    src = SparseTimeFunction(name="src", grid=grid, nt=nt, npoint=1, interpolation='sinc')
    assert isinstance(src.interpolator, SincInterpolator)
    assert src.r == 4

    src = SparseTimeFunction(name="src", grid=grid, nt=nt, npoint=1,
                             interpolation='sinc', r=6)
    assert isinstance(src.interpolator, SincInterpolator)
    assert src.r == 6


@pytest.mark.parametrize('r, tol', [(2, 0.051), (3, 0.003), (4, 0.008),
                                    (5, 0.002), (6, 0.0005), (7, 8e-5),
                                    (8, 6e-5), (9, 5e-5), (10, 4.2e-5)])
def test_sinc_accuracy(r, tol):
    so = max(2, r)
    solver_lin = acoustic_setup(preset='constant-isotropic', shape=(101, 101),
                                spacing=(10, 10), interpolation='linear', space_order=so)
    solver_sinc = acoustic_setup(preset='constant-isotropic', shape=(101, 101),
                                 spacing=(10, 10), interpolation='sinc', r=r,
                                 space_order=so)

    # On node source
    s_node = [500, 500]
    src_n = solver_lin.geometry.src
    src_n.coordinates.data[:] = s_node

    # Half node src
    s_mid = [505, 505]
    src_h = solver_lin.geometry.src
    src_h.coordinates.data[:] = s_mid

    # On node rec
    r_node = [750, 750]
    rec_n = solver_lin.geometry.new_src(name='rec', src_type=None)
    rec_n.coordinates.data[:] = r_node

    # Half node rec for linear
    r_mid = [755, 755]
    rec_hl = solver_lin.geometry.new_src(name='recl', src_type=None)
    rec_hl.coordinates.data[:] = r_mid

    # Half node rec for sinc
    r_mid = [755, 755]
    rec_hs = solver_lin.geometry.new_src(name='recs', src_type=None)
    rec_hs.coordinates.data[:] = r_mid

    # Reference solution, on node
    _, un, _ = solver_lin.forward(src=src_n, rec=rec_n)
    # Linear interp on half node
    _, ul, _ = solver_lin.forward(src=src_h, rec=rec_hl)
    # Sinc interp on half node
    _, us, _ = solver_sinc.forward(src=src_h, rec=rec_hs)

    # Check sinc is more accuracte
    nref = np.linalg.norm(rec_n.data)
    err_lin = np.linalg.norm(rec_n.data - rec_hl.data)/nref
    err_sinc = np.linalg.norm(rec_n.data - rec_hs.data)/nref

    print(f"Error linear: {err_lin}, Error sinc: {err_sinc}")
    assert np.isclose(err_sinc, 0, rtol=0, atol=tol)
    assert err_sinc < err_lin
    assert err_lin > 0.01


@pytest.mark.parametrize('dtype, expected', [(np.complex64, np.float32),
                                             (np.complex128, np.float64)])
def test_point_symbol_types(dtype, expected):
    """Test that positions are always real"""
    grid = Grid(shape=(11,))
    s = SparseFunction(name='src', npoint=1,
                       grid=grid, dtype=dtype)
    point_symbol = s.interpolator._point_symbols[0]

    assert point_symbol.dtype is expected


class SD0(SubDomain):
    name = 'sd0'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 6), y: y}


class SD1(SubDomain):
    name = 'sd1'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 2, 1), y: ('right', 6)}


class SD2(SubDomain):
    name = 'sd2'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 4), y: y}


class SD3(SubDomain):
    name = 'sd3'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 2, 1), y: ('right', 4)}


class SD4(SubDomain):
    name = 'sd4'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 7, 1), y: ('middle', 1, 6)}


class SD5(SubDomain):
    name = 'sd5'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 4, 3), y: ('middle', 4, 3)}


class SD6(SubDomain):
    name = 'sd6'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 9), y: ('right', 9)}


class SD7(SubDomain):
    name = 'sd7'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1)}


class TestSubDomainInterpolation:
    """
    Tests for interpolation onto and off of Functions defined on
    SubDomains.
    """

    def test_interpolate_subdomain(self):
        """
        Test interpolation off of a Function defined on a SubDomain.
        """

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sd0 = SD0(grid=grid)
        sd1 = SD1(grid=grid)

        f0 = Function(name='f0', grid=sd0)
        f1 = Function(name='f1', grid=sd1)
        f2 = Function(name='f2', grid=grid)

        xmsh, ymsh = np.meshgrid(np.arange(11), np.arange(11))
        msh = xmsh*ymsh
        f0.data[:] = msh[:6, :]
        f1.data[:] = msh[2:-1, -6:]
        f2.data[:] = msh

        sr0 = SparseFunction(name='sr0', grid=grid, npoint=8)
        sr1 = SparseFunction(name='sr1', grid=grid, npoint=8)
        sr2 = SparseFunction(name='sr2', grid=grid, npoint=8)

        coords = np.array([[2.5, 1.5], [4.5, 2.], [8.5, 4.],
                           [0.5, 6.], [7.5, 4.], [5.5, 5.5],
                           [1.5, 4.5], [7.5, 8.5]])

        sr0.coordinates.data[:] = coords
        sr1.coordinates.data[:] = coords
        sr2.coordinates.data[:] = coords

        rec0 = sr0.interpolate(f0)
        rec1 = sr1.interpolate(f1)
        rec2 = sr2.interpolate(f1 + f2)

        op = Operator([rec0, rec1, rec2])

        op.apply()

        # Note that interpolation points can go into the halo by
        # the radius of the SparseFunction.
        check0 = np.array([3.75, 9., 0., 3., 0., 13.75, 6.75, 0.])
        check1 = np.array([0., 0., 0., 0., 0., 30.25, 2.5, 63.75])
        check2 = np.array([[0., 0., 34., 3., 30., 60.5, 9.25, 127.5]])

        assert np.all(np.isclose(sr0.data, check0))
        assert np.all(np.isclose(sr1.data, check1))
        assert np.all(np.isclose(sr2.data, check2))
        assert_structure(op,
                         ['p_sr0', 'p_sr0rsr0xrsr0y', 'p_sr1',
                          'p_sr1rsr1xrsr1y', 'p_sr2', 'p_sr2rsr2xrsr2y'],
                         'p_sr0rsr0xrsr0yp_sr1rsr1xrsr1yp_sr2rsr2xrsr2y')

    def test_interpolate_subdomain_sinc(self):
        """
        Check that sinc interpolation off a SubDomain works as expected.
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sd0 = SD6(grid=grid)
        sd1 = SD7(grid=grid)

        f0 = Function(name='f0', grid=sd0, space_order=2)
        f1 = Function(name='f1', grid=sd1, space_order=2)
        f2 = Function(name='f2', grid=grid, space_order=2)

        xmsh, ymsh = np.meshgrid(np.arange(11), np.arange(11))
        msh = xmsh*ymsh
        f0.data[:] = msh[:9, -9:]
        f1.data[:] = msh[1:-1, 1:-1]
        f2.data[:] = msh[:]

        sr0 = SparseFunction(name='sr0', grid=grid, npoint=5, interpolation='sinc', r=2)
        sr1 = SparseFunction(name='sr1', grid=grid, npoint=5, interpolation='sinc', r=2)
        sr2 = SparseFunction(name='sr2', grid=grid, npoint=5, interpolation='sinc', r=2)

        coords = np.array([[2.5, 6.5], [3.5, 4.5], [6., 6.], [5.5, 4.5], [4.5, 6.]])

        sr0.coordinates.data[:] = coords
        sr1.coordinates.data[:] = coords
        sr2.coordinates.data[:] = coords

        rec0 = sr0.interpolate(f0)
        rec1 = sr1.interpolate(f1)
        rec2 = sr2.interpolate(f2)

        op = Operator([rec0, rec1, rec2])

        op.apply()

        assert np.all(np.isclose(sr0.data, sr2.data))
        assert np.all(np.isclose(sr1.data, sr2.data))
        assert_structure(op,
                         ['p_sr0', 'p_sr0rsr0xrsr0y', 'p_sr1',
                          'p_sr1rsr1xrsr1y', 'p_sr2', 'p_sr2rsr2xrsr2y'],
                         'p_sr0rsr0xrsr0yp_sr1rsr1xrsr1yp_sr2rsr2xrsr2y')

    def test_inject_subdomain(self):
        """
        Test injection into a Function defined on a SubDomain.
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sd0 = SD0(grid=grid)
        sd1 = SD1(grid=grid)

        f0 = Function(name='f0', grid=sd0)
        f1 = Function(name='f1', grid=sd1)

        sr0 = SparseFunction(name='sr0', grid=grid, npoint=8)

        coords = np.array([[2.5, 1.5], [4.5, 2.], [8.5, 4.],
                           [0.5, 6.], [7.5, 4.], [5.5, 5.5],
                           [1.5, 4.5], [7.5, 8.5]])

        sr0.coordinates.data[:] = coords

        src0 = sr0.inject(f0, Float(1.))
        src1 = sr0.inject(f1, Float(1.))

        op = Operator([src0, src1])

        op.apply()

        check0 = np.array([[0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.25, 0.25, 0.5, 0., 0., 0., 0.],
                           [0., 0.25, 0.25, 0., 0.25, 0.25, 0., 0., 0., 0., 0.],
                           [0., 0.25, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0.5, 0., 0., 0.25, 0.25, 0., 0., 0., 0.]])
        check1 = np.array([[0.25, 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.],
                           [0.25, 0.25, 0., 0., 0., 0.],
                           [0.25, 0.25, 0., 0., 0., 0.],
                           [0., 0., 0., 0.25, 0.25, 0.],
                           [0., 0., 0., 0.25, 0.25, 0.],
                           [0., 0., 0., 0., 0., 0.]])

        assert np.all(np.isclose(f0.data, check0))
        assert np.all(np.isclose(f1.data, check1))
        assert_structure(op,
                         ['p_sr0rsr0xrsr0y'],
                         'p_sr0rsr0xrsr0y')

    def test_inject_subdomain_sinc(self):
        """
        Check sinc injection into a Function defined on a SubDomain functions
        as expected.
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sd0 = SD6(grid=grid)
        sd1 = SD7(grid=grid)

        f0 = Function(name='f0', grid=sd0, space_order=2)
        f1 = Function(name='f1', grid=sd1, space_order=2)
        f2 = Function(name='f2', grid=grid, space_order=2)

        sr0 = SparseFunction(name='sr0', grid=grid, npoint=5, interpolation='sinc', r=2)

        coords = np.array([[2.5, 6.5], [3.5, 4.5], [6.0, 6.], [5.5, 4.5], [4.5, 6.]])

        sr0.coordinates.data[:] = coords

        src0 = sr0.inject(f0, Float(1.))
        src1 = sr0.inject(f1, Float(1.))
        src2 = sr0.inject(f2, Float(1.))

        op = Operator([src0, src1, src2])
        op.apply()

        assert np.all(np.isclose(f0.data, f2.data[:9, -9:]))
        assert np.all(np.isclose(f1.data, f2.data[1:-1, 1:-1]))
        assert_structure(op,
                         ['p_sr0rsr0xrsr0y'],
                         'p_sr0rsr0xrsr0y')

    @pytest.mark.parallel(mode=4)
    def test_interpolate_subdomain_mpi(self, mode):
        """
        Test interpolation off of a Function defined on a SubDomain with MPI.
        """

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sd2 = SD2(grid=grid)
        sd3 = SD3(grid=grid)
        sd4 = SD4(grid=grid)

        f0 = Function(name='f0', grid=sd2)
        f1 = Function(name='f1', grid=sd3)
        f2 = Function(name='f2', grid=grid)
        f3 = Function(name='f3', grid=sd4)

        xmsh, ymsh = np.meshgrid(np.arange(11), np.arange(11))
        msh = xmsh*ymsh
        f0.data[:] = msh[:6, :]
        f1.data[:] = msh[2:-1, -6:]
        f2.data[:] = msh
        f3.data[:] = msh[7:-1, 1:-6]

        sr0 = SparseFunction(name='sr0', grid=grid, npoint=8)
        sr1 = SparseFunction(name='sr1', grid=grid, npoint=8)
        sr2 = SparseFunction(name='sr2', grid=grid, npoint=8)
        sr3 = SparseFunction(name='sr3', grid=grid, npoint=8)

        coords = np.array([[2.5, 1.5], [4.5, 2.], [8.5, 4.],
                           [0.5, 6.], [7.5, 4.], [5.5, 5.5],
                           [1.5, 4.5], [7.5, 8.5]])

        sr0.coordinates.data[:] = coords
        sr1.coordinates.data[:] = coords
        sr2.coordinates.data[:] = coords
        sr3.coordinates.data[:] = coords

        rec0 = sr0.interpolate(f0)
        rec1 = sr1.interpolate(f1)
        rec2 = sr2.interpolate(f1 + f2)
        rec3 = sr3.interpolate(f3)

        op = Operator([rec0, rec1, rec2, rec3])

        op.apply()

        if grid.distributor.myrank == 0:
            assert np.all(np.isclose(sr0.data, [3.75, 0.]))
            assert np.all(np.isclose(sr1.data, [0., 0.]))
            assert np.all(np.isclose(sr2.data, [0., 0.]))
            assert np.all(np.isclose(sr3.data, [0., 0.]))
        elif grid.distributor.myrank == 1:
            assert np.all(np.isclose(sr0.data, [0., 3.]))
            assert np.all(np.isclose(sr1.data, [0., 0.]))
            assert np.all(np.isclose(sr2.data, [0., 3.]))
            assert np.all(np.isclose(sr3.data, [34., 0.]))
        elif grid.distributor.myrank == 2:
            assert np.all(np.isclose(sr0.data, [0., 0.]))
            assert np.all(np.isclose(sr1.data, [0., 0.]))
            assert np.all(np.isclose(sr2.data, [0., 16.5]))
            assert np.all(np.isclose(sr3.data, [30., 0.]))
        elif grid.distributor.myrank == 3:
            assert np.all(np.isclose(sr0.data, [6.75, 0.]))
            assert np.all(np.isclose(sr1.data, [0., 48.75]))
            assert np.all(np.isclose(sr2.data, [0., 112.5]))
            assert np.all(np.isclose(sr3.data, [0., 0.]))

    @pytest.mark.parallel(mode=4)
    def test_inject_subdomain_mpi(self, mode):
        """
        Test injection into a Function defined on a SubDomain with MPI.
        """

        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sd2 = SD2(grid=grid)
        sd3 = SD3(grid=grid)
        sd4 = SD4(grid=grid)
        sd5 = SD5(grid=grid)

        f0 = Function(name='f0', grid=sd2)
        f1 = Function(name='f1', grid=sd3)
        f2 = Function(name='f2', grid=sd4)
        f3 = Function(name='f3', grid=sd5)

        sr0 = SparseFunction(name='sr0', grid=grid, npoint=8)

        coords = np.array([[2.5, 1.5], [4.5, 2.], [8.5, 4.],
                           [0.5, 6.], [7.5, 4.], [5.5, 5.5],
                           [1.5, 4.5], [7.5, 8.5]])

        sr0.coordinates.data[:] = coords

        src0 = sr0.inject(f0, Float(1.))
        src1 = sr0.inject(f1, Float(1.))
        src2 = sr0.inject(f2, Float(1.))
        src3 = sr0.inject(f3, Float(1.))

        op = Operator([src0, src1, src2, src3])

        op.apply()

        check0 = np.array([[0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.,],
                           [0., 0., 0., 0., 0.25, 0.25, 0.5, 0., 0., 0., 0.],
                           [0., 0.25, 0.25, 0., 0.25, 0.25, 0., 0., 0., 0., 0.],
                           [0., 0.25, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.]])
        check1 = np.array([[0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0.25, 0.25, 0.],
                           [0., 0.25, 0.25, 0.],
                           [0., 0., 0., 0.]])
        check2 = np.array([[0., 0., 0., 0.5],
                           [0., 0., 0., 1.],
                           [0., 0., 0., 0.5]])
        check3 = np.array([[0., 0., 0., 0.],
                           [0., 0.25, 0.25, 0.],
                           [0., 0.25, 0.25, 0.],
                           [0.5, 0., 0., 0.]])

        # Can't gather inside the assert as it hangs due to the if condition
        data0 = f0.data_gather()
        data1 = f1.data_gather()
        data2 = f2.data_gather()
        data3 = f3.data_gather()

        if grid.distributor.myrank == 0:
            assert np.all(data0 == check0)
            assert np.all(data1 == check1)
            assert np.all(data2 == check2)
            assert np.all(data3 == check3)
        else:
            # Size zero array of None, so can't check "is None"
            # But check equal to None works, even though this is discouraged
            assert data0 == None  # noqa
            assert data1 == None  # noqa
            assert data2 == None  # noqa
            assert data3 == None  # noqa
