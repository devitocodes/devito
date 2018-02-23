import numpy as np
from sympy import Eq
import pytest
from conftest import skipif_yask

from devito.cgen_utils import FLOAT
from devito import Grid, Operator, Function, SparseFunction, Dimension, TimeFunction
from examples.seismic import demo_model, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver


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


def points(grid, ranges, npoints, name='points', **kwargs):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseFunction(name=name, grid=grid, npoint=npoints, **kwargs)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


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
@pytest.mark.parametrize('shape, coords, axis', [
    ((11, 11), [(.2, .2), (.5, .5)], 0),
    ((11, 11), [(.2, .2), (.5, .5)], 1),
    ((11, 11, 11), [(.5, .5), (.3, .3), (.2, .2)], 0),
    ((11, 11, 11), [(.5, .5), (.3, .3), (.2, .2)], 1),
    ((11, 11, 11), [(.5, .5), (.3, .3), (.2, .2)], 2)
])
def test_inject_fd(shape, coords, axis):
    """
    Test injection for a spatial derivative of the point
    For the x derivative, he result should be
    point / (h_x/2) at coord_x - .5
    - point / (h_x/2) at coord_x + .5
    which with interpolation leads to
    u[p, y] = point / (h_x/2)
    u[p+1, y] = -point / (h_x/2)
    """
    a = unit_box(shape=shape)
    a.data[:] = 0.
    p = points(a.grid, ranges=coords, npoints=1, space_order=1)
    p.data[:] = 1.0

    derivative = getattr(p, 'd%s' % a.grid.dimensions[axis])
    expr = p.inject(a, derivative)

    Operator(expr)(a=a)

    # Compute coordinat to indices for a .1 grid spacing
    indices1 = [slice(int(10*c[0]), int(10*c[0]) + 1, 1) for c in coords]
    indices1[axis] = slice(int(coords[axis][0]*10) - 1, int(coords[axis][0]*10), 1)
    indices2 = [slice(int(c[0]*10), int(c[0]*10) + 1, 1) for c in coords]
    # Check values
    assert a.data[indices1] == 10.0
    assert a.data[indices2] == -10.0


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
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time_values = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time_values)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 30.

    # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=2,
                                space_order=4)

    rec, u, _ = solver.forward(save=False)

    # Define source geometry (center of domain, just below surface) with 100. origin
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time_values)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5 + 100.
    src.coordinates.data[0, -1] = 130.

    # Define receiver geometry (same as source, but spread across x)
    rec2 = Receiver(name='rec', grid=model.grid, ntime=nt, npoint=nrec)
    rec2.coordinates.data[:, 0] = np.linspace(100., 100. + model.domain_size[0],
                                              num=nrec)
    rec2.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    rec1, u1, _ = solver.forward(save=False, src=src, rec=rec2,
                                 o_x=100., o_y=100., o_z=100.)

    assert(np.allclose(rec.data, rec1.data, atol=1e-5))


@skipif_yask
@pytest.mark.parametrize('shape', [(50, 50, 50), (50, 50)])
def test_time_shift(shape):
    t0 = 0.0  # Start time
    tn = 200.  # Final time

    # Derive timestepping from model spacing
    dt = .1
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time_values = np.linspace(t0, tn, nt)  # Discretized time axis
    nrec = 101

    # stencil
    grid = Grid(shape, extent=(500., 500.))

    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
    eq = [Eq(u.forward, 2*u - u.backward + dt**2*u.laplace)]

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=grid, f0=0.01, time=time_values)
    src.coordinates.data[0, :] = np.array(grid.shape_domain) * 5.

    # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='rec', grid=grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., 500., num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    src_eq = src.inject(field=u.forward, expr=src)
    rec_eq = rec.interpolate(u)
    # Usual operator
    op = Operator(eq + src_eq + rec_eq)
    op.apply(dt=dt)

    # Shift rec
    rec2 = Receiver(name='rec', grid=grid, ntime=nt, npoint=nrec)
    rec2.coordinates.data[:, 0] = np.linspace(0., 500., num=nrec)
    rec2.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]
    rec_eq2 = rec2.interpolate(u, p_t=rec2.indices[0]+1)
    op = Operator(eq + src_eq + rec_eq2)
    u.data[:] = 0.
    op.apply(dt=dt, rec=rec2)

    # Shift u
    rec3 = Receiver(name='rec', grid=grid, ntime=nt, npoint=nrec)
    rec3.coordinates.data[:, 0] = np.linspace(0., 500., num=nrec)
    rec3.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]
    rec_eq3 = rec3.interpolate(u, u_t=u.indices[0] + 1)
    op = Operator(eq + src_eq + rec_eq3)
    u.data[:] = 0.
    op.apply(dt=dt, rec=rec3)

    assert(np.allclose(rec.data[:-1, :], rec2.data[1:, :], atol=1e-10))
    assert(np.allclose(rec.data[1:-1, :], rec3.data[:-2, :], atol=1e-10))


if __name__ == "__main__":
    test_time_shift((51, 51))
