import numpy as np

from devito import Buffer, Grid, Eq, Operator, TimeFunction, solve


def initial(nt, nx, ny):
    xx, yy = np.meshgrid(np.linspace(0., 1., nx, dtype=np.float32),
                         np.linspace(0., 1., ny, dtype=np.float32))
    ui = np.zeros((nt, nx, ny), dtype=np.float32)
    r = (xx - .5)**2. + (yy - .5)**2.
    ui[0, np.logical_and(.05 <= r, r <= .1)] = 1.

    return ui


def initializer(data):
    data[:] = initial(*data.shape)


def run_simulation(save=False, dx=0.01, dy=0.01, a=0.5, timesteps=100):
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

    grid = Grid(shape=(nx, ny))
    u = TimeFunction(name='u', grid=grid, save=timesteps if save else None,
                     initializer=initializer, time_order=1, space_order=2)

    eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
    stencil = solve(eqn, u.forward)
    op = Operator(Eq(u.forward, stencil))
    op.apply(time=timesteps-2, dt=dt)

    return u.data[timesteps - 1]


def test_save():
    assert(np.array_equal(run_simulation(True), run_simulation()))


def test_buffer_api():
    """Tests memory allocation with different values of ``save``."""
    grid = Grid(shape=(3, 3))
    u0 = TimeFunction(name='u', grid=grid, time_order=2)
    u1 = TimeFunction(name='u', grid=grid, save=20, time_order=2)
    u2 = TimeFunction(name='u', grid=grid, save=Buffer(2), time_order=2)

    assert u0.shape[TimeFunction._time_position] == 3
    assert u1.shape[TimeFunction._time_position] == 20
    assert u2.shape[TimeFunction._time_position] == 2

    assert u0._time_buffering
    assert not u1._time_buffering
    assert u2._time_buffering
