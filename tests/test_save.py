import numpy as np
from sympy import solve
from conftest import skipif_yask

from devito import Grid, Eq, Operator, TimeFunction, Forward


def initial(dx=0.01, dy=0.01):
    nx, ny = int(1 / dx), int(1 / dy)
    xx, yy = np.meshgrid(np.linspace(0., 1., nx, dtype=np.float32),
                         np.linspace(0., 1., ny, dtype=np.float32))
    ui = np.zeros((nx, ny), dtype=np.float32)
    r = (xx - .5)**2. + (yy - .5)**2.
    ui[np.logical_and(.05 <= r, r <= .1)] = 1.

    return ui


def initializer(data):
    data[0, :] = initial()


def run_simulation(save=False, dx=0.01, dy=0.01, a=0.5, timesteps=100):
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

    grid = Grid(shape=(nx, ny))
    u = TimeFunction(
        name='u', grid=grid, save=timesteps if save else None,
        initializer=initializer,
        time_order=1, space_order=2
    )

    eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
    stencil = solve(eqn, u.forward)[0]
    op = Operator(Eq(u.forward, stencil), time_axis=Forward)
    op.apply(time=timesteps, dt=dt)

    if save:
        return u.data[timesteps - 1, :]
    else:
        return u.data[(timesteps+1) % 2, :]


@skipif_yask
def test_save():
    assert(np.array_equal(run_simulation(True), run_simulation()))
