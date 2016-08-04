import numpy as np
from sympy import Eq, solve, symbols

from devito.interfaces import TimeData
from devito.operator import Operator


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

    u = TimeData(
        name='u', shape=(nx, ny), time_dim=timesteps,
        time_order=1, space_order=2, save=save, pad_time=save
    )
    u.set_initializer(initializer)

    a, h, s = symbols('a h s')
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
    stencil = solve(eqn, u.forward)[0]
    op = Operator(stencils=Eq(u.forward, stencil), substitutions={a: 0.5, h: dx, s: dt},
                  nt=timesteps, shape=(nx, ny), spc_border=1, time_order=1)
    op.apply()

    if save:
        return u.data[timesteps - 1, :]
    else:
        return u.data[0, :]


def test_save():
    assert(np.array_equal(run_simulation(True), run_simulation()))
