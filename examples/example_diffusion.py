"""
This example encodes multiple ways to solve the 2D diffusion equations
using an explicit finite difference scheme with fixed boundary values
and a given initial value for the density.
"""
import time
from argparse import ArgumentParser

import numpy as np
from sympy import Eq, Function, as_finite_diff, lambdify, solve
from sympy.abc import h, s, t, x, y

from devito import Operator, TimeData
from devito.logger import log


def ring_initial(dx=0.01, dy=0.01):
    """Initialise grid with initial condition ("ring")"""
    nx, ny = int(1 / dx), int(1 / dy)
    xx, yy = np.meshgrid(np.linspace(0., 1., nx, dtype=np.float32),
                         np.linspace(0., 1., ny, dtype=np.float32))
    ui = np.zeros((nx, ny), dtype=np.float32)
    r = (xx - .5)**2. + (yy - .5)**2.
    ui[np.logical_and(.05 <= r, r <= .1)] = 1.
    return ui


def execute_python(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using pure Python list indexing."""
    nx, ny = ui.shape
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.concatenate((ui, np.zeros_like(ui))).reshape((2, nx, ny))

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        t0 = ti % 2
        t1 = (ti + 1) % 2
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                uxx = (u[t0, i+1, j] - 2*u[t0, i, j] + u[t0, i-1, j]) / dx2
                uyy = (u[t0, i, j+1] - 2*u[t0, i, j] + u[t0, i, j-1]) / dy2
                u[t1, i, j] = u[t0, i, j] + dt * a * (uxx + uyy)
    tfinish = time.time()
    log("Python: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (dx, dy, timesteps, tfinish - tstart))
    return u[ti % 2, :, :]


def execute_numpy(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using vectorised numpy array accesses."""
    nx, ny = ui.shape
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.concatenate((ui, np.zeros_like(ui))).reshape((2, nx, ny))

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        t0 = ti % 2
        t1 = (ti + 1) % 2

        uxx = (u[t0, 2:, 1:-1] - 2*u[t0, 1:-1, 1:-1] + u[t0, :-2, 1:-1]) / dx2
        uyy = (u[t0, 1:-1, 2:] - 2*u[t0, 1:-1, 1:-1] + u[t0, 1:-1, :-2]) / dy2
        u[t1, 1:-1, 1:-1] = u[t0, 1:-1, 1:-1] + a * dt * (uxx + uyy)
    tfinish = time.time()
    log("Numpy: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (dx, dy, timesteps, tfinish - tstart))
    return u[ti % 2, :, :]


def execute_lambdify(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using vectorised numpy array accesses."""
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.concatenate((ui, np.zeros_like(ui))).reshape((2, nx, ny))

    def diffusion_stencil():
        """Create stencil and substitutions for the diffusion equation"""
        p = Function('p')
        dx2 = as_finite_diff(p(x, y, t).diff(x, x), [x - h, x, x + h])
        dy2 = as_finite_diff(p(x, y, t).diff(y, y), [y - h, y, y + h])
        dt = as_finite_diff(p(x, y, t).diff(t), [t, t + s])
        eqn = Eq(dt, a * (dx2 + dy2))
        stencil = solve(eqn, p(x, y, t + s))[0]
        return stencil, (p(x, y, t), p(x + h, y, t), p(x - h, y, t),
                         p(x, y + h, t), p(x, y - h, t), s, h)
    stencil, subs = diffusion_stencil()
    kernel = lambdify(subs, stencil, 'numpy')

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        t0 = ti % 2
        t1 = (ti + 1) % 2
        u[t1, 1:-1, 1:-1] = kernel(u[t0, 1:-1, 1:-1], u[t0, 2:, 1:-1],
                                   u[t0, :-2, 1:-1], u[t0, 1:-1, 2:],
                                   u[t0, 1:-1, :-2], dt, dx)
    tfinish = time.time()
    log("Lambdify: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (dx, dy, timesteps, tfinish - tstart))
    return u[ti % 2, :, :]


def execute_devito(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using the devito Operator API."""
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeData(name='u', shape=(nx, ny), time_order=1, space_order=2)
    u.data[0, :] = ui[:]

    # Derive the stencil according to devito conventions
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
    stencil = solve(eqn, u.forward)[0]
    op = Operator(stencils=Eq(u.forward, stencil), subs={h: dx, s: dt},
                  nt=timesteps, shape=(nx, ny), spc_border=1, time_order=1)
    # Execute the generated Devito stencil operator
    tstart = time.time()
    op.apply()
    tfinish = time.time()
    log("Devito: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (dx, dy, timesteps, tfinish - tstart))
    return u.data[1, :]


def animate(field):
    """Animate solution field"""
    import matplotlib.pyplot as plt
    xvals = np.linspace(0, 1., field.shape[0])
    yvals = np.linspace(0, 1., field.shape[1])
    levels = np.linspace(-.1, 1.1, 21)
    cs = plt.contourf(xvals, yvals, field, levels=levels)
    plt.colorbar(cs)
    plt.show()


def test_diffusion2d(dx=0.01, dy=0.01, timesteps=1000):
    ui = ring_initial(dx=dx, dy=dy)
    u = execute_devito(ui, dx=dx, dy=dy, timesteps=timesteps)
    assert(u.max() < 2.4)
    assert(np.linalg.norm(u, ord=2) < 13)


if __name__ == "__main__":
    description = """
Example script demonstrating several approaches to solving the
2D diffusion equation. The various modes are pure Python,
vectorized numpy, a lambdified SymPy equation and the Devito API.

Please not that the default settings for dx, dy and timesteps are
chosen to highlight that Devito and "vectorised numpy" are
faster than pure Python or the sympy.lambdify kernels approach.
For a fair performance comparison between Devito and
"vectorised numpy" we recommend using --spacing 0.001 -t 1000.
"""
    parser = ArgumentParser(description=description)
    parser.add_argument(dest='mode', nargs='+', default='devito',
                        help="Execution modes: python, numpy, lambdify, devito")
    parser.add_argument('-s', '--spacing', type=float, nargs='+', default=[0.01],
                        help='Grid spacing, either single value or dx, dy)')
    parser.add_argument('-t', '--timesteps', type=int, default=20,
                        help='Number of timesteps to run')
    parser.add_argument('--show', action='store_true',
                        help="Show animation of the solution field")
    args = parser.parse_args()
    if len(args.spacing) > 2:
        raise ValueError("Too many arguments encountered for --spacing")

    # Get the relevant execution method
    executor = {'python': execute_python, 'numpy': execute_numpy,
                'lambdify': execute_lambdify, 'devito': execute_devito}
    dx = args.spacing[0]
    dy = args.spacing[1] if len(args.spacing) > 1 else args.spacing[0]
    for mode in args.mode:
        ui = ring_initial(dx=dx, dy=dy)
        if args.show:
            animate(ui)
        u = executor[mode](ui, dx=dx, dy=dy, timesteps=args.timesteps)
        if args.show:
            animate(u)
