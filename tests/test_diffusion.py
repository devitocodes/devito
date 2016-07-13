"""
This example encodes multiple ways to solve the 2D diffusion equations
using an explicit finite difference scheme with fixed boundary values
and a given initial value for the density.
"""
import numpy as np
import time
from sympy import Function, Eq, symbols, as_finite_diff, solve, lambdify
from sympy.abc import x, y, t
from devito import TimeData, Operator


def ring_initial(dx=0.01, dy=0.01):
    """Initialise grid with initial condition ("ring")"""
    nx, ny = int(1 / dx), int(1 / dy)
    xx, yy = np.meshgrid(np.linspace(0., 1., nx, dtype=np.float32),
                         np.linspace(0., 1., ny, dtype=np.float32))
    ui = np.zeros((nx, ny), dtype=np.float32)
    r = (xx - .5)**2. + (yy - .5)**2.
    ui[np.logical_and(.05 <= r, r <= .1)] = 1.
    return ui


def diffusion_stencil():
    """Create stencil and substitutions for the diffusion equation"""
    p = Function('p')
    s, h, a = symbols('s h a')
    dxx = as_finite_diff(p(x, y, t).diff(x, x), [x - h, x, x + h])
    dyy = as_finite_diff(p(x, y, t).diff(y, y), [y - h, y, y + h])
    dt = as_finite_diff(p(x, y, t).diff(t), [t, t + s])
    equation = a * (dxx + dyy) - dt
    stencil = solve(equation, p(x, y, t + s))[0]
    return stencil, (p(x, y, t), p(x + h, y, t), p(x - h, y, t),
                     p(x, y + h, t), p(x, y - h, t), s, h, a)


def execute_python(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using pure Python list indexing."""
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.zeros_like(ui)

    def single_step(u, ui):
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                uxx = (ui[i+1, j] - 2*ui[i, j] + ui[i-1, j]) / dx2
                uyy = (ui[i, j+1] - 2*ui[i, j] + ui[i, j-1]) / dy2
                u[i, j] = ui[i, j] + dt * a * (uxx + uyy)

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        if ti % 2 == 0:
            single_step(u, ui)
        else:
            single_step(ui, u)
    tfinish = time.time()
    print "Python: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"\
        % (dx, dy, timesteps, tfinish - tstart)
    return u if ti % 2 == 0 else ui


def execute_numpy(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using vectorised numpy array accesses."""
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.zeros_like(ui)

    def single_step(u, ui):
        uxx = (ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1]) / dx2
        uyy = (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2]) / dy2
        u[1:-1, 1:-1] = ui[1:-1, 1:-1] + a * dt * (uxx + uyy)

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        if ti % 2 == 0:
            single_step(u, ui)
        else:
            single_step(ui, u)
    tfinish = time.time()
    print "Numpy: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"\
        % (dx, dy, timesteps, tfinish - tstart)
    return u if ti % 2 == 0 else ui


def execute_lambdify(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using vectorised numpy array accesses."""
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.zeros_like(ui)
    stencil, subs = diffusion_stencil()
    kernel = lambdify(subs, stencil, 'numpy')

    def single_step(u, ui):
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[i, j] = kernel(ui[i, j], ui[i+1, j], ui[i-1, j],
                                 ui[i, j+1], ui[i, j-1], dt, dx, a)

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        if ti % 2 == 0:
            single_step(u, ui)
        else:
            single_step(ui, u)
    tfinish = time.time()
    print "Lambdify: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"\
        % (dx, dy, timesteps, tfinish - tstart)
    return u if ti % 2 == 0 else ui


def execute_devito(ui, dx=0.01, dy=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using the devito Operator API."""
    nx, ny = int(1 / dx), int(1 / dy)
    dx2, dy2 = dx**2, dy**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeData(name='u', shape=(nx, ny), time_order=1)
    u.data[0, :] = ui[:]

    # Derive the stencil according to devito conventions
    a, h, s = symbols('a h s')
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
    stencil = solve(eqn, u.forward)[0]
    op = Operator(subs=[a, h, s], nt=timesteps, shape=(nx, ny), spc_border=1,
                  time_order=1, stencils=[(Eq(u.forward, stencil), [0.5, dx, dt])])
    # Execute the generated Devito stencil operator
    tstart = time.time()
    op.apply()
    tfinish = time.time()
    print "Devito: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"\
        % (dx, dy, timesteps, tfinish - tstart)
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
    """Below is a demonstration of various techniques to solve the
    simple 2D diffusion equation, including simple Python, vectorized
    numpy, a lambdified SymPy equation and the Devito API.

    Please not that the current settings for dx, dy, and timesteps are
    chosen to highlight that Devito and "vectorised numpy" are
    significantly faster than the "pure Python" or "lambdified SymPy"
    approach. For a fair performance comparison between Devito and
    "vectorised numpy" we recommend disabling the slow variants and
    using the following values:

    dx, dy = 0.001, 0.001
    timesteps = 1000
    """
    dx, dy = 0.01, 0.01
    timesteps = 20

    # Execute diffusion via devito
    ui = ring_initial(dx=dx, dy=dy)
    animate(ui)
    u = execute_devito(ui, dx=dx, dy=dy, timesteps=timesteps)
    animate(u)

    # Execute diffusion with vectorised numpy arrays
    ui = ring_initial(dx=dx, dy=dy)
    animate(ui)
    u = execute_numpy(ui, dx=dx, dy=dy, timesteps=timesteps)
    animate(u)

    # Execute diffusion with pure Python list accesses
    ui = ring_initial(dx=dx, dy=dy)
    animate(ui)
    u = execute_python(ui, dx=dx, dy=dy, timesteps=timesteps)
    animate(u)

    # Execute diffusion from labdified SymPy expression
    ui = ring_initial(dx=dx, dy=dy)
    animate(ui)
    u = execute_lambdify(ui, dx=dx, dy=dy, timesteps=timesteps)
    animate(u)
