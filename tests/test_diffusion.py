"""
Illustrative example of solving the 2D diffusion equation following:
http://www.timteatro.net/2010/10/29/performance-python-solving-the-2d-diffusion-equation-with-numpy

This example encodes multiple ways to solve the 2D diffusion equations
using an explicit finite difference scheme with fixed boundary values
and a given initial value for the density.
"""
import numpy as np
import time
from sympy import Function, Eq, symbols, as_finite_diff, solve, lambdify
from sympy.abc import x, y, z, t
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
    u = TimeData(name='u', spc_shape=(nx, ny), time_dim=-666, time_order=1,
                 save=False, dtype=np.float32)
    u.data[0, :] = ui[:]

    # Derive the stencil according to devito conventions
    def diffusion_stencil_devito():
        # Note: Apparently 2D grid indexing is assumed to be (x, z)
        p = Function('p')
        s, h, a = symbols('s h a')
        dxx = as_finite_diff(p(x, z, t-s).diff(x, x), [x - h, x, x + h])
        dzz = as_finite_diff(p(x, z, t-s).diff(z, z), [z - h, z, z + h])
        dt = as_finite_diff(p(x, z, t).diff(t), [t - s, t])
        equation = a * (dxx + dzz) - dt
        stencil = solve(equation, p(x, z, t))
        return stencil[0], [a, h, s]

    # Prepare the stencil to make it devito conformant
    # Note: The setup step needs some serious cleaning up
    from examples.fwi_operators import FWIOperator
    stencil, subs = diffusion_stencil_devito()
    stencil = FWIOperator.smart_sympy_replace(num_dim=2, time_order=1, expr=stencil,
                                              fun=Function('p'), arr=u, fw=True)
    op = Operator(subs, timesteps, (nx, ny), spc_border=1, time_order=1,
                  stencils=[(Eq(u[t, x, z], stencil), [a, dx, dt])],
                  input_params=[u], output_params=[])
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


if __name__ == "__main__":
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
