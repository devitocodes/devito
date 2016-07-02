"""
Illustrative example of solving the 2D diffusion equation following:
http://www.timteatro.net/2010/10/29/performance-python-solving-the-2d-diffusion-equation-with-numpy

This example encodes multiple ways to solve the 2D diffusion equations
using an explicit finite difference scheme with fixed boundary values
and a given initial value for the density.
"""
import numpy as np
import time


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
