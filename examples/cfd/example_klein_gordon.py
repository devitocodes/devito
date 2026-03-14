"""
Klein-Gordon equation example
=============================

Solves the Klein-Gordon equation in 2D using Devito's symbolic API:

    ∂²u/∂t² = c² ∇²u - m² u

The Klein-Gordon equation is the relativistic generalization of the wave equation
with a mass term.  It describes spin-0 scalar fields in quantum field theory.
When m=0, it reduces to the standard wave equation.

This example uses an explicit finite-difference scheme with second-order time
stepping and configurable spatial order.

Usage::

    python example_klein_gordon.py run
    python example_klein_gordon.py run --shape 200 200 --timesteps 400
    python example_klein_gordon.py run -so 8 --mass 2.0
    python example_klein_gordon.py run --show  # requires matplotlib
"""

import numpy as np

from devito import Eq, Grid, Operator, TimeFunction, solve


def klein_gordon(shape=(100, 100), spacing=(1.0, 1.0), speed=1.0, mass=1.0,
                 space_order=4, timesteps=200, dt=None, show=False):
    """
    Solve the Klein-Gordon equation on a 2D grid.

    Parameters
    ----------
    shape : tuple of int
        Number of grid points in each dimension.
    spacing : tuple of float
        Grid spacing in each dimension.
    speed : float
        Wave propagation speed *c*.
    mass : float
        Mass parameter *m*.  Set to 0 for the standard wave equation.
    space_order : int
        Spatial finite-difference order (higher = more accurate).
    timesteps : int
        Number of time steps to compute.
    dt : float, optional
        Time step size.  If None, computed from the CFL condition.
    show : bool
        Whether to plot the final wavefield (requires matplotlib).

    Returns
    -------
    numpy.ndarray
        The final wavefield u.
    """
    grid = Grid(shape=shape, extent=tuple(s * (n - 1) for s, n in
                                          zip(spacing, shape)))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)

    # CFL-stable time step: dt <= h_min / (c * sqrt(ndim))
    if dt is None:
        h_min = min(spacing)
        dt = 0.5 * h_min / (speed * np.sqrt(len(shape)))

    # Klein-Gordon PDE: u.dt2 = c^2 * laplace(u) - m^2 * u
    pde = Eq(u.dt2, speed**2 * u.laplace - mass**2 * u)

    # Solve for the forward time step
    stencil = solve(pde, u.forward)
    update = Eq(u.forward, stencil)

    op = Operator([update])

    # Gaussian initial condition centred on the grid
    cx = shape[0] // 2
    cy = shape[1] // 2
    sigma = min(shape) / 10.0
    xx, yy = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        indexing='ij',
    )
    gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    u.data[0, :] = gaussian
    u.data[1, :] = gaussian  # zero initial velocity

    op.apply(time_M=timesteps, dt=dt)

    if show:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 6))
            plt.imshow(u.data[-1].T, origin='lower', cmap='RdBu_r',
                       vmin=-0.5, vmax=0.5)
            plt.colorbar(label='u')
            plt.title(f'Klein-Gordon (c={speed}, m={mass}) at t={timesteps * dt:.1f}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

    return u.data[-1]


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Klein-Gordon equation example')
    parser.add_argument('mode', nargs='?', default='run')
    parser.add_argument('--shape', type=int, nargs='+', default=[100, 100])
    parser.add_argument('--spacing', type=float, nargs='+', default=[1.0, 1.0])
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--mass', type=float, default=1.0)
    parser.add_argument('-so', '--space_order', type=int, default=4)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    result = klein_gordon(
        shape=tuple(args.shape),
        spacing=tuple(args.spacing),
        speed=args.speed,
        mass=args.mass,
        space_order=args.space_order,
        timesteps=args.timesteps,
        dt=args.dt,
        show=args.show,
    )
    print(f'Klein-Gordon completed: wavefield range [{result.min():.4f}, '
          f'{result.max():.4f}]')
