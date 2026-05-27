""" Script that demonstrates the functionality of the superstep in 1D and 2D
with an initial condition
In 1D: "Wave on a string"
In 2d: "Ripple on a pond"
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import reduce
from operator import mul

import matplotlib.pyplot as plt
import numpy as np

from devito import ConditionalDimension, Eq, Function, Grid, Operator, TimeFunction, solve
from devito.timestepping.superstep import superstep_generator


@dataclass
class Parameters:
    # Spatial
    shape: tuple[int]
    origin: tuple[float]
    extent: tuple[float]
    # Time
    t0: float
    t1: float
    critical_dt: float
    # Initial Condition
    mu: float
    sigma_sq: float
    lim: float


def gaussian_1d(x, mu=0, sigma_sq=1):
    """
    Generate a 1D Gaussian initial condition
    """
    return np.exp(-((x - mu)**2)/(2*sigma_sq))/(np.sqrt(2*np.pi*sigma_sq))


def gaussian(dims, mu=0, sigma_sq=1):
    """
    Generate an N-dimensional Gaussian initial condition
    """
    return reduce(mul, [gaussian_1d(d, mu=mu, sigma_sq=sigma_sq) for d in dims])


def simulate_ic(parameters, step=1, snapshots=-1):
    p = parameters
    d = len(p.shape)
    # Construct Grid
    grid = Grid(shape=p.shape, extent=p.extent)

    # Need to ensure that the velocity function supports the largest superstep stencil
    velocity = Function(name="velocity", grid=grid, space_order=(2, step, step))
    velocity.data[:] = 1500 if d == 2 else 1

    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

    pde = (1/velocity**2)*u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Initial condition
    msh = np.meshgrid(*[
        np.linspace(o, e, s) for o, e, s
        in zip(p.origin, p.extent, p.shape, strict=True)
    ])
    ic = gaussian(msh, mu=p.mu, sigma_sq=p.sigma_sq)

    # Stencil and operator
    if step == 1:
        # Non-superstep case
        new_u = u
        stencil = [stencil]
        new_u.data[0, :] = ic
        new_u.data[1, :] = ic
    else:
        new_u, new_u_p, *stencil = superstep_generator(u, stencil.rhs, step)

        new_u.data[0, :] = ic
        new_u.data[1, :] = ic
        new_u_p.data[0, :] = ic
        new_u_p.data[1, :] = ic

    nt = int(np.ceil((p.t1 - p.t0)/p.critical_dt))
    dt = p.t1/nt

    # Snapshot the solution
    if snapshots > 0:
        factor = int(np.ceil(nt/(snapshots + 1)))
        time_dim = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
        save = snapshots//step + 1
    else:
        time_dim = new_u.time_dim
        save = 1

    u_save = TimeFunction(
        name='usave', grid=grid,
        time_order=0, space_order=2,
        save=save, time_dim=time_dim
    )
    save = Eq(u_save, new_u)

    op = Operator([*stencil, save], opt='noop')
    kwargs = {'time': nt - 2} if d == 1 else {}
    op(dt=dt, **kwargs)

    if d == 2 and step == 1:
        u_save.data[0, :] = ic

    return u_save.data


def plot_1d(k, data, parameters):
    p = parameters
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    x = np.linspace(p.origin[0], p.extent[0], *p.shape)
    ax.plot(
        x, gaussian_1d(x, mu=p.mu, sigma_sq=p.sigma_sq),
        color='k', ls='--', label='Initial Condition'
    )

    for step, d in zip(k, data, strict=True):
        label = 'Normal timestepping' if step == 1 else f'Superstep size {step}'
        ax.plot(x, d[-1], label=label)

    ax.set_xlim(p.origin[0], p.extent[0])
    ax.set_ylim(-p.lim, p.lim)
    ax.legend()
    return fig, ax


def plot_2d(k, data, parameters, snapshots=1):
    p = parameters
    fig, axes = plt.subplots(len(data), snapshots)
    fig.set_size_inches(16, 5)

    for step, d, ax_row in zip(k, data, axes, strict=True):
        idx = 0
        for ii, ax in enumerate(ax_row):
            if ii % step == 0:
                ax.imshow(
                    d[idx, :, :].T,
                    extent=[p.origin[0], p.extent[0], p.extent[1], p.origin[1]],
                    vmin=-p.lim, vmax=p.lim,
                    cmap='seismic'
                )
                idx += 1
                if step == 1:
                    ax.set_title(f't = {(ii*p.t1)/(snapshots - 1):0.3f}')
                    if ii != 0:
                        ax.set_yticklabels([])
                    if ii % 2 == 1:
                        ax.set_xticklabels([])
            else:
                ax.remove()
    fig.subplots_adjust(
        left=0.05,
        bottom=0.02,
        right=0.99,
        top=0.96,
        wspace=0.23,
        hspace=0.0
    )
    return fig, ax


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dimension', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    d = args.dimension
    parameters = {}
    # 1D Simulation parameters
    parameters[1] = Parameters(
        shape=(501, ),
        origin=(0, ),
        extent=(1, ),
        t0=0,
        t1=0.15,
        critical_dt=0.0014142,
        mu=0.5,
        sigma_sq=0.005,
        lim=np.ceil(1/np.sqrt(2*np.pi*0.005)),
    )
    # 2D Simulation parameters
    parameters[2] = Parameters(
        shape=(101, 101),
        origin=(0., 0.),
        extent=(1000, 1000),  # 1kmx1km
        # Time Domain
        t0=0,
        t1=0.5,
        critical_dt=0.0047140,
        # Initial Condition
        mu=500,
        sigma_sq=5000,
        lim=1/(4*np.pi*5000)
    )

    # Supersteps
    if d == 1:
        k = range(1, 6)
        m = -1
    elif d == 2:
        k = [1, 3, 4]
        # Snapshots
        m = 13

    data = [simulate_ic(parameters[d], step, snapshots=m) for step in k]

    if d == 1:
        fig, ax = plot_1d(k, data, parameters[d])
    elif d == 2:
        fig, ax = plot_2d(k, data, parameters[d], m)

    fig.savefig(f'{d}d_example.png', dpi=300)
    plt.show()
