''' Script that demonstrates the functionality of the superstep in 2D
Acoustic wave equation with source injection
'''
import shutil
import urllib.request
from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from devito import (
    ConditionalDimension,
    Eq,
    Function,
    Grid,
    Operator,
    SparseTimeFunction,
    TimeFunction,
    solve,
)
from devito.timestepping.superstep import superstep_generator
from scipy.interpolate import interpn


@dataclass
class model:
    name: str
    velocity: Callable
    # Spatial Domain
    shape: tuple[int]
    origin: tuple[float]
    extent: tuple[float]
    # Souce Location
    source: tuple[float]
    # Time Domain
    t0: float
    t1: float
    t2: float
    critical_dt: float
    # Plotting
    zlim: float


def layered_velocity(grid, step=0):
    velocity = Function(name="layered", grid=grid, space_order=(2, step, step))
    q = grid.shape[1]//4
    velocity.data[:] = 1500
    velocity.data[:, q:2*q] = 2000
    velocity.data[:, 2*q:3*q] = 2500
    velocity.data[:, 3*q:] = 3000
    return velocity


def marmousi(grid, step=0):
    # Grab dataset from pwd or download
    url = 'https://github.com/devitocodes/data/raw/refs/heads/master/Simple2D/vp_marmousi_bi'  # noqa: E501
    filename = Path('marmousi.np')
    shape = (1601, 401)
    if not filename.exists():
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as fh:
            shutil.copyfileobj(response, fh)
    data = np.fromfile(filename, dtype=np.float32, sep='').reshape(*shape)
    cropped = data[650:1051, 35:]

    xs = np.linspace(0, 1, cropped.shape[0])
    ys = np.linspace(0, 1, cropped.shape[1])

    xd = np.linspace(0, 1, grid.shape[0])
    yd = np.linspace(0, 1, grid.shape[1])

    velocity = Function(name="marmousi", grid=grid, space_order=(2, step, step))
    velocity.data[:] = 1000*interpn(
        (xs, ys), cropped, np.meshgrid(xd, yd), method='nearest'
    ).T
    return velocity


def ricker(t, f=10, A=1):
    '''The Ricker wavelet
    f - freq in Hz
    A - amplitude
    '''
    trm = (np.pi * f * (t - 1 / f)) ** 2
    return A * (1 - 2 * trm) * np.exp(-trm)


def acoustic_model(model, step=1, snapshots=1):
    # Construct 2D Grid
    grid = Grid(shape=model.shape, extent=model.extent)
    x, y = grid.dimensions

    t0, t1, t2 = model.t0, model.t1, model.t2

    velocity = model.velocity(grid, step)
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

    pde = (1/velocity**2)*u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    tn1 = int(np.ceil((t1 - t0)/model.critical_dt))
    dt = (t1 - t0)/tn1

    # Source
    t = np.linspace(t0, t1, tn1)
    rick = ricker(t)
    source = SparseTimeFunction(
        name="ricker", npoint=1, coordinates=[model.source], nt=tn1, grid=grid,
        time_order=2, space_order=4
    )
    source.data[:, 0] = rick
    src_term = source.inject(field=u.forward, expr=source*velocity*velocity*dt*dt)

    op1 = Operator([stencil] + src_term)
    op1(time=tn1 - 1, dt=dt)

    # Stencil and operator
    idx = tn1 % 3
    if step == 1:
        # Non-superstep case
        new_u = TimeFunction(name="new_u", grid=grid, time_order=2, space_order=2)
        stencil = [stencil.subs(
            {u.forward: new_u.forward, u: new_u, u.backward: new_u.backward}
        )]
        new_u.data[0, :] = u.data[idx - 2]
        new_u.data[1, :] = u.data[idx - 1]
        new_u.data[2, :] = u.data[idx]
    else:
        new_u, new_u_p, *stencil = superstep_generator(u, stencil.rhs, step)

        new_u.data[0, :] = u.data[idx - 1]
        new_u.data[1, :] = u.data[idx]
        new_u_p.data[0, :] = u.data[idx - 2]
        new_u_p.data[1, :] = u.data[idx - 1]

    tn2 = int(np.ceil((t2 - t1)/model.critical_dt))
    dt = (t2 - t1)/tn2

    # Snapshot the solution
    factor = int(np.ceil(tn2/(snapshots + 1)))
    t_sub = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
    u_save = TimeFunction(
        name='usave', grid=grid,
        time_order=0, space_order=2,
        save=snapshots//step + 1, time_dim=t_sub
    )
    save = Eq(u_save, new_u)

    op = Operator([*stencil, save])
    op(dt=dt)

    if step == 1:
        u_save.data[0, :, :] = u.data[idx]

    return u_save.data


layered_model = model(
    name='layered',
    velocity=layered_velocity,
    shape=(101, 101),
    origin=(0., 0.),
    extent=(1000, 1000),  # 1kmx1km
    source=(500, 20),
    t0=0,
    t1=0.2,
    t2=0.65,
    critical_dt=0.002357,
    zlim=30
)

marmousi_model = model(
    name='marmousi',
    velocity=marmousi,
    shape=(274, 301),
    origin=(4875., 262.5),
    extent=(3000, 2737.5),  # 3kmx2.7km
    source=(1000, 1200),
    t0=0,
    t1=0.2,
    t2=0.5,
    critical_dt=0.0013728,
    zlim=20
)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='layered', choices=['layered', 'marmousi'])
    args = parser.parse_args()

    model = layered_model if args.model == 'layered' else marmousi_model

    # Supersteps
    k = [1, 4]
    # Snapshots
    m = 13
    fig, axes = plt.subplots(len(k), m)

    # Velocity model
    grid = Grid(shape=model.shape, extent=model.extent)
    v = model.velocity(grid)
    plot_extent = [
        model.origin[0],
        model.origin[0] + model.extent[0],
        model.origin[1] + model.extent[1],
        model.origin[1]
    ]

    for step, ax_row in zip(k, axes, strict=True):
        data = acoustic_model(model, step=step, snapshots=m)
        time = np.linspace(model.t1, model.t2, (m - 1)//step + 1)
        idx = 0
        for ii, ax in enumerate(ax_row):
            if ii % step == 0:
                ax.imshow(
                    data[idx, :, :].T,
                    extent=plot_extent,
                    vmin=-model.zlim, vmax=model.zlim,
                    cmap='seismic'
                )
                ax.imshow(v.data.T, cmap='grey', extent=plot_extent, alpha=0.2)
                ax.set_title(f't = {time[idx]:0.3f}')
                idx += 1
                if ii > 0:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                else:
                    xticks = ax.get_xticks()
                    ax.set_xticks(np.array((
                        model.origin[0],
                        round(model.origin[0] + model.extent[0]/2, -3),
                        model.origin[0] + model.extent[0]
                    )))
                    ax.set_xlim(model.origin[0], model.origin[0] + model.extent[0])
                    yticks = ax.get_yticks()
                    ax.set_yticks(np.concat(((model.origin[1],), yticks[2::2])))
                    ax.set_ylim(model.origin[1] + model.extent[1], model.origin[1])
            else:
                ax.remove()

    fig.set_size_inches(16, 3.5)
    fig.subplots_adjust(
        left=0.05,
        bottom=0.025,
        right=0.99,
        top=0.97,
        wspace=0.06,
        hspace=0.06
    )
    fig.savefig(f'{model.name}.png', dpi=300)
