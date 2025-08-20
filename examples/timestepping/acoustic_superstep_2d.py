""" Script that demonstrates the functionality of the superstep in 2D
Acoustic wave equation with source injection
"""
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from devito import (
    ConditionalDimension,
    Eq,
    Operator,
    SparseTimeFunction,
    TimeFunction,
    solve,
)
from devito.timestepping.superstep import superstep_generator
from examples.seismic import demo_model, SeismicModel


def ricker(t, f=10, A=1):
    """
    The Ricker wavelet
    f - freq in Hz
    A - amplitude
    """
    trm = (np.pi * f * (t - 1 / f)) ** 2
    return A * (1 - 2 * trm) * np.exp(-trm)


def acoustic_model(model, t0, t1, t2, critical_dt, source, step=1, snapshots=1):
    # Construct 2D Grid
    x, y = model.grid.dimensions
    velocity = model.vp
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

    pde = (1/velocity**2)*u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    nt1 = int(np.ceil((t1 - t0)/critical_dt))
    dt = (t1 - t0)/nt1

    # Source
    t = np.linspace(t0, t1, nt1)
    rick = ricker(t)
    source = SparseTimeFunction(
        name="ricker",
        npoint=1,
        coordinates=[source],
        nt=nt1,
        grid=model.grid,
        time_order=2,
        space_order=4
    )
    source.data[:, 0] = rick
    src_term = source.inject(
        field=u.forward,
        expr=source*velocity**2*dt**2
    )

    op1 = Operator([stencil] + src_term)
    op1(time=nt1 - 1, dt=dt)

    # Stencil and operator
    idx = nt1 % 3
    if step == 1:
        # Non-superstep case
        new_u = TimeFunction(
            name="new_u",
            grid=model.grid,
            time_order=2,
            space_order=2
        )
        stencil = [stencil.subs(
            {u.forward: new_u.forward, u: new_u, u.backward: new_u.backward}
        )]
        new_u.data[0, :] = u.data[idx - 2]
        new_u.data[1, :] = u.data[idx - 1]
        new_u.data[2, :] = u.data[idx]
    else:
        new_u, new_u_p, *stencil = superstep_generator(u, stencil.rhs, step, nt=nt1)

    nt2 = int(np.ceil((t2 - t1)/critical_dt))
    dt = (t2 - t1)/nt2

    # Snapshot the solution
    factor = int(np.ceil(nt2/(snapshots + 1)))
    t_sub = ConditionalDimension(
        't_sub',
        parent=model.grid.time_dim,
        factor=factor
    )
    u_save = TimeFunction(
        name='usave',
        grid=model.grid,
        time_order=0,
        space_order=2,
        save=snapshots//step + 1,
        time_dim=t_sub
    )
    save = Eq(u_save, new_u)

    op = Operator([*stencil, save])
    op(dt=dt)

    if step == 1:
        u_save.data[0, :, :] = u.data[idx]

    return u_save.data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='layered', choices=['layered', 'marmousi'])
    args = parser.parse_args()

    t0 = 0
    t1 = 0.2
    if args.model == 'layered':
        source = (500, 20)
        t2 = 0.65
        critical_dt=0.002357
        zlim = 30
    else:  # Marmousi
        # This requires the `devitocodes/data` repository, which we
        # assume to be checked out at `$VIRTUAL_ENV/src/data`.
        source = (1500, 1500)
        t2 = 0.5
        critical_dt=0.0013728
        zlim = 20
        tmp_model = demo_model(
            'marmousi-isotropic',
            space_order=2,
            data_path=f'{os.environ["VIRTUAL_ENV"]}/src/data',
            nbl=0
        )
        cropped = tmp_model.vp.data[400:701, -321:-20]

    # Supersteps
    k = [1, 4]
    # Snapshots
    m = 13
    fig, axes = plt.subplots(len(k), m)

    for step, ax_row in zip(k, axes, strict=True):
        # Redefine the model every iteration because we need to adjust
        # the space order
        if args.model == 'layered':
            model = demo_model(
                'layers-isotropic',
                space_order=(2, step, step),
                nlayers=4,
                vp_top=1500,
                vp_bottom=3000,
                nbl=0
            )
        else:  # Marmousi
            model = SeismicModel(
                space_order=(2, step, step),
                vp=1000*cropped,
                nbl=0,
                origin=(0, 0),
                shape=cropped.shape,
                spacing=(10, 10)
            )

        plot_extent = [
            model.origin[0],
            model.origin[0] + model.grid.extent[0],
            model.origin[1] + model.grid.extent[1],
            model.origin[1]
        ]
        data = acoustic_model(
            model, t0, t1, t2, critical_dt, source, step=step, snapshots=m
        )
        time = np.linspace(t1, t2, (m - 1)//step + 1)
        idx = 0
        for ii, ax in enumerate(ax_row):
            if ii % step == 0:
                ax.imshow(
                    data[idx, :, :].T,
                    extent=plot_extent,
                    vmin=-zlim, vmax=zlim,
                    cmap='seismic'
                )
                ax.imshow(model.vp.data.T, cmap='grey', extent=plot_extent, alpha=0.2)
                ax.set_title(f't={time[idx]:0.3f}')
                idx += 1
                if ii > 0:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                else:
                    xticks = ax.get_xticks()
                    ax.set_xticks(np.array((
                        model.origin[0],
                        model.origin[0] + model.grid.extent[0]
                    )))
                    ax.set_xlim(
                        model.origin[0],
                        model.origin[0] + model.grid.extent[0]
                    )
                    yticks = ax.get_yticks()
                    ax.set_yticks(np.array((
                        model.origin[1],
                        model.origin[1] + model.grid.extent[1]
                    )))
                    ax.set_ylim(
                        model.origin[1] + model.grid.extent[1],
                        model.origin[1]
                    )
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
    fig.savefig(f'{args.model}.png', dpi=300)
