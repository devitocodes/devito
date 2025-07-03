''' Script that demonstrates the functionality of the superstep in 2D
"Ripple on a pond"
'''
import matplotlib.pyplot as plt
import numpy as np
from devito import ConditionalDimension, Eq, Function, Grid, Operator, TimeFunction, solve
from devito.timestepping.superstep import superstep_generator


def gaussian2d(xx, yy, mu=0, sigma_sq=1):
    return np.exp(-((xx - mu)**2 + (yy - mu)**2)/(2*sigma_sq))/(np.sqrt(2*np.pi*sigma_sq))


# Spatial Domain
shape = (101, 101)
origin = (0., 0.)
extent = (1000, 1000)  # 1kmx1km
# Time Domain
t0 = 0
t1 = 0.5
critical_dt = 0.0047140
# Initial Condition
mu = 500
sigma_sq = 5000
zlim = 1/(2*np.sqrt(2*np.pi*sigma_sq))


def ripple_on_pond(step=1, snapshots=1):
    # Construct 2D Grid
    grid = Grid(shape=shape, extent=extent)
    x, y = grid.dimensions

    # Need to ensure that the velocity function supports the largest superstep stencil
    velocity = Function(name="velocity", grid=grid, space_order=(2, step, step))
    velocity.data[:] = 1500

    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

    pde = (1/velocity**2)*u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Initial condition
    x = np.linspace(origin[0], extent[0], shape[0])
    y = np.linspace(origin[1], extent[1], shape[1])
    xx, yy = np.meshgrid(x, y)
    ic = gaussian2d(xx, yy, mu=mu, sigma_sq=sigma_sq)

    # Stencil and operator
    if step == 1:
        # Non-superstep case
        new_u = u
        stencil = [stencil]
        new_u.data[0, :] = ic
        new_u.data[1, :] = ic
        new_u.data[2, :] = ic
    else:
        new_u, new_u_p, *stencil = superstep_generator(u, stencil.rhs, step)

        new_u.data[0, :] = ic
        new_u.data[1, :] = ic
        new_u_p.data[0, :] = ic
        new_u_p.data[1, :] = ic

    tn = int(np.ceil((t1 - t0)/critical_dt))
    dt = t1/tn

    # Snapshot the solution
    factor = int(np.ceil(tn/(snapshots + 1)))
    t_sub = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
    u_save = TimeFunction(
        name='usave', grid=grid,
        time_order=0, space_order=2,
        save=snapshots//step + 1, time_dim=t_sub
    )
    save = Eq(u_save, new_u)

    op = Operator([*stencil, save], opt='noop')
    op(dt=dt)

    if step == 1:
        u_save.data[0, :] = ic

    return u_save.data


if __name__ == '__main__':
    # Supersteps
    k = [1, 3, 4]
    # Snapshots
    m = 13
    fig, axes = plt.subplots(len(k), m)
    for step, ax_row in zip(k, axes, strict=True):
        data = ripple_on_pond(step=step, snapshots=m)
        idx = 0
        for ii, ax in enumerate(ax_row):
            if ii % step == 0:
                ax.imshow(
                    data[idx, :, :].T,
                    extent=[origin[0], extent[0], extent[1], origin[1]],
                    vmin=-zlim, vmax=zlim,
                    cmap='seismic'
                )
                idx += 1
            else:
                ax.remove()
    fig.subplots_adjust(
        left=0.05,
        bottom=0.025,
        right=0.99,
        top=0.97,
        wspace=0.06,
        hspace=0.06
    )
    plt.show()
