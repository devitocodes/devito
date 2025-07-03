''' Script that demonstrates the functionality of the superstep in 1D
"Wave on a string"
'''
import matplotlib.pyplot as plt
import numpy as np
from devito import Eq, Function, Grid, Operator, TimeFunction, solve
from devito.timestepping.superstep import superstep_generator

# Parameters
# Spatial
shape = (501, )
pad = (0, )
origin = (0, )
extent = (1, )
# Time
t0 = 0
t1 = 0.15
critical_dt = 0.0014142
# Initial Condition
mu = 0.5
sigma_sq = 0.005
ylim = np.ceil(1/np.sqrt(2*np.pi*sigma_sq))
xlim = (0, 1)


def gaussian(x, mu=0, sigma_sq=1):
    ''' Generate a Gaussian initial condition
    '''
    return np.exp(-((x - mu)**2)/(2*sigma_sq))/(np.sqrt(2*np.pi*sigma_sq))


def wave_on_string(step=1):
    grid = Grid(shape=shape, extent=extent)

    velocity = Function(name='velocity', grid=grid, space_order=(0, step, step))
    velocity.data[:] = 1

    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

    pde = (1/velocity**2)*u.dt2 - u.dx2
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Initial condition
    x = np.linspace(0, 1, *shape)
    ic = gaussian(x, mu=mu, sigma_sq=sigma_sq)

    if step == 1:
        # Non-superstep case
        newu = u
        newu.data[0, :] = ic
        newu.data[1, :] = ic
        op = Operator(stencil)
    else:
        # Superstepping
        newu, newu_p, stencil1, stencil2 = superstep_generator(u, stencil.rhs, k=step)

        newu.data[0, :] = ic
        newu.data[1, :] = ic
        newu_p.data[0, :] = ic
        newu_p.data[1, :] = ic

        op = Operator([
            stencil1,
            stencil2,
        ], opt='noop')

    tn = int(np.ceil(t1/critical_dt))
    dt = t1/tn

    op(time=tn, dt=dt)

    idx = tn % 3
    return newu.data[idx]


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 1, *shape)
    ax.plot(
        x, gaussian(x, mu=mu, sigma_sq=sigma_sq),
        color='k', ls='--', label='Initial Condition'
    )

    for ii in range(1, 6):
        label = 'Normal timestepping' if ii == 1 else f'Superstep size {ii}'
        ax.plot(x, wave_on_string(ii), label=label)

    ax.set_xlim(*xlim)
    ax.set_ylim(-ylim, ylim)
    ax.legend()
    plt.show()
