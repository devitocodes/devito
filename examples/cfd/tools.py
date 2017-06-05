from mpl_toolkits.mplot3d import Axes3D  # noqa

import numpy as np
from matplotlib import pyplot, cm


def plot_field(field, xmax=2., ymax=2., zmax=None, view=None, linewidth=0):
    """Utility plotting routine for 2D data

    :param field: Numpy array with field data to plot
    :param xmax: (Optional) Length of the x-axis
    :param ymax: (Optional) Length of the y-axis
    :param view: (Optional) View point to intialise
    """
    x_coord = np.linspace(0, xmax, field.shape[0])
    y_coord = np.linspace(0, ymax, field.shape[1])
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x_coord, y_coord)
    ax.plot_surface(X, Y, field[:], cmap=cm.viridis, rstride=1, cstride=1,
                    linewidth=linewidth, antialiased=False)

    # Enforce axis measures and set view if given
    ax.set_xlim(0., xmax)
    ax.set_ylim(0., ymax)
    if zmax is not None:
        ax.set_zlim(1., zmax)
    if view is not None:
        ax.view_init(*view)

    # Label axis
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    pyplot.show()


def init_hat(field, dx, dy, value=2., bgvalue=1.):
    """Set "hat function" initial condition on an array:

    u(.5<=x<=1 && .5<=y<=1 ) is 2

    :param field: Numpy array with field data to plot
    :param dx: Spacing in the x-dimension
    :param dy: Spacing in the y-dimension
    :param value: Value of the top part of the function, default=2.
    :param bgvalue: Background value for the bottom of the function, default=1.
    """
    field[:] = bgvalue
    field[int(.5 / dx):int(1 / dx + 1), int(.5 / dy):int(1 / dy + 1)] = value
