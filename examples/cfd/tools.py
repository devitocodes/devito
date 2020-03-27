from mpl_toolkits.mplot3d import Axes3D  # noqa

import numpy as np
from matplotlib import pyplot, cm
from devito.logger import warning


def plot_field(field, xmin=0., xmax=2., ymin=0., ymax=2., zmin=None, zmax=None,
               view=None, linewidth=0):
    """
    Utility plotting routine for 2D data.

    Parameters
    ----------
    field : array_like
        Field data to plot.
    xmax : int, optional
        Length of the x-axis.
    ymax : int, optional
        Length of the y-axis.
    view: int, optional
        View point to intialise.
    """
    if xmin > xmax or ymin > ymax:
        raise ValueError("Dimension min cannot be larger than dimension max.")
    if (zmin is not None and zmax is not None):
        if zmin > zmax:
            raise ValueError("Dimension min cannot be larger than dimension max.")
    elif(zmin is None and zmax is not None):
        if np.min(field) >= zmax:
            warning("zmax is less than field's minima. Figure deceptive.")
    elif(zmin is not None and zmax is None):
        if np.max(field) <= zmin:
            warning("zmin is larger than field's maxima. Figure deceptive.")
    x_coord = np.linspace(xmin, xmax, field.shape[0])
    y_coord = np.linspace(ymin, ymax, field.shape[1])
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')
    ax.plot_surface(X, Y, field[:], cmap=cm.viridis, rstride=1, cstride=1,
                    linewidth=linewidth, antialiased=False)

    # Enforce axis measures and set view if given
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if zmin is None:
        zmin = np.min(field)
    if zmax is None:
        zmax = np.max(field)
    ax.set_zlim(zmin, zmax)

    if view is not None:
        ax.view_init(*view)

    # Label axis
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    pyplot.show()


def init_hat(field, dx, dy, value=2., bgvalue=1.):
    """Set "hat function" initial condition on an array:

    u(.5<=x<=1 && .5<=y<=1 ) is 2

    field : array_like
        Field data to plot.
    dx : float
        Spacing in the x-dimension.
    dy : float
        Spacing in the y-dimension.
    value : float, optional
        Value of the top part of the function. Defaults to 2.
    bgvalue : float, optional
        Background value for the bottom of the function. Defaults to 1.
    """
    field[:] = bgvalue
    field[int(.5 / dx):int(1 / dx + 1), int(.5 / dy):int(1 / dy + 1)] = value


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def fin_bump(x):
    if x <= 0 or x >= 1:
        return 0
    else:
        return 100*np.exp(-1./(x-np.power(x, 2.)))


def init_smooth(field, dx, dy):
    nx, ny = field.shape
    for ix in range(nx):
        for iy in range(ny):
            x = ix * dx
            y = iy * dy
            field[ix, iy] = fin_bump(x/1.5) * fin_bump(y/1.5) + 1.
