import numpy as np
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rc('font', size=16)
    mpl.rc('figure', figsize=(8, 6))
except:
    plt = None
    cm = None


def plot_perturbation(model, model1, colorbar=True):
    """
    Plot a two-dimensional velocity difference from two seismic :class:`Model`
    objects.

    :param model: :class:`Model` object of first velocity model.
    :param model1: :class:`Model` object of the second velocity model.
    :param source: Coordinates of the source point.
    :param receiver: Coordinates of the receiver points.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    dv = np.transpose(model.vp) - np.transpose(model1.vp)

    plot = plt.imshow(dv, animated=True, cmap=cm.jet,
                      vmin=min(dv.reshape(-1)), vmax=max(dv.reshape(-1)),
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity perturbation (km/s)')
    plt.show()


def plot_velocity(model, source=None, receiver=None, colorbar=True):
    """
    Plot a two-dimensional velocity field from a seismic :class:`Model`
    object. Optionally also includes point markers for sources and receivers.

    :param model: :class:`Model` object that holds the velocity model.
    :param source: Coordinates of the source point.
    :param receiver: Coordinates of the receiver points.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    plot = plt.imshow(np.transpose(model.vp), animated=True, cmap=cm.jet,
                      vmin=np.min(model.vp), vmax=np.max(model.vp), extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')
    plt.show()


def plot_shotrecord(rec, model, t0, tn, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    :param rec: Receiver data with shape (time, points)
    :param model: :class:`Model` object that holds the velocity model.
    :param t0: Start of time dimension to plot
    :param tn: End of time dimension to plot
    """
    scale = np.max(rec) / 10.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def plot_image(data, vmin=None, vmax=None, colorbar=True, cmap="gray"):
    """
    Plot image data, such as RTM images or FWI gradients.

    :param data: Image data to plot
    :param cmap: Choice of colormap, default is gray scale for images as a
    seismic convention
    """
    plot = plt.imshow(np.transpose(data),
                      vmin=vmin or 0.9 * np.min(data),
                      vmax=vmax or 1.1 * np.max(data),
                      cmap=cmap)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()
