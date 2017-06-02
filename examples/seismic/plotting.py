import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_velocity(model, source=None, receiver=None):
    """
    Plot a two-dimensional velocity field from a seismic :class:`Model`
    object. Optionally also includes point markers for sources and receivers.

    :param model: :class:`Model` object that holds the velocity model.
    :param source: Coordinates of the source point.
    :param receiver: Coordinates of the receiver points.
    """
    fig = plt.figure()
    domain_size = 1.e-3 * (np.asarray(model.shape) - 1) * np.asarray(model.spacing)
    plot = plt.imshow(np.transpose(model.vp), animated=True, cmap=cm.jet,
                      vmin=min(model.vp.reshape(-1)), vmax=max(model.vp.reshape(-1)),
                      extent=[model.origin[0], model.origin[0] + domain_size[0],
                              model.origin[1] + domain_size[1], model.origin[1]])

    if receiver is not None:
        # Plot source points, if provided
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    if source is not None:
        # Plot receiver points, if provided
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    plt.xlabel('X position (km)',  fontsize=20)
    plt.ylabel('Depth (km)',  fontsize=20)
    plt.tick_params(labelsize=20)
    cbar = plt.colorbar(plot)
    cbar.set_label('velocity (km/s)')
    plt.show()

    
def plot_shotrecord(rec, origin, spacing, dimensions, t0, tn):
    """
    Plot a shot record (receiver values over time).

    :param rec: Receiver data with shape (time, points)
    :param origin: Origin of the domain
    :param spacing: Spacing in all dimensions of the domain
    :param dimensions: Number of grid points in each dimension of the domain
    :param t0: Start of time dimension to plot
    :param tn: End of time dimension to plot
    """
    aspect = tn / (dimensions[0] * spacing[0])
    plt.figure()
    plt.imshow(rec, vmin=-1e1, vmax=1e1, cmap=cm.gray, aspect=aspect,
               extent=[origin[0], origin[0] + 1e-3*dimensions[0] * spacing[0],
                       1e-3*tn, t0])
    plt.xlabel('X position (km)',  fontsize=20)
    plt.ylabel('Time (s)',  fontsize=20)
    plt.tick_params(labelsize=20)
    plt.show()
