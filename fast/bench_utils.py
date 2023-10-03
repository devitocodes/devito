import matplotlib.pyplot as plt
import numpy as np
from examples.seismic import plot_image

__all__ = ['plot_2dfunc', 'plot_3dfunc']


def plot_2dfunc(u):
    # Plot a 2D image using devito's machinery
    plot_image(u.data[0], cmap='seismic')
    plot_image(u.data[1], cmap='seismic')


def plot_3dfunc(u):
    # Plot a 3D structured grid using pyvista
    import pyvista as pv
    cmap = plt.colormaps["viridis"]
    values = u.data[0, :, :, :]
    vistagrid = pv.ImageData()
    vistagrid.dimensions = np.array(values.shape) + 1
    vistagrid.spacing = (1, 1, 1)
    vistagrid.origin = (0, 0, 0)  # The bottom left corner of the data set
    vistagrid.cell_data["values"] = values.flatten(order="F")
    vistaslices = vistagrid.slice_orthogonal()
    vistaslices.plot(cmap=cmap)
