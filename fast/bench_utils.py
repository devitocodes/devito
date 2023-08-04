from examples.seismic import plot_image

__all__ = ['plot_2dfunc']


def plot_2dfunc(u):
    # Plot a 3D structured grid using pyvista
    plot_image(u.data[0], cmap='seismic')
