import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_velocity(v, origin, spacing, dimensions):
    fig = plt.figure()
    plt.imshow(np.transpose(v), animated=True, vmin=min(v.reshape(-1)),
               vmax=max(v.reshape(-1)), cmap=cm.jet,
               extent=[origin[0], origin[0] + 1e-3 * (dimensions[0]-1) * spacing[0],
                       origin[1] + 1e-3*(dimensions[1]-1) * spacing[1], origin[1]])
    plt.xlabel('X position (km)',  fontsize=20)
    plt.ylabel('Depth (km)',  fontsize=20)
    plt.tick_params(labelsize=20)
    cbar = plt.colorbar()
    cbar.set_label('velocity (km/s)')
    plt.show()

    
def plot_shotrecord(rec, origin, spacing, dimensions, t0, tn):
    asp = tn / (dimensions[0] * spacing[0])
    plt.figure()
    plt.imshow(rec, vmin=-1e1, vmax=1e1, cmap=cm.gray, aspect=asp,
               extent=[origin[0], origin[0] + 1e-3*dimensions[0] * spacing[0],
                       1e-3*tn, t0])
    plt.xlabel('X position (km)',  fontsize=20)
    plt.ylabel('Time (s)',  fontsize=20)
    plt.tick_params(labelsize=20)
    plt.show()
