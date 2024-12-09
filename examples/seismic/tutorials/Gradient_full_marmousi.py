r'''
Memory profiling is performed to compute the memory usage when computing the gradient computation
using the full time axis.

Memory-profiler run as: mprof run Gradient_full_marmousi.py; mprof plot
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy
from memory_profiler import memory_usage
from examples.seismic import plot_velocity, plot_perturbation, Model, AcquisitionGeometry, plot_shotrecord, Receiver, plot_image
from examples.seismic.acoustic import AcousticWaveSolver
from devito import configuration, Eq, Operator, Function, TimeFunction, norm, mmax

configuration['log-level'] = 'WARNING'

nshots = 30  # Number of shots to create gradient from
nreceivers = 300  # Number of receiver locations per shot 

# function to get water layer mask
def mask(model,value):
    """
    Return a mask for the model (m) using the (value)
    """
    mask = model > value
    mask = mask.astype(int)
    # mask[:21] = 0
    return mask

# Define true and initial model
shape = (601, 221)  # Number of grid point (nx, nz)
spacing = (15., 15.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations
vel_path = '../../../devito/data/Marm.bin'
# Load the true model
model_true = (np.fromfile(vel_path, np.float32).reshape(221, 601))
msk = mask(model_true, 1.5)
model_init = gaussian_filter(model_true, sigma=[10, 15])
model_init = model_init * msk
model_init[model_init==0] = 1.5 # km/s

model = Model(vp=model_true.T, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbl=20, bcs="damp")
model0 = Model(vp=model_init.T, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbl=20, bcs="damp", grid = model.grid)

# plot_velocity(model)
# plot_velocity(model0)
# plot_perturbation(model0, model)

assert model.grid == model0.grid
assert model.vp.grid == model0.vp.grid

# Define acquisition geometry: source
t0 = 0.
tn = 4000. 
f0 = 0.005
# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 20.  # Depth is 20m

# Define acquisition geometry: receivers
# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 1] = 20.

# Geometry
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
# We can plot the time signature to see the wavelet
# geometry.src.show()

# Plot acquisition geometry
# plot_velocity(model, source=geometry.src_positions,
#               receiver=geometry.rec_positions[::4, :])

solver = AcousticWaveSolver(model, geometry, space_order=4)
true_d, u0, _ = solver.forward(vp=model.vp)

# Compute initial data with forward operator 
smooth_d, _, _ = solver.forward(vp=model0.vp)

# Plot shot record for true and smooth velocity model and the difference
# plot_shotrecord(true_d.data, model, t0, tn)
# plot_shotrecord(smooth_d.data, model, t0, tn)
# plot_shotrecord(smooth_d.data - true_d.data, model, t0, tn)

# Prepare the varying source locations sources
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = np.linspace(0., model.domain_size[0], num=nshots)
source_locations[:, 1] = 0

# plot_velocity(model, source=source_locations)

# Computes the residual between observed and synthetic data into the residual
def compute_residual(residual, dobs, dsyn):
    if residual.grid.distributor.is_parallel:
        # If we run with MPI, we have to compute the residual via an operator
        # First make sure we can take the difference and that receivers are at the 
        # same position
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(residual.coordinates.data[:], dsyn.coordinates.data)
        # Create a difference operator
        diff_eq = Eq(residual, dsyn.subs({dsyn.dimensions[-1]: residual.dimensions[-1]}) -
                               dobs.subs({dobs.dimensions[-1]: residual.dimensions[-1]}))
        Operator(diff_eq)()
    else:
        # A simple data difference is enough in serial
        residual.data[:] = dsyn.data[:] - dobs.data[:]
    
    return residual

# Create FWI gradient kernel 
def fwi_gradient(vp_in):    
    # Create symbols to hold the gradient
    grad = Function(name="grad", grid=model.grid)
    # Create placeholders for the data residual and data
    residual = Receiver(name='residual', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    d_obs = Receiver(name='d_obs', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    objective = 0.
    for i in range(nshots):
        # Update source location
        geometry.src_positions[0, :] = source_locations[i, :]
        
        # Generate synthetic data from true model
        _, _, _ = solver.forward(vp=model.vp, rec=d_obs)
        
        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp_in, save=True, rec=d_syn)
        
        # Compute gradient from data residual and update objective function 
        compute_residual(residual, d_obs, d_syn)
        
        objective += .5*norm(residual)**2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad)
    
    return objective, grad

# Compute gradient of initial model
ff, update = fwi_gradient(model0.vp)
mem_grad = memory_usage()[0]
print(f"Memory usage at the end of full gradient: {mem_grad} MiB")

# # Plot the FWI gradient
# plot_image(-update.data, vmin=-1e4, vmax=1e4, cmap="jet")

# # Plot the difference between the true and initial model.
# # This is not known in practice as only the initial model is provided.
# plot_image(model0.vp.data - model.vp.data, vmin=-1e-1, vmax=1e-1, cmap="jet")

# # Show what the update does to the model
# alpha = .5 / mmax(update)
# plot_image(model0.vp.data + alpha*update.data, vmin=2.5, vmax=3.0, cmap="jet")