import numpy as np

from devito import configuration, Function, norm, mmax, mmin

from examples.seismic import demo_model, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

from inversion_utils import compute_residual, update_with_box

# Turn off logging
configuration['log-level'] = "ERROR"
# Setup
nshots = 9  # Number of shots to create gradient from
nreceivers = 101  # Number of receiver locations per shot
fwi_iterations = 5  # Number of outer FWI iterations

# Define true and initial model
shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('circle-isotropic', vp_circle=3.0, vp_background=2.5,
                   origin=origin, shape=shape, spacing=spacing, nbl=40)

model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
                    origin=origin, shape=shape, spacing=spacing, nbl=40,
                    grid=model.grid)

assert model.grid == model0.grid
assert model.vp.grid == model0.vp.grid

# Acquisition geometry
t0 = 0.
tn = 1000.
f0 = 0.010
# Source at 20m depth and center of x
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, 0] = 20.  # Depth is 20m

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 0] = 980.

# Geometry
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                               t0, tn, f0=f0, src_type='Ricker')

# Wave solver
solver = AcousticWaveSolver(model, geometry, space_order=4)

# Prepare the varying source locations sources
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = 20.
source_locations[:, 1] = np.linspace(0., 1000, num=nshots)


def fwi_gradient(vp_in):
    # Create symbols to hold the gradient
    grad = Function(name="grad", grid=model.grid)
    objective = 0.
    for i in range(nshots):
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
        # Update source location
        solver.geometry.src_positions[0, :] = source_locations[i, :]

        # Generate synthetic data from true model
        solver.forward(vp=model.vp, rec=d_obs)

        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp_in, save=True, rec=d_syn)

        # Compute gradient from data residual and update objective function
        residual = compute_residual(residual, d_obs, d_syn)

        objective += .5*norm(residual)**2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad)

    return objective, grad


# Compute gradient of initial model
ff, update = fwi_gradient(model0.vp)
print(ff, mmin(update), mmax(update))
assert np.isclose(ff, 57010, atol=1e1, rtol=0)
assert np.isclose(mmin(update), -1198, atol=1e1, rtol=0)
assert np.isclose(mmax(update), 3558, atol=1e1, rtol=0)

# Run FWI with gradient descent
history = np.zeros((fwi_iterations, 1))
for i in range(0, fwi_iterations):
    # Compute the functional value and gradient for the current
    # model estimate
    phi, direction = fwi_gradient(model0.vp)

    # Store the history of the functional values
    history[i] = phi

    # Artificial Step length for gradient descent
    # In practice this would be replaced by a Linesearch (Wolfe, ...)
    # that would guarantee functional decrease Phi(m-alpha g) <= epsilon Phi(m)
    # where epsilon is a minimum decrease constant
    alpha = .05 / mmax(direction)

    # Update the model estimate and enforce minimum/maximum values
    update_with_box(model0.vp, alpha, direction)

    # Log the progress made
    print('Objective value is %f at iteration %d' % (phi, i+1))

assert np.isclose(history[-1], 5583, atol=1e1, rtol=0)
