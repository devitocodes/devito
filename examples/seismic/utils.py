import numpy as np


__all__ = ['smooth10']


# Velocity models
def smooth10(vel, shape):
    """
    Smooth the input n-dimensional array 'vel' along its last dimension (depth)
    with a 10 points moving averaging kernel.
    """
    # Return a scaled version of the input if the input is a scalar
    if np.isscalar(vel):
        return .9 * vel * np.ones(shape, dtype=np.float32)
    # Initialize output
    out = np.copy(vel)
    # Size of the smoothed dimension
    nz = shape[-1]
    # Indexing is done via slices for YASK compatibility
    # Fist get the full span
    full_dims = [slice(0, d) for d in shape[:-1]]
    for a in range(5, nz-6):
        # Get the a-5 yto ai+5 indices along the last dimension at index a
        slicessum = full_dims + [slice(a - 5, a + 5)]
        # Average input
        out[..., a] = np.sum(vel[slicessum], axis=len(shape)-1) / 10

    return out
