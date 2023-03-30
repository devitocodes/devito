import numpy as np

def prod(iter):
    carry = 1
    for x in iter:
        carry *= x
    return carry

dtype = np.float32

shape = (200, 200, 200)

devito_file = "devito.data"
stencil_file = "stencil.data"

devito_data = np.fromfile(devito_file, dtype=dtype)
stencil_data = np.fromfile(stencil_file, dtype=dtype)

assert prod(devito_data.shape) == prod(shape)

stencil_data = stencil_data.reshape(tuple(s + 8 for s in shape))
stencil_data = stencil_data[4:-4,4:-4,4:-4]

devito_data = devito_data.reshape(shape)

print("Maximal error {}".format(np.absolute(devito_data - stencil_data).max()))

assert np.isclose(stencil_data, devito_data, rtol=1e-6).all()
