import numpy as np

import sys
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=10,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-n", "--name", type=str, help="benchmark name")

args, unknown = parser.parse_known_args()

bench_name = args.name

def prod(iter):
    carry = 1
    for x in iter:
        carry *= x
    return carry

dtype = np.float32

shape = args.shape

devito_file = bench_name + ".devito.data"
stencil_file = bench_name + ".stencil.data"

devito_data = np.fromfile(devito_file, dtype=dtype)
stencil_data = np.fromfile(stencil_file, dtype=dtype)

assert prod(devito_data.shape) == prod(shape), "Wrong shape specified to the compare script!"

# find halo size:
# this assumes that halo is equal in all directions
ndims = len(shape)
# number of elements that are "too many". We have to divide them equally into the halo
total_elms = stencil_data.shape[0]
for halo in range(0, 20, 2):
    if total_elms >= prod(shape_elm + halo for shape_elm in shape):
        break

assert total_elms == prod(shape_elm + halo for shape_elm in shape), "Could not correctly infer halo"

assert halo 

# reshape into expanded form
stencil_data = stencil_data.reshape(tuple(shape_elm + halo for shape_elm in shape))
# cut off the halo
stencil_data = stencil_data[(halo//2):-(halo//2),(halo//2):-(halo//2),(halo//2):-(halo//2)]
# reshape into normal shape
devito_data = devito_data.reshape(shape)

print("Max error: {}".format(np.absolute(devito_data - stencil_data).max()))
print("Max value: {}", np.maximum(devito_data, stencil_data).max())

assert np.isclose(stencil_data, devito_data, rtol=1e-6).all()
