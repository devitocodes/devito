import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser(description="Process arguments.")

parser.add_argument(
    "-d", "--shape", type=int, nargs="+", help="Number of grid points along each axis"
)
parser.add_argument(
    "-so", "--space_order", default=2, type=int, help="Space order of the simulation"
)
parser.add_argument(
    "-to", "--time_order", default=1, type=int, help="Time order of the simulation"
)
parser.add_argument(
    "-nt", "--nt", default=10, type=int, help="Simulation time in millisecond"
)
parser.add_argument("-n", "--name", type=str, help="benchmark name")
parser.add_argument("--mpi", default=False, action="store_true")

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

try:
    assert prod(devito_data.shape) == prod(shape)
except:
    raise AssertionError("Wrong shape specified to the compare script!")

# find halo size:
# this assumes that halo is equal in all directions
ndims = len(shape)
# number of elements that are "too many". We have to divide them equally into the halo
total_elms = stencil_data.shape[0]

for halo in range(0, 20):
    if total_elms <= (prod(shape_elm + halo for shape_elm in shape)):
        break

assert total_elms == prod(
    shape_elm + halo for shape_elm in shape
), "Could not correctly infer halo"

assert halo

nodes = 2
if args.mpi:
    print("Unmangling MPI gathered data")
    # load data and re-order
    stencil = np.zeros(args.shape)
    local_dims = args.shape[0], args.shape[1] // nodes
    for i in range(nodes):
        for i in range(nodes):
            local = stencil_data[(i * prod(local_dims)):((i+1) * prod(local_dims))].reshape(local_dims)
            stencil[:,(i * local_dims[1]):((i+1) * local_dims[1])] = local
    stencil_data = stencil
else:

    # reshape into expanded form
    stencil_data = stencil_data.reshape(tuple(shape_elm + halo for shape_elm in shape))
    # cut off the halo
    if len(shape) == 2:
        stencil_data = stencil_data[(halo // 2) : -(halo // 2), (halo // 2) : -(halo // 2)]
    if len(shape) == 3:
        stencil_data = stencil_data[
            (halo // 2) : -(halo // 2),
            (halo // 2) : -(halo // 2),
            (halo // 2) : -(halo // 2),
        ]


# reshape into normal shape
devito_data = devito_data.reshape(shape)
error_data = devito_data - stencil_data

print("Max error: {}".format(np.absolute(error_data).max()))
print(f"Mean Squred Error: {(error_data**2).mean()}")
abs_max = np.maximum(np.absolute(devito_data), np.absolute(stencil_data)).max()
print("Max abs value: {}".format(abs_max))

devito_norm = np.linalg.norm(devito_data)
stencil_norm = np.linalg.norm(stencil_data)
print(f"Norms (Devito/xDSL) : \n{devito_norm}\n{stencil_norm}")
assert np.isclose(devito_norm, stencil_norm, rtol=1e-6)
assert np.isclose(stencil_data, devito_data, rtol=1e-6).all()
