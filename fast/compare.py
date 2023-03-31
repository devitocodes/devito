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

args = parser.parse_args()

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

import pdb;pdb.set_trace()
assert prod(devito_data.shape) == prod(shape)

stencil_data = stencil_data.reshape(tuple(s + 8 for s in shape))
stencil_data = stencil_data[4:-4,4:-4,4:-4]

devito_data = devito_data.reshape(shape)

print("Maximal error {}".format(np.absolute(devito_data - stencil_data).max()))

assert np.isclose(stencil_data, devito_data, rtol=1e-6).all()
