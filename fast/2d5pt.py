import numpy as np
import sys

from devito import (Grid, Eq, TimeFunction, Operator, norm,
                    Constant, solve, XDSLOperator)

from devito.ir.ietxdsl.cluster_to_ssa import generate_launcher_base
from devito.logger import info, perf

import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=10,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
parser.add_argument("-xdsl", "--xdsl", default=False, action='store_true')
args = parser.parse_args()

BENCH_NAME = __file__.split('.')[0]

nx, ny = args.shape
nt = args.nt
nu = .5
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

so = args.space_order
to = args.time_order

# Initialise u
init_value = 10

# Field initialization
grid = Grid(shape=(nx, ny))
u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
u.data[:, :, :] = 0
u.data[:, int(nx/2), int(nx/2)] = init_value
u.data[:, int(nx/2), -int(nx/2)] = -init_value

u.data_with_halo[0, :, :].tofile(BENCH_NAME + '.input.data')

# Create an equation with second-order derivatives
a = Constant(name='a')
eq = Eq(u.dt, a*u.laplace + 0.01)
stencil = solve(eq, u.forward)
eq0 = Eq(u.forward, stencil)

xop = XDSLOperator([eq0])

if args.xdsl:
    info("Operator in " + BENCH_NAME + ".main.mlir")
    with open(BENCH_NAME + ".main.mlir", "w") as f:
        f.write(generate_launcher_base(xop._module, {
            'time_m': 0,
            'time_M': nt,
            **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
            'a': 0.1,
            'dt': dt,
        }, u.shape_allocated[1:]))

    info("Dump mlir code in  in " + BENCH_NAME + ".mlir")
    with open(BENCH_NAME + ".mlir", "w") as f:
        f.write(xop.mlircode)

    sys.exit(0)

op = Operator([eq0])
op.apply(time_M=nt, a=0.1, dt=dt)
orig_data: np.array = u.data_with_halo.copy()
orig_norm = norm(u)

# get final data step
# this is cursed math, but we assume that:
#  1. Every kernel always writes to t1
#  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
#  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
#  4. time_M is always nt in this example
t1 = (nt + u._time_size - 1) % (2)

res_data: np.array = u.data[t1, :, :]

info("Save result data to " + BENCH_NAME + ".devito.data")
res_data.tofile(BENCH_NAME + '.devito.data')
