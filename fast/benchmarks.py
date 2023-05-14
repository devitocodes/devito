import argparse
from functools import reduce
from math import prod
from typing import Any

import numpy as np

from devito import Constant, Eq, Grid, Operator, TimeFunction, XDSLOperator, solve
from devito.ir.ietxdsl.cluster_to_ssa import generate_launcher_base
from devito.logger import info
from devito.operator.profiling import PerfEntry, PerfKey, PerformanceSummary
import sys


def setup_benchmark(name:str, shape:tuple[int, ...], so: int, to: int, init_value: int):
    match name:
        case '2d5pt':
            nt = args.nt
            nx, ny = shape
            nu = 0.5
            dx = 2.0 / (nx - 1)
            dy = 2.0 / (ny - 1)
            sigma = 0.25
            dt = sigma * dx * dy / nu
            # Field initialization
            grid = Grid(shape=(nx, ny))
            u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
            u.data[:, :, :] = 0
            u.data[:, int(nx / 2), int(nx / 2)] = init_value
            u.data[:, int(nx / 2), -int(nx / 2)] = -init_value

            # Create an equation with second-order derivatives
            a = Constant(name="a")
            eq = Eq(u.dt, a * u.laplace + 0.01)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case '3d_diff':
            nx, ny, nz = shape
            nu = .5
            dx = 2. / (nx - 1)
            dy = 2. / (ny - 1)
            sigma = .25
            grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
            u = TimeFunction(name='u', grid=grid, space_order=so)
            # init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
            u.data[:, :, :, :] = 0
            u.data[:, int(nx/2), :, :] = 1

            a = Constant(name='a')
            # Create an equation with second-order derivatives
            eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case _:
            raise Exception("Unknown benchamark!")

    return (grid, u, eq0)


def dump_input(input: TimeFunction, filename: str):
    input.data_with_halo[0, :, :].tofile(filename)


def dump_main(bench_name: str, grid: Grid, u: TimeFunction, xop: XDSLOperator, dt: float, nt: int):
    info("Operator in " + bench_name + ".main.mlir")
    with open(bench_name + ".main.mlir", "w") as f:
        f.write(
            generate_launcher_base(
                xop._module,
                {
                    "time_m": 0,
                    "time_M": nt,
                    **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
                    "a": 0.1,
                    "dt": dt,
                },
                u.shape_allocated[1:],
            )
        )


def run_operator(op: Operator, nt: int, dt: float):
    res = op.apply(time_M=nt, a=0.1, dt=dt)
    assert isinstance(res, PerformanceSummary)
    o = res[PerfKey("section0", None)]
    assert isinstance(o, PerfEntry)
    return o.time


def main(bench_name: str, nt:int, dt:Any):

    grid, u, eq0 = setup_benchmark(bench_name, args.shape, so, to, init_value)

    if args.xdsl:
        dump_input(u, bench_name + ".input.data")
        xop = XDSLOperator([eq0])
        dump_main(bench_name, grid, u, xop, dt, nt)
        info("Dump mlir code in  in " + bench_name + ".mlir")
        with open(bench_name + ".mlir", "w") as f:
            f.write(xop.mlircode)
    else:
        op = Operator([eq0])
        rt = run_operator(op, nt, dt)
        print(f"Devito finer runtime: {rt} s")
        if args.no_dump:
            info("Skipping result data saving.")
        else:
            # get final data step
            # this is cursed math, but we assume that:
            #  1. Every kernel always writes to t1
            #  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
            #  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
            #  4. time_M is always nt in this example
            t1 = (nt + u._time_size - 1) % (2)

            res_data: np.array = u.data[t1]
            info("Save result data to " + bench_name + ".devito.data")
            res_data.tofile(bench_name + ".devito.data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument('benchmark_name', choices=["2d5pt", "3d_diff"])
    parser.add_argument(
        "-d",
        "--shape",
        default=(11, 11),
        type=int,
        nargs="+",
        help="Number of grid points along each axis",
    )
    parser.add_argument(
        "-so",
        "--space_order",
        default=2,
        type=int,
        help="Space order of the simulation",
    )
    parser.add_argument(
        "-to", "--time_order", default=1, type=int, help="Time order of the simulation"
    )
    parser.add_argument(
        "-nt", "--nt", default=10, type=int, help="Simulation time in millisecond"
    )
    parser.add_argument(
        "-bls", "--blevels", default=2, type=int, nargs="+", help="Block levels"
    )
    parser.add_argument("-xdsl", "--xdsl", default=False, action="store_true")
    parser.add_argument("-nd", "--no_dump", default=False, action="store_true")
    args = parser.parse_args()

    benchmark_dim: int
    match args.benchmark_name:
        case '2d5pt':
            benchmark_dim = 2
        case '3d_diff':
            benchmark_dim = 3
    
    if len(args.shape) != benchmark_dim:
        print(f"Expected {benchmark_dim}d shape for this benchmark, got {args.shape}")
        sys.exit(1)

    d = (2.0 / (n - 1) for n in args.shape)
    nu = .5
    sigma = .25
    dt = sigma * reduce(lambda a,b: a*b, d) / nu
    

    so = args.space_order
    to = args.time_order

    init_value = 10
    main(args.benchmark_name,
         args.nt,
         dt)
