import argparse
import os
import pathlib
import sys
from functools import reduce
from subprocess import PIPE, Popen
from typing import Any

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument("benchmark_name", choices=["2d5pt", "3d_diff"])
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

    parser.add_argument("--mpi", default=False, action="store_true")


    # Local group
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument("--openmp", default=False, action="store_true")
    local_group.add_argument("--gpu", default=False, action="store_true")

    # This main-specific ones
    parser.add_argument("-xdsl", "--xdsl", default=False, action="store_true")

    args = parser.parse_args()
    # TODO: (eventually) register mlir as an devito language or something?
    if args.gpu:
        os.environ["DEVITO_LANGUAGE"] = "openacc"
        os.environ["DEVITO_PLATFORM"] = "nvidiaX"
        os.environ["DEVITO_ARCH"] = "nvc++"
    if args.openmp:
        os.environ["DEVITO_LANGUAGE"] = "openmp"
# Doing this here because Devito does config at import time

from devito import Constant, Eq, Grid, Operator, TimeFunction, XDSLOperator, solve
from devito.logger import info
from devito.operator.profiling import PerfEntry, PerfKey, PerformanceSummary


def get_equation(name: str, shape: tuple[int, ...], so: int, to: int, init_value: int):
    d = (2.0 / (n - 1) for n in shape)
    nu = 0.5
    sigma = 0.25
    dt = sigma * reduce(lambda a, b: a * b, d) / nu
    match name:
        case "2d5pt":
            nx, ny = shape
            # Field initialization
            grid = Grid(shape=shape)
            u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
            u.data[...] = 0
            u.data[..., int(nx / 2), int(ny / 2)] = init_value
            u.data[..., int(nx / 2), -int(ny / 2)] = -init_value

            # Create an equation with second-order derivatives
            a = Constant(name="a")
            eq = Eq(u.dt, a * u.laplace + 0.01)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case "3d_diff":
            nx, _, _ = shape
            grid = Grid(shape=shape, extent=(2.0, 2.0, 2.0))
            u = TimeFunction(name="u", grid=grid, space_order=so)
            # init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
            u.data[...] = 0
            u.data[:, int(nx / 2), ...] = 1

            a = Constant(name="a")
            # Create an equation with second-order derivatives
            eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case other:
            raise Exception(f"Unknown benchamark {other}!")

    return (grid, u, eq0, dt)


def run_operator(op: Operator, nt: int, dt: float) -> float:
    res = op.apply(time_M=nt, a=0.1, dt=dt)
    assert isinstance(res, PerformanceSummary)
    o = res[PerfKey("section0", None)]
    assert isinstance(o, PerfEntry)
    return o.time


def main(bench_name: str, nt: int, dump_main: bool, dump_mlir: bool):
    grid, u, eq0, dt = get_equation(bench_name, args.shape, so, to, init_value)

    if args.xdsl:
        xop = XDSLOperator([eq0])
        rt = run_operator(xop, nt, dt)
        print(f"xDSL finer runtime: {rt} s")
        if args.no_output_dump:
            info("Skipping result data saving.")
        else:
            # get final data step
            # this is cursed math, but we assume that:
            #  1. Every kernel always writes to t1
            #  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
            #  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
            #  4. time_M is always nt in this example
            t1 = (nt + u._time_size - 1) % (2)

            #res_data: np.array = u.data[t1, ...]
            #info("Save result data to " + bench_name + ".stencil.data")
            #res_data.tofile(bench_name + ".stencil.data")

    else:
        op = Operator([eq0])
        rt = run_operator(op, nt, dt)
        print(f"Devito finer runtime: {rt} s")
        if args.no_output_dump:
            info("Skipping result data saving.")
        else:
            # get final data step
            # this is cursed math, but we assume that:
            #  1. Every kernel always writes to t1
            #  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
            #  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
            #  4. time_M is always nt in this example
            t1 = (nt + u._time_size - 1) % (2)

            #res_data: np.array = u.data[t1, ...]
            #info("Save result data to " + bench_name + ".devito.data")
            #res_data.tofile(bench_name + ".devito.data")


if __name__ == "__main__":
    benchmark_dim: int
    match args.benchmark_name:
        case "2d5pt":
            benchmark_dim = 2
        case "3d_diff":
            benchmark_dim = 3
        case _:
            raise Exception("Unhandled benchmark?")

    if len(args.shape) != benchmark_dim:
        print(f"Expected {benchmark_dim}d shape for this benchmark, got {args.shape}")
        sys.exit(1)

    so = args.space_order
    to = args.time_order

    init_value = 10
    main(args.benchmark_name, args.nt, args.dump_main, args.dump_mlir)
