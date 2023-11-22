import argparse
import os
import sys
import json
from functools import reduce
import numpy as np

# This file should not be needed anymore
# To check whether any code would be useful
# flake8: noqa



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

    parser.add_argument(
        "--compare", default=False, action="store_true", help="Compare xdsl/devito output"
    )
    parser.add_argument(
        "--random-init",
        default=False,
        action="store_true",
        help="Initialize data randomly"
    )
    parser.add_argument(
        "--repeat", default=1, type=int, help="Run n times"
    )

    # Local group
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument("--openmp", default=False, action="store_true")
    local_group.add_argument("--gpu", default=False, action="store_true")

    # This main-specific ones
    parser.add_argument("--xdsl", default=False, action="store_true")

    parser.add_argument("--devito", default=False, action="store_true")

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
from devito.operator.profiling import PerfEntry, PerfKey, PerformanceSummary

from mpi4py import MPI


def my_rank(default=None) -> int | None:
    if MPI.Is_initialized():
        return MPI.Comm(MPI.COMM_WORLD).rank
    return default


def initialize_domain(u: TimeFunction, nx: int, ny: int):
    # seed with (reproducable) random noise if requested
    # helps finds bugs faster sometimes
    if args.random_init:
        seed = 123456 + my_rank(0)
        np.random.seed(seed)
        mask = np.random.random(u.data.shape) > 0.8
        u.data[...] = 0
        u.data[mask] = 10
    else:
        u.data[...] = 0
        u.data[..., int(nx / 2), int(ny / 2)] = init_value
        u.data[..., int(nx / 2), -int(ny / 2)] = -init_value


def get_equation(name: str, shape: tuple[int, ...], so: int, to: int, init_value: int):
    d = (2.0 / (n - 1) for n in shape)
    nu = 0.5
    sigma = 0.25
    dt = sigma * reduce(lambda a, b: a * b, d) / nu
    match name:
        case "2d5pt":
            # Field initialization
            grid = Grid(shape=shape)
            u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)

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
    o = res[PerfKey("section0", my_rank())]
    assert isinstance(o, PerfEntry)
    return o.time


def main(bench_name: str, nt: int, runs: int = 1):
    grid, u, eq0, dt = get_equation(bench_name, args.shape, so, to, init_value)
    rank = my_rank()

    for run in range(runs):
        data = []
        if args.xdsl:
            initialize_domain(u, *args.shape)
            xop = XDSLOperator([eq0])
            rt = run_operator(xop, nt, dt)

            print(json.dumps({
                'type': 'runtime',
                'runtime': rt,
                'rank': rank,
                'name': bench_name,
                'impl': 'xdsl',
                'run': run,
            }))

            if args.compare:
                data.append(u.data.copy())

        if args.devito:
            initialize_domain(u, *args.shape)
            op = Operator([eq0])
            rt = run_operator(op, nt, dt)

            print(json.dumps({
                'type': 'runtime',
                'runtime': rt,
                'rank': rank,
                'name': bench_name,
                'impl': 'devito',
                'run': run,
            }))

            if args.compare:
                data.append(u.data.copy())

        if args.compare:
            if len(data) != 2:
                print("cannot compare data, must be run with --xdsl --devito flags to "
                      "run both!")

            compare_data(*data, rank, run)


def compare_data(a: np.ndarray, b: np.ndarray, rank: int, run: int):
    print(json.dumps({
        'rank': rank,
        'type': 'correctness',
        'run': run,
        'mean_squared_error': float(((a - b)**2).mean()),
        'abs_max_val': float(max(np.abs(a).max(), np.abs(b).max())),
        'abs_max_error': float(np.abs(a - b).max()),
    }))


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
    main(args.benchmark_name, args.nt, args.repeat)
