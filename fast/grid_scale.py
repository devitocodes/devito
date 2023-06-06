dims = {"2d5pt": 2, "3d_diff": 3}

import argparse
from math import prod

import fast_benchmarks

from devito.operator.operator import Operator
from devito.operator.xdsl_operator import XDSLOperator

parser = argparse.ArgumentParser(description="Process arguments.")

parser.add_argument("benchmark_name", choices=["2d5pt", "3d_diff"])
parser.add_argument("-i", "--init_size", type=int, default=128)
parser.add_argument("-m", "--max_total_size", type=int, default=2048**3)
parser.add_argument("--gpu", default=False, action="store_true")
parser.add_argument("--mpi", default=False, action="store_true")
parser.add_argument("--openmp", default=False, action="store_true")
parser.add_argument(
    "-p",
    "--points",
    type=int,
    default=10,
    help="Number of measurements to make for each number of threads.",
)

args = parser.parse_args()
args.no_output_dump = True

bench_name = args.benchmark_name
init_size = args.init_size
max_size = args.max_total_size
points = args.points
size = tuple([init_size] * dims[bench_name])
csv_name = f"{bench_name}_grid_runtimes.csv"


def get_runtimes_for_size(
    size: tuple[int, ...]
) -> tuple[tuple[int, ...], list[float], list[float]]:
    grid, u, eq0, dt = fast_benchmarks.get_equation(bench_name, size, 2, 1, 10)
    fast_benchmarks.dump_input(u, bench_name)
    xop = XDSLOperator([eq0])
    nt = 100
    fast_benchmarks.compile_main(bench_name, grid, u, xop, dt, nt, args)
    fast_benchmarks.compile_kernel(bench_name, xop.mlircode, args)
    fast_benchmarks.link_kernel(bench_name, args)
    xdsl_runs = [
        fast_benchmarks.run_kernel(bench_name, args.mpi) for _ in range(points)
    ]

    op = Operator([eq0])
    devito_runs = [fast_benchmarks.run_operator(op, nt, dt) for _ in range(points)]

    return (size, xdsl_runs, devito_runs)


next_mul = len(size) - 1

fast_benchmarks.compile_interop(bench_name, args)

with open(csv_name, "w") as f:
    f.write("Grid Size,Devito/xDSL,Devito\n")
    f.flush()

    while prod(size) <= max_size:
        runtime = get_runtimes_for_size(size)
        f.write(f"{runtime}\n")
        f.flush()
        size = list(size)
        size[next_mul] *= 2
        size = tuple(size)
        next_mul = (next_mul - 1) % len(size)
