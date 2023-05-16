dims = {"2d5pt": 2, "3d_diff": 3}

import argparse
import os, sys
from devito.operator.operator import Operator
from devito.operator.xdsl_operator import XDSLOperator
from devito.parameters import configuration
from devito.types.equation import Eq
import fast_benchmarks


parser = argparse.ArgumentParser(description="Process arguments.")

parser.add_argument('benchmark_name', choices=["2d5pt", "3d_diff"])
parser.add_argument('-i', '--init_threads', type=int, default=1, help="Initial (lowest) number of threads")
parser.add_argument('-m', '--max_threads', type=int, default=os.cpu_count(), help=f"Maximum number of threads. Defaults to the detected {os.cpu_count()}")
parser.add_argument(
        "-d",
        "--shape",
        default=(4096, 4096),
        type=int,
        nargs="+",
        help="Number of grid points along each axis",
    )
parser.add_argument('-p', '--points', type=int, default=10, help="Number of measurements to make for each number of threads.")

args = parser.parse_args()

size = args.shape
bench_name = args.benchmark_name
init_threads = args.init_threads
max_threads = args.max_threads
points = args.points
csv_name = f"{bench_name}_thread_runtimes.csv"



def get_runtimes_for_threads(threads: int, eq: Eq) -> tuple[int, list[float], list[float]]:
    print(f"Running for {threads} threads")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    grid, u, eq0, dt = fast_benchmarks.get_equation(bench_name, size, 2, 1, 10)
    op = Operator([eq0])
    xdsl_runs = [fast_benchmarks.run_kernel(bench_name) for _ in range(points)]
    devito_runs = [fast_benchmarks.run_operator(op, nt, dt) for _ in range(points)]
    return (threads, xdsl_runs, devito_runs)



os.environ["OMP_PLACES"] = "threads"
configuration['language'] = 'openmp'


threads = init_threads

grid, u, eq0, dt = fast_benchmarks.get_equation(bench_name, size, 2, 1, 10)
fast_benchmarks.dump_input(u, bench_name)
xop = XDSLOperator([eq0])
nt = 100

fast_benchmarks.compile_interop(bench_name, True)
fast_benchmarks.compile_main(bench_name, grid, u, xop, dt, nt)
fast_benchmarks.compile_kernel(bench_name, xop.mlircode, fast_benchmarks.XDSL_CPU_PIPELINE, fast_benchmarks.OPENMP_PIPELINE)
fast_benchmarks.link_kernel(bench_name)




with open(csv_name, "w") as f:
    f.write("Threads,Devito/xDSL,Devito\n")
    f.flush()

    while threads <= max_threads:
        runtime = get_runtimes_for_threads(threads, eq0)
        f.write(f"{runtime}\n")
        f.flush()
        threads *= 2
