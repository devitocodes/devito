dims = {"2d5pt": 2, "3d_diff": 3}

import argparse
import os

parser = argparse.ArgumentParser(description="Process arguments.")

# Generic args
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
parser.add_argument(
    "-bls", "--blevels", default=2, type=int, nargs="+", help="Block levels"
)
parser.add_argument("--dump_mlir", default=False, action="store_true")
parser.add_argument("--dump_main", default=False, action="store_true")
parser.add_argument("--mpi", default=False, action="store_true")
# Script-specific args
parser.add_argument(
    "-i",
    "--init_threads",
    type=int,
    default=1,
    help="Initial (lowest) number of threads",
)
parser.add_argument(
    "-m",
    "--max_threads",
    type=int,
    default=os.cpu_count(),
    help=f"Maximum number of threads. Defaults to the detected {os.cpu_count()}",
)
parser.add_argument(
    "-p",
    "--points",
    type=int,
    default=10,
    help="Number of measurements to make for each number of threads.",
)
parser.add_argument("benchmark_mode", choices=["xdsl", "devito"])
args = parser.parse_args()
args.no_output_dump = True

args.openmp = True
args.gpu = False
os.environ["DEVITO_LANGUAGE"] = "openmp"

import fast_benchmarks

from devito.operator.operator import Operator
from devito.operator.xdsl_operator import XDSLOperator
from devito.parameters import configuration
from devito.types.equation import Eq

if __name__ == "__main__":
    size = args.shape
    bench_name = args.benchmark_name
    init_threads = args.init_threads
    max_threads = args.max_threads
    points = args.points
    mode = args.benchmark_mode
    csv_name = f"{bench_name}_threads_{mode}.csv"

    def get_runtimes_for_threads(threads: int, eq: Eq) -> tuple[int, list[float]]:
        print(f"Running for {threads} threads")
        os.environ["OMP_NUM_THREADS"] = str(threads)
        match mode:
            case "xdsl":
                runs = [
                    fast_benchmarks.run_kernel(bench_name, args.mpi)
                    for _ in range(points)
                ]
            case "devito":
                grid, u, eq0, dt = fast_benchmarks.get_equation(
                    bench_name, size, 2, 1, 10
                )
                op = Operator([eq0])
                runs = [
                    fast_benchmarks.run_operator(op, args.nt, dt) for _ in range(points)
                ]
            case _:
                raise Exception("Unknown mode!")
        return (threads, runs)

    os.environ["OMP_PLACES"] = "threads"
    configuration["language"] = "openmp"

    threads = init_threads

    grid, u, eq0, dt = fast_benchmarks.get_equation(bench_name, size, 2, 1, 10)
    fast_benchmarks.dump_input(u, bench_name)
    xop = XDSLOperator([eq0])
    nt = 100

    fast_benchmarks.compile_interop(bench_name, args)
    fast_benchmarks.compile_main(bench_name, grid, u, xop, dt, args)
    fast_benchmarks.compile_kernel(bench_name, xop.mlircode, args)
    fast_benchmarks.link_kernel(bench_name, args)

    label: str
    match mode:
        case "xdsl":
            label = "Devito/xDSL"
        case "devito":
            label = "Devito"
        case _:
            raise Exception("Unknown mode!")

    with open(csv_name, "w") as f:
        f.write(f"Threads,{label}\n")
        f.flush()

        while threads <= max_threads:
            runtime = get_runtimes_for_threads(threads, eq0)
            f.write(f"{runtime}\n")
            f.flush()
            threads *= 2
