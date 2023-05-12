dims = {"2d5pt": 2, "3d_diff": 3}

import multiprocessing as mp
import os
import sys
from math import prod

benchmark = sys.argv[1]
if benchmark not in dims:
    print(
        f"{benchmark} is not handled! Handled banchmarks are {', '.join(dims.keys())}"
    )
    sys.exit(1)

max_size = "2048**3"
if len(sys.argv) < 2 + dims[benchmark] or len(sys.argv) > 4 + dims[benchmark]:
    print(
        f"usage: {sys.argv[0]} <benchmark> <size, e.g 2 2 for 2d>[initial threads] [maximum threads]"
    )
    print(f"Initial thread number defaults to 1.")
    print(f"Max thread number defaults to thread count (here {mp.cpu_count()}).")
    sys.exit(1)

size = tuple(map(lambda s: int(s), sys.argv[2 : 2 + dims[benchmark]]))
init_threads = (
    int(sys.argv[2 + dims[benchmark]]) if len(sys.argv) >= 3 + dims[benchmark] else 1
)
max_threads = (
    int(sys.argv[3 + dims[benchmark]])
    if len(sys.argv) >= 4 + dims[benchmark]
    else mp.cpu_count()
)

print(sys.argv)
print(f"Running with threads from {init_threads} to {max_threads}")

csv_name = f"{benchmark}_thread_runtimes.csv"


def get_runtimes_for_threads(threads: int) -> tuple[int, float, float]:
    print(f"Running for {threads} threads")
    cmd = f'make BENCH_OPTS="-d {" ".join(str(s) for s in size)} -nt 100 -to 1" -B {benchmark}.bench MODE=openmp THREADS={threads} DUMP=0 2>&1'
    out: str
    try:
        wrap = os.popen(cmd)
        out = wrap.read()
        lines = out.split("\n")
        xdsl_line = next(line for line in lines if line.startswith("Elapsed time is: "))
        devito_line = next(
            line for line in lines if line.startswith("Operator `Kernel` ran in")
        )

        pair = (
            threads,
            float(xdsl_line.split(" ")[-2]),
            float(devito_line.split(" ")[-2]),
        )
        print(f"Threads:  {pair[0]} : xDSL time: {pair[1]}, Devito time: {pair[2]}")
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e
    return pair


with open(csv_name, "w") as f:
    f.write("Grid Size,Devito/xDSL,Devito/GCC\n")
    threads = init_threads
    while threads <= max_threads:
        runtime = get_runtimes_for_threads(threads)
        f.write(f"{threads},{runtime[1]},{runtime[2]}\n")
        threads *= 2
