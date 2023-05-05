dims = {
    "2d5pt" : 2,
    "3d_diff" : 3
}

import sys, os

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <benchmark>")
    sys.exit(1)

benchmark = sys.argv[1]

if benchmark not in dims:
    print(f"{benchmark} is not handled! Handled banchmarks are {', '.join(dims.keys())}")
    sys.exit(1)

sizes = list(map(lambda i : 2**i, range(4, 16)))

print(f"Will bench for sizes {sizes}")

def get_runtimes_for_size(size : int):
    grid_size = size ** dims[benchmark]
    print(f"Running for grid size {grid_size}")
    wrap = os.popen(f'OMP_NUM_THREADS=32 OMP_PLACES=threads make BENCH_OPTS="-d {" ".join([str(size)] * dims[benchmark])} -nt 100 -to 1" -B {benchmark}.bench MODE=cpu 2>&1')
    out = wrap.read()
    lines = out.split("\n")
    xdsl_line = next(line for line in lines if line.startswith("Elapsed time is: "))
    devito_line = next(line for line in lines if line.startswith("Operator `Kernel` ran in"))

    pair = (grid_size, xdsl_line.split(" ")[-2], devito_line.split(" ")[-2])
    print(f"Gridsize:  {pair[0]} : xDSL time: {pair[1]}, Devito time: {pair[2]}")
    return pair


runtimes = [get_runtimes_for_size(size) for size in sizes]