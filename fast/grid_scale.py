dims = {"2d5pt": 2, "3d_diff": 3}

from math import prod
import sys, os

max_size = "2048**3"
if len(sys.argv) < 3 or len(sys.argv) > 4:
    print(f"usage: {sys.argv[0]} <benchmark> <initial size> [maximum size]")
    print(f"First grid size will be (<initial size>)*kernel's dim.")
    print(f"maximum size defaults to {max_size}={eval(max_size)}")
    sys.exit(1)

benchmark = sys.argv[1]
init_size = int(sys.argv[2])
max_size = int(sys.argv[3]) if len(sys.argv) == 4 else eval(max_size)

if benchmark not in dims:
    print(
        f"{benchmark} is not handled! Handled banchmarks are {', '.join(dims.keys())}"
    )
    sys.exit(1)

size = [init_size] * dims[benchmark]
csv_name = f"{benchmark}_grid_runtimes.csv"

def get_runtimes_for_size(size : tuple[int, ...]) -> tuple[tuple[int, ...], float, float]:
    print(f"Running for grid size {size} (total: {prod(size)})")
    cmd = f'make BENCH_OPTS="-d {" ".join(str(s) for s in size)} -nt 100 -to 1" -B {benchmark}.bench MODE=cpu DUMP=0 2>&1'
    try:
        wrap = os.popen(cmd)
        out = wrap.read()
        lines = out.split("\n")
        xdsl_line = next(line for line in lines if line.startswith("Elapsed time is: "))
        devito_line = next(line for line in lines if line.startswith("Operator `Kernel` ran in"))

        pair = (size, float(xdsl_line.split(" ")[-2]), float(devito_line.split(" ")[-2]))
        print(f"Gridsize:  {pair[0]} : xDSL time: {pair[1]}, Devito time: {pair[2]}")
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        raise e
    return pair

runtimes: list [tuple[tuple[int, ...], float, float]] = []
next_mul = len(size) -1

with  open(csv_name, "w") as f:
    f.write("Grid Size,Devito/xDSL,Devito/GCC\n")

    while prod(size) <= max_size:
        runtime = get_runtimes_for_size(tuple(size))
        f.write(f"{','.join(str(r) for r in runtime[0])},{runtime[1]},{runtime[2]}\n")
        size[next_mul] *= 2
        next_mul = (next_mul -1)%len(size)
