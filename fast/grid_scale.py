dims = {"2d5pt": 2, "3d_diff": 3}

from math import floor, log, prod
import sys, os
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print(f"usage: {sys.argv[0]} <benchmark> <initial (1D) size> [maximum (total) size, defaulting to 2048^3]")
    sys.exit(1)

benchmark = sys.argv[1]
init_size = int(sys.argv[2])
max_size = int(sys.argv[3]) if len(sys.argv) == 4 else 2048**3

if benchmark not in dims:
    print(
        f"{benchmark} is not handled! Handled banchmarks are {', '.join(dims.keys())}"
    )
    sys.exit(1)

size = [init_size] * dims[benchmark]

def get_runtimes_for_size(size : tuple[int, ...]) -> tuple[tuple[int, ...], float, float]:
    print(f"Running for grid size {size} (total: {prod(size)})")
    wrap = os.popen(f'OMP_NUM_THREADS=32 OMP_PLACES=threads make BENCH_OPTS="-d {" ".join(str(s) for s in size)} -nt 100 -to 1" -B {benchmark}.bench MODE=cpu 2>&1')
    out = wrap.read()
    lines = out.split("\n")
    xdsl_line = next(line for line in lines if line.startswith("Elapsed time is: "))
    devito_line = next(line for line in lines if line.startswith("Operator `Kernel` ran in"))

    pair = (size, float(xdsl_line.split(" ")[-2]), float(devito_line.split(" ")[-2]))
    print(f"Gridsize:  {pair[0]} : xDSL time: {pair[1]}, Devito time: {pair[2]}")
    return pair

runtimes: list [tuple[tuple[int, ...], float, float]] = []
next_mul = len(size) -1
while prod(size) <= max_size:
    runtimes.append(get_runtimes_for_size(tuple(size)))
    size[next_mul] *= 2
    next_mul = (next_mul -1)%len(size)


csv_name = f"{benchmark}_grid_runtimes.csv"
svg_name = f"{benchmark}_grid_runtimes.svg"

with  open(csv_name, "w") as f:
    f.write("Grid Size,Devito/xDSL,Devito/GCC\n")
    for runtime in runtimes:
        f.write(f"{','.join(str(r) for r in runtime[0])},{runtime[1]},{runtime[2]}\n")


def human_format(number: int):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    if magnitude == 0 :
        return str(number)
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

with open(csv_name, "r") as f:
    lines = f.read().split("\n")[:-1]
    header_line = lines[0].split(",")
    x_label = header_line[0]
    labels = header_line[1:]

    def split_line(line: str):
        split = line.split(",")
        size = tuple(int(s) for s in split[0:dims[benchmark]])
        runtimes = [float(t) for t in split[dims[benchmark]:]]
        return (size, runtimes)
        # runtimes = 
    lines = list(map(split_line, lines[1:]))

    species: list[tuple[int, ...]] = []
    values:dict[str, list[float]] = {}
    for label in labels:
        values[label] = []
    for line in lines:
        species.append(line[0])
        for i, label in enumerate(labels):
            print(f"line {line} i {i}")
            values[label].append(line[1][i])

    x = np.arange(len(species))  #type: ignore
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained") #type: ignore

    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute) #type: ignore
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Time (s)") #type: ignore
    ax.set_xlabel(x_label) # type: ignore
    ax.set_xticks(x + width, species)
    ax.legend(loc="upper left", ncols=3) #type: ignore

    plt.savefig(svg_name, format="svg") #type: ignore
    plt.show() #type: ignore
