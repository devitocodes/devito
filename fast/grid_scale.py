dims = {"2d5pt": 2, "3d_diff": 3}

from ast import literal_eval
from math import floor, log, prod
import sys, os
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} <benchmark> <initial (1D) size> <maximum (total) size>")
    sys.exit(1)

benchmark = sys.argv[1]
init_size = int(sys.argv[2])
max_size = int(sys.argv[3])

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

runtimes = []
next_mul = len(size) -1
while prod(size) <= max_size:
    runtimes.append(get_runtimes_for_size(tuple(size)))
    size[next_mul] *= 2
    next_mul = (next_mul -1)%len(size)


csv_name = f"{benchmark}_grid_runtimes.csv"
svg_name = f"{benchmark}_grid_runtimes.svg"

with  open(csv_name, "w") as f:
    f.write("GridSize\txDSL\tDevito\n")
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
    
    for line in lines[1:]:
        literal_eval(line)

    species = tuple(s for s in [t[0] for t in runtimes])
    species = tuple(",".join(map(human_format, t)) for t in species)
    # species = ('a', 'b', 'c')
    penguin_means = {
        "Devito/xDSL": tuple(t[1] for t in runtimes),
        "Devito/GCC": tuple(t[2] for t in runtimes),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x + width, species)
    ax.legend(loc="upper left", ncols=3)
    # ax.set_ylim(0, 250)

    plt.savefig(svg_name, format="svg")
    plt.show()
