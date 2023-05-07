dims = {"2d5pt": 2, "3d_diff": 3}

from math import floor, log
import sys, os
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <benchmark>")
    sys.exit(1)

benchmark = sys.argv[1]

if benchmark not in dims:
    print(
        f"{benchmark} is not handled! Handled banchmarks are {', '.join(dims.keys())}"
    )
    sys.exit(1)

sizes = list(map(lambda i: 2**i, range(8, 16)))

print(f"Will bench for sizes {sizes}")

def get_runtimes_for_size(size : int) -> tuple[int, float, float]:
    grid_size: int = size ** dims[benchmark]
    print(f"Running for grid size {grid_size}")
    wrap = os.popen(f'OMP_NUM_THREADS=32 OMP_PLACES=threads make BENCH_OPTS="-d {" ".join([str(size)] * dims[benchmark])} -nt 100 -to 1" -B {benchmark}.bench MODE=cpu 2>&1')
    out = wrap.read()
    lines = out.split("\n")
    xdsl_line = next(line for line in lines if line.startswith("Elapsed time is: "))
    devito_line = next(line for line in lines if line.startswith("Operator `Kernel` ran in"))

    pair = (grid_size, float(xdsl_line.split(" ")[-2]), float(devito_line.split(" ")[-2]))
    print(f"Gridsize:  {pair[0]} : xDSL time: {pair[1]}, Devito time: {pair[2]}")
    return pair


runtimes = [get_runtimes_for_size(size) for size in sizes]


csv_name = f"{benchmark}_grid_runtimes.csv"
svg_name = f"{benchmark}_grid_runtimes.svg"

with  open(csv_name, "w") as f:
    f.write("GridSize\txDSL\tDevito\n")
    for runtime in runtimes:
        f.write(f"{runtime[0]},{runtime[1]},{runtime[2]}\n")


def human_format(number: int):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

with open(csv_name, "r") as f:
    lines = f.read().split("\n")[:-1]
    print(list(map(lambda l: l.split(","), lines[1:])))
    runtimes = [
        (int(t[0]), float(t[1]), float(t[2]))
        for t in map(lambda l: l.split(","), lines[1:])
    ]

    species = tuple(human_format(t[0]) for t in runtimes)
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
