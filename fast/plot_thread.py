
from math import floor, log
import sys
import matplotlib.pyplot as plt
import numpy as np

dims = {"2d5pt": 2, "3d_diff": 3}

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(f"usage: {sys.argv[0]} <benchmark> [first]")
    print("plot data starting from the 0-indexed [first] line, defaulting to 0.")
    sys.exit(1)

benchmark = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0

csv_name = f"{benchmark}_thread_runtimes.csv"
svg_name = f"devito_{benchmark}_nthreads_cpu.svg"

def human_format(number: int):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    if magnitude == 0 :
        return str(number)
    return '%.2f%s' % (number / k**magnitude, units[magnitude])
try:
    with open(csv_name, "r") as f:
        lines = f.read().split("\n")[:-1]
        header_line = lines[0].split(",")
        x_label = header_line[0]
        labels = header_line[1:]

        def split_line(line: str):
            split = line.split(",")
            size = int(split[0])
            runtimes = [float(t) for t in split[1:]]
            return (size, runtimes)
            # runtimes = 
        lines = list(map(split_line, lines[1+first:]))

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
        fig.autofmt_xdate()

        print(f"saving to {svg_name}")
        plt.savefig(svg_name, format="svg") #type: ignore
        plt.show() #type: ignore
except FileNotFoundError:
    print(f'{csv_name} not found! Try running "python grid_scale.py {sys.argv[1]}" to generate it." ')
    sys.exit(1)