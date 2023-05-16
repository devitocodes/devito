
from ast import literal_eval
from math import floor, log
import sys
import matplotlib.pyplot as plt
import numpy as np
import thread_scale

dims = {"2d5pt": 2, "3d_diff": 3}

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(f"usage: {sys.argv[0]} <benchmark> [first]")
    print("plot data starting from the 0-indexed [first] line, defaulting to 0.")
    sys.exit(1)

benchmark = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0


pdf_name = f"devito_{benchmark}_nthreads_cpu.pdf"

labels:list[str] = []
nthreads: set[int] | list[int] = set()
x_label = "Threads"
values:dict[str, list[float]] = {}
for mode in thread_scale.modes:
    csv_name = f"{benchmark}_threads_{mode}.csv"
    try:
        with open(csv_name, "r") as f:
            lines = f.read().split("\n")[:-1]

            labels.append(mode)

            lines = list(map(literal_eval, lines[1+first:]))
            if mode not in values.keys():
                values[mode] = []
            for line in lines:
                nthreads.add(line[0])
                print(f"line {line}")
                # line[1] is the list of runtimes
                # Here I'm just plotting the average!
                values[mode].append(sum(line[1])/len(line[1]))

    except FileNotFoundError:
        print(f'{csv_name} not found! Try running "python grid_scale.py {sys.argv[1]} {mode}" to generate it." ')
        sys.exit(1)
nthreads = list(nthreads)
nthreads.sort()

print(nthreads)
print(labels)
print(values)
x = np.arange(len(nthreads))  #type: ignore
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
ax.set_xticks(x + width, nthreads)
ax.legend(loc="upper left", ncols=3) #type: ignore
fig.autofmt_xdate()

print(f"saving to {pdf_name}")
plt.savefig(pdf_name, format="pdf") #type: ignore
plt.show() #type: ignore