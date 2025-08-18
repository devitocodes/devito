# Benchmarking a Devito Operator

## Running benchmarks

`benchmark.py` implements a minimalist framework to evaluate the performance of
a Devito Operator while varying:

* the problem size (e.g., shape of the computational grid);
* the discretization (e.g., space- and time-order of the input/output fields);
* the simulation time (in milliseconds);
* the performance optimization level;
* the autotuning level.

Running `python benchmark.py --help` will display a list of useful options.

## A step back: configuring your machine for reliable benchmarking

If you are tempted to use your laptop to run a benchmark, you may want to
reconsider: heat and power management may affect the results you get in an
unpredictable way.

It is important that *both* the Python process running Devito (process*es* if
running with MPI) and the OpenMP threads spawned while running an Operator are
pinned to specific CPU cores, to get reliable and deterministic results. There
are several ways to achieve this:

* Through environment variables. All MPI/OpenMP distributions provide a set of
  environment variables to control process/thread pinning.
* Through a program such as `numactl` or `taskset`.

If running on a NUMA system, where multiple nodes of CPU cores ("sockets") and
memory are available, pinning becomes even more important (and therefore
deserves more attention). On a NUMA system, a core can access both local and
remote memory nodes, but the latency is obviously smaller in the former case.
Thus, if a process/thread is pinned to a given core, it is important that as
much accessed data as possible is allocated in local memory (ideally, the entire
working set). There are multiple scenarios that are worth considering:

* Purely sequential run (no OpenMP, no MPI). Use `numactl` or `taskset` to pin
  the Python process. On a NUMA system, this also ensures that all data gets
  allocated in local memory.
* OpenMP-only run. When encountering a parallel region, an Operator will spawn a
  certain number of OpenMP threads. By default, as many threads as available
  logical cores are created. This can be changed by setting the OpenMP-standard
  `OMP_NUM_THREADS` environment variable to a different value. When might we
  want to do this?
  - Unless on a hyperthreads-centered system, such as an Intel Knights Landing,
    spawning only as many threads as *physical* cores usually results in
    slightly better performance due to less contention for hardware resources.
  - Since, here, we are merely interested in benchmarking, when running on a
    NUMA system we should restrain an Operator to run on a single node of CPU
    cores (i.e., a single socket), as in practice one should always use MPI
    across multiple sockets (hence resorting to the MPI+OpenMP mode). There is
    also one caveat: we need to make sure that the Python process runs on the
    same socket as the OpenMP threads, otherwise some data might get allocated
    on remote memory. For this, `numactl` or `taskset` are our friends.
* MPI-only run. Process pinning should be implemented exploiting the proper MPI
  environment variables.
* MPI+OpenMP. The typical execution mode is: one MPI process per socket, and
  each MPI process spawns a set of OpenMP threads upon entering an OpenMP
  parallel region.  Pinning is typically enforced via environment variables.

Some more information about pinning is available
[here](https://hpc-wiki.info/hpc/Binding/Pinning).

There are many ways one can check that pinning is working as expected. A
recommended tool for rapid visual inspection is [htop](http://hisham.hm/htop/).

## Enabling OpenMP

To switch on multi-threaded execution of Operators via OpenMP, the following
environment variable must be set:
```
DEVITO_LANGUAGE=openmp
```
One has two options: either set it explicitly or prepend it to the Python
command. In the former case, assuming a bash shell:
```bash
export DEVITO_LANGUAGE=openmp
```
In the latter case:
```bash
DEVITO_LANGUAGE=openmp python benchmark.py ...
```

## Enabling MPI

To switch on MPI, one should set
```bash
DEVITO_MPI=1
```
and run with `mpirun -n number_of_processes python benchmark.py ...`

Devito supports multiple MPI schemes for halo exchange.

* Devito's most prevalent MPI modes are three: `basic`, `diag2` and `full`.
and are respectively activated e.g., via `DEVITO_MPI=basic`.
These modes may perform better under different factors such as arithmetic intensity,
or number of fields used in the computation.

## The optimization level

`benchmark.py` allows to set optimization mode, as well as several optimization
options, via the `--opt` argument. Please refer to
[this](https://github.com/devitocodes/devito/blob/main/examples/performance/00_overview.ipynb)
notebook for a comprehensive list of all optimization modes and options
available in Devito. You may also want to take a look at the example command
lines a few sections below.

## Auto-tuning

Auto-tuning can significantly improve the run-time performance of an Operator. It
can be enabled on an Operator basis:
```python
op = Operator(...)
op.apply(autotune=True)
```
The auto-tuner will discover a suitable block shape for each blocked loop nest
in the generated code.

With `autotune=True`, the auto-tuner operates in `basic` mode, which only attempts
a small batch of block shapes. With `autotune='aggressive'`, the auto-tuning phase
will likely take up more time, but it will also evaluate more block shapes.

By default, `benchmark.py` runs Operators with auto-tuning in aggressive mode,
that is as `op.apply(autotune='aggressive')`. This can be changed with the
`-a/--autotune` flags. In particular, `benchmark.py` uses the so called
"pre-emptive" auto-tuning, which implies two things:

* The Operator's output fields are copied, and the Operator will write to these
  copies while auto-tuning. So the memory footprint is temporarily larger during
  this phase.
* The auto-tuning phase produces values that are eventually ditched;
  afterwards, the actual computation takes place. The execution time of the
  latter does not include auto-tuning.

Note that in production runs one should/would rather use the so called "runtime
auto-tuning":
```
op.apply(autotune=('aggressive', 'runtime'))
```
in which auto-tuning, as the name suggests, will be performed during the first N
timesteps of the actual computation, after which the best block shapes are
selected and used for all remaining timesteps.

## Choice of the backend compiler

The "backend compiler" takes as input the code generated by Devito and
translates it into a shared object. Supported backend compilers are `gcc`,
`icc`, `pgcc`, `clang`. For each of these compilers, Devito uses some preset compilation
flags (e.g., -O3, -march=native, etc).

The default backend compiler is `gcc`. To change it, one should set the
`DEVITO_ARCH` environment variable to a different value; run
```
from devito import print_defaults
print_defaults()
```
to get all possible `DEVITO_ARCH` values.

## Benchmark verbosity

Run with `DEVITO_LOGGING=DEBUG` to find out the specific performance
optimizations applied by an Operator, how auto-tuning is getting along, and to
emit more performance metrics.


## Example commands

The isotropic acoustic wave forward Operator in a `512**3` grid, space order
12, and a simulation time of 100ms:
```bash
python benchmark.py run -P acoustic -d 512 512 512 -so 12 --tn 100
```
Like before, but with auto-tuning in `basic` mode:
```bash
python benchmark.py run -P acoustic -d 512 512 512 -so 12 -a basic --tn 100
```
It is also possible to run a TTI forward operator -- here in a 512x402x890
grid:
```bash
python benchmark.py run -P tti -d 512 402 890 -so 12 -a basic --tn 100
```
Same as before, but telling devito not to use temporaries to store the
intermediate values which stem from mixed derivatives:
```bash
python benchmark.py run -P tti -d 512 402 890 -so 12 -a basic --tn 100 --opt
"('advanced', {'cire-mingain: 1000000'})"
```
Do not forget to pin processes, especially on NUMA systems; below, we use
`numactl` to pin processes and threads to one specific NUMA domain.
```bash
numactl --cpubind=0 --membind=0 python benchmark.py ...
```
While a benchmark is running, you can have some useful programs running in
background in other shells. For example, to monitor pinning:
```bash
htop
```
or to keep the memory footprint under control:
```bash
watch numastat -m
```

## The run-jit-backdoor mode

As of Devito v3.5 it is possible to customize the code generated by Devito.
This is often referred to as the ["JIT backdoor"
mode](https://github.com/devitocodes/devito/wiki/FAQ#can-i-manually-modify-the-c-code-generated-by-devito-and-test-these-modifications).
With ``benchmark.py`` we can exploit this feature to manually hack and test the
code generated for a given benchmark. So, we first run a problem, for example
```bash
python benchmark.py run-jit-backdoor -P acoustic -d 512 512 512 -so 12 --tn 100
```
As you may expect, the ``run-jit-backdoor`` mode accepts exactly the same arguments
as the ``run`` mode. Eventually, you will see a message along the lines of
```
You may now edit the generated code in
`/tmp/devito-jitcache-uid1000/31e8d25408f369754e2b7a26f4439944dc7683e2.c`. Then
save the file, and re-run this benchmark.
```
At this point, just follow the instructions on screen. The next time you run
the benchmark, the modified C code will be re-compiled and executed. Thus,
you will see the performance impact of your changes.

## Benchmark output

The GFlops/s and GPoints/s performance, Operational Intensity (OI) and
execution time are emitted to standard output at the end of each run.  You may
find this
[FAQ](https://github.com/devitocodes/devito/wiki/FAQ#how-does-devito-compute-the-performance-of-an-operator)
useful.
