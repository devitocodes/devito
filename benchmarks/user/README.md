# Benchmarking a Devito Operator

## Running benchmarks

`benchmark.py` implements a minimalist framework to evaluate the performance of
a Devito Operator while varying:

* the problem size (e.g., shape of the computational grid);
* the discretization (e.g., space- and time-order of the input/output fields);
* the simulation time (in milliseconds);
* the performance optimizations applied by the Devito compiler, including:
  - the Devito Symbolic Engine (DSE) optimizations for flop reduction,
  - the Devito Loop Engine (DLE) optimizations for parallelism and cache
    locality;
* the autotuning level.

Running `python benchmark.py --help` will display a list of useful options.

## A step back: configuring your machine for reliable benchmarking

If you are tempted to use your laptop to run a benchmark, you may want to
reconsider: heat and power management may affect the results you get in an
unpredictable way.

It is important that *both* the Python process running Devito (process*es* if
running with MPI) and the OpenMP threads spawned while running an Operator are
pinned to specific CPU cores, to get reliable and determinisic results. There
are several ways to achieve this:

* Through environment variables. All MPI/OpenMP distributions provide a set of
  environment variables to control process/thread pinning.  Devito also
  supplies the `set_omp_pinning.sh` program (under `/scripts`), which helps
  with thread pinning (though, currently, only limited to Intel architectures).
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
  - Unless on a hyperthreads-centerd system, such as an Intel Knights Landing,
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
[here](https://perf.readthedocs.io/en/latest/system.html).

There are many ways one can check that pinning is working as expected. A
recommended tool for rapid visual inspection is [htop](http://hisham.hm/htop/).

## Enabling OpenMP

To switch on multi-threaded execution of Operators via OpenMP, the following
environment variable must be set:
```
DEVITO_OPENMP=1
```
One has two options: either set it explicitly or prepend it to the Python
command. In the former case, assuming a bash shell:
```
export DEVITO_OPENMP=1
```
In the latter case:
```
DEVITO_OPENMP=1 python benchmark.py ...
```

## Enabling MPI

To switch on MPI, one should set
```
DEVITO_MPI=1
```
and run with `mpirun -n number_of_processes python benchmark.py ...`

Devito supports multiple MPI schemes for halo exchange. See the `Tips` section
below.

## DSE, DLE: optimization of generated code

`benchmark.py` offers three preset optimization modes:

 * 'O1': DSE=basic, DLE=basic.
   Minimum set of optimizations.
 * 'O2': DSE=advanced, DLE=advanced.
   The default setting when compiling an Operator. Switches on some
   flop-reduction transformations as well as fundamental loop-level
   optimizations such as loop blocking and vectorization.
 * 'O3': DSE=aggressive, DLE=advanced.
   More aggressive flop-reduction transformations, which might improve the
   runtime performance.

## Auto-tuning

Auto-tuning can greatly improve the run-time performance of an Operator. It
can be enabled on an Operator basis (it is off by default):
```
op = Operator(...)
op.apply(autotune=True)
```
The auto-tuner will discover a suitable block shape for each blocked loop nest
in the generated code.

With `autotune=True`, the auto-tuner gets set in `basic` mode, which will only
attempt a small batch of block shapes. With `autotune='aggressive'`, the
auto-tuning phase will take up more time, but it will also evaluate more
block-shapes.

When running `python benchmark.py ...`, the underlying Operators will
automatically be run in aggressive mode, that is as
`op.apply(autotune='aggressive')`.

The autotuning method can be selected with `-a ... ` 

`benchmark.py` uses the so called "pre-emptive" auto-tuning, which implies two
things:

* The Operator's output fields are copied, and the Operator will write to these
  copies while auto-tuning. So the memory footprint is at worst doubled during
  this phase.
* The auto-tuner is separated from the actual computation, as useless values
  get computed and eventually ditched.

Note that when benchmarking is not the goal, one should/would rather exploit
the so called "runtime auto-tuning":
```
op.apply(autotune=('aggressive', 'runtime'))
```
in which auto-tuning, as the name suggests, will be performed during the first N
timesteps of the actual computation, after which the best block shapes will be
selected and used for all remaining timesteps.

## Choice of the backend compiler

The "backend compiler" takes as input the code generated by Devito and
translates it into a shared object. Supported backend compilers are `gcc`,
`icc`, `clang`. For each of these compilers, Devito uses some preset compilation
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

You can also set `DEVITO_DEBUG_COMPILER=1` to emit the command used to compile
the generated code.

## Tips

* The 'O3' mode has been found to be particularly beneficial in TTI Operators,
  by resulting in significant reductions in operation count.
* The most powerful MPI mode is called "full", and is activated setting
  `DEVITO_MPI=full` instead of `DEVITO_MPI=1`. The "full" mode supports
  computation/communication overlap.
* When auto-tuning is enabled, one should always run in performance mode:
  ```
  from devito import mode_performance
  mode_perfomance()
  ```
  This is automatically turned on by `benchmark.py`

## Example commands

The isotropic acoustic wave forward Operator in a `512**3` grid, space order
12, and a simulation time of 100ms:
```
python benchmark.py run -P acoustic -d 512 512 512 -so 12 --tn 100
```
Like before, but with a specific optimization mode (O2) selected and auto-tuning
in `basic` mode:
```
python benchmark.py run -P acoustic -bm O2 -d 512 512 512 -so 12 -a basic --tn 100
```
It is also possible to run a TTI forward operator -- here in a 512x402x890
grid:
```
python benchmark.py run -P tti -bm O3 -d 512 402 890 -so 12 -a basic --tn 100
```
Do not forget to pin processes, especially on NUMA systems; below, we do so with
`numactl` on a dual-socket system.
```
numactl --cpubind=0 --membind=0 python benchmark.py ...
```
While a benchmark is running, you can have some useful programs running in
background in other shells. For example, to monitor pinning:
```
htop
```
or to keep the memory footprint under control:
```
watch numastat -m
```

## Running on HPC clusters

`benchmark.py` can be used to evaluate MPI on multi-node systems:
```
mpiexec python benchmark.py ...
```
In `bench` mode, each MPI rank will produce a different `.json` file
summarizing the achieved performance in a structured format.

Further, we provide `make-pbs.py`, a simple program to generate PBS files
to submit jobs on HPC clusters. Take a look at `python make-pbs.py --help`
for more information, and in particular `python make-pbs.py generate --help`.
`make-pbs.py` is especially indicated if interested in running strong scaling
experiments.

## Benchmarks' output

The GFlops/s and GPoints/s performance, Operational Intensity (OI) and
execution time are emitted to standard output at the end of each run.
Further, when running in bench mode, a `.json` file is produced
(see `python benchmark.py bench --help` for more info) in a folder named
`results` except if otherwise specified with the `-r` option specifying
the results directory.

So the isotropic acoustic wave forward Operator in a `512**3` grid, space order
12, and a simulation time of 100ms:

```
`DEVITO_LOGGING=DEBUG` python benchmark.py bench -P acoustic -d 512 512 512 -so 12 --tn 100
```

## Generating a roofline model

To generate a roofline model from the results obtained in `bench` mode,
one can execute `benchmark.py` in `plot` mode. For example, the command

```
python benchmark.py plot -P acoustic -d 512 512 512 -so 12 --tn 100 -a aggressive
 --max-bw 12.8
--flop-ceil 80 linpack
```

will generate a roofline model for the results obtained from

```
python benchmark.py bench -P acoustic -d 512 512 512 -so 12 --tn 100 -a
```

The `plot` mode expects the same arguments used in `bench` mode plus
two additional arguments to generate the roofline:

*    --max-bw FLOAT
    DRAM bandwidth (GB/s)
*    --flop-ceil <FLOAT TEXT>
    CPU machine peak. A 2-tuple (float, str) is expected,
    representing the performance ceil (GFlops/s) and how the ceil was obtained
     (ideal peak, linpack, ...), respectively

In addition, points can be annotated with the runtime value, passing the
`--point-runtime` argument.

To obtain the DRAM bandwidth of a system, we advise to use
 [STREAM][http://www.cs.virginia.edu/stream/ref.html].

To obtain the ideal CPU peak, one should instantiate this formula

#[cores] · #[avx units] · #[vector lanes] · #[FMA ports] · [ISA base frequency]

More details in this [paper][https://arxiv.org/pdf/1807.03032.pdf].

## Known limitations and possible work arounds

 * The DSE `aggressive` mode might not work in combination with OpenMP if the
   backend compiler is `gcc` version `< 8.3`. This is a known
   [issue](https://github.com/opesci/devito/issues/320).

## Do not hesitate to contact us

Should you encounter any issues, do not hesitate to
[get in touch with the development team](https://opesci-slackin.now.sh/)
