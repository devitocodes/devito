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
[here](https://hpc-wiki.info/hpc/Binding/Pinning).

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

## The optimization level

In Devito, an Operator has two preset optimization levels: `noop` and
`advanced`.  With `noop`, no performance optimizations are introduced by the
compiler. With `advanced`, several flop-reducing and data locality
optimizations are applied. Examples of flop-reducing optimizations are common
sub-expressions elimination and factorization; examples of data locality
optimizations are loop fusion and cache blocking. SIMD vectorization is also
applied through compiler auto-vectorization.

`benchmark.py` has two preset optimization modes, that for historical reasons
are called `O1` and `O2`. Basically, `O1` corresponds to `noop`, while `O2`
corresponds to `advanced`.

## Auto-tuning

Auto-tuning can significantly improve the run-time performance of an Operator. It
can be enabled on an Operator basis:
```
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

## The run-jit-backdoor mode

As of Devito v3.5 it is possible to customize the code generated by Devito. This
is often referred to as the ["JIT backdoor" mode](https://github.com/devitocodes/devito/wiki/FAQ#can-i-manually-modify-the-c-code-generated-by-devito-and-test-these-modifications).
With ``benchmark.py`` we can exploit this feature to manually hack and test the
code generated for a given benchmark. So, we first run a problem, for example
```
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

## Benchmark output

The GFlops/s and GPoints/s performance, Operational Intensity (OI) and
execution time are emitted to standard output at the end of each run.
Further, when running in `bench` mode, a `.json` file is produced
(see `python benchmark.py bench --help` for more info) in a folder named
`results` except if otherwise specified with the `-r` option.

## Generating a roofline model

To generate a roofline model from the results obtained in `bench` mode,
one can execute `benchmark.py` in `plot` mode. For example, the command

```
python benchmark.py plot -P acoustic -d 512 512 512 -so 12 --tn 100 -a aggressive --max-bw 12.8 --flop-ceil 80 linpack
```

will generate a roofline model for the results obtained from

```
python benchmark.py bench -P acoustic -d 512 512 512 -so 12 --tn 100 -a
```

The `plot` mode expects the same arguments used in `bench` mode plus
two additional arguments to generate the roofline:

*    `--max-bw <float>`: DRAM bandwidth (GB/s).
*    `--flop-ceil <float, str>`: CPU machine peak. The CPU performance ceil
        (GFlops/s) and how the ceil was obtained (ideal peak, linpack, ...).

There also are two optional arguments:

*   `--point-runtime` (bool switch): Annotate points with the runtime value.
*   `--section <str>`:  The code section for which the roofline is produced.
        An Operator consists of multiple sections. Each section typically
        comprises a loop nest and a sequence of equations. Different sections
        are created for logically-distinct parts of the computation
        (finite-difference stencils, boundary conditions, interpolation, etc.).
        The naming convention is `sectionX`, where `X` is a progress id (`section0`,
        `section1`, ...). In the generated code the beginning and the end
        of a section are marked with suitable comments. Currently, there is
        no way other than looking at the generated code to understand which
        section the user-provided equations belong to.

To obtain the DRAM bandwidth of a system, we advise to use
 [STREAM](http://www.cs.virginia.edu/stream/ref.html).

To obtain the ideal CPU peak, one should instantiate this formula

#[cores] · #[avx units] · #[vector lanes] · #[FMA ports] · [ISA base frequency]

More details in this [paper](https://arxiv.org/pdf/1807.03032.pdf).

## Do not hesitate to contact us

Should you encounter any issues, do not hesitate to
[get in touch with the development team](https://opesci-slackin.now.sh/)
