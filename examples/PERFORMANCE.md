# Driving Devito for maximum run-time performance

This file provides a number of suggestions to improve the run-time performance
of the Operators generated through Devito. It also describes current
limitations and how they will be addressed in future releases.

## Default execution mode

By default, the execution of Devito Operators is sequential, with only one
thread used. The Devito Symbolic and Loop Engines (DSE and DLE) -- the
components that introduce optimizations prior to code generation -- are set to
the following levels:
 * DSE: `advanced`
 * DLE: `advanced`

This setup attempts to maximize the user experience at code development time,
by minimizing code generation time while ensuring a minimum level of run-time
performance. For analogous reasons, Operator auto-tuning is also by default
disabled.

## Performance maximization

There are a number of knobs that users can play with to improve the run-time
performance of an Operator. In the following, we briefly discuss them. The
full list of Devito options is available through:
```
from devito import print_defaults
print_defaults()
```
Note that not all these options impact the Operator run-time performance.

### Multi-threaded execution

To switch on multi-threaded execution via OpenMP, the following environment
variable must be set:
```
DEVITO_OPENMP=1
```
One has two options: either set it explicitly or prepend it before running
Python. In the former case, assuming a bash shell:
```
export DEVITO_OPENMP=1
```
In the latter case:
```
DEVITO_OPENMP=1 python mydevitocode.py
```

Now OpenMP execution is activated. If the Operator runs for more than just a
few seconds (e.g., in a realistic setting), one may easily see Devito using
multiple threads using a program such as [htop](http://hisham.hm/htop/), which
displays cores usage along with other metrics. `htop` should be launched in a
separate shell while Devito is running.

One can control the number of threads used by OpenMP by setting the
environment variable
```
OMP_NUM_THREADS=X
```
In which case, X threads will be used. If left unset, as many threads as the
number of logical cores available on the node will be used.

One typical issue with multi-threaded execution is that, by default, threads
are not permanently "pinned" to the available cores; that is, a thread can
(more or less sporadically) "migrate" from one core to another, causing a lot
of troubles to run-time performance. In particular, thread migration causes (i)
degradation in run-time performance and (ii) fluctuations in execution times.
Disabling this operating system feature is a must during experimentation. One
can achieve that by setting standard OpenMP environment variables to specific
values. The OpenMP 4.0 standards, supported by most recent compilers, provides
a number of environment variables to drive [thread
pinning](http://www.openmp.org/wp-content/uploads/OpenMP4.0.0.pdf). The most
basic setting that one should do is:
```
OMP_PROC_BIND=1
```

Some compilers provide their own wrappers around these OpenMP standard
variables. The case of the Intel compiler, or `icc`, is remarkable. In
particular, the following flag can be set:
```
KMP_HW_SUBSET=Xc,Yt
```
Meaning that thread pinning will be performed by using X physical cores and by
allocating Y logical threads to each of these cores. Thus, if a node has 16
cores, one could do:
```
export KMP_HW_SUBSET=16c,1t
```
In which case, no hyperthreads would be used (due to `1t`).
Regardless of how it is achieved, it is of fundamental importance to check
that thread pinning is actually happening. One can use a program like htop for
that.

### More aggressive DSE

The DSE can be asked to act smarter than in `advanced` mode by setting it to
`aggressive`. This can be done by providing the kwarg `dse='aggressive'` to an
Operator, or globally by setting
```
DEVITO_DSE=aggressive
```
In the latter case, however, consider that local choices take precedence.
Hence, if `dse=advanced` is passed to the Operator `OP`, then `OP` will be
compiled in `advanced` mode, even though `DEVITO_DSE=aggressive` was set.
One may understand that the switch to aggressive mode was successful by
observing changes in the Devito DSE output, printed to screen when code
generation for an Operator takes place.

Significant reductions in operation count due to using `aggressive` mode have
been observed in several TTI examples. The `aggressive` mode may or may not
increase the Devito processing time.

### Loop tiling with 3D blocks

Loop tiling is applied if the DLE is set to the `advanced` level. By default,
Devito tiles loop nests using 2D blocks. In some circumstances, however,
employing 3D blocks may lead to better performance. This is clearly a
problem-dependent aspect. To switch to using 3D blocks, one should set the
following environment variable:
```
DEVITO_DLE_OPTIONS="blockinner:True"
```

### Auto-tuning

Operator auto-tuning can greatly improve the run-time performance. It can be
activated on an Operator basis by passing the flag `autotune=True` at
Operator application time:
```
op = Operator(...)
op.apply(autotune=True)
```
The auto-tuning system will then attempt to retrieve the optimal block size for
loop blocking. When seeking maximum performance, it is recommended to maximize the
auto-tuner aggressiveness through:
```
DEVITO_AUTOTUNING=aggressive
```

### Choice of the backend compiler

For each Operator, Devito generates C code, which then gets compiled into a
shared object through the user-selected "backend compiler". Supported backend
compilers are `gcc`, `icc`, `clang`. In theory, any C compiler would work, but
for the aforementioned ones, Devito is aware of the best compilation flags. If
multiple compilers are available on the system, one can make a selection by
exporting
```
DEVITO_ARCH=X
```
`print_defaults()`, again, provides a list of legal values for X.

For maximum performance, it is *highly recommended to use the Intel compiler*.
Devito performs SIMD vectorization by resorting to the backend compiler
auto-vectorizer, and Intel's is particularly effective in stencil codes.

### Be aware of what's happening in Devito

Run with
```
DEVITO_LOGGING=DEBUG
```
To get more info from Devito about the performance optimizations applied or
on how auto-tuning is getting along.

# Known limitations and possible work arounds

 * At the moment, there is no support for MPI parallelism. This is perhaps the
   biggest limitation, because it significantly affects execution on
   multi-socket nodes.  The ideal setting on multi-socket nodes would be to
   have 1 MPI process per socket, and OpenMP (or, why not, MPI itself in
   shared-memory mode) for the processes within a socket. One can still use
   OpenMP across all available sockets (the default case if `OMP_NUM_THREADS` is
   unset and more than one sockets are available), but the final performance
   will be very far from the attainable machine peak, due to the well known
   NUMA effect.
   The good news is that MPI support is under development, and will be released
   in the upcoming months; the expectation is way before the end of 2017. Thus,
   if you have a multi-socket machine and you are simply trying to understand
   how Devito ideally performs, the recommendation is to run the experiments on
   a single socket. This can be achieved through suitable thread pinning.  The
   reader is invited to get in touch with the development team if struggling
   with this matter.
 * The DSE `aggressive` mode will not work with backend compilers that are not
   Intel. This is a known
   issue[https://github.com/opesci/devito/issues/320] in devito.

# Do not hesitate to contact us

Should you encounter any issues, or for any sort of questions, do not hesitate
to [get in touch with the development team](https://opesci-slackin.now.sh/)
