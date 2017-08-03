# Driving Devito for maximum runtime performance

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
displays cores usage along with other metrics.

One typical issue with multi-threaded execution is that threads are not
permanently "pinned" to the available cores; that is, a thread can (more or
less sporadically) "migrate" from one core to another, causing a lot of
troubles to run-time performance. In particular, thread migration causes (i)
degradation in run-time performance and (ii) fluctuations in execution times.
Disabling this operating system feature is a must during experimentation. One
can achieve that by setting standard OpenMP environment variables to specific
values.

# Current limitations and possible work arounds

At the moment, there is no support for MPI parallelism,

