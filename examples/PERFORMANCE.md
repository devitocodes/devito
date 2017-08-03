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
by minimizing code generationi time while ensuring a certain degree of run-time
performance optimization. For this reason, Operator auto-tuning is also by
default disabled.

## Performance maximization

There are a number of knobs that users can play with to improve the performance
of Operators. We here discuss them. 

### AAA

...


# Current limitations and possible work arounds

At the moment, there is no support for MPI parallelism,

