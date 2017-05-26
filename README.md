# Devito: Fast Finite Difference Computation from Symbolic Specification

[Devito](http://www.opesci.org/devito-public) is a tool to
perform optimised finite difference (FD) computation from
high-level symbolic problem definitions. Starting from symbolic
equations defined in [SymPy](http://www.sympy.org/en/index.html),
Devito employs automated code generation and just-in-time (JIT)
compilation to execute FD kernels on multiple computer platforms.

Devito is part of the [OPESCI](http://www.opesci.org) seismic imaging
project. A general overview of Devito features and capabilities can be
found [here](http://www.opesci.org/devito-public), including a
detailed [API documentation](http://www.opesci.org/devito).

## Quickstart

Devito can be installed from GitHub via pip:
```
pip install --user git+https://github.com/opesci/devito.git
```

Alternatively Devito can be be installed manually from GitHub via:
```
git clone https://github.com/opesci/devito.git
cd devito && pip install --user -r requirements.txt
```
When manually installing Devito please make sure you also add Devito
to your `PYTHONPATH`.

## Examples

At the core of the Devito API are the so-called `Operator` objects that
allow users to create efficient FD kernels from SymPy expressions.
Examples of how to configure operators are provided:

* A simple example of how to solve the 2D diffusion equation can be
  found in `examples/diffusion/example_diffusion.py`. This example
  also demonstrates how the equation can be solved via pure Python and
  optimised `numpy`, as well as Devito.
* A more practical example of acoustic forward, adjoint, gradient and born
  operators for use in full-waveform inversion (FWI) methods can be found in
  `examples/acoustic`.
* An advanced example of a Tilted Transverse Isotropy forward operator
  for use in FWI can be found in `examples/tti`.
* A benchmark example for the acoustic and TTI forward operators can be
  found in `examples/benchmark.py`

## Compilation

Devito's JIT compiler engine supports multiple backends, with provided
presets for the most common compiler toolchains. By default, Devito
will use the default GNU compiler `g++`, but other toolchains may be
selected by setting the `DEVITO_ARCH` environment variable to one of
the following values:
 * `gcc` or `gnu` - Standard GNU compiler toolchain
 * `clang` or `osx` - Mac OSX compiler toolchain via `clang`
 * `intel` or `icpc` - Intel compiler toolchain via `icpc`
 * `intel-mic` or `mic` - Intel Xeon Phi using offload mode via the
   `pymic` package

Thread parallel execution via OpenMP can also be enabled by setting
`DEVITO_OPENMP=1`.

For a full list of the available environment variables and their
possible values, simply execute:
```
from devito.parameters import print_defaults
print_defaults()
```

## Performance optimizations

Devito supports two classes of code optimizations, which are essential
in a wide range of real-life kernels:
 * Flop-count optimizations - They aim to reduce the operation count of an FD
   kernel. These include, for example, code motion, factorization, and
   detection of cross-stencil redundancies. The flop-count optimizations
   are performed by routines built on top of SymPy, which logically belong
   to the Devito Symbolic Engine (DSE), a sub-module of Devito.
 * Loop optimizations - Examples include SIMD vectorization and parallelism
   (via code annotations) and loop blocking. These are performed by the Devito
   Loop Engine (DLE), a sub-module consisting of a sequence of compiler passes
   manipulating abstract syntax trees. Some existing stencil optimizers
   are being integrated with the DLE: one of these is
   [YASK](https://github.com/01org/yask).
 
## Auto tuning block sizes

Devito supports automatic auto-tuning of block sizes when cache blocking is
enabled. Enabling auto-tuning is trivial, and can be done directly in the
symbolic layer by passing the special flag `autotune=True` to an `Operator`.
Auto-tuning parameters can be set through a special environment variable.
