# Devito: Fast Finite Difference Computation from Symbolic Specification

![Build Status](https://travis-ci.org/opesci/devito.svg?branch=master)

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

## Get in touch

If you're using Devito, we would like to hear from you. Whether you
are facing issues or just trying it out, join the
[conversation](https://opesci-slackin.now.sh). 

## Quickstart

The recommended way to install Devito uses the Conda package manager
for installation of the necessary software dependencies. Please
install either [Anaconda](https://www.continuum.io/downloads) or
[Miniconda](https://conda.io/miniconda.html) using the instructions
provided at the download links.

To install the editable development version Devito, including examples,
tests and tutorial notebooks, please run the following commands:
```
git clone https://github.com/opesci/devito.git
cd devito
conda env create -f environment.yml
source activate devito
pip install -e .
```

If you don't want to use the Conda environment manager, Devito can
also be installed directly from GitHub via pip:
```
pip install --user git+https://github.com/opesci/devito.git
```

## Examples

At the core of the Devito API are the so-called `Operator` objects that
allow users to create efficient FD kernels from SymPy expressions.
Examples of how to configure operators are provided:

* A set of introductory notebook tutorials introducing the basic
  features of Devito operators can be found under
  `examples/cfd`. These tutorials cover a range of well-known examples
  from Computation Fluid Dynamics (CFD) and are based on the excellent
  introductory blog ["CFD Python:12 steps to
  Navier-Stokes"](http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)
  by the Lorena A. Barba group. To run these, simply go into the tutorial
  directory and run `jupyter notebook`.
* A set of tutorial notebooks for seismic inversion examples is currently
  under construction in `examples/acoustic/tutorials`. We will add to these
  in the near future to provide more complex examples of how to use Devito
  operators for seismic imaging algorithms.
* Example implementations of acoustic forward, adjoint, gradient and born
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
from devito import print_defaults
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

Devito supports automatic auto-tuning of block sizes when loop blocking is
enabled. Enabling auto-tuning is simple: it can be done by passing the special
flag `autotune=True` to an `Operator`. Auto-tuning parameters can be set
through the special environment variable `DEVITO_AUTOTUNING`.

For more information on how to drive Devito for maximum run-time performance,
see [here](examples/PERFORMANCE.md).
