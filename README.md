# Devito: Fast Finite Difference Computation from Symbolic Specification

![Build Status](https://travis-ci.org/opesci/devito.svg?branch=master)
![Code Coverage](https://codecov.io/gh/opesci/devito/branch/master/graph/badge.svg)

[Devito](http://www.devitoproject.org) is a software to
implement optimised finite difference (FD) computation from
high-level symbolic problem definitions. Starting from symbolic
equations defined in [SymPy](http://www.sympy.org/en/index.html),
Devito employs automated code generation and just-in-time (JIT)
compilation to execute FD kernels on multiple computer platforms.

## Get in touch

If you're using Devito, we would like to hear from you. Whether you
are facing issues or just trying it out, join the
[conversation](https://opesci-slackin.now.sh).

## Quickstart

The recommended way to install Devito uses the Conda package manager
for installation of the necessary software dependencies. Please
install either [Anaconda](https://www.continuum.io/downloads) or
[Miniconda](https://conda.io/miniconda.html) using the instructions
provided at the download links. You will need the Python 3.6 version.

To install Devito, including examples, tests and tutorial notebooks,
follow these simple passes:

```sh
git clone https://github.com/opesci/devito.git
cd devito
conda env create -f environment.yml
source activate devito
pip install -e .
```

Alternatively, you can also install and run Devito via
[Docker](https://www.docker.com/):

```sh
# get the code
git clone https://github.com/opesci/devito.git
cd devito

# run the tests
docker-compose run devito /tests

# start a jupyter notebook server on port 8888
docker-compose up devito

# start a bash shell with devito
docker-compose run devito /bin/bash
```

## Examples

At the core of the Devito API are the so-called `Operator` objects, which
allow the creation and execution of efficient FD kernels from SymPy
expressions. Examples of how to define operators are provided:

* A set of introductory notebook tutorials introducing the basic
  features of Devito operators can be found under
  `examples/cfd`. These tutorials cover a range of well-known examples
  from Computational Fluid Dynamics (CFD) and are based on the excellent
  introductory blog ["CFD Python:12 steps to
  Navier-Stokes"](http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)
  by the Lorena A. Barba group. To run these, simply go into the tutorial
  directory and run `jupyter notebook`.
* A set of tutorial notebooks for seismic inversion examples is available in
  `examples/seismic/tutorials`.
* A set of tutorial notebooks concerning the Devito compiler can be found in
  `examples/compiler`.
* Devito with MPI can be explored in `examples/MPI`.
* Example implementations of acoustic forward, adjoint, gradient and born
  operators for use in full-waveform inversion (FWI) methods can be found in
  `examples/seismic/acoustic`.
* An advanced example of a Tilted Transverse Isotropy forward operator
  for use in FWI can be found in `examples/seismic/tti`.
* A benchmark script for the acoustic and TTI forward operators can be
  found in `benchmarks/user/benchmark.py`.


## Compilation

Devito's JIT compiler engine supports multiple backends, with provided
presets for the most common compiler toolchains. By default, Devito
will use the default GNU compiler `g++`, but other toolchains may be
selected by setting the `DEVITO_ARCH` environment variable to one of
the following values:
 * `gcc` or `gnu` - Standard GNU compiler toolchain
 * `clang` or `osx` - Mac OSX compiler toolchain via `clang`
 * `intel` or `icpc` - Intel compiler toolchain via `icpc`

Thread parallel execution via OpenMP can also be enabled by setting
`DEVITO_OPENMP=1`.

For the full list of available environment variables and their
possible values, simply run:

```py
from devito import print_defaults
print_defaults()
```

Or with Docker, run:

```sh
docker-compose run devito /print-defaults
```

## Performance optimizations

Devito supports two classes of performance optimizations:
 * Flop-count optimizations - They aim to reduce the operation count of an FD
   kernel. These include, for example, code motion, factorization, and
   detection of cross-stencil redundancies. The flop-count optimizations
   are performed by routines built on top of SymPy, implemented in the
   Devito Symbolic Engine (DSE), a sub-module of Devito.
 * Loop optimizations - Examples include SIMD vectorization and parallelism
   (via code annotations) and loop blocking. These are performed by the Devito
   Loop Engine (DLE), another sub-module of Devito.

Further, [YASK](https://github.com/intel/yask) is being integrated as a Devito
backend, for optimized execution on Intel architectures.

Devito supports automatic auto-tuning of block sizes when loop blocking is
enabled. Enabling auto-tuning is simple: it can be done by passing the special
flag `autotune=True` to an `Operator`. Auto-tuning parameters can be set
through the special environment variable `DEVITO_AUTOTUNING`.

For more information on how to drive Devito for maximum run-time performance,
see [here](benchmarks/user/README.md) or ask the developers on the Slack
channel linked above.
