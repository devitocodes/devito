# Devito: Fast Finite Difference Computation from Symbolic Specification

[![Build Status for the Core backend](https://github.com/devitocodes/devito/workflows/CI-core/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-core)
[![Build Status with MPI](https://github.com/devitocodes/devito/workflows/CI-mpi/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-mpi)
[![Code Coverage](https://codecov.io/gh/devitocodes/devito/branch/master/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito)

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

For the least friction method, see the [Docker](#docker-installation) instructions.

If you are familiar with python and pip, just run `pip install devito`. Also look
[here](#pip-installation) for more information.

If you would like to develop and make contributions to Devito, we recommend taking
a look at our [Conda environment](#conda-environment).

## Installation
There are three main approaches to installing Devito. For those looking for the
least-friction way to try Devito, we recommend the Docker route. For those
looking to use Devito alongside other packages as part of a project, we support
pip installation. If you're looking to develop for Devito, you might benefit from
using the included conda environment that includes all the bells and whistles we
recommend when developing for Devito.

### Docker installation
You can install and run Devito via [Docker](https://www.docker.com/):

```sh
# get the code
git clone https://github.com/devitocodes/devito.git
cd devito

# 1. run the tests
docker-compose run devito /tests

# 2. start a jupyter notebook server on port 8888
docker-compose up devito

# 3. start a bash shell with devito
docker-compose run devito /bin/bash
```

The three sample commands above have only been included as illustrations of typical
uses of Devito inside a container. They are not required to be run in that order.

1. Command 1 above runs the unit tests included with Devito to check whether the
installation succeeded. This is not necessary but a handy first thing to run. As
long as the first few tests are passing, it is possible to press `Ctrl+C` to stop
the testing and continue.
2. Command 2 starts a [Jupyter](https://jupyter.org/) notebook server inside the
docker container and forwards the port to `http://localhost:8888`. After running
this command, you can copy-paste the complete URL from the terminal window where
the command was run - to a browser to open a jupyter session to try out the included
tutorials. Alternatively, you may simply point your browser to `http://localhost:8888`
and, if prompted for a password, copy-paste the authentication token from the command
window.
3. Command 3 above starts a bash (command-line) shell with Devito loaded into the
environment. Essentially, it means that any python code run after this command will
see devito when doing imports like `from devito import Function`. Any code using
Devito can be executed using this method.

Issues? Ask us for help on [Slack](https://opesci-slackin.now.sh).

### Pip installation
This is the recommended method when setting up Devito as part of a larger project
that uses Devito among other python packages. To install Devito using `pip`, simply
do:
```sh
pip install devito
```
to install the latest release from PyPI. Note that you don't need to get the code
using git first in this method. 
Depending on your needs, this might also be the recommended setup for using Devito
in a production-like environment. However, since some of the components need to be
compiled before use, this approach may be sensitive to the C/C++ compilers present
on your system and the related environment including what other packages you might
have installed.

If you are facing any issues, we are happy to help on
[Slack](https://opesci-slackin.now.sh).

### Conda Environment
If your objective is to contribute to and develop for Devito, the recommended way would
be to use the included conda environment that also installs an appropriate C compiler
along with all the bells and whistles we felt were necessary when developing for Devito.
Please install either [Anaconda](https://www.continuum.io/downloads) or
[Miniconda](https://conda.io/miniconda.html) using the instructions
provided at the download links. You will need the Python 3 version.

To install Devito, including examples, tests and tutorial notebooks,
follow these simple passes:

```sh
git clone https://github.com/devitocodes/devito.git
cd devito
conda env create -f environment-dev.yml
source activate devito
pip install -e .
```
If you are facing any issues, we are happy to help on
[Slack](https://opesci-slackin.now.sh).

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
* Devito with MPI can be explored in `examples/mpi`.
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
