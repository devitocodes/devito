# Devito: Fast Tensor Computation from Symbolic Specification

[![Build Status for the Core backend](https://github.com/devitocodes/devito/workflows/CI-core/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-core)
[![Build Status with MPI](https://github.com/devitocodes/devito/workflows/CI-mpi/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-mpi)
[![Code Coverage](https://codecov.io/gh/devitocodes/devito/branch/master/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito)
[![Slack Status](https://img.shields.io/badge/chat-on%20slack-%234A154B)](https://opesci-slackin.now.sh)

[Devito](http://www.devitoproject.org) is a Python package to implement
optimised tensor computation from high-level symbolic problem definitions.
Devito builds on [SymPy](http://www.sympy.org/en/index.html) and employs
automated code generation and just-in-time (JIT) compilation to execute
optimized computational kernels on multiple computer platforms, including
CPUs, GPUs, and clusters thereof.

- [About Devito](#about-devito)
- [Installation](#installation)
- [Resources](#resources)
- [Performance](#performance)
- [Get in touch](#get-in-touch)

## About Devito

Devito provides a symbolic language to implement matrix-free operators, that is
operators that do not require the explicit creation of a dense or sparse
matrix. A typical use case is explicit finite difference (FD) methods for
approximating partial differential equations. For example, a 2D diffusion
operator may be implemented as follows

```python
>>> grid = Grid(shape=(10, 10))
>>> f = TimeFunction(name='f', grid=grid, space_order=2)
>>> eqn = Eq(f.dt, 0.5 * f.laplace)
>>> op = Operator(Eq(f.forward, solve(eqn, f.forward)))
```

An `Operator` generates low-level code from an ordered collection of `Eq`. This
code may also be compiled and executed

```python
>>> op(t=timesteps)
```

There is virtually no limit to the type of matrix-free operators that one can
implement with Devito. The Devito compiler will automatically analyze the input,
detect and apply optimizations (including single- and multi-node parallelism),
and eventually generate code with arbitrarily complex loop nests, as required
by the symbolic specification -- clearly, all this is hidden away to the user.

Key features include:

* A tensor language to express FD operators.
* Straightforward mechanisms to dynamically adjust the discretizion.
* Constructs to express sparse operators (e.g., interpolation), classic linear
  operators (e.g., convolutions), and tensor contractions.
* Seamless support for adjoint operators.
* A flexible API to define custom stencils, sub-domains, sub-sampling,
  staggered grids.
* Distributed NumPy arrays over multi-node (MPI) domain decompositions.
* Generation of highly optimized parallel code (SIMD vectorization,
  CPU/GPU/multi-node parallelism, blocking, aggressive symbolic transformations
  for FLOPs reduction, etc.).
* Inspection and customization of the generated code.
* Autotuning framework to ease performance tuning.
* Smooth integration with with popular Python packages such as NumPy, SymPy,
  Dask and SciPy.

## Installation

The easiest way to try Devito is through Docker using the following commands:
```
# get the code
git clone https://github.com/devitocodes/devito.git
cd devito

# start a jupyter notebook server on port 8888
docker-compose up devito
```
After running the last command above, the terminal will display a URL like
`https://127.0.0.1:8888/?token=XXX`. Copy-paste this URL into a browser window
to start a [Jupyter](https://jupyter.org/) notebook session where you can go
through the [tutorials](https://github.com/devitocodes/devito/tree/master/examples)
provided with Devito or create your own notebooks.

[See here](http://devitocodes.github.io/devito/download.html) for detailed installation
instructions and other options. Also look at the
[installation issues](https://github.com/devitocodes/devito/wiki/Installation-Issues) we
have seen in the past. 

## Resources

To learn how to use Devito,
[here](https://github.com/devitocodes/devito/blob/master/examples) is a good
place to start, with lots of examples and tutorials.

The [website](https://www.devitoproject.org/) also provides access to other
info, including documentation and instructions for citing us.

## Performance

If you are interested in any of the following

* Generation of parallel code (CPU, GPU, multi-node via MPI);
* Performance tuning;
* Benchmarking Devito;
* Any other aspect concerning the application performance;

then you should take a look at the README available
[here](https://github.com/devitocodes/devito/blob/master/benchmarks/user).

## Get in touch

If you're using Devito, we would like to hear from you. Whether you
are facing issues or just trying it out, join the
[conversation](https://opesci-slackin.now.sh).
