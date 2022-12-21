# Devito: Fast Stencil Computation from Symbolic Specification

[![Build Status for the Core backend](https://github.com/devitocodes/devito/workflows/CI-core/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-core)
[![Build Status with MPI](https://github.com/devitocodes/devito/workflows/CI-mpi/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-mpi)
[![Build Status on GPU](https://github.com/devitocodes/devito/workflows/CI-gpu/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-gpu)
[![Code Coverage](https://codecov.io/gh/devitocodes/devito/branch/master/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito)
[![Slack Status](https://img.shields.io/badge/chat-on%20slack-%2336C5F0)](https://join.slack.com/t/devitocodes/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://devitocodes.github.io/devito-performance)
[![PyPI version](https://badge.fury.io/py/devito.svg)](https://badge.fury.io/py/devito)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/devitocodes/devito/master)
[![Docker](https://img.shields.io/badge/dockerhub-images-important.svg?logo=Docker?color=blueviolet&label=docker&sort=semver)](https://hub.docker.com/r/devitocodes/devito)

[Devito](http://www.devitoproject.org) is a Python package to implement
optimized stencil computation (e.g., finite differences, image processing,
machine learning) from high-level symbolic problem definitions.  Devito builds
on [SymPy](http://www.sympy.org/en/index.html) and employs automated code
generation and just-in-time compilation to execute optimized computational
kernels on several computer platforms, including CPUs, GPUs, and clusters
thereof.

- [About Devito](#about-devito)
- [Installation](#installation)
- [Resources](#resources)
- [FAQs](https://github.com/devitocodes/devito/FAQ.md)
- [Performance](#performance)
- [Get in touch](#get-in-touch)
- [Interactive jupyter notebooks](#interactive-jupyter-notebooks)

## About Devito

Devito provides a functional language to implement sophisticated operators that
can be made up of multiple stencil computations, boundary conditions, sparse
operations (e.g., interpolation), and much more.  A typical use case is
explicit finite difference methods for approximating partial differential
equations. For example, a 2D diffusion operator may be implemented with Devito
as follows

```python
>>> grid = Grid(shape=(10, 10))
>>> f = TimeFunction(name='f', grid=grid, space_order=2)
>>> eqn = Eq(f.dt, 0.5 * f.laplace)
>>> op = Operator(Eq(f.forward, solve(eqn, f.forward)))
```

An `Operator` generates low-level code from an ordered collection of `Eq` (the
example above being for a single equation). This code may also be compiled and
executed

```python
>>> op(t=timesteps)
```

There is virtually no limit to the complexity of an `Operator` -- the Devito
compiler will automatically analyze the input, detect and apply optimizations
(including single- and multi-node parallelism), and eventually generate code
with suitable loops and expressions.

Key features include:

* A functional language to express finite difference operators.
* Straightforward mechanisms to adjust the discretization.
* Constructs to express sparse operators (e.g., interpolation), classic linear
  operators (e.g., convolutions), and tensor contractions.
* Seamless support for boundary conditions and adjoint operators.
* A flexible API to define custom stencils, sub-domains, sub-sampling,
  and staggered grids.
* Generation of highly optimized parallel code (SIMD vectorization, CPU and
  GPU parallelism via OpenMP and OpenACC, multi-node parallelism via MPI,
  blocking, aggressive symbolic transformations for FLOP reduction, etc.).
* Distributed NumPy arrays over multi-node (MPI) domain decompositions.
* Inspection and customization of the generated code.
* Autotuning framework to ease performance tuning.
* Smooth integration with popular Python packages such as NumPy, SymPy, Dask,
  and SciPy, as well as machine learning frameworks such as TensorFlow and
  PyTorch.

## Installation

The easiest way to try Devito is through Docker using the following commands:
```
# get the code
git clone https://github.com/devitocodes/devito.git
cd devito

# start a jupyter notebook server on port 8888
docker-compose up devito
```
After running the last command above, the terminal will display a URL such as
`https://127.0.0.1:8888/?token=XXX`. Copy-paste this URL into a browser window
to start a [Jupyter](https://jupyter.org/) notebook session where you can go
through the [tutorials](https://github.com/devitocodes/devito/tree/master/examples)
provided with Devito or create your own notebooks.

[See here](http://devitocodes.github.io/devito/download.html) for detailed installation
instructions and other options. If you encounter a problem during installation, please
see the
[installation issues](https://github.com/devitocodes/devito/wiki/Installation-Issues) we
have seen in the past. 

## Resources

To learn how to use Devito,
[here](https://github.com/devitocodes/devito/blob/master/examples) is a good
place to start, with lots of examples and tutorials.

The [website](https://www.devitoproject.org/) also provides access to other
information, including documentation and instructions for citing us.

Some FAQs are discussed [here](FAQ.md).

## Performance

If you are interested in any of the following

* Generation of parallel code (CPU, GPU, multi-node via MPI);
* Performance tuning;
* Benchmarking operators;

then you should take a look at this
[README](https://github.com/devitocodes/devito/blob/master/benchmarks/user).

You may also be interested in
[TheMatrix](https://www.devitocodes.com/blog/thematrix) -- a cross-architecture
benchmarking framework showing the performance of several production-grade
seismic operators implemented with Devito.

## Get in touch

If you're using Devito, we would like to hear from you. Whether you
are facing issues or just trying it out, join the
[conversation](https://join.slack.com/t/devitocodes/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg).

## Interactive jupyter notebooks
The tutorial jupyter notebook are available interactively at the public [binder](https://mybinder.org/v2/gh/devitocodes/devito/master) jupyterhub. 
