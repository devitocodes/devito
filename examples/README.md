## Examples and tutorials

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
