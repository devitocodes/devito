## How to navigate this directory

Examples and tutorials are provided in the form of single Python files, as Jupyter
notebooks, or as mini-apps built on top of Devito.

Jupyter notebooks are files with extension `.ipynb`. To execute these, run
`jupyter notebook`, and then click on the desired notebook in the window that
pops up in your browser. In alternative, you may explore the pre-rendered
notebooks directly on GitHub or, for a potentially smoother experience, [with
nbviewer](https://nbviewer.jupyter.org/github/devitocodes/devito/tree/master/examples/).

We recommend newcomers to start with the following sets of tutorials:

* `userapi`: Gentle introduction to symbolic computation with Devito.
* `cfd`: A series of introductory notebooks showing how to use Devito to
  implement finite difference operators typical of computational fluid
  dynamics. These are based on the excellent blog ["CFD Python:12 steps to
  Navier-Stokes"](http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)
  by the Lorena A. Barba group.

A set of more advanced examples are available in `seismic`:

* `seismic/tutorials`: A series of Jupyter notebooks of incremental complexity,
  showing a variety of Devito features in the context of seismic inversion
  operators. Among the discussed features are custom stencils, staggered
  grids, tensor notation, and time blocking.
* `seismic/acoustic`: Example implementations of isotropic acoustic forward,
  adjoint, gradient and born operators, suitable for full-waveform inversion
  methods (FWI).
* `seismic/tti`: Example implementations of several anisotropic acoustic
  forward operators (TTI).
* `seismic/elastic`: Example implementation of an isotropic elastic forward
  operator. `elastic`, unlike `acoustic` and `tti`, fully exploits the
  tensorial nature of the Devito symbolic language.
* `seismic/viscoelastic`: Example implementation of an isotropic viscoelastic
  forward operator. Like `elastic`, `viscoelastic` exploits tensor functions
  for a neat and compact representation of the discretized partial differential
  equations.
* `seismic/self-adjoint`: Self-adjoint energy conserving pseudo-acoustic
  operators, including notebooks for implementation of the nonlinear forward,
  the forward and adjoint linearized Jacobian, and tests proving accuracy and 
  correctness.

Further:

* `mpi`: Jupyter notebooks explaining how MPI works in Devito.
* `finance`: Jupyter notebooks with examples of applying Devito to partial differential equations with financial applications.
* `misc`: Example operators outside the context of finite differences and
* `performance`: Jupyter notebooks explaining the optimizations applied by Devito, the options available to steer the optimization process, how to run on GPUs, and much more.

For developers:

* `compiler`: A set of notebooks exploring the architecture of the Devito
  compiler. This is still in its infancy.

## More resources

* Articles, presentations, posters and much more concerning Devito is available
  [here](https://www.devitoproject.org/publications). The entries are ordered
  chronologically -- those at the top being the most recent ones, for each
  section.
* The user documentation is available [here](http://devitocodes.github.io/devito/).
