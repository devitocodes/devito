Devito tutorials
================

Listed below are several sets of tutorials that will help you get
started with Devito. They are grouped around individual scientific
computing topics and gradually introduce the key concepts of the
Devito API through IPython notebooks. All notebooks can be found
by checking out our the repository on github_

.. _github: https://github.com/opesci/devito

Computational Fluid Dynamics
----------------------------

A tutorial series that introduces the core features of Devito through
a set of classic examples from Computational Fluid Dynamics (CFD). The
series is based on the excellent tutorial blog `CFD Python: 12 steps
to Navier-Stokes
<http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/>`_
by the Lorena A. Barba Group and focusses on the implementation with
Devito rather than pure CFD or finite difference theory.

* `01 - Linear convection
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/01_convection.ipynb>`_
  - Building a linear operator with simple zero BCs.
* `01b - Linear convection, revisisted
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/01_convection_revisited.ipynb>`_
  - The above example, with a different initial condition.
* `02 - Nonlinear convection
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/02_convection_nonlinear.ipynb>`_
  - Building an operator with coupled equations and simple BCs.
* `03 - Diffusion
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/03_diffusion.ipynb>`_
  - Building an second-order operator with simple BCs.
* `04 - Burgers' equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/04_burgers.ipynb>`_
  - Coupled operator with mixed discretizations and simple BCs.
* `05 - Laplace equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/05_laplace.ipynb>`_
  - Steady-state example with Python convergence loop and Neumann BCs.
* `06 - Poisson equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/06_poisson.ipynb>`_
  - Pseudo-timestepping example with kernel-driven diffusion loop.


Seismic Modelling and Inversion
-------------------------------

A set of tutorials that introduces some basic concepts of seismic
modelling and highlights how to use Devito operators to solve seismic
inversion problems.

* `01 - Introduction to seismic modelling
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/01_modelling.ipynb>`_
* `02 - Reverse Time Migration
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/02_rtm.ipynb>`_
* `03 - Full-Waveform Inversion (FWI)
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/03_fwi.ipynb>`_
* `04 - Distributed FWI with Dask
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/04_dask.ipynb>`_
* `05 - FWI with total variation (TV) minimisation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/05_skimage_tv.ipynb>`_
* `06 - Acoustic modeling (2D) on a staggerd grid with the first-order wave equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/06_staggered_acoustic.ipynb>`_
* `07 - Elastic modeling (2D) on a staggerd grid with the first-order wave equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/07_elastic.ipynb>`_
