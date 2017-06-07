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
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/test_01_convection.ipynb>`_
  - Building a linear operator with simple zero BCs.
* `02 - Nonlinear convection
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/test_02_convection_nonlinear.ipynb>`_
  - Building an operator with coupled equations and simple BCs.
* `03 - Diffusion
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/test_03_diffusion.ipynb>`_
  - Building an second-order operator with simple BCs.
* `04 - Burgers' equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/test_04_burgers.ipynb>`_
  - Coupled operator with mixed discretizations and simple BCs.
* `05 - Laplace equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/test_05_laplace.ipynb>`_
  - Steady-state example with Python convergence loop and Neumann BCs.
* `06 - Poisson equation
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/cfd/test_06_poisson.ipynb>`_
  - Pseudo-timestepping example with kernel-driven diffusion loop.


Seismic Modelling and Inversion
-------------------------------

A set of tutorials that introduces some basic concepts of seismic
modelling and highlights how to use Devito operators to solve seismic
inversion problems.

* `01 - Introduction to seismic modelling
  <http://nbviewer.jupyter.org/github/opesci/devito/blob/master/examples/seismic/tutorials/test_01_modelling.ipynb>`_
