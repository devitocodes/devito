=========
Tutorials
=========

Listed below are several sets of tutorials that will help you get started with
Devito. They are grouped around individual scientific computing topics and
gradually introduce the key concepts of the Devito API through IPython
notebooks. All notebooks can be found by checking out our the repository on
github_

.. _github: https://github.com/devitocodes/devito

Computational Fluid Dynamics
----------------------------

A tutorial series that introduces the core features of Devito through a set of
classic examples from Computational Fluid Dynamics (CFD). The series is based
on the excellent tutorial blog `CFD Python: 12 steps to Navier-Stokes
<http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/>`_ by the
Lorena A. Barba Group and focuses on the implementation with Devito rather than
pure CFD or finite difference theory.

The tutorial, structured as a series of Jupyter notebooks, is available `here
<http://nbviewer.jupyter.org/github/devitocodes/devito/blob/master/examples/cfd/>`__.
The following topics are covered:

* Simple linear and nonlinear operators.
* Operators with coupled equations and mixed discretizations.
* Dirichlet and Neumann BCs.


Seismic Modelling and Inversion
-------------------------------

A tutorial series that introduces some basic concepts of seismic modelling and
highlights how to use Devito operators to solve seismic inversion problems.

The tutorial, structured as a series of Jupyter notebooks, is available `here
<http://nbviewer.jupyter.org/github/devitocodes/devito/blob/master/examples/seismic/tutorials/>`__.
The following topics are covered:

* Quick introduction to seismic modelling
* Reverse Time Migration
* Full-Waveform Inversion (FWI)
* Distributed FWI with Dask
* FWI with total variation minimisation
* Acoustic and elastic modelling (2D) on staggered grids
