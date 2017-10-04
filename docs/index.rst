.. Devito documentation master file, created by
   sphinx-quickstart on Wed Jul 20 13:02:08 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Devito - Fast Finite Difference Computation
===========================================

Devito is a new tool for performing optimised Finite Difference (FD)
computation from high-level symbolic problem definitions. Devito performs
automated code generation and Just-In-time (JIT) compilation based on symbolic
equations defined in `SymPy <http://www.sympy.org/en/index.html>`_ to create
and execute highly optimised Finite Difference kernels on multiple computer
platforms.

Devito is also intended to provide the driving operator kernels for a prototype
Full Waveform Inversion (FWI) code that can be found 
`here <https://github.com/opesci/inversion>`_.

Getting started
---------------

You can get instructions on how to download and install Devito
:doc:`here </download>`.

To learn how to use Devito, check our :doc:`tutorials and examples </tutorials>`.

You can find the API Documentation :doc:`here </devito>`.

.. toctree::
   :maxdepth: 4
   :hidden:

   Download <download>
   Tutorials <tutorials>
   Developer Documentation <developer>
   Example Documentation <examples>
   API Documentation <devito>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
