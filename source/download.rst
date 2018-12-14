=============
Obtain Devito
=============

The recommended way to install Devito uses the Conda package manager
for installation of the necessary software dependencies. First,
install either Anaconda_ or Miniconda_ using the instructions
provided at the download links.

.. _Anaconda: https://www.continuum.io/downloads

.. _Miniconda: https://conda.io/miniconda.html

Then, to install the editable development version Devito, which includes
examples, tests and tutorial notebooks, simply run the following commands:

.. code-block:: shell

   git clone https://github.com/opesci/devito.git
   cd devito
   conda env create -f environment.yml
   source activate devito
   pip install -e .

If you don't want to use the Conda environment manager, Devito can
also be installed directly from GitHub via pip:

.. code-block:: shell

   pip install --user git+https://github.com/opesci/devito.git
