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

   git clone https://github.com/devitocodes/devito.git
   cd devito
   conda env create -f environment.yml
   source activate devito
   pip install -e .
   
Alternatively, you can also install and run Devito via Docker_:

.. _Docker: https://www.docker.com/  

.. code-block:: shell

   # get the code
   git clone https://github.com/devitocodes/devito.git
   cd devito

   # run the tests
   docker-compose run devito /tests

   # start a jupyter notebook server on port 8888
   docker-compose up devito

   # start a bash shell with devito
   docker-compose run devito /bin/bash

If you don't want to use the Conda environment manager or Docker, Devito can
also be installed directly from GitHub via pip:

.. code-block:: shell

   pip install --user git+https://github.com/devitocodes/devito.git
   
A link to the installation page of devito, where you can find instructions or solutions to possible issues can be found here_.

.. _here: https://github.com/devitocodes/devito/wiki/Installation-Issues  

