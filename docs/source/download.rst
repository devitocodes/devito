===========================
Download and Install Devito
===========================

There are two main approaches to installing Devito.

- `Docker installation`_, for those looking for the least-friction way to try Devito
- `pip/conda installation`_, for those looking to use Devito as part of a project alongside other packages or/and looking to develop with Devito


Docker installation
-------------------

You can install and run Devito via Docker_:

.. _Docker: https://www.docker.com/  

For detailed installation instructions and information on the Devito Docker image library please follow 
the docker/README.md_

.. _README.md: ../../docker/README.md


pip/conda installation
----------------------

User route
``````````

This is the recommended method when setting up Devito as part of a larger project
that uses Devito among other python packages. You can use Devito either in a `Python virtual environment`_ or in a `Conda environment`_.
Devito is available as a `pip package`_ in PyPI. To install the latest Devito release along with any additional dependencies, follow:

.. code-block:: shell

   pip install devito
   # ...or to install additional dependencies:
   pip install -e .[extras,mpi,nvidia]

To install the latest Devito development version (current GitHub master) (without the tutorials), follow:

.. code-block:: shell

   pip install git+https://github.com/devitocodes/devito.git
   # ...or to install additional dependencies:
   pip install git+https://github.com/devitocodes/devito.git#egg=project[extras,mpi,nvidia]

Additional dependencies:

- extras : optional dependencies for Jupyter notebooks, plotting, benchmarking
- mpi : optional dependencies for MPI (mpi4py)
- nvidia : optional dependencies for targetting GPU deployment

.. _pip package: https://pypi.org/project/devito/

Note that you do not need to get the code via `git clone` in this method. 
Depending on your needs, this might also be the recommended setup for using Devito
in a production-like environment. However, since some components need to be
compiled before use, this approach may be sensitive to the C/C++ compilers present
on your system and the related environment, including what other packages you might
have installed.


Developer route
```````````````
This is the recommended method when your objective is to contribute
to and develop for Devito, including examples, tests and tutorial notebooks.
We highly recommend using Devito inside a python virtual environment,
e.g. a Python3 or Conda environment. Devito requires Python3 (3.6 to 3.10 currently supported).
Please install either Anaconda_ or Miniconda_.

.. _Anaconda: https://www.continuum.io/downloads

.. _Miniconda: https://conda.io/miniconda.html

.. _Python virtual environment: https://docs.python.org/3/library/venv.html

.. _Conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html

.. code-block:: shell

   # Clone Devito
   git clone https://github.com/devitocodes/devito.git
   cd devito

Then create a Python environment and activate it.

For a Python virtual environment:

.. code-block:: shell

   python3 -m venv /path/to/new/virtual/environment
   source activate /path/to/new/virtual/environment/bin/activate

For a Conda environment:

.. code-block:: shell

   conda create -n devito
   conda activate devito

and finally, install Devito along with any extra dependencies:

.. code-block:: shell

   pip install -e .
   # ... or to install additional dependencies
   pip install -e .[extras,mpi,nvidia]


Facing issues?
--------------

If you are facing any issues, we are happy to help on Slack_. Also, have a look at our
list of known installation issues_.

.. _issues: https://github.com/devitocodes/devito/wiki/Installation-Issues

.. _Slack: https://join.slack.com/t/devitocodes/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg
