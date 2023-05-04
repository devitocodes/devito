===========================
Download and Install Devito
===========================

There are two main approaches to installing Devito.

- `Docker`_, for those looking for the least-friction way to try Devito
- `Virtual environment`_, for those looking to use Devito as part of a project alongside other packages

Docker
------

For detailed installation instructions and information on the Devito Docker image library please follow 
the docker/README.md_

.. _README.md: https://github.com/devitocodes/devito/tree/master/docker#readme


Virtual environment
-------------------

venv route
``````````

Devito is available as a `pip package`_ in PyPI.

Create a `Python virtual environment`_

.. _Python virtual environment: https://docs.python.org/3/library/venv.html

.. code-block:: shell

  python3 -m venv <your_venv_name>

Source the newly created `venv`. This needs to be repeated each time a new terminal is open.

.. code-block:: shell

  source <your_venv_name>/bin/activate


To install the `latest Devito release`_ along with any additional dependencies, follow:

.. code-block:: shell

   pip install devito
   # ...or to install additional dependencies:
   # pip install devito[extras,mpi,nvidia,tests]

.. _latest Devito release: https://pypi.org/project/devito/#history

To install the latest Devito development version, without the tutorials, follow:

.. code-block:: shell

   pip install git+https://github.com/devitocodes/devito.git
   # ...or to install additional dependencies:
   # pip install git+https://github.com/devitocodes/devito.git#egg=project[extras,mpi,nvidia,tests]

Additional dependencies:

- extras : optional dependencies for Jupyter notebooks, plotting, benchmarking
- tests : optional dependencies required for testing infrastructure
- mpi : optional dependencies for MPI (mpi4py)
- nvidia : optional dependencies for targetting GPU deployment

.. _pip package: https://pypi.org/project/devito/

Note that here, you do not need to get the code via `git clone`.
Depending on your needs, this might also be the recommended setup for using Devito
in a production-like environment. However, since some components need to be
compiled before use, this approach may be sensitive to the C/C++ compilers present
on your system and the related environment, including what other packages you might
have installed.


conda route
```````````
Please install either Anaconda_ or Miniconda_.

.. _Anaconda: https://www.continuum.io/downloads

.. _Miniconda: https://conda.io/miniconda.html

.. _Python virtual environment: https://docs.python.org/3/library/venv.html

.. _Conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html

.. code-block:: shell

   # Create new env with the name devito
   conda create --name devito
   # Activate the environment
   conda activate devito

and finally, install Devito along with any extra dependencies:

.. code-block:: shell

   pip install devito
   # ... or to install additional dependencies
   # pip install devito[extras,mpi,nvidia,tests]


For developers
``````````````
First clone Devito:

.. code-block:: shell

   git clone https://github.com/devitocodes/devito.git
   cd devito

and then install the requirements in your virtual environment (venv or conda):

.. code-block:: shell

   # Install requirements
   pip install -e .
   # ...or to install additional dependencies
   # pip install -e .[extras,mpi,nvidia,tests]


Facing issues?
--------------

If you are facing any issues, we are happy to help on Slack_. Also, have a look at our
list of known installation issues_.

.. _issues: https://github.com/devitocodes/devito/wiki/Installation-Issues

.. _Slack: https://join.slack.com/t/devitocodes/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg
