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

There are several "tags" for the Devito Docker images. We provide `Docker for CPUs`_
and `Docker for GPUs`_ depending on the platform you want to deploy Devito.
For detailed installation instructions and information on the Devito Docker image library please follow 
the docker/README.md_

.. _README.md: ../../docker/README.md

TODROP
Docker for CPUs
```````````````
Available tags:

- cpu-dev: for the latest development version (current GitHub master)
- cpu-latest: for the latest GitHub release
- cpu-vX.X.X: for a specific release (Release tags: https://github.com/devitocodes/devito/tags)

In the absence of a tag, Docker defaults to `latest`, which maps to `cpu-latest`.

Devito:latest, which maps to Devito:cpu-latest

.. code-block:: shell

   # 1. Pull Devito image
   docker pull devitocodes/devito:cpu-latest

   # 2. (Optional but recommended) Test installation
   docker run --rm --name testrun 'devitocodes/devito:cpu-latest' pytest tests/test_operator.py

   # 3. Start a bash shell with Devito
   docker run --rm -it devitocodes/devito:cpu-latest /bin/bash

   # 4. Start a Jupyter notebook server on port 8888
   docker run --rm -it -p 8888:8888 devitocodes/devito:cpu-latest

4. Command 4 starts a Jupyter_ notebook server inside the Docker
container and forwards the port to `http://localhost:8888`.
After running this command, you can copy-paste the complete URL from the terminal window where
the command was run - to a browser to open a Jupyter session to try out the included
tutorials. Alternatively, you may point your browser to `http://localhost:8888`
and, if prompted for a password, copy-paste the authentication token from the command
window. Once successfully in the Jupyter notebook session, proceed to run the tutorials
provided in the `examples` folder or create your own notebooks. 

.. _Jupyter: https://jupyter.org/

For those aiming to develop in Devito using Docker, you can use docker-compose

.. code-block:: shell

   # 1. Clone the code
   git clone https://github.com/devitocodes/devito.git
   cd devito

   # 2. Start a bash shell with Devito
   docker-compose run devito /bin/bash

TODROP
Docker for GPUs
```````````````
The GPU image differs from the CPU image in the additional compilers and toolkits necessary to run on GPUs. It contains the same installation and usage of Devito as the CPU image.

Requirements:

- Install the `NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit>`_.

Available tags:

- gpu-dev: for the latest development version (current GitHub master)
- gpu-latest: for the latest GitHub release
- gpu-vX.X.X: for a specific release (Release tags: https://github.com/devitocodes/devito/tags)

.. code-block:: shell

   # 1. Pull Devito image
   docker pull devitocodes/devito:gpu-latest

   # 2. (Optional but recommended) Test installation
   docker run --gpus all --rm --name testrun 'devitocodes/devito:gpu-latest' pytest tests/test_gpu_openacc.py

   # 3. Start a bash shell with Devito
   docker run --gpus all --rm -it devitocodes/devito:gpu-latest /bin/bash

For those aiming to develop in Devito using Docker, you can use docker-compose

.. code-block:: shell

   # 1. Clone the code
   git clone https://github.com/devitocodes/devito.git
   cd devito

   # 2. Start a bash shell with Devito
   docker-compose run devito.nvidia /bin/bash

pip/conda installation
----------------------

User route
``````````

This is the recommended method when setting up Devito as part of a larger project
that uses Devito among other python packages. You can use Devito either in a `Python3 virtual environment`_ or in a `Conda environment`_.
Devito is available as a `pip package`_ in PyPI. To install the latest Devito release along with any additional dependencies, follow:

.. code-block:: shell

   pip install devito
   # Instead, to install additional dependencies
   # pip install -e .[extras,mpi,nvidia]

To install the latest Devito development version (current GitHub master) (without the tutorials), follow:

.. code-block:: shell

   pip install git+https://github.com/devitocodes/devito.git
   # Instead, to install additional dependencies
   # pip install git+https://github.com/devitocodes/devito.git#egg=project[extras,mpi,nvidia]

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

.. _Python3 virtual environment: https://docs.python.org/3/library/venv.html

.. _Conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html

.. code-block:: shell

   # Clone Devito
   git clone https://github.com/devitocodes/devito.git
   cd devito

Then create a Python environment and activate it.

For a Python3 virtual environment:

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
   # To install additional dependencies
   # pip install -e .[extras,mpi,nvidia]


Facing issues?
--------------

If you are facing any issues, we are happy to help on Slack_. Also, have a look at our
list of known installation issues_.

.. _issues: https://github.com/devitocodes/devito/wiki/Installation-Issues

.. _Slack: https://join.slack.com/t/devitocodes/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg
