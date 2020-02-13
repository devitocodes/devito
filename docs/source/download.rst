=============
Obtain Devito
=============

There are three main approaches to installing Devito. For those looking for the
least-friction way to try Devito, we recommend the Docker route. For those
looking to use Devito alongside other packages as part of a project, we support
pip installation. If you're looking to develop for Devito, you might benefit from
using the included conda environment that includes all the bells and whistles we
recommend when developing for Devito.

Docker installation
-------------------
You can install and run Devito via Docker_:

.. _Docker: https://www.docker.com/  

.. code-block:: shell

   # get the code
   git clone https://github.com/devitocodes/devito.git
   cd devito

   # 1. run the tests
   docker-compose run devito /tests

   # 2. start a jupyter notebook server on port 8888
   docker-compose up devito

   # 3. start a bash shell with devito
   docker-compose run devito /bin/bash

The three sample commands above have only been included as illustrations of typical
uses of Devito inside a container. They are not required to be run in that order.

1. Command 1 above runs the unit tests included with Devito to check whether the
installation succeeded. This is not necessary but a handy first thing to run. As
long as the first few tests are passing, it is possible to press `Ctrl+C` to stop
the testing and continue.

2. Command 2 starts a Jupyter_ notebook server inside the
docker container and forwards the port to `http://localhost:8888`.
After running this command, you can copy-paste the complete URL from the terminal window where
the command was run - to a browser to open a jupyter session to try out the included
tutorials. Alternatively, you may simply point your browser to `http://localhost:8888`
and, if prompted for a password, copy-paste the authentication token from the command
window. Once successfully in the Jupyter notebook session, proceed to run the tutorials
provided in the `examples` folder or create your own notebooks. 

3. Command 3 above starts a bash (command-line) shell with Devito loaded into the
environment. Essentially, it means that any python code run after this command will
see devito when doing imports like `from devito import Function`. Any code using
Devito can be executed using this method.

Issues? Ask us for help on Slack_. Also look at our list of known
installation issues_.

.. _Jupyter: https://jupyter.org/

Pip installation
----------------
This is the recommended method when setting up Devito as part of a larger project
that uses Devito among other python packages. To install Devito using `pip`, simply
do:


.. code-block:: shell

   pip install --user git+https://github.com/devitocodes/devito.git


to install. Note that you don't need to get the code
using git first in this method. 
Depending on your needs, this might also be the recommended setup for using Devito
in a production-like environment. However, since some of the components need to be
compiled before use, this approach may be sensitive to the C/C++ compilers present
on your system and the related environment including what other packages you might
have installed.

If you are facing any issues, we are happy to help on Slack_. Also look at our list of known
installation issues_.

Conda Environment
-----------------
If your objective is to contribute to and develop for Devito, the recommended way would
be to use the included conda environment that also installs an appropriate C compiler
along with all the bells and whistles we felt were necessary when developing for Devito.
Please install either Anaconda_ or Miniconda_ using the instructions
provided at the download links. You will need the Python 3 version.

To install Devito, including examples, tests and tutorial notebooks,
follow these:

.. code-block:: shell

   git clone https://github.com/devitocodes/devito.git
   cd devito
   conda env create -f environment.yml
   source activate devito
   pip install -e .

If you are facing any issues, we are happy to help on Slack_. Also look at our list of known
installation issues_.

.. _issues: https://github.com/devitocodes/devito/wiki/Installation-Issues  

.. _Slack: https://opesci-slackin.now.sh

.. _Anaconda: https://www.continuum.io/downloads

.. _Miniconda: https://conda.io/miniconda.html
