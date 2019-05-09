#!/bin/bash

# Default directory where IPython stores config files
IPYTHONDIR=~/.ipython

# Create a new profile, called "mpi"
ipython profile create --parallel --profile=mpi

# Add the following line as per instructions from
# https://ipyparallel.readthedocs.io/en/latest/process.html#using-ipcluster-in-mpiexec-mpirun-mode 
# This is instructing `ipcluster` to use the MPI launchers
echo "c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'" >> $IPYTHONDIR/profile_mpi/ipcluster_config.py

ver=$(mpiexec --version)
if [[ $ver == *"open-mpi"* ]]; then
  # OpenMPI need to be told that is allowed to oversubscribe cores
  echo "c.MPILauncher.mpi_cmd = ['mpiexec', '--oversubscribe']" >> $IPYTHONDIR/profile_mpi/ipcluster_config.py
fi
