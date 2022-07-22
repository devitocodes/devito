#!/bin/bash

# Default directory where IPython stores config files
IPYTHONDIR=~/.ipython

# Try to stop any cluster that may be active from previous (failed) runs
ipcluster stop --profile=mpi || echo "No active profile_mpi"

# Remove any existing configuration
rm -rf $IPYTHONDIR/profile_mpi/

# Create a new profile, called "mpi"
ipython profile create --parallel --profile=mpi

ver=$(mpiexec --version)
if [[ $ver == *"open-mpi"* ]]; then
  # OpenMPI need to be told that is allowed to oversubscribe cores
  echo "c.MPILauncher.mpi_cmd = ['mpiexec', '--oversubscribe']" >> $IPYTHONDIR/profile_mpi/ipcluster_config.py
fi
