#!/bin/bash

export DEVITO_LANGUAGE=openmp
export DEVITO_ARCH=cray
export DEVITO_LOGGING=DEBUG
unset OMP_NUM_THREADS

export OMP_PLACES=cores
export OMP_PROC_BIND=true
unset OMP_NUM_THREADS

for bench in "setup_wave2d.py -d 8192 8192 --nt 1024" "setup_wave3d.py -d 512 512 512 --nt 512"
do
  for so in 2 4 8
    do
      python $bench -so $so
  done
done
