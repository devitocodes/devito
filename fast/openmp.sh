#!/bin/bash

export DEVITO_LANGUAGE=openmp
export DEVITO_ARCH=cray
export DEVITO_LOGGING=DEBUG
unset DEVITO_AUTOTUNING

export OMP_PLACES=cores
export OMP_PROC_BIND=true

get_runtime() {
    $@ |& grep 'Operator.*ran' | rev | cut -d ' ' -f2 | rev
}

echo bench_name,so,threads,Devito,xDSL
for bench in "wave2d_b.py -d 2048 2048 --nt 512" "wave3d_b.py -d 512 512 512 --nt 512" "diffusion_3D_wBCs.py -d 512 512 512 --nt 512" "diffusion_2D_wBCs.py -d 2048 2048 --nt 512"
do
  bench_name=$(echo $bench | cut -d ' ' -f1)
  for so in 2 4 8
  do
    for threads in 1 2 4 8 16 32
    do
      export OMP_NUM_THREADS=$threads
      # echo OMP_NUM_THREADS=$threads
      # python $bench -so $so --devito 1
      devito_time=$(get_runtime python $bench -so $so --devito 1)
      xdsl_time=$(get_runtime python $bench -so $so --xdsl 1)
      echo $bench_name,$so,$threads,$devito_time,$xdsl_time
    done
  done
done
