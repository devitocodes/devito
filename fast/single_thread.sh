#!/bin/bash

unset DEVITO_LANGUAGE
export DEVITO_ARCH=cray
export DEVITO_LOGGING=DEBUG
export DEVITO_AUTOTUNING=aggressive

get_runtime() {
    $@ |& grep 'Operator.*ran' | rev | cut -d ' ' -f2 | rev
}

echo bench_name,so,Devito,xDSL
for bench in "wave2d_b.py -d 2048 2048 --nt 512" "wave3d_b.py -d 512 512 512 --nt 512" "diffusion_3D_wBCs.py -d 512 512 512 --nt 512" "diffusion_2D_wBCs.py -d 2048 2048 --nt 512"
do
  bench_name=$(echo $bench | cut -d ' ' -f1)
  for so in 2 4 8
    do
      devito_time=$(get_runtime python $bench -so $so --devito 1)
      xdsl_time=$(get_runtime python $bench -so $so --xdsl 1)
      echo $bench_name,$so,$devito_time,$xdsl_time
  done
done
