#!/bin/bash

# Simple script to run multithreaded benchmarks locally,
# for simple sanity checks.

# Use OpenMP
export DEVITO_LANGUAGE=openmp
# Use the cray compiler, if available.
export DEVITO_ARCH=cray
# Enable debug logging.
export DEVITO_LOGGING=BENCH
# Enable (tile size) autotuning.
# I disable it for speed sometimes; NB enabling it requires that no explicit tile size
# is specified in the Operator constructor args.
export DEVITO_AUTOTUNING="aggressive"

# Bind threads to physical cores
export OMP_PLACES=cores
export OMP_PROC_BIND=true

# Just extract the reported throughput from the whole output of the passed command
get_throughput() {
    $@ |& grep Global | head -n 1 | cut -d ' ' -f6
}

# Iterate over benchmarks and cases, print simple CSV data to stdout
# Copy-pastes nicely in Google Sheets
echo bench_name,so,threads,Devito,xDSL
for bench in "wave2d_b.py -d 8192 8192 --nt 1024" "wave3d_b.py -d 512 512 512 --nt 512" "diffusion_2D_wBCs.py -d 8192 8192 --nt 1024" "diffusion_3D_wBCs.py -d 512 512 512 --nt 512"
do
  # Get the benchmark file for printing
  bench_name=$(echo $bench | cut -d ' ' -f1)
  # Iterate over measured space orders
  for so in 2 4 8
  do
    # Iterate over measured thread numbers
    for threads in 1 2 4 8 16 32 64 128
    do
      # Set the thread number to use
      export OMP_NUM_THREADS=$threads

      # To uncomment to check what's going on without capturing the output.
      # echo OMP_NUM_THREADS=$threads
      python $bench -so $so --devito 1
      python $bench -so $so --xdsl 1

      # Get the throughputs
      # devito_time=$(get_throughput python $bench -so $so --devito 1)
      # xdsl_time=$(get_throughput python $bench -so $so --xdsl 1)
      # print CSV line
      # echo $bench_name,$so,$threads,$devito_time,$xdsl_time
    done
  done
done
