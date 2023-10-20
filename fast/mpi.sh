#!/bin/bash

# Simple script to run MPI+OMP benchmarks locally,
# for simple sanity checks.

# Use OpenMP and MPI
export DEVITO_LANGUAGE=openmp
export DEVITO_MPI=1
# Use the default compiler here, cray and MPI doesnt work on my machine.
unset DEVITO_ARCH
# Enable debug logging.
export DEVITO_LOGGING=BENCH
# Enable (tile size) autotuning.
# I disable it for speed sometimes; NB enabling it requires that no explicit tile size
# is specified in the Operator constructor args.
export DEVITO_AUTOTUNING="aggressive"

# Bind threads to physical cores
export OMP_PLACES=cores
export OMP_PROC_BIND=true

# Doing 4 ranks X 4 threads locally here
export OMP_NUM_THREADS=4
export MPI_NUM_RANKS=4
export HYDRA_TOPO_DEBUG=1
# Just extract the reported throughput from the whole output of the passed command
get_throughput() {
    #echo $($@)
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

      # To uncomment to check what's going on without capturing the output.
      # echo OMP_NUM_THREADS=$threads
      #  mpirun -np $MPI_NUM_RANKS --bind-to=core python $bench -so $so --devito 1
      #  mpirun -np $MPI_NUM_RANKS --bind-to=core python $bench -so $so --xdsl 1

      # Get the runtimes
      DEVITO_MPI=diag2 devito_time=$(get_throughput mpirun -n $MPI_NUM_RANKS --bind-to core:$OMP_NUM_THREADS python $bench -so $so --devito 1)
      xdsl_time=$(get_throughput mpirun -n $MPI_NUM_RANKS --bind-to core:$OMP_NUM_THREADS python $bench -so $so --xdsl 1)
      # print CSV line
      echo $bench_name,$so,$devito_time,$xdsl_time
  done
done
