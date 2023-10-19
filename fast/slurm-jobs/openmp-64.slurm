#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=Devito_OpenMP_64
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --switches=1@360 # Each group has 128 nodes

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=d011
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o ./jobs-output/openmp-64.%j.out # STDOUT

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export SHARED=/work/d011/d011/shared
module use $SHARED/modules
module load sc-23
module load cray-mpich


cd $SHARED/software/devito/fast
# Simple script to run multithreaded benchmarks locally,
# for simple sanity checks.

#   We want one thread per physical core
export OMP_PLACES=cores

# Devito-specific env variables
export DEVITO_ARCH=cray
export DEVITO_LANGUAGE=openmp
export DEVITO_LOGGING=BENCH
unset DEVITO_MPI
export DEVITO_AUTOTUNING=aggressive

export OMP_PROC_BIND=true

export CRAY_OMP_CHECK_AFFINITY=TRUE
export SLURM_CPU_FREQ_REQ=2250000
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Just extract the reported throughput from the whole output of the passed command
get_throughput() {
    # echo $($@)
    $@ |& grep Global | head -n 1 | cut -d ' ' -f6
}

# Iterate over benchmarks and cases, print simple CSV data to stdout
# Copy-pastes nicely in Google Sheets
echo bench_name,so,Devito,xDSL
for bench in "wave2d_b.py -d 8192 8192 --nt 1024" "wave3d_b.py -d 512 512 512 --nt 512" "diffusion_2D_wBCs.py -d 8192 8192 --nt 1024" "diffusion_3D_wBCs.py -d 512 512 512 --nt 512"
do
  # Get the benchmark file for printing
  bench_name=$(echo $bench | cut -d ' ' -f1)
  # Iterate over measured space orders
  for so in 2 4 8
  do

    # Get the throughputs
    devito_time=$(get_throughput srun --distribution=block:block --hint=nomultithread python $bench -so $so --devito 1)
    xdsl_time=$(get_throughput srun --distribution=block:block --hint=nomultithread python $bench -so $so --xdsl 1)
    # print CSV line
    echo $bench_name,$so,$devito_time,$xdsl_time
  done
done
