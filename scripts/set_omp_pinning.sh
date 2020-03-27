#!/bin/bash


#function to display commands
exe() { echo "$@" ; "$@" ; }

help() {
    echo "Usage: source set_omp_pinning [PLATFORM | reset | man]"
    echo "Possible values for PLATFORM are: [$platforms]"
    echo "With reset, delete any pre-existing pinning configuration and exit"
    echo "With man, display useful information about pinning-related env vars and exit"
}

platforms="snb ivb hsw bdw skx knl"

if [[ -z $1 || $1 == "--help" ]]; then
    help
elif [[ $1 == "reset" ]]; then
    exe unset OMP_NESTED
    exe unset OMP_MAX_ACTIVE_LEVELS
    exe unset OMP_PLACES
    exe unset OMP_PROC_BIND
    exe unset KMP_HOT_TEAMS
    exe unset KMP_HOT_TEAMS_MAX_LEVELS
elif [[ $1 == "man" ]]; then
    cat << EOF
OpenMP standard environment variables
-------------------------------------
OMP_PLACES:
  a list of places that threads can be pinned on:
  – threads: Each place corresponds to a single hardware thread
             on the target machine
  – cores: Each	place corresponds to a single core (having one or more
           hardware threads) on the target machine
  – sockets: Each place corresponds to a single socket (consisting of one or
             more cores) on the target machine
  – A list with explicit place values, such as:
      "{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}"
      "{0:4},{4:4},{8:4},{12:4}"

OMP_PROC_BIND:
  the thread affinity policy to be used for parallel regions
  at the corresponding nested level:
  – spread: Bind threads as evenly distributed (spread) as possible
  – close: Bind threads close to the master thread while still distributing
           threads for load balancing, wrap around once each place receives
           one thread
  – master: Bind threads the same place as the master thread

OMP_NESTED:
  enable/disable nested parallelism. Possible values are true|false. Most
  compilers keep it disabled by default

Intel-specific environment variables
------------------------------------
KMP_HOT_TEAMS:
  enable/disable hot teams. Disabled by default.

KMP_HOT_TEAMS_MAX_LEVEL:
  The value set will be the maximum depth for which hot teams will be
  maintained. By default, only the outermost parallel region is treated
  as a hot team.
EOF
elif [[ $1 == @(snb|ivb|hsw|bdw|skx) ]]; then
    exe unset OMP_NESTED
    exe export OMP_PLACES=cores
    exe export OMP_PROC_BIND=close
elif [[ $1 == "knl" ]]; then
    echo "Setting for nested parallelism"
    exe export OMP_NESTED=true
    exe export OMP_MAX_ACTIVE_LEVELS=2
    exe export OMP_PLACES=threads
    exe export OMP_PROC_BIND=spread,close
    exe export KMP_HOT_TEAMS=1
    exe export KMP_HOT_TEAMS_MAX_LEVELS=2
    echo "Note: the KMP_ evn vars are Intel-compiler specific"
else
    help
fi
