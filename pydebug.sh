#!/usr/bin/env bash
# script for debugging MPI'd python programs. Debugs only a single rank.
# usage: replace the python call with pydebug. e.g.:
#     mpirun -n 4 pyton example.py <args>
# becomes:
#     mpirun -n 4 ./pydebug.sh example.py <arg>
#
# you will find the output of each node on stdout, and also in 
# log_rank_<rank-number>.log

function print_local() {
    echo "start at" $(date) > log_rank_"$PMI_RANK".log
    while IFS='' read line; do
        printf "$PMI_RANK: %s\n" "$line" | tee -a log_rank_"$PMI_RANK".log
    done
}

if [[ "$PMI_RANK" == "${DEBUGRANK:0}" ]]; then
    echo python -m debugpy --wait-for-client --listen 5678 "$@" | print_local
    python -m debugpy --wait-for-client --listen 5678 "$@" | print_local
else
    echo python "$@" | print_local
    python "$@" | print_local
fi
