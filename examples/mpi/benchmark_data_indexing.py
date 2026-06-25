"""
Benchmark Devito's MPI-distributed data indexing against single-process NumPy.

Devito turns ``data[idx]`` get/set on an MPI-distributed array into the minimal
point-to-point exchange (see ``devito/data/distributed``). This script measures
that machinery against the equivalent operation on a single-process NumPy array
holding the full domain, for three representative cases:

* a routed scatter assignment ``data[idx] = values`` (global-index labelled,
  the path that moves data across ranks);
* the matching routed read ``data[idx]``;
* a basic strided slice ``data[::2]`` (local/induced, no communication).

For the routed cases the distributed timing is reported both *cold* (first call,
including plan construction) and *warm* (cached plan -- the steady state of a
time loop, e.g. sparse injection every step). The NumPy column is the
single-process reference for the same logical operation on the full domain.

Run, for example::

    DEVITO_MPI=1 mpirun -n 4 python examples/mpi/benchmark_data_indexing.py
    DEVITO_MPI=1 mpirun -n 4 python examples/mpi/benchmark_data_indexing.py \\
        --shape 1024 1024 --reps 50
"""

import argparse
from time import perf_counter

import numpy as np

from devito import Function, Grid, configuration
from devito.data.distributed import exchange as _exchange
from devito.mpi import MPI


def _measure(call, reps, comm, clear=None):
    """
    Median wall time of ``call`` over ``reps`` repetitions.

    The wall time of one repetition is the slowest rank (an ``MPI.MAX`` reduce
    of the per-rank time). With ``clear`` given it is invoked before every timed
    call to drop the plan cache, measuring the cold path; otherwise one warm-up
    call is made first and the cached path is measured.
    """
    if clear is None:
        call()
    times = []
    for _ in range(reps):
        if clear is not None:
            clear()
        comm.Barrier()
        start = perf_counter()
        call()
        elapsed = perf_counter() - start
        times.append(comm.allreduce(elapsed, op=MPI.MAX))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs='+', default=[512, 512],
                        help='global grid shape (default: 512 512)')
    parser.add_argument('--reps', type=int, default=30,
                        help='timed repetitions per operation (default: 30)')
    parser.add_argument('--npoint', type=int, default=None,
                        help='routed points (default: an eighth of axis 0)')
    args = parser.parse_args()

    configuration['mpi'] = True

    shape = tuple(args.shape)
    # Creating the Grid initialises the MPI distributor (and MPI itself)
    grid = Grid(shape=shape)
    comm = grid.distributor.comm
    rank, nprocs = grid.distributor.myrank, grid.distributor.nprocs
    f = Function(name='f', grid=grid, dtype=np.float32)

    # Full replicated reference array; also the NumPy single-process baseline
    full = np.arange(int(np.prod(shape)), dtype=f.dtype).reshape(shape)
    f.data[:] = full

    # Global row indices to route (reversed, so they cross rank boundaries);
    # each rank contributes the slice it would hold in an external layout
    npoint = args.npoint or max(1, shape[0] // 8)
    glb_idx = np.arange(shape[0])[::-1][:npoint]
    loc_idx = np.array_split(glb_idx, nprocs)[rank]
    loc_vals = full[loc_idx]
    ref = full.copy()

    clear = _exchange._build.cache_clear

    rows = []

    # 1. Routed scatter assignment: data[idx] = values
    rows.append((
        'scatter  data[idx] = v',
        _measure(lambda: ref.__setitem__(glb_idx, full[glb_idx]), args.reps, comm),
        _measure(lambda: f.data.__setitem__(loc_idx, loc_vals), args.reps, comm,
                 clear=clear),
        _measure(lambda: f.data.__setitem__(loc_idx, loc_vals), args.reps, comm),
    ))

    # 2. Routed gather read: data[idx]
    rows.append((
        'gather   data[idx]',
        _measure(lambda: ref[glb_idx], args.reps, comm),
        _measure(lambda: f.data[loc_idx], args.reps, comm, clear=clear),
        _measure(lambda: f.data[loc_idx], args.reps, comm),
    ))

    # 3. Basic strided slice (local/induced, no communication)
    rows.append((
        'slice    data[::2]',
        _measure(lambda: ref[::2, ::2], args.reps, comm),
        None,
        _measure(lambda: f.data[::2, ::2], args.reps, comm),
    ))

    if rank == 0:
        print(f"\nMPI distributed data indexing vs NumPy "
              f"(shape={shape}, ranks={nprocs}, "
              f"routed points={npoint}, reps={args.reps})\n")
        header = ('operation'.ljust(24) + 'numpy [us]'.rjust(14)
                  + 'devito cold [us]'.rjust(18) + 'devito warm [us]'.rjust(18))
        print(header)
        print('-' * len(header))
        for name, np_t, cold, warm in rows:
            cold_s = '-' if cold is None else format(cold * 1e6, '.1f')
            print(name.ljust(24)
                  + format(np_t * 1e6, '.1f').rjust(14)
                  + cold_s.rjust(18)
                  + format(warm * 1e6, '.1f').rjust(18))
        print("\ncold = first call (plan construction); "
              "warm = cached plan (steady state of a time loop).")


if __name__ == '__main__':
    main()
