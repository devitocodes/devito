"""
Benchmark the overhead of Devito's ``data[idx]`` get/set over plain NumPy.

The script runs one of two studies, picked automatically from the MPI launch
size, each making a single point:

* **Serial** (run without ``mpirun``) -- with MPI off, ``data[idx]`` wraps
  NumPy indexing. The overhead is a fixed per-call cost (the ``.data`` accessor
  plus global-index translation), not a per-element one. It is large *relative*
  to NumPy for a trivial index -- whose NumPy cost is almost nothing -- but, being
  fixed, it amortises: only a few times NumPy at realistic sizes, and negligible
  against actual compute. ::

      python examples/mpi/benchmark_data_indexing.py

* **Parallel** (run under ``mpirun -n N`` with ``N > 1``) -- a routed
  assignment ``data[idx] = values`` moves data point-to-point. The benchmark
  sweeps the number of communicated points and shows the added cost scales with
  that volume (as expected for communication), while the per-rank NumPy work is
  not slowed down -- the NumPy reference time is independent of the rank count. ::

      DEVITO_MPI=1 mpirun -n 4 python examples/mpi/benchmark_data_indexing.py

Pass ``--reps`` to change the number of timed repetitions.
"""

import argparse
import os
from time import perf_counter

import numpy as np

from devito import Function, Grid, configuration

# Global axis length used by both studies; large enough to distribute, small
# enough to stay in memory on one process.
AXIS = 1 << 20

# Numbers of indexed/communicated points to sweep.
POINTS = [16, 128, 1024, 8192, 65536]


def _set(target, idx, vals):
    """Assign ``target[idx] = vals`` (``target`` a NumPy array or ``f.data``)."""
    target[idx] = vals


def _median_time(func, args, reps, comm=None):
    """
    Median wall time of ``func(*args)`` over ``reps`` repetitions, after a
    warm-up call.

    Under MPI the per-repetition time is the slowest rank (an ``MPI.MAX``
    reduction), since a collective operation finishes only when every rank does.
    """
    func(*args)
    times = []
    for _ in range(reps):
        if comm is not None:
            comm.Barrier()
        start = perf_counter()
        func(*args)
        elapsed = perf_counter() - start
        if comm is not None:
            from devito.mpi import MPI
            elapsed = comm.allreduce(elapsed, op=MPI.MAX)
        times.append(elapsed)
    return float(np.median(times))


def _distinct_indices(npoint):
    """``npoint`` distinct global indices, evenly spread over the axis."""
    step = max(1, AXIS // npoint)
    return (np.arange(npoint) * step) % AXIS


def serial_study(reps):
    """Overhead of ``data[idx]`` over NumPy with MPI off (single process)."""
    configuration['mpi'] = False
    grid = Grid(shape=(AXIS,))
    f = Function(name='f', grid=grid, dtype=np.float32)
    full = np.arange(AXIS, dtype=f.dtype)
    f.data[:] = full

    print(f"\nSerial overhead of data[idx] = v over NumPy "
          f"(axis={AXIS}, reps={reps})\n")
    header = ('points'.rjust(10) + 'numpy [us]'.rjust(14)
              + 'devito [us]'.rjust(14) + 'overhead [us]'.rjust(16)
              + 'ratio'.rjust(9))
    print(header)
    print('-' * len(header))
    for npoint in POINTS:
        idx = _distinct_indices(npoint)
        vals = full[idx]
        ref = full.copy()
        t_np = _median_time(_set, (ref, idx, vals), reps)
        t_dv = _median_time(_set, (f.data, idx, vals), reps)
        print(str(npoint).rjust(10)
              + format(t_np * 1e6, '.1f').rjust(14)
              + format(t_dv * 1e6, '.1f').rjust(14)
              + format((t_dv - t_np) * 1e6, '.1f').rjust(16)
              + (format(t_dv / t_np, '.0f') + 'x').rjust(9))
    print("\noverhead = devito - numpy; a fixed per-call cost (the .data accessor "
          "and\nglobal-index translation). It is large relative to NumPy for a "
          "trivial\nindex but, being fixed, amortises to a few x at realistic "
          "sizes.")


def mpi_study(comm, nprocs, reps):
    """Communication cost of a routed assignment vs the volume exchanged."""
    configuration['mpi'] = True
    grid = Grid(shape=(AXIS,))
    f = Function(name='f', grid=grid, dtype=np.float32)
    full = np.arange(AXIS, dtype=f.dtype)
    f.data[:] = full
    rank = grid.distributor.myrank

    if rank == 0:
        print(f"\nRouted assignment data[idx] = v under MPI "
              f"(axis={AXIS}, ranks={nprocs}, reps={reps})\n")
        header = ('points'.rjust(10) + 'KiB sent'.rjust(12)
                  + 'numpy [us]'.rjust(14) + 'devito [us]'.rjust(14)
                  + 'comm [us]'.rjust(14))
        print(header)
        print('-' * len(header))

    for npoint in POINTS:
        global_idx = _distinct_indices(npoint)
        # Each rank supplies the points it would hold in an external layout
        # (reversed, so the values cross rank boundaries when routed).
        loc_idx = np.array_split(global_idx[::-1], nprocs)[rank]
        loc_vals = full[loc_idx]
        ref = full.copy()

        # NumPy reference: the whole operation on a single process. It is the
        # pure-compute cost and is unaffected by the rank count.
        t_np = _median_time(_set, (ref, global_idx, full[global_idx]), reps, comm)
        # Devito: routed, point-to-point, with the plan cached (steady state).
        t_dv = _median_time(_set, (f.data, loc_idx, loc_vals), reps, comm)

        if rank == 0:
            kib = npoint * full.dtype.itemsize / 1024
            print(str(npoint).rjust(10)
                  + format(kib, '.1f').rjust(12)
                  + format(t_np * 1e6, '.1f').rjust(14)
                  + format(t_dv * 1e6, '.1f').rjust(14)
                  + format((t_dv - t_np) * 1e6, '.1f').rjust(14))

    if rank == 0:
        print("\ncomm = devito - numpy; grows with the volume communicated, as "
              "expected,\nwhile the numpy reference (pure compute) is unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=20,
                        help='timed repetitions per point count (default: 20)')
    args = parser.parse_args()

    # Detect the launch size without initialising MPI (so the serial study can
    # keep MPI off and exercise the pure-NumPy path).
    nprocs = int(os.environ.get('OMPI_COMM_WORLD_SIZE',
                                os.environ.get('PMI_SIZE', '1')))

    if nprocs == 1:
        serial_study(args.reps)
    else:
        configuration['mpi'] = True
        grid = Grid(shape=(8,))  # initialise the distributor (and MPI)
        mpi_study(grid.distributor.comm, nprocs, args.reps)


if __name__ == '__main__':
    main()
