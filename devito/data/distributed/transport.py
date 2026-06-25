"""
Transport layer for distributed data redistribution.

This module knows nothing about indexing or `Data`; it only moves contiguous
buffers between MPI ranks. The single primitive, `nbx_exchange`, performs
a sparse "all-to-some" exchange in which only the ranks that actually share data
ever communicate. It can be swapped for neighbor collectives or a persistent
graph communicator without affecting the layers above.
"""

import numpy as np

from devito.mpi import MPI
from devito.tools import dtype_to_mpidtype

__all__ = ['nbx_exchange']


def nbx_exchange(comm, sendbufs, dtype, tag=0):
    """
    Sparse "all-to-some" exchange via nonblocking consensus (NBX).

    Each rank sends a buffer to each peer listed in `sendbufs` and receives
    from whichever ranks send to it, without any rank needing global knowledge
    of the communication pattern and without any dense collective. Only ranks
    that actually exchange data communicate; global termination is detected with
    a single nonblocking barrier (log-depth).

    Parameters
    ----------
    comm : MPI communicator
        The communicator over which to exchange.
    sendbufs : dict
        Maps a destination rank to the buffer (a NumPy array) to send it. An
        entry for the caller's own rank is delivered locally, bypassing MPI.
        Empty buffers are skipped.
    dtype : numpy.dtype
        Element type shared by every buffer.
    tag : int, optional
        MPI tag isolating this exchange. Defaults to 0.

    Returns
    -------
    dict
        Maps each source rank to the 1D buffer received from it. The caller
        reshapes using its known payload shape.

    Notes
    -----
    Implements the NBX algorithm (Hoefler et al., "Scalable Communication
    Protocols for Dynamic Sparse Data Exchange"). Synchronous sends (`Issend`)
    complete only once matched by a receive, so a rank can enter the nonblocking
    barrier only after every message it sent has been taken. Once all ranks are
    in the barrier no message is in flight, so probing can safely stop.
    """
    rank = comm.Get_rank()
    mpitype = dtype_to_mpidtype(dtype)

    recvd = {}

    # Local (self) delivery never goes through MPI
    local = sendbufs.get(rank)
    if local is not None and local.size:
        recvd[rank] = np.ravel(np.ascontiguousarray(local))

    # Post synchronous sends to every other peer. The buffers must stay alive
    # until the matching requests complete, hence `live_bufs`.
    sends = []
    live_bufs = []
    for peer, buf in sendbufs.items():
        if peer == rank or buf.size == 0:
            continue
        buf = np.ascontiguousarray(buf)
        live_bufs.append(buf)
        sends.append(comm.Issend([buf, mpitype], dest=peer, tag=tag))

    barrier = None
    status = MPI.Status()
    while True:
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=tag, status=status):
            src = status.Get_source()
            count = status.Get_count(mpitype)
            buf = np.empty(count, dtype=dtype)
            comm.Recv([buf, mpitype], source=src, tag=tag)
            recvd[src] = buf
        elif barrier is None:
            if MPI.Request.Testall(sends):
                # All my sends were matched -> announce I am done sending
                barrier = comm.Ibarrier()
        elif barrier.Test():
            # Everyone is done sending and nothing is in flight
            break

    return recvd
