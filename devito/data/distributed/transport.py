"""
Transport layer for distributed data redistribution.

This module knows nothing about indexing or `Data`; it only moves contiguous
buffers between MPI ranks. The single primitive, `sparse_exchange`, performs a
sparse "all-to-some" exchange in which only the ranks that actually share data
exchange payloads.

Each rank first learns *how many* peers will send to it via a single small
`Reduce_scatter_block` over one integer per rank, then posts the point-to-point
messages and receives exactly that many. This relies only on standard, widely
portable MPI calls (no synchronous-send / nonblocking-barrier consensus), so it
behaves uniformly across MPI implementations; the payloads themselves still move
strictly point-to-point, so no data all-to-all takes place.
"""

import numpy as np

from devito.mpi import MPI
from devito.tools import mpi4py_mapper

__all__ = ['sparse_exchange']


def sparse_exchange(comm, sendbufs, dtype, tag=0):
    """
    Sparse "all-to-some" exchange of contiguous buffers.

    Each rank sends a buffer to each peer listed in `sendbufs` and receives from
    whichever ranks send to it. The number of incoming messages is discovered
    with one `Reduce_scatter_block` over a length-`nprocs` 0/1 indicator (a few
    bytes per rank); only ranks that share data then exchange payloads, strictly
    point-to-point.

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
    """
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Some MPI builds lack a native datatype for `dtype` (e.g. `float16`); send
    # over a same-size byte-equivalent wire type and view back on receipt, just
    # as the halo exchange does via `comm_dtype`. A no-op for mapped-to-self
    # types. The MPI datatype is left for mpi4py to infer from each buffer (as
    # the halo exchange's `Sendrecv` does): passing an explicit datatype forces
    # the `Type_get_extent` count path, which segfaults under some CUDA-aware MPI
    # builds, whereas buffer-protocol typing is portable.
    wire = np.dtype(mpi4py_mapper.get(np.dtype(dtype).type, dtype))

    recvd = {}

    # Local (self) delivery never goes through MPI
    local = sendbufs.get(rank)
    if local is not None and local.size:
        recvd[rank] = np.ravel(np.ascontiguousarray(local))

    # Discover how many peers will send to this rank: the column sum of a 0/1
    # "rank r sends to rank c" matrix, scattered so each rank gets its own count.
    indicator = np.zeros(nprocs, dtype=np.int32)
    for peer, buf in sendbufs.items():
        if peer != rank and buf.size:
            indicator[peer] = 1
    incoming = np.zeros(1, dtype=np.int32)
    comm.Reduce_scatter_block([indicator, MPI.INT], [incoming, MPI.INT],
                              op=MPI.SUM)

    # Post the point-to-point sends. The buffers must stay alive until the
    # matching requests complete, hence `live_bufs`.
    sends = []
    live_bufs = []
    for peer, buf in sendbufs.items():
        if peer == rank or buf.size == 0:
            continue
        buf = np.ascontiguousarray(buf).view(wire)
        live_bufs.append(buf)
        sends.append(comm.Isend(buf, dest=peer, tag=tag))

    # Receive exactly the expected number of messages, sizing each from its
    # probe. The byte count (a probe against `MPI.BYTE`) divides by the wire
    # item size to give the element count.
    status = MPI.Status()
    for _ in range(int(incoming[0])):
        comm.Probe(source=MPI.ANY_SOURCE, tag=tag, status=status)
        src = status.Get_source()
        count = status.Get_count(MPI.BYTE) // wire.itemsize
        buf = np.empty(count, dtype=wire)
        comm.Recv(buf, source=src, tag=tag)
        recvd[src] = buf.view(dtype)

    MPI.Request.Waitall(sends)
    return recvd
