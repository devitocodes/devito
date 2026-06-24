"""
Plan layer: the rank-to-rank routing induced by a Selection on a Layout.

:class:`ExchangePlan` is a value object built once (no communication) from a
:class:`~devito.data.distributed.selection.Selection` and a
:class:`~devito.data.distributed.layout.Layout`. It computes, for every routed
element, its owner rank and owner-local offset, and arranges the result/value
array as ``(npoints, payload)`` so packing and unpacking are single NumPy
fancy-index operations. The same plan drives ``get`` (pull) and ``put`` (push).

Every axis falls in one of four quadrants and is handled uniformly:

============== ================================== ============================
               structured (scalar / slice)        scattered (array / mask)
============== ================================== ============================
replicated     local payload block                local payload block
distributed    block redistribution (owner index) owner index via decomposition
============== ================================== ============================

The unit of exchange is one "point" (a coordinate tuple over the distributed
axes) carrying a payload block addressed by the replicated axes.
"""

import numpy as np

from devito.data.distributed.selection import Affine, Scalar
from devito.data.distributed.transport import nbx_exchange
from devito.mpi import MPI
from devito.tools import prod

__all__ = ['ExchangePlan', 'nbx_push']


class ExchangePlan:

    def __init__(self, layout, selection, perm, t_shape, payload_shape, owners,
                 peers, block_offsets, repl_total, oob_error, dup_error):
        self.layout = layout
        self.selection = selection
        self._perm = perm                  # result axes -> (T-dims..., payload-dims...)
        self._t_shape = t_shape
        self._payload_shape = payload_shape
        self._owners = owners              # owner rank per T row (-1 if OOB)
        self._peers = peers                # {rank: (rows, dist_lin)}
        self._block_offsets = block_offsets  # (payload_size,) within owner repl block
        self._repl_total = repl_total      # full replicated stride
        self._oob_error = oob_error        # out-of-bounds index (get and set)
        self._dup_error = dup_error        # duplicate target (set only)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, selection, layout):
        """Plan the exchange for ``data[idx]``; serves both get and set."""
        dist = set(layout.distributed_axes)
        repl = set(layout.replicated_axes)

        adv_dist = [a for a in selection.advanced_axes if a in dist]
        adv_repl = [a for a in selection.advanced_axes if a in repl]
        if adv_dist and adv_repl:
            raise NotImplementedError(
                "Advanced indexing coupling distributed and replicated axes is "
                "not supported"
            )
        advanced_distributed = bool(adv_dist)

        dims = selection.result_dims
        is_t = [_dim_is_distributed(d, dist, advanced_distributed) for d in dims]
        t_pos = [i for i, t in enumerate(is_t) if t]
        p_pos = [i for i, t in enumerate(is_t) if not t]
        perm = t_pos + p_pos

        t_shape = tuple(selection.result_shape[i] for i in t_pos)
        payload_shape = tuple(selection.result_shape[i] for i in p_pos)
        t_dims = [dims[i] for i in t_pos]
        p_dims = [dims[i] for i in p_pos]

        gcoords = _distributed_coords(selection, layout, t_dims, t_shape)
        owners, dist_local, sub = _resolve_owners(selection, layout, gcoords)

        block_offsets = _replicated_block(selection, layout, p_dims, payload_shape)
        repl_total = layout.replicated_size

        peers, oob_error, dup_error = _group_peers(layout, owners, dist_local,
                                                   sub, gcoords)
        return cls(layout, selection, perm, t_shape, payload_shape, owners,
                   peers, block_offsets, repl_total, oob_error, dup_error)

    # --------------------------------------------------------------- helpers

    @property
    def comm(self):
        return self.layout.distributor.comm

    @property
    def nprocs(self):
        return self.layout.distributor.nprocs

    @property
    def payload_size(self):
        return prod(self._payload_shape)

    def _raise_on_error(self, check_dup):
        """Reach consensus on local errors and raise consistently on all ranks.

        A single 1-bit ``Allreduce`` gates the (rare) error path so that every
        rank raises together, avoiding a deadlock where one rank raises while the
        others enter the exchange. This is log-depth, not an all-to-all.
        """
        error = self._oob_error or (self._dup_error if check_dup else None)
        if self.nprocs > 1:
            if self.comm.allreduce(error is not None, op=MPI.LOR):
                messages = self.comm.allgather(error)
                joined = "; ".join(f"rank {r}: {m}"
                                   for r, m in enumerate(messages) if m)
                raise ValueError(joined)
        elif error is not None:
            raise ValueError(error)

    def _moved(self, local):
        """View of the rank-local array with distributed axes moved to front."""
        axes = self.layout.distributed_axes
        return np.moveaxis(local, axes, range(len(axes)))

    def _owner_apply(self, moved, dist_lin, block_offsets):
        """Owner-local (row, column) multi-index for a received message."""
        elem = dist_lin[:, None] * self._repl_total + block_offsets[None, :]
        return np.unravel_index(elem.reshape(-1), moved.shape)

    # ------------------------------------------------------------------- get

    def get(self, local):
        """Return ``data[idx]`` as a NumPy array (pull from owner ranks)."""
        self._raise_on_error(check_dup=False)
        comm, ps = self.comm, self.payload_size
        dtype = local.dtype

        headers = {r: _encode(ps, self._block_offsets, dist_lin)
                   for r, (_, dist_lin) in self._peers.items()}
        requests = nbx_exchange(comm, headers, np.int64, tag=41)

        moved = self._moved(local)
        replies = {}
        for src, buf in requests.items():
            block_offsets, dist_lin = _decode(buf)
            midx = self._owner_apply(moved, dist_lin, block_offsets)
            replies[src] = np.ascontiguousarray(moved[midx]).reshape(-1)
        payloads = nbx_exchange(comm, replies, dtype, tag=42)

        rows_flat = np.zeros((self._nrows(), ps), dtype=dtype)
        for r, (rows, _) in self._peers.items():
            if rows.size:
                rows_flat[rows] = payloads[r].reshape(rows.size, ps)
        return self._rows_to_result(rows_flat)

    # ------------------------------------------------------------------- put

    def put(self, local, value):
        """Assign ``data[idx] = value`` (push to owner ranks)."""
        self._raise_on_error(check_dup=True)
        rows_flat = self._value_to_rows(value, local.dtype)
        nbx_push(self.comm, self.layout.distributed_axes, self._repl_total,
                 self._peers, self._block_offsets, self.payload_size, rows_flat,
                 local)

    # ------------------------------------------------------- result <-> rows

    def _nrows(self):
        return prod(self._t_shape)

    def _rows_to_result(self, rows_flat):
        moved_shape = self._t_shape + self._payload_shape
        moved = rows_flat.reshape(moved_shape)
        result = np.moveaxis(moved, range(len(self._perm)), self._perm)
        return np.ascontiguousarray(result).reshape(self.selection.result_shape)

    def _value_to_rows(self, value, dtype):
        value = np.broadcast_to(np.asarray(value, dtype=dtype),
                                self.selection.result_shape)
        moved = np.transpose(value, self._perm)
        return np.ascontiguousarray(moved).reshape(self._nrows(), self.payload_size)


# --------------------------------------------------------------------- free fns


def _dim_is_distributed(dim, dist, advanced_distributed):
    kind, val = dim
    if kind == 'basic':
        return val in dist
    return advanced_distributed


def _distributed_coords(selection, layout, t_dims, t_shape):
    """Global coordinate per distributed axis, one value per T row."""
    nrows = prod(t_shape)
    grids = (np.indices(t_shape).reshape(len(t_dims), -1)
             if t_dims else np.zeros((0, nrows), dtype=np.int64))

    # Index of advanced T dims, flattened into a single point index q
    adv_rows = [ri for ri, d in enumerate(t_dims) if d[0] == 'adv']
    q = None
    if adv_rows:
        q = np.ravel_multi_index([grids[ri] for ri in adv_rows],
                                 selection.advanced_shape)

    gcoords = {}
    for axis in layout.distributed_axes:
        s = selection.selectors[axis]
        if isinstance(s, Scalar):
            gcoords[axis] = np.full(nrows, s.index, dtype=np.int64)
        elif isinstance(s, Affine):
            ri = t_dims.index(('basic', axis))
            gcoords[axis] = s.coords[grids[ri]]
        else:  # Explicit
            gcoords[axis] = s.coords[q]
    return gcoords


def _resolve_owners(selection, layout, gcoords):
    """Owner rank, per-axis local offset, and per-axis sub-rank for each T row."""
    axes = layout.distributed_axes
    nrows = len(next(iter(gcoords.values()))) if gcoords else 0

    sub = np.zeros((len(axes), nrows), dtype=np.int64)
    local = np.zeros((len(axes), nrows), dtype=np.int64)
    valid = np.ones(nrows, dtype=bool)
    for k, axis in enumerate(axes):
        owner_lut, local_lut, _ = layout.axis_maps(axis)
        g = gcoords[axis]
        in_range = (g >= 0) & (g < owner_lut.size)
        safe = np.where(in_range, g, 0)
        sub[k] = np.where(in_range, owner_lut[safe], -1)
        local[k] = np.where(in_range, local_lut[safe], -1)
        valid &= in_range & (sub[k] >= 0)

    # Map sub-rank tuples to flat ranks through the topology
    rank_arr = np.full(layout.topology_shape, -1, dtype=np.int64)
    for coord, r in layout.coord_to_rank.items():
        rank_arr[coord] = r
    owners = np.full(nrows, -1, dtype=np.int64)
    if nrows:
        safe_sub = np.where(valid, sub, 0)
        owners = np.where(valid, rank_arr[tuple(safe_sub)], -1)
    return owners, local, sub


def _replicated_block(selection, layout, p_dims, payload_shape):
    """Offsets of the selected replicated block within the owner's repl block."""
    payload_size = prod(payload_shape)
    offsets = np.zeros(payload_size, dtype=np.int64)
    if not layout.replicated_axes:
        return offsets

    pgrids = (np.indices(payload_shape).reshape(len(p_dims), -1)
              if p_dims else np.zeros((0, payload_size), dtype=np.int64))
    adv_rows = [ri for ri, d in enumerate(p_dims) if d[0] == 'adv']
    q = None
    if adv_rows:
        q = np.ravel_multi_index([pgrids[ri] for ri in adv_rows],
                                 selection.advanced_shape)

    # Strides over replicated axes (increasing order) of the owner-local block
    sizes = [layout.global_shape[a] for a in layout.replicated_axes]
    strides = {a: int(prod(sizes[i + 1:]))
               for i, a in enumerate(layout.replicated_axes)}

    for axis in layout.replicated_axes:
        s = selection.selectors[axis]
        if isinstance(s, Scalar):
            coord = np.full(payload_size, s.index, dtype=np.int64)
        elif isinstance(s, Affine):
            ri = p_dims.index(('basic', axis))
            coord = s.coords[pgrids[ri]]
        else:  # Explicit (replicated advanced)
            coord = s.coords[q]
        offsets += coord * strides[axis]
    return offsets


def _group_peers(layout, owners, dist_local, sub, gcoords):
    """Group T rows by owner, computing the owner-local linear (dist) offset."""
    axes = layout.distributed_axes
    oob_error = dup_error = None

    if owners.size:
        # Within-rank duplicate detection over distributed coordinates
        stacked = np.stack([gcoords[a] for a in axes], axis=1) if axes \
            else np.zeros((owners.size, 0), dtype=np.int64)
        if np.unique(stacked, axis=0).shape[0] != stacked.shape[0]:
            dup_error = "Duplicate global indices in distributed assignment"

    if np.any(owners < 0):
        oob_error = "Advanced index contains out-of-bounds global indices"

    peers = {}
    for r in np.unique(owners[owners >= 0]) if owners.size else []:
        rows = np.where(owners == r)[0]
        subranks = sub[:, rows[0]]
        local_shape = tuple(int(layout.axis_maps(a)[2][subranks[k]])
                            for k, a in enumerate(axes))
        if local_shape:
            dist_lin = np.ravel_multi_index([dist_local[k, rows]
                                             for k in range(len(axes))],
                                            local_shape)
        else:
            dist_lin = np.zeros(rows.size, dtype=np.int64)
        peers[int(r)] = (rows, np.asarray(dist_lin, dtype=np.int64))
    return peers, oob_error, dup_error


def nbx_push(comm, distributed_axes, repl_total, peers, block_offsets,
             payload_size, rows_flat, local):
    """
    Route ``rows_flat`` to the owner ranks (NBX) and scatter each received
    payload into ``local`` at its owner-local position.

    This is the single push primitive behind both :meth:`ExchangePlan.put`
    (advanced/replicated assignment, ``payload_size`` >= 1) and the structured
    redistribution in :mod:`devito.data.distributed.redistribution` (one value
    per point, ``payload_size`` == 1). ``rows_flat`` is ``(nrows, payload_size)``
    in owner-grouped row order; ``block_offsets`` indexes the replicated payload
    block within an owner's local stride ``repl_total``.
    """
    headers = {r: _encode(payload_size, block_offsets, dist_lin)
               for r, (_, dist_lin) in peers.items()}
    payloads = {r: rows_flat[rows].reshape(-1)
                for r, (rows, _) in peers.items() if rows.size}
    requests = nbx_exchange(comm, headers, np.int64, tag=43)
    values = nbx_exchange(comm, payloads, rows_flat.dtype, tag=44)

    moved = np.moveaxis(local, distributed_axes, range(len(distributed_axes)))
    for src, buf in requests.items():
        offsets, dist_lin = _decode(buf)
        elem = dist_lin[:, None] * repl_total + offsets[None, :]
        midx = np.unravel_index(elem.reshape(-1), moved.shape)
        moved[midx] = values[src]


def _encode(payload_size, block_offsets, dist_lin):
    """Pack a request header: [payload_size, *block_offsets, *dist_lin]."""
    return np.concatenate(([payload_size], block_offsets, dist_lin)).astype(np.int64)


def _decode(buf):
    payload_size = int(buf[0])
    block_offsets = buf[1:1 + payload_size]
    dist_lin = buf[1 + payload_size:]
    return block_offsets, dist_lin
