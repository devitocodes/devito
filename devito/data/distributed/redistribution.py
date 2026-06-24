"""
Structured redistribution between distributed arrays (the "structured" quadrant).

``redistribute_set`` assigns ``self[idx] = other`` (``other`` distributed) by
pushing ``other``'s block into a strided region of ``self``, point-to-point.

The source-to-destination mapping is derived directly from the index as a
per-axis affine map (``source_coord = start + k*step``), so it never relies on
the legacy ``_process_args`` machinery. Both reuse the
engine's :class:`Layout` and owner-resolution/transport.

The structured path is chosen only if *every* rank can build it (a single
collective ``LAND``). This keeps the communication pattern identical on all ranks
-- essential, since a value/result with non-uniformly-structured decomposition
metadata would otherwise make the per-rank choice diverge and deadlock.
Unsupported patterns fall back to the legacy path -- no behavior is lost.
"""

import numpy as np

from devito.data.distributed.layout import Layout
from devito.data.distributed.plan import _decode, _encode, _group_peers, _resolve_owners
from devito.data.distributed.selection import Affine, Selection
from devito.data.distributed.transport import nbx_exchange
from devito.mpi import MPI

__all__ = ['redistribute_set']


def redistribute_set(data, glb_idx, other):
    """
    Assign ``data[glb_idx] = other`` (``other`` a distributed ``Data``) via a
    structured point-to-point redistribution.

    Returns ``True`` when handled, ``False`` when the pattern is not supported
    yet and the caller should fall back to the legacy path.

    The structured path is taken only if *every* rank can build it (decided by a
    single collective). This keeps the choice -- and therefore the communication
    pattern -- identical on all ranks, which is essential: a value produced by
    the legacy path (e.g. a reversed slice) may carry decomposition metadata that
    is not uniformly structured, and a diverging choice would deadlock.
    """
    try:
        spec = _structured_spec(data, glb_idx, other)
    except (IndexError, ValueError, TypeError):
        spec = None

    if not data._distributor.comm.allreduce(spec is not None, op=MPI.LAND):
        return False

    layout, gcoords, values = spec
    _push(layout, gcoords, values, np.asarray(data))
    return True


def _structured_spec(data, glb_idx, other):
    """
    Build the (layout, per-axis global coords, flat values) for the supported
    case: ``data`` and ``other`` both fully distributed with matching rank, every
    axis sliced (no scalars/arrays), and each sliced region matching ``other``'s
    extent. Anything else returns ``None`` (fall back to legacy).
    """
    decomposition = data._decomposition
    if any(d is None for d in decomposition):
        return None
    if not (isinstance(other, type(data)) and other._is_distributed):
        return None
    if other.ndim != data.ndim:
        return None
    other_dec = other._decomposition
    if any(d is None for d in other_dec):
        return None

    global_shape = tuple(d.size for d in decomposition)
    selection = Selection.from_index(glb_idx, global_shape)
    if any(not isinstance(s, Affine) for s in selection.selectors):
        return None

    coords_per_axis = []
    for axis, affine in enumerate(selection.selectors):
        # `other` fills the sliced region in the order of its own global indices;
        # map each to the corresponding `self` coordinate: start + c*step. The
        # per-axis owned indices come from this rank's subdomain (a replicated
        # axis owns the full extent), 0-based within `other`'s global space.
        dec = other_dec[axis]
        if affine.size != dec.size:
            return None
        owned = np.asarray(dec.loc_abs_numb, dtype=np.int64) - (dec.glb_min or 0)
        coords_per_axis.append(affine.start + owned*affine.step)

    mesh = np.meshgrid(*coords_per_axis, indexing='ij')
    gcoords = {axis: m.reshape(-1) for axis, m in enumerate(mesh)}
    values = np.ascontiguousarray(np.asarray(other)).reshape(-1)

    layout = Layout(data._distributor, decomposition, global_shape)
    return layout, gcoords, values


def _push(layout, gcoords, values, local):
    """Push ``values`` (one per global coordinate in ``gcoords``) to owners."""
    owners, dist_local, sub = _resolve_owners(None, layout, gcoords)
    peers, _, _ = _group_peers(layout, owners, dist_local, sub, gcoords)

    comm = layout.distributor.comm
    block_offsets = np.zeros(1, dtype=np.int64)   # no replicated payload
    headers = {r: _encode(1, block_offsets, dist_lin)
               for r, (_, dist_lin) in peers.items()}
    payloads = {r: values[rows]
                for r, (rows, _) in peers.items() if rows.size}

    requests = nbx_exchange(comm, headers, np.int64, tag=43)
    recv_values = nbx_exchange(comm, payloads, values.dtype, tag=44)

    axes = layout.distributed_axes
    moved = np.moveaxis(local, axes, range(len(axes)))
    for src, buf in requests.items():
        _, dist_lin = _decode(buf)
        midx = np.unravel_index(dist_lin, moved.shape)
        moved[midx] = recv_values[src]
